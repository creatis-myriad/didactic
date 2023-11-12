import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
from sklearn.decomposition import PCA
from vital.data.cardinal.config import TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.transforms import Interp1d
from vital.utils.decorators import auto_cast_data
from vital.utils.norm import minmax_scaling, scale


class CardiacSequenceAttributesPCA:
    """PCA model specialized for cardiac sequences time-series attrs, where attrs have different range of values."""

    def __init__(
        self,
        n_components: int,
        pca_kwargs: Dict[str, Any] = None,
        strategy: Literal["global_pca", "attr_pca"] = "global_pca",
    ):
        """Initializes class instance.

        Args:
            n_components: Dimensionality of the PCA model's latent space.
            pca_kwargs: Parameters that will be passed along to the `PCA`'s init.
            strategy: Strategies available for handling multi-domain time-series attributes values.
                'global_pca': one global PCA model is trained on the normalized attributes.
                'attr_pca': one PCA model is fitted to each attribute, side-stepping the need for normalization.
        """
        self._n_components = n_components
        self._pca_kwargs = pca_kwargs if pca_kwargs else {}
        self.strategy = strategy

        match self.strategy:
            case "global_pca":
                self.pca = None
                self._attrs_stats = {}
            case "attr_pca":
                self.pca = {}

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the PCA model's latent space."""
        return self._n_components

    @property
    def in_shape(self) -> Tuple[int, int]:
        """Dimensionality of one input sample expected by the PCA model."""
        match self.strategy:
            case "global_pca":
                pca = self.pca
            case "attr_pca":
                pca = next(iter(self.pca.values()))
        return 1, pca.n_features_in_

    def save(self, save_path: Path) -> None:
        """Saves a pickled version of the PCA model to a file.

        Args:
            save_path: Path where to save a pickled version of the PCA model.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, mode="wb") as save_file:
            pickle.dump(self, save_file)

    @classmethod
    def preprocess_attr_data(
        cls, data: np.ndarray, bounds: Tuple[float | np.ndarray, float | np.ndarray] = None, length: int = None
    ) -> np.ndarray:
        """Pre-processes input data, to prepare it for inference using the autoencoder.

        Args:
            data: (N, C, L), Input data to process to prepare for inference.
            bounds: If applicable, minimum/maximum bounds to use to perform min-max scaling on the data.
            length: If applicable, length expected by the autoencoder, to reach by linearly interpolating the data.

        Returns:
            (N, C, M), Pre-processed input ready to be fed to the autoencoder.
        """
        if length is not None and length != data.shape[-1]:
            # Interpolate data to reach the requested number of data points
            normalize_length = Interp1d(length)
            data = np.apply_along_axis(normalize_length, -1, data)
        if bounds is not None:
            # Make sure the data is scaled between 0 and 1
            data = minmax_scaling(data, bounds=bounds)
        return data

    @classmethod
    def postprocess_attr_prediction(
        cls, prediction: np.ndarray, bounds: Tuple[float | np.ndarray, float | np.ndarray] = None, length: int = None
    ) -> np.ndarray:
        """Post-processes reconstructed attributes, undoing pre-processing to make predictions comparable with inputs.

        Args:
            prediction: (N, C, M), Reconstructed input to post-process before comparing to the input.
            bounds: If applicable, minimum/maximum to use to invert the input's min-max scaling on the prediction.
            length: If applicable, target output length to reach by linearly interpolating the prediction.

        Returns:
            (N, C, L), Post-processed prediction ready to be compared to the input.
        """
        if bounds is not None:
            # Undo the scaling of the input data on the reconstructed signal
            prediction = scale(prediction, bounds)
        if length is not None and length != prediction.shape[-1]:
            # Interpolate data to reach the requested number of data points
            resample_to_input_length = Interp1d(length)
            prediction = np.apply_along_axis(resample_to_input_length, -1, prediction)
        return prediction

    @auto_cast_data
    def __call__(
        self,
        x: np.ndarray,
        task: Literal["encode", "decode", "reconstruct"] = "reconstruct",
        attr: Tuple[ViewEnum, TimeSeriesAttribute] = None,
        out_shape: Tuple[int, ...] = None,
    ) -> np.ndarray:
        """Performs test-time inference on the input.

        Args:
            x: - if ``task`` == 'decode': (N, ?), Encoding in the latent space.
               - else: (N, [C,] L), Input, with an optional dimension for the channels.
            task: Flag indicating which type of inference task to perform.
            attr: Key identifying the attribute for which to perform inference.
            out_shape: Target ([C], L) of the reconstructed output. When `task!=decode`, this is normally inferred from
                the data, and providing it in this case will lead to the output shape not matching the input.

        Returns:
            if ``task`` == 'encode':
                (N, ``Z``), Encoding of the input in the latent space.
            else:
                (N, [C,] L), Reconstructed input, with the same channels as in the input.
        """
        attr_bounds = None
        match self.strategy:
            case "global_pca":
                pca = self.pca
                if attr:
                    attr_bounds = self._attrs_stats[attr]
            case "attr_pca":
                pca = self.pca[attr]

        if out_shape is None:
            # If the user doesn't request a specific output shape
            if task == "decode":
                # If we have no input data whose output shape to match, use the default shape used during training
                out_shape = (pca.n_features_in_,)
            else:
                # Match the shape of the input data
                out_shape = x.shape[1:]
        out_channels = out_shape[0] if len(out_shape) > 1 else 0
        out_length = out_shape[-1]

        if task in ["encode", "reconstruct"]:
            # Add channel dimension if it is not in the input data
            if x.ndim == 2:
                x = np.expand_dims(x, axis=1)
            x = self.preprocess_attr_data(x, bounds=attr_bounds, length=pca.n_features_in_ // x.shape[1])
            x = pca.transform(x.reshape(len(x), pca.n_features_in_))
        if task in ["decode", "reconstruct"]:
            x = pca.inverse_transform(x).reshape(len(x), max(out_channels, 1), -1)
            x = self.postprocess_attr_prediction(x, bounds=attr_bounds, length=out_length)
            # Eliminate channel dimension if it was not in the original data
            if not out_channels:
                x = x.squeeze(axis=1)
        return x

    def fit(self, samples: Dict[Tuple[ViewEnum, TimeSeriesAttribute], np.ndarray]) -> "CardiacSequenceAttributesPCA":
        """Fits one or multiple PCA model(s) to the attributes samples, depending on the chosen strategy.

        Args:
            samples: Mapping between attributes and their samples, of shape (N, L), where N is the number of samples and
                L is the length of the time-series.
                The samples are divided by attributes to allow the model to learn models/statistics on each attribute
                independently.

        Returns:
            The instance itself, now fitted to the data.
        """
        match self.strategy:
            case "global_pca":
                # Normalize each attributes' data w.r.t. its own range before learning one PCA model
                self._attrs_stats = {
                    attr_key: (attr_samples.min(), attr_samples.max()) for attr_key, attr_samples in samples.items()
                }
                samples = np.vstack(
                    [
                        minmax_scaling(attr_samples, self._attrs_stats[attr_key])
                        for attr_key, attr_samples in samples.items()
                    ]
                )
                self.pca = PCA(n_components=self._n_components, **self._pca_kwargs).fit(samples)
            case "attr_pca":
                # Learn one PCA model for each attribute
                self.pca = {
                    attr_key: PCA(n_components=self._n_components, **self._pca_kwargs).fit(attr_samples)
                    for attr_key, attr_samples in samples.items()
                }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Wrapper around `__call__` to be compatible with `sklearn`'s PCA API."""
        return self(X, task="encode")

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Wrapper around `__call__` to be compatible with `sklearn`'s PCA API."""
        return self(X, task="decode")


def main():
    """Fit the PCA to the provided patients."""
    import argparse

    import pandas as pd
    from matplotlib import pyplot as plt
    from tqdm.auto import tqdm
    from vital.data.cardinal.config import CardinalTag
    from vital.data.cardinal.utils.attributes import TIME_SERIES_ATTR_LABELS
    from vital.data.cardinal.utils.itertools import Patients
    from vital.utils.logging import configure_logging
    from vital.utils.parsing import yaml_flow_collection
    from vital.utils.signal.decomposition import (
        analyze_pca_wrt_n_components,
        sweep_embedding_dims,
        visualize_embedding_pairwise_dims,
    )

    logger = logging.getLogger(__name__)

    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the time-series attributes",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=128,
        help="Number of points in attributes' data to reach by linearly interpolating the original data, corresponding "
        "to the number of input features for the PCA",
    )
    parser.add_argument("--n_components", type=int, default=8, help="Dimensionality of the PCA model's latent space")
    parser.add_argument(
        "--pca_kwargs",
        type=yaml_flow_collection,
        metavar="{ATTR1:VAL1,ATTR2:VAL2,...}",
        help="Parameters that will be passed along to the `PCA`'s init",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["global_pca", "attr_pca"],
        default="global_pca",
        help="Strategies available for handling multi-domain time-series attributes values. \n"
        "'global_pca': one global PCA model is trained on the normalized attributes. \n"
        "'attr_pca': one PCA model is fitted to each attribute, side-stepping the need for normalization.",
    )
    parser.add_argument(
        "--sweep_bounds",
        type=float,
        default=1,
        help="Factor defining the bounds for each sweep around an average sample when interpreting PCA components. The "
        "bounds are defined as `mean ± margin * stddev`, where stddev is the standard deviation of the PCA dimension.",
    )
    parser.add_argument(
        "--sweep_num_steps",
        type=int,
        default=2,
        help="Number of steps to sweep between the average sample and the min/max bounds when interpreting PCA "
        "components, for each dimension to sweep",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd() / "cardiac_sequence_attrs_pca",
        help="Root directory under which to save the model and the plots analyzing its behavior",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    mask_tag, n_features, n_components, pca_kwargs, strategy, sweep_bounds, sweep_num_steps, output_dir = (
        kwargs.pop("mask_tag"),
        kwargs.pop("n_features"),
        kwargs.pop("n_components"),
        kwargs.pop("pca_kwargs"),
        kwargs.pop("strategy"),
        kwargs.pop("sweep_bounds"),
        kwargs.pop("sweep_num_steps"),
        kwargs.pop("output_dir"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare patients' data for fitting the model
    normalize_length = Interp1d(n_features)
    patients = Patients(**kwargs)
    attrs_data = {
        (patient.id, view, attr): normalize_length(attr_data)
        for patient in tqdm(patients.values(), desc="Collecting attributes' data from patients", unit="patient")
        for view, view_attrs in patient.get_mask_attributes(mask_tag).items()
        for attr, attr_data in view_attrs.items()
    }
    attrs_df = pd.DataFrame(
        attrs_data.values(), index=pd.MultiIndex.from_tuples(attrs_data.keys(), names=["patient", "view", "attr"])
    )
    attr_keys = attrs_df.droplevel("patient").index.unique()
    samples = {attr_key: attrs_df[attrs_df.index.droplevel("patient") == attr_key].to_numpy() for attr_key in attr_keys}

    logger.info("Fitting PCA model to attributes' data...")
    model = CardiacSequenceAttributesPCA(n_components, pca_kwargs=pca_kwargs, strategy=strategy).fit(samples)

    model_save_path = output_dir / "pca-model.pickle"
    logger.info(f"Saving fitted PCA model to '{model_save_path}'...")
    model.save(model_save_path)

    logger.info("Analyzing PCA results over its training data...")
    # Ensure that matplotlib is using 'agg' backend
    # to avoid possible leak of file handles if matplotlib defaults to another backend
    plt.switch_backend("agg")

    sweep_coeffs = {
        f"{coeff:+.1f}σ": coeff
        for coeff in np.linspace(-sweep_bounds, sweep_bounds, num=(sweep_num_steps * 2) + 1)
        if coeff != 0
    }

    def _save_cur_fig(title: str, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        title_pathified = title.lower().replace("/", "_").replace(" ", "_")
        plt.savefig(folder / f"{title_pathified}.png")
        plt.close()  # Close the figure to avoid contamination between plots

    samples_embedding = {}
    for attr_key, attr_samples in tqdm(samples.items(), desc="Analysing PCA results over each attribute", unit="attr"):
        samples_embedding[attr_key] = attr_samples_embedding = model(attr_samples, task="encode", attr=attr_key)

        # Analyze PCA variance explanation
        match model.strategy:
            case "global_pca":
                pca = model.pca
            case "attr_pca":
                pca = model.pca[attr_key]
        title, plot = analyze_pca_wrt_n_components(pca)
        _save_cur_fig("_".join((*attr_key, title)), output_dir / "variance")

        # Plot reconstructions of average samples +/- some coefficient of stddev, for each PCA dimension
        for title, plot in sweep_embedding_dims(
            lambda x: model(x, task="decode", attr=attr_key),
            attr_samples_embedding,
            sweep_coeffs,
            plots_kwargs={"ylabel": TIME_SERIES_ATTR_LABELS[attr_key[1]]},
        ):
            _save_cur_fig(title, output_dir / "sweep" / "_".join(attr_key))

    # Visualize embedding by plotting scatter plots over pairs of dimensions at a time
    samples_embedding_df = (
        pd.concat(
            {
                attr_key: pd.DataFrame(attr_samples_embedding)
                for attr_key, attr_samples_embedding in samples_embedding.items()
            }
        )
        .droplevel(-1)
        .rename_axis(["view", "attr"])
    )
    for title, plot in tqdm(
        visualize_embedding_pairwise_dims(samples_embedding_df, plots_kwargs={"hue": "attr", "style": "view"}),
        desc="Plotting embedding pairwise dimensions",
        unit="pair",
    ):
        _save_cur_fig(title, output_dir / "embedding")


if __name__ == "__main__":
    main()
