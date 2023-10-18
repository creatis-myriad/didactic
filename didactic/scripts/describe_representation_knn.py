import itertools
import logging
from pathlib import Path
from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib.axes import Axes
from sklearn.neighbors import NearestNeighbors
from vital.data.cardinal.config import CardinalTag, ClinicalAttribute, ImageAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.attributes import CLINICAL_ATTR_UNITS, IMAGE_ATTR_LABELS
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients

from didactic.data.cardinal.utils import build_img_attr_by_patient_group_dataframe, build_knn_dataframe
from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask

logger = logging.getLogger(__name__)


def find_nearest_neighbors(
    patients_encodings: Sequence[np.ndarray], patient_ids: Sequence[Patient.Id], **neigh_kwargs
) -> np.ndarray:
    """Finds the nearest neighbors of each patient based on the encodings from each model.

    Args:
        patients_encodings: Encodings of the patients corresponding to multiple representations of the population.
        patient_ids: IDs of the patients to use when identifying the nearest neighbors. Should be of the same length as
            every array in `patients_encodings`.
        neigh_kwargs: Keyword arguments to forward to the `KNeighborsClassifier` constructor.

    Returns:
        Array (of `Patient.Id`s) of shape `(n_encodings, n_patients, n_neighbors)` containing the IDs of the nearest
        neighbors of each patient for each encoding.
    """
    # Check that the number of patients is the same for each encoding, and matches the number of patient IDs
    for idx, encoding in enumerate(patients_encodings):
        if len(encoding) != len(patient_ids):
            raise ValueError(
                f"The number of patients in the encoding for model #{idx} does not match the number of patient IDs"
            )

    # For each encoding, fit a nearest neighbors model on the encoding and find the nearest neighbors of each patient
    kneighbors_indices = np.stack(
        [
            NearestNeighbors(**neigh_kwargs).fit(encoding).kneighbors(return_distance=False)
            for encoding in patients_encodings
        ]
    )

    # Convert the indices of the nearest neighbors to patient IDs
    kneighbors_ids = np.array(patient_ids)[kneighbors_indices]

    return kneighbors_ids


def plot_clinical_attrs_variability_figures(
    patients: Patients,
    kneighbors_ids: np.ndarray,
    var_attr: ClinicalAttribute,
    clinical_attrs: Sequence[ClinicalAttribute] = None,
    agg: str = "mean",
    plot_kwargs: dict = None,
    dots_layer_kwargs: dict = None,
    polyfit_layer_kwargs: dict = None,
) -> Iterator[Tuple[str, so.Plot]]:
    """Plots the variability of locally-aggregated clinical attrs across multiple encodings w.r.t. a ref. clinical attr.

    Args:
        patients: Collection of patients data from which to extract the attributes.
        kneighbors_ids: Array (of `Patient.Id`s) of shape `(n_encodings, n_patients, n_neighbors)` containing the IDs of
            the nearest neighbors of each patient for each encoding.
        var_attr: Reference clinical attribute w.r.t. which to plot the variability of the attributes.
        clinical_attrs: Subset of clinical attributes on which to compile the results. If not provided, will default to
            all available attributes.
        agg: Aggregation function to apply to the neighborhood of each patient.
        plot_kwargs: Parameters to forward to the call to `seaborn.object.Plot`.
        dots_layer_kwargs: Parameters to forward to the call to `seaborn.object.Plot.add` for the scatter plot layer.
        polyfit_layer_kwargs: Parameters to forward to the call to `seaborn.object.Plot.scale` for the polynomial fit
            layer.

    Returns:
        Iterator over figures (and their corresponding titles) plotting the variability of locally-aggregated clinical
        attrs across multiple encodings w.r.t. a ref. clinical attr.
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    if dots_layer_kwargs is None:
        dots_layer_kwargs = {}
    if polyfit_layer_kwargs is None:
        polyfit_layer_kwargs = {}

    # Gather the data of the nearest neighbors of each patient for each encoding
    neigh_data = build_knn_dataframe(patients, kneighbors_ids, cat_to_num=True)
    if clinical_attrs is not None:
        neigh_data = neigh_data[clinical_attrs]

    # Compute the mean of the attributes over the nearest neighbors of each patient
    neigh_agg_data = neigh_data.groupby(level=["model", "patient_id"]).agg(agg)

    # Ignore `matplotlib.category` logger 'INFO' level logs to avoid repeated logs about categorical units parsable
    # as floats
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

    # For each attribute (regularized by aggregation across neighbors),
    # generate a scatter plot of the attribute w.r.t. the reference attribute
    for attr in neigh_agg_data.columns:
        # Drop rows w/ NA values for the attribute to plot to avoid "Cannot cast ufunc 'lstsq_n'" error
        # (the error happens when trying to fit least squares polynomial regression if NA values are present)
        # This really only affects attributes with lots of missing values since agg over neighbors typically
        # "fills in the gaps" for attributes with few missing values
        attr_data = neigh_agg_data.dropna(subset=[attr])

        plot = (
            so.Plot(data=attr_data, x=var_attr, y=attr, **plot_kwargs)
            .add(so.Dots(), so.Jitter(0.3), **dots_layer_kwargs)
            .add(so.Line(), so.PolyFit(), **polyfit_layer_kwargs)
        )

        title = f"{attr}_wrt_{var_attr}"
        axis_labels = {
            axis: "(ratio true/false)" if attr in ClinicalAttribute.boolean_attrs() else CLINICAL_ATTR_UNITS[attr][0]
            for axis, attr in zip(["x", "y"], (var_attr, attr))
        }
        plot = plot.label(title=title, **axis_labels)

        yield title, plot


def plot_img_attrs_variability_figures(
    patients: Patients,
    kneighbors_ids: np.ndarray,
    var_attr: ClinicalAttribute,
    image_attrs: Sequence[Tuple[ViewEnum, ImageAttribute]],
    agg: str = "mean",
    mask_tag: str = CardinalTag.mask,
    n_bins: int = 5,
) -> Iterator[Tuple[str, Axes]]:
    """Plots the variability of locally-aggregated image attrs across multiple encodings w.r.t. a ref. clinical attr.

    Args:
        patients: Collection of patients data from which to extract the attributes.
        kneighbors_ids: Array (of `Patient.Id`s) of shape `(n_encodings, n_patients, n_neighbors)` containing the IDs of
            the nearest neighbors of each patient for each encoding.
        var_attr: Reference clinical attribute w.r.t. which to plot the variability of the attributes.
        image_attrs: Subset of image-based attributes derived from segmentations (identified by view/attribute pairs)
            for which to plot the variability between bins of the reference clinical attribute.
        agg: Aggregation function to apply to the neighborhood of each patient.
        mask_tag: Tag of the segmentation mask for which to extract the image attributes.
        n_bins: Number of bins by which to divide the population and over which to compute the variability of the image
            attributes.

    Returns:
        Iterator over figures (and their corresponding titles) plotting the variability of locally-aggregated image
        attrs across multiple encodings w.r.t. a ref. clinical attr.
    """
    # Gather the reference attribute data of the nearest neighbors of each patient for each encoding
    neigh_data = build_knn_dataframe(patients, kneighbors_ids, cat_to_num=True)[var_attr]

    # Compute the mean of the reference attribute over the nearest neighbors of each patient
    neigh_agg_var = neigh_data.groupby(level=["model", "patient_id"]).agg(agg)

    # Divide the population into bins based on the reference attribute
    bins = np.linspace(min(neigh_agg_var), max(neigh_agg_var), num=n_bins + 1)
    bins[-1] += 1e-6  # Add epsilon to the last bin's upper bound since it's excluded by `np.digitize`
    bin_labels = np.digitize(neigh_agg_var, bins) - 1  # Subtract 1 because bin indexing starts at 1
    # Since the bin labels are attributed based on the aggregation of neighbors, repeat the labels `n_neighbors` times
    # to obtain a list of labels for each neighbor of each patient for each kneighbors model
    bin_labels = pd.Series(np.repeat(bin_labels, kneighbors_ids.shape[-1], axis=0), index=neigh_data.index)

    # For each bin, flatten the list of neighbors of each patient in the bin
    # (to obtain lists of patients, with duplicates, in each bin)
    patient_ids_by_bin = {
        bin_idx: bin_labels[bin_labels == bin_idx].index.get_level_values("neighbor_id").tolist()
        for bin_idx in sorted(bin_labels.unique())
    }
    patients_by_bin = {
        bin_idx: list(Patients.from_dict({patient_id: patients[patient_id] for patient_id in patient_ids}).values())
        for bin_idx, patient_ids in patient_ids_by_bin.items()
    }

    # For each image attribute, build the dataframe of the mean curve for each bin and plot the curves for each bin
    for img_attr in image_attrs:
        neigh_agg_img_attrs_data = build_img_attr_by_patient_group_dataframe(
            patients_by_bin, img_attr, group_desc="bin", mask_tag=mask_tag
        )

        with sns.axes_style("darkgrid"):
            plot = sns.lineplot(
                data=neigh_agg_img_attrs_data, x="time", y="val", hue="bin", hue_order=sorted(bin_labels.unique())
            )
        title = f"{'/'.join(img_attr)}_wrt_{var_attr}_bins"
        plot.set(title=title, ylabel=IMAGE_ATTR_LABELS[img_attr[1]])
        plot.legend(title=f"{var_attr} bin")

        yield title, plot


def main():
    """Run the script."""
    from argparse import ArgumentParser

    from matplotlib import pyplot as plt
    from tqdm import tqdm
    from vital.utils.logging import configure_logging
    from vital.utils.parsing import yaml_flow_collection
    from vital.utils.saving import load_from_checkpoint

    from didactic.tasks.utils import encode_patients

    # Ensure that matplotlib is using 'agg' backend in non-interactive case
    plt.switch_backend("agg")

    configure_logging(log_to_console=True, console_level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument(
        "models_ckpts",
        nargs="+",
        type=Path,
        help="Checkpoints of models for which to compare and compile the local neighborhood",
    )
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--neigh_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the `KNeighborsClassifier` constructor",
    )
    parser.add_argument(
        "--reference_attr",
        type=ClinicalAttribute,
        default=ClinicalAttribute.ht_severity,
        help="Reference clinical attribute w.r.t. which to plot the variability of the other attributes",
    )
    parser.add_argument(
        "--clinical_attrs",
        type=ClinicalAttribute,
        nargs="*",
        choices=list(ClinicalAttribute),
        help="Subset of clinical attributes on which to compile the results. If not provided, will default to all "
        "available attributes",
    )
    parser.add_argument(
        "--image_attrs",
        type=ImageAttribute,
        choices=list(ImageAttribute),
        nargs="*",
        default=list(ImageAttribute),
        help="Subset of image-based attributes derived from segmentations for which to plot the intra/inter-cluster "
        "variability",
    )
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the image attributes",
    )
    parser.add_argument(
        "--agg",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="Aggregation function to apply to the neighborhood of each patient",
    )
    parser.add_argument(
        "--clinical_plot_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.object.Plot` for clinical attributes figures",
    )
    parser.add_argument(
        "--clinical_dots_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.object.Plot.add` for the scatter plot layer in clinical "
        "attributes figures",
    )
    parser.add_argument(
        "--clinical_polyfit_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.object.Plot.add` for the polynomial regression layer in "
        "clinical attributes figures",
    )
    parser.add_argument(
        "--image_n_bins",
        type=int,
        default=5,
        help="Number of bins by which to divide the population and over each of which to aggregate the image "
        "attributes",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("compiled_representation_knn"),
        help="Root directory under which to save the compiled results for all of the methods",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    (
        models_ckpts,
        neigh_kwargs,
        ref_attr,
        clinical_attrs,
        image_attrs,
        mask_tag,
        agg,
        plot_kwargs,
        dots_kwargs,
        polyfit_kwargs,
        img_n_bins,
        output_dir,
    ) = list(
        map(
            kwargs.pop,
            [
                "models_ckpts",
                "neigh_kwargs",
                "reference_attr",
                "clinical_attrs",
                "image_attrs",
                "mask_tag",
                "agg",
                "clinical_plot_kwargs",
                "clinical_dots_kwargs",
                "clinical_polyfit_kwargs",
                "image_n_bins",
                "output_dir",
            ],
        )
    )
    image_attrs_keys = [(view, image_attr) for view, image_attr in itertools.product(args.views, image_attrs)]
    if neigh_kwargs is None:
        neigh_kwargs = {}

    # Load the dataset
    patients = Patients(**kwargs)

    # Load the models and process the patients to generate the encodings
    models = [
        load_from_checkpoint(ckpt, expected_checkpoint_type=CardiacMultimodalRepresentationTask)
        for ckpt in models_ckpts
    ]
    encodings = [
        encode_patients(model, patients.values(), mask_tag=mask_tag)
        for model in tqdm(models, desc="Encoding patients using each model", unit="model")
    ]

    # Find the nearest neighbors of each patient for each encoding
    kneighbors_ids = find_nearest_neighbors(encodings, list(patients), **neigh_kwargs)

    clinical_attrs_plots = plot_clinical_attrs_variability_figures(
        patients,
        kneighbors_ids,
        ref_attr,
        clinical_attrs=clinical_attrs,
        agg=agg,
        plot_kwargs=plot_kwargs,
        dots_layer_kwargs=dots_kwargs,
        polyfit_layer_kwargs=polyfit_kwargs,
    )
    image_attrs_plots = plot_img_attrs_variability_figures(
        patients, kneighbors_ids, ref_attr, image_attrs_keys, agg=agg, mask_tag=mask_tag, n_bins=img_n_bins
    )

    # Plot the variability of the clinical and image attributes
    output_dir.mkdir(parents=True, exist_ok=True)  # Prepare the output folder for the method
    n_plots = (len(clinical_attrs) if clinical_attrs else len(ClinicalAttribute)) + len(image_attrs_keys)
    for title, plot in tqdm(
        itertools.chain(clinical_attrs_plots, image_attrs_plots),
        desc=f"Plotting the variability of the attributes w.r.t. {ref_attr}",
        unit="attr",
        total=n_plots,
    ):
        title_pathified = title.lower().replace("/", "_").replace(" ", "_")
        filepath = output_dir / f"{title_pathified}.svg"

        if isinstance(plot, so.Plot):
            plot.save(filepath, bbox_inches="tight")
        elif isinstance(plot, Axes):
            plt.savefig(filepath)
            plt.close()  # Close the figure to avoid contamination between plots
        else:
            raise ValueError(f"Unable to save the figure for plot type: {type(plot)}.")


if __name__ == "__main__":
    main()
