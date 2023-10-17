import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks.prediction_writer import WriteInterval
from torch import Tensor
from vital.data.cardinal.config import ImageAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.data_module import PREDICT_DATALOADERS_SUBSETS
from vital.data.cardinal.datapipes import PatientData, filter_image_attributes
from vital.data.cardinal.utils.attributes import build_attributes_dataframe, plot_attributes_wrt_time
from vital.utils.loggers import log_figure
from vital.utils.plot import embedding_scatterplot

from didactic.tasks.cardiac_sequence_attrs_ae import CardiacSequenceAttributesAutoencoder


class CardiacSequenceAttributesPredictionWriter(BasePredictionWriter):
    """Prediction writer that plots reconstructed image attributes and plots the latent space manifold."""

    def __init__(self, write_path: str | Path = None, embedding_kwargs: Dict[str, Any] = None):
        """Initializes class instance.

        Args:
            write_path: Root directory under which to save the predictions / analysis plots.
            embedding_kwargs: Parameters to pass along to the PaCMAP embedding.
        """
        super().__init__(write_interval=WriteInterval.BATCH_AND_EPOCH)
        self._write_path = Path(write_path) if write_path else None
        self._embedding_kwargs = {} if embedding_kwargs is None else embedding_kwargs

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Removes results potentially left behind by previous runs of the callback in the same directory."""
        # Write to the same directory as the experiment logger if no custom path is provided
        if self._write_path is None:
            self._write_path = pl_module.log_dir / "predictions_plots"

        # Assign a subdirectory for each dataloader/subset to predict on
        self._dataloaders_write_path = [self._write_path / subset for subset in PREDICT_DATALOADERS_SUBSETS]

        # Delete leftover predictions from previous run
        shutil.rmtree(self._write_path, ignore_errors=True)

        # Ensure that matplotlib is using 'agg' backend
        # to avoid possible leak of file handles if matplotlib defaults to another backend
        plt.switch_backend("agg")

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: CardiacSequenceAttributesAutoencoder,
        prediction: Dict[Tuple[ViewEnum, ImageAttribute], Tuple[Tensor, Tensor]],
        batch_indices: Optional[Sequence[int]],
        batch: PatientData,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Saves plots of the attributes' reconstructed curves vs their input curves.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            prediction: Mapping between attributes' keys and tuples of i) their reconstructions and ii) their encodings
                in the latent space.
            batch_indices: Indices of all the batches whose outputs are provided.
            batch: The current batch used by the model to give its prediction.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """
        patient_id = list(trainer.datamodule.subsets_patients[PREDICT_DATALOADERS_SUBSETS[dataloader_idx]])[batch_idx]

        # Collect the attributes predictions and convert them to numpy arrays
        img_attrs_reconstructions = {
            attr_key: attr_prediction[0].cpu().numpy() for attr_key, attr_prediction in prediction.items()
        }
        # Collect the attributes data and convert it to numpy arrays,
        # only keeping the attributes for which we have predictions
        img_attrs = {
            attr_key: attr.cpu().numpy()
            for attr_key, attr in filter_image_attributes(batch).items()
            if attr_key in img_attrs_reconstructions
        }
        attrs = {"data": img_attrs, "pred": img_attrs_reconstructions}

        # Plot the curves for each attribute w.r.t. time
        attrs_df = build_attributes_dataframe(attrs, normalize_time=True)
        for title, plot in plot_attributes_wrt_time(attrs_df, plot_title_root=patient_id):
            batch_dir = self._dataloaders_write_path[dataloader_idx] / patient_id
            batch_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(batch_dir / f"{title}.png")
            plt.close()  # Close the figure to avoid contamination between plots

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: CardiacSequenceAttributesAutoencoder,
        predictions: Sequence[Sequence[Dict[Tuple[ViewEnum, ImageAttribute], Tuple[Tensor, Tensor]]]],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        """Saves plots of the distribution of attributes' encodings.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            predictions: Sequences of predictions for each patient, with the content of the predictions for each patient
                detailed in the docstring for `write_on_batch_end`. There is one sublist for each prediction dataloader
                provided.
            batch_indices: Indices of all the batches whose outputs are provided.
        """
        # Build a dataframe for the encodings of the whole dataset, with some metadata about each encoding to be able
        # to visualize the distribution of encodings w.r.t. this metadata
        encodings = {
            (subset, patient_id, *attr_key): attr_predictions[1].cpu().numpy()
            # For each prediction dataloader
            for subset, subset_predictions in zip(PREDICT_DATALOADERS_SUBSETS, predictions)
            # For each batch of data in a dataloader
            for patient_id, patient_predictions in zip(trainer.datamodule.subsets_patients[subset], subset_predictions)
            # For each attribute prediction in the batch
            for attr_key, attr_predictions in patient_predictions.items()
        }
        encodings_df = pd.DataFrame(
            encodings.values(),
            index=pd.MultiIndex.from_tuples(encodings.keys(), names=["subset", "patient", "view", "attr"]),
        )

        plots = {
            "latent_space_by_attrs": {"hue": "attr", "style": "view"},
            "latent_space_by_subsets": {"hue": "subset"},
        }
        for plot_filename, _ in zip(
            plots,
            embedding_scatterplot(encodings_df, plots.values(), data_tag="latent space", **self._embedding_kwargs),
        ):
            # Log the plots using the experiment logger
            log_figure(trainer.logger, figure_name=plot_filename)

            # Save the plots locally
            plt.savefig(self._write_path / f"{plot_filename}.png")
            plt.close()  # Close the figure to avoid contamination between plots
