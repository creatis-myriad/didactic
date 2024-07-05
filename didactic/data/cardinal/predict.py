import csv
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks.prediction_writer import WriteInterval
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from torch import Tensor
from vital.data.cardinal.config import TabularAttribute, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.data_module import PREDICT_DATALOADERS_SUBSETS
from vital.data.cardinal.datapipes import PatientData, filter_time_series_attributes
from vital.data.cardinal.utils.attributes import (
    TABULAR_CAT_ATTR_LABELS,
    build_attributes_dataframe,
    plot_attributes_wrt_time,
)
from vital.utils.loggers import log_dataframe, log_figure
from vital.utils.plot import embedding_scatterplot

from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask
from didactic.tasks.cardiac_sequence_attrs_ae import CardiacSequenceAttributesAutoencoder


class CardiacSequenceAttributesPredictionWriter(BasePredictionWriter):
    """Prediction writer that plots reconstructed time-series attributes and plots the latent space manifold."""

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
        prediction: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tuple[Tensor, Tensor]],
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
        time_series_attrs_reconstructions = {
            attr_key: attr_prediction[0].cpu().numpy() for attr_key, attr_prediction in prediction.items()
        }
        # Collect the attributes data and convert it to numpy arrays,
        # only keeping the attributes for which we have predictions
        time_series_attrs = {
            attr_key: attr.cpu().numpy()
            for attr_key, attr in filter_time_series_attributes(batch).items()
            if attr_key in time_series_attrs_reconstructions
        }
        attrs = {"data": time_series_attrs, "pred": time_series_attrs_reconstructions}

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
        predictions: Sequence[Sequence[Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tuple[Tensor, Tensor]]]],
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


class CardiacRepresentationPredictionWriter(BasePredictionWriter):
    """Prediction writer that measures prediction performance for cardiac representation tasks."""

    def __init__(
        self, write_path: str | Path = None, hue_attrs: Sequence[str] = None, embedding_kwargs: Dict[str, Any] = None
    ):
        """Initializes class instance.

        Args:
            write_path: Root directory under which to save the predictions / analysis plots.
            hue_attrs: Attributes to display the scatter plot w.r.t.
            embedding_kwargs: Parameters to pass along to the PaCMAP embedding.
        """
        super().__init__(write_interval=WriteInterval.EPOCH)
        self._write_path = Path(write_path) if write_path else None
        self._hue_attrs = hue_attrs if hue_attrs else []
        self._embedding_kwargs = {} if embedding_kwargs is None else embedding_kwargs

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Removes results potentially left behind by previous runs of the callback in the same directory."""
        # Write to the same directory as the experiment logger if no custom path is provided
        if self._write_path is None:
            self._write_path = pl_module.log_dir / "predictions"

        # Delete leftover predictions from previous run
        shutil.rmtree(self._write_path, ignore_errors=True)

        # Ensure that matplotlib is using 'agg' backend
        # to avoid possible leak of file handles if matplotlib defaults to another backend
        plt.switch_backend("agg")

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: CardiacMultimodalRepresentationTask,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        """Measures and saves prediction performance for cardiac representation tasks.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            predictions: Sequences of encoder output features and predicted tabular attributes for each patient. There
                is one sublist for each prediction dataloader provided.
            batch_indices: Indices of all the batches whose outputs are provided.
        """
        self._write_path.mkdir(parents=True, exist_ok=True)
        self._write_features_plots(trainer, pl_module, predictions)

        # If the model includes prediction heads to predict attributes from the features, analyze the prediction results
        if pl_module.prediction_heads:
            self._write_prediction_scores(trainer, pl_module, predictions)

    def _write_features_plots(
        self, trainer: "pl.Trainer", pl_module: CardiacMultimodalRepresentationTask, predictions: Sequence[Any]
    ) -> None:
        """Plots the distribution of features learned by the encoder w.r.t. tabular attributes and some metadata.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            predictions: Sequences of encoder output features and predicted tabular attributes for each patient. There
                is one sublist for each prediction dataloader provided.
        """
        prediction_example = predictions[0][0]  # 1st: subset, 2nd: batch
        # Pre-compute the list of attributes for which we have an unimodal parameter, since this output might be None
        # and we don't want to access it in that case
        ordinal_attrs = list(prediction_example[2]) if prediction_example[2] else []
        features = {
            (
                subset,
                patient.id,
                *[patient.attrs.get(attr) for attr in self._hue_attrs],
                *[patient_prediction[output_idx][attr].item() for attr in ordinal_attrs for output_idx in (2, 3)],
            ): patient_prediction[0]
            .flatten()
            .cpu()
            .numpy()
            # For each prediction dataloader
            for subset, subset_predictions in zip(PREDICT_DATALOADERS_SUBSETS, predictions)
            # For each batch of data in a dataloader
            for patient, patient_prediction in zip(
                trainer.datamodule.subsets_patients[subset].values(), subset_predictions
            )
        }
        features_df = pd.DataFrame(
            features.values(),
            index=pd.MultiIndex.from_tuples(
                features.keys(),
                names=[
                    "subset",
                    "patient",
                    *self._hue_attrs,
                    *[f"{attr}_unimodal_{pred_desc}" for attr in ordinal_attrs for pred_desc in ("param", "tau")],
                ],
            ),
        )

        # Plot data w.r.t. all indexing data, except for specific patient
        plots = {
            f"features_wrt_{index_name}": {
                "hue": index_name,
                "hue_order": TABULAR_CAT_ATTR_LABELS.get(index_name),  # Use categorical attrs' predefined labels order
            }
            for index_name in features_df.index.names
            if index_name != "patient"
        }
        for plot_filename, _ in zip(
            plots,
            embedding_scatterplot(features_df, plots.values(), data_tag="features", **self._embedding_kwargs),
        ):
            # Log the plots using the experiment logger
            log_figure(trainer.logger, figure_name=plot_filename)

            # Save the plots locally
            plt.savefig(self._write_path / f"{plot_filename}.png")
            plt.close()  # Close the figure to avoid contamination between plots

    def _write_prediction_scores(
        self,
        trainer: "pl.Trainer",
        pl_module: CardiacMultimodalRepresentationTask,
        predictions: Sequence[Any],
    ) -> None:
        """Measure and save prediction scores.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            predictions: Sequences of encoder output features and predicted tabular attributes for each patient. There
                is one sublist for each prediction dataloader provided.
        """
        target_categorical_attrs = [
            attr for attr in pl_module.hparams.predict_losses if attr in TabularAttribute.categorical_attrs()
        ]
        target_numerical_attrs = [
            attr for attr in pl_module.hparams.predict_losses if attr in TabularAttribute.numerical_attrs()
        ]

        for subset, subset_predictions in zip(PREDICT_DATALOADERS_SUBSETS, predictions):
            subset_patients = trainer.datamodule.subsets_patients[subset]

            # Compute metrics on the predictions for all the patients of the subset +
            # collect and structure necessary predictions to compute these metrics
            subset_categorical_data, subset_numerical_data = [], []
            classification_out = {attr: [] for attr in target_categorical_attrs}
            for (patient_id, patient), patient_predictions in zip(subset_patients.items(), subset_predictions):
                attr_predictions = patient_predictions[1]
                if target_categorical_attrs:
                    patient_categorical_data = {"patient": patient_id}
                    for attr in target_categorical_attrs:
                        # Collect the classification logits/probabilities
                        classification_out.setdefault(attr, []).append(attr_predictions[attr].detach().cpu().numpy())
                        # Add the hard prediction and target labels
                        patient_categorical_data.update(
                            {
                                f"{attr}_prediction": TABULAR_CAT_ATTR_LABELS[attr][attr_predictions[attr].argmax()],
                                f"{attr}_target": patient.attrs.get(attr, np.nan),
                            }
                        )
                    subset_categorical_data.append(patient_categorical_data)

                if target_numerical_attrs:
                    patient_numerical_data = {"patient": patient_id}
                    for attr in target_numerical_attrs:
                        patient_numerical_data.update(
                            {
                                f"{attr}_prediction": attr_predictions[attr].item(),
                                f"{attr}_target": patient.attrs.get(attr, np.nan),
                            }
                        )
                    subset_numerical_data.append(patient_numerical_data)

            # Convert the classification logits/probabilities to numpy arrays
            for attr, attr_pred in classification_out.items():
                # Convert to numpy array and ensure float32, to avoid numerical instabilities in case of float16 values
                # coming from AMP models. This is especially important for softmax, which is sensitive to small values.
                attr_pred = np.array(attr_pred, dtype=np.float32)
                if (attr_pred < 0).any() or (attr_pred > 1).any():
                    # If output were logits, compute probabilities from logits
                    attr_pred = softmax(attr_pred, axis=1)
                classification_out[attr] = attr_pred

            if subset_categorical_data:
                subset_categorical_df = pd.DataFrame.from_records(subset_categorical_data, index="patient")
                subset_categorical_stats = subset_categorical_df.describe().drop(["count"])
                # Compute additional custom metrics (i.e. not reported by `describe`) for categorical attributes
                notna_mask = subset_categorical_df.notna()
                subset_categorical_stats.loc["acc"] = {
                    f"{attr}_prediction": accuracy_score(
                        subset_categorical_df[f"{attr}_target"][notna_mask[f"{attr}_target"]],
                        subset_categorical_df[f"{attr}_prediction"][notna_mask[f"{attr}_target"]],
                    )
                    for attr in target_categorical_attrs
                }
                subset_categorical_stats.loc["auroc"] = {
                    f"{attr}_prediction": roc_auc_score(
                        subset_categorical_df[f"{attr}_target"][notna_mask[f"{attr}_target"]],
                        classification_out[attr][notna_mask[f"{attr}_target"]],
                        multi_class="ovr",
                    )
                    for attr in target_categorical_attrs
                }

                # Concatenate the element-wise results + statistics in one dataframe
                subset_categorical_scores = pd.concat([subset_categorical_stats, subset_categorical_df])

            if subset_numerical_data:
                subset_numerical_df = pd.DataFrame.from_records(subset_numerical_data, index="patient")
                subset_numerical_stats = subset_numerical_df.describe(percentiles=[]).drop(["count"])
                # Compute additional custom metrics (i.e. not reported by `describe`) for numerical attributes
                notna_mask = subset_numerical_df.notna()
                subset_numerical_stats.loc["mae"] = {
                    f"{attr}_prediction": mean_absolute_error(
                        subset_numerical_df[f"{attr}_target"][notna_mask[f"{attr}_target"]],
                        subset_numerical_df[f"{attr}_prediction"][notna_mask[f"{attr}_target"]],
                    )
                    for attr in target_numerical_attrs
                }
                subset_numerical_stats.loc["corr"] = {
                    f"{attr}_prediction": stats.pearsonr(
                        subset_numerical_df[f"{attr}_target"][notna_mask[f"{attr}_target"]],
                        subset_numerical_df[f"{attr}_prediction"][notna_mask[f"{attr}_target"]],
                    ).statistic
                    for attr in target_numerical_attrs
                }

                # Concatenate the element-wise results + statistics in one dataframe
                subset_numerical_scores = pd.concat([subset_numerical_stats, subset_numerical_df])

            # Log the prediction scores and statistics using the experiment logger
            prediction_scores_to_log = {}
            if subset_categorical_data:
                prediction_scores_to_log["categorical"] = subset_categorical_scores
            if subset_numerical_data:
                prediction_scores_to_log["numerical"] = subset_numerical_scores
            for tag, prediction_scores in prediction_scores_to_log.items():
                # Log the prediction scores to the (online) experiment logger
                data_filepath = self._write_path / f"{subset}_{tag}_scores.csv"
                log_dataframe(trainer.logger, prediction_scores, filename=data_filepath.name)

                # Save the prediction scores locally
                prediction_scores.to_csv(data_filepath, quoting=csv.QUOTE_NONNUMERIC)
