import itertools
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from vital.data.cardinal.utils.data_struct import View
from vital.data.cardinal.utils.itertools import Views
from vital.results.metrics import Metrics
from vital.utils.parsing import yaml_flow_collection
from vital.utils.saving import load_from_checkpoint

from didactic.tasks.cardiac_sequence_attrs_ae import CardiacSequenceAttributesAutoencoder

# Should not delete this import even if it is not used directly, since it's necessary for unpickling
# serialized `CardiacSequenceAttributesPCA` models
from didactic.tasks.cardiac_sequence_attrs_pca import CardiacSequenceAttributesPCA  # noqa


class TimeSeriesAttributesMetrics(Metrics):
    """Class that measures reconstruction performance of manifold learning models applied to time-series attributes."""

    desc = "img_attrs_scores"
    ResultsCollection = Views
    ProcessingOutput = Dict[str, float]

    def __init__(self, models: Dict[str, str | Path], **kwargs):
        """Initializes class instance.

        Args:
            models: Mapping between model IDs and path to their checkpoint or name in a Comet model registry.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self.models = {}
        for model_tag, ckpt in models.items():
            ckpt = Path(ckpt)
            if ckpt.suffix == ".pickle":
                with open(ckpt, mode="rb") as ckpt_file:
                    model = pickle.load(ckpt_file)
            else:
                model = load_from_checkpoint(ckpt, expected_checkpoint_type=CardiacSequenceAttributesAutoencoder)
            self.models[model_tag] = model

    def process_result(self, result: View) -> Tuple[str, "ProcessingOutput"]:
        """Computes reconstruction metrics on time-series attributes from a sequence.

        Args:
            result: Data structure holding all the relevant information to compute the requested metrics for a single
                sequence.

        Returns:
            - Identifier of the sequence for which the metrics where computed.
            - Mapping between the metrics and their value for the instant.
        """
        # Extract the attributes' data from the result, and predict the reconstruction using each model
        time_series_attrs = result.get_mask_attributes(self.input_tag)
        with torch.inference_mode():
            models_predictions = {
                model_tag: {
                    attr: model(torch.tensor(attr_data[None, :], dtype=torch.float), attr=(result.id.view, attr))
                    .squeeze(dim=0)
                    .cpu()
                    .numpy()
                    for attr, attr_data in time_series_attrs.items()
                }
                for model_tag, model in self.models.items()
            }

        # Compute the reconstruction metrics between models' predictions, and between models' predictions and the input
        # data
        def _mae(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
            return np.mean(np.abs(array1 - array2))

        metrics = {}
        for attr, attr_data in time_series_attrs.items():
            for model1, model2 in itertools.combinations(self.models, 2):
                metrics[f"{model1}_{model2}_{attr}_mae"] = _mae(
                    models_predictions[model1][attr], models_predictions[model2][attr]
                )
            metrics.update(
                {f"{attr}_{model}_mae": _mae(models_predictions[model][attr], attr_data) for model in self.models}
            )

        return result.id, metrics

    def _aggregate_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Computes global statistics for the metrics computed over each result.

        Args:
            metrics: Metrics computed over each result.

        Returns:
            Global statistics for the metrics computed over each result.
        """
        return metrics.agg(["mean", "std", "min", "max"])

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser with support for time-series attributes reconstruction metrics.

        Returns:
            Parser object with support for time-series attributes reconstruction metrics.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--models",
            type=yaml_flow_collection,
            required=True,
            metavar="{MODEL1:CKPT1,MODEL2:CKPT2,...}",
            help="Mapping between model IDs and path to their checkpoint or name in a Comet model registry",
        )
        return parser


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()  # Load .env to configure Comet API key in case some models come from a Comet model registry
    TimeSeriesAttributesMetrics.main()
