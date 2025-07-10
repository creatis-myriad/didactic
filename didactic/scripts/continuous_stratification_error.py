import logging

import numpy as np
from vital.data.cardinal.config import CardinalTag
from vital.data.cardinal.utils.attributes import TABULAR_CAT_ATTR_LABELS
from vital.data.cardinal.utils.itertools import Patients

from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask
from didactic.tasks.utils import encode_patients

logger = logging.getLogger(__name__)


def compute_continuous_stratification_error(
    model: CardiacMultimodalRepresentationTask,
    patients: Patients,
    mask_tag: str = CardinalTag.mask,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Compute the distance between misclassified patients' continuous stratification and their true label's threshold.

    Args:
        model: Transformer encoder model to use for inference.
        patients: (N) Patients to embed.
        mask_tag: Tag of the segmentation mask for which to extract the image attributes.
        progress_bar: If ``True``, enables progress bars detailing the progress of encoding patients.

    Returns:
        Distances between misclassified patients' continuous stratification and their true label's threshold, for each
        target attribute.
    """
    # Compute the predictions for each patient
    predictions_by_target = encode_patients(
        model, patients.values(), task="continuum_param", mask_tag=mask_tag, progress_bar=progress_bar
    )
    targets = list(predictions_by_target.keys())

    # Extract the ground truth labels for each patient
    patients_records = patients.to_dataframe(tabular_attrs=targets, cast_to_pandas_dtypes=False)

    dist2true_by_target = {}
    for target, predictions in predictions_by_target.items():
        # Get information about the target, i.e. labels, count, and bin bounds
        target_labels = np.array(TABULAR_CAT_ATTR_LABELS[target])
        n_classes = len(target_labels)
        label_bins = np.linspace(0, 1, num=n_classes + 1)

        # Get the numerical labels of the target attribute
        true_labels = (patients_records[target].to_numpy().reshape(-1, 1) == target_labels[np.newaxis, :]).argmax(
            axis=1
        )

        # Assign the patients to the appropriate bins, based on their predictions
        pred_labels = np.digitize(predictions, label_bins) - 1  # Subtract 1 because bin indexing starts at 1

        # Identify the patients that are misclassified
        error_patients_mask = true_labels != pred_labels
        error_continuous_pred = predictions[error_patients_mask]
        error_labels = true_labels[error_patients_mask]

        # Get the ground truth class' bin bounds for each misclassified patient
        error_label_bounds_idx = np.stack((error_labels, error_labels + 1), axis=-1)
        error_label_bounds = label_bins[error_label_bounds_idx]

        # Compute the minimal distance between the continuous stratification and the ground truth class' bin bounds
        dist2true = np.min(np.abs(error_continuous_pred[:, np.newaxis] - error_label_bounds), axis=1)

        dist2true_by_target[target] = dist2true

    return dist2true_by_target


def main():
    """Run the script."""
    import argparse
    from pathlib import Path

    from vital.utils.logging import configure_logging
    from vital.utils.saving import load_from_checkpoint

    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pretrained_encoder",
        type=Path,
        help="Path to a model checkpoint, or name of a model from a Comet model registry, of an encoder",
    )
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the image attributes",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    encoder_ckpt, mask_tag = kwargs.pop("pretrained_encoder"), kwargs.pop("mask_tag")

    # Load the data and model
    patients = Patients(**kwargs)
    encoder = load_from_checkpoint(encoder_ckpt, expected_checkpoint_type=CardiacMultimodalRepresentationTask)

    continuous_stratification_error = compute_continuous_stratification_error(
        encoder, patients, mask_tag=mask_tag, progress_bar=True
    )
    for target, error in continuous_stratification_error.items():
        logger.info(
            f"Distance between continuous stratification and '{target}' true label's threshold "
            f"(for misclassified patients): mean={error.mean():.3f} median={np.median(error):.3f}"
        )


if __name__ == "__main__":
    main()
