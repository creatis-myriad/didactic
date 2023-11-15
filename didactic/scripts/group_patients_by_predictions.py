def main():
    """Run the script."""

    from argparse import ArgumentParser
    from pathlib import Path

    import numpy as np
    from vital.data.cardinal.config import CardinalTag
    from vital.data.cardinal.utils.itertools import Patients
    from vital.utils.saving import load_from_checkpoint

    from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask
    from didactic.tasks.utils import encode_patients

    parser = ArgumentParser("Script to create bins of patients w.r.t. a continuous prediction")
    parser.add_argument(
        "pretrained_encoder",
        type=Path,
        help="Path to model checkpoint, or name of a model from a Comet model registry, of an encoder",
    )
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the time-series attributes",
    )
    parser.add_argument(
        "--encoding_task",
        type=str,
        default="unimodal_param",
        choices=["unimodal_param"],
        help="Encoding task used to output the continuous prediction w.r.t. which to bin the patients",
    )
    parser.add_argument("--bins", type=int, default=8, help="Number of bins to group the patients into")
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=2,
        help="Bounds of the prediction to bin the patients by. If not provided, will default to the min and max values "
        "of the prediction",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the root directory where to save the lists of patients in each created bin",
    )
    args = parser.parse_args()

    kwargs = vars(args)
    pretrained_encoder, mask_tag, encoding_task, num_bins, bounds, output_dir = list(
        map(kwargs.pop, ["pretrained_encoder", "mask_tag", "encoding_task", "bins", "bounds", "output_dir"])
    )

    # Compute the predictions for the patients
    patients = Patients(**kwargs)
    encoder = load_from_checkpoint(pretrained_encoder, expected_checkpoint_type=CardiacMultimodalRepresentationTask)
    target_attr = encoder.hparams.target_tabular_attrs[0]  # Take the first target as the prediction
    predictions = encode_patients(encoder, patients.values(), mask_tag=mask_tag, progress_bar=True, task=encoding_task)[
        target_attr
    ]

    # Determine the bounds of the bins
    if not bounds:
        bounds = min(predictions), max(predictions)
    # Compute the bounds of the bins
    bins = np.linspace(*bounds, num=num_bins + 1)
    bins[-1] += 1e-6  # Add epsilon to the last bin's upper bound since it's excluded by `np.digitize`

    # Assign the patients to the appropriate bins, based on their predictions
    patients_bins = np.digitize(predictions, bins) - 1  # Subtract 1 because bin indexing starts at 1

    # Save the lists of patients in each bin to plain text files
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_ids = np.array(list(patients.keys()))
    for bin_idx in range(num_bins):
        bin_patients = sorted(patient_ids[patients_bins == bin_idx])
        (output_dir / f"{bin_idx}.txt").write_text("\n".join(bin_patients))


if __name__ == "__main__":
    main()
