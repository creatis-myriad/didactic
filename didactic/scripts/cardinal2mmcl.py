from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL.Image import Resampling
from vital.data.cardinal.config import CardinalTag, Label, TabularAttribute
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients
from vital.utils.image.transform import resize_image
from vital.utils.image.us.measure import EchoMeasure
from vital.utils.norm import minmax_scaling
from vital.utils.tabular import impute_missing_tabular_data


def _normalize_and_resize_image(
    image: np.ndarray, size: Tuple[int, int], norm_bounds: Tuple[int, int] = (0, 1)
) -> np.ndarray:
    """Normalizes and resizes an image.

    Args:
        image: (H, W), Image to normalize and resize.
        size: Target (H, W) for the input image.
        norm_bounds: (min, max) bounds for the normalization.

    Returns:
        (H, W), Normalized and resized image.
    """
    resized_image = resize_image(image, size=size, resample=Resampling.BILINEAR)
    normalized_image = minmax_scaling(resized_image, bounds=norm_bounds)
    return normalized_image


def extract_patient_images(patient: Patient, **img_transform_kwargs) -> List[np.ndarray]:
    """Extracts specific frames (i.e. ED and ES) from the different sequences available for the patient.

    Args:
        patient: Patient to extract frames from.
        img_transform_kwargs: Keyword arguments to pass to the image transformation function.

    Returns:
        List of images extracted from the patient.
    """
    images = []
    for view_data in patient.views.values():
        img, mask = view_data.data[CardinalTag.bmode], view_data.data[CardinalTag.mask]

        ed_frame = img[0]  # The first frame is the ED frame

        es_frame_idx = np.argmin(EchoMeasure.structure_area(mask, labels=Label.LV))
        es_frame = img[es_frame_idx]  # The ES frame is the one with the smallest LV area

        images.extend([ed_frame, es_frame])

    return [_normalize_and_resize_image(img, **img_transform_kwargs) for img in images]


def save_images_and_tabular_data(
    images: np.ndarray,
    tabular_df: pd.DataFrame,
    label_tag: TabularAttribute,
    output_dir: Path,
    label_as_a_feature: bool = True,
    tabular_tag: str = "tabular",
    subsets: Dict[str, Sequence[int]] = None,
) -> None:
    """Serializes the images to disk using `torch.save`, and saves tabular data to a CSV file.

    Args:
        images: (N, C, H, W), Array images to save.
        tabular_df: Tabular data to save.
        label_tag: Tabular variable to use as the label.
        output_dir: Directory to save the data to.
        label_as_a_feature: Whether to keep the label as a feature (LaaF) in the tabular data.
        tabular_tag: Tag to use to describe the selection of tabular features in the tabular data files.
        subsets: Optional dictionary mapping indices of images/rows in specific subsets. If provided, the subsets will
            be saved to separate 'pt' and CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split the label from the rest of the tabular data, convert it to integer labels
    labels = tabular_df[label_tag.value].cat.codes.to_numpy(copy=True)
    if not label_as_a_feature:
        tabular_df = tabular_df.drop(columns=[label_tag.value])
    else:
        tabular_tag += "_laaf"

    # Inspect the dtypes of the tabular data
    cat_df = tabular_df.select_dtypes(include="category")
    boolean_df = tabular_df.select_dtypes(include=bool)

    # Infer the "length" of the one-hot encoding of each field in the tabular data
    field_lengths = [
        len(cat_df[col].cat.categories) if col in cat_df else 2 if col in boolean_df else 1  # continuous variable
        for col, dtype in tabular_df.dtypes.items()
    ]
    field_lengths = np.array(field_lengths)

    # Convert categorical and boolean labels to numerical labels
    tabular_df = tabular_df.assign(**{col_name: col_data.cat.codes for col_name, col_data in cat_df.items()})
    tabular_df = tabular_df.assign(**{col_name: col_data.astype(int) for col_name, col_data in boolean_df.items()})

    # Save the field lengths as an array
    torch.save(torch.from_numpy(field_lengths), output_dir / f"{tabular_tag}_field_lengths.pt")

    # Group data by subset if necessary
    if subsets:
        imgs_by_subset = {f"{name}_images": images[idxs] for name, idxs in subsets.items()}
        tab_by_subset = {f"{name}_{tabular_tag}": tabular_df.iloc[idxs] for name, idxs in subsets.items()}
        labels_by_subset = {f"{name}_{label_tag}": labels[idxs] for name, idxs in subsets.items()}
    else:
        imgs_by_subset = {"images": images}
        tab_by_subset = {tabular_tag: tabular_df}
        labels_by_subset = {label_tag: labels}

    # Save the images as a tensor
    for tag, subset_imgs in imgs_by_subset.items():
        torch.save(torch.from_numpy(subset_imgs), output_dir / f"{tag}.pt")

    # Save the labels as a tensor
    for tag, subset_labels in labels_by_subset.items():
        torch.save(torch.from_numpy(subset_labels.astype(np.int64)), output_dir / f"{tag}.pt")

    # Save the tabular data as a CSV file
    for tag, subset_tab in tab_by_subset.items():
        subset_tab.to_csv(output_dir / f"{tag}.csv", index=False, header=False)


def main():
    """Run the script."""
    import argparse
    import logging

    from vital.utils.logging import configure_logging

    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = argparse.ArgumentParser("Export CARDINAL data to format compatible with MMCL tabular imaging paper.")
    parser = Patients.add_args(parser)
    parser.add_argument("--img_size", type=int, nargs=2, default=(210, 210), help="Size to resize the images to")
    parser.add_argument("--norm_bounds", type=int, nargs=2, default=(0, 1), help="Bounds for min-max normalization")
    parser.add_argument(
        "--tabular_attrs", type=TabularAttribute, nargs="*", help="Tabular attributes to collect and save"
    )
    parser.add_argument(
        "--tabular_tag",
        type=str,
        default="tabular",
        help="Tag to use to describe the selection of tabular features in the tabular data files",
    )
    parser.add_argument(
        "--label_tag",
        type=TabularAttribute,
        default=TabularAttribute.ht_severity,
        help="Tabular variable to use as the label",
    )
    parser.add_argument(
        "--disable_laaf",
        dest="laaf",
        action="store_false",
        help="Disable label as a feature (LaaF), where the label is included in the tabular data",
    )
    parser.add_argument("--imputer_random_state", type=int, default=42, help="Random state for the imputer")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the data to")
    parser.add_argument(
        "--subsets",
        type=Path,
        nargs="*",
        help="Plain text files listing patients belonging to each subset to save independently",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    img_size, norm_bounds, tabular_attrs, tabular_tag, label_tag, laaf, imp_rand_state, output_dir, subsets = (
        kwargs.pop("img_size"),
        kwargs.pop("norm_bounds"),
        kwargs.pop("tabular_attrs"),
        kwargs.pop("tabular_tag"),
        kwargs.pop("label_tag"),
        kwargs.pop("laaf"),
        kwargs.pop("imputer_random_state"),
        kwargs.pop("output_dir"),
        kwargs.pop("subsets"),
    )

    # Load the data
    patients = Patients(**kwargs)

    # Extract the images, only keeping the ED and ES frames
    images = np.stack(
        [
            np.stack(extract_patient_images(patient, size=img_size, norm_bounds=norm_bounds))
            for patient in patients.values()
        ]
    )

    # Extract the tabular data
    tabular_df = impute_missing_tabular_data(
        patients.to_dataframe(tabular_attrs=tabular_attrs), random_state=imp_rand_state
    )

    # Read the lists of patients in each subset from their respective files
    patient_ids = list(patients.keys())
    if subsets:
        subsets = {
            subset_file.stem: [patient_ids.index(patient_id) for patient_id in subset_file.read_text().splitlines()]
            for subset_file in subsets
        }

    # Save the data
    save_images_and_tabular_data(
        images, tabular_df, label_tag, output_dir, label_as_a_feature=laaf, tabular_tag=tabular_tag, subsets=subsets
    )


if __name__ == "__main__":
    main()
