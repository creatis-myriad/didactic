from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from vital.data.cardinal.config import ClinicalAttribute
from vital.data.cardinal.utils.attributes import CLINICAL_CAT_ATTR_LABELS
from vital.data.cardinal.utils.itertools import Patients

NUM_CLINICAL_ATTR_STATS = defaultdict(lambda: "mean")
NUM_CLINICAL_ATTR_STATS.update(
    {
        ClinicalAttribute.ddd: "quartile",
        ClinicalAttribute.creat: "quartile",
        ClinicalAttribute.gfr: "quartile",
        ClinicalAttribute.nt_probnp: "quartile",
        ClinicalAttribute.mv_dt: "quartile",
    }
)
"Numerical attributes' statistics are summarized using `mean ± std, unless other specified here."

NUM_CLINICAL_ATTR_DECIMALS = defaultdict(int)
NUM_CLINICAL_ATTR_DECIMALS.update(
    {
        ClinicalAttribute.bmi: 1,
        ClinicalAttribute.ddd: 1,
        ClinicalAttribute.gfr: 1,
        ClinicalAttribute.e_velocity: 1,
        ClinicalAttribute.a_velocity: 1,
        ClinicalAttribute.e_e_prime_ratio: 1,
        ClinicalAttribute.la_volume: 1,
        ClinicalAttribute.la_area: 1,
        ClinicalAttribute.vmax_tr: 1,
        ClinicalAttribute.ivs_d: 1,
        ClinicalAttribute.lvid_d: 1,
        ClinicalAttribute.pw_d: 1,
        ClinicalAttribute.tapse: 1,
        ClinicalAttribute.s_prime: 1,
    }
)
"""Attributes' statistics are rounded to the nearest integer, unless specified otherwise here."""


def describe_patients(
    patients: Patients, clinical_attrs: Sequence[ClinicalAttribute] = None, format_summary: bool = False
) -> pd.DataFrame:
    """Computes statistics over patients' clinical attributes, adapting statistics to numerical/categorical attributes.

    Args:
        patients: Patients over which to compute the statistics.
        clinical_attrs: Subset of clinical attributes over which to compute the statistics.
        format_summary: Whether to add a column where a subset of the stats (depending on the attribute) are selected
            and formatted, as a summary of that attribute's statistics.

    Returns:
        Statistics describing patients' clinical attributes.
    """
    patients_attrs = patients.to_dataframe(clinical_attrs=clinical_attrs)
    if clinical_attrs is None:
        clinical_attrs = patients_attrs.columns.tolist()

    # Get the descriptions for the numerical attributes, with attributes as rows and descriptions as columns
    num_stats = ["mean", "std", "50%", "25%", "75%"]
    num_attrs_desc = patients_attrs.describe(include=np.number).loc[num_stats]

    # Manually compute the occurrences of label for boolean/categorical attributes
    cat_stats = ["count", "%"]
    cat_attrs = [attr for attr in ClinicalAttribute.categorical_attrs() if attr in patients_attrs.columns]
    cat_attrs_desc = {}
    for attr in cat_attrs:
        attr_data = patients_attrs[attr]
        attr_data = attr_data[attr_data.notna()]  # Discard missing data

        label_counts = {label: (attr_data == label).sum() for label in CLINICAL_CAT_ATTR_LABELS[attr]}
        label_percentages = {label: round(count * 100 / len(attr_data), 1) for label, count in label_counts.items()}
        cat_attrs_desc[attr] = {"count": label_counts, "%": label_percentages}
    # Structure the boolean/categorical description as dataframe, with attributes as rows and descriptions as columns
    cat_attrs_desc = pd.DataFrame.from_dict(cat_attrs_desc)

    # Join descriptions of numerical and categorical attributes
    patients_attrs_desc = num_attrs_desc.join(cat_attrs_desc, how="outer")
    # Index w.r.t. attributes and sort the attributes and statistics
    patients_attrs_desc = patients_attrs_desc.T.reindex(clinical_attrs)[num_stats + cat_stats]

    # Cast numerical stats to float, since the transpose leads all columns to be of generic 'object' type
    patients_attrs_desc[num_stats] = patients_attrs_desc[num_stats].astype(float)

    if format_summary:
        summaries = {}

        for attr in patients_attrs_desc.index:
            if attr in ClinicalAttribute.boolean_attrs():
                attr_summary = (
                    f"{patients_attrs_desc.loc[attr, 'count'][True]} "
                    f"({patients_attrs_desc.loc[attr, '%'][True]:.0f})"
                )
            elif attr in ClinicalAttribute.categorical_attrs():
                attr_summary = "\n".join(
                    f"{cat_count} ({patients_attrs_desc.loc[attr, '%'][cat]:.0f})"
                    for cat, cat_count in patients_attrs_desc.loc[attr, "count"].items()
                )
            else:  # attr in ClinicalAttribute.numerical_attrs():
                dec = NUM_CLINICAL_ATTR_DECIMALS[attr]
                match summary_stat := NUM_CLINICAL_ATTR_STATS[attr]:
                    case "mean":
                        mean, std = patients_attrs_desc[["mean", "std"]].loc[attr]
                        attr_summary = f"{mean:.{dec}f} ± {std:.{dec}f}"
                    case "quartile":
                        median, q1, q3 = patients_attrs_desc[["50%", "25%", "75%"]].loc[attr]
                        attr_summary = f"{median:.{dec}f} ({q1:.{dec}f}-{q3:.{dec}f})"
                    case _:
                        raise ValueError(
                            f"Unexpected value of '{summary_stat}' for the method to use to format the stats for "
                            f"attribute '{attr}'. Use one of: ['mean', 'quartile']."
                        )

            summaries[attr] = attr_summary

        patients_attrs_desc["summary"] = summaries

    return patients_attrs_desc


def main():
    """Run the script."""
    from argparse import ArgumentParser
    from pathlib import Path

    from tqdm.auto import tqdm

    parser = ArgumentParser()
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--attributes",
        type=ClinicalAttribute,
        nargs="+",
        choices=list(ClinicalAttribute),
        help="Attributes to describe",
    )
    parser.add_argument(
        "--subsets",
        type=Path,
        nargs="*",
        help="Plain text files listing patients belonging to each subset to describe independently",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("describe_patients"),
        help="Root directory under which to save the CSV files in which to save the description of each subsets",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    clinical_attrs, subset_files, output_dir = kwargs.pop("attributes"), kwargs.pop("subsets"), kwargs.pop("output_dir")

    if subset_files:
        subsets = {subset_file.stem: subset_file.read_text().splitlines() for subset_file in subset_files}
    else:
        subsets = {"all": None}

    # Describe the numerical/categorical attributes by subset
    include_patients = kwargs.pop("include_patients")
    patients_attrs_desc_by_subset = {}
    pbar = tqdm(subsets.items(), unit="subset")
    for subset, subset_patients in pbar:
        pbar.set_description(f"Computing statistics over subset of patients '{subset}'")

        # Select the patients for the subset as the intersection of the patients to include overall (if specified) and
        # the patients in the subset (if specified)
        if include_patients is not None:
            subset_patients = [
                patient_id
                for patient_id in include_patients
                if subset_patients is None or patient_id in subset_patients
            ]

        patients_attrs_desc_by_subset[subset] = describe_patients(
            Patients(**kwargs, include_patients=subset_patients), clinical_attrs=clinical_attrs, format_summary=True
        )

    # Save the description for each subset
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset, patients_attrs_desc in patients_attrs_desc_by_subset.items():
        patients_attrs_desc.to_csv(output_dir / f"{subset}.csv")


if __name__ == "__main__":
    main()
