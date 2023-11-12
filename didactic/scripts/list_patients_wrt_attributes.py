def main():
    """Run the script."""
    from argparse import ArgumentParser
    from pathlib import Path

    import pandas as pd
    from tqdm.auto import tqdm
    from vital.data.cardinal.config import TabularAttribute
    from vital.data.cardinal.utils.itertools import Patients

    parser = ArgumentParser()
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--attributes",
        type=TabularAttribute,
        nargs="+",
        choices=TabularAttribute.categorical_attrs(),
        default=TabularAttribute.categorical_attrs(),
        help="Attributes w.r.t. which to list the patients belonging to each class",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("patients_wrt_attributes"),
        help="Root directory under which to save the folders of lists of patients w.r.t. their attributes",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    attrs_to_list, output_dir = kwargs.pop("attributes"), kwargs.pop("output_dir")

    patients = Patients(**kwargs)
    patients_attrs = {patient.id: [patient.attrs.get(attr) for attr in attrs_to_list] for patient in patients.values()}
    patients_attrs = pd.DataFrame.from_dict(patients_attrs, orient="index", columns=attrs_to_list)

    # For each attribute
    for attr in tqdm(attrs_to_list, desc="Dividing patients based on their categorical attributes values", unit="attr"):
        # Create a designated directory under the output directory
        attr_dir = output_dir / attr
        attr_dir.mkdir(parents=True, exist_ok=True)

        # For each of the attribute's labels (excluding the placeholder label for missing data)
        attr_labels = patients_attrs[attr][~patients_attrs[attr].isna()].unique()
        for attr_label in tqdm(attr_labels, desc=f"Attribute: {attr}", unit="label", leave=False):
            # Find the list of the patients with that label
            patients_with_label = patients_attrs.index[patients_attrs[attr] == attr_label]

            # Save the list
            (attr_dir / f"{attr_label}.txt").write_text("\n".join(patients_with_label))


if __name__ == "__main__":
    main()
