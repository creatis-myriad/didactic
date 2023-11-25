from typing import Dict, Iterator, Sequence, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from vital.data.cardinal.config import CardinalTag, TabularAttribute
from vital.data.cardinal.utils.attributes import TABULAR_CAT_ATTR_LABELS
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients
from vital.utils.plot import embedding_scatterplot

from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask
from didactic.tasks.utils import encode_patients


def plot_patients_embeddings(
    model: CardiacMultimodalRepresentationTask,
    patients: Patients,
    plot_tabular_attrs: Sequence[TabularAttribute] = None,
    categorical_attrs_lists: Dict[str, Dict[str, Sequence[Patient.Id]]] = None,
    mask_tag: str = CardinalTag.mask,
    progress_bar: bool = False,
    cat_plot_kwargs: dict = None,
    num_plot_kwargs: dict = None,
    embedding_kwargs: dict = None,
) -> Iterator[Tuple[TabularAttribute | str, Axes]]:
    """Generates 2D scatter plots of patients' encodings, labeled w.r.t. specific attributes.

    Args:
        model: Transformer encoder model to use for inference.
        patients: (N) Patients to embed.
        mask_tag: Tag of the segmentation mask for which to extract the image attributes.
        plot_tabular_attrs: Patients' tabular attributes w.r.t. which to plot the embedding.
        categorical_attrs_lists: Nested mapping listing, for each additional categorical attribute, the patients
            belonging to each of the attribute's labels.
        progress_bar: If ``True``, enables progress bars detailing the progress of encoding patients.
        cat_plot_kwargs: Parameters to forward to the call to `seaborn.scatterplot` for categorical attributes figures.
        num_plot_kwargs: Parameters to forward to the call to `seaborn.scatterplot` for numerical attributes figures.
        embedding_kwargs: If the data has more than 2 dimensions, PaCMAP is used to reduce the dimensionality of the
            data for plotting purposes. These arguments will be passed along to the PaCMAP embedding's `init`.

    Returns:
        An iterator over the attributes and associated scatter plots.
    """
    if plot_tabular_attrs is None and categorical_attrs_lists is None:
        raise ValueError(
            "You have specified neither built-in attributes (`plot_tabular_attrs` is None) nor custom attributes "
            "(`categorical_attrs_lists`) w.r.t. which to plot the embeddings. Specify at least one attribute of either "
            "type to plot embeddings of the patients."
        )

    if plot_tabular_attrs is None:
        plot_tabular_attrs = []
    if categorical_attrs_lists is None:
        categorical_attrs_lists = {}
    plot_attrs = plot_tabular_attrs + list(categorical_attrs_lists)
    if cat_plot_kwargs is None:
        cat_plot_kwargs = {}
    if num_plot_kwargs is None:
        num_plot_kwargs = {}

    # Encode the data using the model
    patient_encodings = pd.DataFrame(
        encode_patients(model, patients.values(), mask_tag=mask_tag, progress_bar=progress_bar), index=list(patients)
    ).rename_axis("patient")

    # If the model enforces an ordinal constraint, add the predicted unimodal parameters to the encodings dataframe
    if model.hparams.ordinal_mode:
        cols_to_add = {}
        for task in ["unimodal_param", "unimodal_tau"]:
            prediction_by_target = encode_patients(
                model, patients.values(), task=task, mask_tag=mask_tag, progress_bar=progress_bar
            )
            cols_to_add.update({f"{target}_{task}": prediction for target, prediction in prediction_by_target.items()})
        plot_attrs.extend(cols_to_add.keys())
        patient_encodings = patient_encodings.assign(**cols_to_add)

    # Add the built-in attributes to the encodings dataframe
    patient_encodings = patient_encodings.join(
        pd.DataFrame.from_dict(
            {
                patient.id: {attr: patient.attrs.get(attr) for attr in plot_tabular_attrs}
                for patient in patients.values()
            },
            orient="index",
        )
    )

    # Add each custom attribute as a column to the encodings dataframe
    # by processing the lists of patients belonging to each label
    for attr, attr_labels in categorical_attrs_lists.items():
        attr_df = pd.DataFrame.from_dict(
            {
                patient: [attr_label]
                for attr_label, patients_with_label in attr_labels.items()
                for patient in patients_with_label
            },
            orient="index",
            columns=[attr],
        )
        patient_encodings = patient_encodings.join(attr_df)

    # Transfer the attributes data from the columns to the index, as required by the generic embedding function later
    patient_encodings = patient_encodings.set_index(plot_attrs, append=True)

    # Determine from the tabular attributes' predefined order or the natural ordering in the custom attributes the
    # hue order for the plots
    cat_attrs_order = TABULAR_CAT_ATTR_LABELS.copy()
    cat_attrs_order.update({attr: list(attr_lists) for attr, attr_lists in categorical_attrs_lists.items()})

    # Plot data w.r.t. attributes
    for attr, plot in zip(
        plot_attrs,
        embedding_scatterplot(
            patient_encodings,
            [
                {
                    "hue": attr,
                    "hue_order": cat_attrs_order.get(attr),
                    # Add categorical/numerical kwargs depending on the attribute type
                    **(
                        {True: cat_plot_kwargs, False: num_plot_kwargs}[
                            attr in [*TabularAttribute.categorical_attrs(), *list(categorical_attrs_lists)]
                        ]
                    ),
                }
                for attr in plot_attrs
            ],
            data_tag="encoding",
            **embedding_kwargs,
        ),
    ):
        plot.set(title=None, xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])
        yield attr, plot


def main():
    """Run the script."""
    import argparse
    import logging
    from pathlib import Path

    from vital.utils.logging import configure_logging
    from vital.utils.parsing import yaml_flow_collection
    from vital.utils.saving import load_from_checkpoint

    # Configure logging to display logs from `vital` but to ignore most logs displayed by default by bokeh and its deps
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
    parser.add_argument(
        "--plot_tabular_attrs",
        type=TabularAttribute,
        nargs="+",
        choices=list(TabularAttribute),
        default=list(TabularAttribute),
        help="Patients' tabular attributes w.r.t. which to plot the embedding",
    )
    parser.add_argument(
        "--plot_categorical_attrs_dirs",
        type=Path,
        nargs="+",
        help="Directory (one for each additional categorical attribute w.r.t. which to plot the embedding) containing "
        "'.txt' files listing patients belonging to each of the attribute's labels",
    )
    parser.add_argument(
        "--embedding_kwargs",
        type=yaml_flow_collection,
        default={},
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to pass along to the PaCMAP estimator",
    )
    parser.add_argument(
        "--cat_plot_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.scatterplot` for categorical attributes figures",
    )
    parser.add_argument(
        "--num_plot_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.boxplot` for numerical attributes figures",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("cardiac_multimodal_representation_plot"),
        help="Root directory under which to save the scatter plots of the embedding w.r.t. the attributes",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    (
        encoder_ckpt,
        mask_tag,
        plot_tabular_attrs,
        plot_categorical_attrs_dirs,
        cat_plot_kwargs,
        num_plot_kwargs,
        embedding_kwargs,
        output_dir,
    ) = list(
        map(
            kwargs.pop,
            [
                "pretrained_encoder",
                "mask_tag",
                "plot_tabular_attrs",
                "plot_categorical_attrs_dirs",
                "cat_plot_kwargs",
                "num_plot_kwargs",
                "embedding_kwargs",
                "output_dir",
            ],
        )
    )

    # Load the data and model
    patients = Patients(**kwargs)
    encoder = load_from_checkpoint(encoder_ckpt, expected_checkpoint_type=CardiacMultimodalRepresentationTask)

    # Load the attributes w.r.t which to plot the embeddings
    if plot_categorical_attrs_dirs is None:
        plot_categorical_attrs_dirs = []
    categorical_attrs_lists = {
        attr_dir.name: {
            attr_label_file.stem: attr_label_file.read_text().splitlines()
            for attr_label_file in sorted(attr_dir.glob("*.txt"))
        }
        for attr_dir in plot_categorical_attrs_dirs
    }

    # Ensure that matplotlib is using 'agg' backend
    # to avoid possible leak of file handles if matplotlib defaults to another backend
    plt.switch_backend("agg")

    # For the plots w.r.t. each attribute
    output_dir.mkdir(parents=True, exist_ok=True)
    for attr, _ in plot_patients_embeddings(
        encoder,
        patients,
        plot_tabular_attrs=plot_tabular_attrs,
        categorical_attrs_lists=categorical_attrs_lists,
        mask_tag=mask_tag,
        progress_bar=True,
        cat_plot_kwargs=cat_plot_kwargs,
        num_plot_kwargs=num_plot_kwargs,
        embedding_kwargs=embedding_kwargs,
    ):
        # Save the plots locally
        plt.savefig(output_dir / f"{attr}.svg", bbox_inches="tight")
        plt.close()  # Close the figure to avoid contamination between plots


if __name__ == "__main__":
    main()
