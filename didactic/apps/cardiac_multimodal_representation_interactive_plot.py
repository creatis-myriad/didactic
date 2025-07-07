import itertools
import logging
from typing import Any, Dict, Sequence

import holoviews as hv
import pacmap
import pandas as pd
import panel as pn
import param
import seaborn as sns
from bokeh.models import HoverTool
from panel.layout import Panel
from vital.data.cardinal.config import CardinalTag, TabularAttribute
from vital.data.cardinal.utils.attributes import TABULAR_ATTR_GROUPS
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients

from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask
from didactic.tasks.utils import encode_patients

logger = logging.getLogger(__name__)

hv.extension("bokeh")

PREDICTION_TASKS = ("continuum_param", "continuum_tau")


def interactive_cardiac_multimodal_representation(
    model: CardiacMultimodalRepresentationTask,
    patients: Patients,
    mask_tag: str = CardinalTag.mask,
    embedding_kwargs: Dict[str, Any] = None,
    categorical_attrs_lists: Dict[str, Dict[str, Sequence[Patient.Id]]] = None,
    points_opts: Dict[str, Any] = None,
) -> Panel:
    """Organizes an interactive layout of widgets and plots to explore a representation of cardiac patients.

    Args:
        model: Transformer encoder model used to represent the patients.
        patients: Patients to project in the latent space.
        mask_tag: Tag of the segmentation mask for which to extract the time-series attributes.
        embedding_kwargs: Parameters to pass along to the PaCMAP embedding.
        categorical_attrs_lists: Nested mapping listing, for each additional categorical attribute, the patients
            belonging to each of the attribute's labels.
        points_opts: Parameters to pass along to the `hv.Points`' `opts` method.

    Returns:
        Interactive layout of widgets and plots to explore a representation of cardiac patients.
    """
    if embedding_kwargs is None:
        embedding_kwargs = {}
    if categorical_attrs_lists is None:
        categorical_attrs_lists = {}
    if points_opts is None:
        points_opts = {}

    # Make all the attributes in the dataset available for labelling the data
    label_attrs = list(TabularAttribute)
    label_attrs_by_group = {
        group: [attr for attr in group_attrs if attr in label_attrs]
        for group, group_attrs in TABULAR_ATTR_GROUPS.items()
    }
    if categorical_attrs_lists:
        label_attrs_by_group["custom"] = list(categorical_attrs_lists)

    # Compute encoding of each patient in the representation
    patient_encodings = encode_patients(model, patients.values(), mask_tag=mask_tag, progress_bar=True)

    logger.info(f"Computing 2D PaCMAP embedding of the model's {model.hparams.embed_dim}D latent space")
    embedding = pacmap.PaCMAP(**embedding_kwargs)
    patient_embeddings = pd.DataFrame(
        embedding.fit_transform(patient_encodings), index=list(patients), columns=["0", "1"]
    )

    # Isolate tabular attributes data for each patient
    patients_records = patients.to_dataframe(tabular_attrs=label_attrs, cast_to_pandas_dtypes=False)
    # For categorical attributes, convert missing data to a valid string
    # NOTE: When not cast to custom pandas dtype, categorical attributes are of type `object`
    cat_attrs = [
        attr
        for attr in label_attrs
        if attr in TabularAttribute.categorical_attrs() and attr not in TabularAttribute.boolean_attrs()
    ]
    patients_records[cat_attrs] = patients_records[cat_attrs].fillna("n/a")
    patients_df = patient_embeddings.join(patients_records)  # Merge embeddings and attributes data

    # If the model enforces an ordinal constraint, add the predicted continuum parameters to the patients dataframe
    if model.hparams.ordinal_mode:
        cols_to_add = {}
        for task in PREDICTION_TASKS:
            prediction_by_target = encode_patients(
                model, patients.values(), task=task, mask_tag=mask_tag, progress_bar=True
            )
            cols_to_add.update({f"{target}_{task}": prediction for target, prediction in prediction_by_target.items()})
        # Add the predictions to the lists of attributes for plotting
        label_attrs.extend(cols_to_add.keys())  # Selectable attributes to color the scatter plots
        label_attrs_by_group["predictions"] = list(cols_to_add.keys())  # Attributes displayable in hover tool
        patients_df = patients_df.assign(**cols_to_add)

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
        patients_df = patients_df.join(attr_df)

    # Make patient ID available as a column (to more easily display this info in plots)
    patients_df = patients_df.reset_index(names="patient_id")

    # Define the widgets to select which attribute is used to style the points in the scatter plot
    cat_attr = pn.widgets.Select(
        name="Categorical attribute (defines cmap for left figure)",
        options=list(categorical_attrs_lists)
        + [attr for attr in sorted(label_attrs) if attr in TabularAttribute.categorical_attrs()],
    )
    num_attr = pn.widgets.Select(
        name="Numerical attribute (defines colorbar for center figure)",
        options=[
            attr
            for attr in sorted(label_attrs)
            if attr in TabularAttribute.numerical_attrs() or attr.endswith(PREDICTION_TASKS)
        ],
    )

    # Define the widgets to select which attributes are displayed in hover tool
    attrs_groups = {
        group: pn.widgets.CheckBoxGroup(name=group, options=group_attrs)
        for group, group_attrs in label_attrs_by_group.items()
        if group_attrs
    }

    # Define the widget to reinitialize the selection across the checkboxes
    def _reset_attrs_groups(event: param.parameterized.Event):
        for group_attrs in attrs_groups.values():
            group_attrs.value = []

    reset_attrs_groups_button = pn.widgets.Button(name="Clear selected attributes", button_type="danger")
    reset_attrs_groups_button.on_click(_reset_attrs_groups)

    def _update_embedding(
        color_attr: str,
        **attrs_groups: Sequence[TabularAttribute],
    ) -> hv.Points:
        attrs = list(itertools.chain.from_iterable(attrs_groups.values()))
        tooltips = [("ID", "@patient_id")] + [(attr, f"@{attr}") for attr in attrs]
        hover = HoverTool(tooltips=tooltips)
        cmap = (
            sns.color_palette(as_cmap=True) if color_attr in cat_attr.options else sns.cubehelix_palette(as_cmap=True)
        )
        return hv.Points(patients_df, kdims=["0", "1"], vdims=list(patients_df.columns.difference(["0", "1"]))).opts(
            tools=[hover],
            color=color_attr,
            cmap=cmap,
            colorbar=True,
            line_color="white",
            legend_position="bottom",
            xaxis=None,
            yaxis=None,
            **points_opts,
        )

    # Configure the dynamic figures
    cat_attrs_fig = hv.DynamicMap(pn.depends(color_attr=cat_attr, **attrs_groups, watch=True)(_update_embedding)).opts(
        framewise=True
    )
    num_attrs_fig = hv.DynamicMap(pn.depends(color_attr=num_attr, **attrs_groups, watch=True)(_update_embedding)).opts(
        framewise=True
    )

    # Configure the widgets in the layout
    hovertool_tooltip_widget = pn.Accordion(*list(attrs_groups.items()))
    widgets = pn.Column(cat_attr, num_attr, reset_attrs_groups_button, hovertool_tooltip_widget)

    # Configure the dynamic figure titles
    def _update_fig_title(attr: str) -> pn.pane.Markdown:
        return pn.pane.Markdown(
            f"**2D {embedding.__class__.__name__} embedding of the model's {model.hparams.embed_dim}D latent space "
            f"w.r.t. {attr}**"
        )

    cat_attrs_fig_title = pn.depends(attr=cat_attr)(_update_fig_title)
    num_attrs_fig_title = pn.depends(attr=num_attr)(_update_fig_title)

    fig_opts = {"width": 750, "height": 750}
    return pn.Row(
        widgets,
        pn.Column(cat_attrs_fig_title, cat_attrs_fig.opts(**fig_opts)),
        pn.Column(num_attrs_fig_title, num_attrs_fig.opts(**fig_opts)),
    )


def main():
    """Run the interactive app."""
    import argparse
    import logging
    from pathlib import Path

    from vital.utils.logging import configure_logging
    from vital.utils.parsing import yaml_flow_collection
    from vital.utils.saving import load_from_checkpoint

    # Configure logging to display logs from `vital` but to ignore most logs displayed by default by bokeh and its deps
    configure_logging(log_to_console=True, console_level=logging.INFO)
    logging.getLogger("bokeh").setLevel(logging.WARNING)
    logging.getLogger("tornado").setLevel(logging.WARNING)

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
        help="Tag of the segmentation mask for which to extract the time-series attributes",
    )
    parser.add_argument(
        "--embedding_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to pass along to the PaCMAP embedding",
    )
    parser.add_argument(
        "--plot_categorical_attrs_dirs",
        type=Path,
        nargs="+",
        help="Directory (one for each additional categorical attribute w.r.t. which to plot the embedding) containing "
        "'.txt' files listing patients belonging to each of the attribute's labels",
    )
    parser.add_argument(
        "--points_opts",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to pass along to the `hv.Points`' `opts` method",
    )
    parser.add_argument("--port", type=int, default=5100, help="Port on which to launch the renderer server")
    args = parser.parse_args()
    kwargs = vars(args)

    encoder_ckpt, mask_tag, embedding_kwargs, plot_categorical_attrs_dirs, points_opts, port = (
        kwargs.pop("pretrained_encoder"),
        kwargs.pop("mask_tag"),
        kwargs.pop("embedding_kwargs"),
        kwargs.pop("plot_categorical_attrs_dirs"),
        kwargs.pop("points_opts"),
        kwargs.pop("port"),
    )

    encoder = load_from_checkpoint(encoder_ckpt, expected_checkpoint_type=CardiacMultimodalRepresentationTask)

    # Load the additional attributes w.r.t which to plot the embeddings
    if plot_categorical_attrs_dirs is None:
        plot_categorical_attrs_dirs = []
    categorical_attrs_lists = {
        attr_dir.name: {
            attr_label_file.stem: attr_label_file.read_text().splitlines()
            for attr_label_file in sorted(attr_dir.glob("*.txt"))
        }
        for attr_dir in plot_categorical_attrs_dirs
    }

    # Organize layout
    panel = interactive_cardiac_multimodal_representation(
        encoder,
        Patients(**kwargs),
        mask_tag,
        embedding_kwargs=embedding_kwargs,
        categorical_attrs_lists=categorical_attrs_lists,
        points_opts=points_opts,
    )

    # Launch server for the interactive app
    pn.serve(panel, title="Cardiac Multimodal Representation Interactive Distribution", port=port)


if __name__ == "__main__":
    main()
