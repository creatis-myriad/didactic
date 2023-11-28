import functools
import itertools
import logging
from typing import Any, Callable, Dict, Sequence

import holoviews as hv
import numpy as np
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
from didactic.tasks.utils import encode_patients, summarize_patient_attn

logger = logging.getLogger(__name__)

hv.extension("bokeh")


def interactive_cardiac_multimodal_representation(
    model: CardiacMultimodalRepresentationTask,
    patients: Patients,
    mask_tag: str = CardinalTag.mask,
    embedding_kwargs: Dict[str, Any] = None,
    categorical_attrs_lists: Dict[str, Dict[str, Sequence[Patient.Id]]] = None,
    summarize_patient_attn_fn: Callable[[CardiacMultimodalRepresentationTask, Patient], np.ndarray] = None,
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
        summarize_patient_attn_fn: Function that measures the attention given by the model w.r.t. each input attribute.
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
    token_tags = list(model.token_tags)
    if model.cls_token:
        token_tags = token_tags[:-1]  # Discard CLS token from token tags if it is there
    label_attrs = list(TabularAttribute)
    label_attrs_by_group = {
        group: [attr for attr in group_attrs if attr in label_attrs]
        for group, group_attrs in TABULAR_ATTR_GROUPS.items()
    }
    if categorical_attrs_lists:
        label_attrs_by_group["custom"] = list(categorical_attrs_lists)
    if summarize_patient_attn_fn:
        label_attrs_by_group["attention"] = [f"{attr}_attn" for attr in token_tags]

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

    # Merge patients' 2D embeddings and attributes data into a single dataframe
    patients_df = patient_embeddings.join(patients_records)

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

    if summarize_patient_attn_fn:
        # Measure the attention given by the model to each input attribute, for each patient
        patients_attrs_attn = np.vstack([summarize_patient_attn_fn(model, patient) for patient in patients.values()])

        # Add the attention w.r.t. each attribute to the data
        patients_attrs_attn_df = pd.DataFrame(
            patients_attrs_attn, index=list(patients), columns=[f"{attr}_attn" for attr in token_tags]
        )
        patients_df = patients_df.join(patients_attrs_attn_df)

        # For each patient, identify the attributes the model attended to the most
        ranked_attn_indices = np.fliplr(patients_attrs_attn.argsort())

        # Create an array of token tags to vectorize identifying the labels of the most attended attributes
        token_tags_by_patient = np.array(token_tags)[None, :].repeat(len(patients), axis=0)

        # Reorder the attention and token tags arrays according to the attention given by the model
        patients_attrs_attn = np.take_along_axis(patients_attrs_attn, ranked_attn_indices, axis=1)
        token_tags_by_patient = np.take_along_axis(token_tags_by_patient, ranked_attn_indices, axis=1)

        # For each rank, add
        # i) a column indicating which attribute holds that rank for the patients, and
        # ii) the exact value of the attention given by the model to the attribute with that rank
        # NOTE: Join a separate dataframe once at the end (instead of columns one at a time) to avoid fragmenting the
        # dataframe, causing poor performance.
        ranked_attrs = {}
        for rank in range(len(token_tags)):
            ranked_attrs[f"ranked_attr_{rank}"] = token_tags_by_patient[:, rank]
            ranked_attrs[f"ranked_attr_{rank}_attn"] = patients_attrs_attn[:, rank]
        ranked_attrs = pd.DataFrame(ranked_attrs, index=list(patients))
        patients_df = patients_df.join(ranked_attrs)

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
        options=[attr for attr in sorted(label_attrs) if attr in TabularAttribute.numerical_attrs()],
    )
    attr_attn = pn.widgets.Select(
        name="Attention w.r.t. attribute (defines colorbar for right figure)",
        options=sorted(patients_attrs_attn_df.columns) if summarize_patient_attn_fn else [],
    )
    attr_attn.disabled = summarize_patient_attn_fn is None  # Disable if there is no func to measure attention

    # Define the widgets to select which attributes are displayed in hover tool
    attrs_groups = {
        group: pn.widgets.CheckBoxGroup(name=group, options=group_attrs)
        for group, group_attrs in label_attrs_by_group.items()
        if group_attrs
    }
    max_attn_rank = len(token_tags) - 1
    attn_rank_slider = pn.widgets.IntSlider(
        name="Display attention up to Nth most important attribute", end=max_attn_rank, value=0
    )
    attn_rank_slider.disabled = summarize_patient_attn_fn is None  # Disable if there is no func to measure attention

    # Define the widget to reinitialize the selection across the checkboxes
    def _reset_attrs_groups(event: param.parameterized.Event):
        for group_attrs in attrs_groups.values():
            group_attrs.value = []

    reset_attrs_groups_button = pn.widgets.Button(name="Clear selected attributes", button_type="danger")
    reset_attrs_groups_button.on_click(_reset_attrs_groups)

    def _update_embedding(
        color_attr: str,
        attn_rank: int,
        **attrs_groups: Sequence[TabularAttribute],
    ) -> hv.Points:
        attention_group = attrs_groups.pop("attention", [])
        attrs = list(itertools.chain.from_iterable(attrs_groups.values()))
        tooltips = [("ID", "@patient_id")] + [(attr, f"@{attr}") for attr in attrs]
        # Handle attention hover tool group apart from the other groups to use custom formatting
        tooltips += [(attr_attn, f"@{{{attr_attn}}}{{.000}}") for attr_attn in attention_group]
        tooltips += [
            (f"ranked_attr_{rank}", f"@ranked_attr_{rank}: @ranked_attr_{rank}_attn{{.000}}")
            for rank in range(attn_rank)
        ]
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
    cat_attrs_fig = hv.DynamicMap(
        pn.depends(color_attr=cat_attr, attn_rank=attn_rank_slider, **attrs_groups, watch=True)(_update_embedding)
    ).opts(framewise=True)
    num_attrs_fig = hv.DynamicMap(
        pn.depends(color_attr=num_attr, attn_rank=attn_rank_slider, **attrs_groups, watch=True)(_update_embedding)
    ).opts(framewise=True)
    attn_fig = hv.DynamicMap(
        pn.depends(color_attr=attr_attn, attn_rank=attn_rank_slider, **attrs_groups, watch=True)(_update_embedding)
    )

    # Configure the widgets in the layout
    hovertool_tooltip_widget = pn.Accordion(*list(attrs_groups.items()))
    widgets = pn.Column(
        cat_attr, num_attr, attr_attn, attn_rank_slider, reset_attrs_groups_button, hovertool_tooltip_widget
    )

    # Configure the dynamic figure titles
    def _update_fig_title(attr: str) -> pn.pane.Markdown:
        return pn.pane.Markdown(
            f"**2D {embedding.__class__.__name__} embedding of the model's {model.hparams.embed_dim}D latent space "
            f"w.r.t. {attr}**"
        )

    cat_attrs_fig_title = pn.depends(attr=cat_attr)(_update_fig_title)
    num_attrs_fig_title = pn.depends(attr=num_attr)(_update_fig_title)
    attn_fig_title = pn.depends(attr=attr_attn)(_update_fig_title)

    fig_opts = {"width": 750, "height": 750}
    return pn.Row(
        widgets,
        pn.Column(cat_attrs_fig_title, cat_attrs_fig.opts(**fig_opts)),
        pn.Column(num_attrs_fig_title, num_attrs_fig.opts(**fig_opts)),
        pn.Column(attn_fig_title, attn_fig.opts(**fig_opts)),
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
        "--disable_attention",
        dest="enable_attention",
        action="store_false",
        help="Whether to deactivate including information about attention w.r.t. input attributes in the figures",
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

    encoder_ckpt, mask_tag, embedding_kwargs, plot_categorical_attrs_dirs, enable_attention, points_opts, port = (
        kwargs.pop("pretrained_encoder"),
        kwargs.pop("mask_tag"),
        kwargs.pop("embedding_kwargs"),
        kwargs.pop("plot_categorical_attrs_dirs"),
        kwargs.pop("enable_attention"),
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

    summarize_patient_attn_fn = None
    if enable_attention:
        # Configure function to measure attention
        summarize_patient_attn_fn = functools.partial(
            summarize_patient_attn,
            mask_tag=mask_tag,
            use_attention_rollout=True,
            attention_rollout_kwargs={"includes_cls_token": encoder.hparams.cls_token},
        )

    # Organize layout
    panel = interactive_cardiac_multimodal_representation(
        encoder,
        Patients(**kwargs),
        mask_tag,
        embedding_kwargs=embedding_kwargs,
        categorical_attrs_lists=categorical_attrs_lists,
        summarize_patient_attn_fn=summarize_patient_attn_fn,
        points_opts=points_opts,
    )

    # Launch server for the interactive app
    pn.serve(panel, title="Cardiac Multimodal Representation Interactive Distribution", port=port)


if __name__ == "__main__":
    main()
