from collections import OrderedDict
from typing import Dict, Sequence

import holoviews as hv
import numpy as np
import panel as pn
import param
import torch
from panel.layout import Panel
from vital.data.cardinal.config import CardinalTag, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.attributes import TIME_SERIES_ATTR_LABELS
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients
from vital.data.transforms import Interp1d

from didactic.tasks.cardiac_sequence_attrs_ae import CardiacSequenceAttributesAutoencoder
from didactic.tasks.cardiac_sequence_attrs_pca import CardiacSequenceAttributesPCA

hv.extension("bokeh")


def interactive_cardiac_sequence_attrs_interpolation(
    models: Dict[str, CardiacSequenceAttributesAutoencoder | CardiacSequenceAttributesPCA],
    patients: Patients,
    mask_tag: str = CardinalTag.mask,
    steps: int = 100,
) -> Panel:
    """Organizes an interactive layout of widgets to interpolate between cardiac sequences time-series attributes.

    Args:
        models: Models that can encode and reconstruct time-series attributes curves.
        patients: Patients whose time-series attributes to encode as samples between which to interpolate.
        mask_tag: Tag of the segmentation mask to get the time-series attributes for.
        steps: Size of the interpolation slider, i.e. finite number of interpolation coefficients.

    Returns:
        Interactive layout of widgets to interpolate between cardiac sequences time-series attributes.
    """
    # Make sure the autoencoder models are in 'eval' mode
    for model in models.values():
        if isinstance(model, CardiacSequenceAttributesAutoencoder):
            model.eval()

    # Gather hyperparameters from the different autoencoder models, and prompt user for which parameters to use to
    # standardize plots if some hyperparameters differ between models
    in_lengths = {model_tag: model.in_shape[1] for model_tag, model in models.items()}

    if len(in_lengths_vals := set(in_lengths.values())) > 1:
        in_length = int(
            input(
                f"Multiple input lengths across the provided models: {in_lengths}. Please indicate the length at which "
                f"to resample the attributes for the plot: "
            )
        )
    else:
        in_length = in_lengths_vals.pop()  # Get the unique `in_length` value

    # Prepare transform to standardize attributes' length, based on the length used by the models
    normalize_length = Interp1d(in_length)

    # Define the widgets for selecting the starting point of the interpolation
    patient_ids = list(patients)
    src_patient_select = pn.widgets.Select(name="Source patient", value=patient_ids[0], options=patient_ids)
    dest_patient_select = pn.widgets.Select(name="Target patient", value=patient_ids[1], options=patient_ids)
    shared_views = list(
        patients[src_patient_select.value].views.keys() & patients[dest_patient_select.value].views.keys()
    )
    view_select = pn.widgets.Select(name="View", value=shared_views[0], options=shared_views)
    attr_select = pn.widgets.Select(name="Attribute", value=TimeSeriesAttribute.gls, options=list(TimeSeriesAttribute))

    # Define the callbacks for updating widgets whose choices must be dynamically updated
    @pn.depends(src_patient=src_patient_select, dest_patient=dest_patient_select, watch=True)
    def _update_view(src_patient: Patient.Id, dest_patient: Patient.Id) -> None:
        shared_views = list(patients[src_patient].views.keys() & patients[dest_patient].views.keys())
        view_select.options = shared_views
        view_select.value = shared_views[0]

    # Define the widget to play with interpolation coefficient
    interpolation_slider = pn.widgets.FloatSlider(name="alpha", value=0.5, start=0, end=1, step=1 / steps)

    # Define the widget to select which methods to display in the plot
    models_checkbox = pn.widgets.CheckBoxGroup(name="Models", value=list(models), options=list(models), inline=True)

    class _AttributeData(param.Parameterized):
        """Parameterized class to handle attribute data streams."""

        data = param.Array()
        z = param.Dict(default={})

        # Set `on_init=True` so that the encodings are computed when the data is set for the first time upon creating
        # the instance (otherwise the encodings won't be computed on instantiation)
        @param.depends("data", watch=True, on_init=True)
        def update_encodings(self, model_tags: Sequence[str] = models_checkbox.value) -> None:
            """Encodes the attribute data using the models, on any update of the attribute data.

            Args:
                model_tags: Models for which to compute the encodings of the data.
            """
            attr_key = (view_select.value, attr_select.value)

            # Preprocess the data before feeding it to dimensionality reduction models
            data_samples = self.data[None, :]
            with torch.inference_mode():
                for model_tag in model_tags:
                    self.z[model_tag] = (
                        models[model_tag](torch.tensor(data_samples, dtype=torch.float), task="encode", attr=attr_key)
                        .squeeze(dim=0)
                        .cpu()
                        .numpy()
                    )

            # Notify of update to parameter `z` as it does not automatically trigger events, being a mutable container
            self.param.trigger("z")

    # Create instances of attribute data streams for both attributes
    src = _AttributeData(
        data=normalize_length(
            patients[src_patient_select.value].views[view_select.value].attrs[mask_tag][attr_select.value]
        )
    )
    dest = _AttributeData(
        data=normalize_length(
            patients[dest_patient_select.value].views[view_select.value].attrs[mask_tag][attr_select.value]
        )
    )

    # Define the callbacks for updating attribute data streams after widget updates
    @pn.depends(patient=src_patient_select, view=view_select, attr=attr_select, watch=True)
    def _update_src(patient: Patient.Id, view: ViewEnum, attr: TimeSeriesAttribute) -> None:
        src.data = normalize_length(patients[patient].views[view].attrs[mask_tag][attr])

    @pn.depends(patient=dest_patient_select, view=view_select, attr=attr_select, watch=True)
    def _update_dest(patient: Patient.Id, view: ViewEnum, attr: TimeSeriesAttribute) -> None:
        dest.data = normalize_length(patients[patient].views[view].attrs[mask_tag][attr])

    @pn.depends(model_tags=models_checkbox, watch=True)
    def _update_models(model_tags: Sequence[str]) -> None:
        src.update_encodings(model_tags=model_tags)
        dest.update_encodings(model_tags=model_tags)

    # Define function called to interpolate attributes' encodings and plot the interpolation and input data
    @pn.depends(
        src_z=src.param.z,
        dest_z=dest.param.z,
        interpolation=interpolation_slider,
        model_tags=models_checkbox,
        watch=True,
    )
    def _update_lineplot(
        src_z: Dict[str, np.ndarray], dest_z: Dict[str, np.ndarray], interpolation: float, model_tags: Sequence[str]
    ) -> hv.NdOverlay:
        attr_key = (view_select.value, attr_select.value)

        # Add data from the source and destination attributes
        curves_data = {"src": src.data, "dest": dest.data}

        # Add data from the autoencoders' interpolation of the source and destination attributes
        with torch.inference_mode():
            for model_tag in model_tags:
                interp_z = src_z[model_tag] * (1 - interpolation) + interpolation * dest_z[model_tag]
                curves_data[model_tag] = (
                    models[model_tag](
                        torch.tensor(interp_z[None, :], dtype=torch.float),
                        task="decode",
                        attr=attr_key,
                    )
                    .squeeze(dim=0)
                    .cpu()
                    .numpy()
                )

        # Use OrderedDict to guarantee an ordering of the data
        # Use `framewise=True` to dynamically adapt plot limits
        overlay_data = OrderedDict({tag: hv.Curve(data).opts(framewise=True) for tag, data in curves_data.items()})

        # Use `sort=False` in the `hv.NdOverlay` to respect the order of the data in the `OrderedDict`
        return hv.NdOverlay(overlay_data, kdims="data", sort=False).opts(
            xlabel="time", ylabel=TIME_SERIES_ATTR_LABELS[attr_select.value], legend_position="bottom_right"
        )

    # Configure the interactive data structure
    interpolation_lineplot = hv.DynamicMap(_update_lineplot)

    # Configure the widgets in the layout
    patients_widgets = pn.Row(src_patient_select, dest_patient_select)
    widgets_layout = pn.Column(patients_widgets, view_select, attr_select, interpolation_slider, models_checkbox)

    # Organize the overall layout and feed starting values
    latent_dims_str = ", ".join(f"{model_tag}={model.latent_dim}" for model_tag, model in models.items())
    return pn.Row(
        widgets_layout,
        interpolation_lineplot.opts(width=800, height=800, title=f"Latent dimensionality by model: {latent_dims_str}"),
    )


def main():
    """Run the interactive app."""
    import argparse
    import logging
    import pickle
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
        "models",
        type=yaml_flow_collection,
        metavar="{MODEL1:CKPT1,MODEL2:CKPT2,...}",
        help="Mapping between model IDs and path to their checkpoint or name in a Comet model registry",
    )
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the time-series attributes",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Size of the interpolation slider, i.e. finite number of interpolation coefficients",
    )
    parser.add_argument("--port", type=int, default=5100, help="Port on which to launch the renderer server")
    args = parser.parse_args()
    kwargs = vars(args)

    models, mask_tag, steps, port = (
        kwargs.pop("models"),
        kwargs.pop("mask_tag"),
        kwargs.pop("steps"),
        kwargs.pop("port"),
    )

    # Load system
    for model_tag, ckpt in models.items():
        ckpt = Path(ckpt)
        if ckpt.suffix == ".pickle":
            with open(ckpt, mode="rb") as ckpt_file:
                model = pickle.load(ckpt_file)
        else:
            model = load_from_checkpoint(ckpt, expected_checkpoint_type=CardiacSequenceAttributesAutoencoder)
        models[model_tag] = model

    # Organize layout
    panel = interactive_cardiac_sequence_attrs_interpolation(models, Patients(**kwargs), mask_tag=mask_tag, steps=steps)

    # Launch server for the interactive app
    pn.serve(panel, title="Cardiac Sequence Time-series Attributes Interpolation", port=port)


if __name__ == "__main__":
    main()
