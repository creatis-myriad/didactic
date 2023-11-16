import logging
from statistics import mean
from typing import Any, Dict, Iterable, Literal, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import default_collate
from tqdm.auto import tqdm
from vital.data.cardinal.config import CardinalTag, TabularAttribute, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.datapipes import process_patient
from vital.data.cardinal.utils.data_dis import check_subsets
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients
from vital.utils.format.torch import numpy_to_torch, torch_apply, torch_to_numpy

from didactic.models.explain import attention_rollout, k_number, register_attn_weights_hook
from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask

logger = logging.getLogger(__name__)


def encode_patients(
    model: CardiacMultimodalRepresentationTask,
    patients: Iterable[Patient],
    mask_tag: str = CardinalTag.mask,
    progress_bar: bool = False,
    **forward_kwargs,
) -> np.ndarray | Dict[str, np.ndarray]:
    """Wrapper around encoder inference to handle boilerplate code (e.g. extracting attributes from patients, etc.).

    Args:
        model: Transformer encoder model to use for inference.
        patients: (N) Patients to encode.
        mask_tag: Tag of the segmentation mask for which to extract the time-series attributes.
        progress_bar: If ``True``, enables progress bars detailing the progress of the processing and encoding patients
            data.
        **forward_kwargs: Keyword arguments to pass along to the encoder's inference method.

    Returns:
        (N, E) | K * (N, E), encodings or mapping of predictions by target for the patients.
    """
    tab_attrs, time_series_attrs = model.hparams.tabular_attrs, model.hparams.time_series_attrs

    if progress_bar:
        patients = tqdm(patients, desc="Processing patients' data to prepare it for inference", unit="patient")
    patients_attrs = [
        process_patient(patient, tabular_attrs=tab_attrs, time_series_attrs=time_series_attrs, mask_tag=mask_tag)
        for patient in patients
    ]
    # Run inference on one patient at a time, instead of in batches, to avoid a few possible issues:
    # i) having to resample time-series attributes to be of the same constant shape to be able to stack them
    # ii) out of memory errors, in case of very large collections of patients
    if progress_bar:
        patients_attrs = tqdm(
            patients_attrs,
            desc=f"Encoding patients to the encoder's {model.hparams.embed_dim}D latent space",
            unit="patients",
        )

    patients_encodings = [
        encode_patients_attrs(
            model,
            {attr: patient_attrs[attr] for attr in tab_attrs},
            {(view, attr): patient_attrs[view][attr] for view in model.hparams.views for attr in time_series_attrs},
            **forward_kwargs,
        )
        for patient_attrs in patients_attrs
    ]
    # Use torch's `default_collate` to batch the encodings for each patient together, regardless of whether they are
    # directly a numpy array or a dict of numpy arrays, then convert back to numpy array recursively
    # This is not the most efficient, but it is the simplest implementation to handle both cases
    patients_encodings = torch_to_numpy(default_collate(patients_encodings))

    return patients_encodings


def encode_patients_attrs(
    model: CardiacMultimodalRepresentationTask,
    tabular_attrs: Dict[TabularAttribute, np.ndarray],
    time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], np.ndarray],
    **forward_kwargs,
) -> np.ndarray | Dict[str, np.ndarray]:
    """Wrapper around encoder inference to handle boilerplate code (e.g. numpy to torch, batching/unbatching, etc.).

    Args:
        model: Transformer encoder model to use for inference.
        tabular_attrs: (K: S, V: [N]) Sequence of (batch of) tabular attributes.
        time_series_attrs: (K: S, V: ([N,] L)), Sequence of (batch of) time-series attributes, where L is the
            dimensionality of each attribute.
        **forward_kwargs: Keyword arguments to pass along to the encoder's inference method.

    Returns:
        ([N,], E) | K * ([N,], E), encoding(s) or mapping of prediction(s) by target for the patient/batch of patients.
    """
    is_batch = list(tabular_attrs.values())[0].ndim == 1

    # If the input isn't a batch of data, add the batch dimension
    tabular_attrs = {k: v if is_batch else v[None, ...] for k, v in tabular_attrs.items()}
    time_series_attrs = {k: v if is_batch else v[None, ...] for k, v in time_series_attrs.items()}

    with torch.inference_mode():
        out_features = model(numpy_to_torch(tabular_attrs), numpy_to_torch(time_series_attrs), **forward_kwargs)

    # Squeeze to remove batch dimension, if it wasn't there in the input
    # Use the `apply` function to apply the squeeze recursively in case `out_features` is a dict of tensors
    if not is_batch:
        out_features = torch_apply(out_features, lambda x: x.squeeze(dim=0))

    return torch_to_numpy(out_features)


def summarize_patient_attn(
    model: CardiacMultimodalRepresentationTask,
    patient: Patient,
    mask_tag: str = CardinalTag.mask,
    use_attention_rollout: bool = False,
    attention_rollout_kwargs: Dict[str, Any] = None,
    head_reduction: Literal["mean", "k_max", "k_min"] = "k_min",
    layer_reduction: Literal["mean", "first", "last", "k_max", "k_min"] = "mean",
) -> np.ndarray:
    """Summarizes a model's attention on one patient across multiple layers and heads in a single attention map.

    Args:
        model: Transformer encoder model for which we want to analyze the attention.
        patient: Patient for which to summarize the model's attention.
        mask_tag: Tag of the segmentation mask for which to extract the time-series attributes.
        use_attention_rollout: Whether to use attention rollout to compute the summary of the attention.
        attention_rollout_kwargs: When using attention rollout (`use_attention_rollout` is True), parameters to forward
            to `didactic.models.explain.attention_rollout`.
        head_reduction: When not using attention rollout, method to use to aggregate/reduce across attention heads.
        layer_reduction: When not using attention rollout, method to use to aggregate/reduce across attention layers
            (once attention heads have already been reduced and each layer is summarized by one attention map).

    Returns:
        (S, S), Attention map summarizing the model's attention on one patient across multiple layers and heads.
    """
    if attention_rollout_kwargs is None:
        attention_rollout_kwargs = {}

    # Setup hooks to capture attention maps
    attn_maps = {}
    hook_handles = register_attn_weights_hook(model, attn_maps, reduction="first")

    # Run inference on the patient to produce attention maps
    _ = encode_patients(model, [patient], mask_tag=mask_tag)

    # Teardown the hooks
    for layer_hook in hook_handles.values():
        layer_hook.remove()

    if use_attention_rollout:
        attn = attention_rollout(list(attn_maps.values()), **attention_rollout_kwargs)

    else:
        k_numbers = {
            layer_name: [k_number(head_attn) for head_idx, head_attn in enumerate(layer_attn)]
            for layer_name, layer_attn in attn_maps.items()
        }

        match head_reduction:
            case "mean":
                layers_attn = {layer_name: layer_attn.mean(dim=0) for layer_name, layer_attn in attn_maps.items()}
                k_numbers_reduce_fn = mean
            case "k_max":
                layers_attn = {
                    layer_name: layer_attn[np.argmax(k_numbers[layer_name])]
                    for layer_name, layer_attn in attn_maps.items()
                }
                k_numbers_reduce_fn = max
            case "k_min":
                layers_attn = {
                    layer_name: layer_attn[np.argmin(k_numbers[layer_name])]
                    for layer_name, layer_attn in attn_maps.items()
                }
                k_numbers_reduce_fn = min
            case _:
                raise ValueError(
                    f"Unexpected value for 'k_number_head_reduction': {head_reduction}. "
                    f"Use one of: ['mean', 'max', 'min']."
                )

        k_numbers = {
            layer_name: k_numbers_reduce_fn(k_number_by_head) for layer_name, k_number_by_head in k_numbers.items()
        }
        layers_attn = torch.stack(list(layers_attn.values()))

        match layer_reduction:
            case "mean":
                attn = layers_attn.mean(dim=0)
            case "first":
                attn = layers_attn[0]
            case "last":
                attn = layers_attn[-1]
            case "k_max":
                attn = layers_attn[np.argmax(list(k_numbers.values()))]
            case "k_min":
                attn = layers_attn[np.argmin(list(k_numbers.values()))]
            case _:
                raise ValueError(
                    f"Unexpected value for 'k_number_layer_reduction': {layer_reduction}. "
                    f"Use one of: ['mean', 'first', 'last', 'k_max', 'k_min']."
                )

    return attn.cpu().numpy()


def summarize_patients_attn(
    model: CardiacMultimodalRepresentationTask,
    patients: Patients,
    subsets: Dict[str, Sequence[Patient.Id]] = None,
    progress_bar: bool = False,
    **summarize_patient_attn_kwargs,
) -> np.ndarray | Dict[str, np.ndarray]:
    """Summarizes a model's attention on (subsets of) patients using a single attention map.

    Args:
        model: Transformer encoder model for which we want to analyze the attention.
        patients: Patients for which to analyze the model's attention.
        subsets: Lists of patients making up each subset to summarize independently.
        progress_bar: If ``True``, enables progress bars detailing the progress of the collecting attention maps from
            patients.
        **summarize_patient_attn_kwargs: Parameters to pass along to the `summarize_patient_attn` function.

    Returns:
        Attention map(s) summarizing the attention on all input patients, or on each subset of patients if subsets were
        provided.
    """
    if subsets is not None:
        check_subsets(list(patients), subsets)

    patients = patients.values()
    msg = "Collecting attention maps from patients"
    if progress_bar:
        patients = tqdm(patients, desc=msg, unit="patient")
    else:
        logger.info(msg + "...")

    patients_attn = {
        patient.id: summarize_patient_attn(model, patient, **summarize_patient_attn_kwargs) for patient in patients
    }
    attn_summary = np.stack(list(patients_attn.values())).mean(axis=0)

    if subsets is None:
        return attn_summary

    if subsets:
        attn_summary_by_subset = {"all": attn_summary}

        attn_summary_by_subset.update(
            {
                subset: np.stack([patients_attn[patient_id] for patient_id in subset_patients]).mean(axis=0)
                for subset, subset_patients in subsets.items()
            }
        )
        return attn_summary_by_subset
