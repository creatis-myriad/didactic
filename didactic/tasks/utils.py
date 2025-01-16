import logging
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import default_collate
from tqdm.auto import tqdm
from vital.data.cardinal.config import CardinalTag, TabularAttribute, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.datapipes import process_patient
from vital.data.cardinal.utils.data_struct import Patient
from vital.utils.format.torch import numpy_to_torch, torch_apply, torch_to_numpy

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
