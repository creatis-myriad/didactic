from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from vital.data.cardinal.config import CardinalTag, ClinicalAttribute, ImageAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.datapipes import process_patient
from vital.data.cardinal.utils.data_struct import Patient
from vital.utils.format.torch import numpy_to_torch

from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask


def encode_patients(
    model: CardiacMultimodalRepresentationTask,
    patients: Iterable[Patient],
    mask_tag: str = CardinalTag.mask,
    progress_bar: bool = False,
) -> np.ndarray:
    """Wrapper around encoder inference to handle boilerplate code (e.g. extracting attributes from patients, etc.).

    Args:
        model: Transformer encoder model to use for inference.
        patients: (N) Patients to encode.
        mask_tag: Tag of the segmentation mask for which to extract the image attributes.
        progress_bar: If ``True``, enables progress bars detailing the progress of the processing and encoding patients
            data.

    Returns:
        (N, E), encodings of the patients.
    """
    clinical_attrs, img_attrs = model.hparams.clinical_attrs, model.hparams.img_attrs

    if progress_bar:
        patients = tqdm(patients, desc="Processing patients' data to prepare it for inference", unit="patient")
    patients_attrs = [
        process_patient(patient, clinical_attributes=clinical_attrs, image_attributes=img_attrs, mask_tag=mask_tag)
        for patient in patients
    ]
    # Run inference on one patient at a time, instead of in batches, to avoid a few possible issues:
    # i) having to resample image attributes to be of the same constant shape to be able to stack them
    # ii) out of memory errors, in case of very large collections of patients
    if progress_bar:
        patients_attrs = tqdm(
            patients_attrs,
            desc=f"Encoding patients to the encoder's {model.hparams.embed_dim}D latent space",
            unit="patients",
        )
    patients_encodings = np.vstack(
        [
            encode_patients_attrs(
                model,
                {attr: patient_attrs[attr] for attr in clinical_attrs},
                {(view, attr): patient_attrs[view][attr] for view in model.hparams.views for attr in img_attrs},
            )
            for patient_attrs in patients_attrs
        ]
    )

    return patients_encodings


def encode_patients_attrs(
    model: CardiacMultimodalRepresentationTask,
    clinical_attrs: Dict[ClinicalAttribute, np.ndarray],
    img_attrs: Dict[Tuple[ViewEnum, ImageAttribute], np.ndarray],
) -> np.ndarray:
    """Wrapper around encoder inference to handle boilerplate code (e.g. numpy to torch, batching/unbatching, etc.).

    Args:
        model: Transformer encoder model to use for inference.
        clinical_attrs: (K: S, V: [N]) Sequence of (batch of) clinical attributes.
        img_attrs: (K: S, V: ([N,] L)), Sequence of (batch of) image attributes, where L is the dimensionality of each
            attribute.

    Returns:
        ([N,], E), encoding(s) of the patient/batch of patients.
    """
    is_batch = list(clinical_attrs.values())[0].ndim == 1

    # If the input isn't a batch of data, add the batch dimension
    clinical_attrs = {k: v if is_batch else v[None, ...] for k, v in clinical_attrs.items()}
    img_attrs = {k: v if is_batch else v[None, ...] for k, v in img_attrs.items()}

    with torch.inference_mode():
        out_features, _ = model(numpy_to_torch(clinical_attrs), numpy_to_torch(img_attrs))

    # Squeeze to remove batch dimension, if it wasn't there in the input
    if not is_batch:
        out_features = out_features.squeeze(dim=0)

    return out_features.cpu().numpy()
