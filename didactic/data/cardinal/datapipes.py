from typing import Any, Dict, Iterator

import pandas as pd
import torch
from torch.distributions import Normal, Uniform
from torch.utils.data import IterDataPipe
from vital.data.cardinal.config import ClinicalAttribute
from vital.data.cardinal.datapipes import PatientData
from vital.data.cardinal.utils.itertools import Patients


def build_clinical_attrs_gen_datapipes(clinical_attrs_gen_kwargs: Dict[str, Any] = None) -> IterDataPipe[PatientData]:
    """Builds a pipeline of datapipes for generating synthetic clinical attributes.

    Args:
        clinical_attrs_gen_kwargs: Parameters to forward to the generator of synthetic clinical attributes.

    Returns:
        Pipeline of datapipes for the Cardinal dataset.
    """
    if clinical_attrs_gen_kwargs is None:
        clinical_attrs_gen_kwargs = {}
    datapipe = ClinicalAttributesGeneratorIterDataPipe(**clinical_attrs_gen_kwargs)
    return datapipe


class ClinicalAttributesGeneratorIterDataPipe(IterDataPipe):
    """Iterable datapipe that generates an infinite stream of synthetic EDV/ESV/EF tuples."""

    def __init__(self, patients: Patients = None):
        """Initializes class instance.

        Args:
            patients: Collection of patients from which to estimate a distribution of clinical attributes to simulate.
                If not provided, the synthetic clinical attributes will be sampled from prior uniform distributions.
        """
        if patients:
            # Collect the data of the attributes to plot from the patients
            patients_clinical_attrs = {
                patient_id: {
                    attr: patient.attrs[attr]
                    for attr in [ClinicalAttribute.ef, ClinicalAttribute.edv, ClinicalAttribute.esv]
                }
                for patient_id, patient in patients.items()
            }

            # Format the patients' data as a dataframe
            patients_clinical_attrs = pd.DataFrame.from_dict(patients_clinical_attrs, orient="index")
            patients_clinical_attrs_stats = patients_clinical_attrs.agg(["mean", "std"])

            # Use torch's random utilities because they are automatically seeded differently for each worker
            # Otherwise we'd have to implement a `worker_init_fn` to make sure to seed other libraries (e.g. numpy)
            # independently on each worker
            ef_dist = Normal(*patients_clinical_attrs_stats[ClinicalAttribute.ef])
            edv_dist = Normal(*patients_clinical_attrs_stats[ClinicalAttribute.edv])
            self.ef_gen = lambda: torch.clamp(ef_dist.sample(), min=0, max=100)
            self.edv_gen = lambda: torch.clamp(edv_dist.sample(), min=1)

        else:
            # Use torch's random utilities because they are automatically seeded differently for each worker
            # Otherwise we'd have to implement a `worker_init_fn` to make sure to seed other libraries (e.g. numpy)
            # independently on each worker
            ef_dist = Uniform(1, 99)
            edv_dist = Uniform(1, 250)
            self.ef_gen = lambda: ef_dist.sample()
            self.edv_gen = lambda: edv_dist.sample()

    def __iter__(self) -> Iterator[PatientData]:  # noqa: D105
        while True:
            # Sample the EF and EDV to follow the distributions from the reference data
            ef, edv = self.ef_gen(), self.edv_gen()
            # Compute the ESV to be consistent with the sampled EF and EDV
            esv = edv - (ef * edv / 100)

            yield {ClinicalAttribute.ef: ef, ClinicalAttribute.edv: edv, ClinicalAttribute.esv: esv}
