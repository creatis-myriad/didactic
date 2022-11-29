from typing import Optional

from pytorch_lightning.trainer.states import TrainerFn
from vital.data.cardinal.data_module import PREDICT_DATALOADERS_SUBSETS, CardinalDataModule
from vital.data.config import Subset

from didactic.data.cardinal.datapipes import build_clinical_attrs_gen_datapipes


class CardinalSyntheticClinicalAttrsDataModule(CardinalDataModule):
    """Override of the Cardinal datamodule to use a generator of synthetic EDV/ESV/EF tuples for training.

    The val/test/predict subsets still come from real Cardinal data, to ease comparisons with models trained on real
    data only. The only thing that changes is that the training subset is substituted for a generator of synthetic
    EDV/ESV/EF tuples, to increase the number of training samples to infinity.
    """

    def __init__(self, simulate_iid: bool = True, **kwargs):
        """Initializes class instance.

        Args:
            simulate_iid: Whether to provide real data samples to the synthetic data generator so that it mimics the
                real distribution of clinical attributes.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        # Determine subset to setup given the stage of training
        subsets_to_setup = []
        match stage:
            case TrainerFn.FITTING:
                clinical_attrs_gen_kwargs = {}
                if self.hparams.simulate_iid:
                    clinical_attrs_gen_kwargs["patients"] = self._partial_patients(
                        include_patients=self._subsets_lists.get(Subset.TRAIN)
                    )
                self.datasets[Subset.TRAIN] = build_clinical_attrs_gen_datapipes(clinical_attrs_gen_kwargs)
                subsets_to_setup.append(Subset.VAL)
            case TrainerFn.VALIDATING:
                subsets_to_setup.append(Subset.VAL)
            case TrainerFn.TESTING:
                subsets_to_setup.append(Subset.TEST)
            case TrainerFn.PREDICTING:
                subsets_to_setup.extend(PREDICT_DATALOADERS_SUBSETS)

        # Update the available collections of patients and datasets
        self.datasets.update({subset: self._build_subset_datapipes(subset) for subset in subsets_to_setup})
