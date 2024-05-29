import itertools
import logging
from typing import Sequence, Tuple

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.base import ClassifierMixin
from vital.data.cardinal.config import TabularAttribute
from vital.data.cardinal.data_module import CardinalDataModule
from vital.data.cardinal.datapipes import MISSING_CAT_ATTR
from vital.data.config import Subset
from vital.utils.config import register_omegaconf_resolvers

logger = logging.getLogger(__name__)


class CardiacRecordsStratificationTask:
    """XGBoost model for stratifying hypertension severity from EHR tabular data."""

    def __init__(
        self, model: ClassifierMixin, tabular_attrs: Sequence[TabularAttribute], target_attr: TabularAttribute
    ):
        """Initializes class instance.

        Args:
            model: Generic classifier model implementing the `sklearn` API.
            tabular_attrs: List of tabular attributes to use as input features for the classifier.
            target_attr: Tabular attribute to use as the target label for the classifier.
        """
        self.model = model

        # Ensure string tags are converted to their appropriate enum types
        self.tabular_attrs = tuple(TabularAttribute[e] for e in tabular_attrs)
        self.target_attr = TabularAttribute[target_attr]

    def _prepare_data_subset(self, data: CardinalDataModule, subset: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract and process from the data module, specifically to handle missing values and categorical attributes.

        Args:
            data: CARDINAL data module.
            subset: Subset of the data to extract (e.g. "train", "test").

        Returns:
            Tuple of data extracted from the subset:
                - DataFrame of tabular data to use as input features, w/ missing values marked as np.nan.
                - Numpy array of target labels.
        """
        # Make sure the subset has been set up before extracting the data
        match subset:
            case Subset.TRAIN | Subset.VAL:
                data.setup(stage=TrainerFn.FITTING)
            case Subset.TEST:
                data.setup(stage=TrainerFn.TESTING)
            case Subset.PREDICT:
                data.setup(stage=TrainerFn.PREDICTING)
            case _:
                raise ValueError(f"Invalid subset: {subset}")

        # Select the appropriate subset of the data
        dataloader = getattr(data, f"{subset}_dataloader")()

        # For each tabular feature, save the vectors of values over each batch
        tab_data = {}
        for batch, attr in itertools.product(dataloader, [*self.tabular_attrs, self.target_attr]):
            attr_batch_data = batch[attr].detach().cpu().numpy()
            tab_data.setdefault(attr, []).append(attr_batch_data)

        # Concatenate the vectors of batches of tabular features into vectors over the entire training set
        tab_data = {tab_attr: np.hstack(batch_vals) for tab_attr, batch_vals in tab_data.items()}

        # Set aside the target labels
        target = tab_data.pop(self.target_attr)

        # Create a dataframe for the training data,
        # and cast categorical attributes to the appropriate data type
        cat_attrs = [attr for attr in self.tabular_attrs if attr in TabularAttribute.categorical_attrs()]
        tab_df = pd.DataFrame(tab_data).astype({attr: "category" for attr in cat_attrs})

        # After casting categorical attributes to the appropriate data type,
        # mark missing values as `np.nan` so that they can be handled properly by the model
        tab_df[cat_attrs] = tab_df[cat_attrs].replace(MISSING_CAT_ATTR, np.nan)

        return tab_df, target

    def fit(self, data: CardinalDataModule) -> "CardiacRecordsStratificationTask":
        """Fit the model to the training set.

        Args:
            data: CARDINAL data module.

        Returns:
            The fitted model.
        """
        X, y = self._prepare_data_subset(data, "train")

        # Fit the model based on sklearn's `BaseEstimator` API
        self.model = self.model.fit(X, y)

        return self

    def score(self, data: CardinalDataModule) -> float:
        """Measure the model's performance on the test set.

        Args:
            data: CARDINAL data module.

        Returns:
            Model's accuracy on the test set.
        """
        # Extract the tabular data (i.e. inputs and target labels) from the test set
        X, y = self._prepare_data_subset(data, "test")

        # Compute the model's performance on the test set
        return self.model.score(X, y)

    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model to.
        """
        self.model.save_model(path)


@hydra.main(version_base=None, config_path="../config", config_name="experiment/cardinal/records-xgb")
def main(cfg: DictConfig):
    """Fit the generic model to the tabular data from the patients."""

    from pathlib import Path

    from hydra.core.hydra_config import HydraConfig
    from vital.utils.logging import configure_logging

    configure_logging(log_to_console=True, console_level=logging.INFO)

    # Set up the task and data components
    task = hydra.utils.instantiate(cfg.task)
    data = hydra.utils.instantiate(cfg.data, _recursive_=False)

    # Fit the model to the tabular data
    task.fit(data)

    # Evaluate the model's performance on the test set
    score = task.score(data)

    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Save the model
    task.save(hydra_output_dir / cfg.model_ckpt)

    # Save the model's performance
    score_df = pd.Series({"acc": score})
    score_df.to_csv(hydra_output_dir / cfg.scores_filename, header=["value"])

    logger.info(f"Logging model and its score ({score:.2%}) to {hydra_output_dir}")


if __name__ == "__main__":
    # Configure environment before calling hydra main function
    # Load environment variables from `.env` file if it exists
    # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
    load_dotenv()
    # Register custom Hydra resolvers
    register_omegaconf_resolvers()

    main()
