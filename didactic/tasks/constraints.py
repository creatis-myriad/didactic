from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

import pytorch_lightning as pl
from torch import Tensor, nn


class Constraint(nn.Module, ABC):
    """Base class defining differentiable constraints on a model's output features space."""

    def setup(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        """Prepares the computation of the constraint at beginning of each stage."""

    @abstractmethod
    def forward(self, item_ids: Sequence[str], out_features: Tensor) -> Tensor:
        """Computes the differentiable constraint.

        Args:
            item_ids: (N,), IDs uniquely identifying each item in the batch.
            out_features: (N, E), Features extracted by the model, for each item the batch.

        Returns:
            (1,), Computed differentiable constraint.
        """


class DistanceBasedConstraint(Constraint, ABC):
    """Abstract class for constraints based on the distance between points in the feature space."""

    def __init__(self, cdist: Callable[[Tensor, Tensor], Tensor], **kwargs):
        """Initializes class instance.

        Args:
            cdist: Function that computes the pairwise distance matrix (P,R) between collections of vectors (P,M)/(R,M),
                using the same shapes as `scipy.cdist`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self.cdist = cdist
