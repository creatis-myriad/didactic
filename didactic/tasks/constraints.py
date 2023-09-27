import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from vital.metrics.train.metric import MonotonicRegularizationLoss

from didactic.tasks.utils import encode_patients


class ModelUnpickler(pickle.Unpickler):
    """Custom unpickler for models that were pickled in various modules in the project."""

    def find_class(self, module, name):
        """Overrides default mechanism to look up class names for custom classes saved from another module."""
        match name:
            case "GridSearchEnsembleClustering":
                from didactic.tasks.cardiac_representation_clustering import GridSearchEnsembleClustering

                return GridSearchEnsembleClustering

            case _:
                return super().find_class(module, name)


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


class ClustersConsistencyConstraint(DistanceBasedConstraint):
    """Constraint that enforces points to maintain the same relative distance to all cluster centroids.

    The constraint itself is a regularization term that enforces a monotonic relationship between i) the distances
    between features and clusters' centroids at the beginning and ii) these same distances computed dynamically during
    training/finetuning, as the model's feature space evolves.
    """

    def __init__(self, clustering_model: str | Path, delta: float, **kwargs):
        """Initializes class instance.

        Args:
            clustering_model: Path to a serialized clustering model following sklearn's `fit/predict` API.feature spae
            delta: Regularization loss hyperparameter that decides the spread of the posterior distribution.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)

        with open(clustering_model, mode="rb") as clustering_model_file:
            self.clustering_model = ModelUnpickler(clustering_model_file).load()
        cluster_means = torch.from_numpy(self.clustering_model.means)
        # Register as a buffer so that it will automatically be moved to the same device as the runtime tensors
        self.register_buffer("cluster_means", cluster_means)

        self._dist2clusters_reg_loss = MonotonicRegularizationLoss(delta)
        self._dist2clusters = {}

    def setup(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        """For each item in the stage's data, memorize it's initial distance to each cluster's centroid."""
        # For each subset of patients in the data
        datamodule = trainer.datamodule
        for subset_patients in datamodule.subsets_patients.values():
            # Run inference on the patients included in the data to get a "starting" encoding for each patient
            subset_encodings = torch.from_numpy(
                encode_patients(
                    pl_module, subset_patients.values(), mask_tag=datamodule._process_patient_kwargs.get("mask_tag")
                )
            ).to(pl_module.device)

            # For each patient, memorize the distances between its encoding and the clusters' centroids
            self._dist2clusters.update(
                {
                    patient_id: patient_dist2clusters
                    for patient_id, patient_dist2clusters in zip(
                        subset_patients, self.cdist(subset_encodings, self.cluster_means.to(subset_encodings.dtype))
                    )
                }
            )

    def forward(self, item_ids: Sequence[str], out_features: Tensor) -> Tensor:
        """Computes a regularization term between distances computed on `out_features` and those memorized.

        Args:
            item_ids: (N,), IDs uniquely identifying each item in the batch.
            out_features: (N, E), Features extracted by the model, for each item the batch.

        Returns:
            (N,), Regularization term, for each item in the batch.
        """
        batch_dist2clusters = self.cdist(out_features, self.cluster_means.to(out_features.dtype))
        return sum(
            # Move the tensors in `self._dist2clusters` to make sure they're on the runtime device
            self._dist2clusters_reg_loss(patient_dist2clusters, self._dist2clusters[id].to(out_features.device))
            for id, patient_dist2clusters in zip(item_ids, batch_dist2clusters)
        ) / len(item_ids)


class NeighborhoodConsistencyConstraint(DistanceBasedConstraint, ABC):
    """Constraint that enforces the points to maintain the starting neighborhoods, i.e. affinity, to each other.

    The constraint itself is a comparison between a reference similarity matrix, and a moving similarity matrix, updated
    by the features predicted for each new batch.
    i) The pairwise similarity between points is configurable, but a standard choice would be the RBF kernel.
    ii) The loss term for the comparison of similarity matrices is configurable.
    """

    def __init__(
        self, affinity_matrices_loss: Callable[[Tensor, Tensor], Tensor], affinity_eps: float = 1e-3, **kwargs
    ):
        """Initializes class instance.

        Args:
            affinity_matrices_loss: Loss term that compares input and target (D, D) square affinity matrices between
                items' encodings.
            affinity_eps: If not `None`, affinity values under this threshold will be clipped to 0, to reduce the effect
                of distant points and outliers.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self.affinity_matrices_loss = affinity_matrices_loss
        self.affinity_eps = affinity_eps

    def setup(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        """Memorizes the pairwise distances between items in the current stage's data."""
        # For each subset of patients in the data
        datamodule = trainer.datamodule
        patient_ids, patient_encodings = [], []
        for subset_patients in datamodule.subsets_patients.values():
            # Run inference on the patients included in the data to get a "starting" encoding for each patient
            subset_encodings = torch.from_numpy(
                encode_patients(
                    pl_module, subset_patients.values(), mask_tag=datamodule._process_patient_kwargs.get("mask_tag")
                )
            )

            patient_ids.extend(list(subset_patients))
            patient_encodings.append(subset_encodings)

        self.item_ids = patient_ids
        ref_item_encodings = torch.vstack(patient_encodings)
        self.moving_item_encodings = ref_item_encodings.clone()

        self.ref_similarity_matrix = self.cdist(ref_item_encodings, ref_item_encodings)
        if self.affinity_eps:
            # Clip small similarities to 0 to avoid too much noise from distant points and outliers
            self.ref_similarity_matrix = torch.where(
                self.ref_similarity_matrix > self.affinity_eps, self.ref_similarity_matrix, 0
            )

    def forward(self, item_ids: Sequence[str], out_features: Tensor) -> Tensor:
        """Computes a regularization term between distances computed on `out_features` and those memorized.

        Args:
            item_ids: (N,), IDs uniquely identifying each item in the batch.
            out_features: (N, E), Features extracted by the model, for each item the batch.

        Returns:
            (N,), Regularization term, for each item in the batch.
        """
        # Detach moving items' encodings to make sure the graph of previous steps is cleared
        # (so that the backward pass won't try to backprop through previous steps)
        self.moving_item_encodings = self.moving_item_encodings.detach()

        # Update the moving items' encodings
        for item_id, item_features in zip(item_ids, out_features):
            self.moving_item_encodings[self.item_ids.index(item_id)] = item_features

        # Compute the similarity between the updated items' encodings
        similarity_matrix = self.cdist(self.moving_item_encodings, self.moving_item_encodings)
        if self.affinity_eps:
            # Clip small similarities to 0 to avoid too much noise from distant points and outliers
            similarity_matrix = torch.where(similarity_matrix > self.affinity_eps, similarity_matrix, 0)

        # Compute the loss between the moving and reference similarity matrices
        return self.affinity_matrices_loss(similarity_matrix, self.ref_similarity_matrix)
