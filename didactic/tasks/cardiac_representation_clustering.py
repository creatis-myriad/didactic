import csv
import pickle
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm
from vital.data.cardinal.config import TabularAttribute


class GridSearchEnsembleClustering:
    """Wrapper around sklearn's GridSearchCV to run it multiple times and keep the best model on average."""

    def __init__(self, num_sweeps: int, param_grid: Dict[str, Any], progress_bar: bool = True):
        """Initializes class instance.

        Args:
            num_sweeps: Number of identical grid searches to run, keeping the params with the best average score in the
                end.
            param_grid: Dictionary of parameters defining the grid of hyperparameters to search, to be passed along to
                `GridSearchCV`'s `init`.
            progress_bar: If ``True``, enables progress bars detailing the progress of the grid search sweeps.
        """
        self.num_sweeps = num_sweeps
        self.param_grid = param_grid
        self.progress_bar = progress_bar
        self.gmm = None
        self.sweeps_results = None
        self.clusters_map = None

    def fit(self, X: np.ndarray, order_clusters_by: np.ndarray = None) -> "GridSearchEnsembleClustering":
        """Run multiple grid searches, selecting the params with best average score as the best params.

        Args:
            X: (N, E), Samples of E-dimensional data points.
            order_clusters_by: (N), Numerical feature associated to each data point, by which to order the clusters by
                ascending mean value over the cluster. If not provided, the (random) cluster IDs will be kept.

        Returns:
            Reference to the fitted model.
        """
        if len(X) != len(order_clusters_by):
            raise ValueError(
                f"The number values to order clusters by: '{len(order_clusters_by)}' should be the same as the number "
                f"of samples in X: '{len(X)}'."
            )

        # Define scoring function used to evaluate clustering performance
        def _gmm_bic_score(gmm: GaussianMixture, X: np.ndarray) -> float:
            return -gmm.bic(X)  # Make it negative since GridSearchCV expects a score to maximize

        sweep_it = range(self.num_sweeps)
        if self.progress_bar:
            sweep_it = tqdm(
                sweep_it,
                desc="Performing identical rounds of grid search to keep best params on average",
                unit="GridSearch",
            )

        # Run `num_sweeps` grid searches to account for variability between grid searches
        sweep_results = {}
        for sweep in sweep_it:
            grid_search = GridSearchCV(GaussianMixture(), param_grid=self.param_grid, scoring=_gmm_bic_score)
            grid_search.fit(X)
            sweep_results[sweep] = pd.DataFrame(grid_search.cv_results_)

        self.sweeps_results = pd.concat(sweep_results)

        # Select param combination with the best average score as the best params
        # NOTE: Convert 'params' col from dict to frozenset so that it is hashable (required for `groupby` to work)
        sweeps_results = self.sweeps_results.copy()
        sweeps_results["params"] = self.sweeps_results["params"].apply(lambda x: frozenset(x.items()))
        mean_score_by_params = sweeps_results.groupby("params")["mean_test_score"].mean()
        params_rank = mean_score_by_params.rank(method="max").astype(int)
        best_params = dict(params_rank.index[params_rank.argmax()])

        # Refit model with best params on average across sweeps
        self.gmm = GaussianMixture(**best_params).fit(X)

        # Compute mapping of cluster IDs so that they're sorted w.r.t. the provided values
        if order_clusters_by is not None:
            pred_df = pd.DataFrame.from_dict({"cluster": self.gmm.predict(X), "order_by": order_clusters_by})
            values_by_cluster = pred_df.groupby(["cluster"])["order_by"].mean()
            clusters_map = values_by_cluster.rank(method="max").astype(int) - 1  # Start cluster ID at 0
            self.clusters_map = clusters_map.to_dict()

        return self

    @property
    def means(self) -> np.ndarray:
        """Centroids of the clusters of the underlying GMM."""
        if self.gmm is None:
            raise RuntimeError(
                f"Cannot access `means` property on instance of {self.__class__.__name__} before having called `fit`"
            )

        means = self.gmm.means_

        # Re-order the clusters' params to follow the custom a posteriori order
        if self.clusters_map:
            sorted_cluster_indices = tuple(np.argsort(list(self.clusters_map.values())))
            means = means[sorted_cluster_indices, :]

        return means

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels for samples using the trained model.

        Args:
            X: (N, E), Samples of E-dimensional data points.

        Returns:
            (N), The label for each sample.
        """
        if self.gmm is None:
            raise RuntimeError(
                f"Cannot call `predict` on instance of {self.__class__.__name__} before having called `fit`"
            )

        # Predict the clusters using the refitted GMM model with the best average params
        pred = self.gmm.predict(X)

        # Re-order the clusters' indices to follow the custom a posteriori order
        if self.clusters_map:
            pred = np.vectorize(self.clusters_map.get)(pred)

        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Evaluates the components' density for each sample.

        Args:
            X: (N, E), Samples of E-dimensional data points.

        Returns:
            (N, K), Density of each K Gaussian component for each sample in X.
        """
        if self.gmm is None:
            raise RuntimeError(
                f"Cannot call `predict_proba` on instance of {self.__class__.__name__} before having called `fit`"
            )

        # Predict the clusters using the refitted GMM model with the best average params
        proba = self.gmm.predict_proba(X)

        # Re-order the clusters' probabilities to follow the custom a posteriori order
        if self.clusters_map:
            sorted_cluster_indices = tuple(np.argsort(list(self.clusters_map.values())))
            proba = proba[:, sorted_cluster_indices]

        return proba

    def save(self, save_path: Path) -> None:
        """Saves a pickled version of the clustering model to a file.

        Args:
            save_path: Path where to save a pickled version of the clustering model.
        """
        if self.gmm is None:
            raise RuntimeError(
                f"Cannot call `save` on instance of {self.__class__.__name__} before having called `fit`"
            )

        # Save the complete model
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, mode="wb") as save_file:
            pickle.dump(self, save_file)

        # Save some metadata about the model (e.g. best params, cluster mapping) in human-readable format
        (save_path.parent / "best_params.yaml").write_text(yaml.dump(self.gmm.get_params(), sort_keys=False))
        if self.clusters_map:
            (save_path.parent / "clusters_map.yaml").write_text(yaml.dump(self.clusters_map))


def save_predictions(
    model: GridSearchEnsembleClustering, X: np.ndarray, save_path: Path, sample_ids: Sequence[str] = None
) -> None:
    """Saves samples predictions i) map of cluster by sample, ii) lists of samples by cluster and iii) scores.

    Args:
        model: Trained model.
        X: (N, E), Samples of E-dimensional data points.
        sample_ids: IDs associated to samples. If not provided, will default to the numerical index of the sample.
        save_path: Root directory under which to save the different formats of cluster predictions.
    """
    if sample_ids is None:
        sample_ids = list(range(len(X)))
    else:
        if len(sample_ids) != len(X):
            raise ValueError(
                f"The number of sample IDs: '{len(sample_ids)}' should be the same as the number "
                f"of samples in X: '{len(X)}'."
            )

    pred = model.predict(X)
    pred_df = pd.DataFrame.from_dict({"patient": sample_ids, "cluster": pred}).set_index("patient")

    # Save the predictions by samples
    pred_df.to_csv(save_path / "predictions.csv", quoting=csv.QUOTE_NONNUMERIC)

    # Save the predictions by clusters
    for cluster in pred_df["cluster"].unique():
        label_keys = sorted(pred_df[pred_df["cluster"] == cluster].index)
        (save_path / f"{cluster}.txt").write_text("\n".join(label_keys))

    # Compute and save the scores over the provided samples
    scores = {"silhouette_score": metrics.silhouette_score(X, pred).round(decimals=3).item()}
    (save_path / "best_scores.yaml").write_text(yaml.dump(scores, sort_keys=False))


def plot_grid_searches(model: GridSearchEnsembleClustering, save_path: Path, x_label: str, hue_label: str) -> None:
    """Saves plots test score w.r.t. params for each grid search run when fitting the model.

    Args:
        model: Trained model.
        save_path: Root directory under which to save the plots of each grid search.
        x_label: Name of the parameter to use as the x-axis for the bar plots.
        hue_label: Name of the parameter to use as the color for the bar plots.
    """
    if model.sweeps_results is None:
        raise RuntimeError("Cannot plot the results of the grid searches before having called `fit` on the model")

    plot_data = model.sweeps_results[[f"param_{x_label}", f"param_{hue_label}", "mean_test_score"]].rename(
        columns={f"param_{x_label}": x_label, f"param_{hue_label}": hue_label, "mean_test_score": "BIC"}
    )
    # Since the grid searches sought to maximize the inverse of the BIC, invert the test score to get the true BIC
    plot_data["BIC"] = -plot_data["BIC"]

    for sweep in range(model.num_sweeps):
        sweep_plot_data = plot_data.loc[sweep]

        sns.catplot(data=sweep_plot_data, kind="bar", x=x_label, y="BIC", hue=hue_label)
        plt.savefig(save_path / f"bic_wrt_{x_label}_{hue_label}_{sweep}.png")


def main():
    """Run the script."""
    import re
    from argparse import ArgumentParser
    from pathlib import Path

    from vital.data.cardinal.config import CardinalTag
    from vital.data.cardinal.utils.itertools import Patients
    from vital.utils.saving import load_from_checkpoint

    from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask
    from didactic.tasks.utils import encode_patients

    params = ["n_components", "covariance_type"]

    parser = ArgumentParser()
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
        "--output_dir",
        type=Path,
        default=Path("cardiac_multimodal_representation_clustering"),
        help="Root directory under which to save the plots and models detailing the grid search's results",
    )
    parser.add_argument(
        "--num_sweeps",
        type=int,
        default=20,
        help="Number of identical grid searches to perform and average over to select best model",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        nargs=2,
        default=(2, 11),
        help="Lower/upper bound on the number of components to test in the GMM's grid search",
    )
    parser.add_argument(
        "--covariance_type",
        type=str,
        nargs="*",
        choices=["tied", "diag", "full"],
        default=["diag"],
        help="Types of covariance to test in the GMM's grid search",
    )
    parser.add_argument(
        "--x_label",
        type=str,
        choices=params,
        default=params[0],
        help="Name of the parameter to plot along the x-axis in the categorical plots",
    )
    parser.add_argument(
        "--hue_label",
        type=str,
        choices=params,
        default=params[1],
        help="Name of the parameter to plot as the hue in the categorical plots",
    )
    parser.add_argument(
        "--order_clusters_by",
        type=TabularAttribute,
        choices=list(TabularAttribute),
        default=TabularAttribute.ht_grade,
        help="Attribute used to order the clusters by ascending mean value by cluster",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    if args.x_label == args.hue_label:
        unused_params = [param for param in params if param != args.x_label]
        raise ValueError(
            f"You specified the same hyperparameter to be used for the x-axis ('x_label') and hue ('hue_label') of the "
            f"categorical plots. Change either one of them to one of the following unused parameters: "
            f"{unused_params}."
        )

    encoder_ckpt, mask_tag, output_dir, num_sweeps, x_label, hue_label, order_clusters_by = (
        kwargs.pop("pretrained_encoder"),
        kwargs.pop("mask_tag"),
        kwargs.pop("output_dir"),
        kwargs.pop("num_sweeps"),
        kwargs.pop("x_label"),
        kwargs.pop("hue_label"),
        kwargs.pop("order_clusters_by"),
    )
    param_grid = {k: kwargs.pop(k) for k in params}
    # Convert hyperparameters that have to be interpreted from their CLI args
    param_grid["n_components"] = range(*param_grid["n_components"])

    patients = Patients(**kwargs)
    encoder = load_from_checkpoint(encoder_ckpt, expected_checkpoint_type=CardiacMultimodalRepresentationTask)
    patients_encodings = pd.DataFrame(
        data=encode_patients(encoder, patients.values(), mask_tag=mask_tag), index=list(patients)
    )
    # Extract the attribute to use as value to order the clusters
    values_to_order_clusters = np.array([patient.attrs[order_clusters_by] for patient in patients.values()])

    cluster_model = GridSearchEnsembleClustering(num_sweeps, param_grid)

    # Perform multiple successive grid searches to find the best clustering parameters
    cluster_model.fit(patients_encodings.to_numpy(), order_clusters_by=values_to_order_clusters)

    # Save the clustering model
    def _convert_camel_case_to_snake_case(string: str) -> str:
        return re.sub("(?!^)([A-Z]+)", r"_\1", string).lower()

    cluster_model.save(output_dir / f"{_convert_camel_case_to_snake_case(cluster_model.__class__.__name__)}.pickle")

    # For each grid search, save the plot of the BIC w.r.t. the param_grid
    # Ensure that matplotlib is using 'agg' backend
    # to avoid possible leak of file handles if matplotlib defaults to another backend
    plt.switch_backend("agg")
    plot_grid_searches(cluster_model, output_dir, x_label, hue_label)

    # Save the predictions of the model on the data
    save_predictions(cluster_model, patients_encodings.to_numpy(), output_dir, sample_ids=patients_encodings.index)


if __name__ == "__main__":
    main()
