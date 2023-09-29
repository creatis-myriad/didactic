import logging
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics.cluster import adjusted_rand_score

logger = logging.getLogger(__name__)


def compare_clustering(
    ref_clustering: np.ndarray, other_clustering: np.ndarray
) -> Tuple[float, ConfusionMatrixDisplay]:
    """Measures global and pairwise metrics of the similarity between clustering labels.

    Args:
        ref_clustering: Clustering labels to be used as reference.
        other_clustering: Clustering labels to compare to the reference.
    """
    # Compute quantitative and global score comparing the similarity of the clustering
    ari = adjusted_rand_score(ref_clustering, other_clustering)

    # Use a confusion matrix to compare the pairwise similarity between clusters from each prediction
    cm = ConfusionMatrixDisplay.from_predictions(ref_clustering, other_clustering, normalize="true")

    return ari, cm


def main():
    """Run the script."""
    from argparse import ArgumentParser
    from pathlib import Path

    import pandas as pd
    from vital.utils.logging import configure_logging

    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "ref_clustering", type=Path, help="Reference CSV predictions of the `cluster` for each `patient`"
    )
    parser.add_argument(
        "other_clustering",
        type=Path,
        help="CSV predictions of the `cluster` for each `patient` to compare to the reference",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        help="Skip interactively showing figures (e.g. confusion matrices between clusters) and rather save them to "
        "the provided directory",
    )
    parser.add_argument(
        "--naming_level",
        type=int,
        default=1,
        help="Level in the folder hierarchy at which to extract the suffix identifying the clustering version",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="-",
        help="Character separating the rest of the dirname from the suffix of the clustering version",
    )
    args = parser.parse_args()

    if args.save_dir:
        # Ensure that matplotlib is using 'agg' backend in non-interactive case
        plt.switch_backend("agg")

    ref_clustering = pd.read_csv(args.ref_clustering, index_col="patient")["cluster"]
    other_clustering = pd.read_csv(args.other_clustering, index_col="patient")["cluster"]
    other_clustering = other_clustering[ref_clustering.index]  # Make sure the indices match between the series

    logger.info(f"Comparing clusters from '{args.ref_clustering}' and '{args.other_clustering}' ...")
    ari, cm = compare_clustering(ref_clustering.to_numpy(), other_clustering.to_numpy())

    logger.info(f"Adjusted Rand Index (ARI): {ari}")
    ref_name, other_name = (
        args.ref_clustering.parents[args.naming_level].name.split(args.delimiter)[-1],
        args.other_clustering.parents[args.naming_level].name.split(args.delimiter)[-1],
    )
    cm.ax_.set(ylabel=ref_name, xlabel=other_name)
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save_dir / f"cm_{ref_name}_{other_name}.png")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()
