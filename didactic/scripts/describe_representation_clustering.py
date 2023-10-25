import itertools
import logging
from pathlib import Path
from typing import Iterator, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib.axes import Axes
from vital.data.cardinal.config import CardinalTag, ClinicalAttribute, ImageAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.attributes import CLINICAL_ATTR_UNITS, CLINICAL_CAT_ATTR_LABELS, IMAGE_ATTR_LABELS
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients

from didactic.data.cardinal.utils import build_clusterings_dataframe, build_img_attr_by_patient_group_dataframe

logger = logging.getLogger(__name__)


def plot_clinical_attrs_variability_figures(
    patients: Patients,
    clusterings: Mapping[str, Mapping[Patient.Id, str]],
    clinical_attrs: Sequence[ClinicalAttribute] = None,
    num_clustering_agg: str = None,
    cat_plot_kwargs: dict = None,
    num_plot_kwargs: dict = None,
) -> Iterator[Tuple[str, Axes]]:
    """Plots the variability of cluster-aggregated clinical attrs across multiple clusterings w.r.t. clusters.

    Args:
        patients: Collection of patients data from which to extract the attributes.
        clusterings: Instances of clustering of the patients population, representation as mappings between patient IDs
            and cluster labels.
        clinical_attrs: Subset of clinical attributes on which to compile the results. If not provided, will default to
            all available attributes.
        num_clustering_agg: Aggregation function to use to aggregate the numerical attributes by clusters, before the
            aggregation across clusterings. If not provided, the attributes are aggregated by clusters and across
            clusterings at the same time, leading to a higher reported variability.
        cat_plot_kwargs: Parameters to forward to the call to `seaborn.heatmap` for categorical attributes.
        num_plot_kwargs: Parameters to forward to the call to `seaborn.boxplot` for numerical attributes.

    Returns:
        Iterator over figures (and their corresponding titles) plotting the variability of cluster-aggregated clinical
        attrs across multiple clusterings w.r.t. clusters.
    """
    if cat_plot_kwargs is None:
        cat_plot_kwargs = {}
    if num_plot_kwargs is None:
        num_plot_kwargs = {}

    # Gather the data of the patients in each cluster for each clustering
    clusterings_data = build_clusterings_dataframe(patients, clusterings)
    if clinical_attrs is not None:
        clusterings_data = clusterings_data[clinical_attrs]

    # Ignore `matplotlib.category` logger 'INFO' level logs to avoid repeated logs about categorical units parsable
    # as floats
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

    # For each attribute, plot the variability of the attribute w.r.t. clusters
    for attr in clusterings_data.columns:
        title = f"{attr}_wrt_clusters"
        attr_data = clusterings_data[attr]

        # Based on whether the attribute is categorical or numerical, define different types of plots
        if attr in ClinicalAttribute.categorical_attrs():
            # Compute the occurrence of each category for each cluster (including NA), across all clusterings
            attr_stats = attr_data.groupby(["model", "cluster"]).value_counts(normalize=True, dropna=False) * 100
            # After the NA values have been taken into account for the count, drop them
            attr_stats = attr_stats.dropna()

            # For unknown reasons, this plot is unable to pickup variables in the multi-index. As a workaround, we
            # reset the index and to make the index levels into columns available to the plot
            attr_stats = attr_stats.reset_index()

            # For boolean attributes, convert the values to string so that seaborn can properly pick up label names
            # Avoids the following error: 'bool' object has no attribute 'startswith'
            # At the same time, assign relevant labels/hues/etc. for either boolean or categorical attributes
            if attr in ClinicalAttribute.boolean_attrs():
                attr_stats = attr_stats.astype({attr: str})
                ylabel = "(% true)"
                hue_order = [str(val) for val in CLINICAL_CAT_ATTR_LABELS[attr]]
            else:
                ylabel = "(% by label)"
                hue_order = CLINICAL_CAT_ATTR_LABELS[attr]

            # Use dodged barplots for categorical attributes
            with sns.axes_style("darkgrid"):
                plot = sns.barplot(
                    data=attr_stats,
                    x="cluster",
                    y="proportion",
                    hue=attr,
                    hue_order=hue_order,
                    estimator="median",
                    errorbar=lambda data: (np.quantile(data, 0.25), np.quantile(data, 0.75)),
                    **cat_plot_kwargs,
                )

            plot.set(title=title, ylabel=ylabel)

        else:  # attr in ClinicalAttribute.numerical_attrs()
            if num_clustering_agg is not None:
                # Aggregate the numerical attributes by clusters, before the aggregation across clusterings
                attr_data = attr_data.groupby(["model", "cluster"]).agg(num_clustering_agg)

            # Use boxplots for numerical attributes
            with sns.axes_style("darkgrid"):
                # Reset index on the data to make the index levels available as values to plot
                plot = sns.boxplot(data=attr_data.reset_index(), x="cluster", y=attr, **num_plot_kwargs)

            plot.set(title=title, ylabel=CLINICAL_ATTR_UNITS[attr][0])

        yield title, plot


def plot_img_attrs_variability_figures(
    patients: Patients,
    clusterings: Mapping[str, Mapping[Patient.Id, str]],
    image_attrs: Sequence[Tuple[ViewEnum, ImageAttribute]],
    mask_tag: str = CardinalTag.mask,
) -> Iterator[Tuple[str, Axes]]:
    """Plots the variability of cluster-aggregated image attrs across multiple clusterings w.r.t. clusters.

    Args:
        patients: Collection of patients data from which to extract the attributes.
        clusterings: Instances of clustering of the patients population, representation as mappings between patient IDs
            and cluster labels.
        image_attrs: Subset of image-based attributes derived from segmentations (identified by view/attribute pairs)
            for which to plot the variability between bins of the reference clinical attribute.
        mask_tag: Tag of the segmentation mask for which to extract the image attributes.

    Returns:
        Iterator over figures (and their corresponding titles) plotting the variability of cluster-aggregated image
        attrs across multiple clusterings w.r.t. clusters.
    """
    # Convert clusterings from mapping between item IDs and cluster IDs to lists of patient IDs by cluster
    clusterings = {
        clustering_id: {
            cluster_label: sorted(
                patient_id for patient_id, patient_cluster in clusters.items() if patient_cluster == cluster_label
            )
            for cluster_label in sorted(set(clusters.values()))
        }
        for clustering_id, clusters in clusterings.items()
    }

    # Merge the lists of patients in each cluster for each clustering, to obtain a single list of patients per cluster
    # (corresponding to the union of the patients in a specific cluster each clustering)
    # At the same time, while we were up until now just working with patient IDs, we now fetch the patient data
    any_clustering_label = list(clusterings.keys())[0]
    cluster_labels = sorted(set(clusterings[any_clustering_label].keys()))
    patients_by_cluster = {
        cluster_label: list(
            map(
                patients.get,
                itertools.chain.from_iterable(clusters[cluster_label] for clusters in clusterings.values()),
            )
        )
        for cluster_label in cluster_labels
    }

    # For each image attribute, build the dataframe of the mean curve for each bin and plot the curves for each bin
    for img_attr in image_attrs:
        neigh_agg_img_attrs_data = build_img_attr_by_patient_group_dataframe(
            patients_by_cluster, img_attr, group_desc="cluster", mask_tag=mask_tag
        )

        with sns.axes_style("darkgrid"):
            plot = sns.lineplot(
                data=neigh_agg_img_attrs_data, x="time", y="val", hue="cluster", hue_order=sorted(cluster_labels)
            )
        title = f"{'/'.join(img_attr)}_wrt_clusters"
        plot.set(title=title, ylabel=IMAGE_ATTR_LABELS[img_attr[1]])

        yield title, plot


def main():
    """Run the script."""
    from argparse import ArgumentParser

    from matplotlib import pyplot as plt
    from tqdm import tqdm
    from vital.utils.logging import configure_logging
    from vital.utils.parsing import yaml_flow_collection

    # Ensure that matplotlib is using 'agg' backend in non-interactive case
    plt.switch_backend("agg")

    configure_logging(log_to_console=True, console_level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument(
        "clusterings",
        nargs="+",
        type=Path,
        help="Path to files/folders describing the different clusterings of the patients for which to describe the "
        "distribution of attributes",
    )
    parser.add_argument(
        "--clusterings_format",
        type=str,
        choices=["csv", "txt"],
        default="csv",
        help="Format in which the clusterings to be loaded are stored. `csv` is a single CSV file mapping a `patient` "
        "column to a `cluster` column. `txt` is multiple text files inside the folder listing the IDs of the patient "
        "in each cluster.",
    )
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--clinical_attrs",
        type=ClinicalAttribute,
        nargs="*",
        choices=list(ClinicalAttribute),
        help="Subset of clinical attributes on which to compile the results. If not provided, will default to all "
        "available attributes",
    )
    parser.add_argument(
        "--image_attrs",
        type=ImageAttribute,
        choices=list(ImageAttribute),
        nargs="*",
        default=list(ImageAttribute),
        help="Subset of image-based attributes derived from segmentations for which to plot the intra/inter-cluster "
        "variability",
    )
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the image attributes",
    )
    parser.add_argument(
        "--num_clustering_agg",
        type=str,
        help="Aggregation function to use to aggregate the numerical attributes by clusters, before the aggregation "
        "across clusterings. If not provided, the attributes are aggregated by clusters and across clusterings at the "
        "same time, leading to a higher reported variability.",
    )
    parser.add_argument(
        "--clinical_cat_plot_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.heatmap` for categorical clinical attributes figures",
    )
    parser.add_argument(
        "--clinical_num_plot_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.boxplot` for categorical clinical attributes figures",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("compiled_representation_results"),
        help="Root directory under which to save the compiled results for all of the methods",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    (
        clustering_paths,
        clusterings_fmt,
        clinical_attrs,
        image_attrs,
        mask_tag,
        num_clustering_agg,
        cat_plot_kwargs,
        num_plot_kwargs,
        output_dir,
    ) = list(
        map(
            kwargs.pop,
            [
                "clusterings",
                "clusterings_format",
                "clinical_attrs",
                "image_attrs",
                "mask_tag",
                "num_clustering_agg",
                "clinical_cat_plot_kwargs",
                "clinical_num_plot_kwargs",
                "output_dir",
            ],
        )
    )
    image_attrs_keys = [(view, image_attr) for view, image_attr in itertools.product(args.views, image_attrs)]

    # Load the dataset
    patients = Patients(**kwargs)

    # Load and interpret the clustering instances
    match clusterings_fmt:
        case "csv":
            clusterings = {
                str(idx): pd.read_csv(clustering_file, index_col=0, dtype={"patient": str, "cluster": str})[
                    "cluster"
                ].to_dict()
                for idx, clustering_file in enumerate(clustering_paths)
            }
        case "txt":
            clusterings = {
                clustering_dir.stem: {
                    patient_id: cluster_file.stem
                    for cluster_file in clustering_dir.glob("*.txt")
                    for patient_id in cluster_file.read_text().split()
                }
                for clustering_dir in clustering_paths
            }
        case _:
            raise ValueError(f"Unknown `clusterings_format`: {clusterings_fmt}")

    clinical_attrs_plots = plot_clinical_attrs_variability_figures(
        patients,
        clusterings,
        clinical_attrs=clinical_attrs,
        num_clustering_agg=num_clustering_agg,
        cat_plot_kwargs=cat_plot_kwargs,
        num_plot_kwargs=num_plot_kwargs,
    )
    image_attrs_plots = plot_img_attrs_variability_figures(patients, clusterings, image_attrs_keys, mask_tag=mask_tag)

    # Plot the variability of the clinical and image attributes
    output_dir.mkdir(parents=True, exist_ok=True)  # Prepare the output folder for the method
    n_plots = (len(clinical_attrs) if clinical_attrs else len(ClinicalAttribute)) + len(image_attrs_keys)
    for title, plot in tqdm(
        itertools.chain(clinical_attrs_plots, image_attrs_plots),
        desc="Plotting the variability of the attributes w.r.t. clusters",
        unit="attr",
        total=n_plots,
    ):
        title_pathified = title.lower().replace("/", "_").replace(" ", "_")
        filepath = output_dir / f"{title_pathified}.svg"

        if isinstance(plot, so.Plot):
            plot.save(filepath, bbox_inches="tight")
        elif isinstance(plot, Axes):
            plt.savefig(filepath)
            plt.close()  # Close the figure to avoid contamination between plots
        else:
            raise ValueError(f"Unable to save the figure for plot type: {type(plot)}.")


if __name__ == "__main__":
    main()
