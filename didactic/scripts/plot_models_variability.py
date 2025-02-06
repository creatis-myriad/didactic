from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import JointGrid
from vital.data.cardinal.config import TabularAttribute
from vital.data.cardinal.utils.attributes import TABULAR_ATTR_TITLES, TABULAR_CAT_ATTR_LABELS
from vital.data.config import Subset
from vital.utils.parsing import yaml_flow_collection


def plot_embeddings_variability(
    embeddings: Dict[str, np.ndarray],
    ref_embedding: str = None,
    hue_name: str = "hue",
    size_name: str = "size",
    style_name: str = "style",
    **plot_kwargs,
) -> JointGrid:
    """Generates a scatter plot of the variability w.r.t. the position on the continuum, for each item in the embedding.

    Notes:
        - The embedding, i.e. position on the continuum, must be a scalar.

    Args:
        embeddings: Mapping between the name of each embedding and the corresponding (N, [1]) embedding vector.
        ref_embedding: Key to a "reference" embedding (in `embeddings`) to use as the reference position on the
            continuum. If not provided, the mean of the embeddings will be used.
        hue_name: Name of the attribute to use for coloring the points in the plot.
        size_name: Name of the attribute to use for sizing the points in the plot.
        style_name: Name of the attribute to use for styling the points in the plot.
        **plot_kwargs: Additional keyword arguments to forward to the call to `seaborn.scatterplot`.

    Returns:
        Scatter plot of the variability w.r.t. the position on the continuum, for each item in the embedding.
    """
    # Stack the embeddings into a single array,
    # making sure to squeeze the singleton embedding dimension if it exists
    embeddings_arr = np.stack([emb.squeeze() for emb in embeddings.values()], axis=1)  # (N, M)

    # Compute the statistics over each set of embedding vectors
    if ref_embedding:
        val = embeddings[ref_embedding].squeeze()  # (N)
    else:
        val = np.mean(embeddings_arr, axis=1)  # (N)
    std = np.std(embeddings_arr, axis=1)  # (N)

    # Replace the group variables in plot kwargs with their names
    # and store the original values in a separate dict
    group_vars = {}
    for attr, group_var in [(hue_name, "hue"), (size_name, "size"), (style_name, "style")]:
        if group_var in plot_kwargs:
            group_vars[attr] = plot_kwargs.pop(group_var)
            plot_kwargs[group_var] = attr

    # For categorical attributes, count the number of patients assigned to each label
    cat_attrs_labels_counts = {
        attr: {
            str(attr_label): sum(val == attr_label for val in vals)
            for attr_label in TABULAR_CAT_ATTR_LABELS.get(attr) or sorted(set(vals))
        }
        for attr, vals in group_vars.items()
        if attr not in TabularAttribute.numerical_attrs()
    }

    # Create a DataFrame with the variability statistics and the group variables
    data = {"val": val, "std": std}
    data.update(group_vars)
    data = pd.DataFrame(data=data)

    # Scatter plot of the position along the continuum (i.e. mean) w.r.t. variability (i.e. std) and user-defined groups
    with sns.axes_style("darkgrid"):
        # Hack to include the cardinalities of each group in the legend
        scatter = sns.scatterplot(data=data, x="val", y="std", **plot_kwargs)

        # Customize the legend
        with sns.axes_style("darkgrid"):
            for legend_entry in scatter.legend().texts:
                entry_label = legend_entry.get_text()

                # Rename attributes used in the paper's figures to their full names
                if entry_label in TABULAR_ATTR_TITLES:
                    legend_entry.set_text(TABULAR_ATTR_TITLES[entry_label])

                # If we have reached a new attribute in the legend
                if entry_label in group_vars:
                    legend_group_attr = entry_label  # Update the current attribute
                # else, if the current attribute is categorical, add the count of patients assigned to the label
                elif legend_group_attr in cat_attrs_labels_counts:
                    legend_entry.set_text(
                        f"{entry_label} (n={cat_attrs_labels_counts[legend_group_attr][entry_label]})"
                    )

        scatter.set(
            xlim=(-0.025, 1.025),
            ylim=(0, 0.16),
            xlabel="Stratification predicted by representative model",
            ylabel="Stratification's SD across models",
        )

    return scatter


def main():
    """Run the script."""

    from argparse import ArgumentParser
    from pathlib import Path

    from matplotlib import pyplot as plt
    from tqdm.auto import tqdm
    from vital.data.cardinal.config import CardinalTag
    from vital.data.cardinal.utils.attributes import TABULAR_CAT_ATTR_LABELS
    from vital.data.cardinal.utils.itertools import Patients
    from vital.utils.saving import load_from_checkpoint

    from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask
    from didactic.tasks.utils import encode_patients

    parser = ArgumentParser(
        "Script to plot the variability of the predictions of an ensemble of similar models w.r.t. the predictions' "
        "continuum"
    )
    parser.add_argument(
        "pretrained_encoder",
        nargs="+",
        type=Path,
        help="Path to model checkpoint, or name of a model from a Comet model registry, of an encoder",
    )
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the time-series attributes",
    )
    parser.add_argument(
        "--encoding_task",
        type=str,
        default="continuum_param",
        choices=["continuum_param"],
        help="Encoding task used to generate the embeddings for the computation of the variability",
    )
    parser.add_argument(
        "--plot_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.scatterplot`",
    )
    parser.add_argument(
        "--plot_categorical_attrs_dirs",
        type=Path,
        nargs="+",
        help="Directory (one for each additional categorical attribute to add to the plot) containing '.txt' files "
        "listing patients belonging to each of the attribute's labels",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the root directory where to save the variability plots",
    )
    args = parser.parse_args()

    if len(args.pretrained_encoder) < 2:
        raise ValueError("At least 2 models are required to compute a meaningful variability")

    kwargs = vars(args)
    pretrained_encoders, mask_tag, encoding_task, plot_kwargs, plot_categorical_attrs_dirs, output_dir = list(
        map(
            kwargs.pop,
            [
                "pretrained_encoder",
                "mask_tag",
                "encoding_task",
                "plot_kwargs",
                "plot_categorical_attrs_dirs",
                "output_dir",
            ],
        )
    )
    # Convert the list of directories into a mapping between the attribute name and the directory
    plot_categorical_attrs_dirs = {dirpath.name: dirpath for dirpath in plot_categorical_attrs_dirs}

    # Compute the embeddings of the patients for each model
    # Load the models directly inside the dict comprehension so that (hopefully) only one model gets loaded into GPU
    # memory at a time, to avoid having all models loaded at any given time
    patients = Patients(**kwargs)
    embeddings = {
        # Take the first target as the prediction (indexing of the dict of predictions returned by `encode_patients`)
        ckpt.stem: encode_patients(
            encoder := load_from_checkpoint(ckpt, expected_checkpoint_type=CardiacMultimodalRepresentationTask),
            patients.values(),
            mask_tag=mask_tag,
            progress_bar=True,
            task=encoding_task,
        )[encoder.hparams.target_tabular_attrs[0]]
        for ckpt in tqdm(pretrained_encoders, desc="Computing embeddings for the patients", unit="model")
    }

    # If requested, add data to use as grouping variable for the plot
    for group_var in "hue", "size", "style":
        if attr := plot_kwargs.get(group_var):
            if attr in list(TabularAttribute):
                # If a built-in attribute is requested, extract it from the patients' attributes
                plot_kwargs.update(
                    {
                        f"{group_var}_name": attr,
                        group_var: [patient.attrs[attr] for patient in patients.values()],
                        f"{group_var}_order": TABULAR_CAT_ATTR_LABELS.get(attr),
                    }
                )
            elif attr:
                # else, the requested data should be provided as additional categorical attributes
                labels_files = {
                    # HACK: Use YAML parser to recover dtypes labels (e.g. bool, int, etc.) from file stems
                    yaml_flow_collection(label_file.stem): label_file
                    for label_file in sorted(plot_categorical_attrs_dirs[attr].glob("*.txt"))
                }
                patients_additional_labels = {
                    patient_id: label
                    for label, label_file in labels_files.items()
                    for patient_id in label_file.read_text().splitlines()
                }
                plot_kwargs.update(
                    {
                        f"{group_var}_name": attr,
                        group_var: [patients_additional_labels[patient_id] for patient_id in patients],
                    }
                )
                # Sort subsets by their enum's order
                if attr.lower() == "subset":
                    plot_kwargs[f"{group_var}_order"] = sorted(labels_files.keys(), key=list(Subset).index)

    # Ensure that matplotlib is using 'agg' backend
    # to avoid possible 'Could not connect to any X display' errors
    # when no X server is available, e.g. in remote terminal
    plt.switch_backend("agg")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Generate a plot of the variability w.r.t. the mean embedding as well as each encoder's embedding
    for ref_embedding in tqdm(
        [None, *list(embeddings.keys())],
        desc=f"Generating variability plots for {encoding_task} embeddings",
        unit="embedding",
    ):
        # Generate the plot of the variability
        plot_embeddings_variability(embeddings, ref_embedding=ref_embedding, **plot_kwargs)

        filename = ref_embedding if ref_embedding else "mean"
        plt.savefig(output_dir / f"{filename}.svg", bbox_inches="tight")
        plt.close()  # Close the figure to avoid contamination between plots


if __name__ == "__main__":
    main()
