from typing import Dict, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import JointGrid


def plot_embeddings_variability(
    embeddings: Dict[str, np.ndarray], hue: Sequence = None, hue_name: str = None, hue_order: Sequence[str] = None
) -> JointGrid:
    """Generates a scatter plot of the variability w.r.t. the position on the continuum, for each item in the embedding.

    Notes:
        - The embedding, i.e. position on the continuum, must be a scalar.

    Args:
        embeddings: Mapping between the name of each embedding and the corresponding (N, [1]) embedding vector.
        hue: (N), Optional sequence of values to use as the hue for the plot.
        hue_name: Name to use for the hue in the plot's legend.
        hue_order: Sequence of hue values to use for ordering the hue values in the plot's legend.

    Returns:
        Scatter plot of the variability w.r.t. the position on the continuum, for each item in the embedding.
    """
    hue_name = hue_name if hue_name else "hue"

    # Stack the embeddings into a single array,
    # making sure to squeeze the singleton embedding dimension if it exists
    embeddings_arr = np.stack([emb.squeeze() for emb in embeddings.values()], axis=1)  # (N, M)

    # Compute the statistics over each set of embedding vectors
    mean = np.mean(embeddings_arr, axis=1)  # (N)
    std = np.std(embeddings_arr, axis=1)  # (N)

    # Scatter plot of the position along the continuum (i.e. mean) w.r.t. variability (i.e. std) and user-defined hue
    data = {"mean": mean, "std": std}
    grid_kwargs = {}
    if hue is not None:
        data[hue_name] = hue
        grid_kwargs.update(hue=hue_name, hue_order=hue_order)
    data = pd.DataFrame(data=data)

    with sns.axes_style("darkgrid"):
        grid = sns.JointGrid(data=data, x="mean", y="std", **grid_kwargs)
        grid.plot_joint(sns.scatterplot, size=std)
        grid.plot_marginals(sns.histplot)

    # Remove the marginal for the y-axis
    grid.ax_marg_y.remove()

    return grid


def main():
    """Run the script."""

    from argparse import ArgumentParser
    from pathlib import Path

    from matplotlib import pyplot as plt
    from tqdm.auto import tqdm
    from vital.data.cardinal.config import CardinalTag, TabularAttribute
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
        default="unimodal_param",
        choices=["unimodal_param"],
        help="Encoding task used to generate the embeddings for the computation of the variability",
    )
    parser.add_argument(
        "--hue_attr",
        type=TabularAttribute,
        default=TabularAttribute.ht_severity,
        help="Name of the tabular attribute to use as the hue for the plot",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Path to the image file where to save the variability plot",
    )
    args = parser.parse_args()

    if len(args.pretrained_encoder) < 2:
        raise ValueError("At least 2 models are required to compute a meaningful variability")

    kwargs = vars(args)
    pretrained_encoders, mask_tag, encoding_task, hue_attr, output_file = list(
        map(kwargs.pop, ["pretrained_encoder", "mask_tag", "encoding_task", "hue_attr", "output_file"])
    )

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

    # If requested, extract an additional attribute to use as the hue for the plot
    plot_kwargs = {}
    if hue_attr:
        plot_kwargs.update(
            hue=[patient.attrs[hue_attr] for patient in patients.values()],
            hue_name=hue_attr,
            hue_order=TABULAR_CAT_ATTR_LABELS[hue_attr],
        )

    # Generate the plot of the variability
    plot = plot_embeddings_variability(embeddings, **plot_kwargs)
    plot.set_axis_labels(f"{encoding_task} mean", f"{encoding_task} std")
    # plot.figure.subplots_adjust(top=0.9)  # Adjust the existing figure to leave place on top for the title
    plot.figure.suptitle(f"{encoding_task} std w.r.t. mean and {hue_attr}", x=0.4, y=1.02)

    # Save the variability plot to disk
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()  # Close the figure to avoid contamination between plots


if __name__ == "__main__":
    main()
