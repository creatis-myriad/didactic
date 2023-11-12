import logging

import numpy as np
from matplotlib import pyplot as plt
from vital.data.cardinal.utils.itertools import Patients
from vital.utils.plot import plot_heatmap

from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask
from didactic.tasks.utils import summarize_patients_attn

logger = logging.getLogger(__name__)


def main():
    """Run the script."""
    from argparse import ArgumentParser
    from pathlib import Path

    import pandas as pd
    from vital.data.cardinal.config import CardinalTag
    from vital.utils.logging import configure_logging
    from vital.utils.saving import load_from_checkpoint

    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "pretrained_encoder",
        type=Path,
        help="Path to a model checkpoint, or name of a model from a Comet model registry, of an encoder",
    )
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--subsets",
        type=Path,
        nargs="+",
        help="Path to plain-text files listing subsets of patients for which to summarize attention independently",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("cardiac_multimodal_encoder_attn"),
        help="Root directory under which to save the attention map(s) summarizing the model's attention",
    )
    parser.add_argument(
        "--rescale_above_n_tokens",
        type=int,
        default=10,
        help="For token sequences longer than this threshold, the size of the heatmap is scaled so that the tick "
        "labels and annotations become visibly smaller, instead of overlapping and becoming unreadable",
    )
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the time-series attributes",
    )
    parser.add_argument(
        "--use_attention_rollout",
        action="store_true",
        help="Whether to use attention rollout to compute the summary of the attention",
    )
    parser.add_argument(
        "--head_reduction",
        type=str,
        choices=["mean", "k_max", "k_min"],
        default="k_min",
        help="When not using attention rollout, method to use to aggregate/reduce across attention heads. Only used "
        "when `use_attention_rollout=False`",
    )
    parser.add_argument(
        "--layer_reduction",
        type=str,
        choices=["mean", "first", "last", "k_max", "k_min"],
        default="mean",
        help="When not using attention rollout, method to use to aggregate/reduce across attention layers. Only used "
        "when `use_attention_rollout=False`",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    encoder_ckpt, subsets, output_dir, rescale_above_n_tokens = (
        kwargs.pop("pretrained_encoder"),
        kwargs.pop("subsets"),
        kwargs.pop("output_dir"),
        kwargs.pop("rescale_above_n_tokens"),
    )
    summarize_patient_attn_kwargs = {
        k: kwargs.pop(k) for k in ["mask_tag", "use_attention_rollout", "head_reduction", "layer_reduction"]
    }

    encoder = load_from_checkpoint(encoder_ckpt, expected_checkpoint_type=CardiacMultimodalRepresentationTask)
    if encoder.hparams.cls_token:
        summarize_patient_attn_kwargs["attention_rollout_kwargs"] = {"includes_cls_token": True}

    # Read the lists of patients in each subset from their respective files
    if subsets:
        subsets = {subset_file.stem: subset_file.read_text().splitlines() for subset_file in subsets}

    attn_summary = summarize_patients_attn(
        encoder, Patients(**kwargs), subsets=subsets, progress_bar=True, **summarize_patient_attn_kwargs
    )

    # Ensure that matplotlib is using 'agg' backend
    # to avoid possible 'Could not connect to any X display' errors
    # when no X server is available, e.g. in remote terminal
    plt.switch_backend("agg")

    def _log_attn_summary(attn_summary: np.ndarray, name: str) -> None:
        if encoder.hparams.cls_token:
            # If we have the attention vector of the CLS token w.r.t. other tokens,
            # reshape it into a matrix to be able to display it as a 2D heatmap
            attn_summary_df = pd.DataFrame(
                attn_summary.reshape((1, -1)), index=encoder.token_tags[-1:], columns=encoder.token_tags[:-1]
            )

        else:
            # If we have the self-attention map, display it directly as a heatmap
            attn_summary_df = pd.DataFrame(attn_summary, index=encoder.token_tags, columns=encoder.token_tags)

        plot_heatmap(attn_summary_df, rescale_above_n_elems=rescale_above_n_tokens)
        plt.savefig(output_dir / f"{name}.png", bbox_inches="tight")
        plt.close()  # Close the figure to avoid contamination between plots

        attn_summary_df.to_csv(output_dir / f"{name}.csv")

    # Save the attention maps summarizing all the data/each subset in 2 formats:
    # i) as heat maps for easy visual inspection
    # ii) as raw CSV data to allow for manual analysis later on
    output_dir.mkdir(parents=True, exist_ok=True)
    if not subsets:
        _log_attn_summary(attn_summary, "all")
    else:
        for subset, subset_attn_summary in attn_summary.items():
            _log_attn_summary(subset_attn_summary, subset)


if __name__ == "__main__":
    main()
