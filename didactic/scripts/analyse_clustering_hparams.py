import logging

import pandas as pd
from vital.utils.parsing import yaml_flow_collection

logger = logging.getLogger(__name__)


def main():
    """Run the script."""
    from argparse import ArgumentParser
    from pathlib import Path

    import seaborn.objects as so
    from vital.utils.logging import configure_logging

    configure_logging(log_to_console=True, console_level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument(
        "hparams_files",
        nargs="+",
        type=Path,
        help="Paths to YAML config file of the best clustering hyperparameters found for an encoder model",
    )
    parser.add_argument(
        "--hparams",
        nargs="+",
        type=str,
        default=["n_components"],
        help="Hyperparameters for which to analyse the distribution",
    )
    parser.add_argument(
        "--folder_levels",
        type=yaml_flow_collection,
        default={-4: "target", -5: "data", -6: "task"},
        help="Folder levels to use as fields by which to analyse the distribution of clustering hyperparameters",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Root directory under which to save the figures of the distribution of clustering hyperparameters",
    )
    args = parser.parse_args()

    import yaml

    hparams_data = [
        {
            **{level_desc: hparams_file.parts[level] for level, level_desc in args.folder_levels.items()},
            **yaml.safe_load(hparams_file.read_text()),
        }
        for hparams_file in args.hparams_files
    ]
    hparams_data = pd.DataFrame.from_records(hparams_data)

    # Prepare the output folder
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _plot_hist_wrt_hparam(hparam: str, color: str = None) -> None:
        """Plot the histogram of the distribution of the given hyperparameter."""
        title = hparam if color is None else f"{hparam}_wrt_{color}"

        # # Same plot but with the seaborn's functions API
        # from matplotlib import pyplot as plt
        #
        # # Ensure that matplotlib is using 'agg' backend in non-interactive case
        # plt.switch_backend("agg")
        #
        # with sns.axes_style("darkgrid"):
        #     fig = sns.histplot(hparams_data, x=hparam, hue=color, kde=True)
        # fig.set_title(title)
        #
        # plt.savefig(args.output_dir / f"{title}.png")
        # plt.close()  # Close the figure to avoid contamination between plots

        stat = "density"
        plot = (
            so.Plot(hparams_data, x=hparam, color=color)
            .add(so.Bar(), so.Hist(stat, discrete=True), so.Dodge())
            .add(so.Line(), so.KDE())
        )
        plot = plot.label(title=title, y=stat)
        plot.save(args.output_dir / f"{title}.png", bbox_inches="tight")

    # Plot the figures w.r.t. the requested metadata (e.g. hparam, folder level, etc.)
    for hparam in args.hparams:
        _plot_hist_wrt_hparam(hparam)

        for color_var in args.folder_levels.values():
            _plot_hist_wrt_hparam(hparam, color_var)


if __name__ == "__main__":
    main()
