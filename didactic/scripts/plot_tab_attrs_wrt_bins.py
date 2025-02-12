_TAB_ATTRS_WRT_BINS = {
    "mean_e_prime": [13.3, 13.1, 10.4, 10.8, 8.2, 7.9],
    "lateral_e_prime": [15.1, 15.1, 11.8, 12.4, 9.3, 8.9],
    "septal_e_prime": [11.7, 12.1, 8.9, 9.1, 7.1, 6.9],
    "la_volume": [21.1, 22.5, 27.6, 27.9, 33.4, 35.3],
    "lvm_ind": [69.3, 72.3, 84.6, 83.9, 107.7, 110.3],
    "e_e_prime_ratio": [5.8, 6.0, 7.5, 7.5, 10.4, 9.7],
}
_TAB_ATTRS_CMAP = {
    "mean_e_prime": "flare_r",
    "lateral_e_prime": "flare_r",
    "septal_e_prime": "flare_r",
    "la_volume": "flare",
    "lvm_ind": "flare",
    "e_e_prime_ratio": "flare",
}


def main():
    """Run the script."""
    from argparse import ArgumentParser
    from pathlib import Path

    import pandas as pd
    import seaborn as sns
    from describe_patients import _NUM_ATTR_DECIMALS
    from matplotlib import pyplot as plt
    from vital.utils.parsing import yaml_flow_collection

    parser = ArgumentParser()
    parser.add_argument(
        "--tabular_attrs",
        nargs="+",
        choices=list(_TAB_ATTRS_WRT_BINS.keys()),
        default=list(_TAB_ATTRS_WRT_BINS.keys()),
        help="Tabular attributes to plot",
    )
    parser.add_argument(
        "--full_precision",
        action="store_true",
        help="Whether to use all precision of the aggregated tabular attribute values, or round to the precision "
        "used in clinic for each attribute",
    )
    parser.add_argument(
        "--attr_label",
        nargs="?",
        choices=["xlabel", "ylabel"],
        help="Whether to label the attribute on the x-axis or y-axis",
    )
    parser.add_argument(
        "--group_label",
        nargs="?",
        choices=["xlabel", "ylabel"],
        help="Whether to label the group on the x-axis or y-axis",
    )
    parser.add_argument(
        "--annot_kws",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Keyword arguments to pass as 'annot_kws' to `seaborn.heatmap`",
    )
    parser.add_argument("--output_dir", type=Path, help="Path to the directory where to save the plots")
    args = parser.parse_args()

    if (args.attr_label == args.group_label) and (args.attr_label is not None):
        raise ValueError(
            f"The attribute and group labels must be used to label different axes, but both are set to "
            f"'{args.attr_label}'."
        )

    # Extract the mean of the tabular attributes w.r.t. the bins
    tab_attrs_df = pd.DataFrame.from_dict(_TAB_ATTRS_WRT_BINS, orient="index").loc[args.tabular_attrs]

    # Ensure that matplotlib is using 'agg' backend
    # to avoid possible leak of file handles if matplotlib defaults to another backend
    plt.switch_backend("agg")

    # Create the output directory if it does not exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # For each tabular attribute, the heatmap of the mean values w.r.t. the bins, with the range of the colorbar
    # set by the min/max values of the tabular attributes on the whole dataset
    for tab_attr in args.tabular_attrs:
        tab_attr_values = tab_attrs_df.loc[tab_attr].to_numpy().reshape(1, -1)

        # Plot the heatmap
        tab_attr_decimals = _NUM_ATTR_DECIMALS.get(tab_attr, 0)
        plot = sns.heatmap(
            tab_attr_values.round(tab_attr_decimals) if not args.full_precision else tab_attr_values,
            cmap=_TAB_ATTRS_CMAP[tab_attr],
            square=True,
            annot=True,
            annot_kws=args.annot_kws,
            fmt=f".{tab_attr_decimals}f",
            yticklabels=False,
            xticklabels=False,
            cbar=False,
        )
        plot.tick_params(bottom=False)

        label_kws = {}
        if args.attr_label:
            label_kws[args.attr_label] = tab_attr
        if args.group_label:
            label_kws[args.group_label] = "Predicted stratification bin"

        plot.set(title=None, **label_kws)

        # Save the plots locally
        plt.savefig(args.output_dir / f"{tab_attr}.svg", bbox_inches="tight")
        plt.close()  # Close the figure to avoid contamination between plots


if __name__ == "__main__":
    main()
