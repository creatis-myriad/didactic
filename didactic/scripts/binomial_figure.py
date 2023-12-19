def main():
    """Script that takes a number of trials and a probability of success and plots the binomial distribution.

    Note: The plot of the binomial distribution is produced using ``seaborn.objects.Bars`` which is more appropriate
    for continuous histograms.
    """
    import argparse
    from pathlib import Path

    import numpy as np
    import scipy.stats as stats
    import seaborn.objects as so
    from scipy.special import softmax

    parser = argparse.ArgumentParser(description="Plot a binomial distribution.")
    parser.add_argument("--n", type=int, default=6, help="Number of trials.")
    parser.add_argument("--x_title", type=str, default="k", help="Title of the x-axis.")
    parser.add_argument("--x_labels", type=str, nargs="+", help="Tick labels of the x-axis.")
    parser.add_argument("--y_title", type=str, default="B(k,p)", help="Title of the y-axis.")
    parser.add_argument("--p", type=float, default=0.4, help="Probability of success.")
    parser.add_argument("--tau", type=float, default=1, help="Temperature parameter for the softmax function.")
    parser.add_argument("--output_name", type=Path, help="Output file name.")
    args = parser.parse_args()

    if len(args.x_labels) != args.n:
        raise ValueError(f"Number of x labels ({len(args.x_labels)}) must match the number of trials ({args.n}).")

    # Compute the binomial distribution
    x = np.arange(args.n)
    y = stats.binom.pmf(x, args.n, args.p)
    y = softmax(y / args.tau)

    # Plot the binomial distribution
    if categorical_x := args.x_labels is not None:
        x = args.x_labels
    plot = so.Plot(x=x, y=y).add(so.Bars() if not categorical_x else so.Bar()).label(x=args.x_title, y=args.y_title)

    if args.output_name:
        args.output_name.parent.mkdir(parents=True, exist_ok=True)
        plot.save(args.output_name, bbox_inches="tight")
    else:
        plot.show()


if __name__ == "__main__":
    main()
