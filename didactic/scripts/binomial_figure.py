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

    parser = argparse.ArgumentParser(description="Plot a binomial distribution.")
    parser.add_argument("--n", type=int, default=6, help="Number of trials.")
    parser.add_argument("--p", type=float, default=0.4, help="Probability of success.")
    parser.add_argument("--output_name", type=Path, default="binomial_distribution.svg", help="Output file name")
    args = parser.parse_args()

    # Compute the binomial distribution
    x = np.arange(args.n)
    y = stats.binom.pmf(x, args.n, args.p)

    # Plot the binomial distribution
    plot = so.Plot(x=x, y=y).add(so.Bars()).label(x="k", y="p(k)")
    plot.save(args.output_name, bbox_inches="tight")


if __name__ == "__main__":
    main()
