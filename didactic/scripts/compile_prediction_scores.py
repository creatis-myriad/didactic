from typing import List, Sequence

import pandas as pd


def compile_prediction_scores(
    prediction_scores: Sequence[pd.DataFrame], scores_to_agg: Sequence[str], agg_functions: List[str]
) -> pd.DataFrame:
    """Aggregates the scores from multiple models into one set of scores.

    Args:
        prediction_scores: Prediction scores for the different models to aggregate.
        scores_to_agg: Names of the rows representing the scores for which to aggregate the results
        agg_functions: Names of the statistics to measure on the aggregate scores.

    Returns:
        Table of the scores aggregated over all the models.
    """
    agg_scores_by_model = [scores.loc[scores_to_agg].dropna(axis="columns") for scores in prediction_scores]
    agg_scores = pd.concat(agg_scores_by_model).astype(float).describe()
    agg_scores = agg_scores.loc[agg_functions]
    return agg_scores


def main():
    """Run the script."""
    import csv
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument(
        "scores_files",
        nargs="+",
        type=Path,
        help="Files containing the results of multiple methods/runs for which to aggregate the scores",
    )
    parser.add_argument(
        "--scores_to_agg",
        nargs="+",
        type=str,
        default=["acc"],
        help="Names of the rows representing the scores for which to aggregate the results",
    )
    parser.add_argument(
        "--agg_funcs",
        nargs="+",
        type=str,
        choices=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        default=["mean", "std", "min", "max", "50%", "25%", "75%"],
        help="Names of the statistics to measure on the aggregate scores",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Path to a CSV file in which to save the compiled results",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    scores_files, scores_to_agg, agg_funcs, output_file = list(
        map(kwargs.pop, ["scores_files", "scores_to_agg", "agg_funcs", "output_file"])
    )

    # Compile the results of the different methods/runs
    agg_predictions = compile_prediction_scores(
        [pd.read_csv(score_file, index_col=0) for score_file in scores_files], scores_to_agg, agg_funcs
    )

    # Determine default name of the method, if it is not explicitly provided
    if output_file is None:
        output_file = Path(f"{scores_files.name}.csv")

    # Prepare the output folder for the method
    output_file.parent.mkdir(parents=True, exist_ok=True)

    agg_predictions.to_csv(output_file, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    main()
