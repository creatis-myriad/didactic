import importlib
from typing import Any, Tuple

import yaml


def import_from_module(dotpath: str) -> Any:
    """Dynamically imports an object from a module based on its "dotpath".

    Args:
        dotpath: "Dotpath" (name that can be looked up via importlib) where the firsts components specify the module to
            look up, and the very last component is the attribute to import from this module.

    Returns:
        Target object.
    """
    module, module_attr = dotpath.rsplit(".", 1)
    return getattr(importlib.import_module(module), module_attr)


def yaml_flow_collection(
    val: str,
    collection_entries_separator: str = ",",
    mapping_separator: str = ":",
    sequence_markers: Tuple[str, str] = ("[", "]"),
    mapping_markers: Tuple[str, str] = ("{", "}"),
) -> Any:
    """Parses a string as a YAML flow collection.

    References:
        - YAML flow collection specification, for more details: https://yaml.org/spec/1.2.2/#74-flow-collection-styles

    Args:
        val: String representation of the flow collection to parse.
        collection_entries_separator: Separator to use to split collection entries.
        mapping_separator: Separator to use to split key-value pairs in flow mappings.
        sequence_markers: Characters denoting the beginning/end of a flow sequence.
        mapping_markers: Characters denoting the beginning/end of a flow mapping.

    Returns:
        Native Python data structure representation of the YAML flow collection parsed from the string.
    """
    yaml_str = (
        val.replace(collection_entries_separator, ",")
        .replace(mapping_separator, ": ")
        .replace(sequence_markers[0], "[")
        .replace(sequence_markers[1], "]")
        .replace(mapping_markers[0], "{")
        .replace(mapping_markers[1], "}")
    )
    return yaml.safe_load(yaml_str)


if __name__ == "__main__":
    from argparse import ArgumentParser

    import numpy as np

    statistical_diff_tests = [
        "scipy.stats.mannwhitneyu",
        "scipy.stats.ttest_rel",
        "scipy.stats.wilcoxon",
    ]
    normality_tests = ["scipy.stats.shapiro", "scipy.stats.normaltest"]

    parser = ArgumentParser("Script to perform statistical tests on results from different models/configurations")
    parser.add_argument(
        "stats_fn",
        type=str,
        help="Dotpath to the statistical function to use",
        choices=[*statistical_diff_tests, *normality_tests],
    )
    parser.add_argument("--scores_set_1", type=float, nargs="+", help="First set of scores to compare")
    parser.add_argument("--scores_set_2", type=float, nargs="+", help="Second set of scores to compare")
    parser.add_argument(
        "--precompute_diff",
        action="store_true",
        help="If set, the script will compute the difference between the two sets of scores before applying the statistical test. Otherwise, it will apply the test directly to the scores.",
    )
    parser.add_argument(
        "--stats_kwargs",
        type=yaml_flow_collection,
        default={},
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to pass along to the statistical function",
    )
    args = parser.parse_args()

    if args.stats_fn in normality_tests and not args.precompute_diff:
        raise ValueError(
            "Normality tests can only be applied to the difference between the two sets of scores. Please set --precompute_diff to True."
        )
    if args.stats_fn in ["scipy.stats.mannwhitneyu", "scipy.stats.ttest_rel"] and args.precompute_diff:
        raise ValueError(
            "The paired t-test can only be applied directly to the two sets of scores, not their differences. Please set remove the --precompute_diff flag."
        )

    data = np.array((args.scores_set_1, args.scores_set_2))
    if args.precompute_diff:
        if len(data[0]) != len(data[1]):
            raise ValueError("The two sets of scores must have the same length to compute the difference.")
        data = (data[0] - data[1],)

    stats_fn = import_from_module(args.stats_fn)
    result = stats_fn(*data, **args.stats_kwargs)
    print(result)
