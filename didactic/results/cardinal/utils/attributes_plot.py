from abc import abstractmethod
from argparse import ArgumentParser
from typing import Dict, Iterator, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from vital.data.cardinal.utils.attributes import build_attributes_dataframe
from vital.data.cardinal.utils.data_struct import View
from vital.data.cardinal.utils.itertools import Views
from vital.results.processor import ResultsProcessor


class AttributesPlots(ResultsProcessor):
    """Abstract class that plots attributes w.r.t. time."""

    desc = "attrs_plots"
    ResultsCollection = Views

    def __init__(
        self,
        inputs: Sequence[str],
        normalize_time: bool = True,
        data_name: str = "data",
        attr_name: str = "attr",
        time_name: str = "frame",
        hue: Optional[str] = "attr",
        style: Optional[str] = "data",
        display_title_in_plot: bool = True,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            inputs: Data item for which to analyze attributes' evolution w.r.t. time.
            normalize_time: Whether to normalize the values in the time axis between 0 and 1. By default, these values
                are between 0 and the count of data points in the attributes' data.
            data_name: Name to give in the legend to the variable representing the data the attributes come from.
            attr_name: Name to give in the legend to the variable representing the attributes' names.
            time_name: Name to give in the plot to the time axis.
            hue: Field of the attributes' data to use to assign the curves' hues.
            style: Field of the attributes' data to use to assign the curves' styles.
            display_title_in_plot: Whether to display the title generated for the plot in the plot itself.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(output_name="_".join([*inputs, self.desc]), **kwargs)
        if hue is None and style is None:
            raise ValueError(
                "You must set at least one grouping variable, i.e. `hue` or `style`, with which to group the "
                "time-series attributes' data."
            )
        self.normalize_time = normalize_time
        self.data_name = data_name
        self.attr_name = attr_name
        self.time_name = time_name
        self.hue = hue
        self.style = style
        self.display_title_in_plot = display_title_in_plot
        self.input_tags = inputs

        # Ensure that matplotlib is using 'agg' backend
        # to avoid possible leak of file handles if matplotlib defaults to another backend
        plt.switch_backend("agg")

    def process_result(self, result: View) -> None:
        """Plots attributes w.r.t. time.

        Args:
            result: Data structure holding all the sequence`s data.
        """
        attrs = self._extract_attributes_data(result)
        for title, plot in self._plot_attributes_wrt_time(attrs, "/".join(result.id)):
            plt.savefig(self.output_path / f"{title}.png")
            plt.close()  # Close the figure to avoid contamination between plots

    def _extract_attributes_data(self, result: View) -> pd.DataFrame:
        """Extracts the attributes' values and metadata and structures them for easy downstream lookup/manipulation.

        Args:
            result: Data structure holding all the sequence`s data.

        Returns:
            Dataframe of attributes data in long format.
        """
        attrs_data = {
            input_tag: self._extract_attributes_from_result(result, input_tag) for input_tag in self.input_tags
        }
        return build_attributes_dataframe(
            attrs_data,
            outer_name=self.data_name,
            inner_name=self.attr_name,
            time_name=self.time_name,
            normalize_time=self.normalize_time,
        )

    @abstractmethod
    def _extract_attributes_from_result(self, result: View, item_tag: str) -> Dict[str, np.ndarray]:
        """Extracts the attributes' values over time for one item of the sequence.

        Args:
            result: Data structure holding all the sequence`s data.
            item_tag: Data sequence (e.g. mask, etc.) for which to extract the attributes' data.

        Returns:
            Attributes' values over time for one item of the sequence.
        """

    @abstractmethod
    def _plot_attributes_wrt_time(self, attrs: pd.DataFrame, plot_title_root: str) -> Iterator[Tuple[str, Axes]]:
        """Plot the evolution of the attributes' values w.r.t. time.

        Args:
            attrs: Dataframe of attributes data in long format.
            plot_title_root: Common base of the plots' titles, to append based on each plot's group of attributes.

        Returns:
            An iterator over pairs of plots for each group of attributes in the data and their titles.
        """

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for attributes plot processor.

        Returns:
            Parser object for attributes plot processor.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--inputs", type=str, nargs="+", help="Data item for which to analyze attributes' evolution w.r.t. time"
        )
        parser.add_argument(
            "--normalize_time",
            action="store_true",
            help="Whether to normalize the values in the time axis between 0 and 1. By default, these values are "
            "between 0 and the count of data points in the attributes' data.",
        )
        parser.add_argument(
            "--data_name",
            type=str,
            default="data",
            help="Name to give in the legend to the variable representing the data the attributes come from",
        )
        parser.add_argument(
            "--attr_name",
            type=str,
            default="attr",
            help="Name to give in the legend to the variable representing the attributes' names",
        )
        parser.add_argument("--time_name", type=str, default="frame", help="Name to give in the plot to the time axis")
        parser.add_argument(
            "--hue",
            type=str,
            nargs="?",
            default="attr",
            help="Field of the attributes' data to use to assign the curves' hues",
        )
        parser.add_argument(
            "--style",
            type=str,
            nargs="?",
            default="data",
            help="Field of the attributes' data to use to assign the curves' styles",
        )
        parser.add_argument(
            "--no_title_in_plot",
            dest="display_title_in_plot",
            action="store_false",
            help="Do not display the title generated for the plot in the plot itself.",
        )
        return parser
