from typing import Iterator, Tuple

import pandas as pd
from matplotlib.axes import Axes
from vital.data.cardinal.utils.attributes import plot_attributes_wrt_time

from didactic.results.cardinal.utils.attributes_plot import AttributesPlots
from didactic.results.cardinal.utils.time_series_attributes import TimeSeriesAttributesMixin


class TimeSeriesAttributesPlots(TimeSeriesAttributesMixin, AttributesPlots):
    """Class that plots time-series attributes w.r.t. time."""

    def _plot_attributes_wrt_time(self, attrs: pd.DataFrame, plot_title_root: str) -> Iterator[Tuple[str, Axes]]:
        yield from plot_attributes_wrt_time(
            attrs,
            plot_title_root=plot_title_root,
            data_name=self.data_name,
            attr_name=self.attr_name,
            time_name=self.time_name,
            hue=self.hue,
            style=self.style,
            display_title_in_plot=self.display_title_in_plot,
        )


def main():
    """Run the script."""
    TimeSeriesAttributesPlots.main()


if __name__ == "__main__":
    main()
