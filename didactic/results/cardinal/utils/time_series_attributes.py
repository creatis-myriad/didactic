import logging
from argparse import ArgumentParser
from typing import Dict, Sequence

import numpy as np
from vital.data.cardinal.config import TimeSeriesAttribute
from vital.data.cardinal.utils.data_struct import View
from vital.results.processor import ResultsProcessor

logger = logging.getLogger(__name__)


class TimeSeriesAttributesMixin(ResultsProcessor):
    """Mixin that groups together various behaviors regarding how to handle time-series attribute data.

    - How to access the attributes;
    """

    def __init__(self, attrs: Sequence[str], **kwargs):
        """Initializes class instance.

        Args:
            attrs: Labels identifying attributes of interest.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self.attrs = attrs

    def _extract_attributes_from_result(self, result: View, item_tag: str) -> Dict[str, np.ndarray]:
        view_data_attrs = result.attrs[item_tag]
        attrs = {attr: view_data_attrs[attr] for attr in self.attrs if attr in view_data_attrs}
        if missing_attrs := set(self.attrs) - attrs.keys():
            logger.warning(
                f"Requested attributes {missing_attrs} were not available for '{result.id}'. The attributes listed "
                f"were thus ignored. To avoid this warning, either remove these attributes from those required for "
                f"your task, or run your task on data that provides those attributes."
            )
        return attrs

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for processors that need to access time-series attributes.

        Returns:
            Parser object for processors that need to access time-series attributes.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--attrs",
            type=str,
            nargs="+",
            default=list(TimeSeriesAttribute),
            choices=list(TimeSeriesAttribute),
            help="Labels identifying attributes of interest",
        )
        return parser
