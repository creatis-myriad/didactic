from vital import get_vital_root

from didactic.results.cardinal.utils.image_attributes import ImageAttributesMixin
from didactic.results.cardinal.utils.temporal_metrics import TemporalMetrics


class ImageTemporalMetrics(ImageAttributesMixin, TemporalMetrics):
    """Class that computes temporal coherence metrics on sequences of image attributes' values."""

    desc = f"seg_{TemporalMetrics.desc}"
    default_attribute_statistics_cfg = get_vital_root() / "data/camus/statistics/image_attr_stats.yaml"


def main():
    """Run the script."""
    ImageTemporalMetrics.main()


if __name__ == "__main__":
    main()
