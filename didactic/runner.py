import comet_ml  # noqa
import hydra
from omegaconf import DictConfig
from vital.runner import VitalRunner


class DidacticRunner(VitalRunner):
    """Entry-point for a `VitalRunner` that adds the `didactic` config dir to the Hydra search path."""

    @staticmethod
    @hydra.main(version_base=None, config_path="config", config_name="vital_default")
    def run_system(cfg: DictConfig) -> None:  # noqa: D102
        VitalRunner.run_system(cfg)


def main():
    """Run the script."""
    DidacticRunner.main()


if __name__ == "__main__":
    main()
