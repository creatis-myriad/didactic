import logging

logger = logging.getLogger(__name__)


def main():
    """Run the script."""
    import shutil
    from argparse import ArgumentParser
    from pathlib import Path

    from vital.utils.logging import configure_logging

    configure_logging(log_to_console=True, console_level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("ckpts", nargs="+", type=Path, help="Paths to model checkpoints to copy")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models"),
        help="Root directory under which to save the copied model checkpoints",
    )
    parser.add_argument(
        "--copy_filename",
        type=str,
        default="model_{}.ckpt",
        help="Path to a CSV file in which to save the compiled results. The variable in the template will be "
        "substituted with the index of the ckpt in the `ckpts` list",
    )
    args = parser.parse_args()

    # Prepare the output folder
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, ckpt in enumerate(args.ckpts):
        dest = args.output_dir / args.copy_filename.format(idx)
        logger.info(f"Copying model ckpt: '{ckpt}' to new destination: '{dest}'")
        shutil.copy(ckpt, dest)


if __name__ == "__main__":
    main()
