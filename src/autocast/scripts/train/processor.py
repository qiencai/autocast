"""Train a ProcessorModel directly on encoded datasets."""

import logging

import lightning as L

from autocast.scripts.cli import parse_common_args
from autocast.scripts.config import load_config, resolve_work_dir
from autocast.scripts.setup import setup_datamodule, setup_processor_model
from autocast.scripts.training import run_training

log = logging.getLogger(__name__)


def main():
    """CLI entrypoint for training the processor."""
    args = parse_common_args(
        description=(
            "Train a ProcessorModel directly on encoded datasets with "
            "Hydra-configured processor components."
        ),
        config_name="processor",
    )

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Compose config
    cfg = load_config(args)
    work_dir = resolve_work_dir(args.overrides)

    # Setup datamodule and resolve config
    datamodule, cfg, stats = setup_datamodule(cfg)

    # Seed
    L.seed_everything(cfg.get("seed", 42), workers=True)

    # Setup Model
    model = setup_processor_model(cfg, stats)

    # Run Training
    run_training(
        cfg,
        model,
        datamodule,
        work_dir,
        skip_test=args.skip_test,
        output_checkpoint_path=args.output_checkpoint,
        job_type="train-processor",
    )


if __name__ == "__main__":
    main()
