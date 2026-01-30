"""Shared configuration loading for AutoCast scripts."""

import argparse

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_config(args: argparse.Namespace) -> DictConfig:
    """Load and resolve the Hydra configuration based on CLI arguments."""
    config_dir = args.config_dir.resolve()
    overrides = args.overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        hydra_cfg = compose(config_name=args.config_name, overrides=list(overrides))
    return hydra_cfg
