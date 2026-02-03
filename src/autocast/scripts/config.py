"""Shared config utilities for AutoCast scripts."""

import logging
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def save_resolved_config(
    config: DictConfig, work_dir: Path, filename: str = "resolved_config.yaml"
) -> Path:
    """Save a resolved config YAML file and return the path."""
    resolved_cfg = OmegaConf.to_container(config, resolve=True)
    output_path = work_dir / filename
    with open(output_path, "w") as f:
        yaml.dump(resolved_cfg, f)
    log.info("Wrote resolved config to %s", output_path.resolve())
    return output_path
