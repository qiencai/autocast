"""Train-then-evaluate encoder-processor-decoder in a single Hydra job."""

import logging
import os
from pathlib import Path
from typing import cast

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf, open_dict

from autocast.scripts.eval.encoder_processor_decoder import run_evaluation
from autocast.scripts.setup import setup_datamodule, setup_epd_model
from autocast.scripts.training import run_training
from autocast.scripts.utils import get_default_config_path

log = logging.getLogger(__name__)


def _apply_umask(cfg: DictConfig) -> None:
    umask_value = cfg.get("umask")
    if umask_value is None:
        return
    parsed = int(str(umask_value), 8)
    os.umask(parsed)
    log.info("Applied process umask %s", umask_value)


def _resolve_train_checkpoint(cfg: DictConfig, work_dir: Path) -> Path:
    train_eval_cfg = cfg.get("train_eval", {})
    checkpoint_override = train_eval_cfg.get("checkpoint")
    if checkpoint_override is not None:
        return Path(checkpoint_override).expanduser().resolve()

    output_cfg = cfg.get("output", {})
    explicit_checkpoint = output_cfg.get("checkpoint_path")
    if explicit_checkpoint is not None:
        return Path(explicit_checkpoint).expanduser().resolve()

    checkpoint_name = output_cfg.get(
        "checkpoint_name", "encoder_processor_decoder.ckpt"
    )
    return (work_dir / checkpoint_name).resolve()


def _apply_eval_overrides(cfg: DictConfig) -> DictConfig:
    train_eval_cfg = cast(DictConfig, cfg.get("train_eval", {}))
    eval_overrides = list(train_eval_cfg.get("eval_overrides", []))
    if not eval_overrides:
        return cast(DictConfig, cfg)
    merged = OmegaConf.merge(cfg, OmegaConf.from_dotlist(eval_overrides))
    if not isinstance(merged, DictConfig):
        msg = "Expected DictConfig after applying eval overrides"
        raise TypeError(msg)
    return merged


@hydra.main(
    version_base=None,
    config_path=get_default_config_path(),
    config_name="encoder_processor_decoder",
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for single-job train→eval flow."""
    logging.basicConfig(level=logging.INFO)
    _apply_umask(cfg)

    work_dir = Path.cwd()

    datamodule, cfg, stats = setup_datamodule(cfg)
    L.seed_everything(cfg.get("seed", 42), workers=True)

    model = setup_epd_model(cfg, stats, datamodule=datamodule)

    output_cfg = cfg.get("output", {})
    skip_test = output_cfg.get("skip_test", False)
    output_checkpoint = output_cfg.get("checkpoint_path")

    run_training(
        cfg,
        model,
        datamodule,
        work_dir,
        skip_test=skip_test,
        output_checkpoint_path=output_checkpoint,
        job_type="train-encoder-processor-decoder",
    )

    checkpoint_path = _resolve_train_checkpoint(cfg, work_dir)

    train_eval_cfg = cast(DictConfig, cfg.get("train_eval", {}))
    eval_subdir = train_eval_cfg.get("eval_subdir", "eval")
    eval_work_dir = (work_dir / eval_subdir).resolve()
    eval_work_dir.mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        eval_cfg_section = cfg.get("eval")
        if not isinstance(eval_cfg_section, DictConfig):
            eval_cfg_section = OmegaConf.create({})
            cfg["eval"] = eval_cfg_section
        eval_cfg_section["checkpoint"] = str(checkpoint_path)

        batch_indices = train_eval_cfg.get("batch_indices")
        if batch_indices is not None:
            eval_cfg_section["batch_indices"] = batch_indices

        video_dir = train_eval_cfg.get("video_dir")
        if video_dir is not None:
            eval_cfg_section["video_dir"] = video_dir

    eval_cfg = _apply_eval_overrides(cfg)
    run_evaluation(eval_cfg, work_dir=eval_work_dir)


if __name__ == "__main__":
    main()
