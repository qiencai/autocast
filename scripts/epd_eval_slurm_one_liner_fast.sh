#!/bin/bash
# Mapping guide:
# - Keys from configs/hydra/launcher/slurm.yaml map to CLI as
#   hydra.launcher.<key>=<value>
#   e.g. timeout_min -> hydra.launcher.timeout_min=30
# - In `autocast train-eval`, positional overrides apply to TRAIN.
# - Eval-specific settings go under `--eval-overrides`.
# - For different SLURM times per step, set:
#   TRAIN: hydra.launcher.timeout_min=<minutes>
#   EVAL:  --eval-overrides hydra.launcher.timeout_min=<minutes>

uv run autocast train-eval --mode slurm --detach \
    --dataset advection_diffusion_multichannel_64_64 \
    hydra.launcher.timeout_min=30 \
    datamodule.use_normalization=false \
    logging.wandb.enabled=false \
    processor@model.processor=flow_matching_vit \
    datamodule.batch_size=128 \
    optimizer.learning_rate=0.0002 \
    encoder@model.encoder=dc_deep_256 \
    decoder@model.decoder=dc_deep_256 \
    model.train_in_latent_space=true \
    model.processor.backbone.hid_channels=512 \
    trainer.max_epochs=1 \
    +trainer.limit_train_batches=1 \
    +trainer.limit_val_batches=1 \
    +trainer.limit_test_batches=1 \
    +autoencoder_checkpoint=/projects/u5gf/ai4physics/outputs/2026-02-06/advection_diffusion_multichannel_64_64_no_norm/autoencoder.ckpt \
    --eval-overrides \
    hydra.launcher.timeout_min=10 \
    trainer.limit_test_batches=1 \
    +model.n_members=10

