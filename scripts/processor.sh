#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
shift 3

WORKDIR="${PWD}/outputs/${LABEL}/${OUTPATH}"

OVERRIDES=(
	"hydra.run.dir=${WORKDIR}"
	"datamodule=${DATAPATH}"
	"datamodule.data_path=${AUTOCAST_DATASETS}/${DATAPATH}"
)

# Run script
# Optional overrides you can add via CLI:
#   logging.wandb.enabled=true
uv run train_processor "${OVERRIDES[@]}" "$@"
    
