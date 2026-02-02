#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
shift 3

# Run script
RUN_DIR="${PWD}/outputs/${LABEL}/${OUTPATH}/eval"
CKPT_PATH="${PWD}/outputs/${LABEL}/${OUTPATH}/encoder_processor_decoder.ckpt"
VIDEO_DIR="${RUN_DIR}/videos"

uv run python -m autocast.scripts.eval.encoder_processor_decoder \
	--config-path=configs/ \
	--config-name=encoder_processor_decoder \
	--checkpoint=${CKPT_PATH} \
	--batch-index=0 \
	--batch-index=1 \
	--batch-index=2 \
	--batch-index=3 \
	--video-dir=${VIDEO_DIR} \
	hydra.run.dir=${RUN_DIR} \
	datamodule=${DATAPATH} \
	datamodule.data_path=${AUTOCAST_DATASETS}/${DATAPATH} \
	"$@"
