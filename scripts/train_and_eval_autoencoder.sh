#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 3:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --tasks-per-node 36
#SBATCH --job-name train_and_eval_autoencoder
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0
module load FFmpeg/6.0-GCCcore-12.3.0

# Activate virtual environment - This assumes you have already created a virtual environment in the project directory
# If you haven't, replace with `uv venv`
source .venv/bin/activate

# Pip install to get current version of code    
uv sync --extra dev

# First define a timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Now define a job name. This will be used to create a unique working directory for outputs.
# Change this as needed
JOB_NAME="autoencoder_run"

# Finally, this builds the working directory path. 
# It follows the structure outputs/JOB_NAME/TIMESTAMP
WORKING_DIR="outputs/${JOB_NAME}/${TIMESTAMP}"

# ---------------- Code to train and evaluate the model ----------------

# Train
uv run train_autoencoder \
    --config-path=configs/ \
	--work-dir=${WORKING_DIR}
	
# Evaluate
uv run evaluate_autoencoder \
	--config-path=configs/ \
	--work-dir=${WORKING_DIR} \
	--checkpoint=${WORKING_DIR}/autoencoder.ckpt \
	--batch-index=0 --batch-index=3 \
	--video-dir=${WORKING_DIR}/videos

