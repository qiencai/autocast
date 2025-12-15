#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 3:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --tasks-per-node 36
#SBATCH --job-name train_and_eval_autoencoder
#SBATCH --output=logs/train_and_eval_autoencoder_%j.out
#SBATCH --error=logs/train_and_eval_autoencoder_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0
module load FFmpeg/6.0-GCCcore-12.3.0

# Activate virtual environment
cd autocast # Change to the directory where your code is located
source venv/bin/activate

# Pip install to get current version of code 
uv sync --extra dev

# Run Autocast Code 

# Train
uv run train_encoder_processor_decoder \
    --config-path=configs/ \
	--work-dir=outputs/encoder_processor_decoder_run
	
# Evaluate
uv run evaluate_encoder_processor_decoder \
	--config-path=configs/ \
	--work-dir=outputs/processor_eval \
	--checkpoint=outputs/encoder_processor_decoder_run/encoder_processor_decoder.ckpt \
	--batch-index=0 --batch-index=3 \
	--video-dir=outputs/encoder_processor_decoder_run/videos