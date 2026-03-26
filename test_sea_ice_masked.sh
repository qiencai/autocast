#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos=turing
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test_sea_ice_masked
#SBATCH --output=logs/test_sea_ice_masked_%j.log
#SBATCH --error=logs/test_sea_ice_masked_%j.err

# Ensure logs directory exists
mkdir -p logs

# Navigate to repo root
cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

# Load environment
module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

# Activate conda environment
source /bask/homes/q/qqaa9560/miniconda3/bin/activate autocast

# Set full error tracing for Hydra
export HYDRA_FULL_ERROR=1

# Run the test with full debugging
echo "Starting test at $(date)"
echo "Python: $(which python)"
echo "Autocast version:"
python -c "import autocast; print(autocast.__file__)"

# Run the command
echo "Running: autocast train-eval experiment=sea_ice_masked_flow_matching"
autocast train-eval experiment=sea_ice_masked_flow_matching

echo "Test completed at $(date)"
