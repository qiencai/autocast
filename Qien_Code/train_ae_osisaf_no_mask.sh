#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 02:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --mem=0
#SBATCH --job-name ae_no_mask_osisaf

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/.conda/envs/autocast/bin/python \
  -m autocast.scripts.train.autoencoder \
  experiment=seaice_autoencoder_no_mask \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  logging.wandb.enabled=true
