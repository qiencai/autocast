#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 01:00:00
#SBATCH --nodes 1
#SBATCH --gpus 0
#SBATCH --mem=0
#SBATCH --job-name climatology_full
#SBATCH --output=logs/climatology_full_%j.out
#SBATCH --error=logs/climatology_full_%j.err

set -e

mkdir -p logs

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

echo "Starting at $(date)"

python3 - << 'EOF'
import torch
from pathlib import Path

data_root = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_full")
train_pt = data_root / "train" / "data.pt"
out_pt   = data_root / "climatology.pt"

print(f"Loading training data from {train_pt} ...")
d = torch.load(train_pt, map_location="cpu", weights_only=False)
data = d["data"].float()   # (N_years, 365, H, W, C)
print(f"  Training tensor shape: {tuple(data.shape)}")

climatology = data.mean(dim=0)  # (365, H, W, C)
print(f"  Climatology shape    : {tuple(climatology.shape)}")
print(f"  Value range          : [{climatology.min():.2f}, {climatology.max():.2f}]")

torch.save(climatology, out_pt)
print(f"  Saved -> {out_pt}")
EOF

echo "Finished at $(date)"
