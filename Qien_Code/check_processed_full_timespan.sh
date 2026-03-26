#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 01:00:00
#SBATCH --nodes 1
#SBATCH --gpus 0
#SBATCH --mem=0
#SBATCH --job-name check_processed_full

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

python3 - << 'EOF'
import torch, os

base = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_full"
total_trajectories = 0

for split in ["train", "valid", "test"]:
    fpath = os.path.join(base, split, "data.pt")
    d = torch.load(fpath, map_location="cpu")
    data = d["data"] if isinstance(d, dict) else d
    n, t = data.shape[0], data.shape[1]
    total_trajectories += n
    print(f"{split:6s}: {n} trajectories x {t} timesteps  (shape: {tuple(data.shape)})")

print(f"\nTotal trajectories (years): {total_trajectories}")
print(f"Each trajectory = 1 year of daily data (T timesteps = days in that year)")

# Check for any metadata alongside the data
for split in ["train", "valid", "test"]:
    split_dir = os.path.join(base, split)
    for f in os.listdir(split_dir):
        if f != "data.pt":
            print(f"  Found extra file in {split}/: {f}")
EOF
