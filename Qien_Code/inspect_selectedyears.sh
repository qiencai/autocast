#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 00:20:00
#SBATCH --nodes 1
#SBATCH --gpus 0
#SBATCH --mem=0
#SBATCH --job-name inspect_selectedyears

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python3 - << 'EOF'
import torch, os

base = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_selectedyears"

for split in ["train", "valid", "test"]:
    print(f"\n=== {split} ===")
    split_dir = os.path.join(base, split)
    for fname in os.listdir(split_dir):
        fpath = os.path.join(split_dir, fname)
        obj = torch.load(fpath, map_location="cpu")
        if isinstance(obj, dict):
            print(f"  {fname}: dict keys={list(obj.keys())}")
            for k, v in obj.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: shape={tuple(v.shape)}, dtype={v.dtype}, min={v.min():.4f}, max={v.max():.4f}")
                else:
                    print(f"    {k}: {type(v)} = {v}")
        elif hasattr(obj, 'shape'):
            print(f"  {fname}: tensor shape={tuple(obj.shape)}, dtype={obj.dtype}, min={obj.min():.4f}, max={obj.max():.4f}")
        else:
            print(f"  {fname}: {type(obj)}")

# Also check climatology
cpath = os.path.join(base, "climatology.pt")
if os.path.exists(cpath):
    obj = torch.load(cpath, map_location="cpu")
    print("\n=== climatology.pt ===")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")
    elif hasattr(obj, 'shape'):
        print(f"  tensor shape={tuple(obj.shape)}, dtype={obj.dtype}")
EOF
