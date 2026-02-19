#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 0:30:00
#SBATCH --nodes 1
#SBATCH --gpus 0
#SBATCH --tasks-per-node 4
#SBATCH --job-name wrap_osisaf_data

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import torch
from pathlib import Path

base_path = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all")

# Wrap each split's raw tensor in the expected dict format
for split in ["train", "valid", "test"]:
    data_file = base_path / split / "data.pt"
    
    print(f"\nProcessing {split}...")
    print(f"  Loading raw tensor from: {data_file}")
    
    # Load the raw tensor
    raw_tensor = torch.load(data_file, map_location='cpu')
    print(f"  Shape: {raw_tensor.shape}")
    
    # Wrap in dict
    wrapped_data = {"data": raw_tensor}
    
    # Save back (overwrites the file)
    print(f"  Wrapping in dict and saving...")
    torch.save(wrapped_data, data_file)
    
    print(f"  ✓ Done!")

print("\n" + "="*70)
print("All data files wrapped successfully!")
print("="*70)
EOF
