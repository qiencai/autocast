#!/bin/bash
#SBATCH --job-name=copy_mask
#SBATCH --nodes=1
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=copy_mask_%j.log

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast
source .venv/bin/activate

python3 << 'EOF'
import torch
from pathlib import Path

mask = torch.load('/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf/land_mask.pt')
print(f"Loaded mask: shape={mask.shape}, dtype={mask.dtype}")

base_dir = Path('/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_selectedyears')

# Copy to train, valid, test
for split in ['train', 'valid', 'test']:
    dest = base_dir / split / 'constant_fields.pt'
    torch.save(mask, dest)
    print(f"✓ Saved to {dest}")

print("\nAll constant_fields.pt files created successfully!")
EOF
