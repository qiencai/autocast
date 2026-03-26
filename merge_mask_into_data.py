#!/usr/bin/env python3
"""Merge land mask into data.pt files as constant_fields."""

import torch
from pathlib import Path

base_dir = Path('/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_selectedyears')

# Load the land mask (created earlier)
mask = torch.load(base_dir / 'train' / 'constant_fields.pt')
print(f"Loaded mask: shape={mask.shape}, dtype={mask.dtype}")

for split in ['train', 'valid', 'test']:
    data_path = base_dir / split / 'data.pt'
    
    # Load data
    data_dict = torch.load(data_path)
    print(f"\n{split.upper()} before merge:")
    print(f"  Keys: {list(data_dict.keys())}")
    print(f"  data.shape: {data_dict['data'].shape}")
    print(f"  Has constant_fields: {'constant_fields' in data_dict}")
    
    # Merge mask as constant_fields: shape (N, W, H, 1) so that
    # dataset[traj_idx] returns (W, H, 1) satisfying Float[Tensor, 'spatial *spatial channel']
    n_traj = data_dict['data'].shape[0]
    mask_expanded = mask.unsqueeze(-1).unsqueeze(0).expand(n_traj, -1, -1, -1).contiguous()
    data_dict['constant_fields'] = mask_expanded
    
    # Save back
    torch.save(data_dict, data_path)
    print(f"\n{split.upper()} after merge:")
    print(f"  Keys: {list(data_dict.keys())}")
    print(f"  constant_fields.shape: {data_dict['constant_fields'].shape}")
    print(f"  ✓ Saved to {data_path}")

print("\n✅ Mask successfully merged into all data.pt files!")
