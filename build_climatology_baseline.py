"""Build a day-of-year climatology baseline from the training split.

Loads the training data (shape N_years × 365 × W × H × C), averages across
years for each DOY, and saves a (365, W, H, C) tensor to:
    <data_root>/climatology.pt

Usage:
    python build_climatology_baseline.py
"""

from pathlib import Path

import torch

DATA_ROOT = Path(
    "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice"
    "/processed_osisaf_selectedyears"
)
TRAIN_PT = DATA_ROOT / "train" / "data.pt"
OUT_PT = DATA_ROOT / "climatology.pt"


def main() -> None:
    print(f"Loading training data from {TRAIN_PT} …")
    d = torch.load(TRAIN_PT, weights_only=False)
    data: torch.Tensor = d["data"]  # (N_years, 365, W, H, C)
    print(f"  Training data shape: {tuple(data.shape)}")

    # Mean across years → (365, W, H, C)
    climatology = data.float().mean(dim=0)
    print(f"  Climatology shape:   {tuple(climatology.shape)}")
    print(f"  Value range: [{climatology.min():.2f}, {climatology.max():.2f}]")

    print(f"Saving climatology to {OUT_PT} …")
    torch.save(climatology, OUT_PT)
    print("Done.")


if __name__ == "__main__":
    main()
