#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 00:30:00
#SBATCH --nodes 1
#SBATCH --gpus 0
#SBATCH --mem=0
#SBATCH --job-name verify_osisaf_full
#SBATCH --output=logs/verify_osisaf_full_%j.out
#SBATCH --error=logs/verify_osisaf_full_%j.err

set -e

mkdir -p logs

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

python3 - << 'EOF'
import torch
import os
from pathlib import Path

base = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_full")
ref  = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_selectedyears")

OK = "\u2713"
FAIL = "\u2717"

issues = []

def check(cond, msg):
    if cond:
        print(f"  {OK}  {msg}")
    else:
        print(f"  {FAIL}  FAIL: {msg}")
        issues.append(msg)

print("=" * 70)
print("VERIFYING processed_osisaf_full")
print("=" * 70)

# ---- per-split checks -------------------------------------------------------
for split in ["train", "valid", "test"]:
    print(f"\n--- {split} ---")
    dp  = base / split / "data.pt"
    cfp = base / split / "constant_fields.pt"

    # file existence
    check(dp.exists(),  f"{split}/data.pt exists")
    check(cfp.exists(), f"{split}/constant_fields.pt exists")
    if not dp.exists():
        continue

    d = torch.load(dp, map_location="cpu", weights_only=False)

    # keys
    check("data" in d,             f"data.pt has 'data' key")
    check("constant_fields" in d,  f"data.pt has 'constant_fields' key")

    data = d["data"]
    cf   = d["constant_fields"]

    # shape checks
    N, T, H, W, C = data.shape
    check(T == 365, f"T=365 (got {T})")
    check(H == 432, f"H=432 (got {H})")
    check(W == 432, f"W=432 (got {W})")
    check(C == 1,   f"C=1   (got {C})")
    check(cf.shape == (N, H, W, 1), f"constant_fields shape ({N},{H},{W},1) (got {tuple(cf.shape)})")
    check(data.dtype == torch.float32, f"dtype float32 (got {data.dtype})")

    # value sanity: ocean values should be in [0, 100]
    finite = torch.isfinite(data)
    pct_finite = finite.float().mean().item() * 100
    check(pct_finite == 100.0, f"no NaN/Inf in data (finite={pct_finite:.2f}%)")
    vmin, vmax = data.min().item(), data.max().item()
    check(vmin >= 0.0,   f"min >= 0   (got {vmin:.3f})")
    check(vmax <= 100.0, f"max <= 100 (got {vmax:.3f})")

    # constant_fields should be 0/1 (land mask)
    cf_unique = cf[0].unique()
    check(len(cf_unique) <= 2, f"constant_fields binary-ish (unique vals: {cf_unique.tolist()[:5]})")

    # file size
    size_gb = dp.stat().st_size / 1024**3
    print(f"  i  {split}/data.pt: N={N} years, {size_gb:.2f} GB")

    # compare shape format with selectedyears reference
    ref_dp = ref / split / "data.pt"
    if ref_dp.exists():
        ref_d = torch.load(ref_dp, map_location="cpu", weights_only=False)
        ref_data = ref_d["data"]
        check(ref_data.shape[1:] == data.shape[1:],
              f"shape[1:] matches selectedyears {tuple(ref_data.shape[1:])} == {tuple(data.shape[1:])}")

    # standalone constant_fields.pt
    cf_standalone = torch.load(cfp, map_location="cpu", weights_only=False)
    check(cf_standalone.shape == (H, W), f"constant_fields.pt shape ({H},{W}) (got {tuple(cf_standalone.shape)})")

# ---- climatology ------------------------------------------------------------
print(f"\n--- climatology ---")
clim_p = base / "climatology.pt"
check(clim_p.exists(), "climatology.pt exists")
if clim_p.exists():
    clim = torch.load(clim_p, map_location="cpu", weights_only=False)
    check(clim.shape == (365, 432, 432, 1), f"shape (365,432,432,1) (got {tuple(clim.shape)})")
    check(torch.isfinite(clim).all().item(), "no NaN/Inf in climatology")
    print(f"  i  value range: [{clim.min():.2f}, {clim.max():.2f}]")

# ---- compare with selectedyears climatology ---------------------------------
ref_clim_p = ref / "climatology.pt"
if ref_clim_p.exists() and clim_p.exists():
    print(f"\n--- vs selectedyears climatology ---")
    ref_clim = torch.load(ref_clim_p, map_location="cpu", weights_only=False)
    clim     = torch.load(clim_p,     map_location="cpu", weights_only=False)
    check(ref_clim.shape == clim.shape, f"same shape as selectedyears ({tuple(ref_clim.shape)})")
    print(f"  i  selectedyears range: [{ref_clim.min():.2f}, {ref_clim.max():.2f}]")
    print(f"  i  full        range  : [{clim.min():.2f}, {clim.max():.2f}]")

# ---- summary ----------------------------------------------------------------
print("\n" + "=" * 70)
if issues:
    print(f"RESULT: {len(issues)} ISSUE(S) FOUND:")
    for i in issues:
        print(f"  {FAIL}  {i}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)
EOF
