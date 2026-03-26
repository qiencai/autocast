"""Process full OSI-SAF dataset into train/valid/test splits.

Reads individual year netCDF files, fills missing days with linear temporal
interpolation (edge gaps filled by nearest-neighbour), and saves in the
same format as processed_osisaf_selectedyears:

  {split}/data.pt           -> {"data": (N,365,432,432,1), "constant_fields": (N,432,432,1)}
  {split}/constant_fields.pt -> land_mask (432,432)

Year splits:
  Train : 1979-2016  (38 years, after gap-filtering)
  Valid : 2017-2018  (2 years)
  Test  : 2019-2020  (2 years)

Years with any consecutive missing gap > MAX_CONSECUTIVE_GAP_DAYS are skipped
automatically (e.g. 1978 has a 297-day gap at the start of the record).
"""

import numpy as np
import pandas as pd
import xarray as xr
import torch
from pathlib import Path

# Years with a consecutive gap longer than this are skipped entirely.
MAX_CONSECUTIVE_GAP_DAYS = 14


# ---------------------------------------------------------------------------
# Gap checking
# ---------------------------------------------------------------------------

def max_consecutive_gap(nc_path: Path, year: int) -> int:
    """Return the longest run of consecutive missing days for this year."""
    ds = xr.open_dataset(nc_path)
    times_midnight = pd.DatetimeIndex(
        pd.to_datetime(ds["ice_conc"].time.values).normalize()
    )
    ds.close()

    full_index = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    full_index = full_index[~((full_index.month == 2) & (full_index.day == 29))]

    present = set(times_midnight.normalize())
    max_gap = 0
    run = 0
    for d in full_index:
        if d not in present:
            run += 1
            max_gap = max(max_gap, run)
        else:
            run = 0
    return max_gap


# ---------------------------------------------------------------------------
# Per-year processing with linear interpolation
# ---------------------------------------------------------------------------

def process_year(nc_path: Path, year: int) -> np.ndarray:
    """Load one year, fill missing days by linear interpolation.

    Returns
    -------
    np.ndarray of shape (365, H, W, 1), dtype float32.
    NaN land pixels are zeroed after interpolation.
    """
    print(f"  [{year}] Loading {nc_path.name}...")
    ds = xr.open_dataset(nc_path)
    sic = ds["ice_conc"]

    # ---- build the target 365-day index (no Feb 29) ----------------------
    full_index = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    # drop Feb 29 on leap years
    full_index = full_index[~((full_index.month == 2) & (full_index.day == 29))]
    assert len(full_index) == 365, f"Expected 365 days, got {len(full_index)}"

    # normalise timestamps to midnight so reindex matches
    times_midnight = pd.DatetimeIndex(
        pd.to_datetime(sic.time.values).normalize()
    )
    sic = sic.assign_coords(time=times_midnight)

    # reindex: days present → kept; missing days → NaN
    sic = sic.reindex(time=full_index)

    n_present = int((~np.isnan(sic.values[:, 0, 0])).sum())
    n_missing = 365 - n_present
    print(f"  [{year}] {n_present}/365 days present, {n_missing} to interpolate")

    if n_missing > 0:
        # linear interpolation for internal gaps; ffill/bfill for any short
        # edge gaps (caller already ensures max gap <= MAX_CONSECUTIVE_GAP_DAYS)
        sic = sic.interpolate_na(dim="time", method="linear")
        sic = sic.ffill("time").bfill("time")

    # compute to numpy: (365, H, W)
    data = sic.values.astype(np.float32)

    # clamp ocean values to valid range [0, 100] (percent conc)
    # land pixels are persistently NaN → set to 0
    data = np.clip(data, 0.0, 100.0)
    data = np.nan_to_num(data, nan=0.0)

    # add channel dim: (365, H, W, 1)
    data = data[:, :, :, np.newaxis]

    print(f"  [{year}] shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
    ds.close()
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_dir = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf")
    output_dir = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_full")
    land_mask_path = raw_dir / "land_mask.pt"

    # --- year splits --------------------------------------------------------
    train_years = list(range(1979, 2017))   # 1979-2016 (up to 38 years, gap-filtered)
    valid_years = list(range(2017, 2019))   # 2017-2018 (2 years)
    test_years  = list(range(2019, 2021))   # 2019-2020 (2 years)

    print("=" * 80)
    print("PROCESSING FULL OSI-SAF DATASET (with linear interpolation)")
    print("=" * 80)
    print(f"  Max consecutive gap allowed : {MAX_CONSECUTIVE_GAP_DAYS} days")
    print(f"  Train candidates: {train_years[0]}-{train_years[-1]}  ({len(train_years)} years)")
    print(f"  Valid : {valid_years[0]}-{valid_years[-1]}  ({len(valid_years)} years)")
    print(f"  Test  : {test_years[0]}-{test_years[-1]}   ({len(test_years)} years)")
    print(f"  Input : {raw_dir}")
    print(f"  Output: {output_dir}")

    # --- load land mask -------------------------------------------------------
    print(f"\nLoading land mask from {land_mask_path}...")
    land_mask = torch.load(land_mask_path, map_location="cpu")  # (H, W) or (H, W, 1)
    if land_mask.dim() == 2:
        land_mask = land_mask  # keep (H, W) for constant_fields.pt
    print(f"  Land mask shape: {tuple(land_mask.shape)}, dtype: {land_mask.dtype}")

    # --- create output dirs ---------------------------------------------------
    for split in ["train", "valid", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_years,
        "valid": valid_years,
        "test":  test_years,
    }

    for split_name, years in splits.items():
        print(f"\n{'=' * 80}")
        print(f"PROCESSING {split_name.upper()}  ({len(years)} years)")
        print("=" * 80)

        year_arrays = []
        processed_years = []

        for year in years:
            nc_path = raw_dir / f"osisaf_nh_{year}.nc"
            if not nc_path.exists():
                print(f"  SKIP  [{year}] file not found")
                continue
            try:
                gap = max_consecutive_gap(nc_path, year)
                if gap > MAX_CONSECUTIVE_GAP_DAYS:
                    print(f"  SKIP  [{year}] max consecutive gap = {gap} days (> {MAX_CONSECUTIVE_GAP_DAYS})")
                    continue
                print(f"  OK    [{year}] max gap = {gap} days — processing...")
                arr = process_year(nc_path, year)
                year_arrays.append(arr)
                processed_years.append(year)
            except Exception as exc:
                print(f"  ERROR [{year}]: {exc}")
                continue

        if not year_arrays:
            print(f"  ERROR: nothing processed for {split_name}!")
            continue

        print(f"\n  Stacking {len(year_arrays)} years → (N,365,H,W,1)...")
        stacked = np.stack(year_arrays, axis=0)   # (N, 365, H, W, 1)
        print(f"  Shape  : {stacked.shape}")
        print(f"  Years  : {processed_years[0]}–{processed_years[-1]}")
        print(f"  Range  : [{stacked.min():.3f}, {stacked.max():.3f}]")

        tensor_data = torch.from_numpy(stacked)   # float32

        # constant_fields: land mask expanded to (N, H, W, 1)
        n = tensor_data.shape[0]
        h, w = tensor_data.shape[2], tensor_data.shape[3]
        # land_mask is (H, W) → (N, H, W, 1)
        cf = land_mask.float().unsqueeze(-1).unsqueeze(0).expand(n, -1, -1, -1).contiguous()

        data_dict = {
            "data": tensor_data,
            "constant_fields": cf,
        }

        out_path = output_dir / split_name / "data.pt"
        print(f"  Saving data.pt → {out_path}")
        torch.save(data_dict, out_path)
        size_gb = out_path.stat().st_size / 1024**3
        print(f"  ✓  {size_gb:.2f} GB")

        # also save standalone constant_fields.pt (matches selectedyears layout)
        cf_path = output_dir / split_name / "constant_fields.pt"
        torch.save(land_mask, cf_path)
        print(f"  ✓  constant_fields.pt saved ({land_mask.shape})")

    # --- climatology from training data -------------------------------------
    # Mean across years for each DOY → (365, H, W, C), same logic as
    # build_climatology_baseline.py used for selectedyears.
    print(f"\n{'=' * 80}")
    print("BUILDING CLIMATOLOGY FROM TRAINING SPLIT")
    print("=" * 80)
    train_pt = output_dir / "train" / "data.pt"
    clim_pt  = output_dir / "climatology.pt"
    if train_pt.exists():
        d = torch.load(train_pt, map_location="cpu")
        train_data = d["data"].float()   # (N, 365, H, W, C)
        print(f"  Training tensor shape: {tuple(train_data.shape)}")
        climatology = train_data.mean(dim=0)  # (365, H, W, C)
        print(f"  Climatology shape    : {tuple(climatology.shape)}")
        print(f"  Value range          : [{climatology.min():.2f}, {climatology.max():.2f}]")
        torch.save(climatology, clim_pt)
        print(f"  ✓  Saved {clim_pt}")
    else:
        print("  WARNING: train/data.pt not found, skipping climatology")

    # --- final summary -------------------------------------------------------
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    for split_name in ["train", "valid", "test"]:
        p = output_dir / split_name / "data.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu")
            shape = tuple(d["data"].shape)
            cf_shape = tuple(d["constant_fields"].shape)
            size = p.stat().st_size / 1024**3
            print(f"  {split_name:6s}: data={shape}  constant_fields={cf_shape}  {size:.2f} GB")
    if clim_pt.exists():
        print(f"  climatology.pt: {tuple(torch.load(clim_pt, map_location='cpu').shape)}")


if __name__ == "__main__":
    main()
