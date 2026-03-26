# Implementation Guide: Adding Cyclic Date-of-Year and Land Mask as Model Inputs

## Overview
This guide documents the refactored approach for adding cyclic date-of-year encoding and binary land masks as forcing inputs to the flow-matching model, without making them prediction targets.

---

## Current Code Changes Summary

### 1. **Type System** (`types/batch.py`)
**What changed:**
- Added `constant_doy_scalars: TensorBC | None = None` field to both `Sample` and `Batch` dataclasses
- Updated `repeat()` and `to()` methods to handle the new field
- Updated `collate_batches()` to properly stack doy scalars across samples

**Why separate storage?**
- Clear semantic separation: date-of-year scalars are a *time feature*, not user-defined scalars
- Prevents dimension indexing bugs if constant_scalars grows in future
- Self-documenting API: readers immediately know what each field is for

---

### 2. **Dataset** (`data/dataset.py`)
**What changed:**
- Added `self.all_constant_doy_scalars = []` list for separate storage
- In subtrajectory loop: compute `t_init = doy_offset + sub_idx * stride + n_steps_input`
  - This is the **initialization date** (first forecast step = last input timestep)
  - Compute phase as `2π * t_init / 365.25`
  - Store `[sin(phase), cos(phase)]` as 2D doy scalars
- Updated `__getitem__()` to include doy scalars in returned Sample

**Key design decision: Initialization date vs alternatives**
- ✅ **Initialization date (current)**: Represents when forecasting begins (most useful for models)
- ❌ Mid-point: Less clear semantically for initialization
- ❌ First timestep: Doesn't reflect forecast timing
- ❌ Last timestep: Same as initialization, just different naming

**The doy_offset parameter:**
- Set in datamodule config to day-of-year of first data timestep
- Example: `doy_offset: 1` if data starts on January 1

---

### 3. **DataModule** (`data/datamodule.py`)
**What changed:**
- Added `doy_offset: int = 0` parameter to `__init__`
- Passes `doy_offset` to all 5 dataset instances (train, val, test, rollout_val, rollout_test)
- Defaults to 0 if not specified (backward compatible)

**Config usage:**
```yaml
# configs/datamodule/osisaf_nh_sic.yaml
_target_: autocast.data.datamodule.SpatioTemporalDataModule
data_path: /path/to/data
doy_offset: 1  # January 1 = day 1 of year
```

---

### 4. **Model Rollout** (`models/encoder_processor_decoder.py`)
**What changed:**
- `_advance_batch()` now calls `_advance_doy_scalars()` on the separate `constant_doy_scalars` field
- `_advance_doy_scalars()` updates the phase angle during rollout:
  1. Recover current phase: `θ = atan2(sin, cos)`
  2. Advance by stride: `θ_new = θ + 2π * stride / 365.25`
  3. Store updated: `[sin(θ_new), cos(θ_new)]`

**Why this matters:**
- Ground truth date at every rollout step (not predicted)
- Allows model to learn seasonal patterns that vary with time
- Memory efficient: only 2 scalars per sample

---

### 5. **Setup/Statistics** (`scripts/setup.py`)
**What changed:**
- Tracks `n_constant_doy_scalars` separately from `n_constant_scalars`
- Allows model wiring to know doy scalars are present

---

## Data Flow: How DOY Scalars Reach the Model

```
Dataset.__getitem__() 
  → returns Sample with constant_doy_scalars=[sin, cos]
  
collate_batches()
  → stacks samples into Batch.constant_doy_scalars (shape: [B, 2])
  
Model.forward()
  → TODO: encode into global_cond or process separately
  
RolloutMixin._advance_batch()
  → _advance_doy_scalars() updates phase for next step
```

---

## Land Mask: Where We Are Now

### Current Status
- ✅ `constant_fields` already loaded from HDF5
- ✅ Carried through rollout unchanged
- ❌ **Not yet used by DCEncoder** (current default)

### Path Forward for Land Mask
Two options:

**Option A: Add to existing constant_fields** (RECOMMENDED)
1. Place binary land mask in HDF5 as `constant_fields` key (shape: `[N, W, H, 1]`)
2. Map this through the model using one of:
   - Switch encoder to `PermuteConcat` (already concatenates constant_fields with input)
   - Add ~3 lines to DCEncoder to concatenate constant_fields
3. **Advantage**: Self-contained in data pipeline, no code changes to Sample/Batch

**Option B: Separate constant_mask_fields**
1. Add new field like we did for doy_scalars
2. Explicit in code but more complex
3. Better if mask has different semantics than other fields

**Recommendation**: Go with Option A for now. Use `constant_fields` for land mask, add support in encoder.

---

## ERA5 Integration: Future Work

### Current Status
✅ Downloaded: `/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/era5_single_levels_2013_2020.grib` (15GB)

### Next Steps (Deferred)
1. Parse GRIB file, extract relevant features (wind, pressure, temperature, etc.)
2. Regrid to OSISaf grid resolution
3. Temporal alignment with OSISaf timestamps
4. Decide: ERA5 as forcing (like doy) or input fields (like channel concatenation)?
5. Potential approach: ERA5 as additional `constant_scalars` or separate input stream

### Why Deferred
- More complex spatial-temporal alignment needed
- Multiple features to extract and process
- Decide on best integration point (global_cond vs input_fields)
- Better to validate DOY + mask first

---

## Best Practices Checklist

### ✅ For Date-of-Year Encoding
- [x] Store as separate `constant_doy_scalars` field (clean semantics)
- [x] Use initialization date (start of forecast)
- [x] Use sin/cos encoding (captures cyclicity)
- [x]Advance during rollout (ground truth throughout)
- [x] Set via `doy_offset` in config

### ✅ For Land Mask Input
- [ ] Store in HDF5 as `constant_fields` with shape `[N, W, H, 1]`
- [ ] Choose encoder that uses constant_fields (PermuteConcat or enhanced DCEncoder)
- [ ] Configure in experiment YAML if using PermuteConcat

### ⚠️ For ERA5 (Later)
- [ ] Parse GRIB and extract minimal needed features
- [ ] Ensure temporal-spatial alignment with OSISaf
- [ ] Test with simplified ERA5 subset first
- [ ] Decide forcing vs input_fields integration

---

## Next Immediate Steps

1. **Create land mask binary array**
   - SpatioTemporal shape: `[N_trajectories, W, H, 1]`
   - Values: 1.0 where ice can exist, 0.0 where land/out-of-domain
   - Save to HDF5 in existing/new file

2. **Test DOY scalars integration**
   - Verify doy_offset parameter propagates through config
   - Check Batch contains constant_doy_scalars after collation
   - Test _advance_doy_scalars in rollout

3. **Wire mask and doy into model**
   - Choose encoder (suggest PermuteConcat for now)
   - Verify shapes flow correctly through encoder→processor→decoder
   - Test training loop

4. **Validate learning**
   - Train small model with doy + mask
   - Check model learns meaningful representations
   - Compare with baseline (no doy/mask)

---

## Files Modified

| File | Changes | Status |
|------|---------|--------|
| `types/batch.py` | Added constant_doy_scalars field | ✅ Complete |
| `data/dataset.py` | Compute & store doy scalars | ✅ Complete |
| `data/datamodule.py` | Accept & pass doy_offset | ✅ Complete |
| `models/encoder_processor_decoder.py` | Advance doy in rollout | ✅ Complete |
| `scripts/setup.py` | Track doy stats | ✅ Complete |

---

## Code Quality Notes

✅ **Strengths:**
- Clean separation of concerns (doy has own field)
- Backward compatible (doy_offset defaults to 0)
- No assumptions about tensor dimensions
- Survives future additions of constant_scalars

⚠️ **Edge cases to test:**
- What if doy_offset=0? (doy scalars all zeros) → Currently creates [0, 0] always
- Rollout with 365+ stride? (phase wraps correctly via atan2)
- Mixed batches (some with doy, some without)? → Currently requires all or none (checked in collate_batches)

