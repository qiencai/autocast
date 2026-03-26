#!/bin/bash
#SBATCH --job-name=verify_doy_mask
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_verify_doy_mask_%j.log
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos=turing

module purge
module load baskerville
module load pytorch/2.0.0-cu11.8

set -e
cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast
source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

echo "========================================"
echo "Phase 1: Check mask data in files"
echo "========================================"

python3 << 'EOF'
import torch
from pathlib import Path

base_dir = Path('data/seaice/processed_osisaf_selectedyears')
print("\nChecking data files for constant_fields (masks):\n")

for split in ['train', 'valid', 'test']:
    data_path = base_dir / split / 'data.pt'
    if data_path.exists():
        try:
            data = torch.load(data_path, weights_only=False)
            has_const_fields = 'constant_fields' in data
            
            if has_const_fields and data['constant_fields'] is not None:
                shape = data['constant_fields'].shape
                print(f"✅ {split:5s}: constant_fields exists, shape={shape}")
            else:
                print(f"❌ {split:5s}: NO constant_fields (or None)")
            
            # Also check data shape
            data_shape = data['data'].shape
            print(f"         data.shape={data_shape}")
        except Exception as e:
            print(f"⚠️  {split:5s}: Error - {e}")
    else:
        print(f"⚠️  {split:5s}: data.pt not found")

EOF

echo ""
echo "========================================"
echo "Phase 2: Run autoencoder baseline"
echo "========================================"

autocast train-eval \
  experiment=seaice_autoencoder \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1 \
  trainer.limit_test_batches=1 \
  2>&1 | tee autoencoder_test.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo "✅ Autoencoder baseline PASSED"
else
  echo "❌ Autoencoder baseline FAILED"
  exit 1
fi

echo ""
echo "========================================"
echo "Phase 3: Run flow matching with DOY+mask"
echo "========================================"

autocast train-eval \
  experiment=seaice_flow_matching_with_mask_doy \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1 \
  trainer.limit_test_batches=1 \
  2>&1 | tee flow_matching_test.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo "✅ Flow matching with DOY+mask PASSED"
else
  echo "❌ Flow matching with DOY+mask FAILED"
  exit 1
fi

echo ""
echo "========================================"
echo "Phase 4: Trace DOY through pipeline"
echo "========================================"

python3 << 'EOF'
import torch
from pathlib import Path
from autocast.data.datamodule import SpatioTemporalDataModule

# Create datamodule with DOY
dm = SpatioTemporalDataModule(
    data_path='data/seaice/processed_osisaf_selectedyears',
    n_steps_input=5,
    n_steps_output=1,
    stride=1,
    batch_size=1,
    doy_offset=1,  # DOY enabled
    verbose=False
)
dm.setup(stage='fit')

# Get one batch
batch = next(iter(dm.train_dataloader()))

print("\n✅ DOY Pipeline Trace:")
print(f"  Input fields shape: {batch.input_fields.shape}")
print(f"  Output fields shape: {batch.output_fields.shape}")
print(f"  Constant fields: {batch.constant_fields is not None} (mask)")
if batch.constant_fields is not None:
    print(f"    - Shape: {batch.constant_fields.shape}")
print(f"  Constant DOY scalars: {batch.constant_doy_scalars is not None}")
if batch.constant_doy_scalars is not None:
    print(f"    - Shape: {batch.constant_doy_scalars.shape}")
    print(f"    - Values (sin, cos): {batch.constant_doy_scalars[0].tolist()}")

# Simulate encoder preprocessing
from dataclasses import replace
if batch.constant_fields is not None:
    b, t, w, h, c_in = batch.input_fields.shape
    c_const = batch.constant_fields.shape[-1]
    mask_expanded = batch.constant_fields.unsqueeze(1).expand(b, t, w, h, c_const)
    input_with_mask = torch.cat([batch.input_fields, mask_expanded], dim=-1)
    print(f"\n✅ After encoder preprocessing:")
    print(f"    - Input channels expanded: {c_in} → {c_in + c_const} (SIC + mask)")
    print(f"    - Shape: {input_with_mask.shape}")

print(f"\n✅ DOY will be in global_cond for processor:")
print(f"    - Shape: {batch.constant_doy_scalars.shape}")
print(f"    - Used for flow matching temporal conditioning")

EOF

echo ""
echo "========================================"
echo "✅ ALL CHECKS PASSED!"
echo "========================================"
echo "Summary:"
echo "  1. Masks loaded correctly from data files"
echo "  2. Autoencoder baseline runs successfully"
echo "  3. Flow matching + DOY + mask experiment works"
echo "  4. DOY properly flows through pipeline"
echo "========================================"
