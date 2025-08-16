# CDVAE Retraining Guide - Compatibility Fixed

This guide will help you retrain your CDVAE model with updated dependencies and configurations to avoid the compatibility issues we encountered.

## Issues Fixed

1. **PyTorch Lightning Version Mismatch**: Updated from v1.3.8 to v2.5.3
2. **OmegaConf Interpolation Errors**: Fixed `${now:%Y-%m-%d}` and `${oc.env:PROJECT_ROOT}` issues
3. **Missing AttributeDict Attributes**: Added `latent_dim`, `encoder`, etc. to model configuration
4. **GPU Configuration**: Updated `gpus` to `accelerator` and `devices` for PyTorch Lightning 2.x

## Prerequisites

### 1. Environment Setup

```bash
# Create a new conda environment for retraining
conda create -n cdvae_retrain python=3.9
conda activate cdvae_retrain

# Install updated requirements
pip install -r requirements_updated.txt

# Install CDVAE in development mode
cd generator/CDVAE
pip install -e .
```

### 2. GPU Setup (Recommended for External GPU)

For external GPU training, ensure:
- CUDA 11.8 or 12.1 compatible with PyTorch 2.x
- At least 8GB VRAM (16GB+ recommended)
- Proper CUDA drivers installed

```bash
# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## Configuration Changes

### 1. Use Fixed Configuration File

Replace the original `hparams.yaml` with `hparams_fixed.yaml`:

```bash
cd generator/CDVAE/cdvae/prop_models/mp20/
cp hparams_fixed.yaml hparams.yaml
```

**Key changes in hparams_fixed.yaml:**
- Fixed `${now:%Y-%m-%d}` → `"2024-01-01"`
- Updated `gpus: 1` → `accelerator: gpu, devices: 1`
- Added missing model attributes: `latent_dim: 256`, `max_atoms: 20`, etc.
- Added PyTorch Lightning 2.x compatibility settings

### 2. Update Data Path

Edit the `root_path` in `hparams_fixed.yaml`:
```yaml
data:
  root_path: /path/to/your/mp20/data  # Update this to your actual data path
```

## Training Commands

### Option 1: Local Training (if you have a good GPU)

```bash
cd generator/CDVAE
python cdvae/run.py data=mp20 model=vae
```

### Option 2: External GPU Training

Transfer these files to your external GPU machine:
- `generator/CDVAE/` (entire directory)
- `requirements_updated.txt`
- This guide

Then run:
```bash
# On external GPU machine
conda create -n cdvae_retrain python=3.9
conda activate cdvae_retrain
pip install -r requirements_updated.txt
cd generator/CDVAE
pip install -e .

# Start training
python cdvae/run.py data=mp20 model=vae
```

## Monitoring Training

### 1. Weights & Biases (W&B)
The training will automatically log to W&B. Check your dashboard at https://wandb.ai

### 2. Local Monitoring
```bash
# Check GPU usage
nvidia-smi

# Monitor training logs
tail -f logs/train.log  # Adjust path as needed
```

## Expected Training Time

- **Local GPU (RTX 3080/4080)**: 12-24 hours
- **External GPU (A100/V100)**: 4-8 hours
- **CPU Only**: Not recommended (days/weeks)

## Post-Training

### 1. Checkpoint Location
The new checkpoint will be saved in:
```
generator/CDVAE/outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/
```

### 2. Update Genetic Algorithm
Once training is complete, update the path in `genetic_algo_true_cdvae.py`:
```python
def __init__(self, model_path="path/to/new/checkpoint/directory"):
```

### 3. Test the New Model
```bash
cd genetic_algo
python genetic_algo_true_cdvae.py
```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   - Reduce batch size in `hparams_fixed.yaml`: `batch_size: {train: 128, val: 128, test: 128}`
   - Reduce `max_atoms` from 20 to 16

2. **Data Path Issues**
   - Ensure MP20 data is downloaded and path is correct
   - Check that `train.csv`, `val.csv`, `test.csv` exist

3. **Import Errors**
   - Ensure all dependencies from `requirements_updated.txt` are installed
   - Run `pip install -e .` in the CDVAE directory

4. **W&B Login Issues**
   ```bash
   wandb login  # Enter your API key
   ```

## Verification

After retraining, the new model should:
- Load without AttributeDict errors
- Generate crystal structures properly
- Work with the genetic algorithm

Test with:
```python
from genetic_algo.fix_cdvae_compatibility import load_cdvae_with_compatibility_fix
model = load_cdvae_with_compatibility_fix("path/to/new/checkpoint")
print("✅ Model loaded successfully!" if model else "❌ Model loading failed")
```

## Files Created/Modified

- ✅ `requirements_updated.txt` - Updated dependencies
- ✅ `hparams_fixed.yaml` - Fixed configuration file
- ✅ `RETRAINING_GUIDE.md` - This guide
- 🔄 Original compatibility issues resolved

Good luck with the retraining! The new model should work seamlessly with your genetic algorithm.