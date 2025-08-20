# CGCNN Bulk Modulus Fine-Tuning Guide

This guide provides a complete solution for fine-tuning the CGCNN bulk modulus model on high bulk modulus inorganic crystals from the OBELiX dataset with Materials Project bulk modulus values.

## 🎯 Problem Statement

The original CGCNN bulk modulus model was trained on the Materials Project database, which includes many soft organic/molecular crystals. This creates a systematic bias toward lower bulk modulus values (~1 GPa), making it unsuitable for predicting bulk modulus of inorganic solid electrolytes (which should be 50-200 GPa).

## 🔧 Solution Overview

Our fine-tuning approach:

1. **Extract bulk modulus values** from Materials Project for OBELiX crystal structures
2. **Filter for high bulk modulus materials** (>20 GPa) suitable for inorganic crystals
3. **Fine-tune the pre-trained CGCNN model** using transfer learning with lower learning rates
4. **Validate performance improvements** on high bulk modulus materials
5. **Generate comprehensive reports** comparing original vs fine-tuned models

## 📁 Files Created

### Core Scripts

- **`extract_mp_bulk_modulus.py`** - Extracts bulk modulus data from Materials Project API
- **`finetune_cgcnn_bulk_modulus.py`** - Fine-tunes the CGCNN model on high bulk modulus materials
- **`validate_bulk_modulus_model.py`** - Validates and compares model performance
- **`run_bulk_modulus_finetuning_pipeline.py`** - Complete automated pipeline

### Configuration Files

- **`bulk_modulus_finetune_config.json`** - Fine-tuning hyperparameters (auto-generated)

## 🚀 Quick Start

### Prerequisites

1. **Materials Project API Key**: `tQ53EaqRe8UndenrzdDrDcg3vZypqn0d` (already configured)
2. **Required Python packages**:
   ```bash
   pip install mp-api pymatgen torch scikit-learn matplotlib seaborn pandas numpy tqdm
   ```
3. **OBELiX dataset**: Must be available at `env/property_predictions/CIF_OBELiX/cifs/`
4. **Original model**: Must be available at `env/property_predictions/bulk-moduli.pth.tar`

### Option 1: Run Complete Pipeline (Recommended)

```bash
python run_bulk_modulus_finetuning_pipeline.py
```

This will automatically:
- Extract bulk modulus data from Materials Project
- Fine-tune the model
- Validate performance
- Generate comprehensive reports

### Option 2: Run Individual Steps

#### Step 1: Extract Bulk Modulus Data
```bash
python extract_mp_bulk_modulus.py
```
**Output**: `bulk_modulus_data/obelix_bulk_modulus_high.csv`

#### Step 2: Fine-tune Model
```bash
python finetune_cgcnn_bulk_modulus.py
```
**Output**: `outputs/bulk_modulus_finetuned/best_bulk_modulus_model.pth.tar`

#### Step 3: Validate Model
```bash
python validate_bulk_modulus_model.py
```
**Output**: `outputs/bulk_modulus_validation/validation_summary.json`

## ⚙️ Configuration

### Fine-tuning Hyperparameters

The fine-tuning uses these optimized parameters:

```json
{
  "learning_rate": 1e-5,        // 100x smaller than original training
  "weight_decay": 1e-4,         // L2 regularization
  "batch_size": 16,             // Smaller for stability
  "num_epochs": 100,            // Sufficient for convergence
  "patience": 15,               // Early stopping
  "weighted_loss": true,        // Emphasize high bulk modulus materials
  "high_bm_weight": 2.0         // 2x weight for materials >50 GPa
}
```

### Data Filtering

Materials are filtered to focus on high-quality, high bulk modulus crystals:

- **Bulk modulus range**: 20-500 GPa
- **Energy above hull**: <0.1 eV/atom (stable materials)
- **Structure match score**: <0.5 (good matches with OBELiX)

## 📊 Expected Results

### Performance Improvements

The fine-tuned model should show significant improvements on high bulk modulus materials:

- **Original Model**: ~1 GPa predictions (biased toward soft materials)
- **Fine-tuned Model**: 50-200 GPa predictions (appropriate for inorganic crystals)

### Validation Metrics

- **MAE (Mean Absolute Error)**: Should decrease by 20-50%
- **R² (Correlation)**: Should increase by 0.1-0.3
- **High BM Materials (>50 GPa)**: Largest improvements expected

## 📁 Output Structure

```
outputs/
├── bulk_modulus_finetuned/
│   ├── best_bulk_modulus_model.pth.tar    # Fine-tuned model
│   ├── training_history.json              # Training metrics
│   └── training_plots.png                 # Loss/MAE curves
├── bulk_modulus_validation/
│   ├── validation_summary.json            # Performance comparison
│   ├── model_comparison.png               # Comparison plots
│   └── test_predictions.csv               # Detailed predictions
└── bulk_modulus_pipeline/
    ├── pipeline_report.json               # Complete pipeline results
    └── PIPELINE_SUMMARY.md                # Human-readable summary

bulk_modulus_data/
├── obelix_bulk_modulus_all.csv           # All extracted data
├── obelix_bulk_modulus_high.csv          # High bulk modulus materials
├── failed_matches.csv                     # Failed structure matches
└── extraction_summary.json                # Extraction statistics
```

## 🔍 Using the Fine-tuned Model

### Replace Original Model

To use the fine-tuned model in your genetic algorithm:

1. **Backup original model**:
   ```bash
   cp env/property_predictions/bulk-moduli.pth.tar env/property_predictions/bulk-moduli_original.pth.tar
   ```

2. **Replace with fine-tuned model**:
   ```bash
   cp outputs/bulk_modulus_finetuned/best_bulk_modulus_model.pth.tar env/property_predictions/bulk-moduli.pth.tar
   ```

3. **Test with genetic algorithm**:
   ```bash
   python genetic_algo/TRUE_genetic_algo.py
   ```

### Expected Improvements in Genetic Algorithm

After fine-tuning, your genetic algorithm should show:

- **Realistic bulk modulus predictions**: 50-200 GPa for inorganic solid electrolytes
- **Better fitness calculations**: More accurate multi-objective optimization
- **Improved candidate ranking**: Better identification of promising materials

## 🛠️ Troubleshooting

### Common Issues

1. **Materials Project API Errors**:
   - Check API key is valid
   - Ensure internet connection
   - API may have rate limits

2. **CUDA/CPU Device Errors**:
   - Models automatically handle device placement
   - Check PyTorch CUDA installation if using GPU

3. **Memory Issues**:
   - Reduce batch size in config
   - Use CPU if GPU memory insufficient

4. **Few Training Samples**:
   - Relax filtering criteria (increase `max_bulk_modulus`)
   - Check OBELiX dataset completeness

### Performance Validation

If fine-tuning doesn't improve performance:

1. **Check data quality**: Ensure good structure matches
2. **Adjust hyperparameters**: Try different learning rates
3. **Increase training data**: Relax filtering criteria
4. **Extend training**: Increase epochs or reduce patience

## 📈 Technical Details

### Transfer Learning Strategy

- **Frozen layers**: None (full fine-tuning)
- **Learning rate**: 100x smaller than original training
- **Loss function**: Weighted MSE emphasizing high bulk modulus materials
- **Regularization**: L2 weight decay to prevent overfitting

### Data Augmentation

- **Structure matching**: Pymatgen StructureMatcher with tolerances
- **Composition filtering**: Focus on inorganic crystals
- **Stability filtering**: Energy above hull <0.1 eV/atom

### Model Architecture

The fine-tuned model maintains the same architecture as the original:
- **Atom features**: 64 dimensions
- **Convolution layers**: 3 layers
- **Hidden features**: 128 dimensions
- **Output**: Single bulk modulus value

## 🎯 Success Criteria

The fine-tuning is considered successful if:

1. **MAE improves by >20%** on high bulk modulus materials
2. **R² increases by >0.1** on test set
3. **Predictions are realistic** (50-200 GPa for inorganic crystals)
4. **No overfitting** (validation loss doesn't increase)

## 📞 Support

If you encounter issues:

1. Check the generated logs in each output directory
2. Review the validation summary for performance metrics
3. Examine the pipeline report for detailed results
4. Adjust hyperparameters in the config files as needed

## 🔄 Integration with Genetic Algorithm

After successful fine-tuning, your genetic algorithm will benefit from:

- **Accurate bulk modulus predictions** for inorganic solid electrolytes
- **Better multi-objective optimization** with realistic property ranges
- **Improved candidate evaluation** leading to better material discovery

The fine-tuned model seamlessly integrates with your existing [`genetic_algo/TRUE_genetic_algo.py`](genetic_algo/TRUE_genetic_algo.py) and [`genetic_algo/property_prediction_script.py`](genetic_algo/property_prediction_script.py) without requiring code changes.