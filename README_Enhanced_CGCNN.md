# Enhanced CGCNN for Ionic Conductivity Prediction

This repository contains an enhanced Crystal Graph Convolutional Neural Network (CGCNN) implementation specifically designed for ionic conductivity prediction. The implementation includes several key improvements over standard CGCNN approaches to better handle the unique challenges of ionic conductivity modeling.

## Key Features

### 1. Log-Space Training
- **Implementation**: `log_conductivity = torch.log10(conductivity + 1e-20)`
- **Purpose**: Handles the wide range of ionic conductivity values (spanning 10+ orders of magnitude)
- **Benefits**: Improved numerical stability and better learning of relative differences

### 2. Enhanced Normalization
- **Custom LogSpaceScaler**: Performs normalization in log space for better statistical properties
- **Robust Statistics**: Handles extreme values and maintains numerical stability
- **Inverse Transform**: Seamless conversion back to original conductivity units

### 3. Positive Activation Functions
- **Softplus Activations**: `F.softplus()` ensures positive outputs throughout the network
- **Physical Consistency**: Guarantees physically meaningful (positive) conductivity predictions
- **Smooth Gradients**: Better optimization compared to ReLU for positive-only outputs

### 4. Data Cleaning and Filtering
- **Placeholder Removal**: Automatically filters out conductivity values ≤ 1e-12 S/cm
- **Quality Control**: Removes unrealistic or placeholder measurements
- **Robust Dataset**: Ensures training on meaningful data points only

## Architecture Overview

### CGCNNConv Layer
```python
class CGCNNConv(MessagePassing):
    - Node transformation with linear layers
    - Edge-aware message passing
    - Positive activation functions (Softplus)
    - Residual connections for deep networks
```

### EnhancedCGCNN Model
```python
class EnhancedCGCNN(nn.Module):
    - Atom embedding layer
    - Multiple CGCNN convolution layers
    - Layer normalization and dropout
    - Final prediction head with Softplus
```

## File Structure

```
├── enhanced_cgcnn_ionic_conductivity.py  # Main implementation
├── cgcnn_config.json                     # Configuration file
├── test_cgcnn_implementation.py          # Test suite
└── README_Enhanced_CGCNN.md             # This documentation
```

## Installation and Setup

### Prerequisites
```bash
# Activate virtual environment
source venv/bin/activate  # or ./venv/bin/activate

# Required packages (should already be installed in the environment)
- torch
- torch-geometric
- torch-scatter
- pandas
- numpy
- pymatgen
- scikit-learn
```

### Verify Installation
```bash
python test_cgcnn_implementation.py
```

## Usage

### Basic Training
```bash
python enhanced_cgcnn_ionic_conductivity.py \
    --data_path data/ionic_conductivity.csv \
    --output_dir outputs/cgcnn_run_1
```

### With Configuration File
```bash
python enhanced_cgcnn_ionic_conductivity.py \
    --config cgcnn_config.json \
    --data_path data/your_data.csv
```

### Configuration Options

The `TrainingConfig` class provides comprehensive configuration:

```python
@dataclass
class TrainingConfig:
    # Data parameters
    data_path: str = "data/ionic_conductivity.csv"
    prop_name: str = "ionic_conductivity"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Model parameters
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    cutoff: float = 8.0
    max_neighbors: int = 20
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 200
    patience: int = 20
    
    # Log-space parameters
    log_epsilon: float = 1e-20
    conductivity_threshold: float = 1e-12
```

## Data Format

### Input Data Requirements
Your CSV file should contain:
- `material_id`: Unique identifier for each crystal
- `cif`: Crystal structure in CIF format
- `ionic_conductivity`: Target conductivity values (S/cm)

### Example Data Format
```csv
material_id,cif,ionic_conductivity
mp-1234,data_mp-1234...,1.5e-3
mp-5678,data_mp-5678...,2.3e-7
```

## Model Architecture Details

### 1. Input Processing
- **Atom Embedding**: Maps atomic numbers to dense vectors
- **Graph Construction**: Uses CrystalNN for bond detection
- **Distance Calculation**: PBC-aware distance computation

### 2. Message Passing
- **Node Features**: Atomic embeddings updated through convolution
- **Edge Features**: Distance-based edge attributes
- **Aggregation**: Sum aggregation with residual connections

### 3. Output Prediction
- **Global Pooling**: Mean pooling over all atoms
- **MLP Head**: Multi-layer perceptron with dropout
- **Final Activation**: Softplus for positive outputs

## Training Process

### 1. Data Preprocessing
```python
# Automatic filtering of low-quality data
filtered_data = data[data['ionic_conductivity'] > threshold]

# Log-space transformation
log_conductivity = torch.log10(conductivity + epsilon)

# Normalization in log space
normalized = (log_values - log_mean) / log_std
```

### 2. Loss Function
- **MSE in Log Space**: `F.mse_loss(pred_log, target_log)`
- **Numerical Stability**: Training in normalized log space
- **Physical Meaning**: Relative error optimization

### 3. Evaluation Metrics
- **MAE (Log Space)**: Mean absolute error in log space
- **MAE (Original)**: Mean absolute error in original units
- **MAPE**: Mean absolute percentage error
- **R² (Log Space)**: Coefficient of determination in log space

## Advanced Features

### Early Stopping
```python
early_stopping = EarlyStopping(patience=20, min_delta=1e-4)
```

### Learning Rate Scheduling
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
```

### Model Checkpointing
- Automatic saving of best model based on validation loss
- Regular checkpoints every N epochs
- Complete state preservation for resuming training

## Testing and Validation

### Test Suite
The implementation includes comprehensive tests:

```bash
python test_cgcnn_implementation.py
```

**Test Coverage:**
- ✅ LogSpaceScaler functionality
- ✅ CGCNNConv layer operations
- ✅ EnhancedCGCNN model forward pass
- ✅ Metrics computation
- ✅ Data filtering
- ✅ Component integration

### Expected Test Output
```
============================================================
Running Enhanced CGCNN Implementation Tests
============================================================
Testing LogSpaceScaler...
  ✓ LogSpaceScaler test passed

Testing CGCNNConv layer...
  ✓ CGCNNConv test passed

Testing EnhancedCGCNN model...
  ✓ EnhancedCGCNN test passed

[... additional tests ...]

✅ ALL TESTS PASSED!
============================================================
```

## Performance Considerations

### Memory Optimization
- **Batch Processing**: Configurable batch sizes
- **Gradient Checkpointing**: Available for large models
- **Efficient Graph Operations**: Optimized message passing

### Computational Efficiency
- **GPU Support**: Automatic device detection
- **Vectorized Operations**: Efficient tensor operations
- **Sparse Graph Representation**: Memory-efficient graph storage

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   ```

2. **CUDA Out of Memory**
   ```python
   # Reduce batch size in config
   config.batch_size = 16
   ```

3. **Poor Convergence**
   ```python
   # Adjust learning rate and regularization
   config.learning_rate = 5e-4
   config.weight_decay = 1e-4
   ```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{enhanced_cgcnn_ionic_conductivity,
  title={Enhanced CGCNN for Ionic Conductivity Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/enhanced-cgcnn}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- Built upon the CDVAE framework
- Inspired by the original CGCNN paper
- Uses PyTorch Geometric for graph operations
- Crystal structure processing via pymatgen

---

**Note**: This implementation is specifically optimized for ionic conductivity prediction and includes domain-specific enhancements that may not be suitable for other materials properties without modification.