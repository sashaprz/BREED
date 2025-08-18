# OBELiX Dataset Preparation for Enhanced CGCNN

This directory contains scripts and tools for preparing the OBELiX dataset for ionic conductivity prediction using the Enhanced CGCNN model.

## Overview

The OBELiX (Oxide-Based Electrolyte Library) dataset contains ionic conductivity measurements for solid-state electrolytes. This preparation script processes the dataset to create training data compatible with the Enhanced CGCNN architecture.

## Files

- [`prepare_obelix_data.py`](prepare_obelix_data.py) - Main data preparation script
- [`obelix_config.json`](obelix_config.json) - Configuration file with default settings
- [`example_prepare_obelix.py`](example_prepare_obelix.py) - Usage examples and demonstrations
- [`README_OBELiX_Preparation.md`](README_OBELiX_Preparation.md) - This documentation

## Features

### Data Processing
- ✅ **Excel Loading**: Reads OBELiX Excel files with ionic conductivity data
- ✅ **CIF Matching**: Matches CIF crystal structures to conductivity values
- ✅ **Data Filtering**: Removes placeholder values (conductivity ≤ 1e-12)
- ✅ **Multiple Matching Strategies**: ID-based, composition-based, or hybrid matching
- ✅ **Crystal Graph Generation**: Creates graph representations using CDVAE utilities

### Output Formats
- ✅ **Enhanced CGCNN Format**: Pickle files with complete data structures
- ✅ **Traditional CGCNN Format**: CIF files + id_prop.csv
- ✅ **CSV Export**: Tabular data for analysis
- ✅ **JSON Export**: Human-readable structured data

### Quality Control
- ✅ **Data Validation**: Checks for valid CIF files and conductivity values
- ✅ **Composition Matching**: Uses pymatgen for accurate composition comparison
- ✅ **Space Group Analysis**: Incorporates crystallographic space group information
- ✅ **Comprehensive Logging**: Detailed logs of processing steps and statistics

## Quick Start

### 1. Basic Usage

```bash
python prepare_obelix_data.py \
    --obelix_excel data/OBELiX_data.xlsx \
    --cif_folder data/cifs \
    --output_dir data/prepared_dataset
```

### 2. With Configuration File

```bash
python prepare_obelix_data.py \
    --obelix_excel data/OBELiX_data.xlsx \
    --cif_folder data/cifs \
    --config_file obelix_config.json
```

### 3. Python API Usage

```python
from prepare_obelix_data import OBELiXDataPreparator

# Configure preparation
config = {
    'output_dir': 'data/my_dataset',
    'conductivity_threshold': 1e-12,
    'matching_strategy': 'composition_spacegroup'
}

# Initialize and run
preparator = OBELiXDataPreparator(config)
preparator.load_obelix_excel('data/OBELiX_data.xlsx')
preparator.parse_cif_files('data/cifs')
preparator.match_cifs_to_obelix()
preparator.create_enhanced_cgcnn_dataset()
preparator.save_dataset('data/my_dataset/dataset.pkl')
```

## Configuration Options

### Core Settings
- `conductivity_threshold`: Minimum conductivity value (default: 1e-12 S/cm)
- `conductivity_column`: Name of conductivity column in Excel file
- `matching_strategy`: How to match CIFs to OBELiX entries
  - `'id'`: Direct ID matching
  - `'composition_spacegroup'`: Match by composition and space group
  - `'composition_only'`: Match by composition only

### Matching Strategies

#### 1. ID Matching (`'id'`)
- **Best for**: Datasets where CIF filenames match OBELiX IDs exactly
- **Pros**: Fast, unambiguous
- **Cons**: Requires exact filename matches

#### 2. Composition + Space Group (`'composition_spacegroup'`)
- **Best for**: Most cases, provides good accuracy
- **Pros**: Robust matching using crystallographic properties
- **Cons**: Slightly slower, may miss some matches

#### 3. Composition Only (`'composition_only'`)
- **Best for**: When space group information is unreliable
- **Pros**: Maximum matching coverage
- **Cons**: May create ambiguous matches

## Command Line Arguments

### Required Arguments
- `--obelix_excel`: Path to OBELiX Excel file
- `--cif_folder`: Path to folder containing CIF files

### Optional Arguments
- `--output_dir`: Output directory (default: `data/obelix_prepared`)
- `--output_format`: Output format - `pickle`, `csv`, or `json` (default: `pickle`)
- `--matching_strategy`: Matching strategy (default: `composition_spacegroup`)
- `--conductivity_threshold`: Minimum conductivity threshold (default: 1e-12)
- `--conductivity_column`: Name of conductivity column (default: `Ionic conductivity (S cm-1)`)
- `--save_cgcnn_format`: Also save in traditional CGCNN format
- `--config_file`: Path to JSON configuration file

## Data Flow

```
OBELiX Excel File + CIF Files
           ↓
    Load and Validate Data
           ↓
    Parse CIF Structures
           ↓
    Match CIFs to Conductivity Values
           ↓
    Filter Low-Quality Data
           ↓
    Generate Crystal Graphs
           ↓
    Save in Multiple Formats
```

## Output Structure

### Enhanced CGCNN Format (Pickle)
```python
[
    {
        'material_id': 'sample_001',
        'cif': 'CIF file content...',
        'ionic_conductivity': 1.5e-6,
        'composition': 'Li2O',
        'spacegroup': 225,
        'graph_arrays': [...],  # Crystal graph data
        'structure_data': {...},
        'obelix_data': {...}    # Original OBELiX entry
    },
    ...
]
```

### Traditional CGCNN Format
```
output_dir/
├── cgcnn_format/
│   ├── cifs/
│   │   ├── sample_001.cif
│   │   ├── sample_002.cif
│   │   ├── ...
│   │   ├── id_prop.csv
│   │   └── atom_init.json
│   └── ...
```

### CSV Format
```csv
material_id,ionic_conductivity,composition,spacegroup,cif_path
sample_001,1.5e-6,Li2O,225,/path/to/sample_001.cif
sample_002,3.2e-7,LiCoO2,166,/path/to/sample_002.cif
...
```

## Data Quality Filters

### 1. Conductivity Filtering
- Removes entries with conductivity ≤ threshold (default: 1e-12 S/cm)
- Filters out placeholder and invalid values
- Converts to numeric format with error handling

### 2. CIF Validation
- Checks CIF file format and readability
- Validates crystal structure using pymatgen
- Extracts composition and space group information

### 3. Matching Quality
- Uses composition tolerance for fuzzy matching
- Validates space group consistency
- Logs unmatched entries for review

## Integration with Enhanced CGCNN

The prepared dataset is directly compatible with [`enhanced_cgcnn_ionic_conductivity.py`](enhanced_cgcnn_ionic_conductivity.py):

```python
# In enhanced_cgcnn_ionic_conductivity.py
config = TrainingConfig()
config.data_path = "data/obelix_prepared/dataset.pkl"

# Load prepared dataset
dataset = IonicConductivityDataset(config.data_path, config)
```

## Troubleshooting

### Common Issues

#### 1. Excel File Not Found
```
FileNotFoundError: OBELiX Excel file not found
```
**Solution**: Check the path to your OBELiX Excel file and ensure it exists.

#### 2. CIF Parsing Errors
```
Warning: Failed to parse sample_001.cif: Invalid CIF format
```
**Solution**: Check CIF file format. Some files may be corrupted or in non-standard format.

#### 3. No Matches Found
```
Matched 0 CIF-OBELiX pairs
```
**Solutions**:
- Try different matching strategies
- Check that CIF filenames or compositions match OBELiX entries
- Verify column names in Excel file

#### 4. Low Match Rate
```
Match Rate: 15.2%
```
**Solutions**:
- Use `composition_spacegroup` strategy for better matching
- Check data quality in both CIF files and Excel file
- Consider adjusting composition tolerance

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use verbose command line:

```bash
python prepare_obelix_data.py --obelix_excel data.xlsx --cif_folder cifs/ --verbose
```

## Performance Optimization

### For Large Datasets
- Use `composition_spacegroup` matching for best balance of speed and accuracy
- Process in batches if memory is limited
- Consider parallel processing for CIF parsing

### Memory Usage
- The script loads all data into memory for processing
- For very large datasets (>10,000 entries), consider streaming processing
- Monitor memory usage during crystal graph generation

## Dependencies

### Required
- `pandas` - Excel file reading and data manipulation
- `numpy` - Numerical computations
- `pymatgen` - Crystal structure analysis
- `pathlib` - File path handling

### Optional
- `cdvae` - Enhanced crystal graph generation (recommended)
- `torch` - For tensor operations (if using CDVAE)
- `openpyxl` - Excel file support

### Installation
```bash
pip install pandas numpy pymatgen openpyxl
```

For enhanced features:
```bash
# Install CDVAE dependencies
cd generator/CDVAE
pip install -e .
```

## Examples

See [`example_prepare_obelix.py`](example_prepare_obelix.py) for detailed usage examples including:
- Basic data preparation
- Advanced configuration options
- Different matching strategies
- Quality control and validation
- Integration with Enhanced CGCNN

## Contributing

When modifying the preparation script:

1. **Test with sample data** before processing full datasets
2. **Validate output format** compatibility with Enhanced CGCNN
3. **Update documentation** for new features or configuration options
4. **Add logging** for new processing steps
5. **Handle edge cases** gracefully with informative error messages

## License

This script is part of the Enhanced CGCNN project for ionic conductivity prediction. See the main project README for license information.