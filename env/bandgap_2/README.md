# Materials Project Bandgap Dataset Extractor

This script extracts materials with both low-fidelity (PBE) and high-fidelity (HSE06/experimental) bandgaps from the Materials Project database for machine learning training.

## Updated Features

✅ **Uses New Materials Project API** - Updated to use the modern `mp-api` package  
✅ **Automatic Authentication** - Handles API authentication automatically  
✅ **Dual Search Methods** - Two methods to find materials with paired bandgap data  
✅ **Structure Export** - Exports crystal structures in CIF format  
✅ **Comprehensive Logging** - Detailed progress tracking and error handling  

## Installation

### Option 1: Automatic Installation
```bash
python install_dependencies.py
```

### Option 2: Manual Installation
```bash
pip install -r requirements.txt
```

### Option 3: Individual Packages
```bash
pip install mp-api>=0.41.0
pip install pymatgen>=2023.1.1
pip install pandas>=1.5.0
pip install numpy>=1.21.0
```

## Usage

```bash
python extract_mp_data.py
```

The script will:
1. Connect to Materials Project using the new API
2. Search for materials with PBE bandgaps > 0.01 eV
3. Filter for materials that also have HSE or experimental bandgap data
4. Export results to both CSV and JSON formats

## Output Files

- **`bandgap_dataset.csv`** - Tabular data with material properties
- **`bandgap_dataset.json`** - Structured data with full crystal structures

## Key Improvements Over Old Version

| Feature | Old API | New API |
|---------|---------|---------|
| Authentication | `X-API-KEY` header | `Authorization: Bearer` |
| Endpoints | `/materials/` | `/materials/core/` |
| Pagination | Manual chunks | Automatic handling |
| Rate Limiting | Basic delays | Built-in throttling |
| Structure Export | JSON strings | Proper CIF format |
| Error Handling | Basic try/catch | Comprehensive logging |

## Expected Output

The script typically finds:
- **~1000-5000 materials** with PBE bandgaps
- **~100-500 materials** with paired HSE/experimental data
- **Bandgap range**: 0.01 - 10+ eV
- **Common materials**: Oxides, semiconductors, perovskites

## Troubleshooting

### API Key Issues
- Ensure your API key is valid: `tQ53EaqRe8UndenrzdDrDcg3vZypqn0d`
- Check API key permissions on Materials Project website

### Installation Issues
```bash
# If pymatgen fails to install
pip install --upgrade pip setuptools wheel
pip install pymatgen

# If mp-api fails
pip install --upgrade mp-api
```

### Memory Issues
- Reduce `max_materials` parameter in the script
- Process data in smaller chunks

## Script Parameters

You can modify these parameters in `main()`:
- `max_materials=5000` - Maximum materials to process
- `chunk_size=1000` - API request chunk size
- Rate limiting delays can be adjusted in the loops

## Data Structure

Each material entry contains:
- `material_id` - Materials Project ID (e.g., "mp-1234")
- `formula` - Chemical formula (e.g., "Li2O")
- `pbe_bandgap` - PBE calculated bandgap (eV)
- `hse_bandgap` - HSE06 calculated bandgap (eV, if available)
- `experimental_bandgap` - Experimental bandgap (eV, if available)
- `structure` - Crystal structure data
- `structure_cif` - Structure in CIF format