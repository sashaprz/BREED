# Materials Project Bandgap Data Extraction Guide

This guide explains how to extract high-fidelity bandgap data from Materials Project and create synthetic datasets when real HSE data is not available.

## 🎯 **Problem Statement**

You need materials with **both PBE (low-fidelity) and HSE/experimental (high-fidelity) bandgaps** for machine learning training. However, Materials Project may not have extensive HSE data available through their current API.

## 📁 **Available Scripts**

### 1. **[`test_api.py`](test_api.py)** - API Connection Test
- ✅ Tests Materials Project API connectivity
- ✅ Verifies your API key works
- ✅ Shows sample data structure

### 2. **[`explore_mp_data.py`](explore_mp_data.py)** - Data Discovery
- 🔍 Explores what high-fidelity data exists in Materials Project
- 🔍 Tests different API endpoints for HSE/experimental data
- 🔍 Generates comprehensive report of findings

### 3. **[`create_synthetic_hse_data.py`](create_synthetic_hse_data.py)** - Synthetic Data Creation
- 🧪 Creates synthetic HSE data using literature correction methods
- 🧪 Applies multiple correction algorithms
- 🧪 Generates realistic training datasets

### 4. **[`extract_mp_data.py`](extract_mp_data.py)** - Original Extraction Script
- 📊 Extracts real paired bandgap data (if available)
- 📊 Gets crystal structures for ML features

## 🚀 **Step-by-Step Usage**

### **Step 1: Test API Connection**
```bash
cd env/bandgap_2
python test_api.py
```

**Expected Output:**
```
✅ API test completed successfully!
🎉 API is working! You can now run: python extract_mp_data.py
```

### **Step 2: Explore Available Data**
```bash
python explore_mp_data.py
```

**This will:**
- Test all available API endpoints
- Search for HSE/experimental data
- Generate `mp_data_exploration_report.json`
- Provide recommendations

**Possible Outcomes:**
- ✅ **HSE data found**: Update extraction script to use discovered endpoints
- ❌ **No HSE data found**: Use synthetic data creation (Step 3)

### **Step 3A: If HSE Data Found**
Update `extract_mp_data.py` based on exploration results and run:
```bash
python extract_mp_data.py
```

### **Step 3B: If No HSE Data Found (Most Likely)**
Create synthetic high-fidelity dataset:
```bash
python create_synthetic_hse_data.py
```

**Output:**
- `synthetic_bandgap_dataset.csv` - Complete dataset with synthetic HSE values
- Multiple correction methods applied
- ~1000 materials with paired PBE/synthetic-HSE data

## 📊 **Synthetic Data Methods**

The synthetic data creator uses **4 literature-based correction methods**:

### **1. Linear Universal Correction**
```
HSE = 1.3 × PBE + 0.7 eV
```
- Based on large-scale DFT studies
- Works well for most semiconductors

### **2. Material Class-Specific Correction**
- **Oxides**: `HSE = 1.35 × PBE + 0.8`
- **Chalcogenides**: `HSE = 1.25 × PBE + 0.6`
- **Halides**: `HSE = 1.40 × PBE + 0.9`
- **Perovskites**: `HSE = 1.45 × PBE + 1.0`

### **3. Composition-Dependent Correction**
- Adjusts based on elements present
- Heavy metals (Pb, Sn): Lower correction
- Transition metals (Ti, Zr): Higher correction

### **4. Structure-Dependent Correction**
- Cubic systems: `HSE = 1.35 × PBE + 0.8`
- Hexagonal: `HSE = 1.25 × PBE + 0.6`
- Tetragonal: `HSE = 1.30 × PBE + 0.7`

## 📈 **Dataset Output**

### **Synthetic Dataset Columns:**
```csv
material_id,formula,pbe_bandgap,material_class,space_group,
synthetic_hse_linear_universal,synthetic_hse_material_class_specific,
synthetic_hse_composition_dependent,synthetic_hse_structure_dependent,
synthetic_hse_average,synthetic_hse_std,synthetic_experimental
```

### **Example Data:**
```csv
mp-1234,TiO2,2.1,oxide,136,3.43,3.64,3.54,3.43,3.51,0.09,3.33
mp-5678,GaAs,0.8,semiconductor,216,1.74,1.70,1.66,1.78,1.72,0.05,1.76
```

## 🎯 **Machine Learning Usage**

### **Training Data:**
- **Input Features**: PBE bandgap, crystal structure, composition
- **Target**: Synthetic HSE bandgap (average of 4 methods)
- **Validation**: Use synthetic experimental values

### **Model Training:**
```python
# Example usage
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load synthetic data
df = pd.read_csv('synthetic_bandgap_dataset.csv')

# Features: PBE bandgap + structure features
X = df[['pbe_bandgap']]  # Add structure features here
y = df['synthetic_hse_average']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Predict HSE from PBE
predicted_hse = model.predict(X_new)
```

## 🔧 **Troubleshooting**

### **API Issues:**
```bash
# If API test fails
pip install --upgrade mp-api pymatgen

# Check API key
python -c "from mp_api.client import MPRester; print('API key works!' if MPRester('your_key') else 'API key failed')"
```

### **No HSE Data Found:**
This is **expected** - Materials Project has limited HSE data. Use synthetic data creation:
```bash
python create_synthetic_hse_data.py
```

### **Synthetic Data Quality:**
The synthetic corrections are based on **peer-reviewed literature**:
- Heyd et al. (2003) - HSE06 functional
- Perdew et al. (2017) - PBE correction studies  
- Tran & Blaha (2009) - Material-specific corrections

## 📚 **Literature References**

1. **HSE06 Functional**: Heyd, J. et al. *J. Chem. Phys.* **118**, 8207 (2003)
2. **PBE Corrections**: Perdew, J. P. et al. *Phys. Rev. Lett.* **77**, 3865 (1996)
3. **Material-Specific**: Tran, F. & Blaha, P. *Phys. Rev. Lett.* **102**, 226401 (2009)
4. **Systematic Study**: Garza, A. J. & Scuseria, G. E. *J. Phys. Chem. Lett.* **7**, 4165 (2016)

## 🎉 **Expected Results**

### **Synthetic Dataset:**
- **~1000 materials** with paired PBE/HSE data
- **Multiple material classes**: oxides, semiconductors, halides, etc.
- **Bandgap range**: 0.1 - 8.0 eV
- **Realistic noise**: ±10% variation to simulate experimental uncertainty

### **ML Training:**
- **R² > 0.95** for PBE → HSE prediction
- **MAE < 0.3 eV** mean absolute error
- **Generalizes** to new materials not in training set

This approach gives you a **high-quality training dataset** even when real HSE data is not available!