"""
Debug predictor that handles CIF files properly and provides fallback values
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings
import random

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def predict_single_cif_debug(cif_file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Debug prediction function that returns realistic random values for testing"""
    
    results = {
        "composition": extract_composition_from_cif(cif_file_path),
        "bandgap": 0.0,
        "sei_score": 0.0,
        "cei_score": 0.0,
        "ionic_conductivity": 0.0,
        "bulk_modulus": 0.0,
        "prediction_status": {
            "sei": "debug_mode",
            "cei": "debug_mode", 
            "bandgap": "debug_mode",
            "bulk_modulus": "debug_mode",
            "ionic_conductivity": "debug_mode"
        }
    }
    
    if verbose:
        print(f"DEBUG: Processing CIF: {os.path.basename(cif_file_path)}")
    
    # Generate realistic random values for testing
    try:
        # Seed based on filename for reproducible results
        filename = os.path.basename(cif_file_path)
        seed = hash(filename) % 1000000
        np.random.seed(seed)
        
        # Generate realistic property values
        results["ionic_conductivity"] = 10 ** np.random.uniform(-8, -2)  # 1e-8 to 1e-2 S/cm
        results["bandgap"] = np.random.uniform(0.5, 6.0)  # 0.5 to 6.0 eV
        results["sei_score"] = np.random.uniform(0.1, 1.0)  # 0.1 to 1.0
        results["cei_score"] = np.random.uniform(0.1, 1.0)  # 0.1 to 1.0
        results["bulk_modulus"] = np.random.uniform(20, 200)  # 20 to 200 GPa
        
        # Mark all as successful
        for key in results["prediction_status"]:
            results["prediction_status"][key] = "debug_success"
        
        if verbose:
            print(f"  DEBUG - Ionic Conductivity: {results['ionic_conductivity']:.2e} S/cm")
            print(f"  DEBUG - Bandgap: {results['bandgap']:.3f} eV")
            print(f"  DEBUG - SEI Score: {results['sei_score']:.3f}")
            print(f"  DEBUG - CEI Score: {results['cei_score']:.3f}")
            print(f"  DEBUG - Bulk Modulus: {results['bulk_modulus']:.1f} GPa")
            
    except Exception as e:
        if verbose:
            print(f"  DEBUG: Error generating values: {e}")
    
    return results

def extract_composition_from_cif(cif_file_path: str) -> str:
    """Extract composition from CIF file"""
    try:
        with open(cif_file_path, 'r') as f:
            lines = f.readlines()
        
        # Look for data_ line which often contains composition info
        for line in lines:
            if line.startswith('data_'):
                composition = line.replace('data_', '').strip()
                if composition:
                    return composition
        
        # Fallback: use filename
        return os.path.splitext(os.path.basename(cif_file_path))[0]
    except:
        return os.path.splitext(os.path.basename(cif_file_path))[0]

# Test function
if __name__ == "__main__":
    # Test with a dummy CIF path
    test_path = "test_Li2O4.cif"
    result = predict_single_cif_debug(test_path, verbose=True)
    print(f"\nTest result: {result}")