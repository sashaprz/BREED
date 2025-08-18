#!/usr/bin/env python3
"""
Test the real ionic conductivity ML model with existing CIF files
"""

import sys
import os
sys.path.append('genetic_algo')
from corrected_predictor import get_corrected_predictor

# Test with an existing CIF file from the dataset
predictor = get_corrected_predictor()
print('Testing real ionic conductivity ML model with existing CIF file...')

# Use one of the existing CIF files from the dataset
cif_dir = r'C:\Users\Sasha\repos\RL-electrolyte-design\env\property_predictions\CIF_OBELiX\cifs'
if os.path.exists(cif_dir):
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    if cif_files:
        # Test with the first few CIF files
        for i, cif_file in enumerate(cif_files[:3]):
            test_cif = os.path.join(cif_dir, cif_file)
            print(f'\n=== Testing with existing CIF {i+1}: {cif_file} ===')
            results = predictor.predict_single_cif(test_cif, verbose=True)
            print(f'Results summary:')
            print(f'  Ionic Conductivity: {results.get("ionic_conductivity", "N/A")} S/cm')
            print(f'  Bandgap: {results.get("bandgap", "N/A")} eV')
            print(f'  Bulk Modulus: {results.get("bulk_modulus", "N/A")} GPa')
            print(f'  SEI Score: {results.get("sei_score", "N/A")}')
            print(f'  CEI Score: {results.get("cei_score", "N/A")}')
            print(f'  Status: {results.get("prediction_status", {})}')
    else:
        print('No CIF files found in dataset')
else:
    print('Dataset directory not found')