#!/usr/bin/env python3
"""
Test the corrected predictor with ionic conductivity fallback
"""

import sys
import os
sys.path.append('genetic_algo')
from corrected_predictor import get_corrected_predictor

# Test with a sample CIF file
predictor = get_corrected_predictor()
print('Testing corrected predictor with ionic conductivity fallback...')

# Use one of the generated CIF files from our previous test
cif_dir = 'final_cdvae_ga_test_results/cifs'
if os.path.exists(cif_dir):
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    if cif_files:
        test_cif = os.path.join(cif_dir, cif_files[0])
        print(f'Testing with: {test_cif}')
        results = predictor.predict_single_cif(test_cif, verbose=True)
        print('\nResults summary:')
        print(f'  Ionic Conductivity: {results.get("ionic_conductivity", "N/A")} S/cm')
        print(f'  Bandgap: {results.get("bandgap", "N/A")} eV')
        print(f'  Bulk Modulus: {results.get("bulk_modulus", "N/A")} GPa')
        print(f'  SEI Score: {results.get("sei_score", "N/A")}')
        print(f'  CEI Score: {results.get("cei_score", "N/A")}')
        print(f'  Status: {results.get("prediction_status", {})}')
    else:
        print('No CIF files found in test results')
else:
    print('Test results directory not found')