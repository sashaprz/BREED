#!/usr/bin/env python3
"""
Debug the ionic conductivity prediction to see why the real ML model isn't working
"""

import sys
import os
sys.path.append('genetic_algo')
from corrected_predictor import run_direct_cgcnn_prediction

# Test the direct CGCNN prediction function
print('Testing run_direct_cgcnn_prediction function directly...')

# Use one of the generated CIF files from our previous test
cif_dir = 'final_cdvae_ga_test_results/cifs'
if os.path.exists(cif_dir):
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    if cif_files:
        test_cif = os.path.join(cif_dir, cif_files[0])
        checkpoint_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\checkpoint.pth.tar"
        
        print(f'Testing direct prediction with: {test_cif}')
        print(f'Using checkpoint: {checkpoint_path}')
        
        results = run_direct_cgcnn_prediction(checkpoint_path, test_cif)
        
        if results:
            print(f'SUCCESS! Direct prediction results: {results}')
            ic_value = results['predictions'][0]
            print(f'Ionic Conductivity from real ML model: {ic_value:.2e} S/cm')
        else:
            print('FAILED! Direct prediction returned None')
    else:
        print('No CIF files found in test results')
else:
    print('Test results directory not found')