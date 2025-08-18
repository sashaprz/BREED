#!/usr/bin/env python3
"""
Test different transformations for ionic conductivity to find the correct one
"""

import sys
import os
import numpy as np
sys.path.append('genetic_algo')
from corrected_predictor import run_direct_cgcnn_prediction

# Test the direct CGCNN prediction function with different transforms
print('Testing ionic conductivity transformations...')

# Use one of the generated CIF files from our previous test
cif_dir = 'final_cdvae_ga_test_results/cifs'
if os.path.exists(cif_dir):
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    if cif_files:
        test_cif = os.path.join(cif_dir, cif_files[0])
        checkpoint_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\checkpoint.pth.tar"
        
        print(f'Testing with: {os.path.basename(test_cif)}')
        
        results = run_direct_cgcnn_prediction(checkpoint_path, test_cif)
        
        if results:
            raw_value = results['predictions'][0]
            print(f'Raw model output: {raw_value}')
            
            # Test different transformations
            print('\nTesting transformations:')
            print(f'1. Raw value: {raw_value:.6f}')
            print(f'2. Absolute value: {abs(raw_value):.6f}')
            
            # Test exponential (natural log inverse)
            if raw_value > -50:  # Avoid overflow
                exp_value = np.exp(raw_value)
                print(f'3. exp(raw_value): {exp_value:.2e} S/cm')
            
            # Test 10^x (log10 inverse)
            if raw_value > -50:  # Avoid overflow
                log10_value = 10 ** raw_value
                print(f'4. 10^(raw_value): {log10_value:.2e} S/cm')
            
            # Compare with known good values from existing CIF files
            print('\nFor comparison, existing CIF files gave:')
            print('  08e_generated.cif: 4.59e-04 S/cm')
            print('  09w_generated.cif: 3.73e-04 S/cm')
            print('  0ew_generated.cif: 9.00e-05 S/cm (estimated)')
            
        else:
            print('FAILED! Direct prediction returned None')
    else:
        print('No CIF files found in test results')
else:
    print('Test results directory not found')