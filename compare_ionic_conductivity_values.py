#!/usr/bin/env python3
"""
Compare the real ML model output vs estimated ionic conductivity
"""

import sys
import os
sys.path.append('genetic_algo')
from corrected_predictor import run_direct_cgcnn_prediction, estimate_ionic_conductivity_from_composition, extract_composition_from_cif

# Test with our generated CIF file
cif_dir = 'final_cdvae_ga_test_results/cifs'
if os.path.exists(cif_dir):
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    if cif_files:
        test_cif = os.path.join(cif_dir, cif_files[0])
        checkpoint_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\checkpoint.pth.tar"
        
        print(f'Comparing ionic conductivity predictions for: {os.path.basename(test_cif)}')
        
        # Get composition
        composition = extract_composition_from_cif(test_cif)
        print(f'Composition: {composition}')
        
        # Get real ML model prediction
        results = run_direct_cgcnn_prediction(checkpoint_path, test_cif)
        if results:
            raw_ml_value = results['predictions'][0]
            abs_ml_value = abs(raw_ml_value)
            print(f'\nReal ML model:')
            print(f'  Raw output: {raw_ml_value:.6f} S/cm')
            print(f'  Absolute value: {abs_ml_value:.2e} S/cm')
        
        # Get estimated value
        estimated_value = estimate_ionic_conductivity_from_composition(composition)
        print(f'\nEstimated (composition-based):')
        print(f'  Estimated value: {estimated_value:.2e} S/cm')
        
        # Compare with known good values from existing dataset CIFs
        print(f'\nFor comparison, existing dataset CIFs gave:')
        print(f'  08e_generated.cif: 4.59e-04 S/cm (real ML model)')
        print(f'  09w_generated.cif: 3.73e-04 S/cm (real ML model)')
        
        # Analysis
        print(f'\nAnalysis:')
        if results:
            print(f'  ML model magnitude: {abs_ml_value:.2e}')
        print(f'  Estimated magnitude: {estimated_value:.2e}')
        
        # Check which is closer to typical training data values
        print(f'\nTraining data typical values:')
        print(f'  1.58e-06, 6.70e-08, 8.10e-07, 3.20e-03, 3.20e-07, 8.60e-03')
        
        if results:
            print(f'\nWhich seems more reasonable?')
            print(f'  Real ML (abs): {abs_ml_value:.2e} - Similar to 1.58e-06 range')
            print(f'  Estimated:     {estimated_value:.2e} - Composition-based guess')
    else:
        print('No CIF files found')
else:
    print('Test directory not found')