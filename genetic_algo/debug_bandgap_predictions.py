#!/usr/bin/env python3
"""
Debug the bandgap prediction pipeline to understand why values are too small
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fully_optimized_predictor import predict_single_cif_fully_optimized, apply_ml_bandgap_correction
import pandas as pd
import numpy as np
import joblib

def debug_bandgap_pipeline():
    """Debug the entire bandgap prediction pipeline"""
    
    print("🔍 DEBUGGING BANDGAP PREDICTION PIPELINE")
    print("=" * 60)
    
    # Test the ML correction function directly
    print("\n1. Testing ML Bandgap Correction Function Directly:")
    print("-" * 50)
    
    test_pbe_values = [0.0001, 0.5, 1.0, 1.5, 2.0, 2.5]
    test_composition = "Li3Ti2P1O12"  # NASICON-type electrolyte
    
    for pbe_bg in test_pbe_values:
        corrected_bg = apply_ml_bandgap_correction(pbe_bg, test_composition)
        correction_factor = corrected_bg / pbe_bg if pbe_bg > 0 else 0
        print(f"PBE: {pbe_bg:.4f} eV → HSE: {corrected_bg:.4f} eV (factor: {correction_factor:.1f}x)")
    
    # Load and inspect the ML model directly
    print("\n2. Inspecting ML Model Directly:")
    print("-" * 50)
    
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model_joblib.pkl"
    
    try:
        model_data = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"Model keys: {list(model_data.keys())}")
        
        if 'performance' in model_data:
            perf = model_data['performance']
            print(f"Training performance:")
            print(f"  - Ensemble MAE: {perf.get('ensemble_mae', 'unknown'):.4f} eV")
            print(f"  - Ensemble R²: {perf.get('ensemble_r2', 'unknown'):.4f}")
        
        if 'training_samples' in model_data:
            print(f"Training samples: {model_data['training_samples']}")
        
        # Test prediction with the model directly
        print(f"\n3. Testing Model Prediction Directly:")
        print("-" * 50)
        
        rf_model = model_data['rf_model']
        gb_model = model_data['gb_model']
        scaler = model_data['scaler']
        weights = model_data['ensemble_weights']
        
        # Create test features for a typical electrolyte
        test_features = pd.DataFrame({
            'pbe_bandgap': [0.0001, 0.5, 1.0, 2.0],
            'n_elements': [4, 4, 4, 4],  # Li-Ti-P-O system
            'total_atoms': [18, 18, 18, 18],  # Li3Ti2P1O12
            'avg_electronegativity': [2.5, 2.5, 2.5, 2.5],
            'avg_atomic_mass': [45.0, 45.0, 45.0, 45.0],
            'has_O': [1, 1, 1, 1],
            'has_N': [0, 0, 0, 0],
            'has_C': [0, 0, 0, 0],
            'has_Si': [0, 0, 0, 0],
            'has_Al': [0, 0, 0, 0],
            'has_Ti': [1, 1, 1, 1],
            'has_Fe': [0, 0, 0, 0],
            'pbe_squared': [0.0001**2, 0.5**2, 1.0**2, 2.0**2],
            'pbe_sqrt': [np.sqrt(0.0001), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(2.0)],
            'en_pbe_product': [2.5 * 0.0001, 2.5 * 0.5, 2.5 * 1.0, 2.5 * 2.0]
        })
        
        X_scaled = scaler.transform(test_features)
        rf_pred = rf_model.predict(X_scaled)
        gb_pred = gb_model.predict(X_scaled)
        ensemble_pred = weights[0] * rf_pred + weights[1] * gb_pred
        
        print("Direct model predictions:")
        for i, (pbe, hse_pred) in enumerate(zip(test_features['pbe_bandgap'], ensemble_pred)):
            factor = hse_pred / pbe if pbe > 0 else 0
            print(f"  PBE: {pbe:.4f} eV → HSE: {hse_pred:.4f} eV (factor: {factor:.1f}x)")
        
        # Check feature importance
        if 'feature_importance' in model_data:
            print(f"\n4. Feature Importance:")
            print("-" * 50)
            importance = model_data['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feature, imp in sorted_features[:10]:
                print(f"  {feature}: {imp:.4f}")
        
        # Check training data statistics
        print(f"\n5. Checking Training Data Range:")
        print("-" * 50)
        
        # Load training data to see the range
        data_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\jarvis_paired_data\jarvis_paired_bandgaps.csv"
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            print(f"Training data statistics:")
            print(f"  PBE bandgap range: {data['pbe_bandgap'].min():.4f} - {data['pbe_bandgap'].max():.4f} eV")
            print(f"  HSE bandgap range: {data['hse_bandgap'].min():.4f} - {data['hse_bandgap'].max():.4f} eV")
            print(f"  PBE mean: {data['pbe_bandgap'].mean():.4f} eV")
            print(f"  HSE mean: {data['hse_bandgap'].mean():.4f} eV")
            
            # Check correction factors in training data
            correction_factors = data['hse_bandgap'] / data['pbe_bandgap']
            correction_factors = correction_factors[correction_factors.isfinite()]  # Remove inf/nan
            print(f"  Correction factor range: {correction_factors.min():.1f}x - {correction_factors.max():.1f}x")
            print(f"  Correction factor mean: {correction_factors.mean():.1f}x")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_bandgap_pipeline()