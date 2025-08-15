#!/usr/bin/env python3
"""
Bandgap correction model loader with numpy compatibility fixes
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_bandgap_correction_model():
    """Load the bandgap correction model with compatibility fixes"""
    
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        return None, False
    
    # Fix numpy compatibility issue
    try:
        import numpy as np
        
        # Create numpy._core if it doesn't exist (compatibility fix)
        if not hasattr(np, '_core'):
            import numpy.core as _core
            np._core = _core
            
        # Also try the alternative approach
        if not hasattr(np, '_core'):
            import sys
            import numpy.core
            sys.modules['numpy._core'] = numpy.core
            np._core = numpy.core
            
    except Exception as e:
        print(f"⚠️ Numpy compatibility fix failed: {e}")
    
    # Try multiple loading strategies
    loading_strategies = [
        ("Standard pickle", lambda: pickle.load(open(model_path, 'rb'))),
        ("Joblib", lambda: __import__('joblib').load(model_path)),
        ("Pickle with protocol", lambda: pickle.load(open(model_path, 'rb'))),
    ]
    
    for strategy_name, loader in loading_strategies:
        try:
            print(f"Trying {strategy_name}...")
            model = loader()
            print(f"✅ SUCCESS: {strategy_name} worked!")
            return model, True
        except Exception as e:
            print(f"❌ {strategy_name} failed: {e}")
            continue
    
    print("❌ All loading strategies failed")
    return None, False

def apply_bandgap_correction(pbe_bandgap: float, composition_str: str = None) -> tuple:
    """Apply bandgap correction and return (corrected_value, success, method)"""
    
    try:
        # Load model if not already loaded
        if not hasattr(apply_bandgap_correction, '_model'):
            model, success = load_bandgap_correction_model()
            if not success:
                return pbe_bandgap, False, "model_load_failed"
            apply_bandgap_correction._model = model
        
        model = apply_bandgap_correction._model
        
        # Extract model components
        rf_model = model['rf_model']
        gb_model = model['gb_model']
        scaler = model['scaler']
        weights = model['ensemble_weights']
        
        # Create features (simplified for single prediction)
        import pandas as pd
        import numpy as np
        
        features = pd.DataFrame({
            'pbe_bandgap': [pbe_bandgap],
            'n_elements': [2],  # Default
            'total_atoms': [2],  # Default
            'avg_electronegativity': [2.0],  # Default
            'avg_atomic_mass': [50.0],  # Default
            'has_O': [1 if composition_str and 'O' in composition_str else 0],
            'has_N': [1 if composition_str and 'N' in composition_str else 0],
            'has_C': [1 if composition_str and 'C' in composition_str else 0],
            'has_Si': [1 if composition_str and 'Si' in composition_str else 0],
            'has_Al': [1 if composition_str and 'Al' in composition_str else 0],
            'has_Ti': [1 if composition_str and 'Ti' in composition_str else 0],
            'has_Fe': [1 if composition_str and 'Fe' in composition_str else 0],
            'pbe_squared': [pbe_bandgap ** 2],
            'pbe_sqrt': [np.sqrt(abs(pbe_bandgap))],
            'en_pbe_product': [2.0 * pbe_bandgap]
        })
        
        # Scale and predict
        X_scaled = scaler.transform(features)
        rf_pred = rf_model.predict(X_scaled)[0]
        gb_pred = gb_model.predict(X_scaled)[0]
        
        # Ensemble prediction
        corrected_bandgap = weights[0] * rf_pred + weights[1] * gb_pred
        
        return corrected_bandgap, True, "ml_ensemble"
        
    except Exception as e:
        print(f"⚠️ Error applying bandgap correction: {e}")
        return pbe_bandgap, False, f"correction_failed: {e}"

if __name__ == "__main__":
    # Test the loader
    print("Testing bandgap correction loader...")
    model, success = load_bandgap_correction_model()
    
    if success:
        print(f"Model loaded successfully!")
        print(f"Model keys: {list(model.keys()) if isinstance(model, dict) else type(model)}")
        
        # Test correction
        test_pbe = 2.5
        corrected, success, method = apply_bandgap_correction(test_pbe, "Li2O")
        print(f"Test correction: {test_pbe} eV (PBE) -> {corrected:.3f} eV (HSE), Success: {success}, Method: {method}")
    else:
        print("Failed to load model")