#!/usr/bin/env python3
"""
Aggressive numpy compatibility fix for loading sklearn models
"""

import sys
import os

def fix_numpy_compatibility():
    """Apply aggressive numpy compatibility fixes"""
    
    try:
        import numpy as np
        
        # Method 1: Direct module assignment
        if not hasattr(np, '_core'):
            import numpy.core as core_module
            np._core = core_module
            print("✅ Method 1: Direct assignment worked")
        
        # Method 2: System modules manipulation
        if 'numpy._core' not in sys.modules:
            import numpy.core
            sys.modules['numpy._core'] = numpy.core
            print("✅ Method 2: System modules worked")
            
        # Method 3: Create fake _core module
        if not hasattr(np, '_core'):
            class FakeCore:
                def __getattr__(self, name):
                    return getattr(np.core, name, None)
            np._core = FakeCore()
            print("✅ Method 3: Fake core worked")
            
        # Method 4: Monkey patch the import system
        original_import = __builtins__.__import__
        
        def patched_import(name, *args, **kwargs):
            if name == 'numpy._core':
                import numpy.core
                return numpy.core
            return original_import(name, *args, **kwargs)
        
        __builtins__.__import__ = patched_import
        print("✅ Method 4: Import patching applied")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility fix failed: {e}")
        return False

def test_model_loading():
    """Test loading the model after compatibility fixes"""
    
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        import pickle
        print("Attempting to load model...")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("✅ SUCCESS: Model loaded successfully!")
        print(f"Model type: {type(model)}")
        if isinstance(model, dict):
            print(f"Model keys: {list(model.keys())}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

if __name__ == "__main__":
    print("=== Applying numpy compatibility fixes ===")
    fix_numpy_compatibility()
    
    print("\n=== Testing model loading ===")
    model = test_model_loading()
    
    if model:
        print("\n=== Testing bandgap correction ===")
        try:
            import pandas as pd
            import numpy as np
            
            # Test correction
            pbe_bandgap = 2.5
            composition_str = "Li2O"
            
            rf_model = model['rf_model']
            gb_model = model['gb_model']
            scaler = model['scaler']
            weights = model['ensemble_weights']
            
            # Create features
            features = pd.DataFrame({
                'pbe_bandgap': [pbe_bandgap],
                'n_elements': [2],
                'total_atoms': [2],
                'avg_electronegativity': [2.0],
                'avg_atomic_mass': [50.0],
                'has_O': [1 if 'O' in composition_str else 0],
                'has_N': [0],
                'has_C': [0],
                'has_Si': [0],
                'has_Al': [0],
                'has_Ti': [0],
                'has_Fe': [0],
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
            
            print(f"✅ Bandgap correction test successful!")
            print(f"   PBE: {pbe_bandgap:.3f} eV -> HSE: {corrected_bandgap:.3f} eV")
            print(f"   RF prediction: {rf_pred:.3f}")
            print(f"   GB prediction: {gb_pred:.3f}")
            print(f"   Ensemble weights: {weights}")
            
        except Exception as e:
            print(f"❌ Bandgap correction test failed: {e}")
    
    print("\n=== Test completed ===")