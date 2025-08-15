#!/usr/bin/env python3
"""
Fix ML bandgap correction model loading with comprehensive compatibility fixes
"""

import os
import sys
import pickle
import numpy as np

def fix_numpy_sklearn_compatibility():
    """Apply comprehensive numpy and sklearn compatibility fixes"""
    
    print("🔧 Applying comprehensive numpy/sklearn compatibility fixes...")
    
    # Fix 1: Import sklearn first to register all classes
    try:
        import sklearn
        import sklearn.ensemble
        import sklearn.preprocessing
        import sklearn.base
        print("   ✅ sklearn imported successfully")
    except ImportError as e:
        print(f"   ❌ sklearn import failed: {e}")
        return False
    
    # Fix 2: Handle numpy version compatibility
    try:
        import numpy as np
        print(f"   📊 numpy version: {np.__version__}")
        
        # Fix numpy._core compatibility
        if not hasattr(np, '_core'):
            import numpy.core as core_module
            np._core = core_module
            sys.modules['numpy._core'] = core_module
            print("   ✅ numpy._core compatibility fixed")
        
        # Fix numpy random BitGenerator compatibility
        if hasattr(np.random, '_pcg64'):
            # Map PCG64 BitGenerator for older pickle compatibility
            sys.modules['numpy.random._pcg64'] = np.random._pcg64
            if hasattr(np.random._pcg64, 'PCG64'):
                sys.modules['numpy.random._pcg64.PCG64'] = np.random._pcg64.PCG64
            print("   ✅ numpy random BitGenerator compatibility fixed")
            
    except Exception as e:
        print(f"   ❌ numpy compatibility fix failed: {e}")
        return False
    
    # Fix 3: Handle joblib compatibility (sklearn uses joblib internally)
    try:
        import joblib
        print(f"   📦 joblib version: {joblib.__version__}")
    except ImportError:
        print("   ⚠️ joblib not available")
    
    return True

def load_ml_model_with_fixes(model_path):
    """Load ML model with all compatibility fixes applied"""
    
    print(f"🔄 Loading ML model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📏 Model file size: {file_size:.1f} MB")
    
    # Apply compatibility fixes first
    if not fix_numpy_sklearn_compatibility():
        print("❌ Compatibility fixes failed")
        return None
    
    # Method 1: Direct pickle loading
    try:
        print("🧪 Attempting direct pickle loading...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Direct pickle loading successful!")
        return model
    except Exception as e:
        print(f"❌ Direct pickle loading failed: {e}")
    
    # Method 2: Joblib loading (sklearn often uses joblib)
    try:
        print("🧪 Attempting joblib loading...")
        import joblib
        model = joblib.load(model_path)
        print("✅ Joblib loading successful!")
        return model
    except Exception as e:
        print(f"❌ Joblib loading failed: {e}")
    
    # Method 3: Protocol-specific loading
    try:
        print("🧪 Attempting protocol-specific loading...")
        with open(model_path, 'rb') as f:
            # Try different pickle protocols
            for protocol in [None, 2, 3, 4, 5]:
                try:
                    f.seek(0)
                    if protocol is None:
                        model = pickle.load(f)
                    else:
                        model = pickle.load(f)  # Protocol is for saving, not loading
                    print(f"✅ Protocol-specific loading successful!")
                    return model
                except Exception:
                    continue
    except Exception as e:
        print(f"❌ Protocol-specific loading failed: {e}")
    
    print("❌ All loading methods failed!")
    return None

def test_model_prediction(model):
    """Test the loaded model with sample data"""
    
    if model is None:
        print("❌ No model to test")
        return False
    
    print("🧪 Testing model prediction...")
    
    try:
        # Check model structure
        if isinstance(model, dict):
            print(f"   📋 Model keys: {list(model.keys())}")
            
            # Extract components
            rf_model = model.get('rf_model')
            gb_model = model.get('gb_model')
            scaler = model.get('scaler')
            weights = model.get('ensemble_weights')
            
            print(f"   🌲 RF model: {type(rf_model).__name__ if rf_model else 'None'}")
            print(f"   🚀 GB model: {type(gb_model).__name__ if gb_model else 'None'}")
            print(f"   📏 Scaler: {type(scaler).__name__ if scaler else 'None'}")
            print(f"   ⚖️ Weights: {weights if weights else 'None'}")
            
            if all([rf_model, gb_model, scaler, weights]):
                # Test prediction
                import pandas as pd
                
                test_data = pd.DataFrame({
                    'pbe_bandgap': [0.005],
                    'n_elements': [3], 'total_atoms': [12], 'avg_electronegativity': [2.5], 'avg_atomic_mass': [45.0],
                    'has_O': [1], 'has_N': [0], 'has_C': [0], 'has_Si': [0],
                    'has_Al': [0], 'has_Ti': [0], 'has_Fe': [0],
                    'pbe_squared': [0.005 ** 2],
                    'pbe_sqrt': [np.sqrt(0.005)],
                    'en_pbe_product': [2.5 * 0.005]
                })
                
                X_scaled = scaler.transform(test_data)
                rf_pred = rf_model.predict(X_scaled)[0]
                gb_pred = gb_model.predict(X_scaled)[0]
                final_pred = weights[0] * rf_pred + weights[1] * gb_pred
                
                print(f"✅ Model prediction test successful!")
                print(f"   📥 Input PBE bandgap: 0.005 eV")
                print(f"   🌲 RF prediction: {rf_pred:.3f} eV")
                print(f"   🚀 GB prediction: {gb_pred:.3f} eV")
                print(f"   🎯 Final corrected bandgap: {final_pred:.3f} eV")
                print(f"   📈 Correction factor: {final_pred/0.005:.1f}x")
                
                return True
            else:
                print("❌ Missing model components")
                return False
        else:
            print(f"❌ Unexpected model type: {type(model)}")
            return False
            
    except Exception as e:
        print(f"❌ Model prediction test failed: {e}")
        return False

def main():
    """Main function to test ML model loading"""
    
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    print("🔍 ML Bandgap Correction Model Loading Test")
    print("=" * 50)
    
    # Load the model
    model = load_ml_model_with_fixes(model_path)
    
    # Test the model
    success = test_model_prediction(model)
    
    if success:
        print("\n🎉 SUCCESS! ML model loaded and tested successfully!")
        print("   The model can now be integrated into the predictor.")
    else:
        print("\n❌ FAILED! ML model could not be loaded or tested.")
        print("   Will continue using literature-based fallback.")
    
    return model, success

if __name__ == "__main__":
    main()