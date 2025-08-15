#!/usr/bin/env python3
"""
Test script to debug ML bandgap correction model loading
"""

import os
import sys
import pickle
import numpy as np

def test_ml_model_loading():
    """Test loading the ML bandgap correction model with various compatibility fixes"""
    
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    print(f"🔍 Testing ML bandgap correction model loading...")
    print(f"📁 Model path: {model_path}")
    print(f"📊 File exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"📏 File size: {file_size:.1f} MB")
    
    # Test 1: Basic loading attempt
    print("\n🧪 Test 1: Basic pickle loading...")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Basic loading successful!")
        print(f"   Model keys: {list(model.keys()) if isinstance(model, dict) else type(model)}")
        return model
    except Exception as e:
        print(f"❌ Basic loading failed: {e}")
    
    # Test 2: With numpy compatibility fixes
    print("\n🧪 Test 2: With numpy compatibility fixes...")
    try:
        # Apply comprehensive numpy fixes
        import numpy as np
        import sys
        
        # Fix numpy._core compatibility
        if not hasattr(np, '_core'):
            try:
                import numpy.core as core_module
                np._core = core_module
                sys.modules['numpy._core'] = core_module
            except ImportError:
                pass
        
        # Fix BitGenerator compatibility
        try:
            import numpy.random
            if hasattr(numpy.random, '_pcg64'):
                sys.modules['numpy.random._pcg64'] = numpy.random._pcg64
                if hasattr(numpy.random._pcg64, 'PCG64'):
                    sys.modules['numpy.random._pcg64.PCG64'] = numpy.random._pcg64.PCG64
        except (ImportError, AttributeError):
            pass
        
        # Try loading with protocol compatibility
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Loading with numpy fixes successful!")
        print(f"   Model keys: {list(model.keys()) if isinstance(model, dict) else type(model)}")
        return model
        
    except Exception as e:
        print(f"❌ Loading with numpy fixes failed: {e}")
    
    # Test 3: With sklearn compatibility
    print("\n🧪 Test 3: With sklearn compatibility...")
    try:
        # Import sklearn first to register classes
        import sklearn
        import sklearn.ensemble
        import sklearn.preprocessing
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Loading with sklearn compatibility successful!")
        print(f"   Model keys: {list(model.keys()) if isinstance(model, dict) else type(model)}")
        return model
        
    except Exception as e:
        print(f"❌ Loading with sklearn compatibility failed: {e}")
    
    # Test 4: Alternative loading methods
    print("\n🧪 Test 4: Alternative loading methods...")
    try:
        import joblib
        model = joblib.load(model_path)
        print(f"✅ Loading with joblib successful!")
        print(f"   Model keys: {list(model.keys()) if isinstance(model, dict) else type(model)}")
        return model
    except Exception as e:
        print(f"❌ Loading with joblib failed: {e}")
    
    print("\n❌ All loading methods failed!")
    return None

def test_model_prediction(model):
    """Test making predictions with the loaded model"""
    if model is None:
        print("❌ No model to test predictions with")
        return
    
    print(f"\n🧪 Testing model predictions...")
    
    try:
        # Extract model components
        rf_model = model.get('rf_model')
        gb_model = model.get('gb_model') 
        scaler = model.get('scaler')
        weights = model.get('ensemble_weights')
        
        print(f"   RF model: {type(rf_model) if rf_model else 'None'}")
        print(f"   GB model: {type(gb_model) if gb_model else 'None'}")
        print(f"   Scaler: {type(scaler) if scaler else 'None'}")
        print(f"   Weights: {weights if weights else 'None'}")
        
        if all([rf_model, gb_model, scaler, weights]):
            # Test prediction with sample data
            import pandas as pd
            
            test_features = pd.DataFrame({
                'pbe_bandgap': [0.005],  # Small PBE bandgap
                'n_elements': [3], 
                'total_atoms': [12], 
                'avg_electronegativity': [2.5], 
                'avg_atomic_mass': [45.0],
                'has_O': [1], 'has_N': [0], 'has_C': [0], 'has_Si': [0],
                'has_Al': [0], 'has_Ti': [0], 'has_Fe': [0],
                'pbe_squared': [0.005 ** 2],
                'pbe_sqrt': [np.sqrt(0.005)],
                'en_pbe_product': [2.5 * 0.005]
            })
            
            X_scaled = scaler.transform(test_features)
            rf_pred = rf_model.predict(X_scaled)[0]
            gb_pred = gb_model.predict(X_scaled)[0]
            final_pred = weights[0] * rf_pred + weights[1] * gb_pred
            
            print(f"✅ Prediction test successful!")
            print(f"   Input PBE bandgap: 0.005 eV")
            print(f"   RF prediction: {rf_pred:.3f} eV")
            print(f"   GB prediction: {gb_pred:.3f} eV")
            print(f"   Final corrected bandgap: {final_pred:.3f} eV")
            
        else:
            print(f"❌ Missing model components")
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")

if __name__ == "__main__":
    model = test_ml_model_loading()
    test_model_prediction(model)