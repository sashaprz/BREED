#!/usr/bin/env python3
"""
Root cause analysis: Debug your specific ML bandgap correction model
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

def test_basic_file_access():
    """Test 1: Can we access the file?"""
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    print("🔍 TEST 1: Basic File Access")
    print("-" * 30)
    
    print(f"📁 Model path: {model_path}")
    print(f"📊 File exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📏 File size: {file_size:.1f} MB")
        
        # Check file permissions
        print(f"🔓 Readable: {os.access(model_path, os.R_OK)}")
        
        return True
    else:
        print("❌ File not found!")
        return False

def test_environment_info():
    """Test 2: Check Python environment"""
    print("\n🔍 TEST 2: Environment Information")
    print("-" * 35)
    
    print(f"🐍 Python version: {sys.version}")
    
    # Check key packages
    packages = ['numpy', 'pandas', 'sklearn', 'joblib', 'pickle']
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"📦 {pkg}: {version}")
        except ImportError:
            print(f"❌ {pkg}: NOT INSTALLED")

def test_raw_pickle_loading():
    """Test 3: Raw pickle loading without any fixes"""
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    print("\n🔍 TEST 3: Raw Pickle Loading")
    print("-" * 30)
    
    try:
        print("🧪 Attempting raw pickle.load()...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("✅ Raw loading successful!")
        print(f"   Model type: {type(model)}")
        
        if isinstance(model, dict):
            print(f"   Model keys: {list(model.keys())}")
            
            # Check each component
            for key, value in model.items():
                print(f"   {key}: {type(value)}")
        
        return model
        
    except Exception as e:
        print(f"❌ Raw loading failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return None

def test_with_sklearn_import():
    """Test 4: Loading with sklearn imported first"""
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    print("\n🔍 TEST 4: Loading with sklearn imported")
    print("-" * 40)
    
    try:
        # Import sklearn components first
        print("📦 Importing sklearn components...")
        import sklearn
        import sklearn.ensemble
        import sklearn.preprocessing
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        print("   ✅ sklearn components imported")
        
        print("🧪 Attempting pickle.load() with sklearn...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("✅ Loading with sklearn successful!")
        return model
        
    except Exception as e:
        print(f"❌ Loading with sklearn failed: {e}")
        return None

def test_joblib_loading():
    """Test 5: Try joblib loading"""
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    print("\n🔍 TEST 5: Joblib Loading")
    print("-" * 25)
    
    try:
        import joblib
        print("🧪 Attempting joblib.load()...")
        model = joblib.load(model_path)
        
        print("✅ Joblib loading successful!")
        return model
        
    except ImportError:
        print("❌ joblib not available")
        return None
    except Exception as e:
        print(f"❌ Joblib loading failed: {e}")
        return None

def test_model_prediction(model):
    """Test 6: Test prediction with your model"""
    print("\n🔍 TEST 6: Model Prediction Test")
    print("-" * 30)
    
    if model is None:
        print("❌ No model to test")
        return False
    
    try:
        print("📋 Analyzing model structure...")
        
        if isinstance(model, dict):
            print(f"   Model keys: {list(model.keys())}")
            
            # Extract components
            rf_model = model.get('rf_model')
            gb_model = model.get('gb_model')
            scaler = model.get('scaler')
            weights = model.get('ensemble_weights')
            
            print(f"   🌲 RF model: {type(rf_model).__name__ if rf_model else 'None'}")
            print(f"   🚀 GB model: {type(gb_model).__name__ if gb_model else 'None'}")
            print(f"   📏 Scaler: {type(scaler).__name__ if scaler else 'None'}")
            print(f"   ⚖️ Weights: {weights}")
            
            if all([rf_model, gb_model, scaler, weights]):
                print("\n🧪 Testing prediction with sample PBE bandgap...")
                
                # Create test input - single PBE bandgap case
                pbe_bandgap = 0.005  # Small PBE bandgap (typical underestimate)
                
                # Create feature vector (matching your model's expected input)
                test_features = pd.DataFrame({
                    'pbe_bandgap': [pbe_bandgap],
                    'n_elements': [3],  # Example: 3 elements
                    'total_atoms': [12],  # Example: 12 atoms
                    'avg_electronegativity': [2.5],  # Example electronegativity
                    'avg_atomic_mass': [45.0],  # Example atomic mass
                    'has_O': [1],  # Has oxygen (oxide electrolyte)
                    'has_N': [0], 'has_C': [0], 'has_Si': [0],
                    'has_Al': [0], 'has_Ti': [0], 'has_Fe': [0],
                    'pbe_squared': [pbe_bandgap ** 2],
                    'pbe_sqrt': [np.sqrt(pbe_bandgap)],
                    'en_pbe_product': [2.5 * pbe_bandgap]
                })
                
                print(f"   📥 Input features shape: {test_features.shape}")
                print(f"   📥 PBE bandgap input: {pbe_bandgap} eV")
                
                # Scale features
                print("   📏 Scaling features...")
                X_scaled = scaler.transform(test_features)
                print(f"   📏 Scaled features shape: {X_scaled.shape}")
                
                # Make predictions
                print("   🌲 RF prediction...")
                rf_pred = rf_model.predict(X_scaled)[0]
                print(f"      RF result: {rf_pred:.4f} eV")
                
                print("   🚀 GB prediction...")
                gb_pred = gb_model.predict(X_scaled)[0]
                print(f"      GB result: {gb_pred:.4f} eV")
                
                # Ensemble prediction
                print("   🎯 Ensemble prediction...")
                final_pred = weights[0] * rf_pred + weights[1] * gb_pred
                print(f"      Final result: {final_pred:.4f} eV")
                
                # Analysis
                correction_factor = final_pred / pbe_bandgap if pbe_bandgap > 0 else 0
                print(f"\n📊 PREDICTION ANALYSIS:")
                print(f"   📥 Input (PBE): {pbe_bandgap:.4f} eV")
                print(f"   📤 Output (Corrected): {final_pred:.4f} eV")
                print(f"   📈 Correction factor: {correction_factor:.1f}x")
                print(f"   ✅ Model is working: {'YES' if final_pred > pbe_bandgap else 'MAYBE'}")
                
                return True
            else:
                print("❌ Missing model components")
                return False
        else:
            print(f"❌ Unexpected model type: {type(model)}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run complete diagnostic"""
    print("🚀 ROOT CAUSE ANALYSIS: Your ML Bandgap Model")
    print("=" * 50)
    
    # Test 1: File access
    if not test_basic_file_access():
        return
    
    # Test 2: Environment
    test_environment_info()
    
    # Test 3: Raw loading
    model = test_raw_pickle_loading()
    if model:
        test_model_prediction(model)
        return
    
    # Test 4: With sklearn
    model = test_with_sklearn_import()
    if model:
        test_model_prediction(model)
        return
    
    # Test 5: With joblib
    model = test_joblib_loading()
    if model:
        test_model_prediction(model)
        return
    
    print("\n❌ ALL LOADING METHODS FAILED")
    print("🔍 Root cause: Model file has compatibility issues with current environment")
    print("💡 Possible solutions:")
    print("   1. Model was saved with different numpy/sklearn versions")
    print("   2. Model contains objects not compatible with current Python version")
    print("   3. Model file is corrupted")
    print("   4. Missing dependencies that were used during model training")

if __name__ == "__main__":
    main()