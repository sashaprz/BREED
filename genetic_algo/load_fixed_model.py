#!/usr/bin/env python3
"""
Load the binary-fixed model with proper error handling
"""

import os
import sys
import pickle
import numpy as np

def setup_environment():
    """Setup complete environment"""
    
    # Fix numpy compatibility
    if not hasattr(np, '_core'):
        import numpy.core
        np._core = numpy.core
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
    
    # Fix random generators
    import numpy.random
    if hasattr(numpy.random, '_pcg64'):
        numpy.random.PCG64 = numpy.random._pcg64.PCG64
        sys.modules['numpy.random.PCG64'] = numpy.random._pcg64.PCG64
        sys.modules['numpy.random._pcg64'] = numpy.random._pcg64
        sys.modules['numpy.random._pcg64.PCG64'] = numpy.random._pcg64.PCG64
    
    # Import sklearn
    import sklearn
    import sklearn.ensemble
    import sklearn.preprocessing
    
    return True

def load_fixed_model():
    """Load the binary-fixed model"""
    
    fixed_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\genetic_algo\fixed_bandgap_model.pkl"
    
    print("🔄 Loading binary-fixed model...")
    
    if not os.path.exists(fixed_path):
        print("❌ Fixed model file not found")
        return None
    
    setup_environment()
    
    class FixedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle numpy random pickle constructor
            if module == 'numpy.random._pickle' and name == '__generator_ctor':
                print(f"   🔧 Fixing random generator constructor")
                # Return a function that creates a default random generator
                def default_generator_ctor(*args):
                    return np.random.default_rng()
                return default_generator_ctor
            
            # Handle other numpy._core remapping
            if module.startswith('numpy._core'):
                new_module = module.replace('numpy._core', 'numpy.core')
                try:
                    mod = __import__(new_module, fromlist=[name])
                    return getattr(mod, name)
                except:
                    pass
            
            # Handle PCG64
            if 'pcg64' in module.lower() or 'PCG64' in name:
                return np.random._pcg64.PCG64
            
            return super().find_class(module, name)
    
    try:
        with open(fixed_path, 'rb') as f:
            unpickler = FixedUnpickler(f)
            model = unpickler.load()
        
        print("✅ Fixed model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Fixed model loading failed: {e}")
        
        # Try with protocol override
        try:
            print("🔄 Trying with protocol override...")
            with open(fixed_path, 'rb') as f:
                # Try loading with different encoding
                model = pickle.load(f, encoding='latin1')
            print("✅ Loaded with latin1 encoding!")
            return model
        except Exception as e2:
            print(f"❌ Protocol override failed: {e2}")
            return None

def test_fixed_model(model):
    """Test the fixed model"""
    
    if model is None:
        return False
    
    print("\n🧪 Testing fixed model...")
    
    try:
        import pandas as pd
        
        print(f"   📋 Model type: {type(model)}")
        
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
            print(f"   ⚖️ Weights: {weights}")
            
            if all([rf_model, gb_model, scaler, weights]):
                # Test prediction
                test_data = pd.DataFrame({
                    'pbe_bandgap': [0.005],
                    'n_elements': [3], 'total_atoms': [12], 'avg_electronegativity': [2.5], 'avg_atomic_mass': [45.0],
                    'has_O': [1], 'has_N': [0], 'has_C': [0], 'has_Si': [0],
                    'has_Al': [0], 'has_Ti': [0], 'has_Fe': [0],
                    'pbe_squared': [0.005 ** 2], 'pbe_sqrt': [np.sqrt(0.005)], 'en_pbe_product': [2.5 * 0.005]
                })
                
                X_scaled = scaler.transform(test_data)
                rf_pred = rf_model.predict(X_scaled)[0]
                gb_pred = gb_model.predict(X_scaled)[0]
                final_pred = weights[0] * rf_pred + weights[1] * gb_pred
                
                print(f"\n   🎯 PREDICTION TEST:")
                print(f"      Input PBE: 0.005 eV")
                print(f"      RF prediction: {rf_pred:.4f} eV")
                print(f"      GB prediction: {gb_pred:.4f} eV")
                print(f"      Final prediction: {final_pred:.4f} eV")
                print(f"      Correction factor: {final_pred/0.005:.1f}x")
                
                if final_pred > 0.005:
                    print("   ✅ Model is working correctly!")
                    return True
                else:
                    print("   ⚠️ Model prediction seems low")
                    return False
            else:
                print("   ❌ Missing model components")
                return False
        else:
            print(f"   ❌ Unexpected model type: {type(model)}")
            return False
            
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def integrate_working_model(model):
    """Integrate the working model into the predictor"""
    
    if model is None:
        return False
    
    print("\n🔧 Integrating working model into predictor...")
    
    # Save the working model in a more accessible location
    working_model_path = "working_bandgap_model.pkl"
    
    try:
        with open(working_model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"   💾 Working model saved to: {working_model_path}")
        
        # Update the predictor to use the working model
        predictor_code = f'''
# Update the model path in fully_optimized_predictor.py
BANDGAP_CORRECTION_MODEL_PATH = r"C:\\Users\\Sasha\\repos\\RL-electrolyte-design\\genetic_algo\\{working_model_path}"
'''
        
        print("   📝 Update fully_optimized_predictor.py with:")
        print(predictor_code)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        return False

def main():
    """Main function"""
    
    print("🎯 LOADING BINARY-FIXED MODEL")
    print("=" * 35)
    
    # Load the fixed model
    model = load_fixed_model()
    
    if model:
        # Test the model
        if test_fixed_model(model):
            print("\n🎉 SUCCESS! Your ML model is now working!")
            
            # Integrate into predictor
            if integrate_working_model(model):
                print("\n✅ Model integrated! Now test with genetic algorithm:")
                print("   1. Update BANDGAP_CORRECTION_MODEL_PATH in fully_optimized_predictor.py")
                print("   2. Run: python genetic_algo_true_cdvae.py")
                print("   3. Look for: correction_method: ml_ensemble")
            
            return True
        else:
            print("\n⚠️ Model loaded but testing failed")
            return False
    else:
        print("\n❌ Could not load the fixed model")
        
        # Final diagnostic
        print("\n🔍 FINAL DIAGNOSTIC:")
        print("Your model has deep compatibility issues that require:")
        print("1. Retraining with current numpy/sklearn versions, OR")
        print("2. Creating a virtual environment with the exact versions used during training")
        print("\nTo find the original versions, check:")
        print("- The training script or notebook")
        print("- requirements.txt from when the model was created")
        print("- Git history of the training environment")
        
        return False

if __name__ == "__main__":
    main()