#!/usr/bin/env python3
"""
Targeted fix for numpy._core compatibility issue
"""

import os
import sys
import pickle
import numpy as np

def fix_numpy_core_compatibility():
    """Fix the specific numpy._core compatibility issue"""
    
    print("🔧 Fixing numpy._core compatibility...")
    
    try:
        # Method 1: Create numpy._core module mapping
        if not hasattr(np, '_core'):
            print("   📦 Creating numpy._core module...")
            
            # Import the actual core modules
            import numpy.core
            import numpy.core.multiarray
            import numpy.core._multiarray_umath
            
            # Create the _core namespace
            np._core = numpy.core
            
            # Register in sys.modules
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
            sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
            
            print("   ✅ numpy._core module created")
        
        # Method 2: Handle specific numpy._core submodules
        core_modules = [
            'multiarray',
            '_multiarray_umath',
            'numeric',
            'fromnumeric',
            'records',
            'function_base',
            'machar',
            'getlimits',
            'shape_base'
        ]
        
        for module_name in core_modules:
            full_name = f'numpy._core.{module_name}'
            old_name = f'numpy.core.{module_name}'
            
            if full_name not in sys.modules and old_name in sys.modules:
                sys.modules[full_name] = sys.modules[old_name]
                print(f"   ✅ Mapped {full_name}")
        
        # Method 3: Handle numpy random compatibility
        if hasattr(np.random, '_pcg64'):
            sys.modules['numpy._core._pcg64'] = np.random._pcg64
            if hasattr(np.random._pcg64, 'PCG64'):
                sys.modules['numpy._core._pcg64.PCG64'] = np.random._pcg64.PCG64
            print("   ✅ numpy random compatibility fixed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ numpy._core fix failed: {e}")
        return False

def test_model_loading_with_fix():
    """Test loading your model with the numpy fix"""
    
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    print("\n🧪 Testing model loading with numpy._core fix...")
    
    # Apply the fix
    if not fix_numpy_core_compatibility():
        return None
    
    # Import sklearn components
    try:
        import sklearn
        import sklearn.ensemble
        import sklearn.preprocessing
        print("   📦 sklearn components imported")
    except Exception as e:
        print(f"   ❌ sklearn import failed: {e}")
        return None
    
    # Try loading the model
    try:
        print("   🔄 Loading model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("   ✅ Model loaded successfully!")
        print(f"   📋 Model type: {type(model)}")
        
        if isinstance(model, dict):
            print(f"   📋 Model keys: {list(model.keys())}")
        
        return model
        
    except Exception as e:
        print(f"   ❌ Model loading still failed: {e}")
        return None

def test_single_prediction(model):
    """Test a single prediction with your model"""
    
    if model is None:
        print("❌ No model to test")
        return False
    
    print("\n🎯 Testing single prediction...")
    
    try:
        import pandas as pd
        
        # Extract model components
        rf_model = model.get('rf_model')
        gb_model = model.get('gb_model')
        scaler = model.get('scaler')
        weights = model.get('ensemble_weights')
        
        print(f"   🌲 RF model: {type(rf_model).__name__ if rf_model else 'None'}")
        print(f"   🚀 GB model: {type(gb_model).__name__ if gb_model else 'None'}")
        print(f"   📏 Scaler: {type(scaler).__name__ if scaler else 'None'}")
        print(f"   ⚖️ Weights: {weights}")
        
        if not all([rf_model, gb_model, scaler, weights]):
            print("   ❌ Missing model components")
            return False
        
        # Test with a single PBE bandgap value
        pbe_bandgap = 0.005  # Small PBE value (typical underestimate)
        
        print(f"\n   📥 Testing with PBE bandgap: {pbe_bandgap} eV")
        
        # Create feature vector
        features = pd.DataFrame({
            'pbe_bandgap': [pbe_bandgap],
            'n_elements': [3],
            'total_atoms': [12],
            'avg_electronegativity': [2.5],
            'avg_atomic_mass': [45.0],
            'has_O': [1],  # Oxide electrolyte
            'has_N': [0], 'has_C': [0], 'has_Si': [0],
            'has_Al': [0], 'has_Ti': [0], 'has_Fe': [0],
            'pbe_squared': [pbe_bandgap ** 2],
            'pbe_sqrt': [np.sqrt(pbe_bandgap)],
            'en_pbe_product': [2.5 * pbe_bandgap]
        })
        
        # Scale and predict
        X_scaled = scaler.transform(features)
        rf_pred = rf_model.predict(X_scaled)[0]
        gb_pred = gb_model.predict(X_scaled)[0]
        final_pred = weights[0] * rf_pred + weights[1] * gb_pred
        
        print(f"   🌲 RF prediction: {rf_pred:.4f} eV")
        print(f"   🚀 GB prediction: {gb_pred:.4f} eV")
        print(f"   🎯 Final prediction: {final_pred:.4f} eV")
        print(f"   📈 Correction factor: {final_pred/pbe_bandgap:.1f}x")
        
        # Check if it's working correctly
        if final_pred > pbe_bandgap:
            print("   ✅ Model is correcting PBE bandgap upward (expected behavior)")
            return True
        else:
            print("   ⚠️ Model prediction is not higher than PBE input")
            return False
            
    except Exception as e:
        print(f"   ❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def integrate_into_predictor(model):
    """Show how to integrate the working model into the predictor"""
    
    if model is None:
        print("❌ No model to integrate")
        return
    
    print("\n🔧 Integration Instructions:")
    print("-" * 30)
    print("1. Copy the numpy._core fix to fully_optimized_predictor.py")
    print("2. Apply the fix before loading the model")
    print("3. The model should then load successfully")
    print("4. Test with genetic algorithm")
    
    # Show the exact code to add
    print("\n📝 Code to add to fully_optimized_predictor.py:")
    print("""
# Add this before the model loading section:
def fix_numpy_core_compatibility():
    import numpy as np
    import sys
    
    if not hasattr(np, '_core'):
        import numpy.core
        np._core = numpy.core
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
    
    return True

# Then call it before loading:
fix_numpy_core_compatibility()
""")

def main():
    """Main function"""
    
    print("🎯 TARGETED FIX: numpy._core Compatibility")
    print("=" * 45)
    
    # Test the fix
    model = test_model_loading_with_fix()
    
    if model:
        # Test prediction
        success = test_single_prediction(model)
        
        if success:
            print("\n🎉 SUCCESS! Your model is working!")
            integrate_into_predictor(model)
        else:
            print("\n⚠️ Model loaded but prediction test failed")
    else:
        print("\n❌ Model loading still failed after numpy fix")

if __name__ == "__main__":
    main()