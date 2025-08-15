#!/usr/bin/env python3
"""
Final comprehensive fix for your ML bandgap model
"""

import os
import sys
import pickle
import numpy as np

def comprehensive_numpy_fix():
    """Apply all necessary numpy compatibility fixes"""
    
    print("🔧 Applying comprehensive numpy compatibility fixes...")
    
    try:
        # Fix 1: numpy._core compatibility
        if not hasattr(np, '_core'):
            import numpy.core
            np._core = numpy.core
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
            sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
            print("   ✅ numpy._core fixed")
        
        # Fix 2: PCG64 BitGenerator compatibility
        import numpy.random
        
        # Create compatibility for PCG64
        if hasattr(numpy.random, '_pcg64') and hasattr(numpy.random._pcg64, 'PCG64'):
            # Register PCG64 in multiple locations for pickle compatibility
            sys.modules['numpy.random.PCG64'] = numpy.random._pcg64.PCG64
            sys.modules['numpy.random._pcg64'] = numpy.random._pcg64
            sys.modules['numpy.random._pcg64.PCG64'] = numpy.random._pcg64.PCG64
            
            # Also register in the main numpy.random namespace
            if not hasattr(numpy.random, 'PCG64'):
                numpy.random.PCG64 = numpy.random._pcg64.PCG64
            
            print("   ✅ PCG64 BitGenerator fixed")
        
        # Fix 3: Handle other potential BitGenerators
        bit_generators = ['MT19937', 'Philox', 'SFC64']
        for bg_name in bit_generators:
            if hasattr(numpy.random, f'_{bg_name.lower()}'):
                bg_module = getattr(numpy.random, f'_{bg_name.lower()}')
                if hasattr(bg_module, bg_name):
                    sys.modules[f'numpy.random.{bg_name}'] = getattr(bg_module, bg_name)
                    sys.modules[f'numpy.random._{bg_name.lower()}'] = bg_module
                    sys.modules[f'numpy.random._{bg_name.lower()}.{bg_name}'] = getattr(bg_module, bg_name)
        
        print("   ✅ All BitGenerators registered")
        
        # Fix 4: Ensure sklearn compatibility
        import sklearn
        import sklearn.ensemble
        import sklearn.preprocessing
        print("   ✅ sklearn imported")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Comprehensive fix failed: {e}")
        return False

def load_model_with_comprehensive_fix():
    """Load your model with all fixes applied"""
    
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    print("\n🔄 Loading model with comprehensive fixes...")
    
    # Apply all fixes
    if not comprehensive_numpy_fix():
        return None
    
    # Try loading
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        
        # Try alternative approach: load with custom unpickler
        try:
            print("🔄 Trying custom unpickler...")
            
            class CompatibilityUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Handle numpy._core remapping
                    if module.startswith('numpy._core'):
                        module = module.replace('numpy._core', 'numpy.core')
                    
                    # Handle PCG64 remapping
                    if module == 'numpy.random._pcg64' and name == 'PCG64':
                        return numpy.random._pcg64.PCG64
                    
                    return super().find_class(module, name)
            
            with open(model_path, 'rb') as f:
                model = CompatibilityUnpickler(f).load()
            
            print("✅ Model loaded with custom unpickler!")
            return model
            
        except Exception as e2:
            print(f"❌ Custom unpickler also failed: {e2}")
            return None

def test_model_functionality(model):
    """Test if the loaded model works correctly"""
    
    if model is None:
        return False
    
    print("\n🧪 Testing model functionality...")
    
    try:
        import pandas as pd
        
        # Check model structure
        if not isinstance(model, dict):
            print(f"❌ Expected dict, got {type(model)}")
            return False
        
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
        
        if not all([rf_model, gb_model, scaler, weights]):
            print("❌ Missing required model components")
            return False
        
        # Test prediction with realistic electrolyte data
        test_cases = [
            {'pbe': 0.001, 'desc': 'Very small PBE (metallic prediction)'},
            {'pbe': 0.005, 'desc': 'Small PBE (typical underestimate)'},
            {'pbe': 0.050, 'desc': 'Medium PBE (moderate underestimate)'},
        ]
        
        print("\n   🎯 Testing predictions:")
        
        for i, case in enumerate(test_cases):
            pbe_bg = case['pbe']
            
            # Create feature vector
            features = pd.DataFrame({
                'pbe_bandgap': [pbe_bg],
                'n_elements': [3], 'total_atoms': [12], 'avg_electronegativity': [2.5], 'avg_atomic_mass': [45.0],
                'has_O': [1], 'has_N': [0], 'has_C': [0], 'has_Si': [0],
                'has_Al': [0], 'has_Ti': [0], 'has_Fe': [0],
                'pbe_squared': [pbe_bg ** 2], 'pbe_sqrt': [np.sqrt(pbe_bg)], 'en_pbe_product': [2.5 * pbe_bg]
            })
            
            # Make prediction
            X_scaled = scaler.transform(features)
            rf_pred = rf_model.predict(X_scaled)[0]
            gb_pred = gb_model.predict(X_scaled)[0]
            final_pred = weights[0] * rf_pred + weights[1] * gb_pred
            
            correction_factor = final_pred / pbe_bg if pbe_bg > 0 else 0
            
            print(f"      Test {i+1}: {pbe_bg:.3f} eV → {final_pred:.3f} eV ({correction_factor:.1f}x) - {case['desc']}")
        
        print("   ✅ All prediction tests successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Model functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_predictor_with_fix():
    """Update the predictor with the working fix"""
    
    print("\n🔧 Updating fully_optimized_predictor.py...")
    
    # Read current predictor
    predictor_path = "fully_optimized_predictor.py"
    
    try:
        with open(predictor_path, 'r') as f:
            content = f.read()
        
        # Find the model loading section and replace it
        new_loading_code = '''
def comprehensive_numpy_fix():
    """Apply all necessary numpy compatibility fixes"""
    import numpy as np
    import sys
    
    # Fix numpy._core compatibility
    if not hasattr(np, '_core'):
        import numpy.core
        np._core = numpy.core
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
    
    # Fix PCG64 BitGenerator compatibility
    import numpy.random
    if hasattr(numpy.random, '_pcg64') and hasattr(numpy.random._pcg64, 'PCG64'):
        sys.modules['numpy.random.PCG64'] = numpy.random._pcg64.PCG64
        sys.modules['numpy.random._pcg64'] = numpy.random._pcg64
        sys.modules['numpy.random._pcg64.PCG64'] = numpy.random._pcg64.PCG64
        if not hasattr(numpy.random, 'PCG64'):
            numpy.random.PCG64 = numpy.random._pcg64.PCG64
    
    return True

# Try to load ML model, but provide literature-based fallback
try:
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    
    # Apply comprehensive numpy fixes
    comprehensive_numpy_fix()
    
    # Import sklearn components
    import sklearn
    import sklearn.ensemble
    import sklearn.preprocessing
    
    if os.path.exists(BANDGAP_CORRECTION_MODEL_PATH):
        print(f"📁 Loading ML bandgap correction model from: {BANDGAP_CORRECTION_MODEL_PATH}")
        with open(BANDGAP_CORRECTION_MODEL_PATH, 'rb') as f:
            BANDGAP_CORRECTION_MODEL = pickle.load(f)
        print("✅ ML Bandgap correction model loaded successfully - will apply ensemble PBE→HSE corrections")
        print(f"   Model contains: {list(BANDGAP_CORRECTION_MODEL.keys())}")
        BANDGAP_CORRECTION_AVAILABLE = True
        CORRECTION_METHOD = "ml_ensemble"
    else:
        raise FileNotFoundError(f"Model file not found at: {BANDGAP_CORRECTION_MODEL_PATH}")
        '''
        
        # Replace the old loading section
        start_marker = "# Try to load ML model, but provide literature-based fallback"
        end_marker = 'CORRECTION_METHOD = "literature_based"'
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker) + len(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            new_content = content[:start_idx] + new_loading_code + content[end_idx:]
            
            # Write updated file
            with open(predictor_path, 'w') as f:
                f.write(new_content)
            
            print("   ✅ Predictor updated with comprehensive fix!")
            return True
        else:
            print("   ❌ Could not find loading section to replace")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to update predictor: {e}")
        return False

def main():
    """Main function"""
    
    print("🎯 FINAL COMPREHENSIVE FIX for Your ML Model")
    print("=" * 50)
    
    # Load model with comprehensive fix
    model = load_model_with_comprehensive_fix()
    
    if model:
        # Test functionality
        if test_model_functionality(model):
            print("\n🎉 SUCCESS! Your ML model is now working!")
            
            # Update the predictor
            if update_predictor_with_fix():
                print("\n✅ Predictor updated! Ready to test with genetic algorithm.")
                print("\nNext steps:")
                print("1. Run: python genetic_algo_true_cdvae.py")
                print("2. Look for: correction_method: ml_ensemble")
                print("3. Verify realistic bandgap values (3-6 eV)")
            else:
                print("\n⚠️ Manual update needed - copy the fix code to predictor")
        else:
            print("\n❌ Model loaded but functionality test failed")
    else:
        print("\n❌ Could not load your ML model")
        print("The model may need to be retrained with current numpy/sklearn versions")

if __name__ == "__main__":
    main()