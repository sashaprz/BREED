#!/usr/bin/env python3
"""
Aggressive pickle fix - intercept and redirect problematic classes during loading
"""

import os
import sys
import pickle
import numpy as np
import io

def create_aggressive_pickle_fix():
    """Create aggressive pickle compatibility by monkey-patching pickle itself"""
    
    print("🔧 Creating aggressive pickle compatibility fix...")
    
    # Store original pickle functions
    original_load = pickle.load
    original_loads = pickle.loads
    
    def patched_load(file, **kwargs):
        """Patched pickle.load with class redirection"""
        try:
            return original_load(file, **kwargs)
        except Exception as e:
            if "PCG64" in str(e) or "BitGenerator" in str(e):
                print("   🔄 PCG64 error detected, applying aggressive fix...")
                
                # Read the pickle data
                file.seek(0)
                data = file.read()
                
                # Replace problematic class references in the pickle bytecode
                # This is a binary replacement approach
                data = data.replace(b'numpy.random._pcg64\nPCG64', b'numpy.random\nPCG64')
                data = data.replace(b'numpy._core', b'numpy.core')
                
                # Try loading the modified data
                return pickle.loads(data)
            else:
                raise e
    
    def patched_loads(data, **kwargs):
        """Patched pickle.loads with class redirection"""
        try:
            return original_loads(data, **kwargs)
        except Exception as e:
            if "PCG64" in str(e) or "BitGenerator" in str(e):
                print("   🔄 PCG64 error in loads, applying fix...")
                
                # Replace problematic class references
                if isinstance(data, bytes):
                    data = data.replace(b'numpy.random._pcg64\nPCG64', b'numpy.random\nPCG64')
                    data = data.replace(b'numpy._core', b'numpy.core')
                
                return original_loads(data, **kwargs)
            else:
                raise e
    
    # Monkey patch pickle
    pickle.load = patched_load
    pickle.loads = patched_loads
    
    print("   ✅ Pickle functions monkey-patched")
    
    return True

def setup_complete_environment():
    """Setup complete compatibility environment"""
    
    print("🔧 Setting up complete compatibility environment...")
    
    # 1. Fix numpy compatibility
    import numpy as np
    
    if not hasattr(np, '_core'):
        import numpy.core
        np._core = numpy.core
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
    
    # 2. Fix random number generators
    import numpy.random
    
    # Create PCG64 in the expected location
    if hasattr(numpy.random, '_pcg64') and hasattr(numpy.random._pcg64, 'PCG64'):
        # Register in all possible locations
        numpy.random.PCG64 = numpy.random._pcg64.PCG64
        sys.modules['numpy.random.PCG64'] = numpy.random._pcg64.PCG64
        sys.modules['numpy.random._pcg64'] = numpy.random._pcg64
        sys.modules['numpy.random._pcg64.PCG64'] = numpy.random._pcg64.PCG64
        
        # Also create a direct reference
        globals()['PCG64'] = numpy.random._pcg64.PCG64
    
    # 3. Import sklearn
    import sklearn
    import sklearn.ensemble
    import sklearn.preprocessing
    
    print("   ✅ Complete environment setup")
    
    return True

def load_with_custom_unpickler():
    """Load using a completely custom unpickler"""
    
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    
    print("🔄 Loading with custom unpickler...")
    
    class AggressiveUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            print(f"   🔍 Looking for: {module}.{name}")
            
            # Handle numpy._core remapping
            if module.startswith('numpy._core'):
                new_module = module.replace('numpy._core', 'numpy.core')
                print(f"      ↳ Remapping to: {new_module}.{name}")
                try:
                    mod = __import__(new_module, fromlist=[name])
                    return getattr(mod, name)
                except (ImportError, AttributeError):
                    pass
            
            # Handle PCG64 specifically
            if module == 'numpy.random._pcg64' and name == 'PCG64':
                print(f"      ↳ Redirecting PCG64 to numpy.random._pcg64.PCG64")
                return numpy.random._pcg64.PCG64
            
            # Handle other numpy.random classes
            if module.startswith('numpy.random') and hasattr(numpy.random, name):
                print(f"      ↳ Found in numpy.random: {name}")
                return getattr(numpy.random, name)
            
            # Try the original approach
            try:
                return super().find_class(module, name)
            except Exception as e:
                print(f"      ❌ Failed to find {module}.{name}: {e}")
                
                # Last resort: try to find it anywhere in numpy
                if module.startswith('numpy'):
                    for submodule_name in ['core', 'random', 'linalg', 'fft']:
                        try:
                            submodule = getattr(numpy, submodule_name)
                            if hasattr(submodule, name):
                                print(f"      ✅ Found in numpy.{submodule_name}: {name}")
                                return getattr(submodule, name)
                        except:
                            continue
                
                raise e
    
    try:
        with open(model_path, 'rb') as f:
            unpickler = AggressiveUnpickler(f)
            model = unpickler.load()
        
        print("✅ Custom unpickler successful!")
        return model
        
    except Exception as e:
        print(f"❌ Custom unpickler failed: {e}")
        return None

def test_model_after_loading(model):
    """Test the model after successful loading"""
    
    if model is None:
        return False
    
    print("\n🧪 Testing loaded model...")
    
    try:
        import pandas as pd
        
        print(f"   📋 Model type: {type(model)}")
        if isinstance(model, dict):
            print(f"   📋 Model keys: {list(model.keys())}")
            
            # Test prediction
            rf_model = model.get('rf_model')
            gb_model = model.get('gb_model')
            scaler = model.get('scaler')
            weights = model.get('ensemble_weights')
            
            if all([rf_model, gb_model, scaler, weights]):
                # Create test data
                test_data = pd.DataFrame({
                    'pbe_bandgap': [0.005],
                    'n_elements': [3], 'total_atoms': [12], 'avg_electronegativity': [2.5], 'avg_atomic_mass': [45.0],
                    'has_O': [1], 'has_N': [0], 'has_C': [0], 'has_Si': [0],
                    'has_Al': [0], 'has_Ti': [0], 'has_Fe': [0],
                    'pbe_squared': [0.005 ** 2], 'pbe_sqrt': [np.sqrt(0.005)], 'en_pbe_product': [2.5 * 0.005]
                })
                
                # Make prediction
                X_scaled = scaler.transform(test_data)
                rf_pred = rf_model.predict(X_scaled)[0]
                gb_pred = gb_model.predict(X_scaled)[0]
                final_pred = weights[0] * rf_pred + weights[1] * gb_pred
                
                print(f"   🎯 Test prediction: 0.005 eV → {final_pred:.4f} eV ({final_pred/0.005:.1f}x correction)")
                print("   ✅ Model is working correctly!")
                return True
            else:
                print("   ❌ Missing model components")
                return False
        else:
            print(f"   ❌ Unexpected model type: {type(model)}")
            return False
            
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        return False

def main():
    """Main function - try everything to load your model"""
    
    print("🚀 AGGRESSIVE APPROACH: Loading Your ML Model")
    print("=" * 50)
    
    # Setup environment
    setup_complete_environment()
    
    # Apply aggressive pickle fix
    create_aggressive_pickle_fix()
    
    # Try loading with custom unpickler
    model = load_with_custom_unpickler()
    
    if model:
        if test_model_after_loading(model):
            print("\n🎉 SUCCESS! Your ML model is now working!")
            print("Now we can integrate it into the predictor.")
            return model
        else:
            print("\n⚠️ Model loaded but testing failed")
            return None
    else:
        print("\n❌ Still could not load your model")
        print("\nLet's try one more approach - manual binary editing...")
        
        # Last resort: try to manually fix the pickle file
        try_binary_fix()
        return None

def try_binary_fix():
    """Last resort: try to fix the pickle file by editing its binary content"""
    
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
    fixed_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\genetic_algo\fixed_bandgap_model.pkl"
    
    print("\n🔧 Attempting binary fix of pickle file...")
    
    try:
        with open(model_path, 'rb') as f:
            data = f.read()
        
        print(f"   📏 Original file size: {len(data)} bytes")
        
        # Replace problematic byte sequences
        replacements = [
            (b'numpy.random._pcg64\nPCG64', b'numpy.random\nPCG64'),
            (b'numpy._core', b'numpy.core'),
            (b'_pcg64\nPCG64', b'random\nPCG64'),
        ]
        
        modified = False
        for old, new in replacements:
            if old in data:
                data = data.replace(old, new)
                modified = True
                print(f"   ✅ Replaced: {old} → {new}")
        
        if modified:
            # Save fixed version
            with open(fixed_path, 'wb') as f:
                f.write(data)
            
            print(f"   💾 Fixed model saved to: {fixed_path}")
            print("   🧪 Try loading the fixed version...")
            
            # Try loading the fixed version
            with open(fixed_path, 'rb') as f:
                model = pickle.load(f)
            
            print("   ✅ Fixed model loaded successfully!")
            return model
        else:
            print("   ⚠️ No problematic sequences found to replace")
            return None
            
    except Exception as e:
        print(f"   ❌ Binary fix failed: {e}")
        return None

if __name__ == "__main__":
    main()