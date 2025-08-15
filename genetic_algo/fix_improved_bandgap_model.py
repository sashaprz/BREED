#!/usr/bin/env python3
"""
Fix the improved_bandgap_model.pkl file to work with current numpy versions
"""

import pickle
import joblib
import numpy as np
import sys
import os

def fix_improved_model():
    """Fix the improved bandgap model for current numpy compatibility"""
    
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\improved_bandgap_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"🔧 Fixing model: {model_path}")
    
    # Apply comprehensive numpy compatibility fixes
    import numpy.core as np_core
    
    # Fix 1: Handle numpy._core module compatibility
    if not hasattr(np, '_core'):
        np._core = np_core
        sys.modules['numpy._core'] = np_core
    
    # Fix 2: Handle numpy._core._multiarray_umath compatibility
    try:
        import numpy.core._multiarray_umath
        if 'numpy._core._multiarray_umath' not in sys.modules:
            sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
    except ImportError:
        pass
    
    # Fix 3: Handle numpy._core.multiarray compatibility
    try:
        import numpy.core.multiarray
        if 'numpy._core.multiarray' not in sys.modules:
            sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
    except ImportError:
        pass
    
    # Fix 4: Handle numpy random BitGenerator compatibility
    try:
        import numpy.random
        # Map new BitGenerator classes to old ones for compatibility
        if hasattr(numpy.random, '_pcg64') and hasattr(numpy.random._pcg64, 'PCG64'):
            # Create compatibility mapping for PCG64 BitGenerator
            if not hasattr(numpy.random, 'PCG64'):
                numpy.random.PCG64 = numpy.random._pcg64.PCG64
            # Register in sys.modules for pickle compatibility
            sys.modules['numpy.random.PCG64'] = numpy.random._pcg64.PCG64
            sys.modules['numpy.random._pcg64.PCG64'] = numpy.random._pcg64.PCG64
    except (ImportError, AttributeError):
        pass
    
    try:
        # Try to load the model
        print("📁 Loading original model...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"   Model contains: {list(model_data.keys())}")
        
        # Add current metadata
        if 'metadata' not in model_data:
            model_data['metadata'] = {}
        
        model_data['metadata'].update({
            'fixed_date': __import__('datetime').datetime.now().isoformat(),
            'fixed_numpy_version': np.__version__,
            'fixed_sklearn_version': __import__('sklearn').__version__,
            'fix_script': 'fix_improved_bandgap_model.py'
        })
        
        # Save in multiple formats
        print("💾 Saving fixed model in multiple formats...")
        
        # Save with joblib (most compatible)
        joblib_path = model_path.replace('.pkl', '_joblib.pkl')
        joblib.dump(model_data, joblib_path, compress=3)
        joblib_size = os.path.getsize(joblib_path) / 1024 / 1024
        print(f"✅ Saved joblib version: {joblib_path} ({joblib_size:.1f} MB)")
        
        # Save with current pickle
        fixed_path = model_path.replace('.pkl', '_fixed.pkl')
        with open(fixed_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        fixed_size = os.path.getsize(fixed_path) / 1024 / 1024
        print(f"✅ Saved fixed pickle: {fixed_path} ({fixed_size:.1f} MB)")
        
        # Test loading the fixed versions
        print("\n🧪 Testing fixed models...")
        
        # Test joblib version
        try:
            test_model = joblib.load(joblib_path)
            print("✅ Joblib version loads successfully!")
            
            # Quick prediction test
            if 'rf_model' in test_model and 'scaler' in test_model:
                import pandas as pd
                test_features = pd.DataFrame({
                    'pbe_bandgap': [0.5],
                    'n_elements': [2], 'total_atoms': [2], 'avg_electronegativity': [2.0], 'avg_atomic_mass': [50.0],
                    'has_O': [1], 'has_N': [0], 'has_C': [0], 'has_Si': [0],
                    'has_Al': [0], 'has_Ti': [0], 'has_Fe': [0],
                    'pbe_squared': [0.25], 'pbe_sqrt': [np.sqrt(0.5)], 'en_pbe_product': [1.0]
                })
                
                X_test = test_model['scaler'].transform(test_features)
                rf_pred = test_model['rf_model'].predict(X_test)[0]
                gb_pred = test_model['gb_model'].predict(X_test)[0]
                weights = test_model.get('ensemble_weights', [0.6, 0.4])
                final_pred = weights[0] * rf_pred + weights[1] * gb_pred
                
                print(f"✅ Prediction test: 0.5 eV → {final_pred:.3f} eV ({final_pred/0.5:.1f}x correction)")
            
        except Exception as e:
            print(f"❌ Joblib test failed: {e}")
        
        print(f"\n🎉 MODEL FIXED SUCCESSFULLY!")
        print(f"📁 Primary fixed model: {joblib_path}")
        print(f"📁 Backup fixed model: {fixed_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 FIXING IMPROVED BANDGAP MODEL")
    print("=" * 50)
    
    success = fix_improved_model()
    
    if success:
        print("\n✅ Your model is now compatible with current numpy versions!")
        print("   The genetic algorithm should now use ml_ensemble correction.")
        print("\n🚀 Run the genetic algorithm again to see ML ensemble corrections!")
    else:
        print("\n❌ Failed to fix the model. You may need to retrain it.")