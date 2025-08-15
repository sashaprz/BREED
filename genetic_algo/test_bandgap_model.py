#!/usr/bin/env python3
"""
Test script to load the bandgap correction model and identify the exact issue
"""

import os
import sys

print("Testing bandgap correction model loading...")
print(f"Python version: {sys.version}")

# Test 1: Check if file exists
model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\bandgap_correction_model.pkl"
print(f"Model file exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print(f"Model file size: {os.path.getsize(model_path)} bytes")
    
    # Test 2: Try different loading methods
    print("\n=== Testing different loading methods ===")
    
    # Method 1: Standard pickle
    try:
        import pickle
        print("✅ pickle imported successfully")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ SUCCESS: Standard pickle loading worked!")
        print(f"Model keys: {list(model.keys()) if isinstance(model, dict) else type(model)}")
    except Exception as e:
        print(f"❌ Standard pickle failed: {e}")
        
        # Method 2: Try joblib
        try:
            import joblib
            print("✅ joblib imported successfully")
            model = joblib.load(model_path)
            print("✅ SUCCESS: joblib loading worked!")
            print(f"Model keys: {list(model.keys()) if isinstance(model, dict) else type(model)}")
        except Exception as e2:
            print(f"❌ joblib failed: {e2}")
            
            # Method 3: Try with different pickle protocols
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    # Try loading with different protocols
                    for protocol in [None, 0, 1, 2, 3, 4, 5]:
                        try:
                            f.seek(0)
                            if protocol is None:
                                model = pickle.load(f)
                            else:
                                model = pickle.load(f)  # Protocol is for saving, not loading
                            print(f"✅ SUCCESS: pickle loading worked!")
                            print(f"Model keys: {list(model.keys()) if isinstance(model, dict) else type(model)}")
                            break
                        except Exception as e3:
                            continue
                    else:
                        print(f"❌ All pickle protocols failed")
            except Exception as e3:
                print(f"❌ Pickle protocol test failed: {e3}")

# Test 3: Check numpy and pandas
print("\n=== Testing dependencies ===")
try:
    import numpy as np
    print(f"✅ numpy {np.__version__} imported successfully")
except Exception as e:
    print(f"❌ numpy failed: {e}")

try:
    import pandas as pd
    print(f"✅ pandas {pd.__version__} imported successfully")
except Exception as e:
    print(f"❌ pandas failed: {e}")

try:
    import sklearn
    print(f"✅ sklearn {sklearn.__version__} imported successfully")
except Exception as e:
    print(f"❌ sklearn failed: {e}")

print("\n=== Test completed ===")