#!/usr/bin/env python3
"""
Inspect the structure of the saved model file
"""

import pickle

def inspect_model_file(filename):
    """Inspect the contents of a pickle file"""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        print(f"📁 Inspecting {filename}")
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print("Dictionary keys:", list(data.keys()))
            for key, value in data.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    Shape: {value.shape}")
                elif hasattr(value, '__len__') and not isinstance(value, str):
                    print(f"    Length: {len(value)}")
        else:
            print("Direct object type:", type(data))
            if hasattr(data, '__dict__'):
                print("Attributes:", list(data.__dict__.keys()))
        
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

if __name__ == "__main__":
    # Check all available model files
    model_files = [
        'improved_bandgap_model.pkl',
        'bandgap_correction_model.pkl',
        'bandgap_correction_model_v4.pkl'
    ]
    
    for filename in model_files:
        print("=" * 60)
        data = inspect_model_file(filename)
        print()