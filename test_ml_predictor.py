#!/usr/bin/env python3
"""
Test the ML predictor directly to isolate the issue
"""

import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_predictor():
    print("🔬 Testing ML Predictor Directly")
    print("=" * 50)
    
    # Import the same predictor used in TRUE_genetic_algo.py
    try:
        from genetic_algo.cached_property_predictor import get_cached_predictor
        predictor = get_cached_predictor()
        print("✅ Cached property predictor imported successfully")
        
        def predict_single_cif(cif_path, verbose=False):
            return predictor.predict_single_cif(cif_path, verbose=verbose)
            
    except ImportError:
        try:
            from genetic_algo.property_prediction_script import get_corrected_predictor
            predictor = get_corrected_predictor()
            print("✅ Corrected property predictor imported successfully")
            
            def predict_single_cif(cif_path, verbose=False):
                return predictor.predict_single_cif(cif_path, verbose=verbose)
                
        except ImportError:
            print("❌ Could not import any predictor")
            return
    
    # Test with a known good CIF file
    test_cif_paths = [
        # Check if there are any CIF files from previous runs
        "true_genetic_algo_results/cifs",
        # Or use any existing CIF files in the project
        ".",
    ]
    
    test_cif = None
    for cif_dir in test_cif_paths:
        cif_path = Path(cif_dir)
        if cif_path.exists():
            cif_files = list(cif_path.glob("*.cif"))
            if cif_files:
                test_cif = cif_files[0]
                break
    
    if test_cif:
        print(f"\n🧪 Testing with CIF file: {test_cif}")
        try:
            results = predict_single_cif(str(test_cif), verbose=True)
            print(f"\n📊 Prediction Results:")
            for key, value in results.items():
                if key == 'ionic_conductivity':
                    print(f"   {key}: {value:.2e}")
                elif isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
                    
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ No CIF files found to test with")
        
        # Create a simple test CIF file
        print("\n🔧 Creating a simple test CIF file...")
        test_cif_content = """data_test
_cell_length_a    10.0
_cell_length_b    10.0
_cell_length_c    10.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
_space_group_IT_number 1

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 Li 0.0 0.0 0.0
O1  O  0.5 0.5 0.5
"""
        
        test_cif_path = Path("test_structure.cif")
        with open(test_cif_path, 'w') as f:
            f.write(test_cif_content)
        
        print(f"✅ Created test CIF: {test_cif_path}")
        
        try:
            results = predict_single_cif(str(test_cif_path), verbose=True)
            print(f"\n📊 Test Prediction Results:")
            for key, value in results.items():
                if key == 'ionic_conductivity':
                    print(f"   {key}: {value:.2e}")
                elif isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
                    
        except Exception as e:
            print(f"❌ Test prediction failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_predictor()