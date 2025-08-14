#!/usr/bin/env python3
"""
Debug script to test the prediction pipeline on generated CIF files
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fully_optimized_predictor import predict_single_cif_fully_optimized

def test_prediction_on_cif():
    """Test prediction on a generated CIF file"""
    
    # Test with one of the generated CIF files
    cif_path = "pareto_electrolyte_ga_results/cifs/gen0_Al1Br1Ge1La1Li1_2822065719632.cif"
    
    if not os.path.exists(cif_path):
        print(f"❌ CIF file not found: {cif_path}")
        return
    
    print(f"🔍 Testing prediction on: {cif_path}")
    print(f"📁 File exists: {os.path.exists(cif_path)}")
    print(f"📏 File size: {os.path.getsize(cif_path)} bytes")
    
    # Test prediction with verbose output
    try:
        results = predict_single_cif_fully_optimized(cif_path, verbose=True)
        
        print("\n📊 Prediction Results:")
        for key, value in results.items():
            if key != 'prediction_status':
                print(f"  {key}: {value}")
        
        print("\n🔧 Prediction Status:")
        for key, status in results['prediction_status'].items():
            print(f"  {key}: {status}")
            
    except Exception as e:
        print(f"❌ Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction_on_cif()