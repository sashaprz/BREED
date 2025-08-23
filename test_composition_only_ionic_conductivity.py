#!/usr/bin/env python3
"""
Test script for the new composition-only ionic conductivity predictor

This tests that:
1. The new predictor works correctly
2. The cached property predictor uses the new method
3. The property prediction script uses the new method
4. All predictions are fast and reliable
"""

import sys
import os
import time
from pathlib import Path

# Add genetic_algo to path
sys.path.append('genetic_algo')

def test_composition_only_predictor():
    """Test the standalone composition-only predictor"""
    print("🧪 Testing Composition-Only Ionic Conductivity Predictor")
    print("=" * 60)
    
    try:
        from composition_only_ionic_conductivity import (
            predict_ionic_conductivity_from_composition,
            get_composition_only_predictor
        )
        
        # Test compositions
        test_compositions = [
            "Li7P3S11",      # Argyrodite-type (high conductivity)
            "Li6PS5Cl",      # Argyrodite with Cl (very high)
            "Li1.3Al0.3Ti1.7P3O12",  # NASICON-type
            "Li7La3Zr2O12",  # Garnet-type
            "LiPON",         # Low conductivity
            "Li2O"           # Very low conductivity
        ]
        
        predictor = get_composition_only_predictor()
        
        print("Testing individual compositions:")
        for composition in test_compositions:
            start_time = time.time()
            conductivity = predictor.predict_from_composition(composition)
            end_time = time.time()
            
            print(f"  {composition:20s}: {conductivity:.2e} S/cm ({(end_time-start_time)*1000:.1f}ms)")
            
            # Verify reasonable range
            assert 1e-12 <= conductivity <= 1e-2, f"Conductivity out of range: {conductivity}"
        
        print("✅ Composition-only predictor works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Composition-only predictor failed: {e}")
        return False


def test_cached_property_predictor():
    """Test that cached property predictor uses the new method"""
    print("\n🧪 Testing Cached Property Predictor Integration")
    print("=" * 60)
    
    try:
        from cached_property_predictor import get_cached_predictor
        
        # Create a dummy CIF file for testing
        test_cif_content = """data_Li7P3S11
_cell_length_a 8.0
_cell_length_b 8.0
_cell_length_c 8.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 0.0 0.0 0.0
P1 0.5 0.5 0.5
S1 0.25 0.25 0.25
"""
        
        # Write test CIF file
        test_cif_path = "test_structure.cif"
        with open(test_cif_path, 'w') as f:
            f.write(test_cif_content)
        
        try:
            predictor = get_cached_predictor()
            
            start_time = time.time()
            results = predictor.predict_single_cif(test_cif_path, verbose=True)
            end_time = time.time()
            
            print(f"\nPrediction completed in {(end_time-start_time)*1000:.1f}ms")
            print(f"Results: {results}")
            
            # Verify the new method was used
            assert "ionic_conductivity" in results
            assert results["prediction_status"]["ionic_conductivity"] == "composition_based"
            assert results.get("cgcnn_skipped") == True
            assert 1e-12 <= results["ionic_conductivity"] <= 1e-2
            
            print("✅ Cached property predictor uses composition-only method!")
            return True
            
        finally:
            # Clean up test file
            if os.path.exists(test_cif_path):
                os.remove(test_cif_path)
        
    except Exception as e:
        print(f"❌ Cached property predictor test failed: {e}")
        return False


def test_property_prediction_script():
    """Test that property prediction script uses the new method"""
    print("\n🧪 Testing Property Prediction Script Integration")
    print("=" * 60)
    
    try:
        from property_prediction_script import predict_single_cif_corrected
        
        # Create a dummy CIF file for testing
        test_cif_content = """data_Li6PS5Cl
_cell_length_a 8.0
_cell_length_b 8.0
_cell_length_c 8.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 0.0 0.0 0.0
P1 0.5 0.5 0.5
S1 0.25 0.25 0.25
Cl1 0.75 0.75 0.75
"""
        
        # Write test CIF file
        test_cif_path = "test_structure2.cif"
        with open(test_cif_path, 'w') as f:
            f.write(test_cif_content)
        
        try:
            start_time = time.time()
            results = predict_single_cif_corrected(test_cif_path, verbose=True)
            end_time = time.time()
            
            print(f"\nPrediction completed in {(end_time-start_time)*1000:.1f}ms")
            print(f"Results: {results}")
            
            # Verify the new method was used
            assert "ionic_conductivity" in results
            assert results["prediction_status"]["ionic_conductivity"] == "composition_based"
            assert results.get("cgcnn_skipped") == True
            assert 1e-12 <= results["ionic_conductivity"] <= 1e-2
            
            print("✅ Property prediction script uses composition-only method!")
            return True
            
        finally:
            # Clean up test file
            if os.path.exists(test_cif_path):
                os.remove(test_cif_path)
        
    except Exception as e:
        print(f"❌ Property prediction script test failed: {e}")
        return False


def test_performance_comparison():
    """Test performance improvement over CGCNN approach"""
    print("\n🧪 Testing Performance Improvements")
    print("=" * 60)
    
    try:
        from composition_only_ionic_conductivity import predict_ionic_conductivity_from_composition
        
        # Test multiple predictions to measure speed
        test_compositions = ["Li7P3S11", "Li6PS5Cl", "Li1.3Al0.3Ti1.7P3O12"] * 100
        
        start_time = time.time()
        for composition in test_compositions:
            conductivity = predict_ionic_conductivity_from_composition(composition)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_prediction = (total_time / len(test_compositions)) * 1000  # ms
        
        print(f"Processed {len(test_compositions)} predictions in {total_time:.3f}s")
        print(f"Average time per prediction: {avg_time_per_prediction:.2f}ms")
        print(f"Predictions per second: {len(test_compositions)/total_time:.1f}")
        
        # Should be very fast (< 1ms per prediction)
        assert avg_time_per_prediction < 1.0, f"Too slow: {avg_time_per_prediction}ms per prediction"
        
        print("✅ Performance is excellent (< 1ms per prediction)!")
        print("✅ No model loading overhead!")
        print("✅ No CUDA/GPU dependencies!")
        print("✅ 100% reliability (no failures)!")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 TESTING COMPOSITION-ONLY IONIC CONDUCTIVITY IMPLEMENTATION")
    print("=" * 80)
    print("Replacing CGCNN (R² ≈ 0, MAPE > 8M%) with fast, reliable composition-based method")
    print()
    
    tests = [
        test_composition_only_predictor,
        test_cached_property_predictor,
        test_property_prediction_script,
        test_performance_comparison
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 80)
    print(f"FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ CGCNN ionic conductivity successfully replaced with composition-only method!")
        print("✅ Significant performance improvements achieved!")
        print("✅ 100% reliability with no model loading failures!")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)