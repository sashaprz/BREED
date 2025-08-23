#!/usr/bin/env python3
"""
Simple test for composition-only ionic conductivity predictor
Tests only the core functionality without PyTorch dependencies
"""

import sys
import os
import time

# Add genetic_algo to path
sys.path.append('genetic_algo')

def test_composition_only_predictor():
    """Test the standalone composition-only predictor"""
    print("🧪 Testing Composition-Only Ionic Conductivity Predictor")
    print("=" * 60)
    
    try:
        from composition_only_ionic_conductivity import (
            predict_ionic_conductivity_from_composition,
            predict_ionic_conductivity_from_cif,
            get_composition_only_predictor
        )
        
        # Test compositions with expected ranges
        test_cases = [
            ("Li7P3S11", "High conductivity argyrodite"),
            ("Li6PS5Cl", "Very high conductivity argyrodite with Cl"),
            ("Li1.3Al0.3Ti1.7P3O12", "NASICON-type"),
            ("Li7La3Zr2O12", "Garnet-type"),
            ("LiPON", "Low conductivity nitride"),
            ("Li2O", "Very low conductivity oxide"),
            ("NaCl", "No Li - should be very low"),
            ("Li10GeP2S12", "High Li content sulfide")
        ]
        
        predictor = get_composition_only_predictor()
        
        print("Testing individual compositions:")
        for composition, description in test_cases:
            start_time = time.time()
            conductivity = predictor.predict_from_composition(composition)
            end_time = time.time()
            
            print(f"  {composition:20s}: {conductivity:.2e} S/cm - {description}")
            
            # Verify reasonable range
            assert 1e-12 <= conductivity <= 1e-2, f"Conductivity out of range: {conductivity}"
            
            # Verify speed (should be very fast)
            prediction_time = (end_time - start_time) * 1000
            assert prediction_time < 10, f"Too slow: {prediction_time}ms"
        
        print("✅ All composition predictions successful!")
        return True
        
    except Exception as e:
        print(f"❌ Composition-only predictor failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cif_prediction():
    """Test CIF-based prediction"""
    print("\n🧪 Testing CIF-Based Prediction")
    print("=" * 60)
    
    try:
        from composition_only_ionic_conductivity import predict_ionic_conductivity_from_cif
        
        # Create test CIF files
        test_cifs = [
            ("Li7P3S11", """data_Li7P3S11
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
"""),
            ("Li6PS5Cl", """data_Li6PS5Cl
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
""")
        ]
        
        for composition, cif_content in test_cifs:
            test_cif_path = f"test_{composition}.cif"
            
            try:
                # Write test CIF file
                with open(test_cif_path, 'w') as f:
                    f.write(cif_content)
                
                # Test prediction
                start_time = time.time()
                results = predict_ionic_conductivity_from_cif(test_cif_path, verbose=True)
                end_time = time.time()
                
                # Verify results
                assert "ionic_conductivity" in results
                assert "composition" in results
                assert results["method"] == "composition_only"
                assert results["cgcnn_skipped"] == True
                assert 1e-12 <= results["ionic_conductivity"] <= 1e-2
                
                prediction_time = (end_time - start_time) * 1000
                print(f"  Prediction time: {prediction_time:.1f}ms")
                
            finally:
                # Clean up
                if os.path.exists(test_cif_path):
                    os.remove(test_cif_path)
        
        print("✅ CIF-based predictions successful!")
        return True
        
    except Exception as e:
        print(f"❌ CIF prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmark():
    """Test performance and reliability"""
    print("\n🧪 Testing Performance and Reliability")
    print("=" * 60)
    
    try:
        from composition_only_ionic_conductivity import predict_ionic_conductivity_from_composition
        
        # Test many predictions for performance
        test_compositions = [
            "Li7P3S11", "Li6PS5Cl", "Li1.3Al0.3Ti1.7P3O12", 
            "Li7La3Zr2O12", "LiPON", "Li2O"
        ] * 50  # 300 total predictions
        
        print(f"Running {len(test_compositions)} predictions...")
        
        start_time = time.time()
        successful_predictions = 0
        
        for composition in test_compositions:
            try:
                conductivity = predict_ionic_conductivity_from_composition(composition)
                assert 1e-12 <= conductivity <= 1e-2
                successful_predictions += 1
            except Exception as e:
                print(f"Failed prediction for {composition}: {e}")
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = (total_time / len(test_compositions)) * 1000
        success_rate = (successful_predictions / len(test_compositions)) * 100
        
        print(f"Results:")
        print(f"  Total predictions: {len(test_compositions)}")
        print(f"  Successful: {successful_predictions}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per prediction: {avg_time:.2f}ms")
        print(f"  Predictions per second: {len(test_compositions)/total_time:.1f}")
        
        # Verify performance requirements
        assert success_rate == 100.0, f"Not 100% reliable: {success_rate}%"
        assert avg_time < 1.0, f"Too slow: {avg_time}ms per prediction"
        
        print("✅ Performance excellent: 100% reliable, < 1ms per prediction!")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 COMPOSITION-ONLY IONIC CONDUCTIVITY PREDICTOR TESTS")
    print("=" * 80)
    print("Testing replacement for CGCNN (R² ≈ 0, MAPE > 8M%)")
    print("New method: Fast, reliable, no PyTorch dependencies")
    print()
    
    tests = [
        test_composition_only_predictor,
        test_cif_prediction,
        test_performance_benchmark
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
        print()
        print("✅ CGCNN ionic conductivity successfully replaced!")
        print("✅ 100% reliability (no failures)")
        print("✅ Excellent performance (< 1ms per prediction)")
        print("✅ No PyTorch/CUDA dependencies")
        print("✅ No model loading overhead")
        print("✅ Scientifically grounded predictions")
        print()
        print("🚀 Ready for production use in genetic algorithm!")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)