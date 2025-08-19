#!/usr/bin/env python3
"""
Comprehensive test script to diagnose:
1. CDVAE usage and structure generation
2. Property prediction performance and caching
3. Model reloading issues
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_cdvae_generation():
    """Test CDVAE structure generation"""
    print("🧬 Testing CDVAE Structure Generation...")
    print("=" * 50)
    
    try:
        from generator.CDVAE.load_trained_model import TrainedCDVAELoader
        
        weights_path = r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\new_cdvae_weights.ckpt'
        scalers_dir = r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE'
        
        print(f"📁 Weights path: {weights_path}")
        print(f"📁 Scalers dir: {scalers_dir}")
        print(f"✅ Weights exist: {os.path.exists(weights_path)}")
        print(f"✅ Scalers dir exists: {os.path.exists(scalers_dir)}")
        
        # Create loader
        loader = TrainedCDVAELoader(weights_path, scalers_dir)
        print("✅ CDVAE loader created")
        
        # Load model
        start_time = time.time()
        loader.load_model()
        load_time = time.time() - start_time
        print(f"✅ CDVAE model loaded in {load_time:.2f}s")
        
        # Load scalers
        loader.load_scalers()
        print("✅ CDVAE scalers loaded")
        
        # Test structure generation
        print("\n🔬 Generating test structures...")
        start_time = time.time()
        structures = loader.generate_structures(3)
        gen_time = time.time() - start_time
        
        print(f"✅ Generated {len(structures)} structures in {gen_time:.2f}s")
        
        for i, struct in enumerate(structures):
            comp_str = ''.join(f'{elem}{count}' for elem, count in sorted(struct['composition'].items()))
            print(f"  Structure {i+1}: {comp_str}")
            print(f"    CIF path: {struct['cif_path']}")
            print(f"    CIF exists: {os.path.exists(struct['cif_path'])}")
        
        return True, structures
        
    except Exception as e:
        print(f"❌ CDVAE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_property_prediction_performance():
    """Test property prediction performance and caching"""
    print("\n⚡ Testing Property Prediction Performance...")
    print("=" * 50)
    
    # Create a test CIF file
    test_cif_content = """data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
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
O1 0.5 0.5 0.5
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
        f.write(test_cif_content)
        test_cif_path = f.name
    
    try:
        # Test 1: Import and create predictor
        print("📦 Testing predictor import and creation...")
        from genetic_algo.property_prediction_script import get_corrected_predictor
        
        start_time = time.time()
        predictor = get_corrected_predictor()
        creation_time = time.time() - start_time
        print(f"✅ Predictor created in {creation_time:.2f}s")
        
        # Test 2: First prediction (should load models)
        print("\n🔬 Testing first prediction (models should load)...")
        start_time = time.time()
        result1 = predictor.predict_single_cif(test_cif_path, verbose=True)
        first_pred_time = time.time() - start_time
        print(f"✅ First prediction completed in {first_pred_time:.2f}s")
        print(f"   Results: {result1}")
        
        # Test 3: Second prediction (should use cached models)
        print("\n⚡ Testing second prediction (should use cached models)...")
        start_time = time.time()
        result2 = predictor.predict_single_cif(test_cif_path, verbose=False)
        second_pred_time = time.time() - start_time
        print(f"✅ Second prediction completed in {second_pred_time:.2f}s")
        print(f"   Results: {result2}")
        
        # Test 4: Multiple predictions to check caching
        print("\n🚀 Testing multiple predictions for caching...")
        times = []
        for i in range(3):
            start_time = time.time()
            result = predictor.predict_single_cif(test_cif_path, verbose=False)
            pred_time = time.time() - start_time
            times.append(pred_time)
            print(f"   Prediction {i+1}: {pred_time:.3f}s")
        
        avg_time = sum(times) / len(times)
        print(f"✅ Average cached prediction time: {avg_time:.3f}s")
        
        # Performance analysis
        print(f"\n📊 Performance Analysis:")
        print(f"   First prediction (with model loading): {first_pred_time:.2f}s")
        print(f"   Subsequent predictions (cached): {avg_time:.3f}s")
        print(f"   Speedup factor: {first_pred_time/avg_time:.1f}x")
        
        if avg_time < 0.1:
            print("✅ EXCELLENT: Cached predictions are very fast!")
        elif avg_time < 0.5:
            print("✅ GOOD: Cached predictions are reasonably fast")
        else:
            print("⚠️  WARNING: Cached predictions are still slow - caching may not be working")
        
        return True, {
            'first_time': first_pred_time,
            'avg_cached_time': avg_time,
            'speedup': first_pred_time/avg_time
        }
        
    except Exception as e:
        print(f"❌ Property prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}
    
    finally:
        # Clean up test file
        try:
            os.unlink(test_cif_path)
        except:
            pass

def test_genetic_algorithm_integration():
    """Test how the genetic algorithm uses predictors"""
    print("\n🧬 Testing Genetic Algorithm Integration...")
    print("=" * 50)
    
    try:
        # Import the genetic algorithm
        from genetic_algo.FINAL_genetic_algo import GeneticAlgorithm
        
        # Check if it's using the cached predictor pattern
        print("🔍 Checking genetic algorithm predictor usage...")
        
        # Read the source code to check implementation
        import inspect
        ga_source = inspect.getsource(GeneticAlgorithm)
        
        if '_global_predictor' in ga_source:
            print("✅ Found _global_predictor pattern in genetic algorithm")
        else:
            print("❌ _global_predictor pattern NOT found in genetic algorithm")
        
        if 'get_corrected_predictor()' in ga_source:
            print("✅ Found get_corrected_predictor() usage")
        else:
            print("❌ get_corrected_predictor() usage NOT found")
        
        # Try to create a small GA instance
        print("\n🧪 Testing GA instantiation...")
        ga = GeneticAlgorithm(
            population_size=10,
            generations=1,
            tournament_size=3,
            elite_count=2,
            output_dir="test_output"
        )
        print("✅ Genetic algorithm instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Genetic algorithm integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔬 COMPREHENSIVE SYSTEM PERFORMANCE TEST")
    print("=" * 60)
    
    results = {}
    
    # Test 1: CDVAE
    cdvae_success, structures = test_cdvae_generation()
    results['cdvae'] = {'success': cdvae_success, 'structures': len(structures)}
    
    # Test 2: Property prediction performance
    pred_success, pred_stats = test_property_prediction_performance()
    results['prediction'] = {'success': pred_success, 'stats': pred_stats}
    
    # Test 3: GA integration
    ga_success = test_genetic_algorithm_integration()
    results['ga_integration'] = {'success': ga_success}
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    print(f"🧬 CDVAE Generation: {'✅ PASS' if cdvae_success else '❌ FAIL'}")
    if cdvae_success:
        print(f"   Generated {len(structures)} structures successfully")
    
    print(f"⚡ Property Prediction: {'✅ PASS' if pred_success else '❌ FAIL'}")
    if pred_success and pred_stats:
        print(f"   Speedup factor: {pred_stats.get('speedup', 0):.1f}x")
        print(f"   Cached prediction time: {pred_stats.get('avg_cached_time', 0):.3f}s")
    
    print(f"🧬 GA Integration: {'✅ PASS' if ga_success else '❌ FAIL'}")
    
    # Overall status
    all_pass = cdvae_success and pred_success and ga_success
    print(f"\n🎯 OVERALL STATUS: {'✅ ALL TESTS PASS' if all_pass else '❌ SOME TESTS FAILED'}")
    
    if not all_pass:
        print("\n🔧 ISSUES TO FIX:")
        if not cdvae_success:
            print("   - CDVAE structure generation is not working")
        if not pred_success:
            print("   - Property prediction system has issues")
        if not ga_success:
            print("   - Genetic algorithm integration has problems")
    
    return results

if __name__ == "__main__":
    main()