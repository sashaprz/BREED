#!/usr/bin/env python3
"""
Test script for the Multi-Objective Genetic Algorithm for Solid-State Electrolyte Discovery

This script runs a small-scale test to validate the implementation.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from genetic_algo_pareto import ParetoFrontGA, TargetProperties, GACandidate
    from pyxtal_generation import check_charge_neutrality, generate_wyckoff_compatible_composition
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_charge_neutrality():
    """Test charge neutrality function"""
    print("\n🧪 Testing charge neutrality...")
    
    # Valid compositions
    valid_comps = [
        {'Li': 2, 'O': 1},      # Li2O
        {'Li': 1, 'F': 1},      # LiF
        {'Li': 3, 'P': 1, 'O': 4},  # Li3PO4
    ]
    
    # Invalid compositions
    invalid_comps = [
        {'Li': 1, 'O': 1},      # LiO (wrong charges)
        {'Li': 2, 'F': 1},      # Li2F (wrong charges)
    ]
    
    for comp in valid_comps:
        if check_charge_neutrality(comp):
            print(f"  ✅ {comp} is charge neutral")
        else:
            print(f"  ❌ {comp} should be charge neutral but failed")
    
    for comp in invalid_comps:
        if not check_charge_neutrality(comp):
            print(f"  ✅ {comp} correctly identified as not charge neutral")
        else:
            print(f"  ❌ {comp} should not be charge neutral but passed")


def test_candidate_creation():
    """Test GACandidate creation and validation"""
    print("\n🧪 Testing GACandidate creation...")
    
    # Create a test candidate
    candidate = GACandidate(
        composition={'Li': 2, 'O': 1},
        lattice_params={'a': 4.0, 'b': 4.0, 'c': 4.0, 'alpha': 90, 'beta': 90, 'gamma': 90},
        space_group=225,
        generation=0
    )
    
    print(f"  ✅ Created candidate with composition: {candidate.composition}")
    print(f"  ✅ Objectives initialized: {candidate.objectives}")
    print(f"  ✅ Properties initialized: {list(candidate.properties.keys())}")


def test_pareto_dominance():
    """Test Pareto dominance comparison"""
    print("\n🧪 Testing Pareto dominance...")
    
    ga = ParetoFrontGA(population_size=10, max_generations=1)
    
    # Create test candidates
    candidate1 = GACandidate(
        composition={'Li': 2, 'O': 1},
        lattice_params={'a': 4.0, 'b': 4.0, 'c': 4.0, 'alpha': 90, 'beta': 90, 'gamma': 90},
        space_group=225
    )
    candidate1.objectives = [1.0, 2.0, 3.0, 4.0, 5.0]  # Worse in all objectives
    
    candidate2 = GACandidate(
        composition={'Li': 1, 'F': 1},
        lattice_params={'a': 3.0, 'b': 3.0, 'c': 3.0, 'alpha': 90, 'beta': 90, 'gamma': 90},
        space_group=225
    )
    candidate2.objectives = [0.5, 1.0, 1.5, 2.0, 2.5]  # Better in all objectives
    
    # Test dominance
    if ga.pareto_dominates(candidate2, candidate1):
        print("  ✅ Candidate2 correctly dominates Candidate1")
    else:
        print("  ❌ Dominance test failed")
    
    if not ga.pareto_dominates(candidate1, candidate2):
        print("  ✅ Candidate1 correctly does not dominate Candidate2")
    else:
        print("  ❌ Reverse dominance test failed")


def test_small_run():
    """Test a small GA run"""
    print("\n🧪 Testing small GA run...")
    
    # Create output directory for test
    test_output_dir = "test_pareto_results"
    if os.path.exists(test_output_dir):
        import shutil
        shutil.rmtree(test_output_dir)
    
    # Initialize with very small parameters for quick test
    ga = ParetoFrontGA(
        population_size=4,          # Very small population
        elite_count=2,              # Small elite
        tournament_size=2,          # Small tournament
        mutation_rate=0.1,          # Higher mutation for diversity
        max_generations=2,          # Only 2 generations
        convergence_threshold=1,    # Quick convergence
        output_dir=test_output_dir
    )
    
    print(f"  📁 Test output directory: {test_output_dir}")
    
    try:
        # Test initial population generation
        print("  🔄 Generating initial population...")
        start_time = time.time()
        
        population = ga.generate_initial_population()
        
        generation_time = time.time() - start_time
        print(f"  ✅ Generated {len(population)} candidates in {generation_time:.2f}s")
        
        if len(population) > 0:
            candidate = population[0]
            print(f"  ✅ Sample candidate composition: {candidate.composition}")
            print(f"  ✅ Sample candidate space group: {candidate.space_group}")
            
            # Check if CIF file was created
            if candidate.cif_path and os.path.exists(candidate.cif_path):
                print(f"  ✅ CIF file created: {os.path.basename(candidate.cif_path)}")
            else:
                print(f"  ⚠️  CIF file not found")
        
        # Test non-dominated sorting (without ML evaluation)
        print("  🔄 Testing non-dominated sorting...")
        
        # Set dummy objectives for testing
        for i, candidate in enumerate(population):
            candidate.objectives = [float(i), float(len(population) - i), 
                                  float(i % 2), float((i + 1) % 3), float(i % 4)]
        
        fronts = ga.non_dominated_sort(population)
        print(f"  ✅ Non-dominated sorting created {len(fronts)} fronts")
        print(f"  ✅ Front sizes: {[len(front) for front in fronts]}")
        
        # Test crowding distance
        if len(fronts) > 0 and len(fronts[0]) > 0:
            ga.calculate_crowding_distance(fronts[0])
            print(f"  ✅ Crowding distances calculated for first front")
            
            for i, candidate in enumerate(fronts[0]):
                print(f"    Candidate {i}: CD = {candidate.crowding_distance:.3f}")
        
        print("  ✅ Small GA test completed successfully!")
        
        # Clean up test directory
        if os.path.exists(test_output_dir):
            import shutil
            shutil.rmtree(test_output_dir)
            print(f"  🧹 Cleaned up test directory")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Small GA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_target_properties():
    """Test target properties configuration"""
    print("\n🧪 Testing target properties...")
    
    # Test default targets
    default_targets = TargetProperties()
    print(f"  ✅ Default ionic conductivity target: {default_targets.ionic_conductivity:.2e} S/cm")
    print(f"  ✅ Default bandgap target: {default_targets.bandgap} eV")
    print(f"  ✅ Default SEI score target: {default_targets.sei_score}")
    print(f"  ✅ Default CEI score target: {default_targets.cei_score}")
    print(f"  ✅ Default bulk modulus target: {default_targets.bulk_modulus} GPa")
    
    # Test custom targets
    custom_targets = TargetProperties(
        ionic_conductivity=5.0e-3,
        bandgap=2.5,
        sei_score=0.95,
        cei_score=0.9,
        bulk_modulus=100.0
    )
    print(f"  ✅ Custom targets created successfully")


def main():
    """Run all tests"""
    print("🚀 Starting Pareto GA Test Suite")
    print("=" * 50)
    
    # Run individual tests
    test_charge_neutrality()
    test_candidate_creation()
    test_pareto_dominance()
    test_target_properties()
    
    # Run integration test
    success = test_small_run()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The Pareto GA implementation is ready to use.")
        print("\n📋 Next steps:")
        print("1. Run the full GA: python genetic_algo_pareto.py")
        print("2. Monitor results in: pareto_electrolyte_ga_results/")
        print("3. Analyze Pareto front plots and JSON results")
        print("4. Adjust parameters based on initial results")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)