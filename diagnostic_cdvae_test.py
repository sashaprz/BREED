#!/usr/bin/env python3
"""
Comprehensive CDVAE Diagnostic Test Script
Tests all components of the CDVAE integration with the genetic algorithm.
"""

import os
import sys
import traceback
import torch
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section"""
    print(f"\n{'─'*40}")
    print(f"📋 {title}")
    print(f"{'─'*40}")

def test_environment():
    """Test the basic environment setup"""
    print_header("ENVIRONMENT DIAGNOSTICS")
    
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📁 Script location: {os.path.abspath(__file__)}")
    
    # Check if we're in the right directory
    expected_files = [
        'generator/CDVAE/new_cdvae_weights.ckpt',
        'generator/CDVAE/load_trained_model.py',
        'genetic_algo/FINAL_genetic_algo.py'
    ]
    
    print(f"\n📂 Checking required files:")
    all_files_exist = True
    for file_path in expected_files:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path}")
        if not exists:
            all_files_exist = False
    
    return all_files_exist

def test_cdvae_import():
    """Test CDVAE import"""
    print_header("CDVAE IMPORT TEST")
    
    try:
        from generator.CDVAE.load_trained_model import TrainedCDVAELoader
        print("✅ TrainedCDVAELoader imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import TrainedCDVAELoader: {e}")
        traceback.print_exc()
        return False

def test_cdvae_loading():
    """Test CDVAE model loading"""
    print_header("CDVAE MODEL LOADING TEST")
    
    try:
        from generator.CDVAE.load_trained_model import TrainedCDVAELoader
        
        # Use absolute paths to be sure
        base_dir = Path(os.getcwd())
        weights_path = base_dir / 'generator' / 'CDVAE' / 'new_cdvae_weights.ckpt'
        scalers_dir = base_dir / 'generator' / 'CDVAE'
        
        print(f"📁 Weights path: {weights_path}")
        print(f"📁 Scalers directory: {scalers_dir}")
        print(f"📁 Weights file exists: {weights_path.exists()}")
        print(f"📁 Scalers directory exists: {scalers_dir.exists()}")
        
        # Create loader
        print("\n🔧 Creating CDVAE loader...")
        loader = TrainedCDVAELoader(str(weights_path), str(scalers_dir))
        print("✅ CDVAE loader created successfully")
        
        # Test initial state
        print(f"📊 Initial loader state:")
        print(f"  - loader exists: {loader is not None}")
        print(f"  - hasattr('model'): {hasattr(loader, 'model')}")
        print(f"  - model value: {getattr(loader, 'model', 'NOT_FOUND')}")
        
        # Load model
        print("\n🔧 Loading CDVAE model...")
        loader.load_model()
        print("✅ CDVAE model loaded successfully")
        
        # Test post-load state
        print(f"📊 Post-load loader state:")
        print(f"  - hasattr('model'): {hasattr(loader, 'model')}")
        print(f"  - model is not None: {loader.model is not None}")
        print(f"  - model type: {type(loader.model)}")
        
        # Test the exact condition from genetic algorithm
        condition = loader and hasattr(loader, 'model') and loader.model is not None
        print(f"  - GA condition result: {condition}")
        
        if condition:
            print("🎉 SUCCESS: CDVAE model is ready for genetic algorithm!")
        else:
            print("❌ FAILURE: CDVAE model is not ready")
            
        return loader, condition
        
    except Exception as e:
        print(f"❌ CDVAE loading failed: {e}")
        traceback.print_exc()
        return None, False

def test_cdvae_generation(loader):
    """Test CDVAE structure generation"""
    print_header("CDVAE STRUCTURE GENERATION TEST")
    
    if not loader:
        print("❌ No loader available for generation test")
        return False
        
    try:
        print("🔧 Testing structure generation...")
        structures = loader.generate_structures(3)
        print(f"✅ Generated {len(structures)} structures successfully")
        
        print("\n📊 Generated structures:")
        for i, struct in enumerate(structures):
            if isinstance(struct, dict) and 'composition' in struct:
                comp_str = ''.join(f'{elem}{count}' for elem, count in sorted(struct['composition'].items()))
                print(f"  Structure {i+1}: {comp_str}")
            else:
                print(f"  Structure {i+1}: {type(struct)} - {struct}")
                
        return True
        
    except Exception as e:
        print(f"❌ Structure generation failed: {e}")
        traceback.print_exc()
        return False

def test_genetic_algorithm_import():
    """Test genetic algorithm import"""
    print_header("GENETIC ALGORITHM IMPORT TEST")
    
    try:
        # Change to genetic_algo directory for import
        original_cwd = os.getcwd()
        genetic_algo_dir = os.path.join(original_cwd, 'genetic_algo')
        
        if os.path.exists(genetic_algo_dir):
            sys.path.insert(0, genetic_algo_dir)
            
        from FINAL_genetic_algo import FinalCDVAEGA
        print("✅ FinalCDVAEGA imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to import FinalCDVAEGA: {e}")
        traceback.print_exc()
        return False
    finally:
        # Restore original directory
        os.chdir(original_cwd)

def test_genetic_algorithm_initialization():
    """Test genetic algorithm initialization"""
    print_header("GENETIC ALGORITHM INITIALIZATION TEST")
    
    try:
        # Ensure we can import
        original_cwd = os.getcwd()
        genetic_algo_dir = os.path.join(original_cwd, 'genetic_algo')
        
        if os.path.exists(genetic_algo_dir):
            sys.path.insert(0, genetic_algo_dir)
            
        from FINAL_genetic_algo import FinalCDVAEGA
        
        print("🔧 Creating genetic algorithm instance...")
        ga = FinalCDVAEGA(population_size=5, max_generations=1)
        print("✅ Genetic algorithm created successfully")
        
        # Test CDVAE integration
        print(f"\n📊 CDVAE integration status:")
        has_loader = hasattr(ga, 'cdvae_loader') and ga.cdvae_loader is not None
        print(f"  - CDVAE loader exists: {has_loader}")
        
        if has_loader:
            has_model = hasattr(ga.cdvae_loader, 'model')
            print(f"  - CDVAE model attribute exists: {has_model}")
            
            if has_model:
                model_ready = ga.cdvae_loader.model is not None
                print(f"  - CDVAE model is ready: {model_ready}")
                
                # Test the exact condition from generate_initial_population
                condition = ga.cdvae_loader and hasattr(ga.cdvae_loader, 'model') and ga.cdvae_loader.model is not None
                print(f"  - GA condition result: {condition}")
                
                if condition:
                    print("🎉 SUCCESS: Genetic algorithm will use TRUE CDVAE generation!")
                    print("🎉 The actual CDVAE model is loaded and ready!")
                    return True, ga
                else:
                    print("❌ FAILURE: Genetic algorithm will still use fallback generation")
                    return False, ga
            else:
                print("❌ CDVAE loader has no model attribute")
                return False, ga
        else:
            print("❌ CDVAE loader is None")
            return False, ga
            
    except Exception as e:
        print(f"❌ Genetic algorithm initialization failed: {e}")
        traceback.print_exc()
        return False, None
    finally:
        os.chdir(original_cwd)

def test_population_generation(ga):
    """Test initial population generation"""
    print_header("POPULATION GENERATION TEST")
    
    if not ga:
        print("❌ No genetic algorithm instance available")
        return False
        
    try:
        print("🔧 Testing initial population generation...")
        
        # This should use CDVAE if everything is working
        population = ga.generate_initial_population()
        
        print(f"✅ Generated population of size: {len(population)}")
        
        # Check if structures look reasonable
        print(f"\n📊 Population sample:")
        for i, individual in enumerate(population[:3]):  # Show first 3
            if hasattr(individual, 'composition'):
                comp_str = ''.join(f'{elem}{count}' for elem, count in sorted(individual.composition.items()))
                print(f"  Individual {i+1}: {comp_str}")
            else:
                print(f"  Individual {i+1}: {type(individual)}")
                
        return True
        
    except Exception as e:
        print(f"❌ Population generation failed: {e}")
        traceback.print_exc()
        return False

def run_full_diagnostic():
    """Run the complete diagnostic suite"""
    print_header("CDVAE DIAGNOSTIC TEST SUITE")
    print("Testing all components of CDVAE integration with genetic algorithm")
    
    results = {}
    
    # Test 1: Environment
    results['environment'] = test_environment()
    
    # Test 2: CDVAE Import
    results['cdvae_import'] = test_cdvae_import()
    
    # Test 3: CDVAE Loading
    if results['cdvae_import']:
        loader, cdvae_ready = test_cdvae_loading()
        results['cdvae_loading'] = cdvae_ready
        
        # Test 4: CDVAE Generation
        if cdvae_ready:
            results['cdvae_generation'] = test_cdvae_generation(loader)
        else:
            results['cdvae_generation'] = False
    else:
        results['cdvae_loading'] = False
        results['cdvae_generation'] = False
        loader = None
    
    # Test 5: Genetic Algorithm Import
    results['ga_import'] = test_genetic_algorithm_import()
    
    # Test 6: Genetic Algorithm Initialization
    if results['ga_import']:
        ga_ready, ga = test_genetic_algorithm_initialization()
        results['ga_initialization'] = ga_ready
        
        # Test 7: Population Generation
        if ga_ready:
            results['population_generation'] = test_population_generation(ga)
        else:
            results['population_generation'] = False
    else:
        results['ga_initialization'] = False
        results['population_generation'] = False
    
    # Final Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("🎉 CDVAE integration is working correctly!")
        print("🎉 Genetic algorithm will use TRUE CDVAE generation!")
        print(f"🔥 PyTorch {torch.__version__} is fully compatible!")
    else:
        print("❌ Some tests failed - see details above")
        print("🔧 Please check the failed components")
    print(f"{'='*60}")
    
    return all_passed

if __name__ == "__main__":
    # Run the full diagnostic
    success = run_full_diagnostic()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)