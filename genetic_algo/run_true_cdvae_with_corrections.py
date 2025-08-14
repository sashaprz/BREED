#!/usr/bin/env python3
"""
Run True CDVAE Genetic Algorithm with Bandgap Corrections

This script runs the True CDVAE genetic algorithm with integrated literature-based
bandgap corrections for more accurate solid-state electrolyte discovery.
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_prerequisites():
    """Check if all prerequisites are available"""
    print("🔍 Checking Prerequisites...")
    print("=" * 50)
    
    issues = []
    
    # Check bandgap correction system
    try:
        from bandgap_correction_system import correct_bandgap_prediction, get_correction_info
        print("✅ Bandgap correction system: Available")
    except ImportError as e:
        issues.append(f"❌ Bandgap correction system: {e}")
    
    # Check ML predictor (prefer fully optimized)
    ml_predictor_status = "Not available"
    try:
        from fully_optimized_predictor import predict_single_cif_fully_optimized
        ml_predictor_status = "✅ Fully optimized (BEST)"
    except ImportError:
        try:
            from env.optimized_ml_predictor import predict_single_cif_optimized
            ml_predictor_status = "⚠️ Optimized (MEDIUM)"
        except ImportError:
            try:
                from env.main_rl import predict_single_cif
                ml_predictor_status = "⚠️ Standard (SLOW)"
            except ImportError:
                issues.append("❌ No ML predictor available")
    
    print(f"ML Predictor: {ml_predictor_status}")
    
    # Check True CDVAE genetic algorithm
    try:
        from genetic_algo_true_cdvae import TrueCDVAEGA
        print("✅ True CDVAE GA: Available")
    except ImportError as e:
        issues.append(f"❌ True CDVAE GA: {e}")
    
    # Check CDVAE components
    try:
        import torch
        print("✅ PyTorch: Available")
    except ImportError:
        issues.append("❌ PyTorch not installed")
    
    try:
        from pymatgen.core import Structure
        print("✅ Pymatgen: Available")
    except ImportError:
        issues.append("❌ Pymatgen not installed")
    
    # Check model files
    model_paths = [
        r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\cgcnn_pretrained\band-gap.pth.tar",
        r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\cgcnn_pretrained\bulk-moduli.pth.tar",
        r"C:\Users\Sasha\repos\RL-electrolyte-design\env\checkpoint.pth.tar"
    ]
    
    missing_models = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ Model file: {os.path.basename(model_path)}")
        else:
            missing_models.append(os.path.basename(model_path))
    
    if missing_models:
        issues.append(f"❌ Missing model files: {', '.join(missing_models)}")
    
    print("\n" + "=" * 50)
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        print("\n🔧 Please resolve these issues before running the genetic algorithm.")
        return False
    else:
        print("✅ ALL PREREQUISITES SATISFIED!")
        print("🚀 Ready to run True CDVAE GA with bandgap corrections!")
        return True

def run_true_cdvae_ga():
    """Run the True CDVAE genetic algorithm with bandgap corrections"""
    
    print("\n🚀 Starting True CDVAE Genetic Algorithm with Bandgap Corrections")
    print("=" * 80)
    
    # Import the genetic algorithm
    from genetic_algo_true_cdvae import TrueCDVAEGA
    
    # Configure the genetic algorithm
    ga_config = {
        'population_size': 20,      # Smaller for demonstration
        'max_generations': 5,       # Limited for demonstration
        'elite_count': 3,
        'tournament_size': 3,
        'mutation_rate': 0.02,
        'convergence_threshold': 10,
        'output_dir': 'true_cdvae_corrected_results'
    }
    
    print(f"Configuration:")
    for key, value in ga_config.items():
        print(f"  • {key}: {value}")
    
    print("\n" + "=" * 80)
    
    # Initialize and run the genetic algorithm
    try:
        start_time = time.time()
        
        # Create GA instance
        ga = TrueCDVAEGA(**ga_config)
        
        # Run the genetic algorithm
        results = ga.run()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print("🎉 TRUE CDVAE GA WITH BANDGAP CORRECTIONS COMPLETED!")
        print(f"{'='*80}")
        print(f"⏱️ Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Generations completed: {results.get('generations_run', 'Unknown')}")
        print(f"👥 Final population size: {results.get('final_population_size', 'Unknown')}")
        print(f"🏆 Pareto front size: {results.get('pareto_front_size', 'Unknown')}")
        print(f"📁 Results saved to: {ga.output_dir}")
        
        # Show top candidates with corrected bandgaps
        if 'pareto_front_candidates' in results and results['pareto_front_candidates']:
            print(f"\n🔬 TOP CANDIDATES WITH CORRECTED BANDGAPS:")
            print("-" * 60)
            
            for i, candidate in enumerate(results['pareto_front_candidates'][:5]):
                comp_str = "".join(f"{elem}{count}" for elem, count in 
                                 sorted(candidate['composition'].items()))
                
                print(f"\n{i+1}. Composition: {comp_str}")
                print(f"   Generation Method: {candidate.get('generation_method', 'unknown')}")
                
                properties = candidate.get('properties', {})
                if properties:
                    print(f"   Properties:")
                    
                    # Show bandgap correction details
                    if 'bandgap' in properties and 'bandgap_raw_pbe' in properties:
                        pbe_bg = properties['bandgap_raw_pbe']
                        hse_bg = properties['bandgap']
                        correction = hse_bg - pbe_bg
                        material_class = properties.get('material_class', 'unknown')
                        
                        print(f"     🔬 Bandgap (corrected): {hse_bg:.3f} eV")
                        print(f"        • Original PBE: {pbe_bg:.3f} eV")
                        print(f"        • HSE correction: +{correction:.3f} eV")
                        print(f"        • Material class: {material_class}")
                    else:
                        print(f"     Bandgap: {properties.get('bandgap', 0.0):.3f} eV")
                    
                    # Other properties
                    ic = properties.get('ionic_conductivity', 0.0)
                    print(f"     Ionic conductivity: {ic:.2e} S/cm")
                    print(f"     SEI score: {properties.get('sei_score', 0.0):.3f}")
                    print(f"     CEI score: {properties.get('cei_score', 0.0):.3f}")
                    print(f"     Bulk modulus: {properties.get('bulk_modulus', 0.0):.1f} GPa")
                
                objectives = candidate.get('objectives', [])
                if objectives:
                    obj_str = [f'{obj:.3f}' for obj in objectives]
                    print(f"   Objectives: {obj_str}")
        
        print(f"\n📈 BANDGAP CORRECTION IMPACT:")
        print("   • PBE predictions corrected to HSE-equivalent values")
        print("   • Material-specific corrections applied based on literature")
        print("   • More realistic bandgap targets (3.0 eV instead of 2.0 eV)")
        print("   • Better candidate selection for experimental validation")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR RUNNING GENETIC ALGORITHM:")
        print(f"   {str(e)}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   • Check that all model files are present")
        print(f"   • Ensure sufficient disk space for CIF files")
        print(f"   • Verify PyTorch and dependencies are installed")
        print(f"   • Try reducing population_size if memory issues occur")
        return None

def main():
    """Main function"""
    print("🧬 TRUE CDVAE GENETIC ALGORITHM WITH BANDGAP CORRECTIONS")
    print("=" * 80)
    print("This script runs the True CDVAE genetic algorithm with integrated")
    print("literature-based bandgap corrections for accurate electrolyte discovery.")
    print("=" * 80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not satisfied. Please fix issues and try again.")
        return False
    
    # Run the genetic algorithm
    results = run_true_cdvae_ga()
    
    if results:
        print(f"\n✅ SUCCESS: True CDVAE GA with bandgap corrections completed!")
        print(f"📊 Check the results directory for detailed outputs and CIF files.")
        return True
    else:
        print(f"\n❌ FAILED: Could not complete the genetic algorithm run.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)