#!/usr/bin/env python3
"""
Clean run of True CDVAE Genetic Algorithm without verbose bandgap correction logging
"""

import sys
import os
from pathlib import Path

# Add the genetic_algo directory to Python path
sys.path.append(str(Path(__file__).parent))

from genetic_algo_true_cdvae import TrueCDVAEGA

def main():
    """Run True CDVAE GA with clean output"""
    
    print("🚀 Running True CDVAE Genetic Algorithm with Clean Output")
    print("=" * 60)
    
    # Initialize GA with smaller parameters for demonstration
    ga = TrueCDVAEGA(
        population_size=20,  # Smaller for faster execution
        max_generations=5,   # Limited for demonstration
        output_dir="clean_true_cdvae_results"
    )
    
    # Run the genetic algorithm
    results = ga.run()
    
    print("\n" + "=" * 60)
    print("🎉 True CDVAE Genetic Algorithm Completed Successfully!")
    print(f"📁 Results saved to: {ga.output_dir}")
    print(f"📊 Final Pareto front size: {results['pareto_front_size']}")
    print(f"🔄 Generations completed: {results['generations_run']}")
    
    return results

if __name__ == "__main__":
    results = main()