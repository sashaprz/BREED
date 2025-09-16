#!/bin/bash
# Script to run the CDVAE genetic algorithm with proper environment

echo "🧬 Running CDVAE Genetic Algorithm for Solid-State Electrolyte Discovery"
echo "=================================================================="

# Activate the correct environment with working PyTorch Geometric
source cdvae_env/bin/activate

# Navigate to genetic algorithm directory
cd genetic_algo

# Run the genetic algorithm with CDVAE structure generation
echo "🚀 Starting genetic algorithm with:"
echo "   - CDVAE structure generation (GPU accelerated + fast mode)"
echo "   - Enhanced bulk modulus prediction (R² = 0.835)"
echo "   - Multi-objective optimization"
echo "   - Real genetic evolution with breeding and mutations"
echo "   - GPU acceleration for much faster structure generation"
echo ""

python TRUE_genetic_algo.py

echo ""
echo "✅ Genetic algorithm completed!"
echo "Results saved in: true_genetic_algo_results/"
echo "CIF files saved in: true_genetic_algo_results/cifs/"