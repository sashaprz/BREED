#!/usr/bin/env python3
"""
Compare Generation Methods: Advanced Generation vs True CDVAE
Analyze and compare the performance of different crystal generation approaches
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns

def load_results(results_dir: str) -> Dict[str, Any]:
    """Load genetic algorithm results from directory"""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return {}
    
    # Load all generation data
    generations = []
    for gen_dir in sorted(results_path.glob("generation_*")):
        if gen_dir.is_dir():
            population_file = gen_dir / "population.json"
            if population_file.exists():
                with open(population_file, 'r') as f:
                    gen_data = json.load(f)
                    generations.append({
                        'generation': int(gen_dir.name.split('_')[1]),
                        'population': gen_data
                    })
    
    return {
        'results_dir': results_dir,
        'generations': generations,
        'total_generations': len(generations)
    }

def analyze_diversity(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze compositional and structural diversity"""
    
    if not results or not results['generations']:
        return {}
    
    # Get final generation data
    final_gen = results['generations'][-1]['population']
    
    # Analyze compositions
    compositions = []
    generation_methods = []
    properties = []
    
    for candidate in final_gen:
        # Extract composition string
        comp_dict = candidate['composition']
        comp_str = "".join(f"{elem}{count}" for elem, count in sorted(comp_dict.items()))
        compositions.append(comp_str)
        
        # Extract generation method
        generation_methods.append(candidate.get('generation_method', 'unknown'))
        
        # Extract properties
        props = candidate.get('properties', {})
        properties.append(props)
    
    # Calculate diversity metrics
    unique_compositions = len(set(compositions))
    total_compositions = len(compositions)
    composition_diversity = unique_compositions / total_compositions if total_compositions > 0 else 0
    
    # Analyze generation methods
    method_counts = {}
    for method in generation_methods:
        method_counts[method] = method_counts.get(method, 0) + 1
    
    # Analyze property ranges
    property_ranges = {}
    if properties:
        for prop_name in properties[0].keys():
            values = [p.get(prop_name, 0) for p in properties if p.get(prop_name) is not None]
            if values:
                property_ranges[prop_name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'range': max(values) - min(values)
                }
    
    return {
        'total_candidates': total_compositions,
        'unique_compositions': unique_compositions,
        'composition_diversity': composition_diversity,
        'generation_methods': method_counts,
        'property_ranges': property_ranges,
        'compositions': compositions[:10],  # Top 10 for display
        'final_generation_size': len(final_gen)
    }

def analyze_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze optimization performance over generations"""
    
    if not results or not results['generations']:
        return {}
    
    # Track performance metrics over generations
    generation_stats = []
    
    for gen_data in results['generations']:
        gen_num = gen_data['generation']
        population = gen_data['population']
        
        # Extract Pareto front (rank 0)
        pareto_front = [cand for cand in population if cand.get('pareto_rank', 0) == 0]
        
        # Calculate performance metrics
        if pareto_front:
            # Best ionic conductivity
            ic_values = [cand['properties'].get('ionic_conductivity', 0) for cand in pareto_front]
            best_ic = max(ic_values) if ic_values else 0
            
            # Average properties
            avg_props = {}
            for prop_name in ['ionic_conductivity', 'bandgap', 'sei_score', 'cei_score', 'bulk_modulus']:
                values = [cand['properties'].get(prop_name, 0) for cand in pareto_front]
                avg_props[prop_name] = np.mean(values) if values else 0
            
            generation_stats.append({
                'generation': gen_num,
                'pareto_front_size': len(pareto_front),
                'best_ionic_conductivity': best_ic,
                'avg_properties': avg_props
            })
    
    return {
        'generation_stats': generation_stats,
        'total_generations': len(generation_stats),
        'final_pareto_size': generation_stats[-1]['pareto_front_size'] if generation_stats else 0,
        'best_final_ic': generation_stats[-1]['best_ionic_conductivity'] if generation_stats else 0
    }

def compare_methods(advanced_results: Dict, cdvae_results: Dict) -> Dict[str, Any]:
    """Compare Advanced Generation vs True CDVAE methods"""
    
    print("🔍 Comparing Generation Methods...")
    print("=" * 60)
    
    # Analyze both methods
    advanced_diversity = analyze_diversity(advanced_results)
    cdvae_diversity = analyze_diversity(cdvae_results)
    
    advanced_performance = analyze_performance(advanced_results)
    cdvae_performance = analyze_performance(cdvae_results)
    
    # Create comparison
    comparison = {
        'advanced_generation': {
            'diversity': advanced_diversity,
            'performance': advanced_performance
        },
        'true_cdvae': {
            'diversity': cdvae_diversity,
            'performance': cdvae_performance
        }
    }
    
    # Print comparison summary
    print("\n📊 DIVERSITY COMPARISON")
    print("-" * 30)
    
    if advanced_diversity and cdvae_diversity:
        print(f"Advanced Generation:")
        print(f"  • Total candidates: {advanced_diversity['total_candidates']}")
        print(f"  • Unique compositions: {advanced_diversity['unique_compositions']}")
        print(f"  • Composition diversity: {advanced_diversity['composition_diversity']:.1%}")
        print(f"  • Generation methods: {list(advanced_diversity['generation_methods'].keys())}")
        
        print(f"\nTrue CDVAE:")
        print(f"  • Total candidates: {cdvae_diversity['total_candidates']}")
        print(f"  • Unique compositions: {cdvae_diversity['unique_compositions']}")
        print(f"  • Composition diversity: {cdvae_diversity['composition_diversity']:.1%}")
        print(f"  • Generation methods: {list(cdvae_diversity['generation_methods'].keys())}")
        
        # Diversity winner
        if cdvae_diversity['composition_diversity'] > advanced_diversity['composition_diversity']:
            print(f"\n🏆 Winner (Diversity): True CDVAE")
        else:
            print(f"\n🏆 Winner (Diversity): Advanced Generation")
    
    print("\n⚡ PERFORMANCE COMPARISON")
    print("-" * 30)
    
    if advanced_performance and cdvae_performance:
        print(f"Advanced Generation:")
        print(f"  • Final Pareto front size: {advanced_performance['final_pareto_size']}")
        print(f"  • Best ionic conductivity: {advanced_performance['best_final_ic']:.2e} S/cm")
        print(f"  • Total generations: {advanced_performance['total_generations']}")
        
        print(f"\nTrue CDVAE:")
        print(f"  • Final Pareto front size: {cdvae_performance['final_pareto_size']}")
        print(f"  • Best ionic conductivity: {cdvae_performance['best_final_ic']:.2e} S/cm")
        print(f"  • Total generations: {cdvae_performance['total_generations']}")
        
        # Performance winner
        if cdvae_performance['best_final_ic'] > advanced_performance['best_final_ic']:
            print(f"\n🏆 Winner (Performance): True CDVAE")
        else:
            print(f"\n🏆 Winner (Performance): Advanced Generation")
    
    return comparison

def create_comparison_plots(comparison: Dict[str, Any], output_dir: str = "comparison_plots"):
    """Create visualization plots comparing the methods"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plt.style.use('default')
    
    # Plot 1: Diversity Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['Advanced\nGeneration', 'True\nCDVAE']
    
    # Composition diversity
    adv_div = comparison['advanced_generation']['diversity']
    cdvae_div = comparison['true_cdvae']['diversity']
    
    if adv_div and cdvae_div:
        diversities = [adv_div['composition_diversity'], cdvae_div['composition_diversity']]
        colors = ['skyblue', 'lightcoral']
        
        bars1 = ax1.bar(methods, diversities, color=colors, alpha=0.7)
        ax1.set_ylabel('Composition Diversity')
        ax1.set_title('Compositional Diversity Comparison')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars1, diversities):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.1%}', ha='center', va='bottom')
        
        # Unique compositions
        unique_counts = [adv_div['unique_compositions'], cdvae_div['unique_compositions']]
        bars2 = ax2.bar(methods, unique_counts, color=colors, alpha=0.7)
        ax2.set_ylabel('Number of Unique Compositions')
        ax2.set_title('Unique Compositions Generated')
        
        # Add value labels on bars
        for bar, value in zip(bars2, unique_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Performance Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    adv_perf = comparison['advanced_generation']['performance']
    cdvae_perf = comparison['true_cdvae']['performance']
    
    if adv_perf and cdvae_perf:
        # Best ionic conductivity
        ic_values = [adv_perf['best_final_ic'], cdvae_perf['best_final_ic']]
        bars1 = ax1.bar(methods, ic_values, color=colors, alpha=0.7)
        ax1.set_ylabel('Ionic Conductivity (S/cm)')
        ax1.set_title('Best Ionic Conductivity')
        ax1.set_yscale('log')
        
        # Add value labels on bars
        for bar, value in zip(bars1, ic_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{value:.2e}', ha='center', va='bottom', rotation=45)
        
        # Pareto front size
        pareto_sizes = [adv_perf['final_pareto_size'], cdvae_perf['final_pareto_size']]
        bars2 = ax2.bar(methods, pareto_sizes, color=colors, alpha=0.7)
        ax2.set_ylabel('Final Pareto Front Size')
        ax2.set_title('Optimization Convergence')
        
        # Add value labels on bars
        for bar, value in zip(bars2, pareto_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plots saved to: {output_path}")

def main():
    """Main comparison function"""
    
    print("🔬 Crystal Generation Methods Comparison")
    print("=" * 60)
    print("Comparing Advanced Generation vs True CDVAE approaches")
    print()
    
    # Load results from both methods
    advanced_results = load_results("advanced_generation_ga_results")
    cdvae_results = load_results("true_cdvae_ga_results")
    
    if not advanced_results['generations']:
        print("⚠️ Advanced Generation results not found. Run genetic_algo_advanced_generation.py first.")
    
    if not cdvae_results['generations']:
        print("⚠️ True CDVAE results not found. Run genetic_algo_true_cdvae.py first.")
    
    if advanced_results['generations'] and cdvae_results['generations']:
        # Perform comparison
        comparison = compare_methods(advanced_results, cdvae_results)
        
        # Create visualization plots
        create_comparison_plots(comparison)
        
        # Save comparison data
        with open('generation_methods_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"\n💾 Detailed comparison saved to: generation_methods_comparison.json")
        print(f"📊 Visualization plots saved to: comparison_plots/")
        
        print(f"\n🎯 SUMMARY")
        print("=" * 60)
        print("✅ Both generation methods successfully compared")
        print("✅ Diversity and performance metrics analyzed")
        print("✅ Visualization plots created")
        print("✅ Detailed comparison data saved")
        
        print(f"\n🚀 Next Steps:")
        print("• Analyze the detailed comparison results")
        print("• Review the visualization plots")
        print("• Choose the best method for your research goals")
        print("• Consider hybrid approaches combining both methods")
        
    else:
        print("❌ Cannot perform comparison - missing results from one or both methods")
        print("   Please run both genetic algorithms first:")
        print("   1. python genetic_algo_advanced_generation.py")
        print("   2. python genetic_algo_true_cdvae.py")

if __name__ == "__main__":
    main()