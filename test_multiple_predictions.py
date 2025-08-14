#!/usr/bin/env python3
"""
Test predictions on multiple CIF files to verify diversity
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fully_optimized_predictor import predict_single_cif_fully_optimized

def test_multiple_predictions():
    """Test predictions on multiple CIF files"""
    
    # Test with multiple generated CIF files
    cif_files = [
        "pareto_electrolyte_ga_results/cifs/gen0_Al1Br1Ge1La1Li1_2822065719632.cif",
        "pareto_electrolyte_ga_results/cifs/gen0_Al1Br1La1Li1Sn1_2822065365136.cif",
        "pareto_electrolyte_ga_results/cifs/gen0_Al1Cl1La1Li1P1_2822065598352.cif",
        "pareto_electrolyte_ga_results/cifs/gen0_Li4O1Ti1_2822065422160.cif",
        "pareto_electrolyte_ga_results/cifs/gen0_F1Li3Y1_2822065365712.cif"
    ]
    
    print("🔍 Testing predictions on multiple CIF files for diversity:")
    print("=" * 80)
    
    results_list = []
    
    for i, cif_path in enumerate(cif_files):
        if not os.path.exists(cif_path):
            print(f"❌ CIF file not found: {cif_path}")
            continue
        
        print(f"\n📁 File {i+1}: {os.path.basename(cif_path)}")
        
        try:
            results = predict_single_cif_fully_optimized(cif_path, verbose=False)
            results_list.append(results)
            
            print(f"  Composition: {results['composition']}")
            print(f"  SEI Score: {results['sei_score']:.4f}")
            print(f"  CEI Score: {results['cei_score']:.4f}")
            print(f"  Bandgap: {results['bandgap']:.4f} eV")
            print(f"  Bulk Modulus: {results['bulk_modulus']:.4f} GPa")
            print(f"  Ionic Conductivity: {results['ionic_conductivity']:.2e} S/cm")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    
    # Check for diversity
    print(f"\n📊 Diversity Analysis:")
    print("=" * 50)
    
    if len(results_list) >= 2:
        properties = ['sei_score', 'cei_score', 'bandgap', 'bulk_modulus', 'ionic_conductivity']
        
        for prop in properties:
            values = [r[prop] for r in results_list]
            unique_values = len(set(f"{v:.6f}" for v in values))  # Round to avoid floating point issues
            print(f"  {prop}: {unique_values}/{len(values)} unique values")
            if unique_values == 1:
                print(f"    ⚠️  All values identical: {values[0]}")
            else:
                print(f"    ✅ Values range: {min(values):.4f} to {max(values):.4f}")

if __name__ == "__main__":
    test_multiple_predictions()