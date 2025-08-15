#!/usr/bin/env python3
"""
Debug the CGCNN bandgap predictions to understand why they're so small
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fully_optimized_predictor import get_fully_optimized_predictor
import glob

def debug_cgcnn_bandgap():
    """Debug CGCNN bandgap predictions"""
    
    print("🔍 DEBUGGING CGCNN BANDGAP PREDICTIONS")
    print("=" * 60)
    
    # Get the predictor
    predictor = get_fully_optimized_predictor()
    
    # Find some CIF files from the genetic algorithm results
    cif_pattern = "true_cdvae_ga_results/cifs/*.cif"
    cif_files = glob.glob(cif_pattern)
    
    if not cif_files:
        print("❌ No CIF files found. Let's create a test structure.")
        return
    
    print(f"Found {len(cif_files)} CIF files from genetic algorithm")
    print("\nTesting CGCNN bandgap predictions on generated structures:")
    print("-" * 60)
    
    # Test first few CIF files
    for i, cif_file in enumerate(cif_files[:5]):
        print(f"\n{i+1}. Testing: {os.path.basename(cif_file)}")
        
        try:
            # Get the bandgap model
            bandgap_model = predictor.get_bandgap_model()
            
            # Predict raw PBE bandgap
            raw_pbe_bandgap = predictor.predict_cgcnn_property(bandgap_model, cif_file)
            
            print(f"   Raw PBE bandgap: {raw_pbe_bandgap:.6f} eV")
            
            # Also test the full prediction pipeline
            full_results = predictor.predict_single_cif(cif_file, verbose=False)
            
            print(f"   Full pipeline results:")
            print(f"     - Raw PBE: {full_results.get('bandgap_raw_pbe', 'N/A'):.6f} eV")
            print(f"     - Corrected: {full_results.get('bandgap', 'N/A'):.6f} eV")
            print(f"     - Correction applied: {full_results.get('bandgap_correction_applied', 'N/A')}")
            print(f"     - Method: {full_results.get('correction_method', 'N/A')}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test with a known good CIF file if available
    print(f"\n6. Testing with dataset CIF files:")
    print("-" * 60)
    
    dataset_cifs = glob.glob("env/cgcnn_bandgap_ionic_cond_bulk_moduli/CIF_OBELiX/cifs/*.cif")
    
    if dataset_cifs:
        print(f"Found {len(dataset_cifs)} dataset CIF files")
        
        # Test a few dataset files
        for i, cif_file in enumerate(dataset_cifs[:3]):
            print(f"\n   Dataset file {i+1}: {os.path.basename(cif_file)}")
            
            try:
                bandgap_model = predictor.get_bandgap_model()
                raw_pbe_bandgap = predictor.predict_cgcnn_property(bandgap_model, cif_file)
                print(f"     Raw PBE bandgap: {raw_pbe_bandgap:.6f} eV")
                
                # Test full pipeline
                full_results = predictor.predict_single_cif(cif_file, verbose=False)
                print(f"     Corrected bandgap: {full_results.get('bandgap', 'N/A'):.6f} eV")
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
    else:
        print("   No dataset CIF files found")
    
    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    print("If all PBE predictions are very small (< 0.01 eV), the issue is likely:")
    print("1. Generated crystal structures are unrealistic/unstable")
    print("2. CGCNN bandgap model has issues with the generated structures")
    print("3. CIF file format or structure issues")

if __name__ == "__main__":
    debug_cgcnn_bandgap()