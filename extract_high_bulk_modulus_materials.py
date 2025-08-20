#!/usr/bin/env python3
"""
Extract materials with high bulk modulus (>50 GPa) from Materials Project
to train a fresh CGCNN model focused on high bulk modulus prediction.

This avoids the bias issue by training from scratch on high bulk modulus materials only.
"""

import os
import json
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import numpy as np
from tqdm import tqdm
import time

# Materials Project API key
MP_API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"

def extract_high_bulk_modulus_materials(min_bulk_modulus=50, max_materials=5000):
    """
    Extract materials with high bulk modulus from Materials Project
    
    Args:
        min_bulk_modulus: Minimum bulk modulus threshold (GPa)
        max_materials: Maximum number of materials to extract
    """
    
    print(f"🔍 Extracting materials with bulk modulus > {min_bulk_modulus} GPa...")
    
    with MPRester(MP_API_KEY) as mpr:
        
        # Query all materials with bulk modulus data (no direct filtering available)
        print("   Querying Materials Project database...")
        docs = mpr.materials.summary.search(
            fields=["material_id", "formula_pretty", "structure", "bulk_modulus",
                   "energy_above_hull", "formation_energy_per_atom", "band_gap",
                   "density", "nsites", "chemsys"],
            energy_above_hull=(None, 0.1),  # Stable or nearly stable materials
            num_chunks=20,
            chunk_size=1000
        )
        
        print(f"   Retrieved {len(docs)} materials from Materials Project")
        
        # Filter for materials with high bulk modulus
        materials = []
        high_bulk_count = 0
        
        for mat in tqdm(docs, desc="Filtering by bulk modulus"):
            if len(materials) >= max_materials:
                break
                
            # Check if material has bulk modulus data and meets threshold
            bulk_modulus_value = None
            if hasattr(mat, 'bulk_modulus') and mat.bulk_modulus is not None:
                # Handle different bulk modulus formats
                if isinstance(mat.bulk_modulus, dict):
                    # Extract bulk modulus value from dict (usually 'vrh' key)
                    bulk_modulus_value = mat.bulk_modulus.get('vrh') or mat.bulk_modulus.get('voigt') or mat.bulk_modulus.get('reuss')
                else:
                    bulk_modulus_value = mat.bulk_modulus
            
            if bulk_modulus_value is not None and bulk_modulus_value >= min_bulk_modulus:
                high_bulk_count += 1
                
                materials.append({
                    'material_id': mat.material_id,
                    'formula': mat.formula_pretty,
                    'bulk_modulus': bulk_modulus_value,
                    'energy_above_hull': mat.energy_above_hull,
                    'formation_energy_per_atom': mat.formation_energy_per_atom,
                    'band_gap': mat.band_gap,
                    'density': mat.density,
                    'nsites': mat.nsites,
                    'structure': mat.structure,
                    'chemsys': mat.chemsys
                })
        
        print(f"   Found {high_bulk_count} materials with bulk modulus > {min_bulk_modulus} GPa")
    
    print(f"✅ Extracted {len(materials)} high bulk modulus materials")
    return materials

def save_training_data(materials, output_dir="high_bulk_modulus_training"):
    """Save materials as CIF files and create training dataset"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "structures"), exist_ok=True)
    
    training_data = []
    failed_cifs = []
    
    print(f"\n💾 Saving {len(materials)} structures...")
    
    for mat in tqdm(materials, desc="Processing materials"):
        try:
            # Save structure as CIF
            cif_filename = f"{mat['material_id']}.cif"
            cif_path = os.path.join(output_dir, "structures", cif_filename)
            
            cif_writer = CifWriter(mat['structure'])
            cif_writer.write_file(cif_path)
            
            # Add to training data
            training_data.append({
                'material_id': mat['material_id'],
                'cif_file': cif_filename,
                'bulk_modulus': mat['bulk_modulus'],
                'formula': mat['formula'],
                'chemsys': mat['chemsys'],
                'energy_above_hull': mat['energy_above_hull'],
                'formation_energy_per_atom': mat['formation_energy_per_atom'],
                'band_gap': mat['band_gap'],
                'density': mat['density'],
                'nsites': mat['nsites']
            })
            
        except Exception as e:
            print(f"Failed to save CIF for {mat['material_id']}: {e}")
            failed_cifs.append(mat['material_id'])
    
    # Save training data as CSV (CGCNN format: no headers)
    df = pd.DataFrame(training_data)
    
    # Split into train/val/test
    np.random.seed(42)
    n_total = len(df)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Save splits
    for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_df = df.iloc[idx][['cif_file', 'bulk_modulus']].copy()
        split_path = os.path.join(output_dir, f"{split_name}.csv")
        split_df.to_csv(split_path, header=False, index=False)
        print(f"   • {split_name}: {len(split_df)} materials")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(training_data, f, indent=2, default=str)
    
    # Calculate and save statistics
    stats = {
        'total_materials': len(training_data),
        'failed_cifs': len(failed_cifs),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'bulk_modulus_stats': {
            'mean': float(df['bulk_modulus'].mean()),
            'std': float(df['bulk_modulus'].std()),
            'min': float(df['bulk_modulus'].min()),
            'max': float(df['bulk_modulus'].max()),
            'median': float(df['bulk_modulus'].median()),
            'q25': float(df['bulk_modulus'].quantile(0.25)),
            'q75': float(df['bulk_modulus'].quantile(0.75))
        },
        'chemical_systems': df['chemsys'].value_counts().head(20).to_dict()
    }
    
    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"   • Total materials: {stats['total_materials']}")
    print(f"   • Train/Val/Test: {stats['train_size']}/{stats['val_size']}/{stats['test_size']}")
    print(f"   • Bulk modulus range: {stats['bulk_modulus_stats']['min']:.1f} - {stats['bulk_modulus_stats']['max']:.1f} GPa")
    print(f"   • Mean: {stats['bulk_modulus_stats']['mean']:.1f} ± {stats['bulk_modulus_stats']['std']:.1f} GPa")
    print(f"   • Median: {stats['bulk_modulus_stats']['median']:.1f} GPa")
    
    print(f"\n✅ Training data saved to: {output_dir}/")
    print(f"   • Train: train.csv ({stats['train_size']} materials)")
    print(f"   • Validation: val.csv ({stats['val_size']} materials)")
    print(f"   • Test: test.csv ({stats['test_size']} materials)")
    print(f"   • Structures: structures/ directory")
    
    return training_data, stats

def main():
    """Main extraction pipeline"""
    
    print("🚀 Extracting high bulk modulus materials for CGCNN training...")
    
    # Extract materials with bulk modulus > 50 GPa
    materials = extract_high_bulk_modulus_materials(
        min_bulk_modulus=50,  # Focus on high bulk modulus materials
        max_materials=5000    # Reasonable dataset size
    )
    
    if not materials:
        print("❌ No materials found!")
        return
    
    # Save training data
    training_data, stats = save_training_data(materials)
    
    print(f"\n🎉 Dataset creation complete!")
    print(f"   Ready to train fresh CGCNN model on {len(training_data)} high bulk modulus materials")
    
    return training_data, stats

if __name__ == "__main__":
    main()