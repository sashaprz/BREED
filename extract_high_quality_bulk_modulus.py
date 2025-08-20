#!/usr/bin/env python3
"""
Extract high-quality bulk modulus data for solid electrolytes with focus on:
1. Experimental validation
2. Li/Na-containing materials (solid electrolytes)
3. Better outlier detection and removal
4. Material class filtering
"""

import os
import json
import pandas as pd
import numpy as np
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm
import time
from scipy import stats

# Materials Project API key
MP_API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"

def get_solid_electrolyte_queries():
    """Define queries specifically for solid electrolyte materials"""
    
    # Li-based solid electrolytes
    li_electrolytes = [
        "Li-P-S",      # Argyrodites (Li6PS5Cl, Li6PS5Br)
        "Li-Si-P-S",   # Thio-LISICON
        "Li-Ge-P-S",   # Thio-LISICON variants
        "Li-La-Zr-O",  # LLZO garnets
        "Li-Al-Ti-P-O", # NASICON-type
        "Li-La-Ti-O",  # Perovskite electrolytes
        "Li-P-O-N",    # LiPON
        "Li-B-O",      # Borate glasses
        "Li-Al-Cl-O",  # Antiperovskites
        "Li-Si-O",     # Silicate electrolytes
    ]
    
    # Na-based solid electrolytes
    na_electrolytes = [
        "Na-Zr-Si-P-O", # NASICON
        "Na-Al-Si-O",   # β-alumina
        "Na-P-S",       # Na3PS4
        "Na-Sb-S",      # Na3SbS4
        "Na-B-O",       # Borate glasses
    ]
    
    return {
        'li_electrolytes': li_electrolytes,
        'na_electrolytes': na_electrolytes
    }

def query_high_quality_materials(mpr, chemical_system, min_bulk_modulus=30):
    """Query materials with quality filters for solid electrolytes"""
    
    try:
        # Query with strict quality filters
        docs = mpr.materials.summary.search(
            chemsys=chemical_system,
            fields=["material_id", "formula_pretty", "structure", "bulk_modulus",
                   "energy_above_hull", "formation_energy_per_atom", "band_gap",
                   "density", "nsites", "theoretical"],
            energy_above_hull=(None, 0.05),  # Very stable materials only
            theoretical=False,  # Prefer experimental data
            num_chunks=1,
            chunk_size=500
        )
        
        # Filter for materials with bulk modulus data
        quality_materials = []
        for mat in docs:
            if not hasattr(mat, 'bulk_modulus') or mat.bulk_modulus is None:
                continue
                
            # Extract bulk modulus value
            bulk_modulus_value = None
            if isinstance(mat.bulk_modulus, dict):
                bulk_modulus_value = mat.bulk_modulus.get('vrh') or mat.bulk_modulus.get('voigt') or mat.bulk_modulus.get('reuss')
            else:
                bulk_modulus_value = mat.bulk_modulus
            
            if bulk_modulus_value is None or bulk_modulus_value < min_bulk_modulus:
                continue
            
            # Additional quality filters
            if (mat.energy_above_hull is not None and mat.energy_above_hull > 0.05):
                continue  # Skip unstable materials
                
            if (mat.nsites is not None and mat.nsites > 100):
                continue  # Skip very large unit cells (likely computational artifacts)
            
            quality_materials.append({
                'material_id': mat.material_id,
                'formula': mat.formula_pretty,
                'bulk_modulus': bulk_modulus_value,
                'energy_above_hull': mat.energy_above_hull,
                'formation_energy_per_atom': mat.formation_energy_per_atom,
                'band_gap': mat.band_gap,
                'density': mat.density,
                'nsites': mat.nsites,
                'structure': mat.structure,
                'chemical_system': chemical_system,
                'theoretical': getattr(mat, 'theoretical', True)
            })
        
        print(f"   Found {len(quality_materials)} high-quality materials for {chemical_system}")
        return quality_materials
        
    except Exception as e:
        print(f"Error querying system {chemical_system}: {e}")
        return []

def advanced_outlier_detection(materials):
    """Advanced outlier detection using multiple methods"""
    
    if len(materials) < 10:
        return materials  # Too few samples for outlier detection
    
    bulk_moduli = [mat['bulk_modulus'] for mat in materials]
    bulk_array = np.array(bulk_moduli)
    
    print(f"🔍 Outlier detection on {len(materials)} materials...")
    print(f"   Initial range: {bulk_array.min():.1f} - {bulk_array.max():.1f} GPa")
    
    # Method 1: Modified Z-score (robust to outliers)
    median = np.median(bulk_array)
    mad = np.median(np.abs(bulk_array - median))  # Median Absolute Deviation
    modified_z_scores = 0.6745 * (bulk_array - median) / mad
    
    # Method 2: Interquartile Range (IQR)
    q1, q3 = np.percentile(bulk_array, [25, 75])
    iqr = q3 - q1
    iqr_lower = q1 - 2.0 * iqr  # More conservative than 1.5
    iqr_upper = q3 + 2.0 * iqr
    
    # Method 3: Physical constraints for solid electrolytes
    # Typical range: 30-300 GPa (very conservative)
    physical_lower = 30.0
    physical_upper = 300.0
    
    # Combine all methods
    outlier_mask = (
        (np.abs(modified_z_scores) > 3.0) |  # Modified Z-score outliers
        (bulk_array < iqr_lower) | (bulk_array > iqr_upper) |  # IQR outliers
        (bulk_array < physical_lower) | (bulk_array > physical_upper)  # Physical outliers
    )
    
    # Keep only non-outliers
    clean_materials = [mat for i, mat in enumerate(materials) if not outlier_mask[i]]
    outliers_removed = len(materials) - len(clean_materials)
    
    if len(clean_materials) > 0:
        clean_bulk_moduli = [mat['bulk_modulus'] for mat in clean_materials]
        clean_array = np.array(clean_bulk_moduli)
        print(f"   Removed {outliers_removed} outliers")
        print(f"   Clean range: {clean_array.min():.1f} - {clean_array.max():.1f} GPa")
        print(f"   Clean mean: {clean_array.mean():.1f} ± {clean_array.std():.1f} GPa")
    
    return clean_materials

def extract_high_quality_bulk_modulus():
    """Extract high-quality bulk modulus data for solid electrolytes"""
    
    print("🔋 Extracting HIGH-QUALITY bulk modulus data for solid electrolytes...")
    
    with MPRester(MP_API_KEY) as mpr:
        all_materials = []
        electrolyte_queries = get_solid_electrolyte_queries()
        
        # Query Li-based electrolytes
        print("\n📊 Querying Li-based solid electrolytes...")
        for system in tqdm(electrolyte_queries['li_electrolytes'], desc="Li systems"):
            materials = query_high_quality_materials(mpr, system, min_bulk_modulus=30)
            for mat in materials:
                mat['electrolyte_type'] = 'Li-based'
            all_materials.extend(materials)
            time.sleep(0.1)  # Rate limiting
        
        # Query Na-based electrolytes
        print("\n📊 Querying Na-based solid electrolytes...")
        for system in tqdm(electrolyte_queries['na_electrolytes'], desc="Na systems"):
            materials = query_high_quality_materials(mpr, system, min_bulk_modulus=30)
            for mat in materials:
                mat['electrolyte_type'] = 'Na-based'
            all_materials.extend(materials)
            time.sleep(0.1)  # Rate limiting
    
    print(f"\n✅ Total materials extracted: {len(all_materials)}")
    
    # Remove duplicates based on material_id
    unique_materials = {}
    for mat in all_materials:
        if mat['material_id'] not in unique_materials:
            unique_materials[mat['material_id']] = mat
    
    all_materials = list(unique_materials.values())
    print(f"✅ Unique materials after deduplication: {len(all_materials)}")
    
    # Advanced outlier detection
    clean_materials = advanced_outlier_detection(all_materials)
    
    return clean_materials

def analyze_data_quality(materials):
    """Analyze the quality of extracted data"""
    
    print(f"\n📈 Data Quality Analysis:")
    
    bulk_moduli = [mat['bulk_modulus'] for mat in materials]
    bulk_array = np.array(bulk_moduli)
    
    # Basic statistics
    print(f"   • Sample size: {len(materials)}")
    print(f"   • Bulk modulus range: {bulk_array.min():.1f} - {bulk_array.max():.1f} GPa")
    print(f"   • Mean ± Std: {bulk_array.mean():.1f} ± {bulk_array.std():.1f} GPa")
    print(f"   • Median: {np.median(bulk_array):.1f} GPa")
    print(f"   • Q1, Q3: {np.percentile(bulk_array, 25):.1f}, {np.percentile(bulk_array, 75):.1f} GPa")
    
    # Electrolyte type distribution
    li_count = sum(1 for mat in materials if mat['electrolyte_type'] == 'Li-based')
    na_count = sum(1 for mat in materials if mat['electrolyte_type'] == 'Na-based')
    print(f"   • Li-based: {li_count} materials")
    print(f"   • Na-based: {na_count} materials")
    
    # Experimental vs theoretical
    experimental_count = sum(1 for mat in materials if not mat.get('theoretical', True))
    print(f"   • Experimental data: {experimental_count} materials")
    print(f"   • Computational data: {len(materials) - experimental_count} materials")
    
    # Stability analysis
    stable_count = sum(1 for mat in materials if mat.get('energy_above_hull', 0) <= 0.01)
    print(f"   • Highly stable (E_hull ≤ 0.01 eV): {stable_count} materials")

def save_high_quality_data(materials, output_dir="high_quality_bulk_modulus"):
    """Save high-quality training data"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "structures"), exist_ok=True)
    
    training_data = []
    failed_cifs = []
    
    print(f"\n💾 Saving {len(materials)} high-quality structures...")
    
    for mat in tqdm(materials, desc="Saving structures"):
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
                'electrolyte_type': mat['electrolyte_type'],
                'chemical_system': mat['chemical_system'],
                'energy_above_hull': mat['energy_above_hull'],
                'formation_energy_per_atom': mat['formation_energy_per_atom'],
                'band_gap': mat['band_gap'],
                'density': mat['density'],
                'nsites': mat['nsites'],
                'theoretical': mat.get('theoretical', True)
            })
            
        except Exception as e:
            print(f"Failed to save CIF for {mat['material_id']}: {e}")
            failed_cifs.append(mat['material_id'])
    
    # Create train/val/test splits with stratification
    df = pd.DataFrame(training_data)
    
    # Stratified split by electrolyte type and bulk modulus range
    np.random.seed(42)
    
    # Split by bulk modulus quartiles to ensure balanced distribution
    df['bulk_quartile'] = pd.qcut(df['bulk_modulus'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    train_data = []
    val_data = []
    test_data = []
    
    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        quartile_data = df[df['bulk_quartile'] == quartile]
        n = len(quartile_data)
        
        indices = np.random.permutation(n)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        train_data.extend(quartile_data.iloc[indices[:n_train]].to_dict('records'))
        val_data.extend(quartile_data.iloc[indices[n_train:n_train+n_val]].to_dict('records'))
        test_data.extend(quartile_data.iloc[indices[n_train+n_val:]].to_dict('records'))
    
    # Save splits
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_df = pd.DataFrame(split_data)[['cif_file', 'bulk_modulus']]
        split_path = os.path.join(output_dir, f"{split_name}.csv")
        split_df.to_csv(split_path, header=False, index=False)
        print(f"   • {split_name}: {len(split_df)} materials")
    
    # Save metadata
    metadata = {
        'total_materials': len(training_data),
        'failed_cifs': len(failed_cifs),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'bulk_modulus_stats': {
            'mean': float(df['bulk_modulus'].mean()),
            'std': float(df['bulk_modulus'].std()),
            'min': float(df['bulk_modulus'].min()),
            'max': float(df['bulk_modulus'].max()),
            'median': float(df['bulk_modulus'].median()),
            'q25': float(df['bulk_modulus'].quantile(0.25)),
            'q75': float(df['bulk_modulus'].quantile(0.75))
        },
        'electrolyte_types': df['electrolyte_type'].value_counts().to_dict(),
        'experimental_ratio': float((~df['theoretical']).mean()),
        'extraction_date': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = os.path.join(output_dir, "high_quality_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ High-quality training data saved to: {output_dir}/")
    print(f"   • Training CSV files with stratified splits")
    print(f"   • Metadata: {metadata_path}")
    print(f"   • CIF files: {output_dir}/structures/")
    
    return training_data, metadata

def main():
    """Main extraction pipeline for high-quality data"""
    
    print("🚀 Starting HIGH-QUALITY bulk modulus extraction for solid electrolytes...")
    
    # Extract high-quality materials
    materials = extract_high_quality_bulk_modulus()
    
    if len(materials) < 50:
        print(f"❌ Too few materials found ({len(materials)}). Need at least 50 for training.")
        return
    
    # Analyze data quality
    analyze_data_quality(materials)
    
    # Save high-quality training data
    training_data, metadata = save_high_quality_data(materials)
    
    print(f"\n🎉 High-quality extraction complete!")
    print(f"   Ready for CGCNN training with {len(training_data)} solid electrolyte materials")
    print(f"   Expected improvement: Better R² due to cleaner, domain-specific data")
    
    return training_data, metadata

if __name__ == "__main__":
    main()