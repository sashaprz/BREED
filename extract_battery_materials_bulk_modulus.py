#!/usr/bin/env python3
"""
Extract bulk modulus data for inorganic solid electrolytes and battery materials
from Materials Project to create a substantial fine-tuning dataset.

Focus on:
- Li-ion battery materials (cathodes, anodes, electrolytes)
- Na-ion battery materials
- Solid electrolytes
- Inorganic crystals with high bulk modulus (>20 GPa)
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

def get_battery_material_queries():
    """Define queries for different types of battery materials"""
    
    # Common battery elements
    li_systems = ["Li-O", "Li-P-O", "Li-S-O", "Li-Si-O", "Li-Al-O", "Li-Ti-O", 
                  "Li-Mn-O", "Li-Fe-O", "Li-Co-O", "Li-Ni-O", "Li-La-O"]
    
    na_systems = ["Na-O", "Na-P-O", "Na-S-O", "Na-Si-O", "Na-Al-O", "Na-Ti-O",
                  "Na-Mn-O", "Na-Fe-O", "Na-Co-O"]
    
    # Solid electrolyte systems
    solid_electrolyte_systems = [
        "Li-La-Ti-O", "Li-La-Zr-O", "Li-Al-Ti-P-O", "Li-Ge-P-S",
        "Li-Si-P-S", "Li-P-S", "Li-B-O", "Li-Al-Cl-O",
        "Na-Zr-Si-P-O", "Na-Al-Si-O", "Na-B-O"
    ]
    
    # Garnet and perovskite electrolytes
    garnet_systems = ["Li-La-Zr-O", "Li-Al-La-Zr-O"]
    perovskite_systems = ["Li-La-Ti-O", "Li-Sr-Ta-O", "Li-Ba-Ta-O"]
    
    return {
        'li_battery': li_systems,
        'na_battery': na_systems, 
        'solid_electrolytes': solid_electrolyte_systems,
        'garnets': garnet_systems,
        'perovskites': perovskite_systems
    }

def query_materials_by_system(mpr, chemical_system, min_bulk_modulus=20):
    """Query materials for a specific chemical system"""
    
    try:
        # Query with bulk modulus filter
        docs = mpr.materials.summary.search(
            chemsys=chemical_system,
            fields=["material_id", "formula_pretty", "structure", "bulk_modulus", 
                   "energy_above_hull", "formation_energy_per_atom", "band_gap",
                   "density", "nsites"],
            bulk_modulus=(min_bulk_modulus, None),  # Only materials with bulk modulus >= min_bulk_modulus
            energy_above_hull=(None, 0.1),  # Stable or nearly stable materials
            num_chunks=1,
            chunk_size=1000
        )
        
        print(f"Found {len(docs)} materials for system {chemical_system}")
        return docs
        
    except Exception as e:
        print(f"Error querying system {chemical_system}: {e}")
        return []

def extract_bulk_modulus_data():
    """Extract bulk modulus data for battery materials"""
    
    print("🔋 Extracting bulk modulus data for battery materials...")
    
    # Initialize Materials Project client
    with MPRester(MP_API_KEY) as mpr:
        
        all_materials = []
        material_queries = get_battery_material_queries()
        
        # Query each category of materials
        for category, systems in material_queries.items():
            print(f"\n📊 Querying {category} materials...")
            
            for system in tqdm(systems, desc=f"Processing {category}"):
                materials = query_materials_by_system(mpr, system)
                
                for mat in materials:
                    if mat.bulk_modulus and mat.bulk_modulus > 20:  # Filter for realistic bulk modulus
                        all_materials.append({
                            'material_id': mat.material_id,
                            'formula': mat.formula_pretty,
                            'bulk_modulus': mat.bulk_modulus,
                            'energy_above_hull': mat.energy_above_hull,
                            'formation_energy_per_atom': mat.formation_energy_per_atom,
                            'band_gap': mat.band_gap,
                            'density': mat.density,
                            'nsites': mat.nsites,
                            'structure': mat.structure,
                            'category': category,
                            'chemical_system': system
                        })
                
                # Rate limiting
                time.sleep(0.1)
    
    print(f"\n✅ Total materials extracted: {len(all_materials)}")
    
    # Remove duplicates based on material_id
    unique_materials = {}
    for mat in all_materials:
        if mat['material_id'] not in unique_materials:
            unique_materials[mat['material_id']] = mat
    
    all_materials = list(unique_materials.values())
    print(f"✅ Unique materials after deduplication: {len(all_materials)}")
    
    return all_materials

def save_structures_and_data(materials, output_dir="battery_materials_bulk_modulus"):
    """Save structures as CIF files and create training data"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "structures"), exist_ok=True)
    
    training_data = []
    failed_cifs = []
    
    print(f"\n💾 Saving {len(materials)} structures and data...")
    
    for i, mat in enumerate(tqdm(materials, desc="Saving structures")):
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
                'category': mat['category'],
                'chemical_system': mat['chemical_system'],
                'energy_above_hull': mat['energy_above_hull'],
                'formation_energy_per_atom': mat['formation_energy_per_atom'],
                'band_gap': mat['band_gap'],
                'density': mat['density'],
                'nsites': mat['nsites']
            })
            
        except Exception as e:
            print(f"Failed to save CIF for {mat['material_id']}: {e}")
            failed_cifs.append(mat['material_id'])
    
    # Save training data as CSV (for CGCNN)
    df = pd.DataFrame(training_data)
    csv_path = os.path.join(output_dir, "battery_materials_bulk_modulus.csv")
    
    # CGCNN format: no headers, columns: [cif_file, target_value]
    cgcnn_df = df[['cif_file', 'bulk_modulus']].copy()
    cgcnn_df.to_csv(csv_path, header=False, index=False)
    
    # Save detailed metadata
    metadata_path = os.path.join(output_dir, "battery_materials_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(training_data, f, indent=2, default=str)
    
    # Save summary statistics
    summary = {
        'total_materials': len(training_data),
        'failed_cifs': len(failed_cifs),
        'bulk_modulus_stats': {
            'mean': float(df['bulk_modulus'].mean()),
            'std': float(df['bulk_modulus'].std()),
            'min': float(df['bulk_modulus'].min()),
            'max': float(df['bulk_modulus'].max()),
            'median': float(df['bulk_modulus'].median())
        },
        'categories': df['category'].value_counts().to_dict(),
        'chemical_systems': df['chemical_system'].value_counts().to_dict()
    }
    
    summary_path = os.path.join(output_dir, "extraction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Extraction Summary:")
    print(f"   • Total materials: {summary['total_materials']}")
    print(f"   • Failed CIFs: {summary['failed_cifs']}")
    print(f"   • Bulk modulus range: {summary['bulk_modulus_stats']['min']:.1f} - {summary['bulk_modulus_stats']['max']:.1f} GPa")
    print(f"   • Mean bulk modulus: {summary['bulk_modulus_stats']['mean']:.1f} ± {summary['bulk_modulus_stats']['std']:.1f} GPa")
    print(f"   • Categories: {summary['categories']}")
    
    print(f"\n✅ Data saved to: {output_dir}/")
    print(f"   • Training CSV: {csv_path}")
    print(f"   • Metadata: {metadata_path}")
    print(f"   • Summary: {summary_path}")
    print(f"   • CIF files: {output_dir}/structures/")
    
    return training_data, summary

def main():
    """Main extraction pipeline"""
    
    print("🚀 Starting battery materials bulk modulus extraction...")
    
    # Extract materials data
    materials = extract_bulk_modulus_data()
    
    if not materials:
        print("❌ No materials found!")
        return
    
    # Save structures and training data
    training_data, summary = save_structures_and_data(materials)
    
    print(f"\n🎉 Extraction complete!")
    print(f"   Ready for CGCNN fine-tuning with {len(training_data)} materials")
    
    return training_data, summary

if __name__ == "__main__":
    main()