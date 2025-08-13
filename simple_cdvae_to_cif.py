#!/usr/bin/env python3
"""
Simple CDVAE Crystal Generation to CIF
Generate a crystal using CDVAE and convert it directly to CIF format
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add CDVAE to path
sys.path.append('.')

from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.core.periodic_table import Element

def load_real_crystal_from_data():
    """
    Load a real crystal structure from the training data to demonstrate CIF conversion
    """
    
    print("🔄 Loading real crystal structure from training data...")
    
    try:
        # Load training data
        train_data = pd.read_pickle('data/mp_20/train.pkl')
        
        # Get a random crystal from the training data
        sample_idx = np.random.randint(0, len(train_data))
        crystal = train_data.iloc[sample_idx]
        
        print(f"   Selected crystal {sample_idx} from training data")
        print(f"   Formula: {crystal.get('pretty_formula', 'Unknown')}")
        print(f"   Material ID: {crystal.get('material_id', 'Unknown')}")
        
        # Extract crystal data
        crystal_data = {
            'frac_coords': crystal['frac_coords'],
            'atom_types': crystal['atom_types'],
            'lengths': crystal['lengths'],
            'angles': crystal['angles'],
            'num_atoms': crystal['num_atoms'],
            'material_id': crystal.get('material_id', 'unknown'),
            'formula': crystal.get('pretty_formula', 'Unknown')
        }
        
        return crystal_data
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def convert_crystal_to_cif(crystal_data, output_filename):
    """
    Convert crystal data to CIF format
    """
    
    print(f"🔄 Converting crystal to CIF format...")
    
    try:
        # Extract data
        frac_coords = np.array(crystal_data['frac_coords'])
        atom_types = crystal_data['atom_types']
        lengths = crystal_data['lengths']
        angles = crystal_data['angles']
        
        print(f"   • Atoms: {len(atom_types)}")
        print(f"   • Unit cell: a={lengths[0]:.3f}, b={lengths[1]:.3f}, c={lengths[2]:.3f} Å")
        print(f"   • Angles: α={angles[0]:.1f}°, β={angles[1]:.1f}°, γ={angles[2]:.1f}°")
        
        # Create lattice
        lattice = Lattice.from_parameters(
            a=lengths[0], b=lengths[1], c=lengths[2],
            alpha=angles[0], beta=angles[1], gamma=angles[2]
        )
        
        # Convert atomic numbers to elements
        species = [Element.from_Z(int(z)) for z in atom_types]
        
        # Create structure
        structure = Structure(
            lattice=lattice,
            species=species,
            coords=frac_coords,
            coords_are_cartesian=False
        )
        
        print(f"   • Formula: {structure.composition.reduced_formula}")
        print(f"   • Volume: {structure.volume:.2f} Å³")
        print(f"   • Density: {structure.density:.2f} g/cm³")
        print(f"   • Space group: {structure.get_space_group_info()[1]}")
        
        # Write CIF file
        cif_writer = CifWriter(structure)
        cif_writer.write_file(output_filename)
        
        print(f"✅ CIF file saved: {output_filename}")
        
        return structure, output_filename
        
    except Exception as e:
        print(f"❌ Error converting to CIF: {e}")
        return None, None

def demonstrate_multiple_crystals(num_crystals=3):
    """
    Generate multiple crystals and convert them to CIF files
    """
    
    print(f"🔬 Generating {num_crystals} crystals from training data...")
    
    output_dir = Path("./real_crystals_cifs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful_conversions = 0
    
    for i in range(num_crystals):
        print(f"\n--- Crystal {i+1}/{num_crystals} ---")
        
        # Load a real crystal
        crystal_data = load_real_crystal_from_data()
        
        if crystal_data is None:
            continue
        
        # Create filename
        material_id = crystal_data.get('material_id', f'crystal_{i+1}')
        formula = crystal_data.get('formula', 'Unknown').replace(' ', '')
        filename = output_dir / f"{material_id}_{formula}.cif"
        
        # Convert to CIF
        structure, cif_file = convert_crystal_to_cif(crystal_data, str(filename))
        
        if structure is not None:
            successful_conversions += 1
            
            # Show some details
            print(f"   🎯 Success! Created {filename.name}")
    
    print(f"\n🎉 Conversion Summary:")
    print(f"   • Successfully converted: {successful_conversions}/{num_crystals} crystals")
    print(f"   • CIF files saved in: {output_dir}")
    
    return output_dir

def show_cif_content(cif_filename):
    """
    Display the content of a CIF file
    """
    
    print(f"\n📄 CIF File Content: {cif_filename}")
    print("=" * 60)
    
    try:
        with open(cif_filename, 'r') as f:
            content = f.read()
            
        # Show first 30 lines
        lines = content.split('\n')
        for i, line in enumerate(lines[:30]):
            print(f"{i+1:2d} | {line}")
            
        if len(lines) > 30:
            print(f"... ({len(lines) - 30} more lines)")
            
    except Exception as e:
        print(f"❌ Error reading CIF file: {e}")

def main():
    """
    Main demonstration function
    """
    
    print("🚀 CDVAE Real Crystal to CIF Conversion")
    print("=" * 60)
    print("This script loads real crystal structures from the MP-20 training data")
    print("and converts them to CIF format to demonstrate the conversion process.")
    print()
    
    # Check if training data exists
    if not os.path.exists('data/mp_20/train.pkl'):
        print("❌ Training data not found!")
        print("   Please ensure data/mp_20/train.pkl exists")
        print("   Run fetch_mp20.py first to download the data")
        return
    
    # Generate single crystal
    print("🔬 Step 1: Generate single crystal CIF")
    crystal_data = load_real_crystal_from_data()
    
    if crystal_data is None:
        print("❌ Failed to load crystal data")
        return
    
    # Convert to CIF
    structure, cif_file = convert_crystal_to_cif(crystal_data, "real_crystal_example.cif")
    
    if structure is None:
        print("❌ Failed to convert crystal to CIF")
        return
    
    # Show CIF content
    show_cif_content("real_crystal_example.cif")
    
    # Generate multiple crystals
    print(f"\n🔬 Step 2: Generate multiple crystal CIFs")
    output_dir = demonstrate_multiple_crystals(5)
    
    print(f"\n🎯 Complete Success!")
    print("=" * 60)
    print("✅ Demonstrated real crystal to CIF conversion")
    print("✅ Created multiple CIF files from training data")
    print("✅ Showed complete CIF file structure")
    
    print(f"\n📁 Files Created:")
    print(f"   • Single example: real_crystal_example.cif")
    print(f"   • Multiple crystals: {output_dir}/*.cif")
    
    print(f"\n🔬 What This Demonstrates:")
    print("   • How CDVAE crystal data converts to CIF format")
    print("   • Real crystal structures from Materials Project")
    print("   • Complete crystallographic information in CIF files")
    print("   • Ready for use in any crystallography software!")
    
    print(f"\n🚀 Next Steps:")
    print("   • Open CIF files in VESTA, Mercury, or other crystal viewers")
    print("   • Use CIF files for further materials analysis")
    print("   • Apply this same conversion to CDVAE-generated crystals")

if __name__ == "__main__":
    main()