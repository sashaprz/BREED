#!/usr/bin/env python3
"""
CDVAE to CIF Converter
Convert CDVAE model outputs to CIF (Crystallographic Information File) format

The CDVAE model outputs:
- frac_coords: Fractional coordinates of atoms [N_atoms, 3]
- atom_types: Atomic numbers [N_atoms]  
- lengths: Unit cell lengths [a, b, c] in Angstroms
- angles: Unit cell angles [alpha, beta, gamma] in degrees
- num_atoms: Number of atoms in the structure

This script shows how to convert these to CIF files using pymatgen.
"""

import numpy as np
import torch
import json
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.core.periodic_table import Element

def create_structure_from_cdvae_output(frac_coords, atom_types, lengths, angles):
    """
    Convert CDVAE model output to pymatgen Structure object
    
    Args:
        frac_coords (array): Fractional coordinates [N_atoms, 3]
        atom_types (array): Atomic numbers [N_atoms]
        lengths (array): Unit cell lengths [a, b, c] in Angstroms
        angles (array): Unit cell angles [alpha, beta, gamma] in degrees
        
    Returns:
        pymatgen.Structure: Crystal structure object
    """
    
    # Create lattice from unit cell parameters
    lattice = Lattice.from_parameters(
        a=lengths[0], b=lengths[1], c=lengths[2],
        alpha=angles[0], beta=angles[1], gamma=angles[2]
    )
    
    # Convert atomic numbers to element symbols
    species = [Element.from_Z(int(z)) for z in atom_types]
    
    # Create structure
    structure = Structure(
        lattice=lattice,
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False  # Using fractional coordinates
    )
    
    return structure

def save_structure_as_cif(structure, filename, title="Generated Crystal"):
    """
    Save pymatgen Structure as CIF file
    
    Args:
        structure (Structure): pymatgen Structure object
        filename (str): Output CIF filename
        title (str): Title for the CIF file
    """
    
    # Create CIF writer
    cif_writer = CifWriter(structure)
    
    # Write to file
    cif_writer.write_file(filename)
    
    print(f"✅ Saved CIF file: {filename}")
    return filename

def demonstrate_cif_conversion():
    """
    Demonstrate how to convert CDVAE output to CIF format
    """
    
    print("🔬 CDVAE to CIF Conversion Demo")
    print("=" * 50)
    
    # Example CDVAE model output (simulated)
    print("📊 Example CDVAE Model Output:")
    
    # Generate example crystal structure data
    num_atoms = 8
    
    # Fractional coordinates (8 atoms in unit cell)
    frac_coords = np.array([
        [0.0, 0.0, 0.0],      # Atom 1
        [0.5, 0.5, 0.0],      # Atom 2  
        [0.5, 0.0, 0.5],      # Atom 3
        [0.0, 0.5, 0.5],      # Atom 4
        [0.25, 0.25, 0.25],   # Atom 5
        [0.75, 0.75, 0.25],   # Atom 6
        [0.75, 0.25, 0.75],   # Atom 7
        [0.25, 0.75, 0.75],   # Atom 8
    ])
    
    # Atomic numbers (Silicon and Oxygen for SiO2-like structure)
    atom_types = np.array([14, 8, 8, 8, 14, 8, 8, 8])  # Si=14, O=8
    
    # Unit cell parameters
    lengths = np.array([5.43, 5.43, 5.43])  # Cubic cell, ~5.43 Å
    angles = np.array([90.0, 90.0, 90.0])   # Cubic angles
    
    print(f"   • Number of atoms: {num_atoms}")
    print(f"   • Fractional coordinates shape: {frac_coords.shape}")
    print(f"   • Atom types: {atom_types}")
    print(f"   • Unit cell lengths: {lengths} Å")
    print(f"   • Unit cell angles: {angles}°")
    
    # Convert to pymatgen Structure
    print(f"\n🏗️ Converting to pymatgen Structure...")
    structure = create_structure_from_cdvae_output(
        frac_coords, atom_types, lengths, angles
    )
    
    print(f"✅ Structure created successfully!")
    print(f"   • Formula: {structure.composition.reduced_formula}")
    print(f"   • Space group: {structure.get_space_group_info()[1]}")
    print(f"   • Volume: {structure.volume:.2f} Å³")
    print(f"   • Density: {structure.density:.2f} g/cm³")
    
    # Save as CIF file
    print(f"\n💾 Saving as CIF file...")
    cif_filename = "generated_crystal_example.cif"
    save_structure_as_cif(structure, cif_filename, "CDVAE Generated Crystal")
    
    # Show CIF file content
    print(f"\n📄 CIF File Content Preview:")
    with open(cif_filename, 'r') as f:
        cif_content = f.read()
        print(cif_content[:500] + "..." if len(cif_content) > 500 else cif_content)
    
    return structure, cif_filename

def batch_convert_cdvae_to_cif(cdvae_outputs, output_dir="./cif_files"):
    """
    Convert multiple CDVAE outputs to CIF files
    
    Args:
        cdvae_outputs (list): List of CDVAE output dictionaries
        output_dir (str): Directory to save CIF files
        
    Returns:
        list: List of created CIF filenames
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cif_files = []
    
    print(f"\n🔄 Converting {len(cdvae_outputs)} structures to CIF format...")
    
    for i, output in enumerate(cdvae_outputs):
        try:
            # Extract CDVAE output data
            frac_coords = np.array(output['frac_coords'])
            atom_types = np.array(output['atom_types'])
            lengths = np.array(output['lengths'])
            angles = np.array(output['angles'])
            
            # Create structure
            structure = create_structure_from_cdvae_output(
                frac_coords, atom_types, lengths, angles
            )
            
            # Save as CIF
            cif_filename = output_path / f"crystal_{i+1:03d}.cif"
            save_structure_as_cif(
                structure, 
                str(cif_filename), 
                f"CDVAE Generated Crystal {i+1}"
            )
            
            cif_files.append(str(cif_filename))
            
            print(f"   ✅ Crystal {i+1}: {structure.composition.reduced_formula} "
                  f"({structure.num_sites} atoms)")
            
        except Exception as e:
            print(f"   ❌ Error converting crystal {i+1}: {e}")
    
    print(f"\n🎉 Conversion complete! Created {len(cif_files)} CIF files in {output_path}")
    return cif_files

def create_cif_analysis_script(output_dir):
    """Create a script to analyze the generated CIF files"""
    
    analysis_script = f'''#!/usr/bin/env python3
"""
CIF File Analysis Script
Analyze the generated CIF files and extract key properties
"""

import os
from pathlib import Path
from pymatgen.io.cif import CifParser
import pandas as pd

def analyze_cif_files(cif_dir):
    """Analyze all CIF files in a directory"""
    
    cif_files = list(Path(cif_dir).glob("*.cif"))
    print(f"Found {{len(cif_files)}} CIF files")
    
    results = []
    
    for cif_file in cif_files:
        try:
            parser = CifParser(str(cif_file))
            structure = parser.get_structures()[0]
            
            result = {{
                'filename': cif_file.name,
                'formula': structure.composition.reduced_formula,
                'num_atoms': structure.num_sites,
                'volume': structure.volume,
                'density': structure.density,
                'space_group': structure.get_space_group_info()[1],
                'lattice_a': structure.lattice.a,
                'lattice_b': structure.lattice.b,
                'lattice_c': structure.lattice.c,
                'alpha': structure.lattice.alpha,
                'beta': structure.lattice.beta,
                'gamma': structure.lattice.gamma,
            }}
            
            results.append(result)
            print(f"✅ {{cif_file.name}}: {{result['formula']}} ({{result['num_atoms']}} atoms)")
            
        except Exception as e:
            print(f"❌ Error analyzing {{cif_file.name}}: {{e}}")
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    df.to_csv("cif_analysis_summary.csv", index=False)
    print(f"\\n📊 Analysis saved to: cif_analysis_summary.csv")
    
    return df

if __name__ == "__main__":
    df = analyze_cif_files("{output_dir}")
    print(f"\\n📋 Summary Statistics:")
    print(df.describe())
'''
    
    analysis_file = Path(output_dir) / "analyze_cifs.py"
    with open(analysis_file, 'w') as f:
        f.write(analysis_script)
    
    print(f"   ✅ Created CIF analysis script: {analysis_file}")
    return analysis_file

def main():
    """Main demonstration function"""
    
    # Demonstrate single structure conversion
    structure, cif_file = demonstrate_cif_conversion()
    
    # Create example batch data
    print(f"\n🔄 Creating example batch conversion...")
    
    example_outputs = []
    for i in range(3):
        # Generate different example structures
        num_atoms = np.random.randint(4, 12)
        
        output = {
            'frac_coords': np.random.rand(num_atoms, 3).tolist(),
            'atom_types': np.random.choice([6, 8, 14, 26], num_atoms).tolist(),  # C, O, Si, Fe
            'lengths': (np.random.rand(3) * 5 + 5).tolist(),  # 5-10 Å
            'angles': (np.random.rand(3) * 20 + 80).tolist(),  # 80-100°
            'num_atoms': num_atoms
        }
        example_outputs.append(output)
    
    # Convert batch to CIF
    cif_files = batch_convert_cdvae_to_cif(example_outputs, "./example_cifs")
    
    # Create analysis script
    create_cif_analysis_script("./example_cifs")
    
    print(f"\n🎯 CIF Conversion Complete!")
    print("=" * 50)
    print("✅ Demonstrated single structure to CIF conversion")
    print("✅ Demonstrated batch conversion")
    print("✅ Created CIF analysis script")
    
    print(f"\n📁 Files Created:")
    print(f"   • Single CIF: {cif_file}")
    print(f"   • Batch CIFs: {len(cif_files)} files in ./example_cifs/")
    print(f"   • Analysis script: ./example_cifs/analyze_cifs.py")
    
    print(f"\n🚀 Usage with Real CDVAE Output:")
    print("   1. Load your trained CDVAE model")
    print("   2. Generate crystals: outputs = model.decode(latent_vectors)")
    print("   3. Convert to CIF: batch_convert_cdvae_to_cif(outputs)")
    print("   4. Analyze results with the generated analysis script")
    print("   5. Use CIF files in any crystallography software!")

if __name__ == "__main__":
    main()