#!/usr/bin/env python3
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
    print(f"Found {len(cif_files)} CIF files")
    
    results = []
    
    for cif_file in cif_files:
        try:
            parser = CifParser(str(cif_file))
            structure = parser.get_structures()[0]
            
            result = {
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
            }
            
            results.append(result)
            print(f"✅ {cif_file.name}: {result['formula']} ({result['num_atoms']} atoms)")
            
        except Exception as e:
            print(f"❌ Error analyzing {cif_file.name}: {e}")
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    df.to_csv("cif_analysis_summary.csv", index=False)
    print(f"\n📊 Analysis saved to: cif_analysis_summary.csv")
    
    return df

if __name__ == "__main__":
    df = analyze_cif_files("./example_cifs")
    print(f"\n📋 Summary Statistics:")
    print(df.describe())
