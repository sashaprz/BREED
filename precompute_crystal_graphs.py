#!/usr/bin/env python3
"""
Pre-compute crystal graphs for fast CGCNN training
This will build all crystal graphs once and save them for GPU training
"""

import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import warnings
warnings.filterwarnings('ignore')


def build_crystal_graph(structure: Structure, cutoff: float = 8.0, max_neighbors: int = 20):
    """Build crystal graph from pymatgen Structure"""
    try:
        # Use CrystalNN for neighbor finding
        nn_strategy = CrystalNN()
        
        # Get all neighbors
        all_neighbors = []
        for i, site in enumerate(structure):
            neighbors = nn_strategy.get_nn_info(structure, i)
            for neighbor in neighbors[:max_neighbors]:  # Limit neighbors
                j = neighbor['site_index']
                distance = neighbor['weight']  # CrystalNN uses weight as distance
                if distance <= cutoff:
                    all_neighbors.append((i, j, distance))
        
        # Create edge indices and distances
        if not all_neighbors:
            # Fallback: create self-loops
            num_atoms = len(structure)
            edge_indices = [[i, i] for i in range(num_atoms)]
            distances = [0.1] * num_atoms  # Small distance for self-loops
        else:
            edge_indices = [(i, j) for i, j, d in all_neighbors]
            distances = [d for i, j, d in all_neighbors]
        
        # Convert to arrays
        edge_index = np.array(edge_indices).T
        distances = np.array(distances)
        
        # Get atomic numbers
        atom_types = [site.specie.Z for site in structure]
        
        # Get fractional coordinates
        frac_coords = structure.frac_coords
        
        # Get lattice parameters
        lattice = structure.lattice
        lengths = [lattice.a, lattice.b, lattice.c]
        angles = [lattice.alpha, lattice.beta, lattice.gamma]
        
        return {
            'frac_coords': frac_coords,
            'atom_types': np.array(atom_types),
            'edge_index': edge_index,
            'distances': distances,
            'lengths': lengths,
            'angles': angles,
            'num_atoms': len(structure)
        }
        
    except Exception as e:
        print(f"Error building crystal graph: {e}")
        # Fallback: minimal graph
        num_atoms = len(structure)
        return {
            'frac_coords': structure.frac_coords,
            'atom_types': np.array([site.specie.Z for site in structure]),
            'edge_index': np.array([[i, i] for i in range(num_atoms)]).T,
            'distances': np.array([0.1] * num_atoms),
            'lengths': [structure.lattice.a, structure.lattice.b, structure.lattice.c],
            'angles': [structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma],
            'num_atoms': num_atoms
        }


def main():
    """Pre-compute crystal graphs for the dataset"""
    print("Loading original dataset...")
    
    # Load original dataset
    with open('data/ionic_conductivity_dataset.pkl', 'rb') as f:
        original_data = pickle.load(f)
    
    print(f"Processing {len(original_data)} samples...")
    
    # Pre-compute crystal graphs
    processed_data = []
    failed_count = 0
    
    for i, sample in enumerate(tqdm(original_data, desc="Building crystal graphs")):
        try:
            # Parse CIF structure
            structure = Structure.from_str(sample['cif'], fmt='cif')
            
            # Build crystal graph
            graph_data = build_crystal_graph(structure, cutoff=8.0, max_neighbors=20)
            
            # Create processed sample with pre-computed graph
            processed_sample = {
                'material_id': sample['material_id'],
                'ionic_conductivity': sample['ionic_conductivity'],
                'cif_path': sample['cif_path'],
                # Pre-computed crystal graph data
                'atom_types': graph_data['atom_types'],
                'edge_index': graph_data['edge_index'],
                'distances': graph_data['distances'],
                'frac_coords': graph_data['frac_coords'],
                'lengths': graph_data['lengths'],
                'angles': graph_data['angles'],
                'num_atoms': graph_data['num_atoms']
            }
            
            processed_data.append(processed_sample)
            
        except Exception as e:
            print(f"Failed to process {sample['material_id']}: {e}")
            failed_count += 1
            continue
    
    print(f"Successfully processed {len(processed_data)} samples")
    print(f"Failed to process {failed_count} samples")
    
    # Save processed dataset
    output_path = 'data/ionic_conductivity_dataset_with_graphs.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Saved pre-computed crystal graphs to {output_path}")
    print(f"Dataset size: {len(processed_data)} samples")
    
    # Show sample structure
    if processed_data:
        sample = processed_data[0]
        print(f"\nSample structure for {sample['material_id']}:")
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} {value.dtype}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()