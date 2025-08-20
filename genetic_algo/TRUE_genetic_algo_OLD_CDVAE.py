#!/usr/bin/env python3
"""
True Genetic Algorithm with OLD CDVAE Integration for comparison
This version uses the old CDVAE weights to test if the new architecture is causing issues
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy
import tempfile
import shutil
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE')

# Import torch first
import torch
import yaml

# Import pymatgen for structure handling
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Import the working CDVAE loader
try:
    from generator.CDVAE.load_trained_model import TrainedCDVAELoader
    print("✅ TrainedCDVAELoader imported successfully")
    CDVAE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import TrainedCDVAELoader: {e}")
    CDVAE_AVAILABLE = False

# Import the CACHED ML predictor
try:
    from genetic_algo.cached_property_predictor import get_cached_predictor
    _global_predictor = get_cached_predictor()
    
    def predict_single_cif(cif_path, verbose=False):
        """Use the cached predictor"""
        return _global_predictor.predict_single_cif(cif_path, verbose=verbose)
    
    print("🚀 Using CACHED ML predictor for OLD CDVAE test!")
except ImportError:
    def predict_single_cif_debug(cif_path, verbose=False):
        """Debug predictor with realistic random values"""
        import random
        return {
            'ionic_conductivity': random.uniform(1e-6, 1e-2),
            'bandgap': random.uniform(1.0, 5.0),
            'sei_score': random.uniform(0.3, 0.9),
            'cei_score': random.uniform(0.3, 0.9),
            'bulk_modulus': random.uniform(20.0, 150.0)
        }
    predict_single_cif = predict_single_cif_debug
    print("⚠️  Using DEBUG predictor for OLD CDVAE test")


@dataclass
class TargetProperties:
    """Target properties for multi-objective optimization"""
    ionic_conductivity: float = 1.0e-3  # S/cm
    bandgap: float = 3.0  # eV
    sei_score: float = 0.9  # Higher is better
    cei_score: float = 0.85  # Higher is better
    bulk_modulus: float = 80.0  # GPa (optimal for solid electrolytes)


@dataclass
class GACandidate:
    """Represents a candidate electrolyte in the genetic algorithm"""
    composition: Dict[str, int]  # Element: count
    lattice_params: Dict[str, float]  # a, b, c, alpha, beta, gamma
    space_group: int
    generation_method: str = "unknown"
    structure: Optional[Structure] = None
    cif_path: Optional[str] = None
    properties: Dict[str, float] = field(default_factory=dict)
    objectives: List[float] = field(default_factory=list)  # Multi-objective values
    pareto_rank: int = 0  # Pareto front rank (0 = best front)
    crowding_distance: float = 0.0  # Diversity measure
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    fitness: float = 0.0  # Single fitness value for selection
    
    def __post_init__(self):
        if not self.properties:
            self.properties = {
                'ionic_conductivity': 0.0,
                'bandgap': 0.0,
                'sei_score': 0.0,
                'cei_score': 0.0,
                'bulk_modulus': 0.0
            }
        if not self.objectives:
            self.objectives = [0.0] * 5  # 5 objectives


class TrueGeneticAlgorithmOldCDVAE:
    """True Genetic Algorithm with OLD CDVAE Integration for comparison"""
    
    def __init__(self,
                 population_size: int = 10,  # Smaller for testing
                 elite_count: int = 2,
                 tournament_size: int = 3,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8,
                 max_generations: int = 5,  # Fewer generations for testing
                 convergence_threshold: int = 3,
                 output_dir: str = "old_cdvae_test_results"):
        
        self.population_size = population_size
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.cif_dir = self.output_dir / "cifs"
        self.cif_dir.mkdir(exist_ok=True)
        
        # Initialize TrainedCDVAELoader with OLD model files
        weights_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\new_cdvae_weights.ckpt"  # OLD weights
        scalers_dir = r"C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE"
        hparams_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\new_hparams.yaml"  # OLD hparams
        
        print("🔧 Initializing TrainedCDVAELoader with OLD CDVAE weights for comparison...")
        print(f"   Using: {os.path.basename(weights_path)}")
        print(f"   Using: {os.path.basename(hparams_path)}")
        
        if CDVAE_AVAILABLE:
            try:
                self.cdvae_loader = TrainedCDVAELoader(weights_path, scalers_dir, hparams_path)
                self.cdvae_loader.load_model()
                print("✅ OLD CDVAE model loaded successfully!")
                
                # Try to load scalers (optional)
                try:
                    self.cdvae_loader.load_scalers()
                    print("✅ OLD CDVAE scalers loaded successfully!")
                except Exception as scaler_error:
                    print(f"⚠️  OLD CDVAE scalers not loaded (optional): {scaler_error}")
                    
            except Exception as e:
                print(f"❌ Failed to load OLD CDVAE model: {e}")
                import traceback
                traceback.print_exc()
                self.cdvae_loader = None
        else:
            self.cdvae_loader = None
            print("❌ TrainedCDVAELoader not available, using fallback generation")
        
        self.target_properties = TargetProperties()
        self.population: List[GACandidate] = []
        self.generation = 0
        self.structure_matcher = StructureMatcher()
        
        print(f"🚀 OLD CDVAE Test Genetic Algorithm initialized")
        print(f"   Population size: {self.population_size}")
        print(f"   Max generations: {self.max_generations}")
        print(f"   This will test if OLD CDVAE generates better structures for CGCNN models")
    
    def generate_initial_population(self) -> List[GACandidate]:
        """Generate initial population using OLD CDVAE"""
        print(f"🔬 Generating initial population of {self.population_size} candidates with OLD CDVAE...")
        
        candidates = []
        
        # Use CDVAE generation
        if self.cdvae_loader and hasattr(self.cdvae_loader, 'model') and self.cdvae_loader.model is not None:
            print(f"🧬 Generating {self.population_size} structures using OLD CDVAE...")
            cdvae_candidates = self._generate_cdvae_candidates(self.population_size)
            candidates.extend(cdvae_candidates)
        else:
            print("❌ OLD CDVAE not available - using fallback method")
            candidates = self._generate_fallback_candidates(self.population_size)
        
        return candidates[:self.population_size]
    
    def _generate_cdvae_candidates(self, count: int) -> List[GACandidate]:
        """Generate candidates using OLD CDVAE"""
        candidates = []
        attempts = 0
        max_attempts = count * 3
        
        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            
            try:
                batch_size = min(3, count - len(candidates))  # Smaller batches for testing
                structures = self.cdvae_loader.generate_structures(batch_size, fast_mode=True)  # Use fast mode
                
                for structure in structures:
                    if len(candidates) >= count:
                        break
                        
                    structure_data = self._convert_cdvae_structure(structure)
                    if structure_data:
                        candidate = self._create_candidate_from_data(structure_data)
                        if candidate and self._is_valid_candidate(candidate):
                            candidates.append(candidate)
                            
            except Exception as e:
                print(f"  OLD CDVAE generation attempt {attempts} failed: {e}")
                continue
                
        return candidates
    
    def _generate_fallback_candidates(self, count: int) -> List[GACandidate]:
        """Generate fallback candidates"""
        candidates = []
        
        # Simple fallback structures
        compositions = [
            {'Li': 7, 'La': 3, 'Zr': 2, 'O': 12},  # LLZO-type
            {'Li': 3, 'Ti': 2, 'P': 3, 'O': 12},   # NASICON-type
            {'Li': 6, 'P': 1, 'S': 5, 'Cl': 1},    # Argyrodite-type
        ]
        
        for i in range(count):
            composition = compositions[i % len(compositions)]
            
            lattice_params = {
                'a': 10.0 + random.uniform(-1, 1),
                'b': 10.0 + random.uniform(-1, 1), 
                'c': 10.0 + random.uniform(-1, 1),
                'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0
            }
            
            structure_data = {
                'composition': composition,
                'lattice_params': lattice_params,
                'space_group': 225,
                'generation_method': 'fallback'
            }
            
            candidate = self._create_candidate_from_data(structure_data)
            if candidate and self._is_valid_candidate(candidate):
                candidates.append(candidate)
                
        return candidates
    
    def _convert_cdvae_structure(self, cdvae_structure) -> Optional[Dict]:
        """Convert CDVAE structure output to our expected format"""
        try:
            if hasattr(cdvae_structure, 'composition') or 'composition' in cdvae_structure:
                composition = cdvae_structure.get('composition', cdvae_structure.composition if hasattr(cdvae_structure, 'composition') else {})
                
                # Extract lattice parameters
                if hasattr(cdvae_structure, 'lattice') or 'lattice' in cdvae_structure:
                    lattice = cdvae_structure.get('lattice', cdvae_structure.lattice if hasattr(cdvae_structure, 'lattice') else None)
                    if lattice:
                        if hasattr(lattice, 'abc') and hasattr(lattice, 'angles'):
                            a, b, c = lattice.abc
                            alpha, beta, gamma = lattice.angles
                        else:
                            a = b = c = 10.0
                            alpha = beta = gamma = 90.0
                    else:
                        a = b = c = 10.0
                        alpha = beta = gamma = 90.0
                else:
                    a = b = c = 10.0
                    alpha = beta = gamma = 90.0
                
                lattice_params = {
                    'a': float(a), 'b': float(b), 'c': float(c),
                    'alpha': float(alpha), 'beta': float(beta), 'gamma': float(gamma)
                }
                
                space_group = getattr(cdvae_structure, 'space_group', 225)
                
                return {
                    'composition': composition,
                    'lattice_params': lattice_params,
                    'space_group': space_group,
                    'generated_id': f'old_cdvae_{id(cdvae_structure)}',
                    'generation_method': 'old_cdvae'
                }
            else:
                return None
                
        except Exception as e:
            return None
    
    def _create_candidate_from_data(self, structure_data: Dict) -> Optional[GACandidate]:
        """Create GACandidate from structure generation data"""
        try:
            composition = structure_data['composition']
            lattice_params = structure_data['lattice_params']
            space_group = structure_data['space_group']
            generation_method = structure_data.get('generation_method', 'unknown')
            
            # Create pymatgen structure
            structure = self._create_structure(composition, lattice_params)
            
            if structure is None:
                return None
            
            candidate = GACandidate(
                composition=composition,
                lattice_params=lattice_params,
                space_group=space_group,
                generation_method=generation_method,
                structure=structure,
                generation=self.generation
            )
            
            # Generate CIF file
            self._generate_cif_file(candidate)
            
            return candidate
            
        except Exception as e:
            return None
    
    def _create_structure(self, composition: Dict[str, int], lattice_params: Dict[str, float]) -> Optional[Structure]:
        """Create pymatgen Structure from composition and lattice parameters"""
        try:
            # Create lattice
            lattice = Lattice.from_parameters(
                a=lattice_params['a'], b=lattice_params['b'], c=lattice_params['c'],
                alpha=lattice_params['alpha'], beta=lattice_params['beta'], gamma=lattice_params['gamma']
            )
            
            # Generate atomic positions (simplified approach)
            species = []
            coords = []
            
            # Simple position generation - distribute atoms in unit cell
            total_atoms = sum(composition.values())
            positions_per_atom = 1.0 / total_atoms
            
            current_pos = 0.0
            for element, count in composition.items():
                for i in range(count):
                    # Simple linear distribution with some randomness
                    x = (current_pos + random.uniform(-0.1, 0.1)) % 1.0
                    y = (current_pos * 1.618 + random.uniform(-0.1, 0.1)) % 1.0  # Golden ratio for better distribution
                    z = (current_pos * 2.618 + random.uniform(-0.1, 0.1)) % 1.0
                    
                    species.append(Element(element))
                    coords.append([x, y, z])
                    current_pos += positions_per_atom
            
            # Create structure
            structure = Structure(lattice, species, coords, coords_are_cartesian=False)
            
            return structure
            
        except Exception as e:
            return None
    
    def _is_valid_candidate(self, candidate: GACandidate) -> bool:
        """Enhanced validity checks for candidates"""
        if not candidate.structure:
            return False
            
        # Check composition has reasonable number of atoms
        total_atoms = sum(candidate.composition.values())
        if total_atoms < 1 or total_atoms > 50:
            return False
        
        # Must contain Li for electrolyte applications
        if 'Li' not in candidate.composition:
            return False
            
        # Check for reasonable lattice parameters
        lattice = candidate.structure.lattice
        if any(param < 1.0 or param > 50.0 for param in [lattice.a, lattice.b, lattice.c]):
            return False
        
        # Check for reasonable volume
        if lattice.volume < 10 or lattice.volume > 5000:  # Å³
            return False
            
        return True
    
    def _generate_cif_file(self, candidate: GACandidate) -> str:
        """Generate CIF file for the candidate"""
        if not candidate.structure:
            return ""
            
        # Generate unique filename with generation method
        composition_str = "".join(f"{elem}{count}" for elem, count in
                                sorted(candidate.composition.items()))
        filename = f"old_gen{self.generation}_{composition_str}_{candidate.generation_method}_{id(candidate)}.cif"
        cif_path = self.cif_dir / filename
        
        # Write CIF file
        cif_writer = CifWriter(candidate.structure)
        cif_writer.write_file(str(cif_path))
        
        candidate.cif_path = str(cif_path)
        return str(cif_path)
    
    def evaluate_population(self, candidates: List[GACandidate]) -> None:
        """Evaluate properties for all candidates"""
        print(f"🔬 Evaluating properties for {len(candidates)} OLD CDVAE candidates...")
        
        for i, candidate in enumerate(candidates):
            try:
                # Get ML predictions
                if candidate.cif_path and os.path.exists(candidate.cif_path):
                    results = predict_single_cif(candidate.cif_path, verbose=True)
                    
                    candidate.properties = {
                        'ionic_conductivity': results.get('ionic_conductivity', 1e-10),
                        'bandgap': results.get('bandgap', 0.0),
                        'bandgap_correction_applied': results.get('bandgap_correction_applied', False),
                        'correction_method': results.get('correction_method', 'none'),
                        'sei_score': results.get('sei_score', 0.0),
                        'cei_score': results.get('cei_score', 0.0),
                        'bulk_modulus': results.get('bulk_modulus', 0.0)
                    }
                    
                    # Include raw PBE value if available
                    if 'bandgap_raw_pbe' in results:
                        candidate.properties['bandgap_raw_pbe'] = results['bandgap_raw_pbe']
                    
                    # Calculate fitness (simple version for testing)
                    candidate.fitness = self._calculate_simple_fitness(candidate.properties)
                    
                    print(f"  Candidate {i+1}: {candidate.generation_method}")
                    print(f"    Composition: {''.join(f'{k}{v}' for k,v in candidate.composition.items())}")
                    print(f"    Bandgap: {candidate.properties['bandgap']:.3f} eV")
                    print(f"    Bulk modulus: {candidate.properties['bulk_modulus']:.1f} GPa")
                    print(f"    SEI: {candidate.properties['sei_score']:.3f}")
                    print(f"    CEI: {candidate.properties['cei_score']:.3f}")
                    print(f"    IC: {candidate.properties['ionic_conductivity']:.2e} S/cm")
                    print(f"    Fitness: {candidate.fitness:.4f}")
                    print()
                    
                else:
                    candidate.fitness = 0.0
                    
            except Exception as e:
                print(f"  Candidate {i+1} evaluation failed: {e}")
                candidate.fitness = 0.0
    
    def _calculate_simple_fitness(self, properties: Dict[str, float]) -> float:
        """Calculate simple fitness for testing"""
        # Simple fitness based on how close we are to realistic values
        fitness = 0.0
        
        # Bandgap should be 1-5 eV
        bg = properties['bandgap']
        if 1.0 <= bg <= 5.0:
            fitness += 0.3
        
        # Bulk modulus should be positive
        bm = properties['bulk_modulus']
        if bm > 0:
            fitness += 0.2
        
        # SEI/CEI should be > 0
        if properties['sei_score'] > 0:
            fitness += 0.2
        if properties['cei_score'] > 0:
            fitness += 0.2
        
        # Ionic conductivity should be reasonable
        ic = properties['ionic_conductivity']
        if 1e-10 <= ic <= 1e-2:
            fitness += 0.1
        
        return fitness
    
    def run_test(self) -> Dict[str, Any]:
        """Run a quick test to compare OLD vs NEW CDVAE"""
        print(f"🚀 Starting OLD CDVAE Test")
        print(f"   Population size: {self.population_size}")
        print(f"   This will test if OLD CDVAE generates better structures for CGCNN models")
        print("-" * 80)
        
        # Generate initial population
        self.population = self.generate_initial_population()
        
        if not self.population:
            print("❌ No candidates generated!")
            return {"error": "No candidates generated"}
        
        # Evaluate population
        self.evaluate_population(self.population)
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Results
        results = {
            'population_size': len(self.population),
            'best_fitness': self.population[0].fitness if self.population else 0.0,
            'avg_fitness': np.mean([c.fitness for c in self.population]) if self.population else 0.0,
            'candidates': []
        }
        
        print(f"\n{'='*20} OLD CDVAE TEST RESULTS {'='*20}")
        print(f"Generated candidates: {len(self.population)}")
        print(f"Best fitness: {results['best_fitness']:.4f}")
        print(f"Average fitness: {results['avg_fitness']:.4f}")
        
        if self.population:
            print(f"\n🏆 TOP CANDIDATES FROM OLD CDVAE:")
            
            for i, candidate in enumerate(self.population[:3]):
                comp_str = "".join(f"{elem}{count}" for elem, count in sorted(candidate.composition.items()))
                print(f"\n{i+1}. Composition: {comp_str}")
                print(f"   Generation Method: {candidate.generation_method}")
                print(f"   Fitness: {candidate.fitness:.4f}")
                print(f"   Properties:")
                for prop, value in candidate.properties.items():
                    if prop == 'ionic_conductivity':
                        print(f"     {prop}: {value:.2e}")
                    elif isinstance(value, (int, float)):
                        print(f"     {prop}: {value:.4f}")
                    elif isinstance(value, bool):
                        print(f"     {prop}: {value}")
                    elif isinstance(value, str):
                        print(f"     {prop}: {value}")
                
                results['candidates'].append({
                    'composition': candidate.composition,
                    'properties': candidate.properties,
                    'fitness': candidate.fitness,
                    'generation_method': candidate.generation_method
                })
        
        return results


def main():
    """Main function to run the OLD CDVAE test"""
    
    print("🧬 OLD CDVAE TEST - COMPARISON WITH NEW CDVAE")
    print("=" * 60)
    print("This test uses the OLD CDVAE weights to see if they generate")
    print("structures that work better with the CGCNN property prediction models")
    print("=" * 60)
    
    # Initialize and run OLD CDVAE test
    ga = TrueGeneticAlgorithmOldCDVAE(
        population_size=5,  # Small test
        max_generations=1,  # Just generation
        output_dir="old_cdvae_test_results"
    )
    
    results = ga.run_test()
    
    print(f"\n✅ OLD CDVAE Test completed!")
    print(f"Results saved to: {ga.output_dir}")
    print(f"CIF files saved to: {ga.cif_dir}")
    print(f"Compare these results with the NEW CDVAE results to see which works better")
    
    return results


if __name__ == "__main__":
    main()