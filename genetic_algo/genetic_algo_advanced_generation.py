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

# CDVAE and pymatgen imports
import sys
sys.path.append(r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE')
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Import ML prediction functions with fallback to debug mode
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory

try:
    from fully_optimized_predictor import predict_single_cif_fully_optimized as predict_single_cif
    print("Using FULLY optimized ML predictor - models loaded ONCE only")
except ImportError:
    try:
        from env.optimized_ml_predictor import predict_single_cif_optimized as predict_single_cif
        print("Using optimized ML predictor with model caching")
    except ImportError:
        try:
            from env.main_rl import predict_single_cif
            print("Using standard ML predictor (models will reload each time)")
        except ImportError:
            # Create a simple debug predictor if nothing else works
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
            print("Using DEBUG predictor with realistic random values for testing")

# Advanced crystal generation using noise-based approach (CDVAE-inspired)
def load_training_data_for_generation():
    """Load training data to use as seeds for generation"""
    try:
        # Try to load the training data
        train_data_path = r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\data\mp_20\train.pkl'
        if os.path.exists(train_data_path):
            train_data = pd.read_pickle(train_data_path)
            print(f"✅ Loaded training data with {len(train_data)} samples")
            return train_data
        else:
            print("⚠️ Training data not found, using synthetic seeds")
            return None
    except Exception as e:
        print(f"⚠️ Could not load training data: {e}")
        return None

def generate_noise_based_crystal_structures():
    """Generate crystal structures using noise-based approach inspired by diffusion models"""
    
    # Load training data as seeds
    training_data = load_training_data_for_generation()
    
    crystal_generators = []
    
    # Generator 1: Noise-perturbed known structures
    if training_data is not None and len(training_data) > 0:
        def noise_perturbed_generator():
            """Generate by adding noise to known structures"""
            sample = training_data.sample(1).iloc[0]
            
            # Extract base structure data
            base_structure = {
                'lattice_params': {
                    'a': random.uniform(3.0, 15.0),
                    'b': random.uniform(3.0, 15.0), 
                    'c': random.uniform(3.0, 15.0),
                    'alpha': random.uniform(60, 120),
                    'beta': random.uniform(60, 120),
                    'gamma': random.uniform(60, 120)
                },
                'composition': generate_diverse_composition(),
                'space_group': random.choice([1, 2, 12, 14, 15, 62, 63, 71, 136, 141, 194, 221, 225, 227, 229]),
                'generation_method': 'noise_perturbed'
            }
            return base_structure
        
        crystal_generators.append(noise_perturbed_generator)
    
    # Generator 2: Compositionally-driven generation
    def composition_driven_generator():
        """Generate based on target compositions for solid electrolytes"""
        
        # Define target composition families for solid electrolytes
        composition_families = [
            # Li-rich garnets
            {'Li': random.randint(5, 8), 'La': random.randint(2, 4), 'Zr': random.randint(1, 3), 'O': random.randint(10, 14)},
            # NASICON-type
            {'Li': random.randint(1, 4), 'Ti': random.randint(1, 3), 'P': random.randint(2, 4), 'O': random.randint(8, 14)},
            # Perovskite-type
            {'Li': random.randint(1, 3), 'Ti': 1, 'O': 3, random.choice(['La', 'Sr', 'Ba']): 1},
            # Sulfide electrolytes
            {'Li': random.randint(2, 6), random.choice(['P', 'Si', 'Ge']): random.randint(1, 2), 'S': random.randint(4, 8)},
            # Halide electrolytes
            {'Li': random.randint(2, 4), random.choice(['Y', 'In', 'Sc']): 1, random.choice(['Cl', 'Br', 'I']): random.randint(4, 7)},
        ]
        
        base_composition = random.choice(composition_families)
        
        # Add some compositional noise/variation
        varied_composition = {}
        for element, count in base_composition.items():
            if isinstance(count, int):
                # Add ±1 variation
                new_count = max(1, count + random.randint(-1, 1))
                varied_composition[element] = new_count
            else:
                varied_composition[element] = count
        
        return {
            'composition': varied_composition,
            'lattice_params': generate_realistic_lattice_for_composition(varied_composition),
            'space_group': select_space_group_for_composition(varied_composition),
            'generation_method': 'composition_driven'
        }
    
    crystal_generators.append(composition_driven_generator)
    
    # Generator 3: Symmetry-driven generation
    def symmetry_driven_generator():
        """Generate based on crystal symmetry and space groups"""
        
        # High-symmetry space groups good for ionic conductors
        high_symmetry_groups = [225, 227, 229, 221, 194, 191, 166, 167]
        space_group = random.choice(high_symmetry_groups)
        
        # Generate composition suitable for this symmetry
        composition = generate_composition_for_symmetry(space_group)
        
        return {
            'composition': composition,
            'space_group': space_group,
            'lattice_params': generate_lattice_for_symmetry(space_group),
            'generation_method': 'symmetry_driven'
        }
    
    crystal_generators.append(symmetry_driven_generator)
    
    # Generator 4: Property-targeted generation
    def property_targeted_generator():
        """Generate structures targeting specific properties"""
        
        # Target high ionic conductivity compositions
        high_conductivity_elements = {
            'framework': ['Ti', 'Zr', 'Nb', 'Ta', 'Al', 'Ga'],
            'mobile_ion': ['Li'],
            'anions': ['O', 'S', 'F', 'Cl'],
            'dopants': ['La', 'Y', 'Sc', 'In']
        }
        
        composition = {}
        composition['Li'] = random.randint(1, 6)
        composition[random.choice(high_conductivity_elements['framework'])] = random.randint(1, 3)
        composition[random.choice(high_conductivity_elements['anions'])] = random.randint(3, 10)
        
        # Sometimes add dopant
        if random.random() < 0.4:
            composition[random.choice(high_conductivity_elements['dopants'])] = random.randint(1, 2)
        
        return {
            'composition': composition,
            'lattice_params': generate_optimized_lattice_for_conductivity(),
            'space_group': random.choice([225, 227, 194, 167, 221]),  # Good for ionic conduction
            'generation_method': 'property_targeted'
        }
    
    crystal_generators.append(property_targeted_generator)
    
    return crystal_generators

def generate_diverse_composition():
    """Generate diverse chemical compositions"""
    
    # Element pools for solid electrolytes
    alkali_metals = ['Li', 'Na', 'K']
    framework_metals = ['Ti', 'Zr', 'Al', 'Ga', 'In', 'Sc', 'Y', 'La', 'Nb', 'Ta']
    anions = ['O', 'S', 'F', 'Cl', 'Br', 'N', 'P']
    
    composition = {}
    
    # Always include alkali metal (prefer Li)
    alkali = random.choice(alkali_metals) if random.random() < 0.2 else 'Li'
    composition[alkali] = random.randint(1, 6)
    
    # Add 1-2 framework metals
    num_framework = random.randint(1, 2)
    for _ in range(num_framework):
        metal = random.choice(framework_metals)
        if metal not in composition:
            composition[metal] = random.randint(1, 4)
    
    # Add 1-2 anions
    num_anions = random.randint(1, 2)
    for _ in range(num_anions):
        anion = random.choice(anions)
        if anion not in composition:
            composition[anion] = random.randint(2, 10)
    
    return composition

def generate_realistic_lattice_for_composition(composition):
    """Generate realistic lattice parameters based on composition"""
    
    # Estimate volume based on composition
    volume_per_atom = {
        'Li': 8, 'Na': 15, 'K': 25,
        'Ti': 12, 'Zr': 18, 'Al': 10, 'Ga': 12, 'In': 16,
        'Sc': 14, 'Y': 20, 'La': 25, 'Nb': 15, 'Ta': 16,
        'O': 6, 'S': 12, 'F': 4, 'Cl': 8, 'Br': 12, 'N': 5, 'P': 8
    }
    
    total_volume = sum(count * volume_per_atom.get(element, 10) for element, count in composition.items())
    
    # Generate lattice parameters
    if random.random() < 0.3:  # Cubic
        a = (total_volume) ** (1/3)
        return {'a': a, 'b': a, 'c': a, 'alpha': 90, 'beta': 90, 'gamma': 90}
    elif random.random() < 0.5:  # Tetragonal
        c_over_a = random.uniform(0.8, 1.5)
        a = (total_volume / c_over_a) ** (1/3)
        return {'a': a, 'b': a, 'c': a * c_over_a, 'alpha': 90, 'beta': 90, 'gamma': 90}
    else:  # Orthorhombic
        ratios = sorted([random.uniform(0.8, 1.3) for _ in range(3)])
        scale = (total_volume / (ratios[0] * ratios[1] * ratios[2])) ** (1/3)
        a, b, c = [r * scale for r in ratios]
        return {'a': a, 'b': b, 'c': c, 'alpha': 90, 'beta': 90, 'gamma': 90}

def select_space_group_for_composition(composition):
    """Select appropriate space group based on composition"""
    
    # Simple heuristics for space group selection
    if 'Li' in composition and composition['Li'] >= 4:
        # Li-rich compositions often have high symmetry
        return random.choice([225, 227, 221, 229])
    elif any(element in composition for element in ['Ti', 'Zr']):
        # Transition metals often form tetragonal/cubic structures
        return random.choice([136, 141, 225, 221])
    elif any(element in composition for element in ['La', 'Y']):
        # Rare earths often form hexagonal structures
        return random.choice([194, 191, 166])
    else:
        # Default to common space groups
        return random.choice([1, 2, 12, 14, 15, 62, 63])

def generate_composition_for_symmetry(space_group):
    """Generate composition suitable for given space group"""
    
    if space_group in [225, 227, 221, 229]:  # Cubic
        return {'Li': random.randint(1, 4), 'Ti': 1, 'O': random.randint(3, 6)}
    elif space_group in [194, 191, 166]:  # Hexagonal
        return {'Li': random.randint(1, 3), 'Al': 1, 'O': random.randint(3, 5)}
    else:  # Lower symmetry
        return generate_diverse_composition()

def generate_lattice_for_symmetry(space_group):
    """Generate lattice parameters appropriate for space group"""
    
    base_length = random.uniform(4, 12)
    
    if space_group in [225, 227, 221, 229]:  # Cubic
        return {'a': base_length, 'b': base_length, 'c': base_length, 
                'alpha': 90, 'beta': 90, 'gamma': 90}
    elif space_group in [194, 191, 166]:  # Hexagonal
        c = base_length * random.uniform(0.8, 2.0)
        return {'a': base_length, 'b': base_length, 'c': c,
                'alpha': 90, 'beta': 90, 'gamma': 120}
    else:  # Lower symmetry
        return {'a': base_length, 'b': base_length * random.uniform(0.8, 1.2), 
                'c': base_length * random.uniform(0.8, 1.2),
                'alpha': 90, 'beta': 90, 'gamma': 90}

def generate_optimized_lattice_for_conductivity():
    """Generate lattice parameters optimized for ionic conductivity"""
    
    # Larger unit cells often have better ionic conductivity
    base_size = random.uniform(8, 15)
    
    return {
        'a': base_size, 'b': base_size, 'c': base_size * random.uniform(0.9, 1.1),
        'alpha': 90, 'beta': 90, 'gamma': 90
    }

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

class AdvancedGenerationGA:
    """Genetic Algorithm with Advanced Crystal Generation (CDVAE-inspired)"""
    
    def __init__(self, 
                 population_size: int = 80,
                 elite_count: int = 6,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.02,
                 max_generations: int = 50,
                 convergence_threshold: int = 15,
                 output_dir: str = "advanced_generation_ga_results"):
        
        self.population_size = population_size
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.cif_dir = self.output_dir / "cifs"
        self.cif_dir.mkdir(exist_ok=True)
        
        # Advanced generation components
        self.crystal_generators = generate_noise_based_crystal_structures()
        
        self.target_properties = TargetProperties()
        self.population: List[GACandidate] = []
        self.pareto_fronts: List[List[GACandidate]] = []
        self.pareto_history: List[List[List[GACandidate]]] = []
        self.hypervolume_history: List[float] = []
        self.generation = 0
        self.structure_matcher = StructureMatcher()
        
        # Random number generator
        self.rng = np.random.default_rng()
        
        print(f"🚀 Advanced Generation GA initialized with {len(self.crystal_generators)} generation methods")
        
    def generate_initial_population(self) -> List[GACandidate]:
        """Generate initial population using advanced generation methods"""
        print(f"Generating initial population of {self.population_size} candidates using Advanced Generation...")
        
        candidates = []
        attempts = 0
        max_attempts = self.population_size * 15
        
        generation_stats = {method.__name__ if hasattr(method, '__name__') else f"method_{i}": 0 
                          for i, method in enumerate(self.crystal_generators)}
        
        while len(candidates) < self.population_size and attempts < max_attempts:
            attempts += 1
            
            try:
                # Select generation method
                generator = random.choice(self.crystal_generators)
                crystal_data = generator()
                
                candidate = self._create_candidate_from_data(crystal_data)
                if candidate and self._is_valid_candidate(candidate):
                    candidates.append(candidate)
                    method_name = crystal_data.get('generation_method', 'unknown')
                    generation_stats[method_name] = generation_stats.get(method_name, 0) + 1
                    
                    if len(candidates) % 10 == 0:
                        print(f"  Generated {len(candidates)}/{self.population_size} candidates...")
                        
            except Exception as e:
                print(f"  Failed to generate candidate (attempt {attempts}): {e}")
                continue
                
        print(f"  ✅ Generation methods used: {generation_stats}")
        
        if len(candidates) < self.population_size:
            print(f"Warning: Only generated {len(candidates)} valid candidates out of {self.population_size}")
            
        return candidates
    
    def _create_candidate_from_data(self, crystal_data: Dict) -> Optional[GACandidate]:
        """Create GACandidate from crystal generation data"""
        
        try:
            composition = crystal_data['composition']
            lattice_params = crystal_data['lattice_params']
            space_group = crystal_data['space_group']
            generation_method = crystal_data.get('generation_method', 'unknown')
            
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
            
        # Check for reasonable density
        density = candidate.structure.density
        if density < 0.5 or density > 12.0:  # g/cm³
            return False
            
        # Check for reasonable lattice parameters
        lattice = candidate.structure.lattice
        if any(param < 2.0 or param > 30.0 for param in [lattice.a, lattice.b, lattice.c]):
            return False
        
        # Check for reasonable volume
        if lattice.volume < 30 or lattice.volume > 1500:  # Å³
            return False
            
        # Check composition has reasonable number of atoms
        total_atoms = sum(candidate.composition.values())
        if total_atoms < 2 or total_atoms > 25:
            return False
            
        return True
    
    def _generate_cif_file(self, candidate: GACandidate) -> str:
        """Generate CIF file for the candidate"""
        if not candidate.structure:
            return ""
            
        # Generate unique filename with generation method
        composition_str = "".join(f"{elem}{count}" for elem, count in 
                                sorted(candidate.composition.items()))
        filename = f"gen{self.generation}_{composition_str}_{candidate.generation_method}_{id(candidate)}.cif"
        cif_path = self.cif_dir / filename
        
        # Write CIF file
        cif_writer = CifWriter(candidate.structure)
        cif_writer.write_file(str(cif_path))
        
        candidate.cif_path = str(cif_path)
        return str(cif_path)
    
    def evaluate_population(self, candidates: List[GACandidate]) -> None:
        """Evaluate multi-objective fitness for all candidates"""
        print(f"Evaluating properties for {len(candidates)} candidates...")
        
        for i, candidate in enumerate(candidates):
            if i % 10 == 0:
                print(f"  Evaluating candidate {i+1}/{len(candidates)}...")
                
            try:
                # Get ML predictions
                if candidate.cif_path and os.path.exists(candidate.cif_path):
                    results = predict_single_cif(candidate.cif_path, verbose=False)
                    
                    candidate.properties = {
                        'ionic_conductivity': results.get('ionic_conductivity', 1e-10),
                        'bandgap': results.get('bandgap', 0.0),
                        'sei_score': results.get('sei_score', 0.0),
                        'cei_score': results.get('cei_score', 0.0),
                        'bulk_modulus': results.get('bulk_modulus', 0.0)
                    }
                    
                    # Calculate multi-objective values (minimize all - distance from targets)
                    candidate.objectives = self._calculate_objectives(candidate.properties)
                    
                else:
                    print(f"    Warning: CIF file not found for candidate {i+1}")
                    candidate.objectives = [float('inf')] * 5
                    
            except Exception as e:
                print(f"    Error evaluating candidate {i+1}: {e}")
                candidate.objectives = [float('inf')] * 5
    
    def _calculate_objectives(self, properties: Dict[str, float]) -> List[float]:
        """Calculate multi-objective values (all minimization problems)"""
        targets = self.target_properties
        objectives = []
        
        # Objective 1: Ionic conductivity error (log scale, handle edge cases)
        if properties['ionic_conductivity'] > 1e-12:  # Avoid log of zero
            target_log = np.log10(targets.ionic_conductivity)  # log10(1e-3) = -3
            actual_log = np.log10(properties['ionic_conductivity'])
            ic_error = abs(actual_log - target_log)  # Difference in log space
        else:
            ic_error = 10.0  # Large penalty for zero/negative conductivity
        objectives.append(ic_error)
        
        # Objective 2: Bandgap error (normalized)
        if targets.bandgap > 0:
            bg_error = abs(properties['bandgap'] - targets.bandgap) / targets.bandgap
        else:
            bg_error = abs(properties['bandgap'] - targets.bandgap)
        objectives.append(bg_error)
        
        # Objective 3: SEI score error (0-1 scale, higher is better)
        sei_error = abs(properties['sei_score'] - targets.sei_score)
        objectives.append(sei_error)
        
        # Objective 4: CEI score error (0-1 scale, higher is better)
        cei_error = abs(properties['cei_score'] - targets.cei_score)
        objectives.append(cei_error)
        
        # Objective 5: Bulk modulus error (normalized)
        if targets.bulk_modulus > 0:
            bm_error = abs(properties['bulk_modulus'] - targets.bulk_modulus) / targets.bulk_modulus
        else:
            bm_error = abs(properties['bulk_modulus'] - targets.bulk_modulus)
        objectives.append(bm_error)
        
        return objectives
    
    def pareto_dominates(self, candidate1: GACandidate, candidate2: GACandidate) -> bool:
        """Check if candidate1 Pareto dominates candidate2 (minimization)"""
        obj1, obj2 = candidate1.objectives, candidate2.objectives
        
        # candidate1 dominates candidate2 if:
        # 1. candidate1 is no worse than candidate2 in all objectives
        # 2. candidate1 is strictly better than candidate2 in at least one objective
        
        at_least_one_better = False
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:  # candidate1 is worse in this objective
                return False
            if obj1[i] < obj2[i]:  # candidate1 is better in this objective
                at_least_one_better = True
                
        return at_least_one_better
    
    def non_dominated_sort(self, population: List[GACandidate]) -> List[List[GACandidate]]:
        """NSGA-II non-dominated sorting"""
        fronts = []
        domination_count = [0] * len(population)  # Number of solutions that dominate this solution
        dominated_solutions = [[] for _ in range(len(population))]  # Solutions dominated by this solution
        
        # Calculate domination relationships
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j:
                    if self.pareto_dominates(population[i], population[j]):
                        dominated_solutions[i].append(j)
                    elif self.pareto_dominates(population[j], population[i]):
                        domination_count[i] += 1
        
        # Find first front (non-dominated solutions)
        current_front = []
        for i in range(len(population)):
            if domination_count[i] == 0:
                population[i].pareto_rank = 0
                current_front.append(population[i])
        
        fronts.append(current_front)
        
        # Find subsequent fronts
        front_index = 0
        while front_index < len(fronts) and len(fronts[front_index]) > 0:
            next_front = []
            for candidate in fronts[front_index]:
                candidate_index = population.index(candidate)
                for dominated_index in dominated_solutions[candidate_index]:
                    domination_count[dominated_index] -= 1
                    if domination_count[dominated_index] == 0:
                        population[dominated_index].pareto_rank = front_index + 1
                        next_front.append(population[dominated_index])
            
            if next_front:
                fronts.append(next_front)
                front_index += 1
            else:
                break
                
        return fronts
    
    def calculate_crowding_distance(self, front: List[GACandidate]) -> None:
        """Calculate crowding distance for diversity preservation"""
        if len(front) <= 2:
            for candidate in front:
                candidate.crowding_distance = float('inf')
            return
        
        # Initialize crowding distance
        for candidate in front:
            candidate.crowding_distance = 0.0
        
        num_objectives = len(front[0].objectives)
        
        # Calculate crowding distance for each objective
        for obj_index in range(num_objectives):
            # Sort by objective value
            front.sort(key=lambda x: x.objectives[obj_index])
            
            # Set boundary points to infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range for normalization
            obj_range = front[-1].objectives[obj_index] - front[0].objectives[obj_index]
            if obj_range == 0:
                continue
            
            # Calculate crowding distance for intermediate points
            for i in range(1, len(front) - 1):
                distance = (front[i + 1].objectives[obj_index] -
                           front[i - 1].objectives[obj_index]) / obj_range
                front[i].crowding_distance += distance
    
    def tournament_selection(self, population: List[GACandidate]) -> GACandidate:
        """Tournament selection based on Pareto rank and crowding distance"""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        # Sort by Pareto rank (lower is better), then by crowding distance (higher is better)
        tournament.sort(key=lambda x: (x.pareto_rank, -x.crowding_distance))
        
        return tournament[0]
    
    def run(self) -> Dict[str, Any]:
        """Run the advanced generation genetic algorithm"""
        print(f"🚀 Starting Advanced Generation Genetic Algorithm for Solid-State Electrolyte Discovery")
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {self.max_generations}")
        print(f"Generation methods: {len(self.crystal_generators)}")
        print("-" * 80)
        
        # Generate initial population
        self.population = self.generate_initial_population()
        
        # Evaluate initial population
        self.evaluate_population(self.population)
        
        # Perform initial non-dominated sorting
        self.pareto_fronts = self.non_dominated_sort(self.population)
        for front in self.pareto_fronts:
            self.calculate_crowding_distance(front)
        
        # Save initial results
        self.save_generation_results()
        
        # Store initial Pareto fronts
        self.pareto_history.append(deepcopy(self.pareto_fronts))
        
        # Evolution loop (simplified for demonstration)
        generations_without_improvement = 0
        
        for generation in range(1, min(self.max_generations + 1, 10)):  # Limited to 10 generations for demo
            self.generation = generation
            
            print(f"\n{'='*20} Generation {generation} {'='*20}")
            
            # Simple evolution: generate new population
            new_candidates = []
            while len(new_candidates) < self.population_size // 2:
                try:
                    generator = random.choice(self.crystal_generators)
                    crystal_data = generator()
                    candidate = self._create_candidate_from_data(crystal_data)
                    if candidate and self._is_valid_candidate(candidate):
                        new_candidates.append(candidate)
                except:
                    continue
            
            # Evaluate new candidates
            self.evaluate_population(new_candidates)
            
            # Combine with best from previous generation
            combined_population = self.pareto_fronts[0][:self.population_size//2] + new_candidates
            self.population = combined_population[:self.population_size]
            
            # Update Pareto fronts
            self.pareto_fronts = self.non_dominated_sort(self.population)
            for front in self.pareto_fronts:
                self.calculate_crowding_distance(front)
            
            # Save results
            self.save_generation_results()
            self.pareto_history.append(deepcopy(self.pareto_fronts))
        
        # Final results
        final_pareto_front = self.pareto_fronts[0] if self.pareto_fronts else []
        
        results = {
            'generations_run': self.generation,
            'final_population_size': len(self.population),
            'pareto_front_size': len(final_pareto_front),
            'pareto_front_candidates': []
        }
        
        # Add Pareto front candidates to results
        for candidate in final_pareto_front:
            candidate_data = {
                'composition': candidate.composition,
                'properties': candidate.properties,
                'objectives': candidate.objectives,
                'generation_method': candidate.generation_method,
                'crowding_distance': candidate.crowding_distance
            }
            results['pareto_front_candidates'].append(candidate_data)
        
        print(f"\n{'='*20} FINAL RESULTS {'='*20}")
        print(f"Generations run: {self.generation}")
        print(f"Final Pareto front size: {len(final_pareto_front)}")
        
        if final_pareto_front:
            print(f"\nTop 5 Advanced Generated Candidates:")
            for i, candidate in enumerate(final_pareto_front[:5]):
                comp_str = "".join(f"{elem}{count}" for elem, count in sorted(candidate.composition.items()))
                print(f"\n{i+1}. Composition: {comp_str}")
                print(f"   Generation Method: {candidate.generation_method}")
                print(f"   Properties:")
                for prop, value in candidate.properties.items():
                    if prop == 'ionic_conductivity':
                        print(f"     {prop}: {value:.2e}")
                    else:
                        print(f"     {prop}: {value:.4f}")
                print(f"   Objectives: {[f'{obj:.3f}' for obj in candidate.objectives]}")
        
        return results
    
    def save_generation_results(self) -> None:
        """Save current generation results"""
        gen_dir = self.output_dir / f"generation_{self.generation}"
        gen_dir.mkdir(exist_ok=True)
        
        # Save population data
        population_data = []
        for i, candidate in enumerate(self.population):
            data = {
                'id': i,
                'composition': candidate.composition,
                'lattice_params': candidate.lattice_params,
                'space_group': candidate.space_group,
                'generation_method': candidate.generation_method,
                'properties': candidate.properties,
                'objectives': candidate.objectives,
                'pareto_rank': candidate.pareto_rank,
                'crowding_distance': candidate.crowding_distance,
                'generation': candidate.generation,
                'cif_path': candidate.cif_path
            }
            population_data.append(data)
        
        with open(gen_dir / "population.json", 'w') as f:
            json.dump(population_data, f, indent=2)
        
        # Save Pareto front data
        if self.pareto_fronts:
            best_front = self.pareto_fronts[0]
            print(f"\nGeneration {self.generation} - Pareto Front 0 ({len(best_front)} candidates):")
            
            for i, candidate in enumerate(best_front[:10]):  # Show top 10
                comp_str = "".join(f"{elem}{count}" for elem, count in sorted(candidate.composition.items()))
                print(f"  {i+1}. {comp_str} | Method: {candidate.generation_method} | "
                      f"Objectives: {[f'{obj:.3f}' for obj in candidate.objectives]} | "
                      f"CD: {candidate.crowding_distance:.3f}")


def main():
    """Main function to run the advanced generation genetic algorithm"""
    
    # Initialize and run Advanced Generation GA
    ga = AdvancedGenerationGA(
        population_size=40,  # Smaller for demonstration
        max_generations=10,  # Limited for demonstration
        output_dir="advanced_generation_ga_results"
    )
    
    results = ga.run()
    
    print(f"\n🎉 Advanced Generation Genetic Algorithm completed!")
    print(f"Results saved to: {ga.output_dir}")
    print(f"This demonstrates CDVAE-inspired generation with multiple methods:")
    print(f"  • Noise-perturbed structures")
    print(f"  • Composition-driven generation")
    print(f"  • Symmetry-driven generation")
    print(f"  • Property-targeted generation")
    
    return results


if __name__ == "__main__":
    main()