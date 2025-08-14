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

# PyXtal and pymatgen imports
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Import ML prediction functions with fallback to debug mode
try:
    from genetic_algo.fully_optimized_predictor import predict_single_cif_fully_optimized as predict_single_cif
    print("Using FULLY optimized ML predictor - models loaded ONCE only")
except ImportError:
    try:
        from optimized_ml_predictor import predict_single_cif_optimized as predict_single_cif
        print("Using optimized ML predictor with model caching")
    except ImportError:
        try:
            from env.main_rl import predict_single_cif
            print("Using standard ML predictor (models will reload each time)")
        except ImportError:
            from debug_predictor import predict_single_cif_debug as predict_single_cif
            print("Using DEBUG predictor with realistic random values for testing")

# Import crystal generation functions
from genetic_algo.pyxtal_generation import (
    charge_dict, all_species, common_space_groups,
    check_charge_neutrality, get_wyckoff_multiplicities,
    generate_wyckoff_compatible_composition
)


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


class ParetoFrontGA:
    """Multi-objective Genetic Algorithm with Pareto front optimization for solid-state electrolyte discovery"""
    
    def __init__(self, 
                 population_size: int = 80,
                 elite_count: int = 6,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.02,
                 max_generations: int = 50,
                 convergence_threshold: int = 15,
                 output_dir: str = "pareto_ga_results"):
        
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
        
        # Enhanced element pool for Li-ion conductors
        self.element_pool = all_species
        
        # Space groups commonly found in solid electrolytes
        self.common_space_groups = common_space_groups
        
        self.target_properties = TargetProperties()
        self.population: List[GACandidate] = []
        self.pareto_fronts: List[List[GACandidate]] = []
        self.pareto_history: List[List[List[GACandidate]]] = []
        self.hypervolume_history: List[float] = []
        self.generation = 0
        self.structure_matcher = StructureMatcher()
        
        # Random number generator
        self.rng = np.random.default_rng()
        
    def generate_initial_population(self) -> List[GACandidate]:
        """Generate initial population using enhanced PyXtal with charge neutrality"""
        print(f"Generating initial population of {self.population_size} candidates...")
        
        candidates = []
        attempts = 0
        max_attempts = self.population_size * 20  # More attempts for better diversity
        
        while len(candidates) < self.population_size and attempts < max_attempts:
            attempts += 1
            
            try:
                candidate = self._generate_valid_crystal()
                if candidate and self._is_valid_candidate(candidate):
                    candidates.append(candidate)
                    if len(candidates) % 10 == 0:
                        print(f"  Generated {len(candidates)}/{self.population_size} candidates...")
                        
            except Exception as e:
                print(f"  Failed to generate candidate (attempt {attempts}): {e}")
                continue
                
        if len(candidates) < self.population_size:
            print(f"Warning: Only generated {len(candidates)} valid candidates out of {self.population_size}")
            
        return candidates
    
    def _generate_valid_crystal(self) -> Optional[GACandidate]:
        """Generate a single valid crystal using enhanced PyXtal generation"""
        
        for attempt in range(100):
            try:
                # Select space group
                space_group = random.choice(self.common_space_groups)
                wyckoff_mults = get_wyckoff_multiplicities(space_group)
                
                # Generate charge-neutral composition with enhanced diversity
                composition = generate_wyckoff_compatible_composition(
                    self.element_pool, wyckoff_mults,
                    min_species=2, max_species=6, max_multiplicity=4  # Increased for more diversity
                )
                
                if composition is None:
                    continue
                
                # Optional: Prefer Li-containing structures (but don't force)
                # This allows chemical diversity while still favoring Li-ion conductors
                if random.random() < 0.7 and 'Li' not in composition:
                    # 70% chance to add Li for Li-ion conduction, but allow 30% diversity
                    elements = list(composition.keys())
                    if len(elements) < 5:  # Only if we have room for more elements
                        replace_element = random.choice(elements)
                        li_count = composition[replace_element]
                        del composition[replace_element]
                        composition['Li'] = li_count
                
                # Check charge neutrality
                if not check_charge_neutrality(composition):
                    continue
                
                # Convert to PyXtal format
                species = list(composition.keys())
                numIons = list(composition.values())
                
                # Generate structure with PyXtal (avoid deepcopy issues)
                crystal = pyxtal()
                crystal.from_random(dim=3, group=space_group, species=species,
                                  numIons=numIons, random_state=None)  # Use None to avoid deepcopy issues
                
                if not crystal.valid:
                    continue
                    
                # Convert to pymatgen Structure
                structure = crystal.to_pymatgen()
                
                # Extract lattice parameters
                lattice = structure.lattice
                lattice_params = {
                    'a': lattice.a,
                    'b': lattice.b,
                    'c': lattice.c,
                    'alpha': lattice.alpha,
                    'beta': lattice.beta,
                    'gamma': lattice.gamma
                }
                
                candidate = GACandidate(
                    composition=composition,
                    lattice_params=lattice_params,
                    space_group=space_group,
                    structure=structure,
                    generation=self.generation
                )
                
                # Generate CIF file
                self._generate_cif_file(candidate)
                
                return candidate
                
            except Exception as e:
                continue
                
        return None
    
    def _is_valid_candidate(self, candidate: GACandidate) -> bool:
        """Enhanced validity checks for candidates"""
        if not candidate.structure:
            return False
            
        # Check for reasonable density
        density = candidate.structure.density
        if density < 1.0 or density > 10.0:  # g/cm³
            return False
            
        # Check for reasonable lattice parameters
        lattice = candidate.structure.lattice
        if any(param < 2.0 or param > 30.0 for param in [lattice.a, lattice.b, lattice.c]):
            return False
            
        # Prefer Li content but don't require it (allows chemical diversity)
        # This is now just a preference, not a hard requirement
        pass
            
        # Check charge neutrality
        if not check_charge_neutrality(candidate.composition):
            return False
            
        return True
    
    def _generate_cif_file(self, candidate: GACandidate) -> str:
        """Generate CIF file for the candidate"""
        if not candidate.structure:
            return ""
            
        # Generate unique filename
        composition_str = "".join(f"{elem}{count}" for elem, count in 
                                sorted(candidate.composition.items()))
        filename = f"gen{self.generation}_{composition_str}_{id(candidate)}.cif"
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
    
    def crossover(self, parent1: GACandidate, parent2: GACandidate) -> GACandidate:
        """Enhanced crossover with chemical validity preservation"""
        
        # Weighted average of lattice parameters
        weight = random.uniform(0.3, 0.7)
        new_lattice_params = {}
        
        for param in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
            new_lattice_params[param] = (weight * parent1.lattice_params[param] + 
                                       (1 - weight) * parent2.lattice_params[param])
        
        # Enhanced composition mixing with charge neutrality preservation
        new_composition = self._crossover_composition(parent1.composition, parent2.composition)
        
        # Select space group (bias toward better parent)
        if parent1.pareto_rank < parent2.pareto_rank:
            space_group = parent1.space_group if random.random() < 0.7 else parent2.space_group
        elif parent2.pareto_rank < parent1.pareto_rank:
            space_group = parent2.space_group if random.random() < 0.7 else parent1.space_group
        else:
            # Same rank, use crowding distance
            if parent1.crowding_distance > parent2.crowding_distance:
                space_group = parent1.space_group if random.random() < 0.7 else parent2.space_group
            else:
                space_group = parent2.space_group if random.random() < 0.7 else parent1.space_group
        
        # Create offspring candidate
        offspring = GACandidate(
            composition=new_composition,
            lattice_params=new_lattice_params,
            space_group=space_group,
            generation=self.generation + 1,
            parent_ids=[id(parent1), id(parent2)]
        )
        
        # Generate structure using PyXtal with error handling
        try:
            # Create new PyXtal instance to avoid deepcopy issues
            crystal = pyxtal()
            species = list(new_composition.keys())
            numIons = list(new_composition.values())
            
            # Try multiple times with different random states to avoid PyXtal issues
            for attempt in range(3):
                try:
                    crystal.from_random(dim=3, group=space_group, species=species,
                                      numIons=numIons, random_state=None)  # Use None to avoid deepcopy issues
                    
                    if crystal.valid:
                        offspring.structure = crystal.to_pymatgen()
                        # Update lattice parameters with actual values
                        lattice = offspring.structure.lattice
                        offspring.lattice_params = {
                            'a': lattice.a, 'b': lattice.b, 'c': lattice.c,
                            'alpha': lattice.alpha, 'beta': lattice.beta, 'gamma': lattice.gamma
                        }
                        self._generate_cif_file(offspring)
                        return offspring
                except Exception as inner_e:
                    if attempt == 2:  # Last attempt
                        print(f"    PyXtal crossover failed after 3 attempts: {inner_e}")
                    continue
                    
        except Exception as e:
            print(f"    Crossover structure generation failed: {e}")
            
        return None
    
    def _crossover_composition(self, comp1: Dict[str, int], comp2: Dict[str, int]) -> Dict[str, int]:
        """Enhanced composition crossover with charge neutrality"""
        
        # Try multiple approaches to maintain charge neutrality
        for attempt in range(10):
            new_composition = {}
            all_elements = set(comp1.keys()) | set(comp2.keys())
            
            # Method 1: Weighted combination
            if attempt < 5:
                weight = random.uniform(0.3, 0.7)
                for element in all_elements:
                    count1 = comp1.get(element, 0)
                    count2 = comp2.get(element, 0)
                    
                    if count1 > 0 and count2 > 0:
                        new_count = int(weight * count1 + (1 - weight) * count2)
                        new_count = max(1, new_count)
                    elif count1 > 0:
                        new_count = count1 if random.random() < 0.7 else 0
                    else:
                        new_count = count2 if random.random() < 0.7 else 0
                        
                    if new_count > 0:
                        new_composition[element] = new_count
            
            # Method 2: Random selection from parents
            else:
                parent_comp = random.choice([comp1, comp2])
                new_composition = parent_comp.copy()
                
                # Randomly modify one element
                if len(new_composition) > 1:
                    element = random.choice(list(new_composition.keys()))
                    if element != 'Li':  # Don't remove Li
                        change = random.randint(-1, 1)
                        new_count = new_composition[element] + change
                        if new_count <= 0:
                            del new_composition[element]
                        else:
                            new_composition[element] = new_count
            
            # Ensure Li is present
            if 'Li' not in new_composition or new_composition['Li'] == 0:
                new_composition['Li'] = max(comp1.get('Li', 1), comp2.get('Li', 1))
            
            # Check charge neutrality
            if check_charge_neutrality(new_composition):
                return new_composition
        
        # Fallback: return parent composition
        return comp1.copy()
    
    def mutate(self, candidate: GACandidate) -> GACandidate:
        """Enhanced mutation with charge neutrality preservation"""
        if random.random() > self.mutation_rate:
            return candidate
            
        mutated = deepcopy(candidate)
        
        # Try multiple mutation approaches
        for attempt in range(5):
            temp_composition = mutated.composition.copy()
            mutation_type = random.choice(['composition', 'lattice', 'space_group'])
            
            if mutation_type == 'composition':
                # Composition mutation with charge neutrality
                if len(temp_composition) > 1:
                    element = random.choice(list(temp_composition.keys()))
                    
                    if element == 'Li':
                        # Modify Li count carefully
                        change = random.randint(-1, 2)
                        temp_composition[element] = max(1, temp_composition[element] + change)
                    else:
                        change = random.randint(-2, 2)
                        new_count = temp_composition[element] + change
                        if new_count <= 0:
                            del temp_composition[element]
                        else:
                            temp_composition[element] = new_count
                
                # Check if mutation maintains charge neutrality
                if check_charge_neutrality(temp_composition):
                    mutated.composition = temp_composition
                    break
                    
            elif mutation_type == 'lattice':
                # Lattice parameter mutation
                param = random.choice(['a', 'b', 'c'])
                factor = random.uniform(0.9, 1.1)
                mutated.lattice_params[param] *= factor
                break
                
            elif mutation_type == 'space_group':
                # Space group mutation
                mutated.space_group = random.choice(self.common_space_groups)
                break
        
        # Regenerate structure with error handling
        try:
            crystal = pyxtal()
            species = list(mutated.composition.keys())
            numIons = list(mutated.composition.values())
            
            # Try multiple times to avoid PyXtal issues
            for attempt in range(3):
                try:
                    crystal.from_random(dim=3, group=mutated.space_group, species=species,
                                      numIons=numIons, random_state=None)  # Use None to avoid deepcopy issues
                    
                    if crystal.valid:
                        mutated.structure = crystal.to_pymatgen()
                        lattice = mutated.structure.lattice
                        mutated.lattice_params = {
                            'a': lattice.a, 'b': lattice.b, 'c': lattice.c,
                            'alpha': lattice.alpha, 'beta': lattice.beta, 'gamma': lattice.gamma
                        }
                        self._generate_cif_file(mutated)
                        return mutated
                except Exception as inner_e:
                    if attempt == 2:  # Last attempt
                        print(f"    PyXtal mutation failed after 3 attempts: {inner_e}")
                    continue
                    
        except Exception as e:
            print(f"    Mutation structure generation failed: {e}")
            
        return candidate  # Return original if mutation failed
    
    def evolve_generation(self) -> List[GACandidate]:
        """Create next generation using NSGA-II"""
        print(f"Evolving generation {self.generation + 1}...")
        
        # Perform non-dominated sorting
        fronts = self.non_dominated_sort(self.population)
        self.pareto_fronts = fronts
        
        # Calculate crowding distance for each front
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        print(f"  Pareto fronts: {[len(front) for front in fronts]}")
        
        # Generate offspring
        offspring = []
        while len(offspring) < self.population_size:
            parent1 = self.tournament_selection(self.population)
            parent2 = self.tournament_selection(self.population)
            
            child = self.crossover(parent1, parent2)
            if child and self._is_valid_candidate(child):
                child = self.mutate(child)
                if child:
                    offspring.append(child)
        
        # Combine parents and offspring
        combined_population = self.population + offspring
        
        # Evaluate offspring
        self.evaluate_population(offspring)
        
        # Select next generation using NSGA-II selection
        next_generation = self._nsga2_selection(combined_population)
        
        print(f"  Next generation size: {len(next_generation)}")
        
        return next_generation
    
    def _nsga2_selection(self, population: List[GACandidate]) -> List[GACandidate]:
        """NSGA-II environmental selection"""
        
        # Perform non-dominated sorting
        fronts = self.non_dominated_sort(population)
        
        # Calculate crowding distance for each front
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        # Select individuals for next generation
        next_generation = []
        front_index = 0
        
        while len(next_generation) + len(fronts[front_index]) <= self.population_size:
            next_generation.extend(fronts[front_index])
            front_index += 1
            if front_index >= len(fronts):
                break
        
        # If we need to partially fill from the next front
        if len(next_generation) < self.population_size and front_index < len(fronts):
            remaining_slots = self.population_size - len(next_generation)
            last_front = fronts[front_index]
            
            # Sort by crowding distance (descending)
            last_front.sort(key=lambda x: x.crowding_distance, reverse=True)
            next_generation.extend(last_front[:remaining_slots])
        
        return next_generation
    
    def calculate_hypervolume(self, front: List[GACandidate], reference_point: List[float] = None) -> float:
        """Calculate hypervolume indicator for convergence assessment"""
        if not front:
            return 0.0
        
        if reference_point is None:
            # Use worst values + 10% as reference point
            reference_point = []
            for obj_idx in range(len(front[0].objectives)):
                max_val = max(candidate.objectives[obj_idx] for candidate in front)
                reference_point.append(max_val * 1.1)
        
        # Simple hypervolume calculation (for small fronts)
        # For larger fronts, consider using specialized algorithms
        if len(front) == 1:
            volume = 1.0
            for i, obj_val in enumerate(front[0].objectives):
                volume *= max(0, reference_point[i] - obj_val)
            return volume
        
        # For multiple points, use Monte Carlo approximation
        num_samples = 10000
        dominated_count = 0
        
        for _ in range(num_samples):
            # Generate random point in objective space
            random_point = [random.uniform(0, ref) for ref in reference_point]
            
            # Check if any solution dominates this random point
            for candidate in front:
                if all(candidate.objectives[i] <= random_point[i] for i in range(len(random_point))):
                    dominated_count += 1
                    break
        
        # Calculate hypervolume
        total_volume = 1.0
        for ref in reference_point:
            total_volume *= ref
        
        return (dominated_count / num_samples) * total_volume
    
    def save_generation_results(self) -> None:
        """Save current generation results including Pareto front analysis"""
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
                'properties': candidate.properties,
                'objectives': candidate.objectives,
                'pareto_rank': candidate.pareto_rank,
                'crowding_distance': candidate.crowding_distance,
                'generation': candidate.generation,
                'parent_ids': candidate.parent_ids,
                'cif_path': candidate.cif_path
            }
            population_data.append(data)
        
        with open(gen_dir / "population.json", 'w') as f:
            json.dump(population_data, f, indent=2)
        
        # Save Pareto front data
        if self.pareto_fronts:
            pareto_data = []
            for front_idx, front in enumerate(self.pareto_fronts):
                front_data = {
                    'front_index': front_idx,
                    'size': len(front),
                    'candidates': []
                }
                
                for candidate in front:
                    candidate_data = {
                        'composition': candidate.composition,
                        'properties': candidate.properties,
                        'objectives': candidate.objectives,
                        'crowding_distance': candidate.crowding_distance
                    }
                    front_data['candidates'].append(candidate_data)
                
                pareto_data.append(front_data)
            
            with open(gen_dir / "pareto_fronts.json", 'w') as f:
                json.dump(pareto_data, f, indent=2)
        
        # Calculate and save hypervolume
        if self.pareto_fronts and len(self.pareto_fronts) > 0:
            hypervolume = self.calculate_hypervolume(self.pareto_fronts[0])
            self.hypervolume_history.append(hypervolume)
        
        # Save best candidates from first Pareto front
        if self.pareto_fronts and len(self.pareto_fronts[0]) > 0:
            best_front = self.pareto_fronts[0]
            print(f"\nGeneration {self.generation} - Pareto Front 0 ({len(best_front)} candidates):")
            
            for i, candidate in enumerate(best_front[:10]):  # Show top 10
                comp_str = "".join(f"{elem}{count}" for elem, count in sorted(candidate.composition.items()))
                print(f"  {i+1}. {comp_str} | Objectives: {[f'{obj:.3f}' for obj in candidate.objectives]} | "
                      f"CD: {candidate.crowding_distance:.3f}")
    
    def plot_pareto_front(self, generation: int = None) -> None:
        """Plot Pareto front for visualization"""
        if not self.pareto_fronts or len(self.pareto_fronts) == 0:
            return
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Plot first 3 objectives in 3D
            fig = plt.figure(figsize=(12, 8))
            
            # 3D plot of first 3 objectives
            ax1 = fig.add_subplot(221, projection='3d')
            
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for front_idx, front in enumerate(self.pareto_fronts[:5]):  # Plot first 5 fronts
                if len(front) == 0:
                    continue
                    
                objectives = np.array([candidate.objectives for candidate in front])
                ax1.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                           c=colors[front_idx % len(colors)],
                           label=f'Front {front_idx}', alpha=0.7)
            
            ax1.set_xlabel('Ionic Conductivity Error')
            ax1.set_ylabel('Bandgap Error')
            ax1.set_zlabel('SEI Score Error')
            ax1.set_title('Pareto Fronts (First 3 Objectives)')
            ax1.legend()
            
            # 2D plots for other objective pairs
            ax2 = fig.add_subplot(222)
            for front_idx, front in enumerate(self.pareto_fronts[:3]):
                if len(front) == 0:
                    continue
                objectives = np.array([candidate.objectives for candidate in front])
                ax2.scatter(objectives[:, 3], objectives[:, 4],
                           c=colors[front_idx % len(colors)],
                           label=f'Front {front_idx}', alpha=0.7)
            ax2.set_xlabel('CEI Score Error')
            ax2.set_ylabel('Bulk Modulus Error')
            ax2.set_title('CEI vs Bulk Modulus')
            ax2.legend()
            
            # Hypervolume evolution
            ax3 = fig.add_subplot(223)
            if len(self.hypervolume_history) > 1:
                ax3.plot(range(len(self.hypervolume_history)), self.hypervolume_history, 'b-o')
                ax3.set_xlabel('Generation')
                ax3.set_ylabel('Hypervolume')
                ax3.set_title('Hypervolume Evolution')
                ax3.grid(True)
            
            # Front size evolution
            ax4 = fig.add_subplot(224)
            if hasattr(self, 'pareto_history') and len(self.pareto_history) > 0:
                front_sizes = []
                for gen_fronts in self.pareto_history:
                    if len(gen_fronts) > 0:
                        front_sizes.append(len(gen_fronts[0]))
                    else:
                        front_sizes.append(0)
                ax4.plot(range(len(front_sizes)), front_sizes, 'g-o')
                ax4.set_xlabel('Generation')
                ax4.set_ylabel('First Front Size')
                ax4.set_title('Pareto Front Size Evolution')
                ax4.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            gen_suffix = f"_gen{generation}" if generation is not None else ""
            plt.savefig(self.output_dir / f"pareto_front{gen_suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plot: {e}")
    
    def run(self) -> Dict[str, Any]:
        """Run the multi-objective genetic algorithm with Pareto front optimization"""
        print(f"Starting Multi-Objective Genetic Algorithm for Solid-State Electrolyte Discovery")
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {self.max_generations}")
        print(f"Elite count: {self.elite_count}")
        print(f"Tournament size: {self.tournament_size}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Convergence threshold: {self.convergence_threshold} generations")
        print(f"Target properties:")
        print(f"  Ionic Conductivity: {self.target_properties.ionic_conductivity:.2e} S/cm")
        print(f"  Bandgap: {self.target_properties.bandgap} eV")
        print(f"  SEI Score: {self.target_properties.sei_score}")
        print(f"  CEI Score: {self.target_properties.cei_score}")
        print(f"  Bulk Modulus: {self.target_properties.bulk_modulus} GPa")
        print("-" * 80)
        
        # Generate initial population
        self.population = self.generate_initial_population()
        
        # Evaluate initial population
        self.evaluate_population(self.population)
        
        # Perform initial non-dominated sorting
        self.pareto_fronts = self.non_dominated_sort(self.population)
        for front in self.pareto_fronts:
            self.calculate_crowding_distance(front)
        
        # Track initial hypervolume
        if self.pareto_fronts and len(self.pareto_fronts[0]) > 0:
            initial_hypervolume = self.calculate_hypervolume(self.pareto_fronts[0])
            self.hypervolume_history.append(initial_hypervolume)
        
        # Save initial results
        self.save_generation_results()
        self.plot_pareto_front(0)
        
        # Store initial Pareto fronts
        self.pareto_history.append(deepcopy(self.pareto_fronts))
        
        # Evolution loop
        generations_without_improvement = 0
        best_hypervolume = self.hypervolume_history[0] if self.hypervolume_history else 0
        
        for generation in range(1, self.max_generations + 1):
            self.generation = generation
            
            print(f"\n{'='*20} Generation {generation} {'='*20}")
            
            # Create next generation
            self.population = self.evolve_generation()
            
            # Calculate current hypervolume
            current_hypervolume = 0
            if self.pareto_fronts and len(self.pareto_fronts[0]) > 0:
                current_hypervolume = self.calculate_hypervolume(self.pareto_fronts[0])
                self.hypervolume_history.append(current_hypervolume)
            
            # Check for improvement
            improvement_threshold = 0.01  # 1% improvement
            if current_hypervolume > best_hypervolume * (1 + improvement_threshold):
                best_hypervolume = current_hypervolume
                generations_without_improvement = 0
                print(f"Hypervolume improved: {current_hypervolume:.6f}")
            else:
                generations_without_improvement += 1
                print(f"No significant improvement for {generations_without_improvement} generations")
                print(f"Current hypervolume: {current_hypervolume:.6f}")
            
            # Save results
            self.save_generation_results()
            
            # Store Pareto fronts history
            self.pareto_history.append(deepcopy(self.pareto_fronts))
            
            # Plot every 10 generations or at the end
            if generation % 10 == 0 or generation == self.max_generations:
                self.plot_pareto_front(generation)
            
            # Check convergence
            if generations_without_improvement >= self.convergence_threshold:
                print(f"\nConverged after {generation} generations (no improvement for {self.convergence_threshold} generations)")
                break
        
        # Final analysis
        final_pareto_front = self.pareto_fronts[0] if self.pareto_fronts else []
        
        # Create final comprehensive plot
        self.plot_pareto_front()
        
        results = {
            'generations_run': self.generation,
            'final_population_size': len(self.population),
            'pareto_front_size': len(final_pareto_front),
            'final_hypervolume': self.hypervolume_history[-1] if self.hypervolume_history else 0,
            'hypervolume_history': self.hypervolume_history,
            'converged': generations_without_improvement >= self.convergence_threshold,
            'pareto_front_candidates': []
        }
        
        # Add Pareto front candidates to results
        for candidate in final_pareto_front:
            candidate_data = {
                'composition': candidate.composition,
                'properties': candidate.properties,
                'objectives': candidate.objectives,
                'crowding_distance': candidate.crowding_distance
            }
            results['pareto_front_candidates'].append(candidate_data)
        
        # Save final results
        with open(self.output_dir / "final_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*20} FINAL RESULTS {'='*20}")
        print(f"Generations run: {self.generation}")
        print(f"Final Pareto front size: {len(final_pareto_front)}")
        print(f"Final hypervolume: {results['final_hypervolume']:.6f}")
        print(f"Converged: {results['converged']}")
        
        if final_pareto_front:
            print(f"\nTop 5 Pareto-optimal candidates:")
            for i, candidate in enumerate(final_pareto_front[:5]):
                comp_str = "".join(f"{elem}{count}" for elem, count in sorted(candidate.composition.items()))
                print(f"\n{i+1}. Composition: {comp_str}")
                print(f"   Properties:")
                for prop, value in candidate.properties.items():
                    if prop == 'ionic_conductivity':
                        print(f"     {prop}: {value:.2e}")
                    else:
                        print(f"     {prop}: {value:.4f}")
                print(f"   Objectives: {[f'{obj:.3f}' for obj in candidate.objectives]}")
                print(f"   Crowding distance: {candidate.crowding_distance:.3f}")
        
        return results


def main():
    """Main function to run the multi-objective genetic algorithm"""
    
    # Initialize and run Pareto GA
    ga = ParetoFrontGA(
        population_size=80,
        elite_count=6,
        tournament_size=5,
        mutation_rate=0.02,
        max_generations=50,
        convergence_threshold=15,
        output_dir="pareto_electrolyte_ga_results"
    )
    
    results = ga.run()
    
    print(f"\nMulti-Objective Genetic Algorithm completed!")
    print(f"Results saved to: {ga.output_dir}")
    print(f"Pareto front plots saved as PNG files")
    
    return results


if __name__ == "__main__":
    main()