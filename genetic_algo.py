import os
import sys
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy
import tempfile
import shutil
from pathlib import Path

# PyXtal and pymatgen imports
from pyxtal import pyxtal
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Import your ML prediction functions
from main_rl import predict_single_cif


@dataclass
class TargetProperties:
    """Target properties for fitness evaluation"""
    ionic_conductivity: float = 1.0e-3  # S/cm
    bandgap: float = 3.5  # eV (wide bandgap for stability)
    sei_score: float = 0.8  # Higher is better
    cei_score: float = 0.8  # Higher is better
    bulk_modulus: float = 50.0  # GPa (moderate stiffness)
    
    # Weights for each property in fitness calculation
    weights: Dict[str, float] = field(default_factory=lambda: {
        'ionic_conductivity': 0.0001,
        'bandgap': 0.3,
        'sei_score': 0.9,
        'cei_score': 0.85,
        'bulk_modulus': 0.1
    })


@dataclass
class GACandidate:
    """Represents a candidate electrolyte in the genetic algorithm"""
    composition: Dict[str, int]  # Element: count
    lattice_params: Dict[str, float]  # a, b, c, alpha, beta, gamma
    space_group: int
    structure: Optional[Structure] = None
    cif_path: Optional[str] = None
    properties: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
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


class ElectrolyteGA:
    """Genetic Algorithm for solid-state electrolyte discovery"""
    
    def __init__(self, 
                 population_size: int = 80,
                 elite_count: int = 6,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.02,
                 max_generations: int = 50,
                 convergence_threshold: int = 15,
                 output_dir: str = "ga_results"):
        
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
        
        # Element pool for Li-ion conductors (common elements in solid electrolytes)
        self.element_pool = {
            'Li': {'min': 1, 'max': 8, 'weight': 0.3},  # Essential for Li conduction
            'O': {'min': 1, 'max': 12, 'weight': 0.25}, # Common in oxides
            'P': {'min': 0, 'max': 4, 'weight': 0.1},   # NASICON-type
            'S': {'min': 0, 'max': 8, 'weight': 0.1},   # Sulfides
            'Ti': {'min': 0, 'max': 4, 'weight': 0.05}, # NASICON-type
            'Zr': {'min': 0, 'max': 4, 'weight': 0.05}, # NASICON-type
            'Al': {'min': 0, 'max': 4, 'weight': 0.05}, # Garnets
            'La': {'min': 0, 'max': 4, 'weight': 0.03}, # Garnets
            'Cl': {'min': 0, 'max': 6, 'weight': 0.03}, # Halides
            'F': {'min': 0, 'max': 6, 'weight': 0.04},  # Halides
        }
        
        # Space groups commonly found in solid electrolytes
        self.common_space_groups = [1, 2, 14, 15, 62, 63, 136, 141, 166, 167, 194, 225]
        
        self.target_properties = TargetProperties()
        self.population: List[GACandidate] = []
        self.best_fitness_history: List[float] = []
        self.generation = 0
        self.structure_matcher = StructureMatcher()
        
    def generate_initial_population(self) -> List[GACandidate]:
        """Generate initial population using PyXtal"""
        print(f"Generating initial population of {self.population_size} candidates...")
        
        candidates = []
        attempts = 0
        max_attempts = self.population_size * 10
        
        while len(candidates) < self.population_size and attempts < max_attempts:
            attempts += 1
            
            try:
                candidate = self._generate_random_candidate()
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
    
    def _generate_random_candidate(self) -> Optional[GACandidate]:
        """Generate a single random candidate using PyXtal"""
        
        # Randomly select composition
        composition = self._generate_random_composition()
        if not composition:
            return None
            
        # Convert composition to PyXtal format
        species = list(composition.keys())
        numIons = 5 #list(composition.values())
        
        # Randomly select space group
        space_group = 255 #random.choice(self.common_space_groups)
        
        # Generate structure with PyXtal
        crystal = pyxtal()
        
        try:
            # Try to generate a valid structure
            crystal.from_random(dim=3, group=space_group, species=species, numIons=numIons)
            
            if not crystal.valid:
                return None
                
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
            print(f"    PyXtal generation failed: {e}")
            return None
    
    def _generate_random_composition(self) -> Dict[str, int]:
        """Generate a random but chemically reasonable composition"""
        composition = {}
        
        # Always include Li
        composition['Li'] = random.randint(
            self.element_pool['Li']['min'],
            self.element_pool['Li']['max']
        )
        
        # Add other elements based on probability
        for element, params in self.element_pool.items():
            if element == 'Li':
                continue
                
            # Higher weight = higher probability of inclusion
            if random.random() < params['weight']:
                count = random.randint(params['min'], max(1, params['max']))
                if count > 0:
                    composition[element] = count
        
        # Ensure we have at least 2 elements
        if len(composition) < 2:
            # Add oxygen if not present
            if 'O' not in composition:
                composition['O'] = random.randint(1, 4)
        
        return composition
    
    def _is_valid_candidate(self, candidate: GACandidate) -> bool:
        """Check if candidate is chemically and physically valid"""
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
            
        # Check for Li content (essential for Li-ion conduction)
        li_count = candidate.composition.get('Li', 0)
        if li_count == 0:
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
        """Evaluate fitness for all candidates in the population"""
        print(f"Evaluating properties for {len(candidates)} candidates...")
        
        for i, candidate in enumerate(candidates):
            if i % 10 == 0:
                print(f"  Evaluating candidate {i+1}/{len(candidates)}...")
                
            try:
                # Get ML predictions
                if candidate.cif_path and os.path.exists(candidate.cif_path):
                    results = predict_single_cif(candidate.cif_path, verbose=False)
                    
                    candidate.properties = {
                        'ionic_conductivity': results.get('ionic_conductivity', 0.0),
                        'bandgap': results.get('bandgap', 0.0),
                        'sei_score': results.get('sei_score', 0.0),
                        'cei_score': results.get('cei_score', 0.0),
                        'bulk_modulus': results.get('bulk_modulus', 0.0)
                    }
                    
                    # Calculate fitness
                    candidate.fitness = self._calculate_fitness(candidate.properties)
                    
                else:
                    print(f"    Warning: CIF file not found for candidate {i+1}")
                    candidate.fitness = 0.0
                    
            except Exception as e:
                print(f"    Error evaluating candidate {i+1}: {e}")
                candidate.fitness = 0.0
    
    def _calculate_fitness(self, properties: Dict[str, float]) -> float:
        """Calculate fitness score based on how close properties are to targets"""
        fitness = 0.0
        targets = self.target_properties
        
        # Ionic conductivity (logarithmic scale, higher is better)
        if properties['ionic_conductivity'] > 0:
            ic_score = 1.0 - abs(np.log10(properties['ionic_conductivity']) - 
                               np.log10(targets.ionic_conductivity)) / 5.0  # 5 orders of magnitude range
            fitness += max(0, ic_score) * targets.weights['ionic_conductivity']
        
        # Bandgap (linear scale, closer to target is better)
        bg_diff = abs(properties['bandgap'] - targets.bandgap) / targets.bandgap
        bg_score = max(0, 1.0 - bg_diff)
        fitness += bg_score * targets.weights['bandgap']
        
        # SEI and CEI scores (higher is better)
        fitness += properties['sei_score'] * targets.weights['sei_score']
        fitness += properties['cei_score'] * targets.weights['cei_score']
        
        # Bulk modulus (closer to target is better)
        if targets.bulk_modulus > 0:
            bm_diff = abs(properties['bulk_modulus'] - targets.bulk_modulus) / targets.bulk_modulus
            bm_score = max(0, 1.0 - bm_diff)
            fitness += bm_score * targets.weights['bulk_modulus']
        
        return fitness
    
    def select_parents(self, population: List[GACandidate]) -> List[GACandidate]:
        """Select parents using tournament selection"""
        parents = []
        
        # Number of offspring needed (population_size - elite_count)
        num_offspring = self.population_size - self.elite_count
        
        for _ in range(num_offspring):
            # Tournament selection
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
            
        return parents
    
    def crossover(self, parent1: GACandidate, parent2: GACandidate) -> GACandidate:
        """Create offspring through crossover"""
        
        # Weighted average of lattice parameters
        new_lattice_params = {}
        weight = random.uniform(0.3, 0.7)  # Bias toward one parent
        
        for param in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
            new_lattice_params[param] = (weight * parent1.lattice_params[param] + 
                                       (1 - weight) * parent2.lattice_params[param])
        
        # Composition mixing - combine elements from both parents
        new_composition = {}
        all_elements = set(parent1.composition.keys()) | set(parent2.composition.keys())
        
        for element in all_elements:
            count1 = parent1.composition.get(element, 0)
            count2 = parent2.composition.get(element, 0)
            
            if count1 > 0 and count2 > 0:
                # Both parents have this element - take weighted average
                new_count = int(weight * count1 + (1 - weight) * count2)
                new_count = max(1, new_count)  # Ensure at least 1
            elif count1 > 0:
                # Only parent1 has this element
                new_count = count1 if random.random() < 0.7 else 0
            else:
                # Only parent2 has this element  
                new_count = count2 if random.random() < 0.7 else 0
                
            if new_count > 0:
                new_composition[element] = new_count
        
        # Ensure Li is present
        if 'Li' not in new_composition or new_composition['Li'] == 0:
            new_composition['Li'] = max(parent1.composition.get('Li', 1),
                                      parent2.composition.get('Li', 1))
        
        # Select space group (bias toward more successful parent)
        if parent1.fitness > parent2.fitness:
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
        
        # Generate structure using PyXtal
        try:
            crystal = pyxtal()
            species = list(new_composition.keys())
            numIons = list(new_composition.values())
            
            crystal.from_random(dim=3, group=space_group, species=species, numIons=numIons)
            
            if crystal.valid:
                offspring.structure = crystal.to_pymatgen()
                # Update lattice parameters with actual values
                lattice = offspring.structure.lattice
                offspring.lattice_params = {
                    'a': lattice.a, 'b': lattice.b, 'c': lattice.c,
                    'alpha': lattice.alpha, 'beta': lattice.beta, 'gamma': lattice.gamma
                }
                self._generate_cif_file(offspring)
                
        except Exception as e:
            print(f"    Crossover structure generation failed: {e}")
            return None
            
        return offspring
    
    def mutate(self, candidate: GACandidate) -> GACandidate:
        """Apply mutation to a candidate"""
        if random.random() > self.mutation_rate:
            return candidate
            
        mutated = deepcopy(candidate)
        mutation_type = random.choice(['composition', 'lattice', 'space_group'])
        
        if mutation_type == 'composition':
            # Randomly modify one element's count
            elements = list(mutated.composition.keys())
            element = random.choice(elements)
            
            if element == 'Li':
                # Don't remove Li completely
                mutated.composition[element] = max(1, mutated.composition[element] + random.randint(-2, 2))
            else:
                new_count = mutated.composition[element] + random.randint(-2, 2)
                if new_count <= 0:
                    del mutated.composition[element]
                else:
                    mutated.composition[element] = new_count
                    
        elif mutation_type == 'lattice':
            # Mutate lattice parameters by ±10%
            param = random.choice(['a', 'b', 'c'])
            factor = random.uniform(0.9, 1.1)
            mutated.lattice_params[param] *= factor
            
        elif mutation_type == 'space_group':
            # Change to a different common space group
            mutated.space_group = random.choice(self.common_space_groups)
        
        # Regenerate structure
        try:
            crystal = pyxtal()
            species = list(mutated.composition.keys())
            numIons = list(mutated.composition.values())
            
            crystal.from_random(dim=3, group=mutated.space_group, species=species, numIons=numIons)
            
            if crystal.valid:
                mutated.structure = crystal.to_pymatgen()
                lattice = mutated.structure.lattice
                mutated.lattice_params = {
                    'a': lattice.a, 'b': lattice.b, 'c': lattice.c,
                    'alpha': lattice.alpha, 'beta': lattice.beta, 'gamma': lattice.gamma
                }
                self._generate_cif_file(mutated)
                return mutated
            else:
                return candidate  # Return original if mutation failed
                
        except Exception:
            return candidate  # Return original if mutation failed
    
    def evolve_generation(self) -> List[GACandidate]:
        """Create next generation through selection, crossover, and mutation"""
        print(f"Evolving generation {self.generation + 1}...")
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Elite selection - top performers move to next generation
        elite = self.population[:self.elite_count]
        print(f"  Elite candidates (top {self.elite_count}): fitness = {[f'{c.fitness:.4f}' for c in elite]}")
        
        # Select parents for breeding
        parents = self.select_parents(self.population)
        
        # Generate offspring through crossover and mutation
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[min(i+1, len(parents)-1)]  # Handle odd number of parents
            
            # Crossover
            child = self.crossover(parent1, parent2)
            if child and self._is_valid_candidate(child):
                # Apply mutation
                child = self.mutate(child)
                if child:
                    offspring.append(child)
        
        # If we don't have enough offspring, fill with additional random candidates
        while len(offspring) < (self.population_size - self.elite_count):
            candidate = self._generate_random_candidate()
            if candidate and self._is_valid_candidate(candidate):
                offspring.append(candidate)
        
        # Combine elite and offspring for next generation
        next_generation = elite + offspring[:self.population_size - self.elite_count]
        
        print(f"  Generated {len(offspring)} offspring")
        print(f"  Next generation size: {len(next_generation)}")
        
        return next_generation
    
    def save_generation_results(self) -> None:
        """Save current generation results to files"""
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
                'fitness': candidate.fitness,
                'generation': candidate.generation,
                'parent_ids': candidate.parent_ids,
                'cif_path': candidate.cif_path
            }
            population_data.append(data)
        
        with open(gen_dir / "population.json", 'w') as f:
            json.dump(population_data, f, indent=2)
        
        # Save best candidates
        top_10 = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:10]
        
        print(f"\nGeneration {self.generation} - Top 10 candidates:")
        for i, candidate in enumerate(top_10):
            comp_str = "".join(f"{elem}{count}" for elem, count in sorted(candidate.composition.items()))
            print(f"  {i+1}. {comp_str} | Fitness: {candidate.fitness:.4f} | "
                  f"IC: {candidate.properties['ionic_conductivity']:.2e} S/cm")
    
    def run(self) -> Dict[str, Any]:
        """Run the genetic algorithm"""
        print(f"Starting Genetic Algorithm for Solid-State Electrolyte Discovery")
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {self.max_generations}")
        print(f"Elite count: {self.elite_count}")
        print(f"Tournament size: {self.tournament_size}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Convergence threshold: {self.convergence_threshold} generations")
        print("-" * 80)
        
        # Generate initial population
        self.population = self.generate_initial_population()
        
        # Evaluate initial population
        self.evaluate_population(self.population)
        
        # Track best fitness
        best_fitness = max(self.population, key=lambda x: x.fitness).fitness
        self.best_fitness_history.append(best_fitness)
        
        self.save_generation_results()
        
        # Evolution loop
        generations_without_improvement = 0
        
        for generation in range(1, self.max_generations + 1):
            self.generation = generation
            
            print(f"\n{'='*20} Generation {generation} {'='*20}")
            
            # Create next generation
            self.population = self.evolve_generation()
            
            # Evaluate new population
            self.evaluate_population(self.population)
            
            # Check for improvement
            current_best = max(self.population, key=lambda x: x.fitness).fitness
            self.best_fitness_history.append(current_best)
            
            if current_best > best_fitness:
                best_fitness = current_best
                generations_without_improvement = 0
                print(f"New best fitness: {best_fitness:.4f}")
            else:
                generations_without_improvement += 1
                print(f"No improvement for {generations_without_improvement} generations")
            
            self.save_generation_results()
            
            # Check convergence
            if generations_without_improvement >= self.convergence_threshold:
                print(f"\nConverged after {generation} generations (no improvement for {self.convergence_threshold} generations)")
                break
        
        # Final results
        final_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        best_candidate = final_population[0]
        
        results = {
            'generations_run': self.generation,
            'final_population_size': len(final_population),
            'best_fitness': best_candidate.fitness,
            'best_composition': best_candidate.composition,
            'best_properties': best_candidate.properties,
            'fitness_history': self.best_fitness_history,
            'converged': generations_without_improvement >= self.convergence_threshold
        }
        
        # Save final results
        with open(self.output_dir / "final_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*20} FINAL RESULTS {'='*20}")
        comp_str = "".join(f"{elem}{count}" for elem, count in sorted(best_candidate.composition.items()))
        print(f"Best composition: {comp_str}")
        print(f"Best fitness: {best_candidate.fitness:.4f}")
        print(f"Properties:")
        for prop, value in best_candidate.properties.items():
            if prop == 'ionic_conductivity':
                print(f"  {prop}: {value:.2e}")
            else:
                print(f"  {prop}: {value:.4f}")
        
        return results


def main():
    """Main function to run the genetic algorithm"""
    
    # Initialize and run GA
    ga = ElectrolyteGA(
        population_size=80,
        elite_count=6,
        tournament_size=5,
        mutation_rate=0.02,
        max_generations=50,
        convergence_threshold=15,
        output_dir="electrolyte_ga_results"
    )
    
    results = ga.run()
    
    print(f"\nGenetic Algorithm completed!")
    print(f"Results saved to: {ga.output_dir}")
    
    return results


if __name__ == "__main__":
    main()