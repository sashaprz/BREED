import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy
import os

from main_rl import predict_single_cif


@dataclass
class ElectrolyteCandidate:
    """Represents a solid state electrolyte candidate with CIF structure"""
    cif_file: str  # Path to CIF file
    composition: str = ""  # Will be extracted from CIF
    bandgap: float = 0.0
    sei_score: float = 0.0
    cei_score: float = 0.0
    ionic_conductivity: float = 0.0
    bulk_modulus: float = 0.0
    fitness: float = 0.0
    pareto_rank: int = 0  # For Pareto ranking
    crowding_distance: float = 0.0  # For diversity preservation
    dominated_count: int = 0  # Number of solutions that dominate this one
    dominates: List[int] = None  # Indices of solutions this one dominates
    
    def __post_init__(self):
        if self.dominates is None:
            self.dominates = []


class ExternalModelInterface:
    def __init__(self):
        # No external script call or path needed; use direct function calls
        pass

    def predict_properties(self, candidates: List[ElectrolyteCandidate]) -> None:
        for candidate in candidates:
            try:
                results = predict_single_cif(candidate.cif_file)
                candidate.composition = results.get("composition", "")
                candidate.bandgap = float(results.get("bandgap", 0.0))
                candidate.sei_score = float(results.get("sei_score", 0.0))
                candidate.cei_score = float(results.get("cei_score", 0.0))
                candidate.ionic_conductivity = float(results.get("ionic_conductivity", 0.0))
                candidate.bulk_modulus = float(results.get("bulk_modulus", 0.0))
            except Exception as e:
                print(f"Prediction failed for {candidate.cif_file}: {e}")
                candidate.composition = ""
                candidate.bandgap = 0.0
                candidate.sei_score = 0.0
                candidate.cei_score = 0.0
                candidate.ionic_conductivity = 0.0
                candidate.bulk_modulus = 0.0


class CrystaLLMInterface:
    """Interface to crystaLLM for generating CIF files"""
    
    @staticmethod
    def generate_cif_candidates(n_candidates: int, output_dir: str = "generated_cifs") -> List[ElectrolyteCandidate]:
        """
        Generate initial population CIF files using crystaLLM.
        Replace this with actual crystaLLM API calls.
        """
        os.makedirs(output_dir, exist_ok=True)
        candidates = []
        
        # Mock CIF generation - replace with actual crystaLLM calls
        base_compositions = [
            "Li7La3Zr2O12", "Li10GeP2S12", "Li6PS5Cl", "Li3PS4", 
            "Li7P3S11", "LiPON", "Li1.3Al0.3Ti1.7(PO4)3", "Li0.33La0.56TiO3",
            "Li6.4La3Zr1.4Ta0.6O12", "Li3N", "Li3OCl", "Li10SnP2S12"
        ]
        
        for i in range(n_candidates):
            cif_filename = f"{output_dir}/candidate_{i:04d}.cif"
            composition = random.choice(base_compositions)
            
            # Create mock CIF file (replace with actual crystaLLM generation)
            CrystaLLMInterface._create_mock_cif(cif_filename, composition)
            
            candidate = ElectrolyteCandidate(
                cif_file=cif_filename,
                composition=composition
            )
            candidates.append(candidate)
        
        return candidates
    
    @staticmethod
    def _create_mock_cif(filename: str, composition: str):
        """Create a mock CIF file - replace with actual crystaLLM output"""
        mock_cif = f"""data_{composition}
_cell_length_a    {np.random.uniform(8, 15):.4f}
_cell_length_b    {np.random.uniform(8, 15):.4f}
_cell_length_c    {np.random.uniform(8, 15):.4f}
_cell_angle_alpha {np.random.uniform(85, 95):.2f}
_cell_angle_beta  {np.random.uniform(85, 95):.2f}
_cell_angle_gamma {np.random.uniform(85, 95):.2f}
_space_group_name_H-M_alt 'P1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 0.0 0.0 0.0
"""
        with open(filename, 'w') as f:
            f.write(mock_cif)


class ParetoOptimizer:
    """Handles Pareto front calculation and NSGA-II style optimization"""
    
    def __init__(self, objectives: List[str], maximize: List[bool]):
        self.objectives = objectives
        self.maximize = maximize
    
    def dominates(self, candidate1: ElectrolyteCandidate, candidate2: ElectrolyteCandidate) -> bool:
        better_in_any = False
        worse_in_any = False
        
        for i, obj in enumerate(self.objectives):
            val1 = getattr(candidate1, obj)
            val2 = getattr(candidate2, obj)
            
            if self.maximize[i]:
                if val1 > val2:
                    better_in_any = True
                elif val1 < val2:
                    worse_in_any = True
            else:  # minimize
                if val1 < val2:
                    better_in_any = True
                elif val1 > val2:
                    worse_in_any = True
        
        return better_in_any and not worse_in_any
    
    def fast_non_dominated_sort(self, population: List[ElectrolyteCandidate]) -> List[List[int]]:
        # Reset domination info
        for i, candidate in enumerate(population):
            candidate.dominated_count = 0
            candidate.dominates = []
        
        # Calculate domination relationships
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j:
                    if self.dominates(population[i], population[j]):
                        population[i].dominates.append(j)
                    elif self.dominates(population[j], population[i]):
                        population[i].dominated_count += 1
        
        # Create fronts
        fronts = []
        current_front = []
        
        # First front (non-dominated solutions)
        for i, candidate in enumerate(population):
            if candidate.dominated_count == 0:
                candidate.pareto_rank = 0
                current_front.append(i)
        
        fronts.append(current_front[:])
        
        # Subsequent fronts
        front_num = 0
        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                for j in population[i].dominates:
                    population[j].dominated_count -= 1
                    if population[j].dominated_count == 0:
                        population[j].pareto_rank = front_num + 1
                        next_front.append(j)
            front_num += 1
            current_front = next_front
            if len(current_front) > 0:
                fronts.append(current_front[:])
        
        return fronts
    
    def calculate_crowding_distance(self, population: List[ElectrolyteCandidate], front: List[int]) -> None:
        if len(front) <= 2:
            for i in front:
                population[i].crowding_distance = float('inf')
            return
        
        for i in front:
            population[i].crowding_distance = 0.0
        
        for obj_idx, obj in enumerate(self.objectives):
            front_sorted = sorted(front, key=lambda x: getattr(population[x], obj))
            
            population[front_sorted[0]].crowding_distance = float('inf')
            population[front_sorted[-1]].crowding_distance = float('inf')
            
            obj_min = getattr(population[front_sorted[0]], obj)
            obj_max = getattr(population[front_sorted[-1]], obj)
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            for i in range(1, len(front_sorted) - 1):
                idx = front_sorted[i]
                prev_val = getattr(population[front_sorted[i-1]], obj)
                next_val = getattr(population[front_sorted[i+1]], obj)
                population[idx].crowding_distance += (next_val - prev_val) / obj_range
    
    def get_pareto_front(self, population: List[ElectrolyteCandidate]) -> List[ElectrolyteCandidate]:
        fronts = self.fast_non_dominated_sort(population)
        if len(fronts) > 0:
            return [population[i] for i in fronts[0]]
        return []


class FitnessEvaluator:
    """Enhanced fitness evaluator with target-based optimization and Pareto support"""
    
    def __init__(self, targets: Optional[Dict[str, float]] = None, pareto_mode: bool = True):
        self.pareto_mode = pareto_mode
        self.targets = targets or {}
        
        # Default property ranges for normalization
        self.property_ranges = {
            "bandgap": {"min": 1.0, "max": 6.0},
            "sei_score": {"min": 0.0, "max": 1.0},
            "cei_score": {"min": 0.0, "max": 1.0},
            "ionic_conductivity": {"min": 1e-6, "max": 1e-2, "log_scale": True},
            "bulk_modulus": {"min": 10, "max": 200}
        }
        
        if self.pareto_mode:
            self.pareto_optimizer = ParetoOptimizer(
                objectives=["sei_score", "cei_score", "ionic_conductivity", "bandgap"],
                maximize=[True, True, True, True]
            )
    
    def calculate_target_based_fitness(self, candidate: ElectrolyteCandidate) -> float:
        if not self.targets:
            return 0.0
        
        total_score = 0.0
        weight_sum = 0.0
        
        for prop, target_val in self.targets.items():
            if hasattr(candidate, prop):
                actual_val = getattr(candidate, prop)
                
                if prop == "ionic_conductivity":
                    if actual_val > 0 and target_val > 0:
                        log_actual = np.log10(actual_val)
                        log_target = np.log10(target_val)
                        distance = abs(log_actual - log_target) / 2.0
                    else:
                        distance = 1.0
                else:
                    prop_range = self.property_ranges.get(prop, {"min": 0, "max": 1})
                    range_size = prop_range["max"] - prop_range["min"]
                    distance = abs(actual_val - target_val) / range_size
                
                score = max(0.0, 1.0 - distance)
                total_score += score
                weight_sum += 1.0
        
        return total_score / weight_sum if weight_sum > 0 else 0.0


class GeneticAlgorithm:
    """Enhanced genetic algorithm with Pareto optimization and target-based selection"""
    
    def __init__(self, 
                 population_size: int = 100,
                 max_generations: int = 100,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.02,
                 tournament_size: int = 5,
                 elite_size: int = 10,
                 stagnation_limit: int = 15,
                 targets: Optional[Dict[str, float]] = None,
                 use_pareto: bool = True):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.stagnation_limit = stagnation_limit
        self.use_pareto = use_pareto
        self.targets = targets
        
        self.model_interface = ExternalModelInterface()
        self.fitness_evaluator = FitnessEvaluator(targets=targets, pareto_mode=use_pareto)
        
        self.generation_stats = []
        self.pareto_history = []
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0
    
    def initialize_population(self) -> List[ElectrolyteCandidate]:
        print(f"Generating initial population of {self.population_size} CIF candidates...")
        population = CrystaLLMInterface.generate_cif_candidates(self.population_size)
        
        print("Running ML model predictions...")
        self.model_interface.predict_properties(population)
        
        for candidate in population:
            if self.targets:
                candidate.fitness = self.fitness_evaluator.calculate_target_based_fitness(candidate)
            else:
                candidate.fitness = (
                    0.25 * candidate.sei_score +
                    0.25 * candidate.cei_score +
                    0.25 * min(1.0, candidate.ionic_conductivity * 1000) +  # Scale conductivity
                    0.20 * min(1.0, candidate.bandgap / 5.0) +  # Normalize bandgap
                    0.05 * min(1.0, candidate.bulk_modulus / 200.0)  # Normalize bulk modulus
                )
        
        return population
    
    def nsga_ii_selection(self, population: List[ElectrolyteCandidate]) -> List[ElectrolyteCandidate]:
        fronts = self.fitness_evaluator.pareto_optimizer.fast_non_dominated_sort(population)
        new_population = []
        front_idx = 0
        
        while len(new_population) + len(fronts[front_idx]) <= self.population_size:
            self.fitness_evaluator.pareto_optimizer.calculate_crowding_distance(population, fronts[front_idx])
            for i in fronts[front_idx]:
                new_population.append(population[i])
            front_idx += 1
            if front_idx >= len(fronts):
                break
        
        if len(new_population) < self.population_size and front_idx < len(fronts):
            remaining_front = fronts[front_idx]
            self.fitness_evaluator.pareto_optimizer.calculate_crowding_distance(population, remaining_front)
            remaining_front.sort(key=lambda x: population[x].crowding_distance, reverse=True)
            needed = self.population_size - len(new_population)
            for i in range(min(needed, len(remaining_front))):
                new_population.append(population[remaining_front[i]])
        
        return new_population
    
    def crossover_cif(self, parent1: ElectrolyteCandidate, parent2: ElectrolyteCandidate, 
                     generation: int) -> Tuple[ElectrolyteCandidate, ElectrolyteCandidate]:
        if random.random() > self.crossover_prob:
            return deepcopy(parent1), deepcopy(parent2)
        
        child1_file = f"generated_cifs/offspring_{generation}_{random.randint(1000,9999)}.cif"
        child2_file = f"generated_cifs/offspring_{generation}_{random.randint(1000,9999)}.cif"
        
        CrystaLLMInterface._create_mock_cif(child1_file, f"Offspring_{generation}_A")
        CrystaLLMInterface._create_mock_cif(child2_file, f"Offspring_{generation}_B")
        
        child1 = ElectrolyteCandidate(cif_file=child1_file)
        child2 = ElectrolyteCandidate(cif_file=child2_file)
        
        return child1, child2
    
    def get_best_candidates(self, population: List[ElectrolyteCandidate], 
                            n_candidates: int = 10) -> List[ElectrolyteCandidate]:
        if self.use_pareto:
            pareto_front = self.fitness_evaluator.pareto_optimizer.get_pareto_front(population)
            if len(pareto_front) <= n_candidates:
                return pareto_front
            else:
                self.fitness_evaluator.pareto_optimizer.calculate_crowding_distance(
                    pareto_front, list(range(len(pareto_front)))
                )
                pareto_front.sort(key=lambda x: x.crowding_distance, reverse=True)
                return pareto_front[:n_candidates]
        else:
            population.sort(key=lambda x: x.fitness, reverse=True)
            return population[:n_candidates]
    
    def run(self) -> Tuple[List[ElectrolyteCandidate], List[Dict]]:
        print("Starting Multi-Objective GA for Solid State Electrolyte Screening")
        print("=" * 70)
        if self.targets:
            print("Target Properties:")
            for prop, val in self.targets.items():
                print(f"  {prop}: {val}")
        print(f"Using {'Pareto' if self.use_pareto else 'Weighted'} optimization")
        print("=" * 70)
        
        population = self.initialize_population()
        
        for generation in range(self.max_generations):
            if self.use_pareto:
                pareto_front = self.fitness_evaluator.pareto_optimizer.get_pareto_front(population)
                best_fitness = np.mean([c.fitness for c in pareto_front]) if pareto_front else 0
                self.pareto_history.append(len(pareto_front))
            else:
                population.sort(key=lambda x: x.fitness, reverse=True)
                best_fitness = population[0].fitness
            
            avg_fitness = np.mean([c.fitness for c in population])
            
            stats = {
                "generation": generation,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "pareto_front_size": len(pareto_front) if self.use_pareto else 1
            }
            self.generation_stats.append(stats)
            
            if abs(best_fitness - self.last_best_fitness) < 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.last_best_fitness = best_fitness
            
            # Print concise progress summary only
            if self.use_pareto:
                print(f"Gen {generation:3d}: Pareto Front Size={len(pareto_front):2d}, "
                      f"Avg Fitness={avg_fitness:.4f}, Stagnation={self.stagnation_counter}")
            else:
                print(f"Gen {generation:3d}: Best={best_fitness:.4f}, "
                      f"Avg={avg_fitness:.4f}, Stagnation={self.stagnation_counter}")
            
            if self.stagnation_counter >= self.stagnation_limit:
                print(f"\nStopping: No improvement for {self.stagnation_limit} generations")
                break
            
            if generation < self.max_generations - 1:
                if self.use_pareto:
                    population = self.nsga_ii_selection(population)
                else:
                    population.sort(key=lambda x: x.fitness, reverse=True)
                    new_population = population[:self.elite_size]
                    
                    while len(new_population) < self.population_size:
                        parent1 = self.tournament_selection(population)
                        parent2 = self.tournament_selection(population)
                        child1, child2 = self.crossover_cif(parent1, parent2, generation)
                        
                        self.model_interface.predict_properties([child1, child2])
                        for child in [child1, child2]:
                            if len(new_population) < self.population_size:
                                child.fitness = self.fitness_evaluator.calculate_target_based_fitness(child)
                                new_population.append(child)
                    
                    population = new_population
        
        print("\nOptimization Complete!")
        
        best_candidates = self.get_best_candidates(population, n_candidates=10)
        
        # Plot generation statistics after run completes
        self.plot_statistics()

        return best_candidates, self.generation_stats
    
    def tournament_selection(self, population: List[ElectrolyteCandidate]) -> ElectrolyteCandidate:
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def print_best_candidates(self, candidates: List[ElectrolyteCandidate]):
        print("\n" + "="*60)
        print(f"TOP {len(candidates)} SOLID STATE ELECTROLYTE CANDIDATES")
        print("="*60)
        
        for i, candidate in enumerate(candidates, 1):
            print(f"\n--- Rank {i} ---")
            print(f"CIF File: {candidate.cif_file}")
            print(f"Composition: {candidate.composition}")
            if self.targets:
                print(f"Target-based Fitness: {candidate.fitness:.4f}")
            if hasattr(candidate, 'pareto_rank'):
                print(f"Pareto Rank: {candidate.pareto_rank}")
            
            print("Properties:")
            print(f"  Bandgap:           {candidate.bandgap:.3f} eV")
            print(f"  SEI Score:         {candidate.sei_score:.3f}")
            print(f"  CEI Score:         {candidate.cei_score:.3f}")
            print(f"  Ionic Conductivity: {candidate.ionic_conductivity:.2e} S/cm")
            print(f"  Bulk Modulus:      {candidate.bulk_modulus:.1f} GPa")
            
            if self.targets:
                print("Target Distances:")
                for prop, target in self.targets.items():
                    actual = getattr(candidate, prop, 0)
                    if prop == "ionic_conductivity":
                        if actual > 0 and target > 0:
                            distance = abs(np.log10(actual) - np.log10(target))
                        else:
                            distance = float('inf')
                        print(f"  {prop}: {distance:.3f} (log scale)")
                    else:
                        distance = abs(actual - target)
                        print(f"  {prop}: {distance:.3f}")
    
    def plot_statistics(self):
        if not self.generation_stats:
            print("No generation statistics to plot.")
            return
        
        generations = [stat['generation'] for stat in self.generation_stats]
        best_fitness = [stat['best_fitness'] for stat in self.generation_stats]
        avg_fitness = [stat['avg_fitness'] for stat in self.generation_stats]
        pareto_sizes = [stat['pareto_front_size'] for stat in self.generation_stats]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(generations, best_fitness, label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Best Fitness over Generations')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(generations, avg_fitness, label='Average Fitness', color='orange')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Average Fitness over Generations')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(generations, pareto_sizes, label='Pareto Front Size', color='green')
        plt.xlabel('Generation')
        plt.ylabel('Size')
        plt.title('Pareto Front Size over Generations')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Define target properties for optimization
    target_properties = {
        "ionic_conductivity": 1e-3,  # Target: 1 mS/cm
        "sei_score": 0.8,            # Target: High SEI stability
        "cei_score": 0.75,           # Target: Good CEI properties
        "bandgap": 3.0               # Target: 3 eV bandgap
    }
    
    # Initialize and run genetic algorithm
    ga = GeneticAlgorithm(
        population_size=100,
        max_generations=100,
        crossover_prob=0.8,
        mutation_prob=0.02,
        tournament_size=5,
        elite_size=10,
        stagnation_limit=15,
        targets=target_properties,
        use_pareto=True
    )

    # Override model_interface with updated one
    ga.model_interface = ExternalModelInterface()
    
    # Run optimization
    best_candidates, stats = ga.run()
    
    # Display results
    ga.print_best_candidates(best_candidates)
    
    print(f"\nReady for DFT screening: {len(best_candidates)} candidates")
    print("CIF files for DFT:")
    for candidate in best_candidates:
        print(f"  {candidate.cif_file}")
