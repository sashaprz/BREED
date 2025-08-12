import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from copy import deepcopy
import os
import matplotlib.pyplot as plt

from main_rl import predict_single_cif


@dataclass
class ElectrolyteCandidate:
    cif_file: str
    composition: str = ""
    bandgap: float = 0.0
    sei_score: float = 0.0
    cei_score: float = 0.0
    ionic_conductivity: float = 0.0
    bulk_modulus: float = 0.0
    fitness: float = 0.0
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    dominated_count: int = 0
    dominates: List[int] = None

    def __post_init__(self):
        if self.dominates is None:
            self.dominates = []


class ExternalInterface:
    def __init__(self):
        pass

    def predict_properties(self, candidates: List[ElectrolyteCandidate]) -> None:
        for candidate in candidates:
            try:
                res = predict_single_cif(candidate.cif_file, verbose=False)
                if res is None:
                    raise ValueError("Prediction returned None")
                candidate.composition = res.get("composition", "")
                candidate.bandgap = float(res.get("bandgap", 0.0))
                candidate.sei_score = float(res.get("sei_score", 0.0))
                candidate.cei_score = float(res.get("cei_score", 0.0))
                candidate.ionic_conductivity = float(res.get("ionic_conductivity", 0.0))
                candidate.bulk_modulus = float(res.get("bulk_modulus", 0.0))
            except Exception as e:
                print(f"Prediction failed for {candidate.cif_file}: {e}")
                candidate.composition = ""
                candidate.bandgap = 0.0
                candidate.sei_score = 0.0
                candidate.cei_score = 0.0
                candidate.ionic_conductivity = 0.0
                candidate.bulk_modulus = 0.0


class CrystaLLMInterface:
    @staticmethod
    def generate_candidates(n: int, outdir="generated_cifs") -> List[ElectrolyteCandidate]:
        os.makedirs(outdir, exist_ok=True)
        base_compositions = [
            "Li7La3Zr2O12", "Li10GeP2S12", "Li6PS5Cl", "Li3PS4",
            "Li7P3S11", "LiPON", "Li1.3Al0.3Ti1.7(PO4)3", "Li0.33La0.56TiO3",
            "Li6.4La3Zr1.4Ta0.6", "Li3N", "Li3OCl", "Li10SnP2S12"
        ]
        candidates = []
        for i in range(n):
            comp = random.choice(base_compositions)
            fname = os.path.join(outdir, f"candidate_{i:04d}.cif")
            CrystaLLMInterface._create_mock_cif(fname, comp)
            candidates.append(ElectrolyteCandidate(cif_file=fname, composition=comp))
        return candidates

    @staticmethod
    def _create_mock_cif(filename, composition):
        content = f"""data_{composition}
_cell_length_a     {np.random.uniform(8, 15):.4f}
_cell_length_b     {np.random.uniform(8, 15):.4f}
_cell_length_c     {np.random.uniform(8, 15):.4f}
_cell_angle_alpha  {np.random.uniform(85, 95):.2f}
_cell_angle_beta   {np.random.uniform(85, 95):.2f}
_cell_angle_gamma  {np.random.uniform(85, 95):.2f}
_space_group_name_H-M 'P1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 0.0 0.0 0.0
"""
        with open(filename, "w") as f:
            f.write(content)


def mutate_cif(candidate: ElectrolyteCandidate, mutation_prob: float) -> ElectrolyteCandidate:
    # Mutate lattice params by ±5%, ensuring positive reasonable bounds
    with open(candidate.cif_file) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith(("_cell_length_", "_cell_angle_")) and random.random() < mutation_prob:
            parts = line.strip().split()
            val = float(parts[1])
            val_mut = val * (1 + random.uniform(-0.05, 0.05))
            # Clamp values
            if "length" in parts[0]:
                val_mut = max(1.0, val_mut)
            elif "angle" in parts[0]:
                val_mut = min(max(30.0, val_mut), 150.0)
            new_line = f"{parts[0]} {val_mut:.4f}\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    mutated_filename = candidate.cif_file.replace(".cif", f"_mut{random.randint(1000,9999)}.cif")
    with open(mutated_filename, "w") as f:
        f.writelines(new_lines)

    return ElectrolyteCandidate(cif_file=mutated_filename)


def crossover_cif_files(parent1: ElectrolyteCandidate, parent2: ElectrolyteCandidate, generation: int) -> Tuple[ElectrolyteCandidate, ElectrolyteCandidate]:
    def read_params(fpath):
        params = {}
        with open(fpath) as f:
            for line in f:
                if line.startswith("_cell_"):
                    parts = line.strip().split()
                    params[parts[0]] = float(parts[1])
        return params

    p1_params = read_params(parent1.cif_file)
    p2_params = read_params(parent2.cif_file)

    child1_params = {}
    child2_params = {}
    for k in p1_params.keys():
        if random.random() < 0.5:
            child1_params[k] = p1_params[k]
            child2_params[k] = p2_params[k]
        else:
            child1_params[k] = p2_params[k]
            child2_params[k] = p1_params[k]

    def write_cif(file, params):
        content = "data_Offspring\n"
        for k, v in params.items():
            if "length" in k:
                content += f"{k} {v:.4f}\n"
            else:
                content += f"{k} {v:.2f}\n"
        content += "_space_group_name_H-M 'P1'\nloop_\n"
        content += "_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        content += "Li1 0.0 0.0 0.0\n"
        with open(file, "w") as f:
            f.write(content)

    fname1 = f"generated_cifs/offspring_{generation}_{random.randint(1000,9999)}.cif"
    fname2 = f"generated_cifs/offspring_{generation}_{random.randint(1000,9999)}.cif"

    write_cif(fname1, child1_params)
    write_cif(fname2, child2_params)

    return ElectrolyteCandidate(cif_file=fname1), ElectrolyteCandidate(cif_file=fname2)


class ParetoOptimizer:
    def __init__(self, objectives: List[str], maximize: List[bool]):
        self.objectives = objectives
        self.maximize = maximize

    def dominates(self, c1, c2) -> bool:
        better = False
        worse = False
        for i, obj in enumerate(self.objectives):
            v1 = getattr(c1, obj)
            v2 = getattr(c2, obj)
            if self.maximize[i]:
                if v1 > v2:
                    better = True
                elif v1 < v2:
                    worse = True
            else:
                if v1 < v2:
                    better = True
                elif v1 > v2:
                    worse = True
        return better and not worse

    def fast_non_dominated_sort(self, pop):
        for c in pop:
            c.dominated_count = 0
            c.dominates = []
        for i in range(len(pop)):
            for j in range(len(pop)):
                if i == j:
                    continue
                if self.dominates(pop[i], pop[j]):
                    pop[i].dominates.append(j)
                elif self.dominates(pop[j], pop[i]):
                    pop[i].dominated_count += 1
        fronts = []
        current = []
        for i, c in enumerate(pop):
            if c.dominated_count == 0:
                c.pareto_rank = 0
                current.append(i)
        fronts.append(current)
        front_no = 0
        while current:
            next_front = []
            for i in current:
                for j in pop[i].dominates:
                    pop[j].dominated_count -= 1
                    if pop[j].dominated_count == 0:
                        pop[j].pareto_rank = front_no + 1
                        next_front.append(j)
            front_no += 1
            current = next_front
            if current:
                fronts.append(current)
        return fronts

    def calculate_crowding_distance(self, pop, front):
        if len(front) <= 2:
            for i in front:
                pop[i].crowding_distance = float("inf")
            return
        for i in front:
            pop[i].crowding_distance = 0.0

        for obj in self.objectives:
            sorted_front = sorted(front, key=lambda idx: getattr(pop[idx], obj))
            pop[sorted_front[0]].crowding_distance = float("inf")
            pop[sorted_front[-1]].crowding_distance = float("inf")
            min_val = getattr(pop[sorted_front[0]], obj)
            max_val = getattr(pop[sorted_front[-1]], obj)
            if max_val == min_val:
                # Can't normalize
                continue
            for i in range(1, len(sorted_front) - 1):
                prev_val = getattr(pop[sorted_front[i - 1]], obj)
                next_val = getattr(pop[sorted_front[i + 1]], obj)
                pop[sorted_front[i]].crowding_distance += (next_val - prev_val) / (max_val - min_val)

    def get_pareto_front(self, pop):
        fronts = self.fast_non_dominated_sort(pop)
        if fronts:
            return [pop[i] for i in fronts[0]]
        return []


class FitnessEvaluator:
    def __init__(self, targets: Optional[Dict[str, float]] = None, pareto_mode: bool = True):
        self.targets = targets or {}
        self.pareto_mode = pareto_mode
        self.property_ranges = {
            "bandgap": {"min": 1.0, "max": 6.0},
            "sei_score": {"min": 0.0, "max": 1.0},
            "cei_score": {"min": 0.0, "max": 1.0},
            "ionic_conductivity": {"min": 1e-6, "max": 1e-2, "log_scale": True},
            "bulk_modulus": {"min": 10, "max": 200},
        }
        if self.pareto_mode:
            self.pareto_optimizer = ParetoOptimizer(
                ["sei_score", "cei_score", "ionic_conductivity", "bandgap"], [True, True, True, True]
            )

    def calculate_target_based_fitness(self, cand: ElectrolyteCandidate) -> float:
        if not self.targets:
            return 0.0
        total = 0.0
        count = 0
        for prop, target in self.targets.items():
            if hasattr(cand, prop):
                actual = getattr(cand, prop)
                if prop == "ionic_conductivity":
                    if actual > 0 and target > 0:
                        total += max(0, 1 - abs(np.log10(actual) - np.log10(target)) / 2)
                        count += 1
                    else:
                        total += 0
                        count += 1
                else:
                    pr = self.property_ranges.get(prop, {"min": 0, "max": 1})
                    dist = abs(actual - target) / (pr["max"] - pr["min"])
                    total += max(0, 1 - dist)
                    count += 1
        return total / count if count else 0.0


class GeneticAlgorithm:
    def __init__(
        self,
        population_size=100,
        max_generations=100,
        crossover_prob=0.8,
        mutation_prob=0.02,
        tournament_size=5,
        elite_size=10,
        stagnation_limit=15,
        targets=None,
        use_pareto=True,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.stagnation_limit = stagnation_limit
        self.targets = targets
        self.use_pareto = use_pareto

        self.model_interface = ExternalInterface()
        self.fitness_evaluator = FitnessEvaluator(targets=targets, pareto_mode=use_pareto)

        self.generation_stats = []
        self.pareto_history = []
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0

    def initialize(self):
        print(f"Generating initial population size {self.population_size}")
        pop = CrystaLLMInterface.generate_candidates(self.population_size)
        self.model_interface.predict_properties(pop)
        for c in pop:
            c.fitness = self.fitness_evaluator.calculate_target_based_fitness(c)
        return pop

    def tournament_select(self, population):
        contenders = random.sample(population, min(self.tournament_size, len(population)))
        return max(contenders, key=lambda c: c.fitness)

    def run(self):
        population = self.initialize()
        for gen in range(self.max_generations):
            if self.use_pareto:
                pareto_front = self.fitness_evaluator.pareto_optimizer.get_pareto_front(population)
                best_fit = np.mean([c.fitness for c in pareto_front]) if pareto_front else 0
                self.pareto_history.append(len(pareto_front))
            else:
                population.sort(key=lambda c: c.fitness, reverse=True)
                best_fit = population[0].fitness

            avg_fit = np.mean([c.fitness for c in population])
            self.generation_stats.append(
                {"generation": gen, "best_fitness": best_fit, "avg_fitness": avg_fit, "pareto_size": len(pareto_front) if pareto_front else 0}
            )
            print(f"Gen {gen}: Best={best_fit:.4f}, Avg={avg_fit:.4f}")

            if abs(best_fit - self.last_best_fitness) < 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.last_best_fitness = best_fit

            if self.stagnation_counter >= self.stagnation_limit:
                print("Stopping due to no improvement")
                break

            # Create next generation
            new_population = []

            if self.use_pareto:
                # Use Pareto selection
                selected = self.fitness_evaluator.pareto_optimizer.get_pareto_front(population)
            else:
                # Sort and select top for breeding
                selected = sorted(population, key=lambda c: c.fitness, reverse=True)

            # Preserve elites
            elites = sorted(selected, key=lambda c: c.fitness, reverse=True)[: self.elite_size]
            new_population.extend(elites)

            # Breed until population full
            while len(new_population) < self.population_size:
                parent1 = self.tournament_select(selected)
                parent2 = self.tournament_select(selected)
                # Crossover
                if random.random() < self.crossover_prob:
                    child1, child2 = crossover_cif_files(parent1, parent2, gen)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)
                # Mutation
                if random.random() < self.mutation_prob:
                    child1 = mutate_cif(child1, self.mutation_prob)
                if random.random() < self.mutation_prob:
                    child2 = mutate_cif(child2, self.mutation_prob)
                # Predict and score
                self.model_interface.predict_properties([child1, child2])
                child1.fitness = self.fitness_evaluator.calculate_target_based_fitness(child1)
                child2.fitness = self.fitness_evaluator.calculate_target_based_fitness(child2)
                if len(new_population) < self.population_size:
                    new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population

        # Return top solutions
        return self.get_top_candidates(population), self.generation_stats

    def get_top_candidates(self, population, n=10):
        if self.use_pareto:
            pf = self.fitness_evaluator.pareto_optimizer.get_pareto_front(population)
            if len(pf) <= n:
                return pf
            self.fitness_evaluator.pareto_optimizer.calculate_crowding_distance(pf, list(range(len(pf))))
            return sorted(pf, key=lambda c: c.crowding_distance, reverse=True)[:n]
        else:
            return sorted(population, key=lambda c: c.fitness, reverse=True)[:n]

    def print_top_candidates(self, candidates):
        print(f"Top {len(candidates)} candidates:")
        for i, c in enumerate(candidates, 1):
            print(f"{i}. CIF: {c.cif_file}, Composition: {c.composition}")
            print(f"   Bandgap: {c.bandgap:.3f} eV, SEI: {c.sei_score:.3f}, CEI: {c.cei_score:.3f}")
            print(f"   Ionic Cond: {c.ionic_conductivity:.2e}, Bulk Modulus: {c.bulk_modulus:.1f} GPa")
            print(f"   Fitness: {c.fitness:.4f}, Rank: {c.pareto_rank}, Crowding Distance: {c.crowding_distance:.4f}")

    def plot_statistics(self):
        if not self.generation_stats:
            print("No generation stats to plot")
            return
        gens = [st["generation"] for st in self.generation_stats]
        bests = [st["best_fitness"] for st in self.generation_stats]
        avgs = [st["avg_fitness"] for st in self.generation_stats]
        sizes = [st["pareto_size"] for st in self.generation_stats]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(gens, bests, label="Best Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Best Fitness over Generations")
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(gens, avgs, label="Average Fitness", color="orange")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Average Fitness over Generations")
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(gens, sizes, label="Pareto Front Size", color="green")
        plt.xlabel("Generation")
        plt.ylabel("Size")
        plt.title("Pareto Front Size over Generations")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    targets = {
        "ionic_conductivity": 1e-3,
        "sei_score": 0.8,
        "cei_score": 0.75,
        "bandgap": 3.0,
    }
    ga = GeneticAlgorithm(targets=targets, use_pareto=True)
    best, stats = ga.run()
    ga.print_top_candidates(best)
    ga.plot_statistics()
