import random
import numpy as np
from pyxtal import pyxtal
from pymatgen.core import Structure, Element

def generate_diverse_garnet_population(pop_size=100, max_attempts=10):
    population = []
    space_groups = [230, 142]  # Ia-3d (cubic garnet), I-41/acd (tetragonal)
    a_elements = ['Li', 'Na', 'K']
    b_elements = ['La', 'Nd', 'Pr']
    c_elements = ['Zr', 'Ta', 'Nb', 'Hf']
    dopants = ['Al', 'Ga', 'Mg', 'Ca']
    o_sub = ['O', 'F', 'S']

    # Wyckoff site multiplicities for garnet (Ia-3d)
    # A: 24d/96h (Li), B: 16a (La), C: 24d/48g (Zr), O: 96h
    wyckoff_multiplicities = {
        230: {'A': [24, 96], 'B': [16], 'C': [24, 48], 'O': [96]},
        142: {'A': [16, 32], 'B': [16], 'C': [16, 32], 'O': [32, 64]}
    }

    for _ in range(pop_size):
        for attempt in range(max_attempts):
            # Select space group
            sg = random.choice(space_groups)
            multis = wyckoff_multiplicities[sg]

            # Generate garnet-like composition (A3B2C3O12 base, scaled)
            scale = random.choice([1, 2])  # 1x or 2x unit cell for diversity
            num_a = random.choice(multis['A']) * scale  # e.g., 24 or 96 for Li
            num_b = random.choice(multis['B']) * scale  # e.g., 16 for La
            num_c = random.choice(multis['C']) * scale  # e.g., 24 or 48 for Zr
            num_o = random.choice(multis['O']) * scale  # e.g., 96 for O

            # Random elements
            elem_a = random.choice(a_elements)
            elem_b = random.choice(b_elements)
            elem_c = random.choice(c_elements)
            elem_o = random.choices(o_sub, weights=[0.9, 0.05, 0.05], k=num_o)

            # Add dopants: Replace 0-20% of A/B sites
            dopant = random.choice(dopants)
            num_dopant_a = int(num_a * random.uniform(0, 0.2))
            num_dopant_b = int(num_b * random.uniform(0, 0.2))

            # Adjust species and counts
            species = [elem_a] * (num_a - num_dopant_a) + [dopant] * num_dopant_a + \
                      [elem_b] * (num_b - num_dopant_b) + [dopant] * num_dopant_b + \
                      [elem_c] * num_c + elem_o
            num_ions = [num_a - num_dopant_a, num_dopant_a,
                        num_b - num_dopant_b, num_dopant_b,
                        num_c, len(elem_o)]

            # Charge neutrality check
            valence_map = {
                'Li': 1, 'Na': 1, 'K': 1,  # A-site
                'La': 3, 'Nd': 3, 'Pr': 3,  # B-site
                'Zr': 4, 'Ta': 5, 'Nb': 5, 'Hf': 4,  # C-site
                'Al': 3, 'Ga': 3, 'Mg': 2, 'Ca': 2,  # Dopants
                'O': -2, 'F': -1, 'S': -2  # Anions
            }
            total_charge = sum([valence_map[s] * n for s, n in zip(
                [elem_a, dopant, elem_b, dopant, elem_c] + [elem_o[0]],  # Approximate O as single species
                num_ions
            ) if s in valence_map])
            if abs(total_charge) > 1e-2:  # Skip if not neutral
                continue

            # Generate structure
            try:
                crystal = pyxtal()
                crystal.from_random(3, sg, species, num_ions, factor=1.2)  # Increased factor for stability
                if crystal.valid:
                    pmg_struct = crystal.to_pymatgen()
                    if pmg_struct.is_valid():  # Additional pymatgen check
                        population.append(pmg_struct)
                        break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for SG {sg}: {e}")
                continue

    return population

# Usage
if __name__ == "__main__":
    initial_pop = generate_diverse_garnet_population(10)
    print(f"Generated {len(initial_pop)} valid diverse structures.")