import random
from pyxtal import pyxtal
from pyxtal.symmetry import Group  # for Wyckoff info
import numpy as np

# Charge dictionary (same as before)
charge_dict = {
    'Li': +1, 'Na': +1, 'K': +1,
    'Mg': +2, 'Ca': +2, 'Sr': +2,
    'Ti': +4, 'Zr': +4, 'Hf': +4,
    'V': +5, 'Cr': +3, 'Mn': +2, 'Fe': +3, 'Co': +2, 'Ni': +2, 'Cu': +2, 'Zn': +2,
    'Al': +3, 'B': +3, 'Si': +4, 'Sn': +4, 'Sb': +3, 'P': +5,
    'O': -2, 'S': -2, 'Se': -2, 'Te': -2,
    'F': -1, 'Cl': -1, 'Br': -1, 'I': -1
}

all_species = [
    'Li', 'Na', 'K', 
    'Mg', 'Ca', 'Sr',
    'Ti', 'Zr', 'Hf',
    'Al', 'B', 'Si', 'Sn', 'Sb', 'P',
    'O', 'S', 'Se', 'Te',
    'F', 'Cl', 'Br', 'I'
]

common_space_groups = [
    225, 227, 230,
    142, 141, 123,
    62, 58, 55,
    166, 167,
    194, 176, 189,
    216, 214, 149, 70
]

rng = np.random.default_rng()

def check_charge_neutrality(composition):
    total_charge = 0
    for species, count in composition.items():
        if species in charge_dict:
            total_charge += charge_dict[species] * count
        else:
            return False
    return total_charge == 0

def get_wyckoff_multiplicities(space_group_number):
    """
    Return a sorted list of unique Wyckoff multiplicities for the given space group
    """
    sg = Group(space_group_number)
    mults = set(wp.multiplicity for wp in sg.Wyckoff_positions)
    return sorted(list(mults))

def generate_wyckoff_compatible_composition(species_pool, wyckoff_mults, min_species=2, max_species=5, max_multiplicity=3, max_tries=200):
    """
    Generate a random composition compatible with given Wyckoff multiplicities:
    - Randomly pick subset of elements
    - For each species, pick a multiplicity from wyckoff_mults multiplied by a small integer multiplier
    - Ensure charge neutrality
    """
    for _ in range(max_tries):
        num_species = random.randint(min_species, max_species)
        chosen_species = random.sample(species_pool, num_species)
        
        counts = []
        for _ in chosen_species:
            base_mult = random.choice(wyckoff_mults)
            multiplier = random.randint(1, max_multiplicity)
            counts.append(base_mult * multiplier)
        
        composition = dict(zip(chosen_species, counts))
        if check_charge_neutrality(composition):
            return composition
    return None

def generate_valid_crystal():
    for attempt in range(100):
        space_group = random.choice(common_space_groups)
        wyckoff_mults = get_wyckoff_multiplicities(space_group)
        
        composition = generate_wyckoff_compatible_composition(all_species, wyckoff_mults)
        if composition is None:
            print("⚠ Could not find charge neutral wyckoff-compatible comp, retrying...")
            continue
        
        species = list(composition.keys())
        numIons = list(composition.values())
        
        xtal = pyxtal()
        try:
            xtal.from_random(3, space_group, species, numIons, random_state=rng)
            if xtal.valid:
                print("\n✅ Successfully generated crystal")
                print(f"Composition: {composition}")
                print(f"Space group: {space_group}")
                print(xtal)
                cif_filename = "wyckoff_aware_generated_solid_electrolyte.cif"
                xtal.to_file(cif_filename)
                print(f"CIF saved as: {cif_filename}")
                return xtal
            else:
                print("❌ Invalid crystal after generation, retrying...")
        except Exception as e:
            print(f"⚠ Generation failed: {e}")
    print("❌ Could not generate valid crystal after many tries.")
    return None

if __name__ == "__main__":
    generate_valid_crystal()
