from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
import matplotlib.pyplot as plt
from collections import defaultdict

# Load structure
structure = Structure.from_file("CIFs/test_CIF.cif")

# Get fractional coordinates and elements
coords = structure.frac_coords
elements = [site.specie.symbol for site in structure.sites]

# Initialize neighbor finder
cnn = CrystalNN()

# Define color map for elements
element_colors = {
    'H': 'white',
    'Li': 'green',
    'Na': 'blue',
    'K': 'purple',
    'Mg': 'lightgreen',
    'Ca': 'darkgreen',
    'Al': 'gray',
    'Si': 'orange',
    'P': 'violet',
    'S': 'yellow',
    'Cl': 'cyan',
    'O': 'red',
    'F': 'lightblue',
    'Br': 'brown',
    'I': 'darkviolet',
    'Zn': 'lightgray',
    'Cu': 'teal',
    'Fe': 'darkred',
    'Mn': 'indigo',
    'Co': 'navy',
    'Ni': 'slateblue',
    'Ti': 'silver',
    'Zr': 'lightsteelblue',
    'Sn': 'gold',
    'Pb': 'black',
    'Gd': 'darkorange',
    'La': 'mediumorchid',
    'Ce': 'mediumslateblue',
    'Y': 'mediumturquoise',
    'Nb': 'steelblue',
    'Ta': 'midnightblue',
    'W': 'darkslategray',
    'Mo': 'cadetblue',
    'Sb': 'peru',
    'Bi': 'darkgoldenrod',
    'Ge': 'chocolate',
    'Te': 'darkkhaki',
    'Se': 'olive',
    'Ag': 'lightgray',
    'Au': 'goldenrod',
    'Pt': 'plum',
    'Pd': 'orchid',
    'Rh': 'mediumblue',
    'Ru': 'royalblue',
    'Re': 'darkcyan',
    'Os': 'darkblue',
    'Ir': 'darkmagenta',
}

# Group atoms by element
element_coords = defaultdict(list)
for i, (x, y, z) in enumerate(coords):
    element_coords[elements[i]].append((x, y))

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Plot atoms by group
for element, positions in element_coords.items():
    xs, ys = zip(*positions)
    color = element_colors.get(element, 'black')  # default to black if not in map
    ax.plot(xs, ys, 'o', color=color, label=element)
    for x, y in positions:
        ax.text(x, y, element, fontsize=9, ha='center', va='center', color=color)

# Draw bonds
for i, site in enumerate(structure):
    neighbors = cnn.get_nn_info(structure, i)
    for neighbor in neighbors:
        j = neighbor['site_index']
        xi, yi = coords[i][0], coords[i][1]
        xj, yj = coords[j][0], coords[j][1]
        ax.plot([xi, xj], [yi, yj], 'k-', linewidth=0.8)

# Final plot settings
ax.set_xlabel('Fractional X')
ax.set_ylabel('Fractional Y')
ax.set_title('2D Projection of Atomic Positions with Bonds')
ax.grid(True)
ax.legend()

plt.show()