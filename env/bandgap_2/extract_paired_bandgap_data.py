from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
import csv, json, os

# 1. Load the 3D DFT dataset from JARVIS
print("Downloading JARVIS-DFT dataset...")
dft_data = data('dft_3d')

# 2. Prepare output lists
paired_entries = []

# 3. Output folder for CIFs
os.makedirs("structures_cif", exist_ok=True)

# 4. Loop through dataset and filter
for entry in dft_data:
    pbe_gap = entry.get("optb88vdw_bandgap", None)   # PBE-like functional
    # Try common high-fidelity keys — adjust as needed depending on dataset availability
    hse_gap = entry.get("hse_gap", None)
    mbj_gap = entry.get("mbj_bandgap", None)  # meta-GGA TBmBJ
    gw_gap = entry.get("gw_bandgap", None)

    # Choose the first available high-fidelity gap
    high_gap = hse_gap or mbj_gap or gw_gap

    if pbe_gap is not None and high_gap is not None:
        atoms = Atoms.from_dict(entry["atoms"])
        formula = atoms.composition.reduced_formula
        cif_str = atoms.write_cif()

        # Save CIF file
        cif_filename = f"{entry['jid']}.cif"
        cif_path = os.path.join("structures_cif", cif_filename)
        with open(cif_path, "w") as f:
            f.write(cif_str)

        paired_entries.append({
            "material_id": entry["jid"],
            "formula": formula,
            "cif_path": cif_path,
            "pbe_bandgap": pbe_gap,
            "high_bandgap": high_gap
        })

print(f"Found {len(paired_entries)} materials with both PBE and high-fidelity bandgaps.")

# 5. Save CSV
csv_file = "paired_bandgaps.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["material_id", "formula", "cif_path", "pbe_bandgap", "high_bandgap"])
    writer.writeheader()
    writer.writerows(paired_entries)

# 6. Save JSON
json_file = "paired_bandgaps.json"
with open(json_file, "w") as f:
    json.dump(paired_entries, f, indent=2)

print(f"Saved CSV to {csv_file} and JSON to {json_file}")
print("CIF structures saved in 'structures_cif' folder.")
