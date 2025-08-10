import os
import pandas as pd

# Path to your OBELiX Excel file
obelix_xlsx_path = r'C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX\for_extraction\OBELiX_data.xlsx'

# Directory containing your CIF files
cif_folder = r'C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX\cifs'  # adjust if different path

# Load OBELiX dataset
df = pd.read_excel(obelix_xlsx_path)

print("Columns in dataset:", df.columns)

# Normalize the 'ID' column: strip whitespace, convert to str and lowercase for consistent matching
df['ID'] = df['ID'].astype(str).str.strip().str.lower()

# Convert ionic conductivity column to numeric, coercing errors to NaN
cond_col = 'Ionic conductivity (S cm-1)'  # adjust if different column name
df[cond_col] = pd.to_numeric(df[cond_col], errors='coerce')

# Drop rows with missing ionic conductivity
df = df.dropna(subset=[cond_col])

# Get list of CIF filenames (without '.cif' extension), normalized lower-case
cif_filenames = [
    os.path.splitext(f)[0].lower()
    for f in os.listdir(cif_folder)
    if f.endswith('.cif')
]

cif_set = set(cif_filenames)
obelix_ids = set(df['ID'].unique())

matched_ids = cif_set.intersection(obelix_ids)

cifs_unmatched = cif_set - obelix_ids
print(f"CIF files with no OBELiX match ({len(cifs_unmatched)}): {sorted(cifs_unmatched)}")

ids_unmatched = obelix_ids - cif_set
print(f"OBELiX IDs with no CIF file found ({len(ids_unmatched)}): {sorted(list(ids_unmatched)[:20])} ...")  # show first 20

# Then proceed with filtering and saving your matched data
df_filtered = df[df['ID'].isin(cif_filenames)]

print(f"Found {len(cif_filenames)} CIF files in '{cif_folder}' folder.")

# Filter OBELiX rows where ID is in the CIF filename list
df_filtered = df[df['ID'].isin(cif_filenames)]

print(f"Matched {len(df_filtered)} entries from OBELiX with CIF files.")

# Select ID and ionic conductivity columns
out = df_filtered[['ID', cond_col]]

# Save to id_prop.csv without header or index
out.to_csv('id_prop.csv', index=False, header=False)

print(f"Saved {len(out)} entries to id_prop.csv")
