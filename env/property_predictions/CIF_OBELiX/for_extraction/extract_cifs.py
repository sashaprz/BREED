import zipfile

cifs_zip = "all_cifs.zip"
extract_dir = r'C:\Users\YourName\Documents\project\fr8\RL-electrolyte-design\CIFs\CIF_OBELiX\cifs'


with zipfile.ZipFile(cifs_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Extraction complete")