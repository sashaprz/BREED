import os
import pandas as pd
import time
from mp_api.client import MPRester
from pymatgen.core import Composition
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"  # Replace with your actual API key
OBELIX_XLSX_PATH = r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\CIF_OBELiX\for_extraction\OBELiX_data.xlsx"
OUTPUT_FOLDER = "materials_project_cifs"
DELAY_BETWEEN_REQUESTS = 0.2  # Reduced delay since we're making more requests per composition

def normalize_composition(comp_str):
    """Normalize composition string to standard format"""
    try:
        comp = Composition(comp_str)
        return comp.reduced_formula
    except:
        logger.warning(f"Could not parse composition: {comp_str}")
        return None

def search_materials_project(mpr, composition, space_group=None):
    """Search Materials Project for materials matching composition with multiple strategies"""
    try:
        normalized_comp = normalize_composition(composition)
        if not normalized_comp:
            return []
        
        results = []
        
        # Strategy 1: Exact formula match
        try:
            exact_results = mpr.materials.summary.search(
                formula=normalized_comp,
                fields=["material_id", "formula_pretty", "symmetry", "structure"]
            )
            results.extend(exact_results)
        except:
            pass
        
        # Strategy 2: If no exact matches, try chemical system search
        if not results:
            try:
                comp_obj = Composition(composition)
                elements = [str(el) for el in comp_obj.elements]
                chemsys = "-".join(sorted(elements))
                
                chemsys_results = mpr.materials.summary.search(
                    chemsys=chemsys,
                    fields=["material_id", "formula_pretty", "symmetry", "structure"]
                )
                
                # Filter to only include compositions with similar element ratios
                filtered_chemsys = []
                for result in chemsys_results[:20]:  # Limit to first 20 to avoid too many
                    try:
                        result_comp = Composition(result.formula_pretty)
                        # Check if it has the same elements
                        if set(result_comp.elements) == set(comp_obj.elements):
                            filtered_chemsys.append(result)
                    except:
                        continue
                
                results.extend(filtered_chemsys[:5])  # Take top 5 matches
            except:
                pass
        
        # Strategy 3: If still no results, try element-based search with fewer constraints
        if not results:
            try:
                comp_obj = Composition(composition)
                main_elements = [str(el) for el in comp_obj.elements if comp_obj[el] > 0.1]
                
                if len(main_elements) <= 4:  # Only for simpler systems
                    element_results = mpr.materials.summary.search(
                        elements=main_elements,
                        fields=["material_id", "formula_pretty", "symmetry", "structure"]
                    )
                    
                    # Take a few representative structures
                    results.extend(element_results[:3])
            except:
                pass
        
        # Filter by space group if provided
        if space_group is not None and results:
            filtered_results = []
            for result in results:
                if hasattr(result, 'symmetry') and result.symmetry:
                    if result.symmetry.number == int(space_group):
                        filtered_results.append(result)
            # If space group filtering gives us nothing, return some unfiltered results
            if filtered_results:
                return filtered_results
            else:
                return results[:2]  # Return a couple even without space group match
        
        return results
    
    except Exception as e:
        logger.error(f"Error searching for {composition}: {str(e)}")
        return []

def save_cif_file(structure, material_id, output_folder, composition_info=""):
    """Save structure as CIF file with better error handling"""
    try:
        cif_path = os.path.join(output_folder, f"{material_id}.cif")
        
        # Check if file already exists
        if os.path.exists(cif_path):
            print(f"    Warning: {material_id}.cif already exists, skipping...")
            return cif_path
        
        structure.to(filename=cif_path, fmt="cif")
        return cif_path
    except Exception as e:
        print(f"    Error saving CIF for {material_id}: {str(e)}")
        return None

def main():
    # Load the OBELiX dataset
    try:
        df = pd.read_excel(OBELIX_XLSX_PATH)
        print(f"Loaded dataset with {len(df)} entries")
    except Exception as e:
        print(f"Error loading Excel file: {str(e)}")
        return
    
    # Print column names to help debug
    print(f"Available columns: {list(df.columns)}")
    
    # Try to identify the correct column names (case insensitive)
    composition_col = None
    space_group_col = None
    cif_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'composition' in col_lower or 'formula' in col_lower:
            composition_col = col
        elif 'space' in col_lower and 'group' in col_lower:
            space_group_col = col
        elif 'cif' in col_lower:
            cif_col = col
    
    if not composition_col:
        print("Error: Could not find composition column. Please check column names.")
        return
    
    print(f"Using columns - Composition: {composition_col}, Space Group: {space_group_col}, CIF: {cif_col}")
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Initialize Materials Project client
    try:
        mpr = MPRester(API_KEY)
        print("Successfully connected to Materials Project API")
    except Exception as e:
        print(f"Error connecting to Materials Project: {str(e)}")
        return
    
    # Filter for entries that need CIF extraction
    if cif_col:
        mask = df[cif_col].isna() | (df[cif_col].astype(str).str.contains("No Match", case=False, na=False))
        entries_to_process = df[mask]
    else:
        # If no CIF column found, process all entries
        entries_to_process = df
        logger.warning("No CIF column found, processing all entries")
    
    logger.info(f"Found {len(entries_to_process)} entries to process")
    
    found_count = 0
    processed_count = 0
    
    print(f"\nProcessing {len(entries_to_process)} entries with missing CIFs...")
    
    for idx, row in entries_to_process.iterrows():
        composition = row.get(composition_col)
        space_group = row.get(space_group_col) if space_group_col else None
        
        # Skip if missing composition
        if pd.isna(composition) or str(composition).strip() == '':
            continue
        
        processed_count += 1
        
        # Show progress every 20 entries or for first few
        if processed_count <= 5 or processed_count % 20 == 0:
            print(f"Processing {processed_count}/{len(entries_to_process)}: {composition}")
        
        # Search Materials Project
        results = search_materials_project(mpr, composition, space_group)
        
        if results:
            # Save CIF file for the FIRST matching material only
            try:
                result = results[0]  # Take only the first result
                material_id = result.material_id
                structure = result.structure
                
                if processed_count <= 15:  # Debug info for first 15
                    print(f"    Found {len(results)} matches, using {material_id} ({result.formula_pretty})")
                
                if structure:
                    cif_path = save_cif_file(structure, material_id, OUTPUT_FOLDER, composition)
                    if cif_path:
                        found_count += 1
                        if processed_count <= 15:  # Show details for first 15
                            print(f"  ✓ Saved CIF for {material_id} ({result.formula_pretty})")
                        elif found_count % 10 == 0:  # Show every 10th success
                            print(f"  ✓ Found CIF #{found_count}: {material_id} for {composition}")
                    else:
                        if processed_count <= 15:
                            print(f"  ✗ Failed to save CIF for {material_id}")
                else:
                    if processed_count <= 15:
                        print(f"  ✗ No structure data for {material_id}")
                    
            except Exception as e:
                print(f"  ✗ Error processing result for {composition}: {str(e)}")
        else:
            if processed_count <= 15:
                print(f"  - No matches found for {composition}")
        
        # Add delay to respect rate limits
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print(f"\n=== SUMMARY ===")
    print(f"Processed {processed_count} entries")
    print(f"Found and saved {found_count} CIF files to folder '{OUTPUT_FOLDER}'")
    
    # List some of the saved files
    saved_files = os.listdir(OUTPUT_FOLDER)
    if saved_files:
        print(f"Sample saved files: {saved_files[:5]}")
    
    print(f"\nSUCCESS: Found and saved {found_count} CIF files!")

if __name__ == "__main__":
    main()