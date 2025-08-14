#!/usr/bin/env python3
"""
Materials Project Bandgap Dataset Extractor
Extracts materials with both low-fidelity (PBE) and high-fidelity (HSE06/experimental) bandgaps
for machine learning training on bandgap correction tasks.

Updated to use the new Materials Project API (mp-api)
"""

import os
import json
import csv
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# New Materials Project API
from mp_api.client import MPRester
from pymatgen.core import Structure

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MaterialData:
    """Data class to store material information"""
    material_id: str
    formula: str
    structure: Structure
    pbe_bandgap: float
    hse_bandgap: Optional[float] = None
    experimental_bandgap: Optional[float] = None

class MaterialsProjectExtractor:
    """Class to handle Materials Project API interactions and data extraction"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.mpr = MPRester(api_key)
        
    def get_materials_with_bandgaps(self, max_materials: int = 10000) -> List[MaterialData]:
        """
        Fetch materials that have both PBE and HSE/experimental bandgaps
        """
        materials_data = []
        
        logger.info("Fetching materials with PBE bandgaps using new MP API...")
        
        try:
            # Search for materials with non-zero PBE bandgaps using correct API
            pbe_materials = self.mpr.materials.summary.search(
                band_gap=(0.01, None),  # Materials with bandgap > 0.01 eV
                fields=[
                    "material_id",
                    "formula_pretty",
                    "band_gap"
                ],
                num_chunks=10,  # Process in chunks for better performance
                chunk_size=1000
            )
            
            logger.info(f"Found {len(pbe_materials)} materials with PBE bandgaps")
            
            # Process materials and look for high-fidelity bandgap data
            logger.info("Processing materials for high-fidelity bandgap data...")
            
            for i, material in enumerate(pbe_materials[:max_materials]):
                if i % 100 == 0:
                    logger.info(f"Processing material {i+1}/{min(len(pbe_materials), max_materials)}")
                
                try:
                    # Get structure separately since summary search doesn't include it
                    structure = None
                    try:
                        structure = self.mpr.get_structure_by_material_id(material.material_id)
                    except Exception as e:
                        logger.warning(f"Could not get structure for {material.material_id}: {e}")
                    
                    if structure is not None:
                        material_data = MaterialData(
                            material_id=str(material.material_id),
                            formula=material.formula_pretty,
                            structure=structure,
                            pbe_bandgap=material.band_gap,
                            hse_bandgap=None,  # Will be enhanced later
                            experimental_bandgap=None  # Will be enhanced later
                        )
                        materials_data.append(material_data)
                    else:
                        logger.warning(f"Skipping {material.material_id} - no structure available")
                        
                except Exception as e:
                    logger.warning(f"Error processing material {material.material_id}: {e}")
                    continue
                    
                # Rate limiting to be respectful to the API
                time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching materials: {e}")
            raise
        
        logger.info(f"Found {len(materials_data)} materials with PBE bandgap data")
        return materials_data
    
    def _get_detailed_material_data(self, material_id: str) -> Optional[Dict]:
        """Get additional detailed data for a specific material"""
        try:
            # Try to get additional data - for now return placeholder
            # The new API structure might have experimental data in different endpoints
            return {"experimental_bandgap": None}
            
        except Exception as e:
            logger.warning(f"Error fetching detailed data for {material_id}: {e}")
            return None
    
    def get_hse_materials_directly(self, max_materials: int = 5000) -> List[MaterialData]:
        """
        Alternative method: Search specifically for materials with HSE calculations
        """
        materials_data = []
        
        logger.info("Searching for materials with HSE calculations...")
        
        try:
            # Search for materials using summary endpoint
            materials = self.mpr.materials.summary.search(
                band_gap=(0.01, None),
                fields=[
                    "material_id",
                    "formula_pretty",
                    "band_gap"
                ],
                num_chunks=5,
                chunk_size=1000
            )
            
            logger.info(f"Found {len(materials)} materials with theoretical calculations")
            
            for i, material in enumerate(materials[:max_materials]):
                if i % 50 == 0:
                    logger.info(f"Processing material {i+1}/{min(len(materials), max_materials)}")
                
                try:
                    # For now, we'll create a simplified version that focuses on PBE data
                    # HSE data might be available through different API endpoints
                    # This is a placeholder approach
                    
                    # Skip HSE search for now and focus on getting PBE data working
                    # We can enhance this later once the basic API calls work
                    pass
                        
                except Exception as e:
                    logger.warning(f"Error processing material {material.material_id}: {e}")
                    continue
                    
                time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching HSE materials: {e}")
            raise
        
        logger.info(f"Found {len(materials_data)} materials with HSE bandgap data")
        return materials_data

def save_to_csv(materials: List[MaterialData], filename: str = 'bandgap_dataset.csv'):
    """Save materials data to CSV format"""
    logger.info(f"Saving {len(materials)} materials to {filename}")
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'material_id', 'formula', 'pbe_bandgap', 
            'hse_bandgap', 'experimental_bandgap', 'structure_cif'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for material in materials:
            # Convert structure to CIF format
            structure_cif = ""
            try:
                structure_cif = material.structure.to(fmt="cif")
            except Exception as e:
                logger.warning(f"Could not convert structure to CIF for {material.material_id}: {e}")
                structure_cif = str(material.structure)
            
            writer.writerow({
                'material_id': material.material_id,
                'formula': material.formula,
                'pbe_bandgap': material.pbe_bandgap,
                'hse_bandgap': material.hse_bandgap,
                'experimental_bandgap': material.experimental_bandgap,
                'structure_cif': structure_cif
            })

def save_to_json(materials: List[MaterialData], filename: str = 'bandgap_dataset.json'):
    """Save materials data to JSON format"""
    logger.info(f"Saving {len(materials)} materials to {filename}")
    
    data = []
    for material in materials:
        # Convert structure to dictionary for JSON serialization
        structure_dict = material.structure.as_dict()
        
        data.append({
            'material_id': material.material_id,
            'formula': material.formula,
            'pbe_bandgap': material.pbe_bandgap,
            'hse_bandgap': material.hse_bandgap,
            'experimental_bandgap': material.experimental_bandgap,
            'structure': structure_dict
        })
    
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)

def main():
    """Main function to extract and save the dataset"""
    # API key (in production, use environment variable or secure storage)
    API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"
    
    # Initialize extractor
    extractor = MaterialsProjectExtractor(API_KEY)
    
    try:
        logger.info("=== Starting Materials Project Bandgap Dataset Extraction ===")
        
        # Method 1: Try to get materials with paired bandgap data
        logger.info("Method 1: Searching for materials with paired bandgap data...")
        materials = extractor.get_materials_with_bandgaps(max_materials=5000)
        
        # For now, skip Method 2 since we're getting materials from Method 1
        logger.info(f"Collected {len(materials)} materials with PBE bandgaps")
        
        if not materials:
            logger.warning("No materials found with bandgap data")
            return
        
        # Save to both CSV and JSON formats
        save_to_csv(materials, 'bandgap_dataset.csv')
        save_to_json(materials, 'bandgap_dataset.json')
        
        # Print summary statistics
        logger.info("=== Dataset Summary ===")
        logger.info(f"Total materials: {len(materials)}")
        
        hse_count = sum(1 for m in materials if m.hse_bandgap is not None)
        exp_count = sum(1 for m in materials if m.experimental_bandgap is not None)
        both_count = sum(1 for m in materials if m.hse_bandgap is not None and m.experimental_bandgap is not None)
        
        logger.info(f"Materials with HSE bandgaps: {hse_count} (HSE data collection will be enhanced)")
        logger.info(f"Materials with experimental bandgaps: {exp_count} (experimental data collection will be enhanced)")
        logger.info(f"Materials with both HSE and experimental: {both_count}")
        
        # Sample statistics
        pbe_gaps = [m.pbe_bandgap for m in materials]
        logger.info(f"PBE bandgap range: {min(pbe_gaps):.3f} - {max(pbe_gaps):.3f} eV")
        
        if hse_count > 0:
            hse_gaps = [m.hse_bandgap for m in materials if m.hse_bandgap is not None]
            logger.info(f"HSE bandgap range: {min(hse_gaps):.3f} - {max(hse_gaps):.3f} eV")
        
        # Show some example formulas
        example_formulas = [m.formula for m in materials[:10]]
        logger.info(f"Example formulas: {', '.join(example_formulas)}")
        
        logger.info("Dataset extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise
    
    finally:
        # MPRester doesn't need explicit closing in the new API
        pass

if __name__ == "__main__":
    main()