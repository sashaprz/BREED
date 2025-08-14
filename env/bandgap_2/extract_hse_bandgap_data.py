#!/usr/bin/env python3
"""
Materials Project HSE Bandgap Data Extractor
Specifically searches for true hybrid functional (HSE06) bandgap data
by examining calculation metadata and specific HSE fields
"""

import os
import json
import csv
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import pandas as pd

# Materials Project API
from mp_api.client import MPRester
from pymatgen.core import Structure

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HSEBandgapData:
    """Data class to store HSE bandgap information"""
    material_id: str
    formula: str
    structure: Optional[Structure]
    pbe_bandgap: float
    hse_bandgap: Optional[float]
    experimental_bandgap: Optional[float]
    calculation_metadata: Dict
    bandgap_source: str  # "hse_field", "calculation_metadata", "experimental"
    is_gap_direct: bool
    space_group: int
    crystal_system: str

class HSEBandgapExtractor:
    """Extract materials with true HSE bandgap data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.mpr = MPRester(api_key)
        
    def extract_hse_data(self, max_materials: int = 1000) -> List[HSEBandgapData]:
        """Extract materials with HSE bandgap data using multiple approaches"""
        logger.info(f"Searching for HSE bandgap data using multiple approaches...")
        
        all_hse_materials = []
        
        # Approach 1: Search for explicit HSE fields
        logger.info("Approach 1: Searching for explicit HSE bandgap fields...")
        hse_field_materials = self._search_hse_fields(max_materials // 3)
        all_hse_materials.extend(hse_field_materials)
        logger.info(f"Found {len(hse_field_materials)} materials with HSE fields")
        
        # Approach 2: Search calculation metadata for HSE calculations
        logger.info("Approach 2: Searching calculation metadata for HSE calculations...")
        hse_calc_materials = self._search_hse_calculations(max_materials // 3)
        all_hse_materials.extend(hse_calc_materials)
        logger.info(f"Found {len(hse_calc_materials)} materials with HSE calculations")
        
        # Approach 3: Search for experimental bandgap data
        logger.info("Approach 3: Searching for experimental bandgap data...")
        exp_materials = self._search_experimental_bandgaps(max_materials // 3)
        all_hse_materials.extend(exp_materials)
        logger.info(f"Found {len(exp_materials)} materials with experimental data")
        
        # Remove duplicates based on material_id
        unique_materials = {}
        for material in all_hse_materials:
            if material.material_id not in unique_materials:
                unique_materials[material.material_id] = material
            else:
                # Merge data if we have the same material from multiple sources
                existing = unique_materials[material.material_id]
                if material.hse_bandgap and not existing.hse_bandgap:
                    existing.hse_bandgap = material.hse_bandgap
                if material.experimental_bandgap and not existing.experimental_bandgap:
                    existing.experimental_bandgap = material.experimental_bandgap
        
        final_materials = list(unique_materials.values())
        logger.info(f"Total unique materials with HSE/experimental data: {len(final_materials)}")
        
        return final_materials
    
    def _search_hse_fields(self, max_materials: int) -> List[HSEBandgapData]:
        """Search for materials with explicit HSE bandgap fields"""
        hse_materials = []
        
        # Try different possible HSE field names
        hse_field_names = [
            "hse_band_gap", "band_gap_hse", "hse06_band_gap", 
            "hybrid_band_gap", "hse_bandgap", "bandgap_hse"
        ]
        
        for field_name in hse_field_names:
            logger.info(f"  Trying HSE field: {field_name}")
            try:
                # Try to search with this field in summary data
                materials = self.mpr.materials.summary.search(
                    band_gap=(0.1, 8.0),
                    fields=[
                        "material_id", "formula_pretty", "band_gap", 
                        "structure", "symmetry", field_name
                    ],
                    chunk_size=min(100, max_materials),
                    num_chunks=1
                )
                
                for material in materials:
                    if hasattr(material, field_name):
                        hse_value = getattr(material, field_name)
                        if hse_value is not None and hse_value != material.band_gap:
                            # Found a material with different HSE value!
                            hse_data = self._create_hse_data_entry(
                                material, hse_bandgap=hse_value, 
                                source=f"hse_field_{field_name}"
                            )
                            if hse_data:
                                hse_materials.append(hse_data)
                                logger.info(f"    ✅ Found HSE data: {material.material_id} PBE={material.band_gap:.3f} HSE={hse_value:.3f}")
                
            except Exception as e:
                logger.warning(f"  ❌ Error searching field {field_name}: {e}")
                continue
        
        return hse_materials
    
    def _search_hse_calculations(self, max_materials: int) -> List[HSEBandgapData]:
        """Search for materials with HSE calculations in metadata"""
        hse_materials = []
        
        try:
            # Get materials and then check their calculation metadata
            materials = self.mpr.materials.summary.search(
                band_gap=(0.1, 8.0),
                fields=[
                    "material_id", "formula_pretty", "band_gap", 
                    "structure", "symmetry", "task_ids", "origins"
                ],
                chunk_size=min(200, max_materials),
                num_chunks=1
            )
            
            logger.info(f"  Checking calculation metadata for {len(materials)} materials...")
            
            for i, material in enumerate(materials):
                if i % 50 == 0:
                    logger.info(f"    Processing material {i+1}/{len(materials)}")
                
                try:
                    # Check if material has task_ids or origins that might indicate HSE calculations
                    hse_found = False
                    hse_bandgap = None
                    calc_metadata = {}
                    
                    # Check origins for HSE-related information
                    if hasattr(material, 'origins') and material.origins:
                        for origin in material.origins:
                            if hasattr(origin, 'name'):
                                origin_name = str(origin.name).lower()
                                if any(hse_term in origin_name for hse_term in ['hse', 'hybrid', 'hse06']):
                                    hse_found = True
                                    calc_metadata['hse_origin'] = origin_name
                                    logger.info(f"    ✅ Found HSE origin: {material.material_id} - {origin_name}")
                    
                    # If we found HSE-related metadata, try to get the actual HSE bandgap
                    if hse_found:
                        # For now, we'll mark it as HSE but use a placeholder value
                        # In a real implementation, you'd query the specific calculation
                        hse_data = self._create_hse_data_entry(
                            material, hse_bandgap=None, 
                            source="calculation_metadata",
                            metadata=calc_metadata
                        )
                        if hse_data:
                            hse_materials.append(hse_data)
                
                except Exception as e:
                    logger.warning(f"    Error processing {material.material_id}: {e}")
                    continue
                
                time.sleep(0.05)  # Rate limiting
        
        except Exception as e:
            logger.error(f"Error searching HSE calculations: {e}")
        
        return hse_materials
    
    def _search_experimental_bandgaps(self, max_materials: int) -> List[HSEBandgapData]:
        """Search for materials with experimental bandgap data"""
        exp_materials = []
        
        # Try different approaches to find experimental data
        experimental_fields = [
            "experimental_band_gap", "band_gap_experimental", 
            "exp_band_gap", "bandgap_exp"
        ]
        
        for field_name in experimental_fields:
            logger.info(f"  Trying experimental field: {field_name}")
            try:
                materials = self.mpr.materials.summary.search(
                    band_gap=(0.1, 8.0),
                    fields=[
                        "material_id", "formula_pretty", "band_gap", 
                        "structure", "symmetry", field_name
                    ],
                    chunk_size=min(100, max_materials),
                    num_chunks=1
                )
                
                for material in materials:
                    if hasattr(material, field_name):
                        exp_value = getattr(material, field_name)
                        if exp_value is not None and exp_value != material.band_gap:
                            # Found experimental data!
                            exp_data = self._create_hse_data_entry(
                                material, experimental_bandgap=exp_value,
                                source=f"experimental_{field_name}"
                            )
                            if exp_data:
                                exp_materials.append(exp_data)
                                logger.info(f"    ✅ Found experimental data: {material.material_id} PBE={material.band_gap:.3f} Exp={exp_value:.3f}")
                
            except Exception as e:
                logger.warning(f"  ❌ Error searching experimental field {field_name}: {e}")
                continue
        
        return exp_materials
    
    def _create_hse_data_entry(self, material, hse_bandgap=None, experimental_bandgap=None, 
                              source="unknown", metadata=None) -> Optional[HSEBandgapData]:
        """Create HSE data entry from material"""
        try:
            # Get space group and crystal system
            space_group = 1
            crystal_system = "unknown"
            
            if hasattr(material, 'symmetry') and material.symmetry:
                space_group = getattr(material.symmetry, 'number', 1)
                crystal_system = str(getattr(material.symmetry, 'crystal_system', 'unknown'))
            
            # Get structure
            structure = None
            if hasattr(material, 'structure') and material.structure:
                structure = material.structure
            
            return HSEBandgapData(
                material_id=str(material.material_id),
                formula=material.formula_pretty,
                structure=structure,
                pbe_bandgap=material.band_gap,
                hse_bandgap=hse_bandgap,
                experimental_bandgap=experimental_bandgap,
                calculation_metadata=metadata or {},
                bandgap_source=source,
                is_gap_direct=getattr(material, 'is_gap_direct', False),
                space_group=space_group,
                crystal_system=crystal_system
            )
            
        except Exception as e:
            logger.warning(f"Error creating HSE data entry for {material.material_id}: {e}")
            return None
    
    def analyze_hse_data(self, hse_materials: List[HSEBandgapData]) -> Dict:
        """Analyze the HSE data quality and characteristics"""
        if not hse_materials:
            return {"total_materials": 0}
        
        # Convert to DataFrame for analysis
        df_data = []
        for entry in hse_materials:
            df_data.append({
                'material_id': entry.material_id,
                'formula': entry.formula,
                'pbe_bandgap': entry.pbe_bandgap,
                'hse_bandgap': entry.hse_bandgap,
                'experimental_bandgap': entry.experimental_bandgap,
                'bandgap_source': entry.bandgap_source,
                'space_group': entry.space_group,
                'crystal_system': entry.crystal_system,
                'has_structure': entry.structure is not None,
                'has_hse': entry.hse_bandgap is not None,
                'has_experimental': entry.experimental_bandgap is not None
            })
        
        df = pd.DataFrame(df_data)
        
        # Calculate differences where we have both PBE and HSE/experimental
        hse_diff = []
        exp_diff = []
        
        for _, row in df.iterrows():
            if row['has_hse']:
                hse_diff.append(row['hse_bandgap'] - row['pbe_bandgap'])
            if row['has_experimental']:
                exp_diff.append(row['experimental_bandgap'] - row['pbe_bandgap'])
        
        analysis = {
            'total_materials': len(df),
            'materials_with_hse': df['has_hse'].sum(),
            'materials_with_experimental': df['has_experimental'].sum(),
            'materials_with_both': ((df['has_hse']) & (df['has_experimental'])).sum(),
            'pbe_bandgap_range': (df['pbe_bandgap'].min(), df['pbe_bandgap'].max()),
            'bandgap_sources': df['bandgap_source'].value_counts().to_dict(),
            'crystal_systems': df['crystal_system'].value_counts().to_dict(),
            'materials_with_structure': df['has_structure'].sum()
        }
        
        if hse_diff:
            analysis['hse_differences'] = {
                'mean': float(pd.Series(hse_diff).mean()),
                'std': float(pd.Series(hse_diff).std()),
                'min': float(pd.Series(hse_diff).min()),
                'max': float(pd.Series(hse_diff).max())
            }
        
        if exp_diff:
            analysis['experimental_differences'] = {
                'mean': float(pd.Series(exp_diff).mean()),
                'std': float(pd.Series(exp_diff).std()),
                'min': float(pd.Series(exp_diff).min()),
                'max': float(pd.Series(exp_diff).max())
            }
        
        return analysis
    
    def save_hse_data(self, hse_materials: List[HSEBandgapData], 
                     csv_filename: str = 'hse_bandgap_dataset.csv',
                     json_filename: str = 'hse_bandgap_dataset.json'):
        """Save HSE bandgap data to both CSV and JSON formats"""
        logger.info(f"Saving {len(hse_materials)} HSE materials to {csv_filename} and {json_filename}")
        
        # Save CSV format
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'material_id', 'formula', 'pbe_bandgap', 'hse_bandgap', 
                'experimental_bandgap', 'bandgap_source', 'space_group', 
                'crystal_system', 'structure_cif'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in hse_materials:
                # Convert structure to CIF
                structure_cif = ""
                if entry.structure:
                    try:
                        structure_cif = entry.structure.to(fmt="cif")
                    except Exception as e:
                        logger.warning(f"Could not convert structure to CIF for {entry.material_id}: {e}")
                        structure_cif = str(entry.structure) if entry.structure else ""
                
                writer.writerow({
                    'material_id': entry.material_id,
                    'formula': entry.formula,
                    'pbe_bandgap': entry.pbe_bandgap,
                    'hse_bandgap': entry.hse_bandgap,
                    'experimental_bandgap': entry.experimental_bandgap,
                    'bandgap_source': entry.bandgap_source,
                    'space_group': entry.space_group,
                    'crystal_system': entry.crystal_system,
                    'structure_cif': structure_cif
                })
        
        # Save JSON format with full material data
        json_data = []
        for entry in hse_materials:
            # Convert structure to dictionary for JSON serialization
            structure_dict = None
            structure_cif = ""
            
            if entry.structure:
                try:
                    structure_dict = entry.structure.as_dict()
                    structure_cif = entry.structure.to(fmt="cif")
                except Exception as e:
                    logger.warning(f"Could not process structure for {entry.material_id}: {e}")
                    structure_dict = None
                    structure_cif = ""
            
            # Determine the high-fidelity target
            high_fidelity_target = entry.hse_bandgap if entry.hse_bandgap else entry.experimental_bandgap
            
            material_entry = {
                "material_id": entry.material_id,
                "composition": entry.formula,
                "structure": {
                    "cif": structure_cif,
                    "pymatgen_dict": structure_dict
                },
                "bandgap_data": {
                    "low_fidelity_pbe": entry.pbe_bandgap,
                    "high_fidelity_hse": entry.hse_bandgap,
                    "experimental_bandgap": entry.experimental_bandgap,
                    "bandgap_source": entry.bandgap_source,
                    "calculation_metadata": entry.calculation_metadata
                },
                "crystal_properties": {
                    "space_group": entry.space_group,
                    "crystal_system": entry.crystal_system,
                    "is_direct_gap": entry.is_gap_direct
                },
                "ml_ready": {
                    "input_features": {
                        "pbe_bandgap": entry.pbe_bandgap,
                        "space_group": entry.space_group,
                        "crystal_system": entry.crystal_system,
                        "is_direct_gap": entry.is_gap_direct
                    },
                    "target": high_fidelity_target,
                    "has_structure": entry.structure is not None,
                    "data_quality": entry.bandgap_source
                }
            }
            
            json_data.append(material_entry)
        
        # Save JSON file
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(json_data)} materials in ML-ready JSON format")

def main():
    """Main function to extract HSE bandgap data"""
    API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"
    
    logger.info("=== Materials Project HSE Bandgap Data Extraction ===")
    
    extractor = HSEBandgapExtractor(API_KEY)
    
    try:
        # Extract HSE data using multiple approaches
        hse_materials = extractor.extract_hse_data(max_materials=1000)
        
        if not hse_materials:
            logger.error("❌ No HSE/experimental bandgap data found!")
            logger.info("This confirms that Materials Project has limited high-fidelity bandgap data.")
            logger.info("Recommendation: Use the synthetic data creation approach instead.")
            return
        
        # Analyze data quality
        analysis = extractor.analyze_hse_data(hse_materials)
        
        # Save data in both formats
        extractor.save_hse_data(hse_materials, 
                               csv_filename='hse_bandgap_dataset.csv',
                               json_filename='hse_bandgap_dataset.json')
        
        # Print results
        logger.info("=== HSE EXTRACTION RESULTS ===")
        logger.info(f"Total materials found: {analysis['total_materials']}")
        logger.info(f"Materials with HSE data: {analysis['materials_with_hse']}")
        logger.info(f"Materials with experimental data: {analysis['materials_with_experimental']}")
        logger.info(f"Materials with both HSE and experimental: {analysis['materials_with_both']}")
        logger.info(f"PBE bandgap range: {analysis['pbe_bandgap_range'][0]:.3f} - {analysis['pbe_bandgap_range'][1]:.3f} eV")
        
        logger.info("Data sources:")
        for source, count in analysis['bandgap_sources'].items():
            logger.info(f"  {source}: {count}")
        
        logger.info("Crystal systems:")
        for system, count in analysis['crystal_systems'].items():
            logger.info(f"  {system}: {count}")
        
        if 'hse_differences' in analysis:
            hse_stats = analysis['hse_differences']
            logger.info(f"HSE-PBE differences:")
            logger.info(f"  Mean: {hse_stats['mean']:.3f} eV")
            logger.info(f"  Std:  {hse_stats['std']:.3f} eV")
            logger.info(f"  Range: {hse_stats['min']:.3f} to {hse_stats['max']:.3f} eV")
        
        if 'experimental_differences' in analysis:
            exp_stats = analysis['experimental_differences']
            logger.info(f"Experimental-PBE differences:")
            logger.info(f"  Mean: {exp_stats['mean']:.3f} eV")
            logger.info(f"  Std:  {exp_stats['std']:.3f} eV")
            logger.info(f"  Range: {exp_stats['min']:.3f} to {exp_stats['max']:.3f} eV")
        
        # Save analysis
        with open('hse_bandgap_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info("✅ HSE bandgap extraction completed successfully!")
        logger.info("Files created:")
        logger.info("  - hse_bandgap_dataset.csv (tabular format)")
        logger.info("  - hse_bandgap_dataset.json (ML-ready format with structures)")
        logger.info("  - hse_bandgap_analysis.json (statistical analysis)")
        
        if analysis['materials_with_hse'] == 0 and analysis['materials_with_experimental'] == 0:
            logger.warning("⚠️  No true high-fidelity bandgap data found!")
            logger.warning("Consider using the synthetic data creation approach:")
            logger.warning("  python create_synthetic_hse_data.py")
        
    except Exception as e:
        logger.error(f"Error during HSE extraction: {e}")
        raise

if __name__ == "__main__":
    main()