#!/usr/bin/env python3
"""
Materials Project Paired Bandgap Data Extractor
Extracts materials with both PBE (summary) and high-fidelity (electronic_structure) bandgaps
Based on exploration findings showing electronic_structure endpoint has different bandgap values
"""

import os
import json
import csv
import time
from typing import Dict, List, Optional, Tuple
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
class PairedBandgapData:
    """Data class to store paired bandgap information"""
    material_id: str
    formula: str
    structure: Optional[Structure]
    pbe_bandgap: float  # From summary endpoint
    es_bandgap: float   # From electronic_structure endpoint
    bandgap_difference: float  # es_bandgap - pbe_bandgap
    is_gap_direct: bool
    space_group: int
    crystal_system: str

class PairedBandgapExtractor:
    """Extract materials with both PBE and electronic structure bandgaps"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.mpr = MPRester(api_key)
        
    def extract_paired_data(self, max_materials: int = 2000) -> List[PairedBandgapData]:
        """Extract materials with both PBE and electronic structure bandgaps"""
        logger.info(f"Extracting paired bandgap data for up to {max_materials} materials...")
        
        # Step 1: Get materials from electronic structure endpoint (these have high-fidelity data)
        logger.info("Step 1: Getting materials with electronic structure data...")
        es_materials = self._get_electronic_structure_materials(max_materials)
        logger.info(f"Found {len(es_materials)} materials with electronic structure data")
        
        if not es_materials:
            logger.warning("No electronic structure materials found!")
            return []
        
        # Step 2: Get corresponding summary data for the same materials
        logger.info("Step 2: Getting corresponding summary data...")
        paired_data = self._get_paired_data(es_materials)
        logger.info(f"Successfully paired {len(paired_data)} materials")
        
        return paired_data
    
    def _get_electronic_structure_materials(self, max_materials: int) -> List:
        """Get materials from electronic structure endpoint"""
        try:
            # Use the working electronic structure endpoint
            es_materials = self.mpr.materials.electronic_structure.search(
                band_gap=(0.1, 8.0),  # Semiconductors and insulators
                fields=[
                    "material_id", "formula_pretty", "band_gap", 
                    "is_gap_direct", "symmetry", "structure"
                ],
                num_chunks=max(1, max_materials // 500),
                chunk_size=min(500, max_materials)
            )
            
            logger.info(f"Electronic structure search returned {len(es_materials)} materials")
            return es_materials[:max_materials]
            
        except Exception as e:
            logger.error(f"Error getting electronic structure materials: {e}")
            return []
    
    def _get_paired_data(self, es_materials: List) -> List[PairedBandgapData]:
        """Get corresponding summary data and create paired dataset"""
        paired_data = []
        
        # Get material IDs from electronic structure data
        material_ids = [str(mat.material_id) for mat in es_materials]
        logger.info(f"Getting summary data for {len(material_ids)} materials...")
        
        # Get summary data in batches
        batch_size = 100
        summary_materials = []
        
        for i in range(0, len(material_ids), batch_size):
            batch_ids = material_ids[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(material_ids)-1)//batch_size + 1}")
            
            try:
                # Get summary data for this batch
                batch_summary = self.mpr.materials.summary.search(
                    material_ids=batch_ids,
                    fields=[
                        "material_id", "formula_pretty", "band_gap", 
                        "structure", "symmetry"
                    ]
                )
                summary_materials.extend(batch_summary)
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error getting summary data for batch: {e}")
                continue
        
        logger.info(f"Got summary data for {len(summary_materials)} materials")
        
        # Create lookup dictionary for summary data
        summary_lookup = {str(mat.material_id): mat for mat in summary_materials}
        
        # Pair the data
        for es_mat in es_materials:
            es_id = str(es_mat.material_id)
            
            if es_id in summary_lookup:
                summary_mat = summary_lookup[es_id]
                
                try:
                    # Get structure (prefer from summary, fallback to electronic structure)
                    structure = None
                    if hasattr(summary_mat, 'structure') and summary_mat.structure:
                        structure = summary_mat.structure
                    elif hasattr(es_mat, 'structure') and es_mat.structure:
                        structure = es_mat.structure
                    
                    # Get space group and crystal system
                    space_group = 1
                    crystal_system = "unknown"
                    
                    if hasattr(summary_mat, 'symmetry') and summary_mat.symmetry:
                        space_group = getattr(summary_mat.symmetry, 'number', 1)
                        crystal_system = str(getattr(summary_mat.symmetry, 'crystal_system', 'unknown'))
                    elif hasattr(es_mat, 'symmetry') and es_mat.symmetry:
                        space_group = getattr(es_mat.symmetry, 'number', 1)
                        crystal_system = str(getattr(es_mat.symmetry, 'crystal_system', 'unknown'))
                    
                    # Create paired data entry
                    paired_entry = PairedBandgapData(
                        material_id=es_id,
                        formula=es_mat.formula_pretty,
                        structure=structure,
                        pbe_bandgap=summary_mat.band_gap,  # PBE from summary
                        es_bandgap=es_mat.band_gap,       # Higher-fidelity from electronic structure
                        bandgap_difference=es_mat.band_gap - summary_mat.band_gap,
                        is_gap_direct=getattr(es_mat, 'is_gap_direct', False),
                        space_group=space_group,
                        crystal_system=crystal_system
                    )
                    
                    paired_data.append(paired_entry)
                    
                except Exception as e:
                    logger.warning(f"Error creating paired data for {es_id}: {e}")
                    continue
            else:
                logger.warning(f"No summary data found for {es_id}")
        
        return paired_data
    
    def analyze_data_quality(self, paired_data: List[PairedBandgapData]) -> Dict:
        """Analyze the quality and characteristics of paired data"""
        if not paired_data:
            return {}
        
        # Convert to DataFrame for analysis
        df_data = []
        for entry in paired_data:
            df_data.append({
                'material_id': entry.material_id,
                'formula': entry.formula,
                'pbe_bandgap': entry.pbe_bandgap,
                'es_bandgap': entry.es_bandgap,
                'bandgap_difference': entry.bandgap_difference,
                'is_gap_direct': entry.is_gap_direct,
                'space_group': entry.space_group,
                'crystal_system': entry.crystal_system,
                'has_structure': entry.structure is not None
            })
        
        df = pd.DataFrame(df_data)
        
        analysis = {
            'total_materials': len(df),
            'pbe_bandgap_range': (df['pbe_bandgap'].min(), df['pbe_bandgap'].max()),
            'es_bandgap_range': (df['es_bandgap'].min(), df['es_bandgap'].max()),
            'bandgap_difference_stats': {
                'mean': df['bandgap_difference'].mean(),
                'std': df['bandgap_difference'].std(),
                'min': df['bandgap_difference'].min(),
                'max': df['bandgap_difference'].max()
            },
            'correlation': df['pbe_bandgap'].corr(df['es_bandgap']),
            'direct_gap_count': df['is_gap_direct'].sum(),
            'materials_with_structure': df['has_structure'].sum(),
            'crystal_systems': df['crystal_system'].value_counts().to_dict(),
            'space_group_distribution': df['space_group'].value_counts().head(10).to_dict()
        }
        
        return analysis
    
    def save_paired_data(self, paired_data: List[PairedBandgapData],
                        csv_filename: str = 'paired_bandgap_dataset.csv',
                        json_filename: str = 'paired_bandgap_dataset.json'):
        """Save paired bandgap data to both CSV and JSON formats"""
        logger.info(f"Saving {len(paired_data)} paired materials to {csv_filename} and {json_filename}")
        
        # Save CSV format
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'material_id', 'formula', 'pbe_bandgap', 'es_bandgap',
                'bandgap_difference', 'is_gap_direct', 'space_group',
                'crystal_system', 'structure_cif'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in paired_data:
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
                    'es_bandgap': entry.es_bandgap,
                    'bandgap_difference': entry.bandgap_difference,
                    'is_gap_direct': entry.is_gap_direct,
                    'space_group': entry.space_group,
                    'crystal_system': entry.crystal_system,
                    'structure_cif': structure_cif
                })
        
        # Save JSON format with full material data
        json_data = []
        for entry in paired_data:
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
            
            material_entry = {
                "material_id": entry.material_id,
                "composition": entry.formula,
                "structure": {
                    "cif": structure_cif,
                    "pymatgen_dict": structure_dict
                },
                "bandgap_data": {
                    "low_fidelity_pbe": entry.pbe_bandgap,
                    "high_fidelity_es": entry.es_bandgap,
                    "difference": entry.bandgap_difference,
                    "is_direct_gap": entry.is_gap_direct
                },
                "crystal_properties": {
                    "space_group": entry.space_group,
                    "crystal_system": entry.crystal_system
                },
                "ml_ready": {
                    "input_features": {
                        "pbe_bandgap": entry.pbe_bandgap,
                        "space_group": entry.space_group,
                        "crystal_system": entry.crystal_system,
                        "is_direct_gap": entry.is_gap_direct
                    },
                    "target": entry.es_bandgap,
                    "has_structure": entry.structure is not None
                }
            }
            
            json_data.append(material_entry)
        
        # Save JSON file
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(json_data)} materials in ML-ready JSON format")

def main():
    """Main function to extract paired bandgap data"""
    API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"
    
    logger.info("=== Materials Project Paired Bandgap Data Extraction ===")
    
    extractor = PairedBandgapExtractor(API_KEY)
    
    try:
        # Extract paired data
        paired_data = extractor.extract_paired_data(max_materials=1000)
        
        if not paired_data:
            logger.error("No paired bandgap data found!")
            return
        
        # Analyze data quality
        analysis = extractor.analyze_data_quality(paired_data)
        
        # Save data in both formats
        extractor.save_paired_data(paired_data,
                                  csv_filename='paired_bandgap_dataset.csv',
                                  json_filename='paired_bandgap_dataset.json')
        
        # Print results
        logger.info("=== EXTRACTION RESULTS ===")
        logger.info(f"Total paired materials: {analysis['total_materials']}")
        logger.info(f"PBE bandgap range: {analysis['pbe_bandgap_range'][0]:.3f} - {analysis['pbe_bandgap_range'][1]:.3f} eV")
        logger.info(f"ES bandgap range: {analysis['es_bandgap_range'][0]:.3f} - {analysis['es_bandgap_range'][1]:.3f} eV")
        logger.info(f"Bandgap difference (ES - PBE):")
        logger.info(f"  Mean: {analysis['bandgap_difference_stats']['mean']:.3f} eV")
        logger.info(f"  Std:  {analysis['bandgap_difference_stats']['std']:.3f} eV")
        logger.info(f"  Range: {analysis['bandgap_difference_stats']['min']:.3f} to {analysis['bandgap_difference_stats']['max']:.3f} eV")
        logger.info(f"PBE-ES correlation: {analysis['correlation']:.3f}")
        logger.info(f"Direct bandgap materials: {analysis['direct_gap_count']}")
        logger.info(f"Materials with structure: {analysis['materials_with_structure']}")
        
        logger.info("Crystal systems:")
        for system, count in analysis['crystal_systems'].items():
            logger.info(f"  {system}: {count}")
        
        # Save analysis
        with open('paired_bandgap_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info("✅ Paired bandgap extraction completed successfully!")
        logger.info("Files created:")
        logger.info("  - paired_bandgap_dataset.csv (tabular format)")
        logger.info("  - paired_bandgap_dataset.json (ML-ready format with structures)")
        logger.info("  - paired_bandgap_analysis.json (statistical analysis)")
        
        # Check if we actually got different bandgap values
        if analysis['bandgap_difference_stats']['std'] == 0.0:
            logger.warning("⚠️  WARNING: All bandgap differences are 0.0!")
            logger.warning("This suggests the electronic structure endpoint has the same values as summary.")
            logger.warning("The 'high-fidelity' data might not actually be HSE/experimental values.")
            logger.warning("Consider using the synthetic data creation approach instead.")
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise

if __name__ == "__main__":
    main()