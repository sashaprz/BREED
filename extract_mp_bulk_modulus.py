#!/usr/bin/env python3
"""
Extract Bulk Modulus Values from Materials Project for OBELiX Dataset

This script queries the Materials Project API to match OBELiX crystal structures
with their bulk modulus values, focusing on inorganic crystals with high bulk modulus.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import time

# Materials Project API
try:
    from mp_api.client import MPRester
    from pymatgen.core import Structure, Composition
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.cif import CifParser
    MP_AVAILABLE = True
except ImportError:
    print("❌ Materials Project API not available. Install with:")
    print("pip install mp-api pymatgen")
    MP_AVAILABLE = False
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaterialsProjectBulkModulusExtractor:
    """Extract bulk modulus values from Materials Project for OBELiX structures"""
    
    def __init__(self, api_key: str, obelix_data_path: str = "env/property_predictions/CIF_OBELiX"):
        self.api_key = api_key
        self.obelix_path = Path(obelix_data_path)
        self.cifs_path = self.obelix_path / "cifs"
        
        # Output paths
        self.output_dir = Path("bulk_modulus_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Structure matcher for comparing crystals
        self.structure_matcher = StructureMatcher(
            ltol=0.2,  # Length tolerance
            stol=0.3,  # Site tolerance  
            angle_tol=5,  # Angle tolerance in degrees
            primitive_cell=True,
            scale=True,
            attempt_supercell=False
        )
        
        # Results storage
        self.bulk_modulus_data = []
        self.matched_structures = {}
        self.failed_matches = []
        
    def load_obelix_structures(self) -> List[Tuple[str, Structure, str]]:
        """Load OBELiX CIF structures"""
        logger.info(f"Loading OBELiX structures from {self.cifs_path}")
        
        structures = []
        cif_files = list(self.cifs_path.glob("*.cif"))
        
        if not cif_files:
            logger.error(f"No CIF files found in {self.cifs_path}")
            return structures
            
        logger.info(f"Found {len(cif_files)} CIF files")
        
        for cif_file in tqdm(cif_files, desc="Loading CIF files"):
            try:
                parser = CifParser(str(cif_file))
                structure = parser.get_structures()[0]  # Get first structure
                
                # Basic validation
                if len(structure) > 200:  # Skip very large structures
                    continue
                    
                structures.append((cif_file.stem, structure, str(cif_file)))
                
            except Exception as e:
                logger.warning(f"Failed to parse {cif_file}: {e}")
                continue
                
        logger.info(f"Successfully loaded {len(structures)} structures")
        return structures
    
    def query_materials_project_bulk_modulus(self, composition: str, max_results: int = 10) -> List[Dict]:
        """Query Materials Project for bulk modulus data by composition"""
        try:
            with MPRester(self.api_key) as mpr:
                # Search by composition
                docs = mpr.materials.summary.search(
                    formula=composition,
                    fields=[
                        "material_id", 
                        "formula_pretty", 
                        "structure",
                        "bulk_modulus",
                        "shear_modulus", 
                        "universal_anisotropy",
                        "homogeneous_poisson",
                        "energy_above_hull",
                        "formation_energy_per_atom",
                        "is_stable"
                    ]
                )
                
                # Filter for materials with bulk modulus data
                bulk_modulus_docs = []
                for doc in docs[:max_results]:
                    try:
                        # Handle different bulk modulus data formats
                        bulk_modulus_vrh = None
                        bulk_modulus_reuss = None
                        bulk_modulus_voigt = None
                        shear_modulus_vrh = None
                        
                        if hasattr(doc, 'bulk_modulus') and doc.bulk_modulus is not None:
                            if hasattr(doc.bulk_modulus, 'vrh'):
                                # New API format with vrh attribute
                                bulk_modulus_vrh = doc.bulk_modulus.vrh
                                bulk_modulus_reuss = getattr(doc.bulk_modulus, 'reuss', None)
                                bulk_modulus_voigt = getattr(doc.bulk_modulus, 'voigt', None)
                            elif isinstance(doc.bulk_modulus, (int, float)):
                                # Simple numeric format
                                bulk_modulus_vrh = float(doc.bulk_modulus)
                            elif isinstance(doc.bulk_modulus, dict):
                                # Dictionary format
                                bulk_modulus_vrh = doc.bulk_modulus.get('vrh', doc.bulk_modulus.get('value', None))
                                bulk_modulus_reuss = doc.bulk_modulus.get('reuss', None)
                                bulk_modulus_voigt = doc.bulk_modulus.get('voigt', None)
                        
                        # Handle shear modulus
                        if hasattr(doc, 'shear_modulus') and doc.shear_modulus is not None:
                            if hasattr(doc.shear_modulus, 'vrh'):
                                shear_modulus_vrh = doc.shear_modulus.vrh
                            elif isinstance(doc.shear_modulus, (int, float)):
                                shear_modulus_vrh = float(doc.shear_modulus)
                            elif isinstance(doc.shear_modulus, dict):
                                shear_modulus_vrh = doc.shear_modulus.get('vrh', doc.shear_modulus.get('value', None))
                        
                        # Only include if we have valid bulk modulus data
                        if bulk_modulus_vrh is not None and bulk_modulus_vrh > 0:
                            bulk_modulus_docs.append({
                                'material_id': doc.material_id,
                                'formula': doc.formula_pretty,
                                'structure': doc.structure,
                                'bulk_modulus_vrh': float(bulk_modulus_vrh),
                                'bulk_modulus_reuss': float(bulk_modulus_reuss) if bulk_modulus_reuss is not None else None,
                                'bulk_modulus_voigt': float(bulk_modulus_voigt) if bulk_modulus_voigt is not None else None,
                                'shear_modulus': float(shear_modulus_vrh) if shear_modulus_vrh is not None else None,
                                'energy_above_hull': getattr(doc, 'energy_above_hull', 0.0),
                                'formation_energy': getattr(doc, 'formation_energy_per_atom', 0.0),
                                'is_stable': getattr(doc, 'is_stable', True)
                            })
                    except Exception as e:
                        logger.debug(f"Error processing document {getattr(doc, 'material_id', 'unknown')}: {e}")
                        continue
                
                return bulk_modulus_docs
                
        except Exception as e:
            logger.error(f"Error querying Materials Project for {composition}: {e}")
            return []
    
    def find_best_structure_match(self, target_structure: Structure, mp_candidates: List[Dict]) -> Optional[Dict]:
        """Find the best matching structure from Materials Project candidates"""
        best_match = None
        best_score = float('inf')
        
        for candidate in mp_candidates:
            try:
                mp_structure = candidate['structure']
                
                # Try to match structures
                if self.structure_matcher.fit(target_structure, mp_structure):
                    # Calculate a simple matching score based on composition and volume
                    target_comp = target_structure.composition
                    mp_comp = mp_structure.composition
                    
                    # Composition similarity (normalized difference)
                    comp_diff = sum(abs(target_comp.get_atomic_fraction(el) - 
                                      mp_comp.get_atomic_fraction(el)) 
                                  for el in set(target_comp.elements) | set(mp_comp.elements))
                    
                    # Volume similarity
                    vol_diff = abs(target_structure.volume - mp_structure.volume) / target_structure.volume
                    
                    # Combined score (lower is better)
                    score = comp_diff + 0.1 * vol_diff
                    
                    if score < best_score:
                        best_score = score
                        best_match = candidate
                        best_match['match_score'] = score
                        
            except Exception as e:
                logger.debug(f"Structure matching failed for {candidate['material_id']}: {e}")
                continue
                
        return best_match
    
    def extract_bulk_modulus_for_obelix(self) -> None:
        """Main extraction process"""
        logger.info("🚀 Starting bulk modulus extraction from Materials Project")
        
        # Load OBELiX structures
        obelix_structures = self.load_obelix_structures()
        
        if not obelix_structures:
            logger.error("No OBELiX structures loaded. Exiting.")
            return
        
        logger.info(f"Processing {len(obelix_structures)} OBELiX structures...")
        
        # Process each structure
        for obelix_id, structure, cif_path in tqdm(obelix_structures, desc="Matching structures"):
            try:
                # Get composition string
                composition = structure.composition.reduced_formula
                
                # Query Materials Project
                mp_candidates = self.query_materials_project_bulk_modulus(composition)
                
                if not mp_candidates:
                    self.failed_matches.append({
                        'obelix_id': obelix_id,
                        'composition': composition,
                        'reason': 'No MP candidates with bulk modulus'
                    })
                    continue
                
                # Find best structure match
                best_match = self.find_best_structure_match(structure, mp_candidates)
                
                if best_match:
                    # Store successful match
                    bulk_modulus_entry = {
                        'obelix_id': obelix_id,
                        'cif_path': cif_path,
                        'composition': composition,
                        'mp_material_id': best_match['material_id'],
                        'mp_formula': best_match['formula'],
                        'bulk_modulus_vrh': best_match['bulk_modulus_vrh'],
                        'bulk_modulus_reuss': best_match['bulk_modulus_reuss'],
                        'bulk_modulus_voigt': best_match['bulk_modulus_voigt'],
                        'shear_modulus': best_match['shear_modulus'],
                        'energy_above_hull': best_match['energy_above_hull'],
                        'formation_energy': best_match['formation_energy'],
                        'is_stable': best_match['is_stable'],
                        'match_score': best_match['match_score'],
                        'num_atoms': len(structure),
                        'volume': structure.volume,
                        'density': structure.density
                    }
                    
                    self.bulk_modulus_data.append(bulk_modulus_entry)
                    self.matched_structures[obelix_id] = best_match
                    
                    logger.debug(f"✅ Matched {obelix_id} -> {best_match['material_id']} "
                               f"(BM: {best_match['bulk_modulus_vrh']:.1f} GPa)")
                else:
                    self.failed_matches.append({
                        'obelix_id': obelix_id,
                        'composition': composition,
                        'reason': 'No structure match found',
                        'num_candidates': len(mp_candidates)
                    })
                
                # Rate limiting
                time.sleep(0.1)  # Be nice to the API
                
            except Exception as e:
                logger.error(f"Error processing {obelix_id}: {e}")
                self.failed_matches.append({
                    'obelix_id': obelix_id,
                    'composition': composition if 'composition' in locals() else 'unknown',
                    'reason': f'Processing error: {str(e)}'
                })
                continue
    
    def filter_high_bulk_modulus_materials(self, min_bulk_modulus: float = 20.0) -> List[Dict]:
        """Filter for materials with high bulk modulus suitable for fine-tuning"""
        logger.info(f"Filtering for materials with bulk modulus > {min_bulk_modulus} GPa")
        
        high_bm_materials = []
        for entry in self.bulk_modulus_data:
            bulk_modulus = entry['bulk_modulus_vrh']
            
            # Filter criteria
            if (bulk_modulus > min_bulk_modulus and 
                entry['energy_above_hull'] < 0.1 and  # Stable or nearly stable
                entry['match_score'] < 0.5):  # Good structure match
                
                high_bm_materials.append(entry)
        
        logger.info(f"Found {len(high_bm_materials)} high bulk modulus materials")
        return high_bm_materials
    
    def save_results(self) -> None:
        """Save extraction results"""
        logger.info("💾 Saving results...")
        
        # Save all bulk modulus data
        df_all = pd.DataFrame(self.bulk_modulus_data)
        df_all.to_csv(self.output_dir / "obelix_bulk_modulus_all.csv", index=False)
        
        # Save high bulk modulus materials for fine-tuning
        high_bm_materials = self.filter_high_bulk_modulus_materials()
        df_high_bm = pd.DataFrame(high_bm_materials)
        df_high_bm.to_csv(self.output_dir / "obelix_bulk_modulus_high.csv", index=False)
        
        # Save failed matches for analysis
        df_failed = pd.DataFrame(self.failed_matches)
        df_failed.to_csv(self.output_dir / "failed_matches.csv", index=False)
        
        # Save summary statistics
        summary = {
            'total_obelix_structures': len(self.bulk_modulus_data) + len(self.failed_matches),
            'successful_matches': len(self.bulk_modulus_data),
            'failed_matches': len(self.failed_matches),
            'high_bulk_modulus_materials': len(high_bm_materials),
            'bulk_modulus_stats': {}
        }
        
        # Only calculate stats if we have data
        if self.bulk_modulus_data:
            bulk_moduli = [entry['bulk_modulus_vrh'] for entry in self.bulk_modulus_data]
            summary['bulk_modulus_stats'] = {
                'mean': float(np.mean(bulk_moduli)),
                'std': float(np.std(bulk_moduli)),
                'min': float(np.min(bulk_moduli)),
                'max': float(np.max(bulk_moduli)),
                'median': float(np.median(bulk_moduli))
            }
        else:
            summary['bulk_modulus_stats'] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        with open(self.output_dir / "extraction_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("📊 EXTRACTION SUMMARY:")
        logger.info(f"   Total structures processed: {summary['total_obelix_structures']}")
        logger.info(f"   Successful matches: {summary['successful_matches']}")
        logger.info(f"   Failed matches: {summary['failed_matches']}")
        logger.info(f"   High bulk modulus materials: {summary['high_bulk_modulus_materials']}")
        logger.info(f"   Bulk modulus range: {summary['bulk_modulus_stats']['min']:.1f} - {summary['bulk_modulus_stats']['max']:.1f} GPa")
        logger.info(f"   Mean bulk modulus: {summary['bulk_modulus_stats']['mean']:.1f} ± {summary['bulk_modulus_stats']['std']:.1f} GPa")
        
        logger.info(f"📁 Results saved to: {self.output_dir}")


def main():
    """Main function"""
    print("🔍 Materials Project Bulk Modulus Extractor for OBELiX Dataset")
    print("=" * 70)
    
    # Materials Project API key
    API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"
    
    # Check if OBELiX data exists
    obelix_path = Path("env/property_predictions/CIF_OBELiX/cifs")
    if not obelix_path.exists():
        print(f"❌ OBELiX CIF directory not found: {obelix_path}")
        print("Please ensure the OBELiX dataset is available.")
        return
    
    # Create extractor
    extractor = MaterialsProjectBulkModulusExtractor(API_KEY)
    
    # Run extraction
    try:
        extractor.extract_bulk_modulus_for_obelix()
        extractor.save_results()
        
        print("\n🎉 Bulk modulus extraction completed successfully!")
        print(f"📁 Results saved to: {extractor.output_dir}")
        print("\nNext step: Run fine-tuning with 'python finetune_cgcnn_bulk_modulus.py'")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()