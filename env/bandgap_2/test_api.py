#!/usr/bin/env python3
"""
Simple test script to verify Materials Project API is working
"""

import logging
from mp_api.client import MPRester

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api():
    """Test basic API functionality"""
    API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"
    
    try:
        logger.info("Testing Materials Project API connection...")
        
        with MPRester(API_KEY) as mpr:
            # Test 1: Simple search for a few materials
            logger.info("Test 1: Basic search for materials with bandgaps...")
            materials = mpr.materials.summary.search(
                band_gap=(0.1, 2.0),  # Small range for testing
                fields=["material_id", "formula_pretty", "band_gap", "structure"],
                num_chunks=1,
                chunk_size=10  # Just get 10 materials for testing
            )
            
            logger.info(f"Found {len(materials)} materials")
            
            # Show first few results
            for i, mat in enumerate(materials[:5]):
                logger.info(f"  {i+1}. {mat.material_id}: {mat.formula_pretty} (bandgap: {mat.band_gap:.3f} eV)")
            
            # Test 2: Get structure for one material
            if materials:
                logger.info("Test 2: Getting structure data...")
                first_material = materials[0]
                logger.info(f"Structure for {first_material.material_id}:")
                logger.info(f"  Formula: {first_material.formula_pretty}")
                
                if first_material.structure is not None:
                    logger.info(f"  Lattice: {first_material.structure.lattice}")
                    logger.info(f"  Number of sites: {len(first_material.structure.sites)}")
                    
                    # Test CIF conversion
                    try:
                        cif_string = first_material.structure.to(fmt="cif")
                        logger.info(f"  CIF conversion: Success ({len(cif_string)} characters)")
                    except Exception as e:
                        logger.warning(f"  CIF conversion failed: {e}")
                else:
                    logger.warning("  Structure data is None - this is normal for summary search")
                    logger.info("  Trying to get structure separately...")
                    
                    # Get structure separately
                    try:
                        structure = mpr.get_structure_by_material_id(first_material.material_id)
                        if structure:
                            logger.info(f"  Retrieved structure: {len(structure.sites)} sites")
                            cif_string = structure.to(fmt="cif")
                            logger.info(f"  CIF conversion: Success ({len(cif_string)} characters)")
                        else:
                            logger.warning("  Could not retrieve structure separately")
                    except Exception as e:
                        logger.warning(f"  Failed to get structure separately: {e}")
            
            logger.info("✅ API test completed successfully!")
            return True
            
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    if success:
        print("\n🎉 API is working! You can now run: python extract_mp_data.py")
    else:
        print("\n❌ API test failed. Please check your API key and internet connection.")