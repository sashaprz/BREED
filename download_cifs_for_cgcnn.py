#!/usr/bin/env python3
"""
CIF Downloader for Enhanced CGCNN Training

This script downloads CIF files from Materials Project for all materials
listed in the id_prop.csv file, preparing them for Enhanced CGCNN training.
"""

import os
import pandas as pd
import time
from pathlib import Path
from mp_api.client import MPRester
import logging
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CIFDownloader:
    """Downloads CIF files from Materials Project for CGCNN training"""
    
    def __init__(self, api_key: str, output_dir: str = "data/cifs", delay: float = 0.2):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.mpr = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_materials': 0,
            'downloaded': 0,
            'already_exists': 0,
            'failed': 0,
            'not_found': 0
        }
    
    def connect_to_mp(self) -> bool:
        """Connect to Materials Project API"""
        try:
            self.mpr = MPRester(self.api_key)
            logger.info("Successfully connected to Materials Project API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Materials Project: {e}")
            return False
    
    def download_cif(self, material_id: str) -> bool:
        """Download CIF file for a single material"""
        try:
            # Check if file already exists
            cif_path = self.output_dir / f"{material_id}.cif"
            if cif_path.exists():
                logger.debug(f"CIF already exists: {material_id}")
                self.stats['already_exists'] += 1
                return True
            
            # Search for material
            try:
                results = self.mpr.materials.summary.search(
                    material_ids=[material_id],
                    fields=["material_id", "structure"]
                )
                
                if not results:
                    logger.warning(f"Material not found: {material_id}")
                    self.stats['not_found'] += 1
                    return False
                
                # Get structure and save as CIF
                structure = results[0].structure
                if structure:
                    structure.to(filename=str(cif_path), fmt="cif")
                    logger.debug(f"Downloaded CIF: {material_id}")
                    self.stats['downloaded'] += 1
                    return True
                else:
                    logger.warning(f"No structure data for: {material_id}")
                    self.stats['failed'] += 1
                    return False
                    
            except Exception as e:
                logger.error(f"Error downloading {material_id}: {e}")
                self.stats['failed'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error for {material_id}: {e}")
            self.stats['failed'] += 1
            return False
    
    def download_from_id_prop(self, id_prop_path: str) -> None:
        """Download CIF files for all materials in id_prop.csv"""
        logger.info(f"Loading materials from {id_prop_path}")
        
        # Read id_prop.csv
        try:
            df = pd.read_csv(id_prop_path, header=None, names=['material_id', 'ionic_conductivity'])
            material_ids = df['material_id'].unique().tolist()
            self.stats['total_materials'] = len(material_ids)
            
            logger.info(f"Found {len(material_ids)} unique materials to download")
            
        except Exception as e:
            logger.error(f"Error reading {id_prop_path}: {e}")
            return
        
        # Connect to Materials Project
        if not self.connect_to_mp():
            return
        
        # Download CIF files
        logger.info("Starting CIF downloads...")
        
        for i, material_id in enumerate(material_ids, 1):
            # Progress reporting
            if i <= 10 or i % 50 == 0 or i == len(material_ids):
                logger.info(f"Processing {i}/{len(material_ids)}: {material_id}")
            
            # Download CIF
            self.download_cif(material_id)
            
            # Rate limiting
            time.sleep(self.delay)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print download summary"""
        logger.info("=" * 60)
        logger.info("CIF DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total materials: {self.stats['total_materials']}")
        logger.info(f"Downloaded: {self.stats['downloaded']}")
        logger.info(f"Already existed: {self.stats['already_exists']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Not found: {self.stats['not_found']}")
        
        total_available = self.stats['downloaded'] + self.stats['already_exists']
        logger.info(f"Total CIF files available: {total_available}")
        
        if total_available > 0:
            success_rate = (total_available / self.stats['total_materials']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
            logger.info(f"CIF files saved to: {self.output_dir}")
        
        logger.info("=" * 60)


def main():
    """Main function"""
    # Configuration
    API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"  # Your Materials Project API key
    ID_PROP_PATH = "env/property_predictions/CIF_OBELiX/cifs/id_prop.csv"
    OUTPUT_DIR = "data/cifs"
    DELAY = 0.2  # seconds between requests
    
    # Create downloader
    downloader = CIFDownloader(
        api_key=API_KEY,
        output_dir=OUTPUT_DIR,
        delay=DELAY
    )
    
    # Download CIF files
    downloader.download_from_id_prop(ID_PROP_PATH)
    
    print(f"\n✅ CIF download completed!")
    print(f"📁 CIF files saved to: {OUTPUT_DIR}")
    print(f"🚀 Ready for Enhanced CGCNN training!")


if __name__ == "__main__":
    main()