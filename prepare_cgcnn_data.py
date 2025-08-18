#!/usr/bin/env python3
"""
Data Preparation Script for Enhanced CGCNN Training

This script converts CIF files and id_prop.csv data into the format
expected by enhanced_cgcnn_ionic_conductivity.py
"""

import os
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CGCNNDataPreparator:
    """Prepares data for Enhanced CGCNN training"""
    
    def __init__(self, cif_dir: str, id_prop_path: str, output_path: str):
        self.cif_dir = Path(cif_dir)
        self.id_prop_path = Path(id_prop_path)
        self.output_path = Path(output_path)
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'matched_cifs': 0,
            'missing_cifs': 0,
            'filtered_low_conductivity': 0,
            'final_dataset_size': 0
        }
    
    def load_id_prop_data(self) -> pd.DataFrame:
        """Load ionic conductivity data from id_prop.csv"""
        logger.info(f"Loading ionic conductivity data from {self.id_prop_path}")
        
        try:
            df = pd.read_csv(self.id_prop_path, header=None, names=['material_id', 'ionic_conductivity'])
            
            # Convert to numeric and drop invalid values
            df['ionic_conductivity'] = pd.to_numeric(df['ionic_conductivity'], errors='coerce')
            df = df.dropna(subset=['ionic_conductivity'])
            
            self.stats['total_entries'] = len(df)
            logger.info(f"Loaded {len(df)} entries with valid ionic conductivity values")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading id_prop data: {e}")
            raise
    
    def find_cif_files(self) -> Dict[str, str]:
        """Find all CIF files and create material_id -> filepath mapping"""
        logger.info(f"Scanning for CIF files in {self.cif_dir}")
        
        cif_files = {}
        if self.cif_dir.exists():
            for cif_file in self.cif_dir.glob("*.cif"):
                material_id = cif_file.stem  # filename without extension
                cif_files[material_id] = str(cif_file)
        
        logger.info(f"Found {len(cif_files)} CIF files")
        return cif_files
    
    def create_enhanced_cgcnn_dataset(self, conductivity_threshold: float = 1e-12) -> List[Dict[str, Any]]:
        """Create dataset in Enhanced CGCNN format"""
        logger.info("Creating Enhanced CGCNN dataset format")
        
        # Load data
        df = self.load_id_prop_data()
        cif_files = self.find_cif_files()
        
        # Filter by conductivity threshold
        initial_count = len(df)
        df = df[df['ionic_conductivity'] > conductivity_threshold]
        self.stats['filtered_low_conductivity'] = initial_count - len(df)
        
        if self.stats['filtered_low_conductivity'] > 0:
            logger.info(f"Filtered out {self.stats['filtered_low_conductivity']} entries with conductivity <= {conductivity_threshold}")
        
        # Create dataset entries
        dataset = []
        missing_cifs = []
        
        for _, row in df.iterrows():
            material_id = str(row['material_id']).strip()
            conductivity = float(row['ionic_conductivity'])
            
            # Check if CIF file exists
            if material_id in cif_files:
                cif_path = cif_files[material_id]
                
                try:
                    # Read CIF content
                    with open(cif_path, 'r') as f:
                        cif_content = f.read()
                    
                    # Create data entry
                    data_entry = {
                        'material_id': material_id,
                        'cif': cif_content,
                        'ionic_conductivity': conductivity,
                        'cif_path': cif_path
                    }
                    
                    dataset.append(data_entry)
                    self.stats['matched_cifs'] += 1
                    
                except Exception as e:
                    logger.warning(f"Error reading CIF file {cif_path}: {e}")
                    missing_cifs.append(material_id)
            else:
                missing_cifs.append(material_id)
                self.stats['missing_cifs'] += 1
        
        self.stats['final_dataset_size'] = len(dataset)
        
        # Log missing CIFs (first 10)
        if missing_cifs:
            logger.warning(f"Missing CIF files for {len(missing_cifs)} materials")
            logger.warning(f"First 10 missing: {missing_cifs[:10]}")
        
        logger.info(f"Created dataset with {len(dataset)} entries")
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], format: str = 'pickle') -> None:
        """Save dataset in specified format"""
        logger.info(f"Saving dataset to {self.output_path}")
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(self.output_path, 'wb') as f:
                pickle.dump(dataset, f)
        elif format == 'csv':
            # Create CSV with basic information
            csv_data = []
            for entry in dataset:
                csv_entry = {
                    'material_id': entry['material_id'],
                    'ionic_conductivity': entry['ionic_conductivity'],
                    'cif_path': entry['cif_path']
                }
                csv_data.append(csv_entry)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(self.output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dataset saved successfully")
    
    def print_statistics(self) -> None:
        """Print preparation statistics"""
        logger.info("=" * 60)
        logger.info("DATA PREPARATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total entries in id_prop.csv: {self.stats['total_entries']}")
        logger.info(f"Entries with matching CIF files: {self.stats['matched_cifs']}")
        logger.info(f"Entries with missing CIF files: {self.stats['missing_cifs']}")
        logger.info(f"Entries filtered (low conductivity): {self.stats['filtered_low_conductivity']}")
        logger.info(f"Final dataset size: {self.stats['final_dataset_size']}")
        
        if self.stats['total_entries'] > 0:
            match_rate = (self.stats['matched_cifs'] / self.stats['total_entries']) * 100
            logger.info(f"CIF match rate: {match_rate:.1f}%")
        
        logger.info("=" * 60)
    
    def prepare_data(self, conductivity_threshold: float = 1e-12, format: str = 'pickle') -> None:
        """Main data preparation function"""
        logger.info("Starting data preparation for Enhanced CGCNN")
        
        # Create dataset
        dataset = self.create_enhanced_cgcnn_dataset(conductivity_threshold)
        
        # Save dataset
        self.save_dataset(dataset, format)
        
        # Print statistics
        self.print_statistics()


def main():
    """Main function"""
    # Configuration
    CIF_DIR = "env/property_predictions/CIF_OBELiX/cifs"  # Use existing CIF directory
    ID_PROP_PATH = "env/property_predictions/CIF_OBELiX/cifs/id_prop.csv"
    OUTPUT_PATH = "data/ionic_conductivity_dataset.pkl"
    CONDUCTIVITY_THRESHOLD = 1e-12
    
    # Create preparator
    preparator = CGCNNDataPreparator(
        cif_dir=CIF_DIR,
        id_prop_path=ID_PROP_PATH,
        output_path=OUTPUT_PATH
    )
    
    # Prepare data
    preparator.prepare_data(
        conductivity_threshold=CONDUCTIVITY_THRESHOLD,
        format='pickle'
    )
    
    print(f"\n✅ Data preparation completed!")
    print(f"📁 Dataset saved to: {OUTPUT_PATH}")
    print(f"🚀 Ready for Enhanced CGCNN training!")


if __name__ == "__main__":
    main()