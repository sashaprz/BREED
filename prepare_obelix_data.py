"""
OBELiX Dataset Preparation Script for Enhanced CGCNN Training

This script prepares the OBELiX dataset for ionic conductivity prediction by:
1. Loading OBELiX Excel file with ionic conductivity data
2. Matching CIF files to ionic conductivity values
3. Creating proper data structure for enhanced CGCNN training
4. Filtering out placeholder values (conductivity <= 1e-12)
5. Outputting data in format expected by enhanced_cgcnn_ionic_conductivity.py

Based on patterns from genetic_algo/fully_optimized_predictor.py and enhanced_cgcnn_ionic_conductivity.py
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import logging
from datetime import datetime
from collections import defaultdict
import shutil

# Scientific computing imports
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser

# Add CDVAE to path for data utilities
sys.path.append(str(Path(__file__).parent / "generator" / "CDVAE"))
try:
    from cdvae.common.data_utils import (
        preprocess, build_crystal, build_crystal_graph
    )
    CDVAE_AVAILABLE = True
except ImportError:
    print("Warning: CDVAE not available. CIF processing will be limited.")
    CDVAE_AVAILABLE = False


class OBELiXDataPreparator:
    """
    Comprehensive OBELiX dataset preparation for Enhanced CGCNN training
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Data storage
        self.obelix_data = []
        self.cif_metadata = {}
        self.matched_pairs = []
        self.processed_data = []
        
        # Statistics
        self.stats = {
            'total_obelix_entries': 0,
            'valid_conductivity_entries': 0,
            'total_cif_files': 0,
            'matched_pairs': 0,
            'filtered_pairs': 0,
            'final_dataset_size': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = Path(self.config.get('output_dir', 'outputs')) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger('obelix_prep')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_dir / f'obelix_prep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def load_obelix_excel(self, excel_path: str) -> pd.DataFrame:
        """
        Load and preprocess OBELiX Excel file
        
        Args:
            excel_path: Path to OBELiX Excel file
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info(f"Loading OBELiX dataset from {excel_path}")
        
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"OBELiX Excel file not found: {excel_path}")
        
        # Load Excel file
        df = pd.read_excel(excel_path)
        self.stats['total_obelix_entries'] = len(df)
        
        self.logger.info(f"Loaded {len(df)} entries from OBELiX dataset")
        self.logger.info(f"Columns: {list(df.columns)}")
        
        # Normalize ID column
        df['ID'] = df['ID'].astype(str).str.strip().str.lower()
        
        # Handle ionic conductivity column
        cond_col = self.config.get('conductivity_column', 'Ionic conductivity (S cm-1)')
        if cond_col not in df.columns:
            # Try to find similar column names
            possible_cols = [col for col in df.columns if 'conductivity' in col.lower()]
            if possible_cols:
                cond_col = possible_cols[0]
                self.logger.warning(f"Using column '{cond_col}' for ionic conductivity")
            else:
                raise ValueError(f"Ionic conductivity column not found. Available columns: {list(df.columns)}")
        
        # Convert to numeric and drop NaNs
        df[cond_col] = pd.to_numeric(df[cond_col], errors='coerce')
        initial_count = len(df)
        df = df.dropna(subset=[cond_col])
        self.stats['valid_conductivity_entries'] = len(df)
        
        self.logger.info(f"Dropped {initial_count - len(df)} entries with invalid conductivity values")
        
        # Filter out placeholder values
        conductivity_threshold = self.config.get('conductivity_threshold', 1e-12)
        initial_count = len(df)
        df = df[df[cond_col] > conductivity_threshold]
        filtered_count = len(df)
        
        self.logger.info(f"Filtered out {initial_count - filtered_count} entries with conductivity <= {conductivity_threshold}")
        self.logger.info(f"Remaining entries: {filtered_count}")
        
        # Normalize composition using pymatgen
        if 'Composition' in df.columns:
            df['Norm_Composition'] = df['Composition'].apply(
                lambda x: Composition(x).reduced_formula if pd.notna(x) else ""
            )
        
        # Normalize space group number
        if 'Space group number' in df.columns:
            df['Space group number'] = pd.to_numeric(df['Space group number'], errors='coerce').fillna(0).astype(int)
        
        # Convert to list of dictionaries for easier processing
        self.obelix_data = []
        for _, row in df.iterrows():
            entry = {
                'ID': row['ID'],
                'IonicConductivity': row[cond_col],
                'Composition': row.get('Composition', ''),
                'Norm_Composition': row.get('Norm_Composition', ''),
                'SpaceGroup': row.get('Space group number', 0),
                'raw_data': row.to_dict()
            }
            self.obelix_data.append(entry)
        
        return df
    
    def parse_cif_files(self, cif_folder: str) -> Dict[str, Dict]:
        """
        Parse CIF files and extract metadata
        
        Args:
            cif_folder: Path to folder containing CIF files
            
        Returns:
            Dictionary mapping CIF IDs to metadata
        """
        self.logger.info(f"Parsing CIF files in {cif_folder}")
        
        if not os.path.exists(cif_folder):
            raise FileNotFoundError(f"CIF folder not found: {cif_folder}")
        
        cif_files = [f for f in os.listdir(cif_folder) if f.lower().endswith('.cif')]
        self.stats['total_cif_files'] = len(cif_files)
        
        self.logger.info(f"Found {len(cif_files)} CIF files")
        
        self.cif_metadata = {}
        failed_count = 0
        
        for cif_file in cif_files:
            cif_path = os.path.join(cif_folder, cif_file)
            cif_id = os.path.splitext(cif_file)[0].lower()
            
            try:
                # Parse CIF file
                struct = Structure.from_file(cif_path)
                sg_analyzer = SpacegroupAnalyzer(struct)
                sg_num = sg_analyzer.get_space_group_number()
                comp = struct.composition.reduced_formula
                
                # Read CIF content for enhanced CGCNN
                with open(cif_path, 'r') as f:
                    cif_content = f.read()
                
                self.cif_metadata[cif_id] = {
                    'composition': comp,
                    'spacegroup': sg_num,
                    'filepath': cif_path,
                    'structure': struct,
                    'cif_content': cif_content,
                    'filename': cif_file
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to parse {cif_file}: {e}")
                failed_count += 1
                continue
        
        self.logger.info(f"Successfully parsed {len(self.cif_metadata)} CIF files")
        if failed_count > 0:
            self.logger.warning(f"Failed to parse {failed_count} CIF files")
        
        return self.cif_metadata
    
    def match_cifs_to_obelix(self, matching_strategy: str = 'composition_spacegroup') -> List[Tuple]:
        """
        Match CIF files to OBELiX entries
        
        Args:
            matching_strategy: Strategy for matching ('id', 'composition_spacegroup', 'composition_only')
            
        Returns:
            List of matched pairs (cif_id, obelix_entry, ionic_conductivity)
        """
        self.logger.info(f"Matching CIF files to OBELiX entries using strategy: {matching_strategy}")
        
        self.matched_pairs = []
        
        if matching_strategy == 'id':
            # Direct ID matching
            obelix_ids = {entry['ID']: entry for entry in self.obelix_data}
            
            for cif_id in self.cif_metadata:
                if cif_id in obelix_ids:
                    entry = obelix_ids[cif_id]
                    self.matched_pairs.append((cif_id, entry, entry['IonicConductivity']))
        
        elif matching_strategy == 'composition_spacegroup':
            # Match by composition and space group
            obelix_index = defaultdict(list)
            for entry in self.obelix_data:
                key = (entry['SpaceGroup'], entry['Norm_Composition'])
                obelix_index[key].append(entry)
            
            for cif_id, meta in self.cif_metadata.items():
                matched = False
                key = (meta['spacegroup'], meta['composition'])
                candidates = obelix_index.get(key, [])
                
                if candidates:
                    # If multiple candidates, pick first one with close composition
                    for candidate in candidates:
                        if self._compositions_close(meta['composition'], candidate['Norm_Composition']):
                            self.matched_pairs.append((cif_id, candidate, candidate['IonicConductivity']))
                            matched = True
                            break
                
                if not matched:
                    # Looser matching: same spacegroup, close composition
                    possible_candidates = [e for e in self.obelix_data if e['SpaceGroup'] == meta['spacegroup']]
                    for candidate in possible_candidates:
                        if self._compositions_close(meta['composition'], candidate['Norm_Composition']):
                            self.matched_pairs.append((cif_id, candidate, candidate['IonicConductivity']))
                            matched = True
                            break
        
        elif matching_strategy == 'composition_only':
            # Match by composition only
            for cif_id, meta in self.cif_metadata.items():
                for entry in self.obelix_data:
                    if self._compositions_close(meta['composition'], entry['Norm_Composition']):
                        self.matched_pairs.append((cif_id, entry, entry['IonicConductivity']))
                        break
        
        self.stats['matched_pairs'] = len(self.matched_pairs)
        self.logger.info(f"Matched {len(self.matched_pairs)} CIF-OBELiX pairs")
        
        return self.matched_pairs
    
    def _compositions_close(self, comp_a: str, comp_b: str, tolerance: float = 0.05) -> bool:
        """
        Check if two compositions are close by comparing atomic fractions
        
        Args:
            comp_a: First composition string
            comp_b: Second composition string
            tolerance: Tolerance for atomic fraction differences
            
        Returns:
            True if compositions are close
        """
        try:
            a = Composition(comp_a)
            b = Composition(comp_b)
        except Exception:
            return False
        
        elements = set(a.elements) | set(b.elements)
        for el in elements:
            frac_a = a.get_atomic_fraction(el)
            frac_b = b.get_atomic_fraction(el)
            if abs(frac_a - frac_b) > tolerance:
                return False
        return True
    
    def create_enhanced_cgcnn_dataset(self) -> List[Dict]:
        """
        Create dataset in format expected by enhanced_cgcnn_ionic_conductivity.py
        
        Returns:
            List of data dictionaries
        """
        self.logger.info("Creating Enhanced CGCNN dataset format")
        
        self.processed_data = []
        failed_count = 0
        
        for cif_id, obelix_entry, conductivity in self.matched_pairs:
            try:
                cif_meta = self.cif_metadata[cif_id]
                
                # Create data entry for enhanced CGCNN
                data_entry = {
                    'material_id': cif_id,
                    'cif': cif_meta['cif_content'],
                    'ionic_conductivity': float(conductivity),
                    'composition': cif_meta['composition'],
                    'spacegroup': cif_meta['spacegroup'],
                    'structure_data': {
                        'filepath': cif_meta['filepath'],
                        'filename': cif_meta['filename']
                    },
                    'obelix_data': obelix_entry['raw_data']
                }
                
                # Add crystal graph processing if CDVAE is available
                if CDVAE_AVAILABLE:
                    try:
                        crystal = build_crystal(cif_meta['cif_content'], niggli=True, primitive=False)
                        graph_arrays = build_crystal_graph(crystal, graph_method='crystalnn')
                        data_entry['graph_arrays'] = graph_arrays
                        data_entry['crystal'] = crystal
                    except Exception as e:
                        self.logger.warning(f"Failed to build crystal graph for {cif_id}: {e}")
                        data_entry['graph_arrays'] = None
                        data_entry['crystal'] = None
                
                self.processed_data.append(data_entry)
                
            except Exception as e:
                self.logger.error(f"Failed to process {cif_id}: {e}")
                failed_count += 1
                continue
        
        self.stats['final_dataset_size'] = len(self.processed_data)
        self.logger.info(f"Created {len(self.processed_data)} data entries for Enhanced CGCNN")
        
        if failed_count > 0:
            self.logger.warning(f"Failed to process {failed_count} entries")
        
        return self.processed_data
    
    def save_dataset(self, output_path: str, format: str = 'pickle') -> None:
        """
        Save processed dataset
        
        Args:
            output_path: Output file path
            format: Output format ('pickle', 'csv', 'json')
        """
        self.logger.info(f"Saving dataset to {output_path} in {format} format")
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(self.processed_data, f)
        
        elif format == 'csv':
            # Create CSV with basic information
            csv_data = []
            for entry in self.processed_data:
                csv_entry = {
                    'material_id': entry['material_id'],
                    'ionic_conductivity': entry['ionic_conductivity'],
                    'composition': entry['composition'],
                    'spacegroup': entry['spacegroup'],
                    'cif_path': entry['structure_data']['filepath']
                }
                csv_data.append(csv_entry)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
        
        elif format == 'json':
            # Convert to JSON-serializable format
            json_data = []
            for entry in self.processed_data:
                json_entry = {
                    'material_id': entry['material_id'],
                    'cif': entry['cif'],
                    'ionic_conductivity': entry['ionic_conductivity'],
                    'composition': entry['composition'],
                    'spacegroup': entry['spacegroup'],
                    'structure_data': entry['structure_data'],
                    'obelix_data': entry['obelix_data']
                }
                json_data.append(json_entry)
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        self.logger.info(f"Dataset saved successfully to {output_path}")
    
    def save_cgcnn_format(self, output_dir: str) -> None:
        """
        Save dataset in traditional CGCNN format (CIF files + id_prop.csv)
        
        Args:
            output_dir: Output directory
        """
        self.logger.info(f"Saving dataset in CGCNN format to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        cifs_dir = os.path.join(output_dir, 'cifs')
        os.makedirs(cifs_dir, exist_ok=True)
        
        # Copy CIF files and create id_prop.csv
        id_prop_data = []
        
        for entry in self.processed_data:
            material_id = entry['material_id']
            conductivity = entry['ionic_conductivity']
            
            # Copy CIF file
            src_path = entry['structure_data']['filepath']
            dst_path = os.path.join(cifs_dir, f"{material_id}.cif")
            shutil.copy2(src_path, dst_path)
            
            # Add to id_prop data
            id_prop_data.append([material_id, conductivity])
        
        # Save id_prop.csv
        id_prop_df = pd.DataFrame(id_prop_data, columns=['material_id', 'ionic_conductivity'])
        id_prop_df.to_csv(os.path.join(cifs_dir, 'id_prop.csv'), index=False, header=False)
        
        # Copy atom_init.json if available
        atom_init_paths = [
            'generator/CDVAE/cdvae/pl_data/atom_init.json',
            'env/property_predictions/cgcnn_pretrained/atom_init.json'
        ]
        
        for atom_init_path in atom_init_paths:
            if os.path.exists(atom_init_path):
                shutil.copy2(atom_init_path, os.path.join(cifs_dir, 'atom_init.json'))
                break
        
        self.logger.info(f"CGCNN format dataset saved to {output_dir}")
    
    def print_statistics(self) -> None:
        """Print dataset preparation statistics"""
        self.logger.info("Dataset Preparation Statistics:")
        self.logger.info("=" * 50)
        
        for key, value in self.stats.items():
            self.logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        if self.stats['total_obelix_entries'] > 0:
            match_rate = (self.stats['matched_pairs'] / self.stats['total_obelix_entries']) * 100
            self.logger.info(f"Match Rate: {match_rate:.2f}%")
        
        if self.stats['matched_pairs'] > 0:
            success_rate = (self.stats['final_dataset_size'] / self.stats['matched_pairs']) * 100
            self.logger.info(f"Processing Success Rate: {success_rate:.2f}%")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Prepare OBELiX dataset for Enhanced CGCNN training')
    
    # Input arguments
    parser.add_argument('--obelix_excel', type=str, required=True,
                       help='Path to OBELiX Excel file')
    parser.add_argument('--cif_folder', type=str, required=True,
                       help='Path to folder containing CIF files')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='data/obelix_prepared',
                       help='Output directory for prepared dataset')
    parser.add_argument('--output_format', type=str, choices=['pickle', 'csv', 'json'], default='pickle',
                       help='Output format for dataset')
    
    # Processing arguments
    parser.add_argument('--matching_strategy', type=str, 
                       choices=['id', 'composition_spacegroup', 'composition_only'],
                       default='composition_spacegroup',
                       help='Strategy for matching CIF files to OBELiX entries')
    parser.add_argument('--conductivity_threshold', type=float, default=1e-12,
                       help='Minimum conductivity threshold for filtering')
    parser.add_argument('--conductivity_column', type=str, default='Ionic conductivity (S cm-1)',
                       help='Name of ionic conductivity column in Excel file')
    
    # Additional options
    parser.add_argument('--save_cgcnn_format', action='store_true',
                       help='Also save in traditional CGCNN format')
    parser.add_argument('--config_file', type=str,
                       help='Path to JSON configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'output_dir': args.output_dir,
        'conductivity_threshold': args.conductivity_threshold,
        'conductivity_column': args.conductivity_column,
        'matching_strategy': args.matching_strategy
    }
    
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Initialize preparator
    preparator = OBELiXDataPreparator(config)
    
    try:
        # Load OBELiX Excel data
        preparator.load_obelix_excel(args.obelix_excel)
        
        # Parse CIF files
        preparator.parse_cif_files(args.cif_folder)
        
        # Match CIFs to OBELiX entries
        preparator.match_cifs_to_obelix(args.matching_strategy)
        
        # Create Enhanced CGCNN dataset
        preparator.create_enhanced_cgcnn_dataset()
        
        # Save dataset
        output_path = os.path.join(args.output_dir, f'obelix_dataset.{args.output_format}')
        preparator.save_dataset(output_path, args.output_format)
        
        # Save in CGCNN format if requested
        if args.save_cgcnn_format:
            cgcnn_dir = os.path.join(args.output_dir, 'cgcnn_format')
            preparator.save_cgcnn_format(cgcnn_dir)
        
        # Print statistics
        preparator.print_statistics()
        
        # Save configuration and statistics
        with open(os.path.join(args.output_dir, 'preparation_config.json'), 'w') as f:
            json.dump({
                'config': config,
                'statistics': preparator.stats,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n✅ Dataset preparation completed successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📊 Final dataset size: {preparator.stats['final_dataset_size']} entries")
        
    except Exception as e:
        preparator.logger.error(f"Dataset preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()