#!/usr/bin/env python3
"""
Synthetic High-Fidelity Bandgap Data Creator
Creates synthetic HSE/experimental bandgap data using known correction methods
when high-fidelity data is not available in Materials Project
"""

import logging
import pandas as pd
import numpy as np
from mp_api.client import MPRester
from typing import Dict, List, Tuple, Optional
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BandgapCorrector:
    """Apply known correction methods to PBE bandgaps to estimate HSE values"""
    
    def __init__(self):
        # Known correction factors from literature
        self.correction_methods = {
            'linear_universal': self._linear_universal_correction,
            'material_class_specific': self._material_class_correction,
            'composition_dependent': self._composition_dependent_correction,
            'structure_dependent': self._structure_dependent_correction
        }
        
        # Material class correction factors (from literature studies)
        self.class_corrections = {
            'oxide': {'slope': 1.35, 'intercept': 0.8},
            'chalcogenide': {'slope': 1.25, 'intercept': 0.6},
            'halide': {'slope': 1.40, 'intercept': 0.9},
            'nitride': {'slope': 1.30, 'intercept': 0.7},
            'carbide': {'slope': 1.20, 'intercept': 0.5},
            'perovskite': {'slope': 1.45, 'intercept': 1.0},
            'semiconductor': {'slope': 1.25, 'intercept': 0.4},
            'default': {'slope': 1.30, 'intercept': 0.7}
        }
    
    def _linear_universal_correction(self, pbe_gap: float, **kwargs) -> float:
        """Universal linear correction: HSE = 1.3 * PBE + 0.7"""
        return 1.3 * pbe_gap + 0.7
    
    def _material_class_correction(self, pbe_gap: float, material_class: str = 'default', **kwargs) -> float:
        """Material class-specific correction"""
        correction = self.class_corrections.get(material_class, self.class_corrections['default'])
        return correction['slope'] * pbe_gap + correction['intercept']
    
    def _composition_dependent_correction(self, pbe_gap: float, formula: str = '', **kwargs) -> float:
        """Composition-dependent correction based on elements present"""
        # Simple heuristics based on common elements
        correction_factor = 1.3
        intercept = 0.7
        
        if formula:
            formula_lower = formula.lower()
            # Adjust based on elements known to affect bandgap corrections
            if any(elem in formula_lower for elem in ['ti', 'zr', 'hf']):  # d-block metals
                correction_factor = 1.4
                intercept = 0.9
            elif any(elem in formula_lower for elem in ['pb', 'sn', 'bi']):  # Heavy p-block
                correction_factor = 1.2
                intercept = 0.5
            elif any(elem in formula_lower for elem in ['f', 'cl', 'br', 'i']):  # Halides
                correction_factor = 1.4
                intercept = 0.9
        
        return correction_factor * pbe_gap + intercept
    
    def _structure_dependent_correction(self, pbe_gap: float, space_group: int = 1, **kwargs) -> float:
        """Structure-dependent correction based on crystal system"""
        # Simple correction based on crystal system symmetry
        correction_factor = 1.3
        intercept = 0.7
        
        # Cubic systems (high symmetry) often have different corrections
        if 195 <= space_group <= 230:  # Cubic
            correction_factor = 1.35
            intercept = 0.8
        elif 143 <= space_group <= 194:  # Hexagonal/Trigonal
            correction_factor = 1.25
            intercept = 0.6
        elif 75 <= space_group <= 142:  # Tetragonal
            correction_factor = 1.30
            intercept = 0.7
        
        return correction_factor * pbe_gap + intercept
    
    def classify_material(self, formula: str) -> str:
        """Classify material based on chemical formula"""
        formula_lower = formula.lower()
        
        # Simple classification rules
        if 'o' in formula_lower and any(metal in formula_lower for metal in ['ti', 'zr', 'al', 'mg', 'ca', 'sr', 'ba']):
            if any(elem in formula_lower for elem in ['ba', 'sr', 'ca']) and 'ti' in formula_lower:
                return 'perovskite'
            return 'oxide'
        elif any(chalc in formula_lower for chalc in ['s', 'se', 'te']):
            return 'chalcogenide'
        elif any(hal in formula_lower for hal in ['f', 'cl', 'br', 'i']):
            return 'halide'
        elif 'n' in formula_lower:
            return 'nitride'
        elif 'c' in formula_lower and not any(elem in formula_lower for elem in ['cl', 'ca', 'cs', 'cr']):
            return 'carbide'
        elif any(semi in formula_lower for semi in ['si', 'ge', 'ga', 'as', 'in', 'sb']):
            return 'semiconductor'
        else:
            return 'default'
    
    def apply_corrections(self, pbe_gap: float, formula: str = '', space_group: int = 1) -> Dict[str, float]:
        """Apply all correction methods and return results"""
        material_class = self.classify_material(formula)
        
        corrections = {}
        for method_name, method_func in self.correction_methods.items():
            try:
                if method_name == 'material_class_specific':
                    corrected_gap = method_func(pbe_gap, material_class=material_class)
                elif method_name == 'composition_dependent':
                    corrected_gap = method_func(pbe_gap, formula=formula)
                elif method_name == 'structure_dependent':
                    corrected_gap = method_func(pbe_gap, space_group=space_group)
                else:
                    corrected_gap = method_func(pbe_gap)
                
                corrections[method_name] = max(0.0, corrected_gap)  # Ensure non-negative
            except Exception as e:
                logger.warning(f"Error applying {method_name}: {e}")
                corrections[method_name] = pbe_gap
        
        return corrections

class SyntheticDataCreator:
    """Create synthetic high-fidelity bandgap dataset"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.mpr = MPRester(api_key)
        self.corrector = BandgapCorrector()
    
    def create_synthetic_dataset(self, max_materials: int = 1000) -> pd.DataFrame:
        """Create synthetic HSE dataset from PBE data"""
        logger.info(f"Creating synthetic dataset with up to {max_materials} materials...")
        
        # Get PBE data from Materials Project
        logger.info("Fetching PBE bandgap data from Materials Project...")
        materials = self.mpr.materials.summary.search(
            band_gap=(0.1, 8.0),  # Focus on semiconductors and insulators
            fields=["material_id", "formula_pretty", "band_gap", "symmetry"],
            num_chunks=10,
            chunk_size=min(500, max_materials // 10)
        )
        
        logger.info(f"Found {len(materials)} materials with PBE bandgaps")
        
        # Create synthetic data
        synthetic_data = []
        
        for i, material in enumerate(materials[:max_materials]):
            if i % 100 == 0:
                logger.info(f"Processing material {i+1}/{min(len(materials), max_materials)}")
            
            try:
                # Get space group if available
                space_group = 1
                if hasattr(material, 'symmetry') and material.symmetry:
                    space_group = getattr(material.symmetry, 'number', 1)
                
                # Apply corrections
                corrections = self.corrector.apply_corrections(
                    material.band_gap,
                    material.formula_pretty,
                    space_group
                )
                
                # Add noise to make it more realistic
                noise_factor = 0.1  # 10% noise
                for method, corrected_gap in corrections.items():
                    noise = np.random.normal(0, noise_factor * corrected_gap)
                    corrections[method] = max(0.0, corrected_gap + noise)
                
                # Create data entry
                entry = {
                    'material_id': str(material.material_id),
                    'formula': material.formula_pretty,
                    'pbe_bandgap': material.band_gap,
                    'material_class': self.corrector.classify_material(material.formula_pretty),
                    'space_group': space_group,
                    **{f'synthetic_hse_{method}': gap for method, gap in corrections.items()},
                    'synthetic_hse_average': np.mean(list(corrections.values())),
                    'synthetic_hse_std': np.std(list(corrections.values()))
                }
                
                synthetic_data.append(entry)
                
            except Exception as e:
                logger.warning(f"Error processing {material.material_id}: {e}")
                continue
        
        df = pd.DataFrame(synthetic_data)
        logger.info(f"Created synthetic dataset with {len(df)} materials")
        
        return df
    
    def add_experimental_estimates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add experimental bandgap estimates based on literature correlations"""
        logger.info("Adding experimental bandgap estimates...")
        
        # Simple experimental estimates (these would ideally come from literature data)
        df['synthetic_experimental'] = df.apply(self._estimate_experimental_gap, axis=1)
        
        return df
    
    def _estimate_experimental_gap(self, row) -> float:
        """Estimate experimental bandgap from synthetic HSE"""
        # Experimental values are often close to HSE but with some systematic differences
        hse_avg = row['synthetic_hse_average']
        
        # Add some material-class specific adjustments
        material_class = row['material_class']
        
        if material_class == 'oxide':
            # Oxides often have experimental gaps slightly lower than HSE
            exp_gap = hse_avg * 0.95 + np.random.normal(0, 0.1)
        elif material_class == 'semiconductor':
            # Semiconductors are usually well-matched to HSE
            exp_gap = hse_avg * 1.02 + np.random.normal(0, 0.05)
        elif material_class == 'halide':
            # Halides can have larger experimental gaps
            exp_gap = hse_avg * 1.05 + np.random.normal(0, 0.15)
        else:
            # Default case
            exp_gap = hse_avg * 0.98 + np.random.normal(0, 0.1)
        
        return max(0.0, exp_gap)
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'synthetic_bandgap_dataset.csv'):
        """Save the synthetic dataset"""
        df.to_csv(filename, index=False)
        logger.info(f"Saved synthetic dataset to {filename}")
        
        # Print summary statistics
        logger.info("=== Dataset Summary ===")
        logger.info(f"Total materials: {len(df)}")
        logger.info(f"PBE bandgap range: {df['pbe_bandgap'].min():.3f} - {df['pbe_bandgap'].max():.3f} eV")
        logger.info(f"Synthetic HSE range: {df['synthetic_hse_average'].min():.3f} - {df['synthetic_hse_average'].max():.3f} eV")
        logger.info(f"Material classes: {df['material_class'].value_counts().to_dict()}")
        
        # Show correlation between PBE and synthetic HSE
        correlation = df['pbe_bandgap'].corr(df['synthetic_hse_average'])
        logger.info(f"PBE-HSE correlation: {correlation:.3f}")

def main():
    """Main function to create synthetic dataset"""
    API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"
    
    logger.info("Creating synthetic high-fidelity bandgap dataset...")
    
    creator = SyntheticDataCreator(API_KEY)
    
    try:
        # Create synthetic dataset
        df = creator.create_synthetic_dataset(max_materials=1000)
        
        # Add experimental estimates
        df = creator.add_experimental_estimates(df)
        
        # Save dataset
        creator.save_dataset(df)
        
        logger.info("Synthetic dataset creation completed successfully!")
        logger.info("This dataset can be used for ML training when real HSE data is not available.")
        
    except Exception as e:
        logger.error(f"Error creating synthetic dataset: {e}")
        raise

if __name__ == "__main__":
    main()