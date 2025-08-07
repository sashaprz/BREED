import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

class CEIProperty(Enum):
    IONIC_CONDUCTIVITY = "ionic_conductivity"
    ELECTRONIC_INSULATION = "electronic_insulation"
    MECHANICAL_STABILITY = "mechanical_stability"
    CHEMICAL_STABILITY = "chemical_stability"
    OXIDATIVE_STABILITY = "oxidative_stability"  # Critical for high-voltage cathode operation
    CYCLING_STABILITY = "cycling_stability"
    INTERFACIAL_ADHESION = "interfacial_adhesion"  # Important for cathode contact

@dataclass
class CEIComponent:
    """Represents a single CEI component with its properties"""
    formula: str
    source_element: str
    properties: Dict[CEIProperty, float]  # Scores from 0-1
    weight_fraction: float = 0.0

class CEIPredictor:
    """Predicts CEI composition and properties from SSE oxidation at cathode interface"""
    
    def __init__(self):
        self.decomposition_database = self._initialize_database()
        self.property_weights = {
            CEIProperty.IONIC_CONDUCTIVITY: 0.20,
            CEIProperty.ELECTRONIC_INSULATION: 0.15,  # Less critical than SEI
            CEIProperty.MECHANICAL_STABILITY: 0.15,
            CEIProperty.CHEMICAL_STABILITY: 0.15,
            CEIProperty.OXIDATIVE_STABILITY: 0.25,  # Most critical for CEI
            CEIProperty.CYCLING_STABILITY: 0.05,
            CEIProperty.INTERFACIAL_ADHESION: 0.05
        }
    
    def _initialize_database(self) -> Dict:
        """Database with SSE oxidation products at cathode interface (oxidizing environment)"""
        return {
            'S': {
                'products': ['Li2SO4', 'SO2'],  # Oxidation products of sulfides
                'properties': {
                    'Li2SO4': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.2,
                        CEIProperty.ELECTRONIC_INSULATION: 0.85,
                        CEIProperty.MECHANICAL_STABILITY: 0.75,
                        CEIProperty.CHEMICAL_STABILITY: 0.9,
                        CEIProperty.OXIDATIVE_STABILITY: 0.95,  # Sulfate is already oxidized
                        CEIProperty.CYCLING_STABILITY: 0.8,
                        CEIProperty.INTERFACIAL_ADHESION: 0.7
                    },
                    'SO2': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.1,  # Gas, poor conductor
                        CEIProperty.ELECTRONIC_INSULATION: 0.9,
                        CEIProperty.MECHANICAL_STABILITY: 0.2,  # Gas phase
                        CEIProperty.CHEMICAL_STABILITY: 0.4,
                        CEIProperty.OXIDATIVE_STABILITY: 0.7,
                        CEIProperty.CYCLING_STABILITY: 0.3,
                        CEIProperty.INTERFACIAL_ADHESION: 0.1  # Gas doesn't adhere
                    }
                }
            },
            'P': {
                'products': ['Li3PO4', 'LiPO3'],  # Phosphate oxidation products
                'properties': {
                    'Li3PO4': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.4,
                        CEIProperty.ELECTRONIC_INSULATION: 0.9,
                        CEIProperty.MECHANICAL_STABILITY: 0.9,
                        CEIProperty.CHEMICAL_STABILITY: 0.9,
                        CEIProperty.OXIDATIVE_STABILITY: 0.98,  # Very stable phosphate
                        CEIProperty.CYCLING_STABILITY: 0.9,
                        CEIProperty.INTERFACIAL_ADHESION: 0.8
                    },
                    'LiPO3': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.35,
                        CEIProperty.ELECTRONIC_INSULATION: 0.85,
                        CEIProperty.MECHANICAL_STABILITY: 0.8,
                        CEIProperty.CHEMICAL_STABILITY: 0.85,
                        CEIProperty.OXIDATIVE_STABILITY: 0.92,
                        CEIProperty.CYCLING_STABILITY: 0.8,
                        CEIProperty.INTERFACIAL_ADHESION: 0.75
                    }
                }
            },
            'O': {
                'products': [ 'O2'],  # Oxide decomposition under oxidizing conditions
                'properties': {
                    'O2': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.05,  # Gas, poor conductor
                        CEIProperty.ELECTRONIC_INSULATION: 0.95,
                        CEIProperty.MECHANICAL_STABILITY: 0.1,  # Gas phase
                        CEIProperty.CHEMICAL_STABILITY: 0.3,
                        CEIProperty.OXIDATIVE_STABILITY: 0.5,  # Reactive oxygen
                        CEIProperty.CYCLING_STABILITY: 0.2,
                        CEIProperty.INTERFACIAL_ADHESION: 0.05
                    }
                }
            },
            'F': {
                'products': ['LiF'],  # Fluoride is already highly oxidized
                'properties': {
                    'LiF': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.25,
                        CEIProperty.ELECTRONIC_INSULATION: 1.0,
                        CEIProperty.MECHANICAL_STABILITY: 0.7,
                        CEIProperty.CHEMICAL_STABILITY: 1.0,
                        CEIProperty.OXIDATIVE_STABILITY: 1.0,  # Extremely stable
                        CEIProperty.CYCLING_STABILITY: 0.95,
                        CEIProperty.INTERFACIAL_ADHESION: 0.6  # Can be brittle but stable
                    }
                }
            },
            'Cl': {
                'products': ['LiCl', 'Cl2'],  # Chloride oxidation products
                'properties': {
                    'LiCl': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.7,
                        CEIProperty.ELECTRONIC_INSULATION: 0.6,
                        CEIProperty.MECHANICAL_STABILITY: 0.5,
                        CEIProperty.CHEMICAL_STABILITY: 0.6,
                        CEIProperty.OXIDATIVE_STABILITY: 0.4,  # Can be oxidized to Cl2
                        CEIProperty.CYCLING_STABILITY: 0.6,
                        CEIProperty.INTERFACIAL_ADHESION: 0.55
                    },
                    'Cl2': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.05,  # Gas
                        CEIProperty.ELECTRONIC_INSULATION: 0.9,
                        CEIProperty.MECHANICAL_STABILITY: 0.1,  # Gas phase
                        CEIProperty.CHEMICAL_STABILITY: 0.2,
                        CEIProperty.OXIDATIVE_STABILITY: 0.8,  # Already oxidized
                        CEIProperty.CYCLING_STABILITY: 0.2,
                        CEIProperty.INTERFACIAL_ADHESION: 0.05
                    }
                }
            },
            'Br': {
                'products': ['LiBr', 'Br2'],  # Bromide oxidation
                'properties': {
                    'LiBr': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.75,
                        CEIProperty.ELECTRONIC_INSULATION: 0.5,
                        CEIProperty.MECHANICAL_STABILITY: 0.4,
                        CEIProperty.CHEMICAL_STABILITY: 0.5,
                        CEIProperty.OXIDATIVE_STABILITY: 0.3,  # Easily oxidized
                        CEIProperty.CYCLING_STABILITY: 0.5,
                        CEIProperty.INTERFACIAL_ADHESION: 0.45
                    },
                    'Br2': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.02,
                        CEIProperty.ELECTRONIC_INSULATION: 0.8,
                        CEIProperty.MECHANICAL_STABILITY: 0.1,
                        CEIProperty.CHEMICAL_STABILITY: 0.2,
                        CEIProperty.OXIDATIVE_STABILITY: 0.9,
                        CEIProperty.CYCLING_STABILITY: 0.1,
                        CEIProperty.INTERFACIAL_ADHESION: 0.02
                    }
                }
            },
            'I': {
                'products': ['LiI', 'I2'],  # Iodide oxidation
                'properties': {
                    'LiI': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.85,
                        CEIProperty.ELECTRONIC_INSULATION: 0.4,
                        CEIProperty.MECHANICAL_STABILITY: 0.3,
                        CEIProperty.CHEMICAL_STABILITY: 0.4,
                        CEIProperty.OXIDATIVE_STABILITY: 0.2,  # Very easily oxidized
                        CEIProperty.CYCLING_STABILITY: 0.4,
                        CEIProperty.INTERFACIAL_ADHESION: 0.35
                    },
                    'I2': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.01,
                        CEIProperty.ELECTRONIC_INSULATION: 0.7,
                        CEIProperty.MECHANICAL_STABILITY: 0.2,
                        CEIProperty.CHEMICAL_STABILITY: 0.3,
                        CEIProperty.OXIDATIVE_STABILITY: 0.95,  # Already oxidized
                        CEIProperty.CYCLING_STABILITY: 0.2,
                        CEIProperty.INTERFACIAL_ADHESION: 0.1
                    }
                }
            },
            # Metal oxide SSE components - these can form higher oxidation states
            'Ti': {
                'products': ['TiO2'],  # Already in highest common oxidation state
                'properties': {
                    'TiO2': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.15,
                        CEIProperty.ELECTRONIC_INSULATION: 0.9,
                        CEIProperty.MECHANICAL_STABILITY: 0.9,
                        CEIProperty.CHEMICAL_STABILITY: 0.9,
                        CEIProperty.OXIDATIVE_STABILITY: 0.98,  # Very stable
                        CEIProperty.CYCLING_STABILITY: 0.85,
                        CEIProperty.INTERFACIAL_ADHESION: 0.8
                    }
                }
            },
            'Zr': {
                'products': ['ZrO2'],  # Highest oxidation state
                'properties': {
                    'ZrO2': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.15,
                        CEIProperty.ELECTRONIC_INSULATION: 0.9,
                        CEIProperty.MECHANICAL_STABILITY: 0.9,
                        CEIProperty.CHEMICAL_STABILITY: 0.9,
                        CEIProperty.OXIDATIVE_STABILITY: 0.99,  # Extremely stable
                        CEIProperty.CYCLING_STABILITY: 0.85,
                        CEIProperty.INTERFACIAL_ADHESION: 0.75
                    }
                }
            },
            'Al': {
                'products': ['Al2O3'],  # Highest oxidation state
                'properties': {
                    'Al2O3': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.05,
                        CEIProperty.ELECTRONIC_INSULATION: 1.0,
                        CEIProperty.MECHANICAL_STABILITY: 0.9,
                        CEIProperty.CHEMICAL_STABILITY: 1.0,
                        CEIProperty.OXIDATIVE_STABILITY: 1.0,  # Extremely stable
                        CEIProperty.CYCLING_STABILITY: 0.95,
                        CEIProperty.INTERFACIAL_ADHESION: 0.7
                    }
                }
            },
            # Rare earth elements (common in garnets)
            'La': {
                'products': ['La2O3'],  # Stable oxide
                'properties': {
                    'La2O3': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.25,
                        CEIProperty.ELECTRONIC_INSULATION: 0.9,
                        CEIProperty.MECHANICAL_STABILITY: 0.85,
                        CEIProperty.CHEMICAL_STABILITY: 0.9,
                        CEIProperty.OXIDATIVE_STABILITY: 0.95,
                        CEIProperty.CYCLING_STABILITY: 0.85,
                        CEIProperty.INTERFACIAL_ADHESION: 0.8
                    }
                }
            },
            # Carbon-containing SSEs
            'C': {
                'products': ['Li2CO3', 'CO2'],  # Carbonate formation and CO2 evolution
                'properties': {
                    'Li2CO3': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.45,
                        CEIProperty.ELECTRONIC_INSULATION: 0.85,
                        CEIProperty.MECHANICAL_STABILITY: 0.8,
                        CEIProperty.CHEMICAL_STABILITY: 0.8,
                        CEIProperty.OXIDATIVE_STABILITY: 0.7,  # Can decompose at high voltage
                        CEIProperty.CYCLING_STABILITY: 0.7,
                        CEIProperty.INTERFACIAL_ADHESION: 0.7
                    },
                    'CO2': {
                        CEIProperty.IONIC_CONDUCTIVITY: 0.02,  # Gas
                        CEIProperty.ELECTRONIC_INSULATION: 0.95,
                        CEIProperty.MECHANICAL_STABILITY: 0.1,
                        CEIProperty.CHEMICAL_STABILITY: 0.6,
                        CEIProperty.OXIDATIVE_STABILITY: 0.9,  # Already oxidized
                        CEIProperty.CYCLING_STABILITY: 0.3,
                        CEIProperty.INTERFACIAL_ADHESION: 0.05
                    }
                }
            }
        }
    
    def extract_composition_from_cif(self, cif_file_path: str) -> Dict[str, float]:
        """Extract elemental composition from CIF file"""
        try:
            from pymatgen.io.cif import CifParser
        except ImportError:
            raise ImportError("pymatgen is required for CIF parsing. Install with: pip install pymatgen")
        
        parser = CifParser(cif_file_path)
        structure = parser.parse_structures(primitive=True)[0]
        
        # Get composition and normalize to atomic fractions
        composition = structure.composition.fractional_composition
        return {str(element): float(fraction) for element, fraction in composition.items()}

    def predict_cei_composition(self, sse_composition: Dict[str, float]) -> List[CEIComponent]:
        """
        Predict CEI composition from SSE elemental composition based on oxidation chemistry
        
        Args:
            sse_composition: Dict of {element: atomic_fraction}
            
        Returns:
            List of CEIComponent objects representing oxidation products
        """
        cei_components = []
        
        for element, atomic_fraction in sse_composition.items():
            if element in self.decomposition_database:
                products = self.decomposition_database[element]['products']
                properties_db = self.decomposition_database[element]['properties']
                
                # Weight products based on thermodynamic favorability under oxidizing conditions
                # For now, using equal distribution but this could be improved with thermodynamic data
                fraction_per_product = atomic_fraction / len(products)
                
                for product in products:
                    if product in properties_db:
                        component = CEIComponent(
                            formula=product,
                            source_element=element,
                            properties=properties_db[product].copy(),
                            weight_fraction=fraction_per_product
                        )
                        cei_components.append(component)
        
        return cei_components
    
    def calculate_overall_properties(self, cei_components: List[CEIComponent]) -> Dict[CEIProperty, float]:
        """Calculate weighted average properties of the CEI"""
        overall_properties = {prop: 0.0 for prop in CEIProperty}
        total_weight = sum(comp.weight_fraction for comp in cei_components)
        
        if total_weight == 0:
            return overall_properties
        
        for component in cei_components:
            weight = component.weight_fraction / total_weight
            for prop, value in component.properties.items():
                overall_properties[prop] += weight * value
        
        return overall_properties
    
    def calculate_cei_score(self, overall_properties: Dict[CEIProperty, float]) -> float:
        """Calculate overall CEI performance score (0-1) with emphasis on oxidative stability"""
        weighted_score = 0.0
        for prop, value in overall_properties.items():
            weighted_score += self.property_weights[prop] * value
        return weighted_score
    
    def predict_from_cif(self, cif_file_path: str) -> Dict:
        """
        Main interface: Predict CEI properties from CIF file
        
        Args:
            cif_file_path: Path to CIF file
            
        Returns:
            Dict containing CEI analysis results with 'cei_score' key for RL agent
        """
        # Extract composition from CIF
        composition = self.extract_composition_from_cif(cif_file_path)
        
        # Predict composition based on oxidation chemistry
        cei_components = self.predict_cei_composition(composition)
        
        # Calculate properties
        overall_properties = self.calculate_overall_properties(cei_components)
        
        # Calculate overall score
        cei_score = self.calculate_cei_score(overall_properties)
        
        # Prepare results
        results = {
            'cei_components': [
                {
                    'formula': comp.formula,
                    'source_element': comp.source_element,
                    'weight_fraction': comp.weight_fraction,
                    'properties': {prop.value: value for prop, value in comp.properties.items()}
                }
                for comp in cei_components
            ],
            'overall_properties': {prop.value: value for prop, value in overall_properties.items()},
            'cei_score': cei_score,  # This is what your RL agent will use
            'input_composition': composition,
        }
        
        return results
    
    def predict_from_composition(self, composition: Dict[str, float]) -> Dict:
        """
        Alternative interface: Predict CEI properties from composition dict
        
        Args:
            composition: Dict of {element: atomic_fraction}
            
        Returns:
            Dict containing CEI analysis results
        """
        # Predict composition based on oxidation chemistry
        cei_components = self.predict_cei_composition(composition)
        
        # Calculate properties
        overall_properties = self.calculate_overall_properties(cei_components)
        
        # Calculate overall score
        cei_score = self.calculate_cei_score(overall_properties)
        
        # Prepare results
        results = {
            'cei_components': [
                {
                    'formula': comp.formula,
                    'source_element': comp.source_element,
                    'weight_fraction': comp.weight_fraction,
                    'properties': {prop.value: value for prop, value in comp.properties.items()}
                }
                for comp in cei_components
            ],
            'overall_properties': {prop.value: value for prop, value in overall_properties.items()},
            'cei_score': cei_score,
            'input_composition': composition,
        }
        
        return results

if __name__ == "__main__":
    predictor = CEIPredictor()
    
    # Example usage with composition dict (typical sulfide SSE)
    example_composition = {'Li': 0.15, 'P': 0.12, 'S': 0.48, 'Cl': 0.25}
    results = predictor.predict_from_composition(example_composition)
    
    print("Predicted CEI Score:", f"{results['cei_score']:.3f}")
    print("\nOverall CEI Properties:")
    for prop, value in results['overall_properties'].items():
        print(f" - {prop}: {value:.3f}")
    print("\nCEI Components (SSE oxidation products):")
    for comp in results['cei_components']:
        print(f" - {comp['formula']} (from {comp['source_element']}, wt frac: {comp['weight_fraction']:.3f})")