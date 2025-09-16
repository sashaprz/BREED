import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

class SEIProperty(Enum):
    IONIC_CONDUCTIVITY = "ionic_conductivity"
    ELECTRONIC_INSULATION = "electronic_insulation"
    MECHANICAL_STABILITY = "mechanical_stability"
    CHEMICAL_STABILITY = "chemical_stability"
    DENDRITE_RESISTANCE = "dendrite_resistance"
    CYCLING_STABILITY = "cycling_stability"

@dataclass
class SEIComponent:
    """Represents a single SEI component with its properties"""
    formula: str
    source_element: str
    properties: Dict[SEIProperty, float]  # Scores from 0-1
    weight_fraction: float = 0.0

class SEIPredictor:
    """Predicts SEI composition and properties from SSE composition"""
    
    def __init__(self):
        self.decomposition_database = self._initialize_database()
        self.property_weights = {
            SEIProperty.IONIC_CONDUCTIVITY: 0.20,
            SEIProperty.ELECTRONIC_INSULATION: 0.25,
            SEIProperty.MECHANICAL_STABILITY: 0.15,
            SEIProperty.CHEMICAL_STABILITY: 0.20,
            SEIProperty.DENDRITE_RESISTANCE: 0.15,
            SEIProperty.CYCLING_STABILITY: 0.05
        }
    
    def _initialize_database(self) -> Dict:
        """Initialize the decomposition product database with property scores"""
        return {
            'F': {
                'products': ['LiF'],
                'properties': {
                    'LiF': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.25,
                        SEIProperty.ELECTRONIC_INSULATION: 1.0,
                        SEIProperty.MECHANICAL_STABILITY: 0.7,
                        SEIProperty.CHEMICAL_STABILITY: 1.0,
                        SEIProperty.DENDRITE_RESISTANCE: 0.95,
                        SEIProperty.CYCLING_STABILITY: 0.95
                    }
                }
            },
            'O': {
                'products': ['Li2O', 'Li2CO3'],
                'properties': {
                    'Li2O': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.55,
                        SEIProperty.ELECTRONIC_INSULATION: 0.85,
                        SEIProperty.MECHANICAL_STABILITY: 0.7,
                        SEIProperty.CHEMICAL_STABILITY: 0.85,
                        SEIProperty.DENDRITE_RESISTANCE: 0.75,
                        SEIProperty.CYCLING_STABILITY: 0.75
                    },
                    'Li2CO3': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.45,
                        SEIProperty.ELECTRONIC_INSULATION: 0.85,
                        SEIProperty.MECHANICAL_STABILITY: 0.8,
                        SEIProperty.CHEMICAL_STABILITY: 0.8,
                        SEIProperty.DENDRITE_RESISTANCE: 0.75,
                        SEIProperty.CYCLING_STABILITY: 0.7,
                    }
                }
            },
            'C': {
                'products': ['Li2CO3', 'ROCO2Li'],
                'properties': {
                    'Li2CO3': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.45,
                        SEIProperty.ELECTRONIC_INSULATION: 0.85,
                        SEIProperty.MECHANICAL_STABILITY: 0.8,
                        SEIProperty.CHEMICAL_STABILITY: 0.8,
                        SEIProperty.DENDRITE_RESISTANCE: 0.75,
                        SEIProperty.CYCLING_STABILITY: 0.7,
                    },
                    'ROCO2Li': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.6,
                        SEIProperty.ELECTRONIC_INSULATION: 0.65,
                        SEIProperty.MECHANICAL_STABILITY: 0.55,
                        SEIProperty.CHEMICAL_STABILITY: 0.5,
                        SEIProperty.DENDRITE_RESISTANCE: 0.6,
                        SEIProperty.CYCLING_STABILITY: 0.45
                    }
                }
            },
            'P': {
                'products': ['Li3PO4'],
                'properties': {
                    'Li3PO4': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.4,
                        SEIProperty.ELECTRONIC_INSULATION: 0.9,
                        SEIProperty.MECHANICAL_STABILITY: 0.9,
                        SEIProperty.CHEMICAL_STABILITY: 0.9,
                        SEIProperty.DENDRITE_RESISTANCE: 0.9,
                        SEIProperty.CYCLING_STABILITY: 0.9,
                    }
                }
            },
            'S': {
                'products': ['Li2S', 'Li2SO3'],
                'properties': {
                    'Li2S': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.65,
                        SEIProperty.ELECTRONIC_INSULATION: 0.8,
                        SEIProperty.MECHANICAL_STABILITY: 0.6,
                        SEIProperty.CHEMICAL_STABILITY: 0.55,
                        SEIProperty.DENDRITE_RESISTANCE: 0.6,
                        SEIProperty.CYCLING_STABILITY: 0.4,
                    },
                    'Li2SO3': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.5,
                        SEIProperty.ELECTRONIC_INSULATION: 0.7,
                        SEIProperty.MECHANICAL_STABILITY: 0.55,
                        SEIProperty.CHEMICAL_STABILITY: 0.6,
                        SEIProperty.DENDRITE_RESISTANCE: 0.5,
                        SEIProperty.CYCLING_STABILITY: 0.5,
                    }
                }
            },
            'Cl': {
                'products': ['LiCl'],
                'properties': {
                    'LiCl': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.7,
                        SEIProperty.ELECTRONIC_INSULATION: 0.6,
                        SEIProperty.MECHANICAL_STABILITY: 0.5,
                        SEIProperty.CHEMICAL_STABILITY: 0.6,
                        SEIProperty.DENDRITE_RESISTANCE: 0.5,
                        SEIProperty.CYCLING_STABILITY: 0.6,
                    }
                }
            },
            'Br': {
                'products': ['LiBr'],
                'properties': {
                    'LiBr': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.75,
                        SEIProperty.ELECTRONIC_INSULATION: 0.5,
                        SEIProperty.MECHANICAL_STABILITY: 0.4,
                        SEIProperty.CHEMICAL_STABILITY: 0.5,
                        SEIProperty.DENDRITE_RESISTANCE: 0.4,
                        SEIProperty.CYCLING_STABILITY: 0.5,
                    }
                }
            },
            'I': {
                'products': ['LiI'],
                'properties': {
                    'LiI': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.85,
                        SEIProperty.ELECTRONIC_INSULATION: 0.4,
                        SEIProperty.MECHANICAL_STABILITY: 0.3,
                        SEIProperty.CHEMICAL_STABILITY: 0.4,
                        SEIProperty.DENDRITE_RESISTANCE: 0.3,
                        SEIProperty.CYCLING_STABILITY: 0.4,
                    }
                }
            },
            'Ti': {
                'products': ['TiO2'],
                'properties': {
                    'TiO2': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.15,
                        SEIProperty.ELECTRONIC_INSULATION: 0.9,
                        SEIProperty.MECHANICAL_STABILITY: 0.9,
                        SEIProperty.CHEMICAL_STABILITY: 0.9,
                        SEIProperty.DENDRITE_RESISTANCE: 0.85,
                        SEIProperty.CYCLING_STABILITY: 0.85,
                    }
                }
            },
            'Zr': {
                'products': ['ZrO2'],
                'properties': {
                    'ZrO2': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.15,
                        SEIProperty.ELECTRONIC_INSULATION: 0.9,
                        SEIProperty.MECHANICAL_STABILITY: 0.9,
                        SEIProperty.CHEMICAL_STABILITY: 0.9,
                        SEIProperty.DENDRITE_RESISTANCE: 0.85,
                        SEIProperty.CYCLING_STABILITY: 0.85,
                    }
                }
            },
            'Al': {
                'products': ['Al2O3'],
                'properties': {
                    'Al2O3': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.05,
                        SEIProperty.ELECTRONIC_INSULATION: 1.0,
                        SEIProperty.MECHANICAL_STABILITY: 0.9,
                        SEIProperty.CHEMICAL_STABILITY: 1.0,
                        SEIProperty.DENDRITE_RESISTANCE: 0.95,
                        SEIProperty.CYCLING_STABILITY: 0.95,
                    }
                }
            },
            'La': {
                'products': ['La2O3'],
                'properties': {
                    'La2O3': {
                        SEIProperty.IONIC_CONDUCTIVITY: 0.25,
                        SEIProperty.ELECTRONIC_INSULATION: 0.9,
                        SEIProperty.MECHANICAL_STABILITY: 0.85,
                        SEIProperty.CHEMICAL_STABILITY: 0.9,
                        SEIProperty.DENDRITE_RESISTANCE: 0.85,
                        SEIProperty.CYCLING_STABILITY: 0.85,
                    }
                }
            },

            'Ce': {
            'products': ['CeO2'],
            'properties': {
                'CeO2': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.3,
                    SEIProperty.ELECTRONIC_INSULATION: 0.9,
                    SEIProperty.MECHANICAL_STABILITY: 0.85,
                    SEIProperty.CHEMICAL_STABILITY: 0.9,
                    SEIProperty.DENDRITE_RESISTANCE: 0.85,
                    SEIProperty.CYCLING_STABILITY: 0.85,
                }
            }
        },
        'Nd': {
            'products': ['Nd2O3'],
            'properties': {
                'Nd2O3': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.25,
                    SEIProperty.ELECTRONIC_INSULATION: 0.85,
                    SEIProperty.MECHANICAL_STABILITY: 0.8,
                    SEIProperty.CHEMICAL_STABILITY: 0.85,
                    SEIProperty.DENDRITE_RESISTANCE: 0.8,
                    SEIProperty.CYCLING_STABILITY: 0.8,
                }
            }
        },
        'Sm': {
            'products': ['Sm2O3'],
            'properties': {
                'Sm2O3': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.25,
                    SEIProperty.ELECTRONIC_INSULATION: 0.85,
                    SEIProperty.MECHANICAL_STABILITY: 0.8,
                    SEIProperty.CHEMICAL_STABILITY: 0.85,
                    SEIProperty.DENDRITE_RESISTANCE: 0.8,
                    SEIProperty.CYCLING_STABILITY: 0.8,
                }
            }
        },
        'Ga': {
            'products': ['Ga2O3'],
            'properties': {
                'Ga2O3': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.1,
                    SEIProperty.ELECTRONIC_INSULATION: 0.95,
                    SEIProperty.MECHANICAL_STABILITY: 0.85,
                    SEIProperty.CHEMICAL_STABILITY: 0.9,
                    SEIProperty.DENDRITE_RESISTANCE: 0.85,
                    SEIProperty.CYCLING_STABILITY: 0.85,
                }
            }
        },
        'In': {
            'products': ['In2O3'],
            'properties': {
                'In2O3': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.1,
                    SEIProperty.ELECTRONIC_INSULATION: 0.9,
                    SEIProperty.MECHANICAL_STABILITY: 0.85,
                    SEIProperty.CHEMICAL_STABILITY: 0.85,
                    SEIProperty.DENDRITE_RESISTANCE: 0.8,
                    SEIProperty.CYCLING_STABILITY: 0.8,
                }
            }
        },
        'Hf': {
            'products': ['HfO2'],
            'properties': {
                'HfO2': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.15,
                    SEIProperty.ELECTRONIC_INSULATION: 0.9,
                    SEIProperty.MECHANICAL_STABILITY: 0.9,
                    SEIProperty.CHEMICAL_STABILITY: 0.9,
                    SEIProperty.DENDRITE_RESISTANCE: 0.85,
                    SEIProperty.CYCLING_STABILITY: 0.85,
                }
            }
        },
        'Ta': {
            'products': ['Ta2O5'],
            'properties': {
                'Ta2O5': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.1,
                    SEIProperty.ELECTRONIC_INSULATION: 0.95,
                    SEIProperty.MECHANICAL_STABILITY: 0.9,
                    SEIProperty.CHEMICAL_STABILITY: 0.95,
                    SEIProperty.DENDRITE_RESISTANCE: 0.9,
                    SEIProperty.CYCLING_STABILITY: 0.9,
                }
            }
        },
        'Nb': {
            'products': ['Nb2O5'],
            'properties': {
                'Nb2O5': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.12,
                    SEIProperty.ELECTRONIC_INSULATION: 0.9,
                    SEIProperty.MECHANICAL_STABILITY: 0.85,
                    SEIProperty.CHEMICAL_STABILITY: 0.9,
                    SEIProperty.DENDRITE_RESISTANCE: 0.85,
                    SEIProperty.CYCLING_STABILITY: 0.85,
                }
            }
        },
        'Na': {
            'products': ['Na2O', 'NaCl'],
            'properties': {
                'Na2O': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.5,
                    SEIProperty.ELECTRONIC_INSULATION: 0.8,
                    SEIProperty.MECHANICAL_STABILITY: 0.7,
                    SEIProperty.CHEMICAL_STABILITY: 0.75,
                    SEIProperty.DENDRITE_RESISTANCE: 0.7,
                    SEIProperty.CYCLING_STABILITY: 0.7,
                },
                'NaCl': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.7,
                    SEIProperty.ELECTRONIC_INSULATION: 0.6,
                    SEIProperty.MECHANICAL_STABILITY: 0.5,
                    SEIProperty.CHEMICAL_STABILITY: 0.6,
                    SEIProperty.DENDRITE_RESISTANCE: 0.5,
                    SEIProperty.CYCLING_STABILITY: 0.6,
                }
            }
        },
        'K': {
            'products': ['K2O', 'KCl'],
            'properties': {
                'K2O': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.5,
                    SEIProperty.ELECTRONIC_INSULATION: 0.75,
                    SEIProperty.MECHANICAL_STABILITY: 0.65,
                    SEIProperty.CHEMICAL_STABILITY: 0.7,
                    SEIProperty.DENDRITE_RESISTANCE: 0.65,
                    SEIProperty.CYCLING_STABILITY: 0.65,
                },
                'KCl': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.68,
                    SEIProperty.ELECTRONIC_INSULATION: 0.6,
                    SEIProperty.MECHANICAL_STABILITY: 0.55,
                    SEIProperty.CHEMICAL_STABILITY: 0.6,
                    SEIProperty.DENDRITE_RESISTANCE: 0.55,
                    SEIProperty.CYCLING_STABILITY: 0.6,
                }
            }
        },
        # Hydride systems
        'H': {
            'products': ['LiH', 'ComplexLiHydrides'],
            'properties': {
                'LiH': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.2,
                    SEIProperty.ELECTRONIC_INSULATION: 0.6,
                    SEIProperty.MECHANICAL_STABILITY: 0.4,
                    SEIProperty.CHEMICAL_STABILITY: 0.4,
                    SEIProperty.DENDRITE_RESISTANCE: 0.3,
                    SEIProperty.CYCLING_STABILITY: 0.35,
                },
                'ComplexLiHydrides': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.25,
                    SEIProperty.ELECTRONIC_INSULATION: 0.55,
                    SEIProperty.MECHANICAL_STABILITY: 0.5,
                    SEIProperty.CHEMICAL_STABILITY: 0.45,
                    SEIProperty.DENDRITE_RESISTANCE: 0.4,
                    SEIProperty.CYCLING_STABILITY: 0.45,
                }
            }
        },
        # Mixed compound decomposition products (typically from Li-P-S, Li-N-O systems)
        'LiPS': {
            'products': ['Li3PS4', 'Li7P3S11'],
            'properties': {
                'Li3PS4': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.75,
                    SEIProperty.ELECTRONIC_INSULATION: 0.7,
                    SEIProperty.MECHANICAL_STABILITY: 0.6,
                    SEIProperty.CHEMICAL_STABILITY: 0.6,
                    SEIProperty.DENDRITE_RESISTANCE: 0.65,
                    SEIProperty.CYCLING_STABILITY: 0.6,
                },
                'Li7P3S11': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.8,
                    SEIProperty.ELECTRONIC_INSULATION: 0.65,
                    SEIProperty.MECHANICAL_STABILITY: 0.58,
                    SEIProperty.CHEMICAL_STABILITY: 0.6,
                    SEIProperty.DENDRITE_RESISTANCE: 0.6,
                    SEIProperty.CYCLING_STABILITY: 0.58,
                },
            }
        },
        'LiNO': {
            'products': ['Li3N', 'Li2O·N2'],
            'properties': {
                'Li3N': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.65,
                    SEIProperty.ELECTRONIC_INSULATION: 0.75,
                    SEIProperty.MECHANICAL_STABILITY: 0.6,
                    SEIProperty.CHEMICAL_STABILITY: 0.65,
                    SEIProperty.DENDRITE_RESISTANCE: 0.6,
                    SEIProperty.CYCLING_STABILITY: 0.6,
                },
                'Li2O·N2': {
                    SEIProperty.IONIC_CONDUCTIVITY: 0.55,
                    SEIProperty.ELECTRONIC_INSULATION: 0.7,
                    SEIProperty.MECHANICAL_STABILITY: 0.58,
                    SEIProperty.CHEMICAL_STABILITY: 0.65,
                    SEIProperty.DENDRITE_RESISTANCE: 0.6,
                    SEIProperty.CYCLING_STABILITY: 0.58,
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

    def predict_sei_composition(self, sse_composition: Dict[str, float]) -> List[SEIComponent]:
        """
        Predict SEI composition from SSE elemental composition
        
        Args:
            sse_composition: Dict of {element: atomic_fraction}
            
        Returns:
            List of SEIComponent objects
        """
        sei_components = []
        
        for element, atomic_fraction in sse_composition.items():
            if element in self.decomposition_database:
                products = self.decomposition_database[element]['products']
                properties_db = self.decomposition_database[element]['properties']
                
                # Distribute atomic fraction among products (simplified equal distribution)
                fraction_per_product = atomic_fraction / len(products)
                
                for product in products:
                    if product in properties_db:
                        component = SEIComponent(
                            formula=product,
                            source_element=element,
                            properties=properties_db[product].copy(),
                            weight_fraction=fraction_per_product
                        )
                        sei_components.append(component)
        
        return sei_components
    
    def calculate_overall_properties(self, sei_components: List[SEIComponent]) -> Dict[SEIProperty, float]:
        """Calculate weighted average properties of the SEI"""
        overall_properties = {prop: 0.0 for prop in SEIProperty}
        total_weight = sum(comp.weight_fraction for comp in sei_components)
        
        if total_weight == 0:
            return overall_properties
        
        for component in sei_components:
            weight = component.weight_fraction / total_weight
            for prop, value in component.properties.items():
                overall_properties[prop] += weight * value
        
        return overall_properties
    
    def calculate_sei_score(self, overall_properties: Dict[SEIProperty, float]) -> float:
        """Calculate overall SEI performance score (0-1)"""
        weighted_score = 0.0
        for prop, value in overall_properties.items():
            weighted_score += self.property_weights[prop] * value
        return weighted_score
    
    def predict_from_cif(self, cif_file_path: str) -> Dict:
        """
        Main interface: Predict SEI properties from CIF file
        
        Args:
            cif_file_path: Path to CIF file
            
        Returns:
            Dict containing SEI analysis results with 'sei_score' key for RL agent
        """
        # Extract composition from CIF
        composition = self.extract_composition_from_cif(cif_file_path)
        
        # Predict composition
        sei_components = self.predict_sei_composition(composition)
        
        # Calculate properties
        overall_properties = self.calculate_overall_properties(sei_components)
        
        # Calculate overall score
        sei_score = self.calculate_sei_score(overall_properties)
        
        # Prepare results
        results = {
            'sei_components': [
                {
                    'formula': comp.formula,
                    'source_element': comp.source_element,
                    'weight_fraction': comp.weight_fraction,
                    'properties': {prop.value: value for prop, value in comp.properties.items()}
                }
                for comp in sei_components
            ],
            'overall_properties': {prop.value: value for prop, value in overall_properties.items()},
            'sei_score': sei_score,  # This is what your RL agent will use
            'input_composition': composition,
        }
        
        return results
    
if __name__ == "__main__":
    predictor = SEIPredictor()
    cif_file = "path/to/your/solid_electrolyte.cif"
    results = predictor.predict_from_cif(cif_file)
    print("Predicted SEI Score:", results['sei_score'])
    print("Overall SEI Properties:")
    for prop, value in results['overall_properties'].items():
        print(f" - {prop}: {value:.3f}")
    print("SEI Components:")
    for comp in results['sei_components']:
        print(f" - {comp['formula']} (from {comp['source_element']}, wt frac: {comp['weight_fraction']:.3f})")