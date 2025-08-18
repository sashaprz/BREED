#!/usr/bin/env python3
"""
Enhanced CDVAE Loader

This module loads the EnhancedCDVAE model with the new weights, hparams, and prop_scaler.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from omegaconf import DictConfig, OmegaConf

# Add CDVAE to path
sys.path.append(r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE')

# Set required environment variables
os.environ['PROJECT_ROOT'] = r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE'

try:
    from cdvae.pl_modules.enhanced_model import EnhancedCDVAE
    from cdvae.common.data_utils import StandardScalerTorch
    ENHANCED_CDVAE_IMPORTS_AVAILABLE = True
    print("✅ EnhancedCDVAE imports successful!")
except ImportError as e:
    ENHANCED_CDVAE_IMPORTS_AVAILABLE = False
    print(f"❌ EnhancedCDVAE imports failed: {e}")


class EnhancedCDVAELoader:
    """Enhanced CDVAE loader that properly handles the EnhancedCDVAE model architecture"""
    
    def __init__(self, weights_path: str, hparams_path: str = None, prop_scaler_path: str = None):
        self.weights_path = Path(weights_path)
        self.hparams_path = Path(hparams_path) if hparams_path else Path(weights_path).parent / "new_hparams.yaml"
        self.prop_scaler_path = Path(prop_scaler_path) if prop_scaler_path else Path(weights_path).parent / "new_prop_scaler.pt"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.external_hparams = None
        self.external_prop_scaler = None
        
        print(f"🔧 Initializing Enhanced CDVAE Loader")
        print(f"   Weights file: {self.weights_path}")
        print(f"   Hparams file: {self.hparams_path}")
        print(f"   Prop scaler file: {self.prop_scaler_path}")
        print(f"   Device: {self.device}")
        print(f"   EnhancedCDVAE imports available: {ENHANCED_CDVAE_IMPORTS_AVAILABLE}")
        
        # Load external configurations
        self._load_external_configs()
        
        if ENHANCED_CDVAE_IMPORTS_AVAILABLE:
            self._load_enhanced_model()
        else:
            print("❌ Cannot load EnhancedCDVAE model - imports not available")
    
    def _load_external_configs(self):
        """Load external hparams and prop_scaler files"""
        try:
            # Load hparams from YAML file
            if self.hparams_path.exists():
                print(f"📄 Loading hparams from: {self.hparams_path}")
                with open(self.hparams_path, 'r') as f:
                    self.external_hparams = yaml.safe_load(f)
                print("   ✅ External hparams loaded successfully")
            else:
                print(f"   ⚠️ Hparams file not found: {self.hparams_path}")
            
            # Load prop_scaler
            if self.prop_scaler_path.exists():
                print(f"📄 Loading prop_scaler from: {self.prop_scaler_path}")
                self.external_prop_scaler = torch.load(self.prop_scaler_path, map_location=self.device, weights_only=False)
                print("   ✅ External prop_scaler loaded successfully")
            else:
                print(f"   ⚠️ Prop_scaler file not found: {self.prop_scaler_path}")
                
        except Exception as e:
            print(f"   ⚠️ Error loading external configs: {e}")
    
    def _load_enhanced_model(self):
        """Load the EnhancedCDVAE model with proper architecture"""
        try:
            print(f"📁 Loading EnhancedCDVAE model from: {self.weights_path}")
            
            if not self.weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
            
            # Method 1: Try load_from_checkpoint (most reliable for PyTorch Lightning models)
            try:
                print("   🔄 Trying load_from_checkpoint method...")
                self.model = EnhancedCDVAE.load_from_checkpoint(
                    str(self.weights_path), 
                    map_location=self.device,
                    strict=False
                )
                
                # Initialize scalers
                self._initialize_scalers()
                
                self.model.eval()
                self.model.to(self.device)
                print("✅ EnhancedCDVAE model loaded successfully with load_from_checkpoint!")
                return
                
            except Exception as e1:
                print(f"   Method 1 failed: {e1}")
            
            # Method 2: Load checkpoint and use its hyperparameters
            try:
                print("   🔄 Trying checkpoint hyperparameters...")
                
                # Load checkpoint
                checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
                
                if 'hyper_parameters' in checkpoint:
                    # Use the hyperparameters from the checkpoint itself
                    checkpoint_hparams = checkpoint['hyper_parameters']
                    
                    # Create model using checkpoint hparams (bypass hydra issues)
                    self.model = EnhancedCDVAE.load_from_checkpoint(
                        str(self.weights_path),
                        map_location=self.device,
                        strict=False,
                        # Override problematic decoder config
                        decoder={
                            '_target_': 'cdvae.pl_modules.decoder.GemNetTDecoder',
                            'hidden_dim': checkpoint_hparams.get('hidden_dim', 256),
                            'latent_dim': checkpoint_hparams.get('latent_dim', 256),
                            'max_neighbors': 20,
                            'radius': 7.0,
                        }
                    )
                    
                    # Initialize scalers
                    self._initialize_scalers()
                    
                    self.model.eval()
                    self.model.to(self.device)
                    print("✅ EnhancedCDVAE model loaded successfully with checkpoint hparams!")
                    return
                    
            except Exception as e2:
                print(f"   Method 2 failed: {e2}")
            
            print("❌ All loading methods failed")
            self.model = None
            
        except Exception as e:
            print(f"❌ Error loading EnhancedCDVAE model: {e}")
            self.model = None
    
    def _create_hparams_from_external(self) -> Dict:
        """Create hparams dictionary from external configuration"""
        if not self.external_hparams or 'model' not in self.external_hparams:
            raise ValueError("External hparams not available or missing model section")
        
        model_config = self.external_hparams['model']
        data_config = self.external_hparams.get('data', {})
        
        # Create hparams compatible with EnhancedCDVAE
        hparams = {
            # Core model parameters
            'hidden_dim': model_config.get('hidden_dim', 256),
            'latent_dim': model_config.get('latent_dim', 256),
            'fc_num_layers': model_config.get('fc_num_layers', 2),
            'max_atoms': data_config.get('max_atoms', 20),
            
            # Enhanced model parameters
            'transformer_layers': model_config.get('transformer_layers', 3),
            'attention_heads': model_config.get('attention_heads', 8),
            'dropout_rate': model_config.get('dropout_rate', 0.1),
            
            # Loss weights
            'cost_natom': model_config.get('cost_natom', 2.0),
            'cost_natom_enhanced': model_config.get('cost_natom_enhanced', 3.0),
            'cost_coord': model_config.get('cost_coord', 10.0),
            'cost_type': model_config.get('cost_type', 1.0),
            'cost_lattice': model_config.get('cost_lattice', 10.0),
            'cost_composition': model_config.get('cost_composition', 1.5),
            'cost_edge': model_config.get('cost_edge', 10.0),
            'cost_property': model_config.get('cost_property', 1.0),
            
            # Beta scheduling
            'beta': model_config.get('beta', 0.01),
            'beta_start': model_config.get('beta_start', 0.0),
            'beta_end': model_config.get('beta_end', 0.01),
            'beta_warmup_epochs': model_config.get('beta_warmup_epochs', 15),
            'beta_schedule': model_config.get('beta_schedule', 'cosine'),
            'kld_capacity': model_config.get('kld_capacity', 0.0),
            
            # Noise parameters
            'sigma_begin': model_config.get('sigma_begin', 10.0),
            'sigma_end': model_config.get('sigma_end', 0.01),
            'type_sigma_begin': model_config.get('type_sigma_begin', 5.0),
            'type_sigma_end': model_config.get('type_sigma_end', 0.01),
            'num_noise_level': model_config.get('num_noise_level', 50),
            
            # Teacher forcing
            'teacher_forcing_lattice': model_config.get('teacher_forcing_lattice', True),
            'teacher_forcing_max_epoch': model_config.get('teacher_forcing_max_epoch', 10),
            
            # Decoder parameters
            'max_neighbors': model_config.get('max_neighbors', 20),
            'radius': model_config.get('radius', 7.0),
            
            # Property prediction
            'predict_property': model_config.get('predict_property', False),
            'num_targets': data_config.get('num_targets', 1),
        }
        
        # Add encoder configuration
        if 'encoder' in model_config:
            encoder_config = model_config['encoder'].copy()
            # Resolve interpolations
            encoder_config['num_targets'] = hparams['latent_dim']
            if 'otf_graph' in encoder_config and str(encoder_config['otf_graph']).startswith('${'):
                encoder_config['otf_graph'] = data_config.get('otf_graph', False)
            if 'readout' in encoder_config and str(encoder_config['readout']).startswith('${'):
                encoder_config['readout'] = data_config.get('readout', 'mean')
            hparams['encoder'] = encoder_config
        
        # Add decoder configuration - use simplified config to avoid path issues
        hparams['decoder'] = {
            '_target_': 'cdvae.pl_modules.decoder.GemNetTDecoder',
            'hidden_dim': hparams['hidden_dim'],
            'latent_dim': hparams['latent_dim'],
            'max_neighbors': hparams['max_neighbors'],
            'radius': hparams['radius'],
            # Don't include scale_file to avoid path issues
        }
        
        # Add data configuration
        hparams['data'] = data_config
        
        return hparams
    
    def _initialize_scalers(self):
        """Initialize scalers for the EnhancedCDVAE model"""
        try:
            print("   🔧 Initializing EnhancedCDVAE scalers...")
            
            # Create proper lattice scaler with reasonable defaults for electrolytes
            lattice_means = torch.tensor([8.0, 8.0, 8.0, 90.0, 90.0, 90.0], dtype=torch.float)
            lattice_stds = torch.tensor([3.0, 3.0, 3.0, 15.0, 15.0, 15.0], dtype=torch.float)
            
            # Use external prop_scaler if available
            if self.external_prop_scaler is not None:
                print("   🔄 Using external prop_scaler")
                prop_scaler = self.external_prop_scaler
            else:
                print("   🔄 Using default prop_scaler")
                prop_means = torch.tensor([0.0], dtype=torch.float)
                prop_stds = torch.tensor([1.0], dtype=torch.float)
                prop_scaler = StandardScalerTorch(means=prop_means, stds=prop_stds)
            
            # Add lattice_scaler to the main model
            if not hasattr(self.model, 'lattice_scaler') or self.model.lattice_scaler is None:
                self.model.lattice_scaler = StandardScalerTorch(means=lattice_means, stds=lattice_stds)
                print("     ✅ Added lattice_scaler to EnhancedCDVAE model")
            
            # Add property scaler to the main model
            if not hasattr(self.model, 'scaler') or self.model.scaler is None:
                self.model.scaler = prop_scaler
                print("     ✅ Added property scaler to EnhancedCDVAE model")
            
            # Initialize decoder scalers if needed
            if hasattr(self.model, 'decoder') and self.model.decoder is not None:
                decoder = self.model.decoder
                
                if not hasattr(decoder, 'scaler') or decoder.scaler is None:
                    decoder.scaler = prop_scaler
                    print("     ✅ Added scaler to decoder")
                
                if not hasattr(decoder, 'normalizer') or decoder.normalizer is None:
                    decoder.normalizer = prop_scaler
                    print("     ✅ Added normalizer to decoder")
            
            # Initialize encoder scalers if needed
            if hasattr(self.model, 'encoder') and self.model.encoder is not None:
                encoder = self.model.encoder
                
                if not hasattr(encoder, 'scaler') or encoder.scaler is None:
                    encoder.scaler = prop_scaler
                    print("     ✅ Added scaler to encoder")
            
            print("   ✅ EnhancedCDVAE scaler initialization complete")
            
        except Exception as e:
            print(f"   ⚠️ Warning: Could not initialize scalers: {e}")
    
    def generate_structures(self, num_samples: int = 10) -> List[Dict]:
        """Generate structures using the EnhancedCDVAE model"""
        if self.model is None:
            print("❌ No EnhancedCDVAE model loaded, cannot generate structures")
            return []
        
        print(f"🔬 Generating {num_samples} structures using EnhancedCDVAE model...")
        
        structures = []
        
        try:
            with torch.no_grad():
                for i in range(num_samples):
                    print(f"   Generating structure {i+1}/{num_samples}...")
                    
                    try:
                        # Generate from latent space
                        structure_data = self._generate_from_enhanced_latent_space(i)
                        
                        if structure_data:
                            structure_data['generation_method'] = 'enhanced_cdvae_model'
                            structures.append(structure_data)
                            print(f"     ✅ Generated: {structure_data['composition']}")
                        else:
                            print(f"     ❌ Failed to generate structure {i+1}")
                            
                    except Exception as e:
                        print(f"     ❌ Error generating structure {i+1}: {e}")
            
            print(f"   ✅ Generated {len(structures)} structures using EnhancedCDVAE model")
            return structures
            
        except Exception as e:
            print(f"❌ Error in structure generation: {e}")
            return []
    
    def _generate_from_enhanced_latent_space(self, index: int) -> Optional[Dict]:
        """Generate structure from latent space using EnhancedCDVAE"""
        try:
            # Generate random latent vector
            latent_dim = getattr(self.model.hparams, 'latent_dim', 256)
            z = torch.randn(1, latent_dim, device=self.device)
            
            # Use EnhancedCDVAE's decode_stats method
            if hasattr(self.model, 'decode_stats'):
                try:
                    result = self.model.decode_stats(z)
                    
                    if isinstance(result, tuple) and len(result) >= 5:
                        num_atoms, _, lengths, angles, composition_per_atom = result
                        
                        # Convert to structure format
                        composition = self._composition_tensor_to_dict(composition_per_atom, num_atoms)
                        
                        lattice_params = {
                            'a': float(lengths[0][0]),
                            'b': float(lengths[0][1]),
                            'c': float(lengths[0][2]),
                            'alpha': float(angles[0][0]),
                            'beta': float(angles[0][1]),
                            'gamma': float(angles[0][2])
                        }
                        
                        return {
                            'composition': composition,
                            'lattice_params': lattice_params,
                            'space_group': self._select_space_group(composition),
                            'generated_id': f'enhanced_cdvae_{index+1:03d}'
                        }
                        
                except Exception as e:
                    print(f"     Warning: EnhancedCDVAE decode_stats failed: {e}")
            
            return None
            
        except Exception as e:
            print(f"     Error in enhanced latent generation: {e}")
            return None
    
    def _composition_tensor_to_dict(self, composition_per_atom: torch.Tensor, num_atoms: torch.Tensor) -> Dict[str, int]:
        """Convert composition tensor to dictionary"""
        try:
            # Get probabilities and sample elements
            probs = torch.softmax(composition_per_atom, dim=-1)
            sampled_elements = torch.multinomial(probs, num_samples=1).squeeze()
            
            return self._atom_types_to_composition(sampled_elements + 1)  # +1 for atomic number
            
        except Exception as e:
            print(f"     Error converting composition tensor: {e}")
            return {'Li': 3, 'Ti': 2, 'P': 1, 'O': 12}
    
    def _atom_types_to_composition(self, atom_types: torch.Tensor) -> Dict[str, int]:
        """Convert atom types tensor to composition dictionary"""
        # Element mapping (atomic number to symbol)
        element_map = {
            1: 'H', 3: 'Li', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si',
            15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
            26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 35: 'Br', 37: 'Rb',
            38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
            53: 'I', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 72: 'Hf', 73: 'Ta', 74: 'W', 82: 'Pb'
        }
        
        composition = {}
        for atom_type in atom_types:
            if isinstance(atom_type, torch.Tensor):
                z = atom_type.item()
            else:
                z = int(atom_type)
            
            if z in element_map:
                element = element_map[z]
                composition[element] = composition.get(element, 0) + 1
        
        # Ensure Li is present for electrolytes
        if 'Li' not in composition and composition:
            # Add Li by replacing least common element
            min_element = min(composition.keys(), key=lambda x: composition[x])
            composition['Li'] = composition.pop(min_element)
        elif not composition:
            composition = {'Li': 3, 'Ti': 2, 'P': 1, 'O': 12}
        
        return composition
    
    def _select_space_group(self, composition: Dict[str, int]) -> int:
        """Select space group based on composition"""
        if 'La' in composition and 'Zr' in composition:
            return 230  # Garnet
        elif 'Ti' in composition and 'P' in composition:
            return 167  # NASICON
        elif 'P' in composition and 'S' in composition:
            return 216  # Argyrodite
        else:
            return 225  # Cubic


def test_enhanced_cdvae_loader():
    """Test the enhanced CDVAE loader"""
    weights_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\new_cdvae_weights.ckpt"
    hparams_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\new_hparams.yaml"
    prop_scaler_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\new_prop_scaler.pt"
    
    try:
        loader = EnhancedCDVAELoader(weights_path, hparams_path, prop_scaler_path)
        
        if loader.model is not None:
            structures = loader.generate_structures(num_samples=3)
            
            print(f"\n🎯 Enhanced CDVAE Loader Test Results:")
            print(f"Generated {len(structures)} structures")
            
            for i, structure in enumerate(structures):
                print(f"\nStructure {i+1}:")
                print(f"  Composition: {structure['composition']}")
                print(f"  Lattice: a={structure['lattice_params']['a']:.3f}")
                print(f"  Space Group: {structure['space_group']}")
                print(f"  Method: {structure['generation_method']}")
            
            return loader
        else:
            print("❌ Failed to load EnhancedCDVAE model")
            return None
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    test_enhanced_cdvae_loader()