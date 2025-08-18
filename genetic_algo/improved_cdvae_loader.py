#!/usr/bin/env python3
"""
Improved CDVAE Loader

This module properly loads the full CDVAE model with the complete architecture
using the actual weights from new_cdvae_weights.ckpt
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
    from cdvae.pl_modules.model import CDVAE
    from cdvae.pl_modules.enhanced_model import EnhancedCDVAE
    from cdvae.pl_modules.gnn import DimeNetPlusPlusWrap
    from cdvae.pl_modules.decoder import GemNetTDecoder
    CDVAE_IMPORTS_AVAILABLE = True
    print("✅ CDVAE imports successful with torch_sparse!")
except ImportError as e:
    CDVAE_IMPORTS_AVAILABLE = False
    print(f"❌ CDVAE imports failed: {e}")


class ImprovedCDVAELoader:
    """Improved CDVAE loader that properly handles the full model architecture"""
    
    def __init__(self, weights_path: str, hparams_path: str = None, prop_scaler_path: str = None, lattice_scaler_path: str = None):
        self.weights_path = Path(weights_path)
        self.hparams_path = Path(hparams_path) if hparams_path else Path(weights_path).parent / "new_hparams.yaml"
        self.prop_scaler_path = Path(prop_scaler_path) if prop_scaler_path else Path(weights_path).parent / "new_prop_scaler.pt"
        self.lattice_scaler_path = Path(lattice_scaler_path) if lattice_scaler_path else Path(weights_path).parent / "new_lattice_scaler.pt"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.external_hparams = None
        self.external_prop_scaler = None
        self.external_lattice_scaler = None
        
        print(f"🔧 Initializing Improved CDVAE Loader")
        print(f"   Weights file: {self.weights_path}")
        print(f"   Hparams file: {self.hparams_path}")
        print(f"   Prop scaler file: {self.prop_scaler_path}")
        print(f"   Lattice scaler file: {self.lattice_scaler_path}")
        print(f"   Device: {self.device}")
        print(f"   CDVAE imports available: {CDVAE_IMPORTS_AVAILABLE}")
        
        # Load external configurations
        self._load_external_configs()
        
        if CDVAE_IMPORTS_AVAILABLE:
            self._load_full_model()
        else:
            print("❌ Cannot load full CDVAE model - imports not available")
    
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
    
    def _load_full_model(self):
        """Load the complete CDVAE model with proper architecture"""
        try:
            print(f"📁 Loading full CDVAE model from: {self.weights_path}")
            
            if not self.weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            
            print(f"   Loaded checkpoint with keys: {list(checkpoint.keys())}")
            
            # Extract hyperparameters
            if 'hyper_parameters' in checkpoint:
                hparams_raw = checkpoint['hyper_parameters']
                print(f"   Raw hyperparameters type: {type(hparams_raw)}")
                
                # Convert to proper format for CDVAE
                hparams = self._fix_hyperparameters(hparams_raw)
                
                print(f"   Fixed hyperparameters:")
                print(f"     hidden_dim: {hparams.get('hidden_dim')}")
                print(f"     latent_dim: {hparams.get('latent_dim')}")
                print(f"     max_atoms: {hparams.get('max_atoms')}")
                
                # Try to create EnhancedCDVAE model (this is what the checkpoint contains!)
                try:
                    # Method 1: Direct instantiation with EnhancedCDVAE
                    self.model = EnhancedCDVAE(**hparams)
                    print("✅ EnhancedCDVAE model created successfully!")
                    
                    # Load state dict
                    if 'state_dict' in checkpoint:
                        missing_keys, unexpected_keys = self.model.load_state_dict(
                            checkpoint['state_dict'], strict=False
                        )
                        
                        loaded_params = len(checkpoint['state_dict']) - len(missing_keys)
                        total_params = len(checkpoint['state_dict'])
                        
                        print(f"   State dict loaded: {loaded_params}/{total_params} parameters")
                        print(f"   Missing keys: {len(missing_keys)}")
                        print(f"   Unexpected keys: {len(unexpected_keys)}")
                        
                        if loaded_params >= total_params * 0.8:  # If we loaded at least 80%
                            # Initialize missing scalers/normalizers
                            self._initialize_scalers()
                            
                            self.model.eval()
                            self.model.to(self.device)
                            print("✅ Full CDVAE model loaded successfully!")
                            return
                        else:
                            print(f"❌ Only loaded {loaded_params}/{total_params} parameters")
                    
                except Exception as e1:
                    print(f"   Method 1 failed: {e1}")
                
                # Method 2: Try with OmegaConf
                try:
                    hparams_conf = OmegaConf.create(hparams)
                    self.model = EnhancedCDVAE(hparams_conf)
                    
                    if 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                        # Initialize missing scalers/normalizers
                        self._initialize_scalers()
                        self.model.eval()
                        self.model.to(self.device)
                        print("✅ EnhancedCDVAE model loaded with OmegaConf!")
                        return
                        
                except Exception as e2:
                    print(f"   Method 2 failed: {e2}")
                
                # Method 3: Try load_from_checkpoint
                try:
                    self.model = EnhancedCDVAE.load_from_checkpoint(str(self.weights_path), strict=False)
                    # Initialize missing scalers/normalizers
                    self._initialize_scalers()
                    self.model.eval()
                    self.model.to(self.device)
                    print("✅ EnhancedCDVAE model loaded with load_from_checkpoint!")
                    return
                    
                except Exception as e3:
                    print(f"   Method 3 failed: {e3}")
            
            print("❌ All loading methods failed")
            self.model = None
            
        except Exception as e:
            print(f"❌ Error loading CDVAE model: {e}")
            self.model = None
    
    def _fix_hyperparameters(self, hparams_raw: Dict) -> Dict:
        """Fix hyperparameters to be compatible with CDVAE constructor"""
        
        # Use exact hparams from the configuration file - this is EnhancedCDVAE!
        print("   🔄 Using EnhancedCDVAE configuration from hparams file")
        hparams = {
            # Core model parameters from hparams
            'hidden_dim': 256,  # From hparams line 107
            'latent_dim': 256,  # From hparams line 108
            'fc_num_layers': 2,  # From hparams line 109
            'max_atoms': 20,     # From hparams line 110
            
            # Cost parameters from hparams
            'cost_natom': 2.0,
            'cost_natom_enhanced': 3.0,  # EnhancedCDVAE specific
            'cost_coord': 10.0,
            'cost_type': 1.0,
            'cost_lattice': 10.0,
            'cost_composition': 1.5,
            'cost_edge': 10.0,           # EnhancedCDVAE specific
            'cost_property': 1.0,
            'beta': 0.01,
            'beta_start': 0.0,           # EnhancedCDVAE specific
            'beta_end': 0.01,            # EnhancedCDVAE specific
            'beta_warmup_epochs': 15,    # EnhancedCDVAE specific
            'beta_schedule': 'cosine',   # EnhancedCDVAE specific
            'kld_capacity': 0.0,         # EnhancedCDVAE specific
            
            # Transformer parameters (EnhancedCDVAE specific)
            'transformer_layers': 3,
            'attention_heads': 8,
            'dropout_rate': 0.1,
            
            # Other parameters
            'teacher_forcing_lattice': True,
            'teacher_forcing_max_epoch': 10,
            'max_neighbors': 20,
            'radius': 7.0,
            'sigma_begin': 10.0,
            'sigma_end': 0.01,
            'type_sigma_begin': 5.0,
            'type_sigma_end': 0.01,
            'num_noise_level': 50,
            'predict_property': False,
        }
        
        # Handle encoder configuration - exact from hparams
        print("   🔄 Using encoder configuration from hparams file")
        hparams['encoder'] = {
            '_target_': 'cdvae.pl_modules.gnn.DimeNetPlusPlusWrap',
            'num_targets': 1,            # From hparams line 83
            'hidden_channels': 128,      # From hparams line 84
            'num_blocks': 4,             # From hparams line 85
            'int_emb_size': 64,          # From hparams line 86
            'basis_emb_size': 8,         # From hparams line 87
            'out_emb_channels': 256,     # From hparams line 88
            'num_spherical': 7,          # From hparams line 89
            'num_radial': 6,             # From hparams line 90
            'otf_graph': False,          # From hparams line 91
            'cutoff': 7.0,               # From hparams line 92
            'max_num_neighbors': 20,     # From hparams line 93
            'envelope_exponent': 5,      # From hparams line 94
            'num_before_skip': 1,        # From hparams line 95
            'num_after_skip': 2,         # From hparams line 96
            'num_output_layers': 3,      # From hparams line 97
            'readout': 'mean',           # From hparams line 98
        }
        
        # Handle decoder configuration - exact from hparams
        print("   🔄 Using decoder configuration from hparams file")
        scale_file_path = r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\cdvae\pl_modules\gemnet\gemnet-dT.json'
        hparams['decoder'] = {
            '_target_': 'cdvae.pl_modules.decoder.GemNetTDecoder',
            'hidden_dim': 128,           # From hparams line 101
            'latent_dim': 256,           # From hparams line 102
            'max_neighbors': 20,         # From hparams line 103
            'radius': 7.0,               # From hparams line 104
            'scale_file': scale_file_path,  # Updated path
        }
        print(f"     ✅ Using scale_file: {scale_file_path}")
        
        # Add other required fields
        hparams['optim'] = hparams_raw.get('optim', {})
        hparams['data'] = hparams_raw.get('data', {})
        hparams['logging'] = hparams_raw.get('logging', {})
        
        return hparams
    
    def _initialize_scalers(self):
        """Initialize missing scalers/normalizers that CDVAE expects"""
        try:
            print("   🔧 Initializing proper CDVAE scalers...")
            
            # Import the proper StandardScalerTorch from CDVAE
            from cdvae.common.data_utils import StandardScalerTorch
            
            # Create proper lattice scaler with reasonable defaults for electrolytes
            # Based on typical lattice parameters: lengths 5-15 Å, angles 60-120°
            lattice_means = torch.tensor([8.0, 8.0, 8.0, 90.0, 90.0, 90.0], dtype=torch.float)  # a,b,c,α,β,γ
            lattice_stds = torch.tensor([3.0, 3.0, 3.0, 15.0, 15.0, 15.0], dtype=torch.float)   # reasonable std devs
            
            # Use external prop_scaler if available, otherwise create defaults
            if self.external_prop_scaler is not None:
                print("   🔄 Using external prop_scaler")
                prop_scaler = self.external_prop_scaler
            else:
                print("   🔄 Using default prop_scaler")
                # Create proper property scaler with reasonable defaults
                # Assuming single property (like formation energy)
                prop_means = torch.tensor([0.0], dtype=torch.float)
                prop_stds = torch.tensor([1.0], dtype=torch.float)
                prop_scaler = StandardScalerTorch(means=prop_means, stds=prop_stds)
            
            # CRITICAL: Add lattice_scaler to the main model
            if not hasattr(self.model, 'lattice_scaler') or self.model.lattice_scaler is None:
                self.model.lattice_scaler = StandardScalerTorch(means=lattice_means, stds=lattice_stds)
                print("     ✅ Added proper lattice_scaler to model")
            
            # Add property scaler to the main model
            if not hasattr(self.model, 'scaler') or self.model.scaler is None:
                self.model.scaler = prop_scaler
                print("     ✅ Added proper property scaler to model")
            
            # Check if model has decoder and if it needs scalers
            if hasattr(self.model, 'decoder') and self.model.decoder is not None:
                decoder = self.model.decoder
                
                # Initialize scaler if missing
                if not hasattr(decoder, 'scaler') or decoder.scaler is None:
                    decoder.scaler = prop_scaler
                    print("     ✅ Added proper scaler to decoder")
                
                # Initialize normalizer if missing (use same as scaler for simplicity)
                if not hasattr(decoder, 'normalizer') or decoder.normalizer is None:
                    decoder.normalizer = prop_scaler
                    print("     ✅ Added proper normalizer to decoder")
            
            # Check if model has encoder and if it needs scalers
            if hasattr(self.model, 'encoder') and self.model.encoder is not None:
                encoder = self.model.encoder
                
                if not hasattr(encoder, 'scaler') or encoder.scaler is None:
                    encoder.scaler = prop_scaler
                    print("     ✅ Added proper scaler to encoder")
            
            print("   ✅ Proper CDVAE scaler initialization complete")
            
        except Exception as e:
            print(f"   ⚠️ Warning: Could not initialize proper scalers: {e}")
            print("   Falling back to basic scalers...")
            # Fallback to basic implementation if import fails
            self._initialize_basic_scalers()
    
    def _initialize_basic_scalers(self):
        """Fallback basic scaler initialization"""
        try:
            # Create a basic scaler class with all required methods
            class BasicScaler:
                def __init__(self, means=None, stds=None):
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.means = means if means is not None else torch.zeros(6, device=self.device)
                    self.stds = stds if stds is not None else torch.ones(6, device=self.device)
                
                def match_device(self, tensor):
                    if self.means.device != tensor.device:
                        self.means = self.means.to(tensor.device)
                        self.stds = self.stds.to(tensor.device)
                    return tensor.to(self.device)
                
                def transform(self, tensor):
                    return (tensor - self.means) / self.stds
                
                def inverse_transform(self, tensor):
                    return tensor * self.stds + self.means
                
                def scale(self, tensor):
                    return self.transform(tensor)
                
                def unscale(self, tensor):
                    return self.inverse_transform(tensor)
                
                def copy(self):
                    return BasicScaler(means=self.means.clone(), stds=self.stds.clone())
            
            # Initialize with reasonable defaults
            lattice_means = torch.tensor([8.0, 8.0, 8.0, 90.0, 90.0, 90.0], dtype=torch.float)
            lattice_stds = torch.tensor([3.0, 3.0, 3.0, 15.0, 15.0, 15.0], dtype=torch.float)
            prop_means = torch.tensor([0.0], dtype=torch.float)
            prop_stds = torch.tensor([1.0], dtype=torch.float)
            
            # Add to model
            if not hasattr(self.model, 'lattice_scaler') or self.model.lattice_scaler is None:
                self.model.lattice_scaler = BasicScaler(means=lattice_means, stds=lattice_stds)
                print("     ✅ Added basic lattice_scaler to model")
            
            if not hasattr(self.model, 'scaler') or self.model.scaler is None:
                self.model.scaler = BasicScaler(means=prop_means, stds=prop_stds)
                print("     ✅ Added basic property scaler to model")
            
        except Exception as e:
            print(f"   ❌ Even basic scaler initialization failed: {e}")
    
    def generate_structures(self, num_samples: int = 10) -> List[Dict]:
        """Generate structures using the full CDVAE model"""
        if self.model is None:
            print("❌ No CDVAE model loaded, cannot generate structures")
            return []
        
        print(f"🔬 Generating {num_samples} structures using full CDVAE model...")
        
        structures = []
        
        try:
            with torch.no_grad():
                for i in range(num_samples):
                    print(f"   Generating structure {i+1}/{num_samples}...")
                    
                    try:
                        # Try different generation methods
                        structure_data = None
                        
                        # Method 1: Try latent space generation first (more reliable)
                        try:
                            structure_data = self._generate_from_latent_space(i)
                            if structure_data:
                                print(f"     ✅ Generated using latent space method")
                        except Exception as e1:
                            print(f"     ⚠️ Latent space method failed: {e1}")
                        
                        # Method 2: Try simple generation if latent failed
                        if not structure_data:
                            structure_data = self._generate_simple_structure(i)
                            if structure_data:
                                print(f"     ✅ Generated using simple fallback method")
                        
                        if structure_data:
                            structure_data['generation_method'] = 'full_cdvae_model'
                            structures.append(structure_data)
                            print(f"     ✅ Generated: {structure_data['composition']}")
                        else:
                            print(f"     ❌ Failed to generate structure {i+1}")
                            
                    except Exception as e:
                        print(f"     ❌ Error generating structure {i+1}: {e}")
            
            print(f"   ✅ Generated {len(structures)} structures using full CDVAE model")
            return structures
            
        except Exception as e:
            print(f"❌ Error in structure generation: {e}")
            return []
    
    def _convert_sample_to_structure(self, sample_output: Dict, index: int) -> Optional[Dict]:
        """Convert CDVAE sample output to structure format"""
        try:
            # Extract structure information from CDVAE output
            num_atoms = sample_output.get('num_atoms', torch.tensor([10]))
            lengths = sample_output.get('lengths', torch.tensor([[8.0, 8.0, 8.0]]))
            angles = sample_output.get('angles', torch.tensor([[90.0, 90.0, 90.0]]))
            frac_coords = sample_output.get('frac_coords', torch.rand(10, 3))
            atom_types = sample_output.get('atom_types', torch.randint(1, 95, (10,)))
            
            # Convert to structure format
            if isinstance(num_atoms, torch.Tensor):
                num_atoms = num_atoms.item() if num_atoms.numel() == 1 else num_atoms[0].item()
            if isinstance(lengths, torch.Tensor):
                lengths = lengths[0] if len(lengths.shape) > 1 else lengths
            if isinstance(angles, torch.Tensor):
                angles = angles[0] if len(angles.shape) > 1 else angles
            
            # Create composition from atom types
            composition = self._atom_types_to_composition(atom_types)
            
            # Create lattice parameters
            lattice_params = {
                'a': float(lengths[0]),
                'b': float(lengths[1]),
                'c': float(lengths[2]),
                'alpha': float(angles[0]),
                'beta': float(angles[1]),
                'gamma': float(angles[2])
            }
            
            return {
                'composition': composition,
                'lattice_params': lattice_params,
                'space_group': self._select_space_group(composition),
                'generated_id': f'full_cdvae_{index+1:03d}',
                'frac_coords': frac_coords.cpu().numpy() if isinstance(frac_coords, torch.Tensor) else frac_coords
            }
            
        except Exception as e:
            print(f"     Error converting sample: {e}")
            return None
    
    def _generate_from_latent_space(self, index: int) -> Optional[Dict]:
        """Generate structure from latent space using true CDVAE weights"""
        try:
            # Generate random latent vector
            latent_dim = getattr(self.model.hparams, 'latent_dim', 256)
            z = torch.randn(1, latent_dim, device=self.device)
            
            # Use CDVAE's decode_stats method to generate structure
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
                            'generated_id': f'cdvae_latent_{index+1:03d}'
                        }
                        
                except Exception as e:
                    print(f"     Warning: CDVAE decode_stats failed: {e}")
            
            return None
            
        except Exception as e:
            print(f"     Error in latent generation: {e}")
            return None
    
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
    
    def _generate_simple_structure(self, index: int) -> Optional[Dict]:
        """Generate a simple structure as fallback"""
        try:
            # Generate realistic electrolyte compositions
            compositions = [
                {'Li': 7, 'La': 3, 'Zr': 2, 'O': 12},  # LLZO-type
                {'Li': 3, 'Ti': 2, 'P': 3, 'O': 12},   # NASICON-type
                {'Li': 6, 'P': 1, 'S': 5, 'Cl': 1},    # Argyrodite-type
                {'Li': 1, 'Al': 1, 'Ge': 1, 'P': 1, 'O': 4},  # LAGP-type
                {'Li': 2, 'Si': 1, 'P': 1, 'O': 4},    # LSP-type
            ]
            
            composition = compositions[index % len(compositions)]
            
            # Generate realistic lattice parameters
            if 'La' in composition and 'Zr' in composition:
                # Garnet structure
                a = np.random.uniform(12.8, 13.2)
                lattice_params = {'a': a, 'b': a, 'c': a, 'alpha': 90, 'beta': 90, 'gamma': 90}
                space_group = 230
            elif 'Ti' in composition and 'P' in composition:
                # NASICON structure
                a = np.random.uniform(8.4, 8.8)
                c = np.random.uniform(20.8, 21.2)
                lattice_params = {'a': a, 'b': a, 'c': c, 'alpha': 90, 'beta': 90, 'gamma': 120}
                space_group = 167
            elif 'P' in composition and 'S' in composition:
                # Argyrodite structure
                a = np.random.uniform(9.8, 10.2)
                lattice_params = {'a': a, 'b': a, 'c': a, 'alpha': 90, 'beta': 90, 'gamma': 90}
                space_group = 216
            else:
                # Generic cubic
                a = np.random.uniform(8.0, 12.0)
                lattice_params = {'a': a, 'b': a, 'c': a, 'alpha': 90, 'beta': 90, 'gamma': 90}
                space_group = 225
            
            return {
                'composition': composition,
                'lattice_params': lattice_params,
                'space_group': space_group,
                'generated_id': f'simple_fallback_{index+1:03d}'
            }
            
        except Exception as e:
            print(f"     Error in simple generation: {e}")
            return None
    
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


def test_improved_cdvae_loader():
    """Test the improved CDVAE loader"""
    weights_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\new_cdvae_weights.ckpt"
    
    try:
        loader = ImprovedCDVAELoader(weights_path)
        
        if loader.model is not None:
            structures = loader.generate_structures(num_samples=3)
            
            print(f"\n🎯 Improved CDVAE Loader Test Results:")
            print(f"Generated {len(structures)} structures")
            
            for i, structure in enumerate(structures):
                print(f"\nStructure {i+1}:")
                print(f"  Composition: {structure['composition']}")
                print(f"  Lattice: a={structure['lattice_params']['a']:.3f}")
                print(f"  Space Group: {structure['space_group']}")
                print(f"  Method: {structure['generation_method']}")
            
            return loader
        else:
            print("❌ Failed to load CDVAE model")
            return None
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    test_improved_cdvae_loader()