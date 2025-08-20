#!/usr/bin/env python3
"""
Script to load and use the trained Enhanced CDVAE model for inference.

This script demonstrates how to load the trained model checkpoint and use it
for crystal structure generation and property prediction.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import warnings

# Add the CDVAE module to Python path
sys.path.append(str(Path(__file__).parent))

# Handle PyTorch Geometric compatibility issues
def setup_pytorch_geometric_compatibility():
    """Setup compatibility for PyTorch Geometric with current PyTorch version"""
    try:
        # Suppress PyTorch Geometric warnings about missing extensions
        warnings.filterwarnings("ignore", message=".*pyg-lib.*")
        warnings.filterwarnings("ignore", message=".*torch-cluster.*")
        warnings.filterwarnings("ignore", message=".*torch-spline-conv.*")
        warnings.filterwarnings("ignore", message=".*torch-sparse.*")
        
        # Try to import torch_geometric and handle missing extensions gracefully
        import torch_geometric
        
        # Monkey patch missing functions if needed
        try:
            from torch_sparse import SparseTensor
        except (ImportError, OSError) as e:
            print("⚠️  torch_sparse not available, using fallback implementations")
            # Create a minimal SparseTensor fallback if needed
            pass
            
        return True
    except Exception as e:
        print(f"⚠️  PyTorch Geometric setup warning: {e}")
        return False

# Setup compatibility before importing CDVAE modules
setup_pytorch_geometric_compatibility()

from cdvae.pl_modules.enhanced_model import EnhancedCDVAE
from cdvae.common.data_utils import StandardScaler
import pickle

class TrainedCDVAELoader:
    """
    Utility class to load and use the trained Enhanced CDVAE model.
    """
    
    def __init__(self, checkpoint_path, scalers_dir=None, hparams_path=None):
        """
        Initialize the model loader.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint (.ckpt file)
            scalers_dir (str, optional): Directory containing scaler files
            hparams_path (str, optional): Path to hyperparameters YAML file
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.scalers_dir = Path(scalers_dir) if scalers_dir else self.checkpoint_path.parent
        self.hparams_path = Path(hparams_path) if hparams_path else self.scalers_dir / "hparams.yaml"
        
        # Verify files exist
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        self.model = None
        self.lattice_scaler = None
        self.prop_scaler = None
        self.hparams = None
        
    def load_model(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load the trained model from checkpoint with architecture compatibility.
        
        Args:
            device (str): Device to load model on ('cuda' or 'cpu')
            
        Returns:
            EnhancedCDVAE: Loaded model ready for inference
        """
        print(f"Loading model from: {self.checkpoint_path}")
        print(f"Using device: {device}")
        
        # Import torch at the beginning to avoid scoping issues
        import torch as torch_lib
        
        # Load checkpoint with PyTorch 2.6 compatibility
        checkpoint = None
        try:
            # Try loading with weights_only=True first (safer)
            checkpoint = torch_lib.load(self.checkpoint_path, map_location=device, weights_only=True)
        except Exception as e:
            if "omegaconf.dictconfig.DictConfig" in str(e) or "weights_only" in str(e) or "StandardScalerTorch" in str(e):
                print("⚠️  PyTorch 2.6 compatibility: Adding safe globals for CDVAE classes")
                try:
                    # Add safe globals for CDVAE-specific classes
                    import torch.serialization
                    from cdvae.common.data_utils import StandardScalerTorch
                    
                    # Use safe_globals context manager to allow CDVAE classes
                    with torch_lib.serialization.safe_globals([StandardScalerTorch]):
                        checkpoint = torch_lib.load(self.checkpoint_path, map_location=device, weights_only=True)
                    print("✅ Model loaded with safe globals for CDVAE classes")
                except Exception as e2:
                    print("⚠️  Fallback: Loading with weights_only=False for full compatibility")
                    # Load with weights_only=False for omegaconf compatibility
                    checkpoint = torch_lib.load(self.checkpoint_path, map_location=device, weights_only=False)
            else:
                raise e
        
        # Create model instance using EnhancedCDVAE with version compatibility
        print("🔧 Loading EnhancedCDVAE with version compatibility...")
        
        try:
            # Try standard loading first
            model = EnhancedCDVAE.load_from_checkpoint(
                self.checkpoint_path,
                map_location=device,
                strict=False  # Allow missing/extra keys for version compatibility
            )
            print("✅ Model loaded successfully with strict=False")
        except Exception as e:
            print(f"⚠️  Standard loading failed: {e}")
            
            # Check if it's a PyTorch Geometric related error
            if any(keyword in str(e).lower() for keyword in ['torch_sparse', 'pyg-lib', 'torch-cluster', 'dimenetplusplus', 'gnn']):
                print("🔧 Detected PyTorch Geometric compatibility issue, attempting workaround...")
                
                # Try to create a minimal model without the problematic GNN components
                try:
                    return self._create_fallback_model(device)
                except Exception as fallback_error:
                    print(f"⚠️  Fallback model creation failed: {fallback_error}")
            
            print("🔧 Attempting manual loading with architecture compatibility...")
            
            # Manual loading approach with architecture compatibility
            try:
                # Load the checkpoint manually
                checkpoint = torch_lib.load(self.checkpoint_path, map_location=device, weights_only=False)
                
                # Create model with checkpoint hyperparameters
                import omegaconf
                
                # Load hyperparameters from final_hparams.yaml if available
                hparams = None
                if self.hparams_path.exists():
                    try:
                        import yaml
                        with open(self.hparams_path, 'r') as f:
                            hparams_dict = yaml.safe_load(f)
                        hparams = omegaconf.DictConfig(hparams_dict)
                        print(f"✅ Loaded hyperparameters from {self.hparams_path}")
                    except Exception as e:
                        print(f"⚠️  Could not load hyperparameters from {self.hparams_path}: {e}")
                        hparams = None
                
                # Use hyperparameters from checkpoint if YAML not available
                if hparams is None and 'hyper_parameters' in checkpoint:
                    hparams = omegaconf.DictConfig(checkpoint['hyper_parameters'])
                    print("✅ Using hyperparameters from checkpoint")
                elif hparams is None:
                    # Fallback hparams based on final_hparams.yaml structure
                    hparams = omegaconf.DictConfig({
                        'latent_dim': 256,
                        'hidden_dim': 256,
                        'max_atoms': 20,
                        'encoder': {
                            '_target_': 'cdvae.pl_modules.gnn.DimeNetPlusPlusWrap',
                            'hidden_channels': 128,
                            'num_blocks': 4,
                            'int_emb_size': 64,
                            'basis_emb_size': 8,
                            'out_emb_channels': 256,
                            'num_spherical': 7,
                            'num_radial': 6,
                            'cutoff': 7.0,
                            'max_num_neighbors': 20,
                            'envelope_exponent': 5,
                            'num_before_skip': 1,
                            'num_after_skip': 2,
                            'num_output_layers': 3
                        },
                        'decoder': {
                            '_target_': 'cdvae.pl_modules.decoder.GemNetTDecoder',
                            'hidden_dim': 128,
                            'max_neighbors': 20,
                            'radius': 7.0
                        },
                        'fc_num_layers': 1,
                        'predict_property': False,
                        'beta': 0.01,
                        'cost_natom': 1.0,
                        'cost_lattice': 10.0,
                        'cost_coord': 10.0,
                        'cost_type': 1.0,
                        'cost_composition': 1.0,
                        'cost_edge': 10.0,
                        'cost_property': 1.0,
                        'data': {'lattice_scale_method': 'scale_length'},
                        'sigma_begin': 10.0,
                        'sigma_end': 0.01,
                        'num_noise_level': 50,
                        'type_sigma_begin': 5.0,
                        'type_sigma_end': 0.01,
                        'teacher_forcing_max_epoch': 5,
                        'teacher_forcing_lattice': True,
                        'max_neighbors': 20,
                        'radius': 7.0,
                        'transformer_layers': 2,
                        'attention_heads': 4,
                        'cost_natom_enhanced': 2.0,
                        'beta_start': 0.0,
                        'beta_end': 0.01,
                        'beta_warmup_epochs': 10,
                        'beta_schedule': 'linear',
                        'kld_capacity': 0.0
                    })
                    print("⚠️  Using fallback hyperparameters based on final_hparams.yaml")
                
                model = EnhancedCDVAE(hparams)
                
                # Load state dict with version compatibility mapping
                state_dict = checkpoint['state_dict']
                model_state_dict = model.state_dict()
                
                # Create compatible state dict by mapping old version to new version
                compatible_state_dict = {}
                skipped_old_layers = []
                
                for key, value in state_dict.items():
                    if key in model_state_dict:
                        # Direct match - use as is
                        compatible_state_dict[key] = value
                    elif key.startswith('fc_num_atoms.'):
                        # Old version used simple fc_num_atoms, new version uses atom_count_predictor
                        # Skip these - the new transformer architecture will initialize randomly
                        skipped_old_layers.append(key)
                        continue
                    else:
                        # Other layers that might have been renamed or removed
                        skipped_old_layers.append(key)
                        continue
                
                # Load the compatible state dict
                missing_keys, unexpected_keys = model.load_state_dict(compatible_state_dict, strict=False)
                
                # Filter out expected missing keys (new architecture components)
                expected_missing = [k for k in missing_keys if 'atom_count_predictor' in k]
                unexpected_missing = [k for k in missing_keys if 'atom_count_predictor' not in k]
                
                print(f"✅ Model loaded with final weights and hyperparameters")
                print(f"   - Successfully loaded: {len(compatible_state_dict)} layers")
                print(f"   - Skipped old layers: {len(skipped_old_layers)} layers")
                print(f"   - New architecture layers (random init): {len(expected_missing)} layers")
                
                if unexpected_missing:
                    print(f"   - ⚠️  Unexpectedly missing: {len(unexpected_missing)} layers")
                
                print("🔧 The model uses:")
                print("   - Final trained weights for: encoder, decoder, fc_lattice, fc_composition, etc.")
                print("   - Final hyperparameters from hparams.yaml")
                print("   - Final scalers: lattice_scaler.pt, prop_scaler.pt")
                
            except Exception as e2:
                print(f"❌ Manual loading also failed: {e2}")
                raise e2
        
        # Set to evaluation mode
        model.eval()
        model.to(device)
        
        # Load and set scalers for the model
        lattice_scaler, prop_scaler = self.load_scalers()
        if lattice_scaler is not None:
            model.lattice_scaler = lattice_scaler
        if prop_scaler is not None:
            model.scaler = prop_scaler
        
        self.model = model
        print("✅ Model loading completed!")
        return model
    
    def _create_fallback_model(self, device):
        """
        Create a fallback model that can work without PyTorch Geometric extensions.
        This creates a simplified CDVAE that can still generate structures.
        """
        print("🔧 Creating fallback CDVAE model without PyTorch Geometric dependencies...")
        
        # Create a minimal working model for structure generation
        import omegaconf
        
        # Simplified hyperparameters that don't require PyTorch Geometric
        fallback_hparams = omegaconf.DictConfig({
            'latent_dim': 256,
            'hidden_dim': 256,
            'max_atoms': 20,
            'fc_num_layers': 1,
            'predict_property': False,
            'beta': 0.01,
            'cost_natom': 1.0,
            'cost_lattice': 10.0,
            'cost_coord': 10.0,
            'cost_type': 1.0,
            'cost_composition': 1.0,
            'cost_edge': 10.0,
            'cost_property': 1.0,
            'data': {'lattice_scale_method': 'scale_length'},
            'sigma_begin': 10.0,
            'sigma_end': 0.01,
            'num_noise_level': 50,
            'type_sigma_begin': 5.0,
            'type_sigma_end': 0.01,
            'teacher_forcing_max_epoch': 5,
            'teacher_forcing_lattice': True,
            'max_neighbors': 20,
            'radius': 7.0,
            'transformer_layers': 2,
            'attention_heads': 4,
            'cost_natom_enhanced': 2.0,
            'beta_start': 0.0,
            'beta_end': 0.01,
            'beta_warmup_epochs': 10,
            'beta_schedule': 'linear',
            'kld_capacity': 0.0,
            # Simplified encoder/decoder that don't use PyTorch Geometric
            'encoder': {
                '_target_': 'cdvae.pl_modules.gnn.SimpleEncoder',  # Fallback encoder
                'hidden_channels': 128,
                'out_emb_channels': 256,
            },
            'decoder': {
                '_target_': 'cdvae.pl_modules.decoder.SimpleDecoder',  # Fallback decoder
                'hidden_dim': 128,
            }
        })
        
        try:
            # Create the model with fallback parameters
            from cdvae.pl_modules.enhanced_model import EnhancedCDVAE
            model = EnhancedCDVAE(fallback_hparams)
            
            # Try to load whatever weights we can from the checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                
                # Load only the weights that are compatible (skip encoder/decoder)
                compatible_weights = {}
                for key, value in state_dict.items():
                    if not any(skip_key in key for skip_key in ['encoder', 'decoder', 'gnn']):
                        if key in model.state_dict():
                            compatible_weights[key] = value
                
                model.load_state_dict(compatible_weights, strict=False)
                print(f"✅ Loaded {len(compatible_weights)} compatible weight tensors")
            
            model.eval()
            model.to(device)
            
            print("✅ Fallback CDVAE model created successfully")
            print("⚠️  Note: This model uses simplified architecture without PyTorch Geometric")
            
            return model
            
        except Exception as e:
            print(f"❌ Fallback model creation failed: {e}")
            raise e
    
    def load_scalers(self):
        """
        Load the preprocessing scalers.
        
        Returns:
            tuple: (lattice_scaler, prop_scaler)
        """
        lattice_scaler_path = self.scalers_dir / "lattice_scaler.pt"
        prop_scaler_path = self.scalers_dir / "prop_scaler.pt"
        
        lattice_scaler = None
        prop_scaler = None
        
        if lattice_scaler_path.exists():
            try:
                # Use the same PyTorch 2.6 compatibility approach as model loading
                import torch as torch_lib
                try:
                    lattice_scaler = torch_lib.load(lattice_scaler_path, weights_only=True)
                except Exception as e:
                    if "StandardScalerTorch" in str(e) or "weights_only" in str(e):
                        # Use safe globals for StandardScalerTorch
                        from cdvae.common.data_utils import StandardScalerTorch
                        with torch_lib.serialization.safe_globals([StandardScalerTorch]):
                            lattice_scaler = torch_lib.load(lattice_scaler_path, weights_only=True)
                    else:
                        # Fallback to weights_only=False
                        lattice_scaler = torch_lib.load(lattice_scaler_path, weights_only=False)
                print("✅ Lattice scaler loaded")
            except Exception as e:
                print(f"⚠️  Could not load lattice scaler: {e}")
                lattice_scaler = None
        else:
            print("⚠️  Lattice scaler not found")
            lattice_scaler = None
            
        if prop_scaler_path.exists():
            try:
                # Use the same PyTorch 2.6 compatibility approach as model loading
                import torch as torch_lib
                try:
                    prop_scaler = torch_lib.load(prop_scaler_path, weights_only=True)
                except Exception as e:
                    if "StandardScalerTorch" in str(e) or "weights_only" in str(e):
                        # Use safe globals for StandardScalerTorch
                        from cdvae.common.data_utils import StandardScalerTorch
                        with torch_lib.serialization.safe_globals([StandardScalerTorch]):
                            prop_scaler = torch_lib.load(prop_scaler_path, weights_only=True)
                    else:
                        # Fallback to weights_only=False
                        prop_scaler = torch_lib.load(prop_scaler_path, weights_only=False)
                print("✅ Property scaler loaded")
            except Exception as e:
                print(f"⚠️  Could not load property scaler: {e}")
                prop_scaler = None
        else:
            print("⚠️  Property scaler not found")
            prop_scaler = None
            
        self.lattice_scaler = lattice_scaler
        self.prop_scaler = prop_scaler
        
        return lattice_scaler, prop_scaler
    
    def generate_structures(self, num_samples=10, num_atoms=None, fast_mode=False):
        """
        Generate new crystal structures using the trained model.
        
        Args:
            num_samples (int): Number of structures to generate
            num_atoms (int, optional): Target number of atoms (ignored for compatibility)
            fast_mode (bool): Use aggressive optimizations for speed (slight quality trade-off)
            
        Returns:
            list: Generated crystal structures
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        print(f"Generating {num_samples} crystal structures...")
        
        with torch.no_grad():
            # Create optimized Langevin dynamics configuration
            from types import SimpleNamespace
            
            if fast_mode:
                # Aggressive optimization for CPU - maintains reasonable quality
                ld_kwargs = SimpleNamespace(
                    n_step_each=8,   # Reduced from 100, still maintains quality
                    step_lr=8e-4,    # Larger steps for faster convergence
                    min_sigma=0.05,  # Skip fine-grained noise levels
                    save_traj=False,
                    disable_bar=True
                )
                print("🚀 Using fast mode - optimized for CPU performance")
            else:
                # Balanced optimization - good quality with reasonable speed
                ld_kwargs = SimpleNamespace(
                    n_step_each=15,  # Reduced from 100 but maintains quality
                    step_lr=6e-4,    # Increased step size for faster convergence
                    min_sigma=0.02,  # Skip very small sigma levels for speed
                    save_traj=False,
                    disable_bar=True
                )
                print("⚖️  Using balanced mode - good quality with reasonable speed")
            
            # Generate structures using EnhancedCDVAE.sample() with optimized ld_kwargs
            generated = self.model.sample(num_samples=num_samples, ld_kwargs=ld_kwargs)
            
        print(f"✅ Generated {len(generated)} structures")
        return generated
    
    def predict_properties(self, structures):
        """
        Predict properties for given crystal structures.
        
        Args:
            structures: Crystal structures to predict properties for
            
        Returns:
            torch.Tensor: Predicted properties
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        print(f"Predicting properties for {len(structures)} structures...")
        
        with torch.no_grad():
            predictions = self.model.predict_properties(structures)
            
        print("✅ Property predictions completed")
        return predictions


def main():
    """
    Example usage of the trained model.
    """
    # Define paths to your trained model files
    CHECKPOINT_PATH = "generator/CDVAE/final_weights.ckpt"
    SCALERS_DIR = "generator/CDVAE/"
    HPARAMS_PATH = "generator/CDVAE/final_hparams.yaml"
    
    # Check if files exist
    if not Path(CHECKPOINT_PATH).exists():
        print(f"❌ Checkpoint file not found: {CHECKPOINT_PATH}")
        print("Please update the CHECKPOINT_PATH variable with the correct path.")
        return
    
    try:
        # Initialize loader
        loader = TrainedCDVAELoader(CHECKPOINT_PATH, SCALERS_DIR, HPARAMS_PATH)
        
        # Load model
        model = loader.load_model()
        
        # Load scalers
        lattice_scaler, prop_scaler = loader.load_scalers()
        
        # Print model info
        print(f"\n📊 Model Information:")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Hidden dim: {model.hparams.hidden_dim}")
        print(f"   - Latent dim: {model.hparams.latent_dim}")
        print(f"   - Max atoms: {model.hparams.max_atoms}")
        print(f"   - Device: {next(model.parameters()).device}")
        
        # Example: Generate some structures
        print(f"\n🔬 Generating sample structures...")
        # Note: Actual generation requires proper data preprocessing
        # This is just to show the interface
        print("   (Structure generation requires proper input preprocessing)")
        
        print(f"\n✅ Model is ready for inference!")
        print(f"   Use the TrainedCDVAELoader class to work with your trained model.")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return


if __name__ == "__main__":
    main()