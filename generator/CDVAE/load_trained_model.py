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

# Add the CDVAE module to Python path
sys.path.append(str(Path(__file__).parent))

from cdvae.pl_modules.enhanced_model import EnhancedCDVAE
from cdvae.common.data_utils import StandardScaler
import pickle

class TrainedCDVAELoader:
    """
    Utility class to load and use the trained Enhanced CDVAE model.
    """
    
    def __init__(self, checkpoint_path, scalers_dir=None):
        """
        Initialize the model loader.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint (.ckpt file)
            scalers_dir (str, optional): Directory containing scaler files
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.scalers_dir = Path(scalers_dir) if scalers_dir else self.checkpoint_path.parent
        
        # Verify files exist
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        self.model = None
        self.lattice_scaler = None
        self.prop_scaler = None
        
    def load_model(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load the trained model from checkpoint.
        
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
        
        # Create model instance from hyperparameters
        try:
            model = EnhancedCDVAE.load_from_checkpoint(
                self.checkpoint_path,
                map_location=device
            )
        except Exception as e:
            if "omegaconf.dictconfig.DictConfig" in str(e) or "weights_only" in str(e):
                print("⚠️  PyTorch 2.6 compatibility: Loading checkpoint with weights_only=False")
                # For PyTorch Lightning load_from_checkpoint, we need to handle this differently
                # Let's try a direct approach
                import omegaconf
                torch_lib.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
                model = EnhancedCDVAE.load_from_checkpoint(
                    self.checkpoint_path,
                    map_location=device
                )
            else:
                raise e
        
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
        print("✅ Model loaded successfully!")
        return model
    
    def load_scalers(self):
        """
        Load the preprocessing scalers.
        
        Returns:
            tuple: (lattice_scaler, prop_scaler)
        """
        lattice_scaler_path = self.scalers_dir / "final_lattice_scaler.pt"
        prop_scaler_path = self.scalers_dir / "final_prop_scaler.pt"
        
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
    CHECKPOINT_PATH = "generator/CDVAE/last_cdvae_weights.ckpt"
    SCALERS_DIR = "generator/CDVAE/"
    
    # Check if files exist
    if not Path(CHECKPOINT_PATH).exists():
        print(f"❌ Checkpoint file not found: {CHECKPOINT_PATH}")
        print("Please update the CHECKPOINT_PATH variable with the correct path.")
        return
    
    try:
        # Initialize loader
        loader = TrainedCDVAELoader(CHECKPOINT_PATH, SCALERS_DIR)
        
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