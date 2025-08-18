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
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        
        # Create model instance from hyperparameters
        model = EnhancedCDVAE.load_from_checkpoint(
            self.checkpoint_path,
            map_location=device
        )
        
        # Set to evaluation mode
        model.eval()
        model.to(device)
        
        self.model = model
        print("✅ Model loaded successfully!")
        return model
    
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
            lattice_scaler = torch.load(lattice_scaler_path)
            print("✅ Lattice scaler loaded")
        else:
            print("⚠️  Lattice scaler not found")
            
        if prop_scaler_path.exists():
            prop_scaler = torch.load(prop_scaler_path)
            print("✅ Property scaler loaded")
        else:
            print("⚠️  Property scaler not found")
            
        self.lattice_scaler = lattice_scaler
        self.prop_scaler = prop_scaler
        
        return lattice_scaler, prop_scaler
    
    def generate_structures(self, num_samples=10, num_atoms=None):
        """
        Generate new crystal structures using the trained model.
        
        Args:
            num_samples (int): Number of structures to generate
            num_atoms (int, optional): Target number of atoms
            
        Returns:
            list: Generated crystal structures
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        print(f"Generating {num_samples} crystal structures...")
        
        with torch.no_grad():
            # Generate structures
            generated = self.model.sample(
                num_samples=num_samples,
                num_atoms=num_atoms
            )
            
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
    CHECKPOINT_PATH = "generator/CDVAE/outputs/singlerun/2025-08-18/enhanced_cdvae/generator/CDVAE/outputs/singlerun/2025-08-18/enhanced_cdvae/epoch=136-step=14659.ckpt"
    SCALERS_DIR = "generator/CDVAE/outputs/singlerun/2025-08-18/enhanced_cdvae/generator/CDVAE/outputs/singlerun/2025-08-18/enhanced_cdvae/"
    
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