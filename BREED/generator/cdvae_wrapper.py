#!/usr/bin/env python3
"""
Simple wrapper for CDVAE model to be used in genetic algorithm.
This provides a clean interface for structure generation.
"""

import sys
import os
from pathlib import Path

# Add CDVAE to path
sys.path.append(str(Path(__file__).parent))

from load_trained_model import TrainedCDVAELoader

class CDVAEWrapper:
    """Simple wrapper for CDVAE structure generation"""
    
    def __init__(self):
        """Initialize the CDVAE wrapper with the trained model"""
        # Use the enhanced production model
        base_path = Path(__file__).parent
        checkpoint_path = base_path / "outputs/singlerun/2025-08-20/enhanced_cdvae_production_200epochs/outputs/singlerun/2025-08-20/enhanced_cdvae_production_200epochs/epoch=26-step=2889.ckpt"
        hparams_path = base_path / "outputs/singlerun/2025-08-20/enhanced_cdvae_production_200epochs/outputs/singlerun/2025-08-20/enhanced_cdvae_production_200epochs/hparams.yaml"
        scalers_dir = base_path / "outputs/singlerun/2025-08-20/enhanced_cdvae_production_200epochs/outputs/singlerun/2025-08-20/enhanced_cdvae_production_200epochs/"
        
        self.loader = TrainedCDVAELoader(
            checkpoint_path=str(checkpoint_path),
            scalers_dir=str(scalers_dir),
            hparams_path=str(hparams_path)
        )
        self.model = None
        
    def load_model(self, device='cpu'):
        """Load the CDVAE model"""
        if self.model is None:
            print("Loading CDVAE model...")
            self.model = self.loader.load_model(device=device)
            print("✅ CDVAE model loaded successfully")
        return self.model
    
    def generate_structures(self, num_samples=1, fast_mode=True):
        """
        Generate crystal structures using CDVAE
        
        Args:
            num_samples (int): Number of structures to generate
            fast_mode (bool): Use fast generation mode
            
        Returns:
            dict: Generated structures with keys:
                - num_atoms: torch.Tensor of atom counts
                - lengths: torch.Tensor of lattice parameters
                - angles: torch.Tensor of lattice angles  
                - frac_coords: torch.Tensor of fractional coordinates
                - atom_types: torch.Tensor of atom types
        """
        if self.model is None:
            self.load_model()
            
        return self.loader.generate_structures(
            num_samples=num_samples,
            fast_mode=fast_mode
        )

# Global instance for easy import
_cdvae_wrapper = None

def get_cdvae_wrapper():
    """Get the global CDVAE wrapper instance"""
    global _cdvae_wrapper
    if _cdvae_wrapper is None:
        _cdvae_wrapper = CDVAEWrapper()
    return _cdvae_wrapper

def generate_structures_simple(num_samples=1):
    """Simple function to generate structures"""
    wrapper = get_cdvae_wrapper()
    return wrapper.generate_structures(num_samples=num_samples, fast_mode=True)

if __name__ == "__main__":
    # Test the wrapper
    print("Testing CDVAE wrapper...")
    wrapper = CDVAEWrapper()
    wrapper.load_model()
    
    structures = wrapper.generate_structures(num_samples=1)
    print(f"Generated structures: {type(structures)}")
    print(f"Keys: {list(structures.keys())}")
    print(f"Number of atoms: {structures['num_atoms']}")
    print("✅ CDVAE wrapper test successful!")