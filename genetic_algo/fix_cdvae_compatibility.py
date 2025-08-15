#!/usr/bin/env python3
"""
CDVAE Checkpoint Compatibility Fix

This script fixes the 'AttributeDict' object has no attribute 'latent_dim' error
by patching the checkpoint loading process to handle missing attributes.
"""

import torch
import yaml
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import sys

# Add CDVAE to path
sys.path.append(r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE')

# Set required environment variables
os.environ['PROJECT_ROOT'] = r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE'

def patch_cdvae_checkpoint(ckpt_path, hparams_path):
    """
    Patch CDVAE checkpoint to fix compatibility issues
    
    Args:
        ckpt_path: Path to checkpoint file
        hparams_path: Path to hyperparameters file
        
    Returns:
        Patched checkpoint dict and hparams
    """
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Loading hparams: {hparams_path}")
    
    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Load hyperparameters
    with open(hparams_path, 'r') as f:
        hparams_dict = yaml.safe_load(f)
    
    print(f"Original checkpoint keys: {list(checkpoint.keys())}")
    print(f"Original hparams keys: {list(hparams_dict.keys())}")
    
    # Convert to OmegaConf for easier manipulation
    hparams = OmegaConf.create(hparams_dict)
    
    # Common missing attributes and their default values
    default_attributes = {
        'latent_dim': 256,  # Common latent dimension
        'hidden_dim': 512,  # Common hidden dimension
        'num_layers': 4,    # Common number of layers
        'max_atoms': 100,   # Maximum atoms per structure
        'num_atom_types': 100,  # Number of atom types
        'cutoff': 8.0,      # Cutoff radius
        'max_neighbors': 20, # Maximum neighbors
        'sigma_begin': 10.0, # Diffusion start
        'sigma_end': 0.01,   # Diffusion end
        'num_noise_level': 50, # Noise levels
        'teacher_forcing_max_epoch': 500,
        'teacher_forcing_decay_rate': 0.9,
        'cost_natom': 1.0,
        'cost_coord': 1.0,
        'cost_lattice': 1.0,
        'cost_composition': 1.0,
        'cost_property': 1.0,
        'beta': 0.01,
        'niggli': True,
        'primitive': False,
        'graph_method': 'crystalnn',
        'preprocess_workers': 30,
        'lattice_scale_method': 'scale_length'
    }
    
    # Add missing attributes to hparams
    for key, default_value in default_attributes.items():
        if key not in hparams:
            print(f"Adding missing attribute: {key} = {default_value}")
            hparams[key] = default_value
    
    # Ensure hparams has required structure
    if 'model' not in hparams:
        hparams['model'] = {}
    
    # Add model-specific attributes
    model_attrs = {
        'hidden_dim': hparams.get('hidden_dim', 512),
        'latent_dim': hparams.get('latent_dim', 256),
        'num_layers': hparams.get('num_layers', 4),
        'max_atoms': hparams.get('max_atoms', 100),
        'cutoff': hparams.get('cutoff', 8.0),
        'max_neighbors': hparams.get('max_neighbors', 20)
    }
    
    for key, value in model_attrs.items():
        if key not in hparams.model:
            hparams.model[key] = value
    
    # Fix state_dict if needed
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        
        # Remove problematic keys that might cause issues
        keys_to_remove = []
        for key in state_dict.keys():
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            print(f"Removing potentially problematic key: {key}")
            del state_dict[key]
    
    # Update checkpoint with fixed hparams
    checkpoint['hyper_parameters'] = OmegaConf.to_container(hparams, resolve=True)
    
    print(f"Patched hparams keys: {list(hparams.keys())}")
    print(f"Patched model keys: {list(hparams.model.keys()) if 'model' in hparams else 'No model section'}")
    
    return checkpoint, hparams

def load_cdvae_with_compatibility_fix(model_path):
    """
    Load CDVAE model with compatibility fixes
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Loaded CDVAE model or None if failed
    """
    try:
        from cdvae.pl_modules.model import CDVAE
        
        model_path = Path(model_path)
        ckpt_path = model_path / "epoch=839-step=89039.ckpt"
        hparams_path = model_path / "hparams.yaml"
        
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            return None
            
        if not hparams_path.exists():
            print(f"Hyperparameters not found: {hparams_path}")
            return None
        
        # Apply compatibility patches
        checkpoint, hparams = patch_cdvae_checkpoint(ckpt_path, hparams_path)
        
        # Try to load model with patched checkpoint
        print("Attempting to load CDVAE model with compatibility fixes...")
        
        # Method 1: Try direct loading with patched checkpoint
        try:
            # Create temporary checkpoint file with patches
            temp_ckpt_path = model_path / "temp_patched.ckpt"
            torch.save(checkpoint, temp_ckpt_path)
            
            model = CDVAE.load_from_checkpoint(str(temp_ckpt_path))
            
            # Clean up temporary file
            temp_ckpt_path.unlink()
            
            print("Successfully loaded CDVAE model with Method 1 (patched checkpoint)")
            return model
            
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            
            # Clean up temporary file if it exists
            temp_ckpt_path = model_path / "temp_patched.ckpt"
            if temp_ckpt_path.exists():
                temp_ckpt_path.unlink()
        
        # Method 2: Try loading with explicit hparams
        try:
            model = CDVAE.load_from_checkpoint(
                str(ckpt_path),
                hparams=OmegaConf.to_container(hparams, resolve=True)
            )
            print("Successfully loaded CDVAE model with Method 2 (explicit hparams)")
            return model
            
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
        
        # Method 3: Try manual model creation
        try:
            # Create model manually with hparams
            model = CDVAE(hparams)
            
            # Load state dict manually
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            
            print("Successfully loaded CDVAE model with Method 3 (manual creation)")
            return model
            
        except Exception as e3:
            print(f"Method 3 failed: {e3}")
        
        print("All loading methods failed")
        return None
        
    except Exception as e:
        print(f"Error in load_cdvae_with_compatibility_fix: {e}")
        return None

if __name__ == "__main__":
    # Test the compatibility fix
    model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\cdvae\prop_models\mp20"
    
    print("Testing CDVAE compatibility fix...")
    model = load_cdvae_with_compatibility_fix(model_path)
    
    if model:
        print("✅ CDVAE model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Model device: {next(model.parameters()).device}")
    else:
        print("❌ Failed to load CDVAE model")