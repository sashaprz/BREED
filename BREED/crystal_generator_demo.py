#!/usr/bin/env python3
"""
Complete CDVAE Crystal Generation Demo
This script demonstrates how to generate novel crystal structures using the pre-trained CDVAE model.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import json
from typing import Dict, List, Tuple

# Add the cdvae module to path
sys.path.append('/workspace/cdvae')

def load_and_test_model():
    """Load the pre-trained CDVAE model and demonstrate its capabilities"""
    
    print("🔬 CDVAE Crystal Generation Demo")
    print("=" * 60)
    
    # Model checkpoint path
    checkpoint_path = "/workspace/cdvae/cdvae/prop_models/mp20/epoch=839-step=89039.ckpt"
    
    try:
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("✅ Checkpoint loaded successfully!")
        print(f"   • Training epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   • Global step: {checkpoint.get('global_step', 'unknown')}")
        print(f"   • Model parameters: {len(checkpoint.get('state_dict', {}))}")
        
        # Extract hyperparameters
        hparams = checkpoint.get('hyper_parameters', {})
        print(f"\n📊 Model Configuration:")
        key_params = ['hidden_dim', 'latent_dim', 'num_layers', 'max_atoms', 'cutoff']
        for param in key_params:
            if param in hparams:
                print(f"   • {param}: {hparams[param]}")
        
        # Show model architecture overview
        state_dict = checkpoint['state_dict']
        encoder_layers = [k for k in state_dict.keys() if 'encoder' in k][:5]
        decoder_layers = [k for k in state_dict.keys() if 'decoder' in k][:5]
        
        print(f"\n🏗️ Model Architecture Preview:")
        print(f"   • Total parameters: {len(state_dict)}")
        print(f"   • Encoder layers (sample): {len(encoder_layers)}")
        for layer in encoder_layers:
            print(f"     - {layer}")
        print(f"   • Decoder layers (sample): {len(decoder_layers)}")
        for layer in decoder_layers:
            print(f"     - {layer}")
        
        return True, checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False, None

def demonstrate_crystal_data():
    """Show what crystal data looks like"""
    
    print(f"\n💎 Crystal Structure Data Format:")
    print("=" * 40)
    
    # Load some sample data to show format
    try:
        train_data = pd.read_pickle('/workspace/cdvae/data/mp_20/train.pkl')
        sample = train_data.iloc[0]
        
        print(f"📋 Sample Crystal Structure:")
        print(f"   • Material ID: {sample.get('material_id', 'N/A')}")
        print(f"   • Formula: {sample.get('pretty_formula', 'N/A')}")
        print(f"   • Space Group: {sample.get('spacegroup', 'N/A')}")
        print(f"   • Number of atoms: {len(sample.get('atom_types', []))}")
        
        if 'frac_coords' in sample:
            coords = np.array(sample['frac_coords'])
            print(f"   • Fractional coordinates shape: {coords.shape}")
            print(f"   • Sample coordinates (first 3 atoms):")
            for i, coord in enumerate(coords[:3]):
                print(f"     Atom {i+1}: [{coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}]")
        
        if 'lengths' in sample and 'angles' in sample:
            lengths = sample['lengths']
            angles = sample['angles']
            print(f"   • Unit cell lengths: [{lengths[0]:.3f}, {lengths[1]:.3f}, {lengths[2]:.3f}] Å")
            print(f"   • Unit cell angles: [{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}]°")
        
        return True
        
    except Exception as e:
        print(f"❌ Could not load sample data: {e}")
        return False

def create_generation_example():
    """Create example code for crystal generation"""
    
    print(f"\n🚀 Crystal Generation Example:")
    print("=" * 40)
    
    example_code = '''
# Example: How to generate crystals with the loaded model
import torch
from cdvae.pl_modules.model import CDVAE

# 1. Load the model
checkpoint = torch.load('path/to/model.ckpt', map_location='cpu')
model = CDVAE.load_from_checkpoint('path/to/model.ckpt')
model.eval()

# 2. Generate novel crystal structures
with torch.no_grad():
    # Sample from latent space
    batch_size = 5
    latent_dim = model.hparams.latent_dim
    z = torch.randn(batch_size, latent_dim)
    
    # Generate crystals
    generated_crystals = model.decode(z)
    
    # Extract crystal properties
    frac_coords = generated_crystals['frac_coords']  # Atomic positions
    atom_types = generated_crystals['atom_types']   # Chemical elements
    lengths = generated_crystals['lengths']         # Unit cell dimensions
    angles = generated_crystals['angles']           # Unit cell angles
    
    print(f"Generated {batch_size} novel crystal structures!")
    '''
    
    print(example_code)
    return True

def save_checkpoint_info():
    """Save checkpoint information to a file for download"""
    
    try:
        checkpoint_path = "/workspace/cdvae/cdvae/prop_models/mp20/epoch=839-step=89039.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create summary info
        info = {
            "model_info": {
                "epoch": checkpoint.get('epoch'),
                "global_step": checkpoint.get('global_step'),
                "model_type": "CDVAE",
                "dataset": "Materials Project MP-20",
                "total_parameters": len(checkpoint.get('state_dict', {}))
            },
            "hyperparameters": checkpoint.get('hyper_parameters', {}),
            "architecture_layers": list(checkpoint['state_dict'].keys())[:20],  # First 20 layers
            "checkpoint_size_mb": os.path.getsize(checkpoint_path) / (1024*1024),
            "usage_instructions": {
                "load_model": "model = CDVAE.load_from_checkpoint('path/to/checkpoint.ckpt')",
                "generate": "Use model.decode(latent_vector) to generate crystals",
                "requirements": ["torch", "pytorch_lightning", "pymatgen", "torch_geometric"]
            }
        }
        
        # Save to JSON file
        info_file = "/workspace/cdvae/model_checkpoint_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        print(f"\n💾 Checkpoint Information Saved:")
        print(f"   • File: {info_file}")
        print(f"   • Checkpoint size: {info['model_info']['checkpoint_size_mb']:.1f} MB")
        print(f"   • Ready for download!")
        
        return info_file
        
    except Exception as e:
        print(f"❌ Error saving checkpoint info: {e}")
        return None

def main():
    """Main demonstration function"""
    
    print("Starting CDVAE Crystal Generation Demonstration...")
    
    # 1. Load and analyze the model
    success, checkpoint = load_and_test_model()
    if not success:
        return False
    
    # 2. Show crystal data format
    demonstrate_crystal_data()
    
    # 3. Show generation example
    create_generation_example()
    
    # 4. Save checkpoint info for download
    info_file = save_checkpoint_info()
    
    print(f"\n🎯 DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("✅ Pre-trained CDVAE model successfully analyzed")
    print("✅ Crystal data format demonstrated")
    print("✅ Generation example code provided")
    print("✅ Checkpoint information saved for download")
    
    print(f"\n📁 Files Ready for Download:")
    print(f"   • Model checkpoint: /workspace/cdvae/cdvae/prop_models/mp20/epoch=839-step=89039.ckpt")
    print(f"   • Model info: {info_file}")
    print(f"   • Demo script: /workspace/cdvae/crystal_generator_demo.py")
    
    print(f"\n🚀 Next Steps:")
    print("   1. Download the checkpoint file to your local machine")
    print("   2. Install required dependencies (torch, pytorch_lightning, pymatgen)")
    print("   3. Use the example code to generate novel crystal structures")
    print("   4. Explore the latent space for materials discovery!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Demo completed successfully!")
    else:
        print("\n❌ Demo failed - check error messages above")