#!/usr/bin/env python3
"""
Test script to verify that the enhanced model configuration works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path

# Add the CDVAE directory to Python path
cdvae_dir = Path(__file__).parent
sys.path.insert(0, str(cdvae_dir))

def test_enhanced_model_instantiation():
    """Test that the enhanced model can be instantiated from config."""
    
    print("Testing Enhanced CDVAE model instantiation...")
    
    try:
        # Import the enhanced model
        from cdvae.pl_modules.enhanced_model import EnhancedCDVAE
        print("✓ Successfully imported EnhancedCDVAE")
        
        # Create a minimal config for testing
        test_config = {
            '_target_': 'cdvae.pl_modules.enhanced_model.EnhancedCDVAE',
            'hidden_dim': 256,
            'latent_dim': 256,
            'fc_num_layers': 1,
            'max_atoms': 20,
            'cost_natom': 1.0,
            'cost_coord': 10.0,
            'cost_type': 1.0,
            'cost_lattice': 10.0,
            'cost_composition': 1.0,
            'cost_edge': 10.0,
            'cost_property': 1.0,
            'beta': 0.01,
            'teacher_forcing_lattice': True,
            'teacher_forcing_max_epoch': 100,
            'max_neighbors': 20,
            'radius': 7.0,
            'sigma_begin': 10.0,
            'sigma_end': 0.01,
            'type_sigma_begin': 5.0,
            'type_sigma_end': 0.01,
            'num_noise_level': 50,
            'predict_property': False,
            'transformer_layers': 2,
            'attention_heads': 4,
            'cost_natom_enhanced': 2.0,
            'beta_start': 0.0,
            'beta_end': 0.01,
            'beta_warmup_epochs': 10,
            'beta_schedule': 'linear',
            'kld_capacity': 0.0,
            'encoder': {
                '_target_': 'cdvae.pl_modules.gnn.DimeNetPlusPlus',
                'num_targets': 256,
                'hidden_channels': 256,
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
                'num_output_layers': 3,
            },
            'decoder': {
                '_target_': 'cdvae.pl_modules.decoder.GemNetTDecoder',
                'hidden_dim': 256,
                'latent_dim': 256,
                'max_neighbors': 20,
                'radius': 7.0,
                'scale_file': None,
            }
        }
        
        # Test direct instantiation
        model = EnhancedCDVAE(**test_config)
        print("✓ Successfully instantiated EnhancedCDVAE directly")
        
        # Test that key components exist
        assert hasattr(model, 'atom_count_predictor'), "Missing atom_count_predictor"
        assert hasattr(model, 'beta_scheduler'), "Missing beta_scheduler"
        assert hasattr(model, 'enhanced_num_atom_loss'), "Missing enhanced_num_atom_loss method"
        print("✓ All enhanced components are present")
        
        # Test that the model has the expected architecture optimizations
        assert model.atom_count_predictor.extended_max_atoms >= 100, "Extended max atoms not set correctly"
        assert len(model.atom_count_predictor.transformer_layers) == 2, "Transformer layers not optimized"
        print("✓ Architecture optimizations applied correctly")
        
        print("\n🎉 All tests passed! Enhanced model configuration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_model_instantiation()
    sys.exit(0 if success else 1)