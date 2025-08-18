#!/usr/bin/env python3
"""
Test script for the enhanced CDVAE setup with caching and improved architecture.
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cdvae.common.caching import get_data_cache
from cdvae.pl_modules.enhanced_model import EnhancedCDVAE
from cdvae.pl_data.datamodule import CrystDataModule

def test_caching_system():
    """Test the caching system."""
    print("Testing caching system...")
    
    try:
        cache = get_data_cache("test_cache")
        
        # Test basic cache operations
        test_key = "test_key"
        test_value = {"data": [1, 2, 3], "metadata": "test"}
        
        # Set and get
        cache.set(test_key, test_value)
        retrieved_value = cache.get(test_key)
        
        assert retrieved_value == test_value, "Cache set/get failed"
        print("✓ Basic cache operations work")
        
        # Test cache statistics
        stats = cache.get_stats()
        print(f"✓ Cache stats: {stats}")
        
        # Clear test cache
        cache.clear()
        print("✓ Caching system test passed")
        return True
        
    except Exception as e:
        print(f"✗ Caching system test failed: {e}")
        return False

def test_enhanced_model():
    """Test the enhanced model architecture."""
    print("\nTesting enhanced model architecture...")
    
    # Create a minimal config for testing
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'latent_dim': 256,
        'hidden_dim': 128,
        'fc_num_layers': 2,
        'max_atoms': 20,
        'cost_natom': 2.0,
        'cost_coord': 10.0,
        'cost_type': 1.0,
        'cost_lattice': 10.0,
        'cost_composition': 1.5,
        'cost_property': 1.0,
        'beta': 0.01,
        'beta_start': 0.0,
        'beta_end': 0.01,
        'beta_warmup_epochs': 10,
        'transformer_layers': 2,
        'attention_heads': 4,
        'predict_property': False,
        'teacher_forcing_lattice': True,
        'teacher_forcing_max_epoch': 5,
        'sigma_begin': 10.0,
        'sigma_end': 0.01,
        'type_sigma_begin': 5.0,
        'type_sigma_end': 0.01,
        'num_noise_level': 50,
        'data': {
            'lattice_scale_method': 'scale_length'
        },
        'encoder': {
            '_target_': 'cdvae.pl_modules.gnn.DimeNetPlusPlusWrap',
            'num_targets': 256,
            'hidden_channels': 128,
            'num_blocks': 4,
            'int_emb_size': 64,
            'basis_emb_size': 8,
            'out_emb_channels': 256,
            'num_spherical': 7,
            'num_radial': 6,
            'otf_graph': False,
            'cutoff': 7.0,
            'max_num_neighbors': 20,
            'envelope_exponent': 5,
            'num_before_skip': 1,
            'num_after_skip': 2,
            'num_output_layers': 3,
            'readout': 'mean',
        },
        'decoder': {
            '_target_': 'cdvae.pl_modules.decoder.GemNetTDecoder',
            'hidden_dim': 128,
            'latent_dim': 256,
            'max_neighbors': 20,
            'radius': 7.0,
        }
    })
    
    try:
        # Test model instantiation
        model = EnhancedCDVAE(**config)
        print("✓ Enhanced model instantiation successful")
        
        # Test model components
        assert hasattr(model, 'atom_count_predictor'), "Missing atom count predictor"
        assert hasattr(model, 'beta_scheduler'), "Missing beta scheduler"
        print("✓ Enhanced model components present")
        
        # Test beta scheduling
        beta_0 = model.beta_scheduler.get_beta(0)
        beta_5 = model.beta_scheduler.get_beta(5)
        beta_15 = model.beta_scheduler.get_beta(15)
        
        assert beta_0 == 0.0, f"Expected beta_0=0.0, got {beta_0}"
        assert 0.0 < beta_5 < 0.01, f"Expected 0 < beta_5 < 0.01, got {beta_5}"
        assert beta_15 == 0.01, f"Expected beta_15=0.01, got {beta_15}"
        print("✓ Beta scheduling works correctly")
        
        # Test atom count predictor
        batch_size = 4
        z = torch.randn(batch_size, config.latent_dim)
        discrete_logits, continuous_pred = model.atom_count_predictor(z)
        
        assert discrete_logits.shape == (batch_size, config.max_atoms + 1), \
            f"Wrong discrete logits shape: {discrete_logits.shape}"
        assert continuous_pred.shape == (batch_size, 1), \
            f"Wrong continuous pred shape: {continuous_pred.shape}"
        print("✓ Enhanced atom count predictor works")
        
        print("✓ Enhanced model architecture test passed")
        
    except Exception as e:
        print(f"✗ Enhanced model test failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test loading the enhanced configurations."""
    print("\nTesting configuration loading...")
    
    try:
        # Test if we can load the enhanced config
        config_path = project_root / "conf"
        
        # Check if config files exist
        enhanced_configs = [
            "enhanced_default.yaml",
            "model/enhanced_vae.yaml",
            "data/enhanced_mp_20.yaml",
            "optim/enhanced_optim.yaml",
            "train/enhanced_train.yaml"
        ]
        
        for config_file in enhanced_configs:
            config_path_full = config_path / config_file
            assert config_path_full.exists(), f"Config file missing: {config_file}"
        
        print("✓ All enhanced configuration files exist")
        print("✓ Configuration loading test passed")
        
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        return False
    
    return True

def test_cuda_availability():
    """Test CUDA availability and setup."""
    print("\nTesting CUDA setup...")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print("✓ CUDA setup is ready")
    else:
        print("⚠ CUDA not available, will use CPU")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("ENHANCED CDVAE SETUP VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("CUDA Setup", test_cuda_availability),
        ("Caching System", test_caching_system),
        ("Enhanced Model", test_enhanced_model),
        ("Configuration Loading", test_config_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<25} {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Enhanced CDVAE setup is ready.")
        print("\nTo train with the enhanced model, run:")
        print("python cdvae/run.py --config-name=enhanced_default")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)