#!/usr/bin/env python3
"""
Test script to diagnose CDVAE loading issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE')

try:
    from generator.CDVAE.load_trained_model import TrainedCDVAELoader
    print('✅ TrainedCDVAELoader imported successfully')
    
    # Initialize loader
    weights_path = r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\new_cdvae_weights.ckpt'
    scalers_dir = r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE'
    
    loader = TrainedCDVAELoader(weights_path, scalers_dir)
    print('✅ Loader initialized')
    
    # Try to load model
    model = loader.load_model()
    print(f'✅ Model loaded: {type(model).__name__}')
    print(f'Model device: {next(model.parameters()).device}')
    print(f'Model has sample method: {hasattr(model, "sample")}')
    print(f'Model hparams latent_dim: {model.hparams.latent_dim}')
    print(f'Model hparams hidden_dim: {model.hparams.hidden_dim}')
    
    # Try to generate structures
    from types import SimpleNamespace
    ld_kwargs = SimpleNamespace(
        n_step_each=10,  # Reduced for testing
        step_lr=1e-4,
        min_sigma=0.0,
        save_traj=False,
        disable_bar=True
    )
    
    print('🧬 Attempting to generate 2 structures...')
    structures = loader.generate_structures(2)
    print(f'✅ Generated {len(structures)} structures successfully!')
    print(f'Structure type: {type(structures[0]) if structures else "No structures"}')
    
    if structures:
        print(f'First structure keys: {list(structures[0].keys()) if hasattr(structures[0], "keys") else "Not a dict"}')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()