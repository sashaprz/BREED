"""
Common utilities for CDVAE
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def log_hyperparameters(trainer, model, cfg):
    """Log hyperparameters for tracking"""
    hparams = {}
    
    # Model hyperparameters
    if hasattr(model, 'hparams'):
        hparams.update(model.hparams)
    
    # Config hyperparameters
    if cfg is not None:
        hparams.update(flatten_dict(cfg))
    
    # Log to trainer
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(hparams)


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('loss', 0.0)


def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_output_dir(base_dir: str, experiment_name: str) -> Path:
    """Create output directory for experiment"""
    output_dir = Path(base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to file"""
    import json
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from file"""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)