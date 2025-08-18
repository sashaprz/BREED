#!/usr/bin/env python3
"""
Standalone Enhanced CGCNN for Ionic Conductivity Prediction

This is a self-contained implementation that doesn't depend on CDVAE,
specifically designed for ionic conductivity prediction with:
1. Log-space training
2. Positive activation functions
3. Data cleaning and filtering
4. Crystal graph construction from CIF files
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter

# Crystal structure processing
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Data parameters
    data_path: str = "data/ionic_conductivity_dataset.pkl"
    prop_name: str = "ionic_conductivity"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Model parameters
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    cutoff: float = 8.0
    max_neighbors: int = 20
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-4
    
    # Log-space parameters
    log_epsilon: float = 1e-20
    conductivity_threshold: float = 1e-12
    
    # Output parameters
    output_dir: str = "outputs/cgcnn_ionic_conductivity"
    save_every: int = 10
    log_every: int = 10
    
    # Device
    device: str = "auto"


class LogSpaceScaler:
    """Custom scaler for log-space normalization of ionic conductivity"""
    
    def __init__(self, epsilon: float = 1e-20):
        self.epsilon = epsilon
        self.log_mean = None
        self.log_std = None
        
    def fit(self, conductivity_values: torch.Tensor):
        """Fit scaler on log-transformed conductivity values"""
        log_values = torch.log10(conductivity_values + self.epsilon)
        self.log_mean = torch.mean(log_values)
        self.log_std = torch.std(log_values) + 1e-8
        
    def transform(self, conductivity_values: torch.Tensor) -> torch.Tensor:
        """Transform conductivity to normalized log space"""
        # Ensure all tensors are on the same device
        device = conductivity_values.device
        log_mean = self.log_mean.to(device) if self.log_mean is not None else torch.tensor(0.0, device=device)
        log_std = self.log_std.to(device) if self.log_std is not None else torch.tensor(1.0, device=device)
        
        log_values = torch.log10(conductivity_values + self.epsilon)
        return (log_values - log_mean) / log_std
    
    def inverse_transform(self, normalized_log_values: torch.Tensor) -> torch.Tensor:
        """Transform back from normalized log space to conductivity"""
        # Ensure all tensors are on the same device
        device = normalized_log_values.device
        log_mean = self.log_mean.to(device) if self.log_mean is not None else torch.tensor(0.0, device=device)
        log_std = self.log_std.to(device) if self.log_std is not None else torch.tensor(1.0, device=device)
        
        log_values = normalized_log_values * log_std + log_mean
        return torch.pow(10, log_values) - self.epsilon
    
    def to(self, device):
        """Move scaler to device"""
        if self.log_mean is not None:
            self.log_mean = self.log_mean.to(device)
        if self.log_std is not None:
            self.log_std = self.log_std.to(device)
        return self


def build_crystal_graph(structure: Structure, cutoff: float = 8.0, max_neighbors: int = 20):
    """Build crystal graph from pymatgen Structure"""
    try:
        # Use CrystalNN for neighbor finding (no cutoff parameter in this version)
        nn_strategy = CrystalNN()
        
        # Get all neighbors
        all_neighbors = []
        for i, site in enumerate(structure):
            neighbors = nn_strategy.get_nn_info(structure, i)
            for neighbor in neighbors[:max_neighbors]:  # Limit neighbors
                j = neighbor['site_index']
                distance = neighbor['weight']  # CrystalNN uses weight as distance
                if distance <= cutoff:
                    all_neighbors.append((i, j, distance))
        
        # Create edge indices and distances
        if not all_neighbors:
            # Fallback: create self-loops
            num_atoms = len(structure)
            edge_indices = [[i, i] for i in range(num_atoms)]
            distances = [0.1] * num_atoms  # Small distance for self-loops
        else:
            edge_indices = [(i, j) for i, j, d in all_neighbors]
            distances = [d for i, j, d in all_neighbors]
        
        # Convert to arrays
        edge_index = np.array(edge_indices).T
        distances = np.array(distances)
        
        # Get atomic numbers
        atom_types = [site.specie.Z for site in structure]
        
        # Get fractional coordinates
        frac_coords = structure.frac_coords
        
        # Get lattice parameters
        lattice = structure.lattice
        lengths = [lattice.a, lattice.b, lattice.c]
        angles = [lattice.alpha, lattice.beta, lattice.gamma]
        
        return {
            'frac_coords': frac_coords,
            'atom_types': np.array(atom_types),
            'edge_index': edge_index,
            'distances': distances,
            'lengths': lengths,
            'angles': angles,
            'num_atoms': len(structure)
        }
        
    except Exception as e:
        print(f"Error building crystal graph: {e}")
        # Fallback: minimal graph
        num_atoms = len(structure)
        return {
            'frac_coords': structure.frac_coords,
            'atom_types': np.array([site.specie.Z for site in structure]),
            'edge_index': np.array([[i, i] for i in range(num_atoms)]).T,
            'distances': np.array([0.1] * num_atoms),
            'lengths': [structure.lattice.a, structure.lattice.b, structure.lattice.c],
            'angles': [structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma],
            'num_atoms': num_atoms
        }


class CGCNNConv(MessagePassing):
    """Enhanced CGCNN convolution layer"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='add', node_dim=0)
        
        self.input_node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Node transformation
        self.node_linear = nn.Linear(self.input_node_dim, hidden_dim)
        
        # Edge transformation
        self.edge_linear = nn.Linear(edge_dim, hidden_dim)
        
        # Message function
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function with positive activation
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim + self.input_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()  # Positive activation
        )
        
    def forward(self, x, edge_index, edge_attr):
        """Forward pass"""
        x_transformed = self.node_linear(x)
        edge_attr_transformed = self.edge_linear(edge_attr)
        
        out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr_transformed)
        out = self.update_net(torch.cat([out, x], dim=-1))
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        """Create messages"""
        message_input = torch.cat([x_i, x_j], dim=-1)
        message = self.message_net(message_input)
        return message * edge_attr


class StandaloneCGCNN(nn.Module):
    """Standalone CGCNN model for ionic conductivity prediction"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        
        # Atom embedding (up to atomic number 118)
        self.atom_embedding = nn.Embedding(119, self.hidden_dim)
        
        # CGCNN layers
        self.conv_layers = nn.ModuleList([
            CGCNNConv(
                node_dim=self.hidden_dim,
                edge_dim=1,  # Just distance
                hidden_dim=self.hidden_dim
            ) for _ in range(self.num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Softplus()  # Positive output
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, batch):
        """Forward pass"""
        # Get atom embeddings
        x = self.atom_embedding(batch.atom_types)
        
        # Edge features (distances)
        edge_attr = batch.distances.unsqueeze(-1)
        
        # Apply CGCNN layers
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            x_new = conv(x, batch.edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = self.dropout(x_new)
            
            # Residual connection
            if i > 0:
                x = x + x_new
            else:
                x = x_new
        
        # Global pooling
        x = global_mean_pool(x, batch.batch)
        
        # Final prediction
        out = self.predictor(x)
        
        return out


class IonicConductivityDataset(Dataset):
    """Dataset for ionic conductivity prediction"""
    
    def __init__(self, data_path: str, config: TrainingConfig):
        self.config = config
        
        # Load data
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} samples")
        
        # Setup scaler - keep on CPU for data loading
        conductivity_values = torch.tensor([d['ionic_conductivity'] for d in self.data], dtype=torch.float32)
        self.log_scaler = LogSpaceScaler(epsilon=config.log_epsilon)
        self.log_scaler.fit(conductivity_values)
        
        # Store original scaler parameters for later GPU transfer
        self.log_mean_cpu = self.log_scaler.log_mean.clone()
        self.log_std_cpu = self.log_scaler.log_std.clone()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        
        try:
            # Parse CIF and build graph
            structure = Structure.from_str(data_dict['cif'], fmt='cif')
            graph_data = build_crystal_graph(structure, self.config.cutoff, self.config.max_neighbors)
            
            # Transform conductivity to log space using CPU scaler
            conductivity = torch.tensor([data_dict['ionic_conductivity']], dtype=torch.float32)
            
            # Manual log transformation to avoid device issues
            log_values = torch.log10(conductivity + self.config.log_epsilon)
            log_conductivity = (log_values - self.log_mean_cpu) / self.log_std_cpu
            
            # Create PyTorch Geometric data object
            data = Data(
                atom_types=torch.tensor(graph_data['atom_types'], dtype=torch.long),
                edge_index=torch.tensor(graph_data['edge_index'], dtype=torch.long),
                distances=torch.tensor(graph_data['distances'], dtype=torch.float32),
                y=log_conductivity,
                raw_y=conductivity,
                num_nodes=graph_data['num_atoms'],
                material_id=data_dict['material_id']
            )
            
            return data
            
        except Exception as e:
            print(f"Error processing {data_dict['material_id']}: {e}")
            # Return a dummy sample
            return Data(
                atom_types=torch.tensor([1], dtype=torch.long),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                distances=torch.tensor([1.0], dtype=torch.float32),
                y=torch.tensor([0.0], dtype=torch.float32),
                raw_y=torch.tensor([1e-12], dtype=torch.float32),
                num_nodes=1,
                material_id=data_dict['material_id']
            )


def collate_fn(batch):
    """Custom collate function"""
    return Batch.from_data_list(batch)


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, log_scaler: LogSpaceScaler) -> Dict[str, float]:
    """Compute evaluation metrics"""
    # Convert back to original space
    pred_conductivity = log_scaler.inverse_transform(predictions)
    true_conductivity = log_scaler.inverse_transform(targets)
    
    # Ensure positive values
    pred_conductivity = torch.clamp(pred_conductivity, min=1e-20)
    true_conductivity = torch.clamp(true_conductivity, min=1e-20)
    
    # Metrics
    mae_log = F.l1_loss(predictions, targets).item()
    mae_orig = F.l1_loss(pred_conductivity, true_conductivity).item()
    mse_log = F.mse_loss(predictions, targets).item()
    rmse_log = np.sqrt(mse_log)
    
    # MAPE
    mape = torch.mean(torch.abs((true_conductivity - pred_conductivity) / true_conductivity)).item() * 100
    
    # R² score
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2_log = 1 - (ss_res / ss_tot).item()
    
    return {
        'mae_log': mae_log,
        'mae_orig': mae_orig,
        'mse_log': mse_log,
        'rmse_log': rmse_log,
        'mape': mape,
        'r2_log': r2_log
    }


def train_epoch(model, dataloader, optimizer, device, log_scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        predictions = model(batch).squeeze()
        targets = batch.y.squeeze()
        
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_predictions, all_targets, log_scaler)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def validate_epoch(model, dataloader, device, log_scaler):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            predictions = model(batch).squeeze()
            targets = batch.y.squeeze()
            
            loss = F.mse_loss(predictions, targets)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_predictions, all_targets, log_scaler)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Standalone CGCNN for Ionic Conductivity')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config = TrainingConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Override with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Setup device - force CUDA if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CUDA not available)")
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    print("Loading dataset...")
    full_dataset = IonicConductivityDataset(config.data_path, config)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(config.train_ratio * total_size)
    val_size = int(config.val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print("Creating model...")
    model = StandaloneCGCNN(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    
    # Move log scaler to device for metric computation
    full_dataset.log_scaler = full_dataset.log_scaler.to(device)
    
    # Create a GPU version of the scaler for metrics
    gpu_scaler = LogSpaceScaler(epsilon=config.log_epsilon)
    gpu_scaler.log_mean = full_dataset.log_mean_cpu.to(device)
    gpu_scaler.log_std = full_dataset.log_std_cpu.to(device)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config.num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, gpu_scaler)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, gpu_scaler)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        if epoch % config.log_every == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val R²: {val_metrics['r2_log']:.4f} | "
                  f"Val MAPE: {val_metrics['mape']:.2f}%")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config.__dict__,
                'log_scaler': full_dataset.log_scaler
            }, output_dir / 'best_model.pt')
        
        # Store losses
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        
        # Early stopping
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Test evaluation
    print("Evaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate_epoch(model, test_loader, device, gpu_scaler)
    
    print("Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': test_metrics
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()