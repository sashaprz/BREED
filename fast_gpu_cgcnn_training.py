#!/usr/bin/env python3
"""
Fast GPU-Optimized CGCNN Training with Pre-computed Crystal Graphs
Uses pre-computed crystal graphs for immediate GPU training
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
    # Data parameters
    data_path: str = "data/ionic_conductivity_dataset_with_graphs.pkl"
    prop_name: str = "ionic_conductivity"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    use_stratified_split: bool = True
    n_bins_stratify: int = 10
    
    # Enhanced model parameters
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.2
    cutoff: float = 8.0
    max_neighbors: int = 20
    
    # Optimized training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 300
    patience: int = 30
    min_delta: float = 1e-5
    
    # Advanced scheduling
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"  # reduce_on_plateau, step, cosine
    scheduler_factor: float = 0.5
    scheduler_patience: int = 15
    scheduler_step_size: int = 50
    
    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 5
    
    # Log-space parameters
    log_epsilon: float = 1e-20
    conductivity_threshold: float = 1e-12
    
    # Output parameters
    output_dir: str = "outputs/fast_gpu_cgcnn"
    save_every: int = 10
    log_every: int = 5
    device: str = "auto"


class LogSpaceScaler:
    """Optimized log-space scaler for GPU"""
    
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
        device = conductivity_values.device
        log_mean = self.log_mean.to(device)
        log_std = self.log_std.to(device)
        
        log_values = torch.log10(conductivity_values + self.epsilon)
        return (log_values - log_mean) / log_std
    
    def inverse_transform(self, normalized_log_values: torch.Tensor) -> torch.Tensor:
        """Transform back from normalized log space to conductivity"""
        device = normalized_log_values.device
        log_mean = self.log_mean.to(device)
        log_std = self.log_std.to(device)
        
        log_values = normalized_log_values * log_std + log_mean
        return torch.pow(10, log_values) - self.epsilon
    
    def to(self, device):
        """Move scaler to device"""
        if self.log_mean is not None:
            self.log_mean = self.log_mean.to(device)
        if self.log_std is not None:
            self.log_std = self.log_std.to(device)
        return self


class EnhancedCGCNNConv(MessagePassing):
    """Enhanced CGCNN convolution with attention"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention
        self.q_linear = nn.Linear(node_dim, hidden_dim)
        self.k_linear = nn.Linear(node_dim, hidden_dim)
        self.v_linear = nn.Linear(node_dim, hidden_dim)
        
        # Edge transformation
        self.edge_linear = nn.Linear(edge_dim, hidden_dim)
        
        # Message function
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()
        )
        
    def forward(self, x, edge_index, edge_attr):
        """Forward pass with attention"""
        # Multi-head attention
        q = self.q_linear(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(-1, self.num_heads, self.head_dim)
        
        # Edge features
        edge_attr_transformed = self.edge_linear(edge_attr)
        
        # Message passing
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_attr=edge_attr_transformed)
        out = out.view(-1, self.hidden_dim)
        
        # Update
        out = self.update_net(torch.cat([out, x], dim=-1))
        
        return out
    
    def message(self, q_i, k_j, v_j, edge_attr):
        """Create attention-weighted messages"""
        # Attention scores
        attention = torch.sum(q_i * k_j, dim=-1, keepdim=True) / np.sqrt(self.head_dim)
        attention = torch.softmax(attention, dim=0)
        
        # Weighted messages
        message = attention * v_j
        message = message.view(-1, self.hidden_dim)
        
        return message * edge_attr


class FastGPUCGCNN(nn.Module):
    """Fast GPU-optimized CGCNN model"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        
        # Atom embedding (up to atomic number 118)
        self.atom_embedding = nn.Embedding(119, self.hidden_dim)
        
        # Enhanced CGCNN layers
        self.conv_layers = nn.ModuleList([
            EnhancedCGCNNConv(
                node_dim=self.hidden_dim,
                edge_dim=1,
                hidden_dim=self.hidden_dim,
                num_heads=config.num_heads
            ) for _ in range(self.num_layers)
        ])
        
        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)
        
        # Enhanced prediction head
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Softplus()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, batch):
        """Forward pass"""
        # Atom embeddings
        x = self.atom_embedding(batch.atom_types)
        
        # Edge features
        edge_attr = batch.distances.unsqueeze(-1)
        
        # CGCNN layers with residual connections
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
        
        # Prediction
        out = self.predictor(x)
        
        return out


class FastDataset(Dataset):
    """Fast dataset using pre-computed crystal graphs"""
    
    def __init__(self, data_path: str, config: TrainingConfig):
        self.config = config
        
        # Load pre-computed data
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} samples with pre-computed graphs")
        
        # Setup scaler
        conductivity_values = torch.tensor([d['ionic_conductivity'] for d in self.data], dtype=torch.float32)
        self.log_scaler = LogSpaceScaler(epsilon=config.log_epsilon)
        self.log_scaler.fit(conductivity_values)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Transform conductivity to log space
        conductivity = torch.tensor([sample['ionic_conductivity']], dtype=torch.float32)
        log_conductivity = self.log_scaler.transform(conductivity)
        
        # Create PyTorch Geometric data object from pre-computed graph
        data = Data(
            atom_types=torch.tensor(sample['atom_types'], dtype=torch.long),
            edge_index=torch.tensor(sample['edge_index'], dtype=torch.long),
            distances=torch.tensor(sample['distances'], dtype=torch.float32),
            y=log_conductivity,
            raw_y=conductivity,
            num_nodes=sample['num_atoms'],
            material_id=sample['material_id']
        )
        
        return data


def stratified_split(dataset, train_ratio=0.8, val_ratio=0.1, n_bins=10, random_state=42):
    """Stratified split in log space with fallback to random split"""
    try:
        # Get conductivity values
        conductivities = [dataset[i].raw_y.item() for i in range(len(dataset))]
        log_conductivities = np.log10(np.array(conductivities) + 1e-20)
        
        # Create bins for stratification - use fewer bins for small datasets
        effective_bins = min(n_bins, len(dataset) // 10)  # At least 10 samples per bin
        effective_bins = max(effective_bins, 3)  # At least 3 bins
        
        bins = np.linspace(log_conductivities.min(), log_conductivities.max(), effective_bins + 1)
        bin_indices = np.digitize(log_conductivities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, effective_bins - 1)
        
        # Check if all bins have at least 2 samples
        unique_bins, counts = np.unique(bin_indices, return_counts=True)
        if np.any(counts < 2):
            print(f"⚠️  Some bins have <2 samples, falling back to random split")
            raise ValueError("Insufficient samples for stratification")
        
        # Split indices
        indices = np.arange(len(dataset))
        train_indices, temp_indices = train_test_split(
            indices, train_size=train_ratio, stratify=bin_indices, random_state=random_state
        )
        
        temp_bin_indices = bin_indices[temp_indices]
        val_size = val_ratio / (1 - train_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices, train_size=val_size, stratify=temp_bin_indices, random_state=random_state
        )
        
        print(f"✅ Stratified split successful with {effective_bins} bins")
        return train_indices, val_indices, test_indices
        
    except Exception as e:
        print(f"⚠️  Stratified split failed ({e}), using random split")
        # Fallback to random split
        indices = np.arange(len(dataset))
        train_indices, temp_indices = train_test_split(
            indices, train_size=train_ratio, random_state=random_state
        )
        
        val_size = val_ratio / (1 - train_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices, train_size=val_size, random_state=random_state
        )
        
        return train_indices, val_indices, test_indices


def compute_enhanced_metrics(predictions: torch.Tensor, targets: torch.Tensor, log_scaler: LogSpaceScaler) -> Dict[str, float]:
    """Enhanced metrics without problematic MAPE"""
    # Convert back to original space
    pred_conductivity = log_scaler.inverse_transform(predictions)
    true_conductivity = log_scaler.inverse_transform(targets)
    
    # Ensure positive values
    pred_conductivity = torch.clamp(pred_conductivity, min=1e-20)
    true_conductivity = torch.clamp(true_conductivity, min=1e-20)
    
    # Log-space metrics
    mae_log = F.l1_loss(predictions, targets).item()
    mse_log = F.mse_loss(predictions, targets).item()
    rmse_log = np.sqrt(mse_log)
    
    # Raw space metrics
    mae_raw_log = F.l1_loss(torch.log10(pred_conductivity + 1e-20), 
                           torch.log10(true_conductivity + 1e-20)).item()
    
    # Orders of magnitude error
    log_error = torch.abs(torch.log10(pred_conductivity + 1e-20) - torch.log10(true_conductivity + 1e-20))
    mean_log_error = torch.mean(log_error).item()
    median_log_error = torch.median(log_error).item()
    
    # R² score in log space
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2_log = 1 - (ss_res / ss_tot).item()
    
    return {
        'mae_log': mae_log,
        'mae_raw_log': mae_raw_log,
        'mse_log': mse_log,
        'rmse_log': rmse_log,
        'mean_log_error': mean_log_error,  # Orders of magnitude
        'median_log_error': median_log_error,
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_enhanced_metrics(all_predictions, all_targets, log_scaler)
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
    metrics = compute_enhanced_metrics(all_predictions, all_targets, log_scaler)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def setup_logging(output_dir: Path):
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('fast_gpu_cgcnn')


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Fast GPU CGCNN Training')
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
    
    # Force GPU usage
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  GPU not available, using CPU")
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting Fast GPU CGCNN training for ionic conductivity prediction")
    logger.info(f"Configuration: {config}")
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Load dataset
    logger.info("Loading dataset with pre-computed crystal graphs...")
    full_dataset = FastDataset(config.data_path, config)
    
    # Stratified split
    if config.use_stratified_split:
        logger.info("Performing stratified split in log space...")
        train_indices, val_indices, test_indices = stratified_split(
            full_dataset, config.train_ratio, config.val_ratio, config.n_bins_stratify
        )
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    else:
        # Random split
        total_size = len(full_dataset)
        train_size = int(config.train_ratio * total_size)
        val_size = int(config.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    logger.info(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                             collate_fn=Batch.from_data_list, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                           collate_fn=Batch.from_data_list, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                            collate_fn=Batch.from_data_list, num_workers=4, pin_memory=True)
    
    # Create model
    logger.info("Creating enhanced CGCNN model...")
    model = FastGPUCGCNN(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} trainable parameters")
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Setup scheduler
    if config.use_scheduler:
        if config.scheduler_type == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.scheduler_factor, 
                                        patience=config.scheduler_patience, verbose=True)
        elif config.scheduler_type == "step":
            scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_factor)
        elif config.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        else:
            scheduler = None
    else:
        scheduler = None
    
    # Move scaler to GPU
    full_dataset.log_scaler = full_dataset.log_scaler.to(device)
    
    # Training loop
    logger.info("🚀 Starting GPU training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, full_dataset.log_scaler)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, full_dataset.log_scaler)
        
        # Update scheduler
        if scheduler is not None:
            if config.scheduler_type == "reduce_on_plateau":
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Log metrics
        if epoch % config.log_every == 0:
            logger.info(f"Epoch {epoch:3d} | "
                       f"Train Loss: {train_metrics['loss']:.4f} | "
                       f"Val Loss: {val_metrics['loss']:.4f} | "
                       f"Val R²: {val_metrics['r2_log']:.4f} | "
                       f"Mean Log Error: {val_metrics['mean_log_error']:.3f} orders")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss - config.min_delta:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config.__dict__,
                'log_scaler': full_dataset.log_scaler
            }, output_dir / 'best_model.pt')
            
            logger.info(f"💾 New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Store losses
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        
        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"🛑 Early stopping at epoch {epoch} (patience: {config.patience})")
            break
    
    # Test evaluation
    logger.info("📊 Evaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate_epoch(model, test_loader, device, full_dataset.log_scaler)
    
    logger.info("🎯 Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
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
    
    logger.info(f"✅ Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()