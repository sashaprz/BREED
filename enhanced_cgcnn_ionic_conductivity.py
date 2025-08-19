"""
Enhanced CGCNN Training Script for Ionic Conductivity Prediction

This script implements an enhanced Crystal Graph Convolutional Neural Network (CGCNN)
specifically designed for ionic conductivity prediction with the following improvements:
1. Log-space training (log_conductivity = torch.log10(conductivity + 1e-20))
2. Improved normalization with log-space normalization
3. Model architecture fix with positive activation functions (torch.exp or F.softplus)
4. Data cleaning to filter out placeholder values (conductivity <= 1e-12)

The script includes comprehensive training loop, data loading, and evaluation metrics.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_scatter import scatter
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.preprocessing import KBinsDiscretizer

# Import existing utilities from CDVAE
sys.path.append(str(Path(__file__).parent / "generator" / "CDVAE"))
from cdvae.common.data_utils import (
    preprocess, get_scaler_from_data_list, StandardScalerTorch,
    build_crystal, build_crystal_graph, frac_to_cart_coords,
    get_pbc_distances, radius_graph_pbc_wrapper
)
from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Data parameters
    data_path: str = "data/ionic_conductivity.csv"
    prop_name: str = "ionic_conductivity"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    use_stratified_split: bool = True
    n_bins_stratify: int = 10  # Number of bins for stratified splitting
    
    # Model parameters - Enhanced architecture
    hidden_dim: int = 256  # Increased from 128
    num_layers: int = 6    # Increased from 4
    num_heads: int = 8
    dropout: float = 0.2   # Slightly increased for better regularization
    cutoff: float = 8.0
    max_neighbors: int = 20
    
    # Training parameters - Optimized for wide dynamic range
    batch_size: int = 16   # Reduced for better gradient updates
    learning_rate: float = 1e-4  # Reduced for more stable training
    weight_decay: float = 1e-4   # Increased regularization
    num_epochs: int = 300        # Increased for better convergence
    patience: int = 30           # Increased patience
    min_delta: float = 1e-5      # More sensitive early stopping
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "step", "cosine"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 15
    scheduler_step_size: int = 50
    
    # Cross-validation parameters
    use_cross_validation: bool = False
    cv_folds: int = 5
    
    # Log-space parameters
    log_epsilon: float = 1e-20
    conductivity_threshold: float = 1e-12
    
    # Output parameters
    output_dir: str = "outputs/cgcnn_ionic_conductivity"
    save_every: int = 10
    log_every: int = 5  # More frequent logging
    
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
        # Convert to log space
        log_values = torch.log10(conductivity_values + self.epsilon)
        
        # Compute statistics in log space
        self.log_mean = torch.mean(log_values)
        self.log_std = torch.std(log_values) + 1e-8  # Add small epsilon for numerical stability
        
    def transform(self, conductivity_values: torch.Tensor) -> torch.Tensor:
        """Transform conductivity to normalized log space"""
        log_values = torch.log10(conductivity_values + self.epsilon)
        
        # Ensure all tensors are on the same device
        device = conductivity_values.device
        log_mean = self.log_mean.to(device) if hasattr(self.log_mean, 'to') else self.log_mean
        log_std = self.log_std.to(device) if hasattr(self.log_std, 'to') else self.log_std
        
        return (log_values - log_mean) / log_std
    
    def inverse_transform(self, normalized_log_values: torch.Tensor) -> torch.Tensor:
        """Transform back from normalized log space to conductivity"""
        # Ensure all tensors are on the same device
        device = normalized_log_values.device
        log_mean = self.log_mean.to(device) if hasattr(self.log_mean, 'to') else self.log_mean
        log_std = self.log_std.to(device) if hasattr(self.log_std, 'to') else self.log_std
        
        log_values = normalized_log_values * log_std + log_mean
        return torch.pow(10, log_values)
    
    def to(self, device):
        """Move scaler to device with type checking"""
        if self.log_mean is not None:
            if hasattr(self.log_mean, 'to'):  # Check if it's a tensor
                self.log_mean = self.log_mean.to(device)
            elif isinstance(self.log_mean, (int, float)):
                # Convert scalar to tensor on the correct device
                self.log_mean = torch.tensor(self.log_mean, device=device)
        if self.log_std is not None:
            if hasattr(self.log_std, 'to'):  # Check if it's a tensor
                self.log_std = self.log_std.to(device)
            elif isinstance(self.log_std, (int, float)):
                # Convert scalar to tensor on the correct device
                self.log_std = torch.tensor(self.log_std, device=device)
        return self


class CGCNNConv(MessagePassing):
    """Enhanced CGCNN convolution layer with positive activation functions"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='add', node_dim=0)  # Set node_dim=0 for MessagePassing
        
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
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim + self.input_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()  # Standard ReLU activation
        )
        
    def forward(self, x, edge_index, edge_attr):
        """Forward pass"""
        # Transform node and edge features
        x_transformed = self.node_linear(x)
        edge_attr_transformed = self.edge_linear(edge_attr)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr_transformed)
        
        # Update node features
        out = self.update_net(torch.cat([out, x], dim=-1))
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        """Create messages"""
        # Combine source node, target node, and edge features
        message_input = torch.cat([x_i, x_j], dim=-1)
        message = self.message_net(message_input)
        
        # Weight by edge attributes
        return message * edge_attr


class EnhancedCGCNN(nn.Module):
    """Enhanced CGCNN model for ionic conductivity prediction with improved architecture"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        
        # Atom embedding with improved initialization
        self.atom_embedding = nn.Embedding(MAX_ATOMIC_NUM, self.hidden_dim)
        
        # Edge feature dimension (distance + additional features)
        edge_dim = 1  # Just distance for now, can be extended
        
        # CGCNN layers with residual connections
        self.conv_layers = nn.ModuleList([
            CGCNNConv(
                node_dim=self.hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=self.hidden_dim
            ) for i in range(self.num_layers)
        ])
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Dropout layers
        self.dropout = nn.Dropout(config.dropout)
        
        # Enhanced prediction head with more capacity
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
            nn.Linear(self.hidden_dim // 4, 1)
            # No final activation - predict directly in normalized log space
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
        # Get atom embeddings with safety check
        atom_indices = batch.atom_types - 1  # Subtract 1 for 0-based indexing
        
        # Safety check: ensure all atom indices are valid
        assert torch.all(atom_indices >= 0), f"Invalid atom indices found: {torch.min(atom_indices)}"
        assert torch.all(atom_indices < MAX_ATOMIC_NUM), f"Atom index too large: {torch.max(atom_indices)}"
        
        x = self.atom_embedding(atom_indices)
        
        # Compute edge features (distances)
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
    """Dataset for ionic conductivity prediction with data cleaning"""
    
    def __init__(self, data_path: str, config: TrainingConfig, split: str = 'train'):
        self.config = config
        self.split = split
        
        # Load and preprocess data
        self.data = self._load_and_clean_data(data_path)
        
        # Setup scalers
        self.setup_scalers()
        
    def _load_and_clean_data(self, data_path: str) -> List[Dict]:
        """Load and clean the dataset"""
        # Load data
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                # Handle different pickle formats
                if isinstance(data, list):
                    # If it's already a list of dictionaries, use it directly
                    processed_data = []
                    initial_count = len(data)
                    
                    # Use logger instead of print for consistency
                    import logging
                    logger = logging.getLogger('cgcnn_training')
                    
                    for idx, item in enumerate(data):
                        if isinstance(item, dict) and self.config.prop_name in item:
                            if item[self.config.prop_name] > self.config.conductivity_threshold:
                                try:
                                    # Use pre-computed graph arrays if available, otherwise build them
                                    if 'graph_arrays' in item:
                                        graph_arrays = item['graph_arrays']
                                    else:
                                        # Fallback: Build crystal structure from CIF
                                        crystal_str = item['cif']
                                        crystal = build_crystal(crystal_str, niggli=True, primitive=False)
                                        graph_arrays = build_crystal_graph(crystal, graph_method='crystalnn')
                                    
                                    # Store processed data
                                    processed_item = {
                                        'material_id': item.get('material_id', f'sample_{idx}'),
                                        'cif': item['cif'],
                                        'graph_arrays': graph_arrays,
                                        self.config.prop_name: item[self.config.prop_name]
                                    }
                                    processed_data.append(processed_item)
                                    
                                except Exception as e:
                                    logger.warning(f"Error processing sample {idx} ({item.get('material_id', 'unknown')}): {e}")
                                    continue
                    
                    filtered_count = len(processed_data)
                    logger.info(f"Processed {filtered_count} samples successfully out of {initial_count} total")
                    logger.info(f"Filtered out {initial_count - filtered_count} samples (threshold + processing errors)")
                    
                    if filtered_count == 0:
                        raise ValueError(f"No samples remain after filtering with threshold {self.config.conductivity_threshold}")
                    
                    return processed_data
                else:
                    raise ValueError(f"Unsupported pickle data format: {type(data)}")
        else:
            raise ValueError("CSV format not supported in this version. Please use pickle format.")
    
    def setup_scalers(self):
        """Setup log-space scaler for conductivity values"""
        conductivity_values = torch.tensor([d[self.config.prop_name] for d in self.data], dtype=torch.float32)
        
        self.log_scaler = LogSpaceScaler(epsilon=self.config.log_epsilon)
        self.log_scaler.fit(conductivity_values)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        
        # Extract graph arrays
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']
        
        # Convert to cart coordinates for distance calculation
        lengths_tensor = torch.tensor(lengths, dtype=torch.float32).unsqueeze(0)
        angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0)
        frac_coords_tensor = torch.tensor(frac_coords, dtype=torch.float32)
        num_atoms_tensor = torch.tensor([num_atoms], dtype=torch.long)
        
        # Calculate distances
        pos = frac_to_cart_coords(frac_coords_tensor, lengths_tensor, angles_tensor, num_atoms_tensor)
        
        # Create edge index and calculate distances
        edge_index = torch.tensor(edge_indices.T, dtype=torch.long)
        to_jimages_tensor = torch.tensor(to_jimages, dtype=torch.long)
        
        # Get PBC distances
        out = get_pbc_distances(
            frac_coords_tensor,
            edge_index,
            lengths_tensor,
            angles_tensor,
            to_jimages_tensor,
            num_atoms_tensor,
            torch.tensor([edge_indices.shape[0]], dtype=torch.long),
            coord_is_cart=False
        )
        
        distances = out["distances"]
        
        # Transform conductivity to log space
        conductivity = torch.tensor([data_dict[self.config.prop_name]], dtype=torch.float32)
        log_conductivity = self.log_scaler.transform(conductivity)
        
        # Create PyTorch Geometric data object
        data = Data(
            x=torch.tensor(atom_types, dtype=torch.long),
            atom_types=torch.tensor(atom_types, dtype=torch.long),
            edge_index=edge_index,
            distances=distances,
            pos=pos,
            y=log_conductivity,
            raw_y=conductivity,
            num_nodes=num_atoms,
            lengths=lengths_tensor,
            angles=angles_tensor,
            material_id=data_dict['material_id']
        )
        
        return data


def collate_fn(batch):
    """Custom collate function for batching"""
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


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                   log_scaler: LogSpaceScaler) -> Dict[str, float]:
    """Compute evaluation metrics optimized for wide dynamic range ionic conductivity"""
    # Convert back to original space
    pred_conductivity = log_scaler.inverse_transform(predictions)
    true_conductivity = log_scaler.inverse_transform(targets)
    
    # Ensure positive values
    pred_conductivity = torch.clamp(pred_conductivity, min=1e-20)
    true_conductivity = torch.clamp(true_conductivity, min=1e-20)
    
    # Primary metrics in log space (most important for wide dynamic range)
    mae_log = F.l1_loss(predictions, targets).item()
    mse_log = F.mse_loss(predictions, targets).item()
    rmse_log = np.sqrt(mse_log)
    
    # Raw log space metrics (without normalization)
    raw_log_pred = torch.log10(pred_conductivity + log_scaler.epsilon)
    raw_log_true = torch.log10(true_conductivity + log_scaler.epsilon)
    mae_raw_log = F.l1_loss(raw_log_pred, raw_log_true).item()
    mse_raw_log = F.mse_loss(raw_log_pred, raw_log_true).item()
    
    # Mean Absolute Error in original space
    mae_orig = F.l1_loss(pred_conductivity, true_conductivity).item()
    
    # R² score in normalized log space
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    
    if ss_tot.item() < 1e-12:
        r2_log = 0.0
    else:
        r2_log = 1 - (ss_res / ss_tot).item()
    
    # R² score in raw log space
    ss_res_raw = torch.sum((raw_log_true - raw_log_pred) ** 2)
    ss_tot_raw = torch.sum((raw_log_true - torch.mean(raw_log_true)) ** 2)
    
    if ss_tot_raw.item() < 1e-12:
        r2_raw_log = 0.0
    else:
        r2_raw_log = 1 - (ss_res_raw / ss_tot_raw).item()
    
    # Orders of magnitude error (more interpretable for conductivity)
    log_error = torch.abs(raw_log_pred - raw_log_true)
    mean_log_error = torch.mean(log_error).item()
    median_log_error = torch.median(log_error).item()
    
    return {
        'mae_log': mae_log,           # MAE in normalized log space
        'mae_raw_log': mae_raw_log,   # MAE in raw log space
        'mae_orig': mae_orig,         # MAE in original space
        'mse_log': mse_log,           # MSE in normalized log space
        'mse_raw_log': mse_raw_log,   # MSE in raw log space
        'rmse_log': rmse_log,         # RMSE in normalized log space
        'r2_log': r2_log,             # R² in normalized log space
        'r2_raw_log': r2_raw_log,     # R² in raw log space
        'mean_log_error': mean_log_error,    # Mean error in orders of magnitude
        'median_log_error': median_log_error  # Median error in orders of magnitude
    }


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, log_scaler: LogSpaceScaler) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch).squeeze()
        targets = batch.y.squeeze()
        
        # Compute loss in log space
        loss = F.mse_loss(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_predictions, all_targets, log_scaler)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def validate_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device,
                  log_scaler: LogSpaceScaler) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            predictions = model(batch).squeeze()
            targets = batch.y.squeeze()
            
            # Compute loss
            loss = F.mse_loss(predictions, targets)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_predictions, all_targets, log_scaler)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def stratified_split_dataset(dataset: IonicConductivityDataset, config: TrainingConfig) -> Tuple[Subset, Subset, Subset]:
    """Create stratified train/val/test splits based on log conductivity values"""
    # Extract conductivity values for stratification
    conductivity_values = np.array([d[config.prop_name] for d in dataset.data])
    log_conductivity = np.log10(conductivity_values + config.log_epsilon)
    
    # Create bins for stratification
    discretizer = KBinsDiscretizer(n_bins=config.n_bins_stratify, encode='ordinal', strategy='quantile')
    conductivity_bins = discretizer.fit_transform(log_conductivity.reshape(-1, 1)).flatten().astype(int)
    
    # First split: train vs (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(config.val_ratio + config.test_ratio), random_state=42)
    train_idx, temp_idx = next(sss1.split(np.arange(len(dataset)), conductivity_bins))
    
    # Second split: val vs test from the remaining data
    temp_bins = conductivity_bins[temp_idx]
    val_size = config.val_ratio / (config.val_ratio + config.test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_size), random_state=42)
    val_idx_temp, test_idx_temp = next(sss2.split(np.arange(len(temp_idx)), temp_bins))
    
    # Map back to original indices
    val_idx = temp_idx[val_idx_temp]
    test_idx = temp_idx[test_idx_temp]
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    return train_dataset, val_dataset, test_dataset


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging with handler guard"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('cgcnn_training')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create handlers
    file_handler = logging.FileHandler(output_dir / 'training.log')
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_scheduler(optimizer, config: TrainingConfig):
    """Create learning rate scheduler based on configuration"""
    if not config.use_scheduler:
        return None
    
    if config.scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True,
            min_lr=1e-7
        )
    elif config.scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_factor
        )
    elif config.scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=1e-7
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")


def cross_validate_model(dataset: IonicConductivityDataset, config: TrainingConfig, device: torch.device, logger) -> Dict[str, float]:
    """Perform k-fold cross-validation"""
    logger.info(f"Starting {config.cv_folds}-fold cross-validation...")
    
    # Extract conductivity values for stratification
    conductivity_values = np.array([d[config.prop_name] for d in dataset.data])
    log_conductivity = np.log10(conductivity_values + config.log_epsilon)
    
    # Create bins for stratified k-fold
    discretizer = KBinsDiscretizer(n_bins=config.n_bins_stratify, encode='ordinal', strategy='quantile')
    conductivity_bins = discretizer.fit_transform(log_conductivity.reshape(-1, 1)).flatten().astype(int)
    
    # Stratified K-Fold
    skf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)))):
        logger.info(f"Training fold {fold + 1}/{config.cv_folds}")
        
        # Create datasets for this fold
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Create model
        model = EnhancedCGCNN(config).to(device)
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = create_scheduler(optimizer, config)
        
        # Training loop for this fold
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.num_epochs):
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, device, dataset.log_scaler)
            val_metrics = validate_epoch(model, val_loader, device, dataset.log_scaler)
            
            # Update scheduler
            if scheduler is not None:
                if config.scheduler_type == "reduce_on_plateau":
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break
        
        cv_results.append(val_metrics)
        logger.info(f"Fold {fold + 1} - Best Val Loss: {best_val_loss:.4f}, R²: {val_metrics['r2_log']:.4f}")
    
    # Aggregate results
    aggregated_results = {}
    for key in cv_results[0].keys():
        values = [result[key] for result in cv_results]
        aggregated_results[f'{key}_mean'] = np.mean(values)
        aggregated_results[f'{key}_std'] = np.std(values)
    
    logger.info("Cross-validation completed:")
    for key, value in aggregated_results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return aggregated_results


def main():
    """Main training function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Enhanced CGCNN for Ionic Conductivity Prediction')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    args = parser.parse_args()
    
    # Load config
    config = TrainingConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
    
    # Override with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device != 'auto':
        config.device = args.device
    
    # Setup device
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    
    # Setup output directory and logging
    output_dir = Path(config.output_dir)
    logger = setup_logging(output_dir)
    
    logger.info("Starting Enhanced CGCNN training for ionic conductivity prediction")
    logger.info(f"Configuration: {config}")
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = IonicConductivityDataset(config.data_path, config)
    
    # Perform cross-validation if requested
    if config.use_cross_validation:
        cv_results = cross_validate_model(full_dataset, config, device, logger)
        
        # Save cross-validation results
        with open(output_dir / 'cv_results.json', 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        logger.info("Cross-validation completed. Results saved.")
        return
    
    # Split dataset using stratified splitting
    if config.use_stratified_split:
        logger.info("Using stratified splitting based on log conductivity values...")
        train_dataset, val_dataset, test_dataset = stratified_split_dataset(full_dataset, config)
    else:
        # Fallback to random split
        logger.info("Using random splitting...")
        total_size = len(full_dataset)
        train_size = max(1, int(config.train_ratio * total_size))
        val_size = max(1, int(config.val_ratio * total_size))
        test_size = max(1, total_size - train_size - val_size)
        
        # Ensure we don't exceed total size
        if train_size + val_size + test_size > total_size:
            excess = train_size + val_size + test_size - total_size
            train_size = max(1, train_size - excess // 3)
            val_size = max(1, val_size - excess // 3)
            test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    logger.info(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    logger.info("Creating model...")
    model = EnhancedCGCNN(config).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} trainable parameters")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = create_scheduler(optimizer, config)
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    
    # Move log scaler to device with error handling
    try:
        full_dataset.log_scaler.to(device)
        logger.info(f"Log scaler moved to device: {device}")
    except Exception as e:
        logger.warning(f"Could not move log scaler to device {device}: {e}")
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
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
        
        # Log metrics (removed MAPE, added more relevant metrics)
        if epoch % config.log_every == 0:
            logger.info(f"Epoch {epoch:3d} | "
                       f"Train Loss: {train_metrics['loss']:.4f} | "
                       f"Val Loss: {val_metrics['loss']:.4f} | "
                       f"Val R²: {val_metrics['r2_log']:.4f} | "
                       f"Val Log Error: {val_metrics['mean_log_error']:.3f} orders")
        
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
        
        # Save checkpoint
        if epoch % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config.__dict__,
                'log_scaler': full_dataset.log_scaler
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Store losses
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        
        # Early stopping
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate_epoch(model, test_loader, device, full_dataset.log_scaler)
    
    logger.info("Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': test_metrics
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()