#!/usr/bin/env python3
"""
Simple GPU CGCNN Training - Avoiding multiprocessing issues
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
import warnings
warnings.filterwarnings('ignore')


class LogSpaceScaler:
    """Simple log-space scaler"""
    
    def __init__(self, epsilon: float = 1e-20):
        self.epsilon = epsilon
        self.log_mean = None
        self.log_std = None
        
    def fit(self, conductivity_values: torch.Tensor):
        log_values = torch.log10(conductivity_values + self.epsilon)
        self.log_mean = torch.mean(log_values)
        self.log_std = torch.std(log_values) + 1e-8
        
    def transform(self, conductivity_values: torch.Tensor) -> torch.Tensor:
        log_values = torch.log10(conductivity_values + self.epsilon)
        return (log_values - self.log_mean) / self.log_std
    
    def inverse_transform(self, normalized_log_values: torch.Tensor) -> torch.Tensor:
        log_values = normalized_log_values * self.log_std + self.log_mean
        return torch.pow(10, log_values) - self.epsilon


class SimpleCGCNNConv(MessagePassing):
    """Simple CGCNN convolution"""
    
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__(aggr='add', node_dim=0)
        
        self.node_linear = nn.Linear(node_dim, hidden_dim)
        self.edge_linear = nn.Linear(1, hidden_dim)
        
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()
        )
        
    def forward(self, x, edge_index, edge_attr):
        x_transformed = self.node_linear(x)
        edge_attr_transformed = self.edge_linear(edge_attr)
        
        out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr_transformed)
        out = self.update_net(torch.cat([out, x], dim=-1))
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        message_input = torch.cat([x_i, x_j], dim=-1)
        message = self.message_net(message_input)
        return message * edge_attr


class SimpleCGCNN(nn.Module):
    """Simple CGCNN model"""
    
    def __init__(self, hidden_dim=256, num_layers=6, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Atom embedding
        self.atom_embedding = nn.Embedding(119, hidden_dim)
        
        # CGCNN layers
        self.conv_layers = nn.ModuleList([
            SimpleCGCNNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, batch):
        x = self.atom_embedding(batch.atom_types)
        edge_attr = batch.distances.unsqueeze(-1)
        
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            x_new = conv(x, batch.edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = self.dropout(x_new)
            
            if i > 0:
                x = x + x_new
            else:
                x = x_new
        
        x = global_mean_pool(x, batch.batch)
        out = self.predictor(x)
        
        return out


class SimpleDataset(Dataset):
    """Simple dataset without multiprocessing issues"""
    
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} samples")
        
        # Setup scaler on CPU
        conductivity_values = torch.tensor([d['ionic_conductivity'] for d in self.data], dtype=torch.float32)
        self.log_scaler = LogSpaceScaler()
        self.log_scaler.fit(conductivity_values)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        conductivity = torch.tensor([sample['ionic_conductivity']], dtype=torch.float32)
        log_conductivity = self.log_scaler.transform(conductivity)
        
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


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, log_scaler: LogSpaceScaler) -> dict:
    """Compute metrics"""
    pred_conductivity = log_scaler.inverse_transform(predictions)
    true_conductivity = log_scaler.inverse_transform(targets)
    
    pred_conductivity = torch.clamp(pred_conductivity, min=1e-20)
    true_conductivity = torch.clamp(true_conductivity, min=1e-20)
    
    mae_log = F.l1_loss(predictions, targets).item()
    mse_log = F.mse_loss(predictions, targets).item()
    rmse_log = np.sqrt(mse_log)
    
    # Orders of magnitude error
    log_error = torch.abs(torch.log10(pred_conductivity + 1e-20) - torch.log10(true_conductivity + 1e-20))
    mean_log_error = torch.mean(log_error).item()
    
    # R² score
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2_log = 1 - (ss_res / ss_tot).item()
    
    return {
        'mae_log': mae_log,
        'mse_log': mse_log,
        'rmse_log': rmse_log,
        'mean_log_error': mean_log_error,
        'r2_log': r2_log
    }


def train_epoch(model, dataloader, optimizer, device, log_scaler):
    """Train epoch"""
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    """Validate epoch"""
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
    # Force specific GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1 which is free
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU")
    
    # Setup output directory
    output_dir = Path('outputs/simple_gpu_cgcnn')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = SimpleDataset('data/ionic_conductivity_dataset_with_graphs.pkl')
    
    # Simple random split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders (no multiprocessing)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                             collate_fn=Batch.from_data_list, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                           collate_fn=Batch.from_data_list, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                            collate_fn=Batch.from_data_list, num_workers=0)
    
    # Create model
    print("Creating model...")
    model = SimpleCGCNN(hidden_dim=256, num_layers=6, dropout=0.2).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    
    # Training loop
    print("🚀 Starting GPU training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    for epoch in range(300):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, full_dataset.log_scaler)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, full_dataset.log_scaler)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val R²: {val_metrics['r2_log']:.4f} | "
                  f"Mean Log Error: {val_metrics['mean_log_error']:.3f} orders")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss - 1e-5:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_metrics['loss'],
                'log_scaler': full_dataset.log_scaler
            }, output_dir / 'best_model.pt')
            
            print(f"💾 New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch}")
            break
    
    # Test evaluation
    print("📊 Evaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate_epoch(model, test_loader, device, full_dataset.log_scaler)
    
    print("🎯 Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"✅ Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()