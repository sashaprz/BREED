#!/usr/bin/env python3
"""
Fine-tune CGCNN Bulk Modulus Model on OBELiX Dataset

This script fine-tunes the pre-trained CGCNN bulk modulus model on high bulk modulus
inorganic crystals from the OBELiX dataset with Materials Project bulk modulus values.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add paths for CGCNN imports
sys.path.append('env/property_predictions')
sys.path.append('env/property_predictions/cgcnn_pretrained')

from env.property_predictions.cgcnn_pretrained.cgcnn.model import CrystalGraphConvNet
from env.property_predictions.cgcnn_pretrained.cgcnn.data import CIFData, collate_pool
from env.property_predictions.main import Normalizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BulkModulusFineTuner:
    """Fine-tune CGCNN bulk modulus model for high bulk modulus inorganic crystals"""
    
    def __init__(self, config_path: str = "bulk_modulus_finetune_config.json"):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths
        self.data_dir = Path("bulk_modulus_data")
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model paths
        self.pretrained_model_path = Path("env/property_predictions/bulk-moduli.pth.tar")
        
        # Training state
        self.model = None
        self.normalizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Results tracking
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_maes': [],
            'val_maes': [],
            'learning_rates': []
        }
        
        logger.info(f"Fine-tuner initialized with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load fine-tuning configuration"""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            # Default fine-tuning configuration
            config = {
                "data_file": "bulk_modulus_data/obelix_bulk_modulus_high.csv",
                "min_bulk_modulus": 20.0,
                "max_bulk_modulus": 500.0,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "batch_size": 16,
                "learning_rate": 1e-5,
                "weight_decay": 1e-4,
                "num_epochs": 100,
                "patience": 15,
                "min_lr": 1e-7,
                "lr_factor": 0.5,
                "lr_patience": 8,
                "output_dir": "outputs/bulk_modulus_finetuned",
                "save_best_only": True,
                "weighted_loss": True,
                "high_bm_weight": 2.0,
                "validation_freq": 5
            }
            
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created default configuration: {config_path}")
        
        return config
    
    def load_bulk_modulus_data(self) -> pd.DataFrame:
        """Load bulk modulus data from Materials Project extraction"""
        data_file = Path(self.config["data_file"])
        
        if not data_file.exists():
            logger.error(f"Bulk modulus data file not found: {data_file}")
            logger.error("Please run 'python extract_mp_bulk_modulus.py' first")
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} bulk modulus entries")
        
        # Filter by bulk modulus range
        min_bm = self.config["min_bulk_modulus"]
        max_bm = self.config["max_bulk_modulus"]
        
        df_filtered = df[
            (df['bulk_modulus_vrh'] >= min_bm) & 
            (df['bulk_modulus_vrh'] <= max_bm) &
            (df['energy_above_hull'] < 0.1) &  # Stable materials
            (df['match_score'] < 0.5)  # Good structure matches
        ].copy()
        
        logger.info(f"Filtered to {len(df_filtered)} materials with bulk modulus {min_bm}-{max_bm} GPa")
        
        if len(df_filtered) < 10:
            logger.warning("Very few materials for fine-tuning. Consider relaxing filters.")
        
        return df_filtered
    
    def create_cif_dataset(self, df: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """Create CIF file list and target values for CGCNN"""
        cif_files = []
        targets = []
        
        for _, row in df.iterrows():
            cif_path = Path(row['cif_path'])
            if cif_path.exists():
                cif_files.append(str(cif_path))
                targets.append(float(row['bulk_modulus_vrh']))
            else:
                logger.warning(f"CIF file not found: {cif_path}")
        
        logger.info(f"Created dataset with {len(cif_files)} CIF files")
        return cif_files, targets
    
    def prepare_data_loaders(self, cif_files: List[str], targets: List[float]) -> None:
        """Prepare PyTorch data loaders for training"""
        logger.info("Preparing data loaders...")
        
        # Split data
        train_ratio = self.config["train_ratio"]
        val_ratio = self.config["val_ratio"]
        test_ratio = self.config["test_ratio"]
        
        # First split: train + val vs test
        train_val_files, test_files, train_val_targets, test_targets = train_test_split(
            cif_files, targets, test_size=test_ratio, random_state=42, stratify=None
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train_files, val_files, train_targets, val_targets = train_test_split(
            train_val_files, train_val_targets, test_size=val_size, random_state=42
        )
        
        logger.info(f"Data split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Create temporary directories with CIF files and targets
        temp_dir = self.output_dir / "temp_data"
        temp_dir.mkdir(exist_ok=True)
        
        # Create datasets
        def create_temp_dataset(files, targets, name):
            dataset_dir = temp_dir / name
            dataset_dir.mkdir(exist_ok=True)
            
            # Copy CIF files
            for i, (cif_file, target) in enumerate(zip(files, targets)):
                src_path = Path(cif_file)
                dst_path = dataset_dir / f"{i:04d}.cif"
                
                # Copy CIF file
                import shutil
                shutil.copy2(src_path, dst_path)
            
            # Create id_prop.csv (no header - CGCNN expects raw data only)
            id_prop_path = dataset_dir / "id_prop.csv"
            with open(id_prop_path, 'w') as f:
                for i, target in enumerate(targets):
                    f.write(f"{i:04d},{'%.6f' % target}\n")
            
            # Copy atom_init.json
            atom_init_src = Path("env/property_predictions/cgcnn_pretrained/atom_init.json")
            atom_init_dst = dataset_dir / "atom_init.json"
            if atom_init_src.exists():
                shutil.copy2(atom_init_src, atom_init_dst)
            
            return str(dataset_dir)
        
        # Create temporary datasets
        train_dataset_path = create_temp_dataset(train_files, train_targets, "train")
        val_dataset_path = create_temp_dataset(val_files, val_targets, "val")
        test_dataset_path = create_temp_dataset(test_files, test_targets, "test")
        
        # Create CGCNN datasets
        train_dataset = CIFData(train_dataset_path)
        val_dataset = CIFData(val_dataset_path)
        test_dataset = CIFData(test_dataset_path)
        
        # Create data loaders
        batch_size = self.config["batch_size"]
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=collate_pool, num_workers=0
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_pool, num_workers=0
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_pool, num_workers=0
        )
        
        logger.info("Data loaders created successfully")
    
    def load_pretrained_model(self) -> None:
        """Load pre-trained CGCNN bulk modulus model"""
        logger.info(f"Loading pre-trained model from {self.pretrained_model_path}")
        
        if not self.pretrained_model_path.exists():
            logger.error(f"Pre-trained model not found: {self.pretrained_model_path}")
            raise FileNotFoundError(f"Model file not found: {self.pretrained_model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
        
        # Get model architecture parameters from a sample
        sample_data = next(iter(self.train_loader))
        input_data, _, _ = sample_data
        
        orig_atom_fea_len = input_data[0].shape[-1]
        nbr_fea_len = input_data[1].shape[-1]
        
        # Create model with same architecture as pre-trained model
        # Based on the error, the pre-trained model has:
        # - 4 conv layers (convs.3 exists)
        # - h_fea_len=32 (conv_to_fc.weight shape is [32, 64])
        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=64,
            n_conv=4,
            h_fea_len=32,
            n_h=1,
            classification=False
        ).to(self.device)
        
        # Load pre-trained weights
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # Load normalizer
        if 'normalizer' in checkpoint:
            self.normalizer = Normalizer(torch.tensor([0.0]).to(self.device))
            self.normalizer.load_state_dict(checkpoint['normalizer'])
            # Ensure normalizer tensors are on correct device
            if hasattr(self.normalizer.mean, 'to'):
                self.normalizer.mean = self.normalizer.mean.to(self.device)
            else:
                self.normalizer.mean = torch.tensor(self.normalizer.mean).to(self.device)
            
            if hasattr(self.normalizer.std, 'to'):
                self.normalizer.std = self.normalizer.std.to(self.device)
            else:
                self.normalizer.std = torch.tensor(self.normalizer.std).to(self.device)
        else:
            # Create new normalizer from training data
            logger.info("Creating new normalizer from training data")
            train_targets = []
            for batch in self.train_loader:
                _, targets, _ = batch
                train_targets.extend(targets.tolist())
            
            self.normalizer = Normalizer(torch.tensor(train_targets).to(self.device))
        
        logger.info("Pre-trained model loaded successfully")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def create_weighted_loss_function(self) -> nn.Module:
        """Create weighted loss function that emphasizes high bulk modulus materials"""
        if not self.config["weighted_loss"]:
            return nn.MSELoss()
        
        class WeightedMSELoss(nn.Module):
            def __init__(self, high_bm_threshold=50.0, high_bm_weight=2.0):
                super().__init__()
                self.threshold = high_bm_threshold
                self.weight = high_bm_weight
                self.mse = nn.MSELoss(reduction='none')
            
            def forward(self, predictions, targets):
                # Denormalize targets to get actual bulk modulus values
                if hasattr(self, 'normalizer') and self.normalizer is not None:
                    actual_targets = self.normalizer.denorm(targets)
                else:
                    actual_targets = targets
                
                # Calculate base loss
                losses = self.mse(predictions, targets)
                
                # Apply higher weight to high bulk modulus materials
                weights = torch.where(
                    actual_targets > self.threshold,
                    torch.tensor(self.weight, device=targets.device),
                    torch.tensor(1.0, device=targets.device)
                )
                
                weighted_losses = losses * weights
                return weighted_losses.mean()
        
        loss_fn = WeightedMSELoss(
            high_bm_threshold=self.config.get("high_bm_threshold", 50.0),
            high_bm_weight=self.config["high_bm_weight"]
        )
        loss_fn.normalizer = self.normalizer
        
        return loss_fn
    
    def train_epoch(self, model: nn.Module, train_loader, optimizer, criterion, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (input_data, targets, _) in enumerate(pbar):
            # Move data to device
            input_vars = (
                input_data[0].to(self.device),
                input_data[1].to(self.device),
                input_data[2].to(self.device),
                [idx.to(self.device) for idx in input_data[3]]
            )
            targets = targets.to(self.device)
            
            # Normalize targets
            targets_norm = self.normalizer.norm(targets)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(*input_vars)
            loss = criterion(outputs, targets_norm.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                outputs_denorm = self.normalizer.denorm(outputs.squeeze())
                mae = torch.mean(torch.abs(outputs_denorm - targets)).item()
            
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{mae:.2f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate(self, model: nn.Module, val_loader, criterion) -> Tuple[float, float, List[float], List[float]]:
        """Validate the model"""
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for input_data, targets, _ in val_loader:
                # Move data to device
                input_vars = (
                    input_data[0].to(self.device),
                    input_data[1].to(self.device),
                    input_data[2].to(self.device),
                    [idx.to(self.device) for idx in input_data[3]]
                )
                targets = targets.to(self.device)
                
                # Normalize targets
                targets_norm = self.normalizer.norm(targets)
                
                # Forward pass
                outputs = model(*input_vars)
                loss = criterion(outputs, targets_norm.unsqueeze(1))
                
                # Denormalize for metrics
                outputs_denorm = self.normalizer.denorm(outputs.squeeze())
                mae = torch.mean(torch.abs(outputs_denorm - targets)).item()
                
                total_loss += loss.item()
                total_mae += mae
                num_batches += 1
                
                # Store predictions and targets
                all_predictions.extend(outputs_denorm.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae, all_predictions, all_targets
    
    def fine_tune(self) -> None:
        """Main fine-tuning process"""
        logger.info("🚀 Starting CGCNN bulk modulus fine-tuning")
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config["lr_factor"],
            patience=self.config["lr_patience"],
            min_lr=self.config["min_lr"],
            verbose=True
        )
        
        criterion = self.create_weighted_loss_function()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.config["num_epochs"] + 1):
            # Train
            train_loss, train_mae = self.train_epoch(
                self.model, self.train_loader, optimizer, criterion, epoch
            )
            
            # Validate
            if epoch % self.config["validation_freq"] == 0:
                val_loss, val_mae, val_preds, val_targets = self.validate(
                    self.model, self.val_loader, criterion
                )
                
                # Update learning rate
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Store history
                self.training_history['train_losses'].append(train_loss)
                self.training_history['val_losses'].append(val_loss)
                self.training_history['train_maes'].append(train_mae)
                self.training_history['val_maes'].append(val_mae)
                self.training_history['learning_rates'].append(current_lr)
                
                # Log progress
                logger.info(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train MAE={train_mae:.2f}, "
                          f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.2f}, LR={current_lr:.2e}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    if self.config["save_best_only"]:
                        self.save_model(epoch, val_loss, is_best=True)
                        logger.info(f"✅ New best model saved (Val Loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config["patience"]:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
            else:
                # Store training history only
                self.training_history['train_losses'].append(train_loss)
                self.training_history['train_maes'].append(train_mae)
                
                logger.info(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train MAE={train_mae:.2f}")
        
        logger.info("🎉 Fine-tuning completed!")
    
    def save_model(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'normalizer': self.normalizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if is_best:
            model_path = self.output_dir / "best_bulk_modulus_model.pth.tar"
        else:
            model_path = self.output_dir / f"bulk_modulus_model_epoch_{epoch}.pth.tar"
        
        torch.save(checkpoint, model_path)
    
    def evaluate_on_test_set(self) -> Dict:
        """Evaluate fine-tuned model on test set"""
        logger.info("📊 Evaluating on test set...")
        
        # Load best model
        best_model_path = self.output_dir / "best_bulk_modulus_model.pth.tar"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded best model for evaluation")
        
        # Evaluate
        criterion = nn.MSELoss()
        test_loss, test_mae, test_preds, test_targets = self.validate(
            self.model, self.test_loader, criterion
        )
        
        # Calculate additional metrics
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        test_r2 = r2_score(test_targets, test_preds)
        
        # Calculate metrics for high bulk modulus materials (>50 GPa)
        high_bm_mask = np.array(test_targets) > 50
        if np.any(high_bm_mask):
            high_bm_targets = np.array(test_targets)[high_bm_mask]
            high_bm_preds = np.array(test_preds)[high_bm_mask]
            high_bm_mae = mean_absolute_error(high_bm_targets, high_bm_preds)
            high_bm_rmse = np.sqrt(mean_squared_error(high_bm_targets, high_bm_preds))
            high_bm_r2 = r2_score(high_bm_targets, high_bm_preds)
        else:
            high_bm_mae = high_bm_rmse = high_bm_r2 = None
        
        results = {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'high_bm_mae': high_bm_mae,
            'high_bm_rmse': high_bm_rmse,
            'high_bm_r2': high_bm_r2,
            'num_test_samples': len(test_targets),
            'num_high_bm_samples': np.sum(high_bm_mask) if high_bm_mask is not None else 0,
            'bulk_modulus_range': [float(np.min(test_targets)), float(np.max(test_targets))],
            'predictions': test_preds,
            'targets': test_targets
        }
        
        # Save results
        with open(self.output_dir / "test_results.json", 'w') as f:
            json.dump({k: v for k, v in results.items() if k not in ['predictions', 'targets']}, f, indent=2)
        
        # Save detailed predictions
        pred_df = pd.DataFrame({
            'target': test_targets,
            'prediction': test_preds,
            'error': np.array(test_preds) - np.array(test_targets),
            'abs_error': np.abs(np.array(test_preds) - np.array(test_targets))
        })
        pred_df.to_csv(self.output_dir / "test_predictions.csv", index=False)
        
        # Log results
        logger.info("📊 TEST SET RESULTS:")
        logger.info(f"   Test MAE: {test_mae:.2f} GPa")
        logger.info(f"   Test RMSE: {test_rmse:.2f} GPa")
        logger.info(f"   Test R²: {test_r2:.3f}")
        if high_bm_mae is not None:
            logger.info(f"   High BM (>50 GPa) MAE: {high_bm_mae:.2f} GPa")
            logger.info(f"   High BM (>50 GPa) R²: {high_bm_r2:.3f}")
        
        return results
    
    def save_training_history(self) -> None:
        """Save training history and create plots"""
        # Save history
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        epochs = range(1, len(self.training_history['train_losses']) + 1)
        ax1.plot(epochs, self.training_history['train_losses'], 'b-', label='Train Loss')
        if self.training_history['val_losses']:
            val_epochs = range(self.config["validation_freq"], 
                             len(self.training_history['val_losses']) * self.config["validation_freq"] + 1,
                             self.config["validation_freq"])
            ax1.plot(val_epochs, self.training_history['val_losses'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(epochs, self.training_history['train_maes'], 'b-', label='Train MAE')
        if self.training_history['val_maes']:
            ax2.plot(val_epochs, self.training_history['val_maes'], 'r-', label='Val MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (GPa)')
        ax2.set_title('Training and Validation MAE')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        if self.training_history['learning_rates']:
            ax3.plot(val_epochs, self.training_history['learning_rates'], 'g-')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_yscale('log')
            ax3.grid(True)
        
        # Remove empty subplot
        ax4.remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history saved to {self.output_dir}")
    
    def run_complete_finetuning(self) -> None:
        """Run the complete fine-tuning pipeline"""
        logger.info("🚀 CGCNN Bulk Modulus Fine-Tuning Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load data
            logger.info("Step 1: Loading bulk modulus data...")
            df = self.load_bulk_modulus_data()
            
            # Step 2: Prepare datasets
            logger.info("Step 2: Preparing datasets...")
            cif_files, targets = self.create_cif_dataset(df)
            self.prepare_data_loaders(cif_files, targets)
            
            # Step 3: Load pre-trained model
            logger.info("Step 3: Loading pre-trained model...")
            self.load_pretrained_model()
            
            # Step 4: Fine-tune
            logger.info("Step 4: Fine-tuning model...")
            self.fine_tune()
            
            # Step 5: Evaluate
            logger.info("Step 5: Evaluating on test set...")
            test_results = self.evaluate_on_test_set()
            
            # Step 6: Save results
            logger.info("Step 6: Saving results...")
            self.save_training_history()
            
            logger.info("🎉 Fine-tuning pipeline completed successfully!")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            logger.info(f"🤖 Best model: {self.output_dir / 'best_bulk_modulus_model.pth.tar'}")
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function"""
    print("🔧 CGCNN Bulk Modulus Fine-Tuning")
    print("=" * 50)
    
    # Check if bulk modulus data exists
    data_file = Path("bulk_modulus_data/obelix_bulk_modulus_high.csv")
    if not data_file.exists():
        print(f"❌ Bulk modulus data not found: {data_file}")
        print("Please run 'python extract_mp_bulk_modulus.py' first")
        return
    
    # Create fine-tuner
    finetuner = BulkModulusFineTuner()
    
    # Run complete fine-tuning pipeline
    try:
        finetuner.run_complete_finetuning()
        print("\n🎉 Fine-tuning completed successfully!")
        print("Your fine-tuned CGCNN bulk modulus model is ready!")
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()