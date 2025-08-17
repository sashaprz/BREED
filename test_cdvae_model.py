#!/usr/bin/env python3
"""
CDVAE Model Testing Script

This script tests the CDVAE model on validation data, calculating and printing
performance metrics including loss, accuracy, and various reconstruction metrics.

Usage:
    python test_cdvae_model.py

Requirements:
    - CDVAE model weights at: generator/CDVAE/cdvae_weights.ckpt
    - Validation data at: generator/CDVAE/data/mp_20/val.pkl
"""

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from tqdm import tqdm

# Add CDVAE to path
sys.path.append(str(Path(__file__).parent / "generator" / "CDVAE"))

# CDVAE imports
from cdvae.pl_modules.model import CDVAE
from cdvae.pl_data.dataset import CrystDataset
from cdvae.pl_data.datamodule import CrystDataModule
from cdvae.common.data_utils import get_scaler_from_data_list
from cdvae.common.utils import PROJECT_ROOT

class CDVAEModelTester:
    """
    Class for testing CDVAE model performance on validation data.
    """
    
    def __init__(self, 
                 model_weights_path: str = "generator/CDVAE/cdvae_weights.ckpt",
                 val_data_path: str = "generator/CDVAE/data/mp_20/val.pkl"):
        """
        Initialize the CDVAE model tester.
        
        Args:
            model_weights_path: Path to the model checkpoint file
            val_data_path: Path to the validation data pickle file
        """
        self.model_weights_path = model_weights_path
        self.val_data_path = val_data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Initializing CDVAE Model Tester")
        print(f"   Device: {self.device}")
        print(f"   Model weights: {model_weights_path}")
        print(f"   Validation data: {val_data_path}")
        
        # Initialize components
        self.model = None
        self.val_dataset = None
        self.val_dataloader = None
        self.scaler = None
        self.lattice_scaler = None
        
    def load_validation_data(self) -> None:
        """
        Load and preprocess the validation data.
        """
        print(f"\n📊 Loading validation data...")
        
        if not os.path.exists(self.val_data_path):
            raise FileNotFoundError(f"Validation data not found at: {self.val_data_path}")
        
        # Load the pickle file to understand its structure
        with open(self.val_data_path, 'rb') as f:
            val_data = pickle.load(f)
        
        print(f"   Validation data type: {type(val_data)}")
        if isinstance(val_data, (list, np.ndarray)):
            print(f"   Number of validation samples: {len(val_data)}")
        elif isinstance(val_data, pd.DataFrame):
            print(f"   Validation DataFrame shape: {val_data.shape}")
            print(f"   Columns: {list(val_data.columns)}")
        
        # Create dataset configuration for validation data
        # We'll use similar parameters as the default configuration
        dataset_config = {
            'name': 'mp_20_val',
            'path': self.val_data_path,
            'prop': 'formation_energy_per_atom',  # Default property
            'niggli': True,
            'primitive': False,
            'graph_method': 'crystalnn',
            'preprocess_workers': 1,
            'lattice_scale_method': 'scale_length',
        }
        
        try:
            # Create the validation dataset
            self.val_dataset = CrystDataset(**dataset_config)
            print(f"   ✅ Successfully created validation dataset with {len(self.val_dataset)} samples")
            
            # Get scalers from the dataset
            self.lattice_scaler = get_scaler_from_data_list(
                self.val_dataset.cached_data, key='scaled_lattice')
            self.scaler = get_scaler_from_data_list(
                self.val_dataset.cached_data, key=self.val_dataset.prop)
            
            # Set scalers to dataset
            self.val_dataset.lattice_scaler = self.lattice_scaler
            self.val_dataset.scaler = self.scaler
            
            print(f"   ✅ Scalers initialized successfully")
            
        except Exception as e:
            print(f"   ❌ Error creating validation dataset: {e}")
            raise
    
    def load_model(self) -> None:
        """
        Load the pre-trained CDVAE model from checkpoint.
        """
        print(f"\n🤖 Loading CDVAE model...")
        
        if not os.path.exists(self.model_weights_path):
            raise FileNotFoundError(f"Model weights not found at: {self.model_weights_path}")
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(self.model_weights_path, map_location=self.device)
            print(f"   ✅ Checkpoint loaded successfully")
            
            # Extract hyperparameters from checkpoint
            if 'hyper_parameters' in checkpoint:
                hparams = checkpoint['hyper_parameters']
                print(f"   Model hyperparameters found in checkpoint")
            else:
                # Use default hyperparameters if not found in checkpoint
                print(f"   ⚠️  No hyperparameters in checkpoint, using defaults")
                hparams = {
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
                }
            
            # Create model instance
            self.model = CDVAE(**hparams)
            
            # Load state dict
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                print(f"   ✅ Model state dict loaded successfully")
            else:
                # Try loading the checkpoint directly as state dict
                self.model.load_state_dict(checkpoint)
                print(f"   ✅ Model weights loaded directly from checkpoint")
            
            # Set scalers to model
            if self.scaler is not None and self.lattice_scaler is not None:
                self.model.scaler = self.scaler
                self.model.lattice_scaler = self.lattice_scaler
                print(f"   ✅ Scalers set to model")
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            print(f"   ✅ Model loaded and ready for evaluation")
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            raise
    
    def create_dataloader(self, batch_size: int = 32) -> None:
        """
        Create DataLoader for validation data.
        
        Args:
            batch_size: Batch size for evaluation
        """
        print(f"\n📦 Creating validation DataLoader...")
        
        if self.val_dataset is None:
            raise ValueError("Validation dataset not loaded. Call load_validation_data() first.")
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"   ✅ DataLoader created with batch size {batch_size}")
        print(f"   Number of batches: {len(self.val_dataloader)}")
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data and compute metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n🔍 Evaluating model on validation data...")
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.val_dataloader is None:
            raise ValueError("DataLoader not created. Call create_dataloader() first.")
        
        # Initialize metrics storage
        all_losses = []
        all_coord_losses = []
        all_type_losses = []
        all_lattice_losses = []
        all_natom_losses = []
        all_kld_losses = []
        all_composition_losses = []
        
        # Accuracy metrics
        all_natom_accuracies = []
        all_type_accuracies = []
        
        # Lattice prediction metrics
        all_lengths_mard = []
        all_angles_mae = []
        all_volumes_mard = []
        
        total_samples = 0
        
        print(f"   Processing {len(self.val_dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc="Evaluating")):
                try:
                    # Move batch to device
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch, teacher_forcing=False, training=False)
                    
                    # Compute losses and metrics using model's compute_stats method
                    log_dict, total_loss = self.model.compute_stats(batch, outputs, prefix='val')
                    
                    # Store losses
                    all_losses.append(total_loss.item())
                    all_coord_losses.append(outputs['coord_loss'].item())
                    all_type_losses.append(outputs['type_loss'].item())
                    all_lattice_losses.append(outputs['lattice_loss'].item())
                    all_natom_losses.append(outputs['num_atom_loss'].item())
                    all_kld_losses.append(outputs['kld_loss'].item())
                    all_composition_losses.append(outputs['composition_loss'].item())
                    
                    # Store accuracy metrics from log_dict
                    if 'val_natom_accuracy' in log_dict:
                        all_natom_accuracies.append(log_dict['val_natom_accuracy'].item())
                    if 'val_type_accuracy' in log_dict:
                        all_type_accuracies.append(log_dict['val_type_accuracy'].item())
                    
                    # Store lattice metrics
                    if 'val_lengths_mard' in log_dict:
                        all_lengths_mard.append(log_dict['val_lengths_mard'].item())
                    if 'val_angles_mae' in log_dict:
                        all_angles_mae.append(log_dict['val_angles_mae'].item())
                    if 'val_volumes_mard' in log_dict:
                        all_volumes_mard.append(log_dict['val_volumes_mard'].item())
                    
                    total_samples += batch.num_graphs
                    
                except Exception as e:
                    print(f"   ⚠️  Error processing batch {batch_idx}: {e}")
                    continue
        
        # Compute average metrics
        metrics = {
            'total_loss': np.mean(all_losses),
            'coord_loss': np.mean(all_coord_losses),
            'type_loss': np.mean(all_type_losses),
            'lattice_loss': np.mean(all_lattice_losses),
            'natom_loss': np.mean(all_natom_losses),
            'kld_loss': np.mean(all_kld_losses),
            'composition_loss': np.mean(all_composition_losses),
            'total_samples': total_samples,
        }
        
        # Add accuracy metrics if available
        if all_natom_accuracies:
            metrics['natom_accuracy'] = np.mean(all_natom_accuracies)
        if all_type_accuracies:
            metrics['type_accuracy'] = np.mean(all_type_accuracies)
        
        # Add lattice metrics if available
        if all_lengths_mard:
            metrics['lengths_mard'] = np.mean(all_lengths_mard)
        if all_angles_mae:
            metrics['angles_mae'] = np.mean(all_angles_mae)
        if all_volumes_mard:
            metrics['volumes_mard'] = np.mean(all_volumes_mard)
        
        print(f"   ✅ Evaluation completed on {total_samples} samples")
        
        return metrics
    
    def print_results(self, metrics: Dict[str, float]) -> None:
        """
        Print formatted evaluation results.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        print(f"\n📊 CDVAE Model Evaluation Results")
        print(f"=" * 60)
        print(f"Total samples evaluated: {metrics['total_samples']}")
        print(f"")
        
        # Loss metrics
        print(f"🔥 Loss Metrics:")
        print(f"   Total Loss:        {metrics['total_loss']:.6f}")
        print(f"   Coordinate Loss:   {metrics['coord_loss']:.6f}")
        print(f"   Type Loss:         {metrics['type_loss']:.6f}")
        print(f"   Lattice Loss:      {metrics['lattice_loss']:.6f}")
        print(f"   Num Atoms Loss:    {metrics['natom_loss']:.6f}")
        print(f"   KLD Loss:          {metrics['kld_loss']:.6f}")
        print(f"   Composition Loss:  {metrics['composition_loss']:.6f}")
        print(f"")
        
        # Accuracy metrics
        if 'natom_accuracy' in metrics or 'type_accuracy' in metrics:
            print(f"🎯 Accuracy Metrics:")
            if 'natom_accuracy' in metrics:
                print(f"   Num Atoms Accuracy: {metrics['natom_accuracy']:.4f} ({metrics['natom_accuracy']*100:.2f}%)")
            if 'type_accuracy' in metrics:
                print(f"   Atom Type Accuracy: {metrics['type_accuracy']:.4f} ({metrics['type_accuracy']*100:.2f}%)")
            print(f"")
        
        # Lattice prediction metrics
        if any(key in metrics for key in ['lengths_mard', 'angles_mae', 'volumes_mard']):
            print(f"🏗️  Lattice Prediction Metrics:")
            if 'lengths_mard' in metrics:
                print(f"   Lengths MARD:      {metrics['lengths_mard']:.6f}")
            if 'angles_mae' in metrics:
                print(f"   Angles MAE:        {metrics['angles_mae']:.6f}°")
            if 'volumes_mard' in metrics:
                print(f"   Volumes MARD:      {metrics['volumes_mard']:.6f}")
            print(f"")
        
        print(f"=" * 60)
        
        # Performance interpretation
        print(f"📈 Performance Interpretation:")
        if 'natom_accuracy' in metrics:
            if metrics['natom_accuracy'] > 0.8:
                print(f"   ✅ Excellent number of atoms prediction (>{80:.0f}%)")
            elif metrics['natom_accuracy'] > 0.6:
                print(f"   ⚠️  Good number of atoms prediction (>{60:.0f}%)")
            else:
                print(f"   ❌ Poor number of atoms prediction (<{60:.0f}%)")
        
        if 'type_accuracy' in metrics:
            if metrics['type_accuracy'] > 0.7:
                print(f"   ✅ Excellent atom type prediction (>{70:.0f}%)")
            elif metrics['type_accuracy'] > 0.5:
                print(f"   ⚠️  Good atom type prediction (>{50:.0f}%)")
            else:
                print(f"   ❌ Poor atom type prediction (<{50:.0f}%)")
        
        if 'total_loss' in metrics:
            if metrics['total_loss'] < 1.0:
                print(f"   ✅ Low total loss (<1.0)")
            elif metrics['total_loss'] < 5.0:
                print(f"   ⚠️  Moderate total loss (<5.0)")
            else:
                print(f"   ❌ High total loss (≥5.0)")
    
    def run_evaluation(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Load validation data
            self.load_validation_data()
            
            # Load model
            self.load_model()
            
            # Create dataloader
            self.create_dataloader(batch_size=batch_size)
            
            # Evaluate model
            metrics = self.evaluate_model()
            
            # Print results
            self.print_results(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            raise


def main():
    """
    Main function to run CDVAE model evaluation.
    """
    print("🚀 Starting CDVAE Model Evaluation")
    print("=" * 60)
    
    # Initialize tester
    tester = CDVAEModelTester()
    
    # Run evaluation
    try:
        metrics = tester.run_evaluation(batch_size=16)  # Use smaller batch size for stability
        
        print(f"\n✅ Evaluation completed successfully!")
        return metrics
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()