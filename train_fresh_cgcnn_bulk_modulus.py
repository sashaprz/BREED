#!/usr/bin/env python3
"""
Train a fresh CGCNN model from scratch on high bulk modulus materials.
This avoids the bias issue by training only on materials with realistic high bulk modulus values.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Add parent directory to path so we can import from env (same as property_prediction_script.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import exactly the same modules as property_prediction_script.py
from env.property_predictions.cgcnn_pretrained.cgcnn.model import CrystalGraphConvNet
from env.property_predictions.cgcnn_pretrained.cgcnn.data import CIFData, collate_pool

# Set cgcnn_path for atom_init.json
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cgcnn_path = os.path.join(base_dir, "env", "property_predictions", "cgcnn_pretrained")

def clean_bulk_modulus_data(data_dir="high_bulk_modulus_training", max_bulk_modulus=500):
    """Clean the dataset by removing outliers and unrealistic values"""
    
    print(f"🧹 Cleaning bulk modulus data (removing values > {max_bulk_modulus} GPa)...")
    
    # Load and clean each split
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(data_dir, f"{split}.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        # Read data
        df = pd.read_csv(csv_path, header=None, names=['cif_file', 'bulk_modulus'])
        original_count = len(df)
        
        # Remove outliers
        df_clean = df[df['bulk_modulus'] <= max_bulk_modulus].copy()
        cleaned_count = len(df_clean)
        
        # Save cleaned data
        df_clean.to_csv(csv_path, header=False, index=False)
        
        print(f"   • {split}: {original_count} → {cleaned_count} materials ({original_count - cleaned_count} outliers removed)")
    
    print("✅ Data cleaning complete")

def create_cgcnn_model(atom_fea_len=64, h_fea_len=128, n_conv=3, n_h=1):
    """Create a fresh CGCNN model for bulk modulus prediction - matches existing architecture"""
    
    # Load atom features from the existing pretrained models directory
    atom_init_file = "env/property_predictions/cgcnn_pretrained/atom_init.json"
    
    if not os.path.exists(atom_init_file):
        raise FileNotFoundError(f"Atom initialization file not found: {atom_init_file}")
    
    with open(atom_init_file) as f:
        atom_features = json.load(f)
    
    # Create model with EXACT same architecture as existing models
    model = CrystalGraphConvNet(
        orig_atom_fea_len=len(atom_features[list(atom_features.keys())[0]]),
        nbr_fea_len=41,  # Standard bond feature length
        atom_fea_len=atom_fea_len,
        n_conv=n_conv,  # 3 conv layers (same as existing models)
        h_fea_len=h_fea_len,
        n_h=n_h,
        classification=False  # Regression task for bulk modulus prediction
    )
    
    return model

def train_model(data_dir="high_bulk_modulus_training", 
                model_save_path="fresh_cgcnn_bulk_modulus.pth",
                epochs=200,
                batch_size=32,
                learning_rate=0.01):
    """Train fresh CGCNN model on high bulk modulus materials"""
    
    print("🚀 Training fresh CGCNN model for bulk modulus prediction...")
    
    # Clean data first
    clean_bulk_modulus_data(data_dir)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Prepare data in CGCNN format (single directory with id_prop.csv)
    structures_dir = os.path.join(data_dir, "structures")
    
    # Copy atom_init.json to structures directory
    atom_init_source = "env/property_predictions/cgcnn_pretrained/atom_init.json"
    atom_init_dest = os.path.join(structures_dir, "atom_init.json")
    if not os.path.exists(atom_init_dest):
        import shutil
        shutil.copy2(atom_init_source, atom_init_dest)
    
    # Create separate datasets for train/val/test by copying CSV files as id_prop.csv
    import tempfile
    import shutil
    
    # Create temporary directories for each split
    train_dir = os.path.join(data_dir, "train_data")
    val_dir = os.path.join(data_dir, "val_data")
    test_dir = os.path.join(data_dir, "test_data")
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
        # Copy atom_init.json to each split directory
        shutil.copy2(atom_init_source, os.path.join(split_dir, "atom_init.json"))
    
    # Copy CSV files as id_prop.csv and create symlinks to CIF files
    for split, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        csv_path = os.path.join(data_dir, f"{split}.csv")
        id_prop_path = os.path.join(split_dir, "id_prop.csv")
        shutil.copy2(csv_path, id_prop_path)
        
        # Read CSV to get CIF files needed for this split
        import pandas as pd
        df = pd.read_csv(csv_path, header=None, names=['cif_file', 'bulk_modulus'])
        
        # Create symlinks to CIF files
        for _, row in df.iterrows():
            cif_file = row['cif_file']
            cif_id = os.path.splitext(cif_file)[0]  # Remove .cif extension for id_prop.csv
            source_cif = os.path.join(structures_dir, cif_file)
            dest_cif = os.path.join(split_dir, cif_file)  # Keep original filename with .cif
            
            if os.path.exists(source_cif) and not os.path.exists(dest_cif):
                os.symlink(os.path.abspath(source_cif), dest_cif)
        
        # Update the CSV to use cif_id (without .cif extension) as expected by CIFData
        df_updated = df.copy()
        df_updated['cif_file'] = df_updated['cif_file'].apply(lambda x: os.path.splitext(x)[0])
        df_updated.to_csv(id_prop_path, header=False, index=False)
    
    # Create datasets
    train_dataset = CIFData(train_dir)
    val_dataset = CIFData(val_dir)
    test_dataset = CIFData(test_dir)
    
    print(f"   • Train: {len(train_dataset)} materials")
    print(f"   • Validation: {len(val_dataset)} materials") 
    print(f"   • Test: {len(test_dataset)} materials")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_pool, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_pool, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_pool, num_workers=0)
    
    # Create model with same architecture as existing CGCNN models
    model = create_cgcnn_model(atom_fea_len=64, h_fea_len=128, n_conv=3, n_h=1)
    model.to(device)
    
    print(f"   • Model architecture: atom_fea_len=64, h_fea_len=128, n_conv=3, n_h=1")
    print(f"   • Task: Bulk modulus regression (NOT ionic conductivity)")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=20)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🏋️ Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target, cif_ids = batch
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
            target = target.to(device).float()
            
            optimizer.zero_grad()
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(target)
            train_count += len(target)
        
        train_loss /= train_count
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target, cif_ids = batch
                atom_fea = atom_fea.to(device)
                nbr_fea = nbr_fea.to(device)
                nbr_fea_idx = nbr_fea_idx.to(device)
                crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
                target = target.to(device).float()
                
                output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                loss = criterion(output.squeeze(), target)
                
                val_loss += loss.item() * len(target)
                val_count += len(target)
        
        val_loss /= val_count
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_args': {
                    'atom_fea_len': 64,
                    'h_fea_len': 128,
                    'n_conv': 3,  # Match existing architecture
                    'n_h': 1
                }
            }, model_save_path)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}: Train Loss = {train_loss:.2f}, Val Loss = {val_loss:.2f}")
    
    print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.2f}")
    print(f"   Model saved to: {model_save_path}")
    
    # Test evaluation
    print("\n📊 Evaluating on test set...")
    model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
    model.eval()
    
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target, cif_ids = batch
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
            
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            
            test_predictions.extend(output.squeeze().cpu().numpy())
            test_targets.extend(target.numpy())
    
    # Calculate metrics
    mae = mean_absolute_error(test_targets, test_predictions)
    r2 = r2_score(test_targets, test_predictions)
    
    print(f"   • Test MAE: {mae:.2f} GPa")
    print(f"   • Test R²: {r2:.3f}")
    
    # Save training history and results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'test_mae': mae,
        'test_r2': r2,
        'test_predictions': test_predictions,
        'test_targets': test_targets
    }
    
    results_path = model_save_path.replace('.pth', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.scatter(test_targets, test_predictions, alpha=0.6)
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
    plt.xlabel('True Bulk Modulus (GPa)')
    plt.ylabel('Predicted Bulk Modulus (GPa)')
    plt.title(f'Test Set Predictions (R² = {r2:.3f})')
    
    plt.tight_layout()
    plot_path = model_save_path.replace('.pth', '_training_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   • Training plots saved to: {plot_path}")
    
    return model, results

def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description='Train fresh CGCNN for bulk modulus prediction')
    parser.add_argument('--data-dir', default='high_bulk_modulus_training', 
                       help='Directory containing training data')
    parser.add_argument('--model-path', default='fresh_cgcnn_bulk_modulus.pth',
                       help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Train model
    model, results = train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print(f"\n🎉 Fresh CGCNN bulk modulus model training complete!")
    print(f"   Model ready for high bulk modulus prediction")

if __name__ == "__main__":
    main()