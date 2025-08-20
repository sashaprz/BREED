#!/usr/bin/env python3
"""
Validate Fine-tuned CGCNN Bulk Modulus Model

This script validates the fine-tuned bulk modulus model against the original model
to ensure improved performance on high bulk modulus inorganic crystals.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# Add paths for CGCNN imports
sys.path.append('env/property_predictions')
sys.path.append('env/property_predictions/cgcnn_pretrained')

from env.property_predictions.cgcnn_pretrained.cgcnn.model import CrystalGraphConvNet
from env.property_predictions.cgcnn_pretrained.cgcnn.data import CIFData, collate_pool
from env.property_predictions.main import Normalizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BulkModulusModelValidator:
    """Validate and compare bulk modulus models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model paths
        self.original_model_path = Path("env/property_predictions/bulk-moduli.pth.tar")
        self.finetuned_model_path = Path("outputs/bulk_modulus_finetuned/best_bulk_modulus_model.pth.tar")
        
        # Data paths
        self.test_data_path = Path("bulk_modulus_data/obelix_bulk_modulus_high.csv")
        self.temp_data_dir = Path("validation_temp_data")
        
        # Results
        self.validation_results = {}
        
        logger.info(f"Validator initialized with device: {self.device}")
    
    def load_model(self, model_path: Path, model_name: str) -> Tuple[CrystalGraphConvNet, Normalizer]:
        """Load a CGCNN model and its normalizer"""
        logger.info(f"Loading {model_name} model from {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model architecture (we'll get dimensions from test data)
        # For now, use standard architecture
        model = CrystalGraphConvNet(
            orig_atom_fea_len=92,  # Standard atom features
            nbr_fea_len=41,        # Standard neighbor features
            atom_fea_len=64,
            n_conv=3,
            h_fea_len=128,
            n_h=1,
            classification=False
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        # Load normalizer
        if 'normalizer' in checkpoint:
            normalizer = Normalizer(torch.tensor([0.0]).to(self.device))
            normalizer.load_state_dict(checkpoint['normalizer'])
            normalizer.mean = normalizer.mean.to(self.device)
            normalizer.std = normalizer.std.to(self.device)
        else:
            logger.warning(f"No normalizer found in {model_name} model")
            normalizer = None
        
        logger.info(f"{model_name} model loaded successfully")
        return model, normalizer
    
    def prepare_test_data(self) -> torch.utils.data.DataLoader:
        """Prepare test data for validation"""
        logger.info("Preparing test data...")
        
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.test_data_path}")
        
        # Load test data
        df = pd.read_csv(self.test_data_path)
        
        # Filter for high bulk modulus materials (>30 GPa)
        df_high_bm = df[
            (df['bulk_modulus_vrh'] > 30) & 
            (df['energy_above_hull'] < 0.1) &
            (df['match_score'] < 0.5)
        ].copy()
        
        logger.info(f"Using {len(df_high_bm)} high bulk modulus materials for validation")
        
        # Create temporary dataset
        self.temp_data_dir.mkdir(exist_ok=True)
        
        # Copy CIF files and create dataset structure
        valid_entries = []
        for idx, row in df_high_bm.iterrows():
            cif_path = Path(row['cif_path'])
            if cif_path.exists():
                # Copy CIF file
                dst_path = self.temp_data_dir / f"{len(valid_entries):04d}_{cif_path.stem}.cif"
                import shutil
                shutil.copy2(cif_path, dst_path)
                valid_entries.append((len(valid_entries), row['bulk_modulus_vrh']))
        
        # Create id_prop.csv
        id_prop_path = self.temp_data_dir / "id_prop.csv"
        with open(id_prop_path, 'w') as f:
            f.write("id,target\n")
            for idx, target in valid_entries:
                f.write(f"{idx:04d},{target:.6f}\n")
        
        # Copy atom_init.json
        atom_init_src = Path("env/property_predictions/cgcnn_pretrained/atom_init.json")
        atom_init_dst = self.temp_data_dir / "atom_init.json"
        if atom_init_src.exists():
            shutil.copy2(atom_init_src, atom_init_dst)
        
        # Create CGCNN dataset
        dataset = CIFData(str(self.temp_data_dir))
        
        # Create data loader
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=False,
            collate_fn=collate_pool, num_workers=0
        )
        
        logger.info(f"Test dataset created with {len(dataset)} samples")
        return test_loader
    
    def evaluate_model(self, model: CrystalGraphConvNet, normalizer: Normalizer, 
                      test_loader: torch.utils.data.DataLoader, model_name: str) -> Dict:
        """Evaluate a model on test data"""
        logger.info(f"Evaluating {model_name} model...")
        
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for input_data, batch_targets, _ in test_loader:
                # Move data to device
                input_vars = (
                    input_data[0].to(self.device),
                    input_data[1].to(self.device),
                    input_data[2].to(self.device),
                    [idx.to(self.device) for idx in input_data[3]]
                )
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                outputs = model(*input_vars)
                
                # Denormalize predictions
                if normalizer is not None:
                    outputs_denorm = normalizer.denorm(outputs.squeeze())
                else:
                    outputs_denorm = outputs.squeeze()
                
                # Store results
                predictions.extend(outputs_denorm.cpu().numpy().tolist())
                targets.extend(batch_targets.cpu().numpy().tolist())
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        # Calculate metrics for different bulk modulus ranges
        ranges = [
            (30, 50, "Medium (30-50 GPa)"),
            (50, 100, "High (50-100 GPa)"),
            (100, 500, "Very High (>100 GPa)")
        ]
        
        range_metrics = {}
        for min_bm, max_bm, range_name in ranges:
            mask = (targets >= min_bm) & (targets < max_bm)
            if np.any(mask):
                range_targets = targets[mask]
                range_preds = predictions[mask]
                range_metrics[range_name] = {
                    'count': len(range_targets),
                    'mae': mean_absolute_error(range_targets, range_preds),
                    'rmse': np.sqrt(mean_squared_error(range_targets, range_preds)),
                    'r2': r2_score(range_targets, range_preds),
                    'mean_target': np.mean(range_targets),
                    'mean_pred': np.mean(range_preds)
                }
        
        results = {
            'model_name': model_name,
            'overall_mae': mae,
            'overall_rmse': rmse,
            'overall_r2': r2,
            'num_samples': len(targets),
            'target_range': [float(np.min(targets)), float(np.max(targets))],
            'pred_range': [float(np.min(predictions)), float(np.max(predictions))],
            'range_metrics': range_metrics,
            'predictions': predictions.tolist(),
            'targets': targets.tolist()
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Overall MAE: {mae:.2f} GPa")
        logger.info(f"  Overall RMSE: {rmse:.2f} GPa")
        logger.info(f"  Overall R²: {r2:.3f}")
        
        for range_name, metrics in range_metrics.items():
            logger.info(f"  {range_name}: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f} ({metrics['count']} samples)")
        
        return results
    
    def compare_models(self) -> None:
        """Compare original and fine-tuned models"""
        logger.info("🔍 Comparing Original vs Fine-tuned Models")
        logger.info("=" * 60)
        
        # Prepare test data
        test_loader = self.prepare_test_data()
        
        # Load and evaluate original model
        try:
            original_model, original_normalizer = self.load_model(
                self.original_model_path, "Original"
            )
            original_results = self.evaluate_model(
                original_model, original_normalizer, test_loader, "Original"
            )
            self.validation_results['original'] = original_results
        except Exception as e:
            logger.error(f"Failed to evaluate original model: {e}")
            original_results = None
        
        # Load and evaluate fine-tuned model
        try:
            finetuned_model, finetuned_normalizer = self.load_model(
                self.finetuned_model_path, "Fine-tuned"
            )
            finetuned_results = self.evaluate_model(
                finetuned_model, finetuned_normalizer, test_loader, "Fine-tuned"
            )
            self.validation_results['finetuned'] = finetuned_results
        except Exception as e:
            logger.error(f"Failed to evaluate fine-tuned model: {e}")
            finetuned_results = None
        
        # Compare results
        if original_results and finetuned_results:
            self.analyze_improvements(original_results, finetuned_results)
        
        # Save results
        self.save_validation_results()
    
    def analyze_improvements(self, original: Dict, finetuned: Dict) -> None:
        """Analyze improvements from fine-tuning"""
        logger.info("\n📊 IMPROVEMENT ANALYSIS:")
        logger.info("=" * 40)
        
        # Overall improvements
        mae_improvement = ((original['overall_mae'] - finetuned['overall_mae']) / original['overall_mae']) * 100
        rmse_improvement = ((original['overall_rmse'] - finetuned['overall_rmse']) / original['overall_rmse']) * 100
        r2_improvement = finetuned['overall_r2'] - original['overall_r2']
        
        logger.info(f"Overall Improvements:")
        logger.info(f"  MAE: {original['overall_mae']:.2f} → {finetuned['overall_mae']:.2f} GPa ({mae_improvement:+.1f}%)")
        logger.info(f"  RMSE: {original['overall_rmse']:.2f} → {finetuned['overall_rmse']:.2f} GPa ({rmse_improvement:+.1f}%)")
        logger.info(f"  R²: {original['overall_r2']:.3f} → {finetuned['overall_r2']:.3f} ({r2_improvement:+.3f})")
        
        # Range-specific improvements
        logger.info(f"\nRange-specific Improvements:")
        for range_name in original['range_metrics']:
            if range_name in finetuned['range_metrics']:
                orig_mae = original['range_metrics'][range_name]['mae']
                fine_mae = finetuned['range_metrics'][range_name]['mae']
                mae_imp = ((orig_mae - fine_mae) / orig_mae) * 100
                
                orig_r2 = original['range_metrics'][range_name]['r2']
                fine_r2 = finetuned['range_metrics'][range_name]['r2']
                r2_imp = fine_r2 - orig_r2
                
                logger.info(f"  {range_name}:")
                logger.info(f"    MAE: {orig_mae:.2f} → {fine_mae:.2f} GPa ({mae_imp:+.1f}%)")
                logger.info(f"    R²: {orig_r2:.3f} → {fine_r2:.3f} ({r2_imp:+.3f})")
        
        # Assessment
        logger.info(f"\n🎯 ASSESSMENT:")
        if mae_improvement > 0 and r2_improvement > 0:
            logger.info("✅ Fine-tuning was SUCCESSFUL!")
            logger.info("   Both MAE and R² improved on high bulk modulus materials.")
        elif mae_improvement > 0 or r2_improvement > 0:
            logger.info("⚠️  Fine-tuning showed MIXED results.")
            logger.info("   Some metrics improved, others may have degraded.")
        else:
            logger.info("❌ Fine-tuning was NOT effective.")
            logger.info("   Consider adjusting hyperparameters or training data.")
    
    def create_comparison_plots(self) -> None:
        """Create comparison plots"""
        if 'original' not in self.validation_results or 'finetuned' not in self.validation_results:
            logger.warning("Cannot create plots - missing model results")
            return
        
        logger.info("Creating comparison plots...")
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        orig_targets = np.array(self.validation_results['original']['targets'])
        orig_preds = np.array(self.validation_results['original']['predictions'])
        fine_targets = np.array(self.validation_results['finetuned']['targets'])
        fine_preds = np.array(self.validation_results['finetuned']['predictions'])
        
        # Plot 1: Prediction vs Target scatter plots
        ax1 = axes[0, 0]
        ax1.scatter(orig_targets, orig_preds, alpha=0.6, label='Original Model', s=50)
        ax1.scatter(fine_targets, fine_preds, alpha=0.6, label='Fine-tuned Model', s=50)
        
        # Perfect prediction line
        min_val = min(np.min(orig_targets), np.min(fine_targets))
        max_val = max(np.max(orig_targets), np.max(fine_targets))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax1.set_xlabel('True Bulk Modulus (GPa)')
        ax1.set_ylabel('Predicted Bulk Modulus (GPa)')
        ax1.set_title('Prediction vs Target Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        ax2 = axes[0, 1]
        orig_errors = orig_preds - orig_targets
        fine_errors = fine_preds - fine_targets
        
        ax2.hist(orig_errors, bins=20, alpha=0.6, label='Original Model', density=True)
        ax2.hist(fine_errors, bins=20, alpha=0.6, label='Fine-tuned Model', density=True)
        ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Prediction Error (GPa)')
        ax2.set_ylabel('Density')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance by bulk modulus range
        ax3 = axes[1, 0]
        ranges = ['Medium (30-50 GPa)', 'High (50-100 GPa)', 'Very High (>100 GPa)']
        orig_maes = []
        fine_maes = []
        
        for range_name in ranges:
            if range_name in self.validation_results['original']['range_metrics']:
                orig_maes.append(self.validation_results['original']['range_metrics'][range_name]['mae'])
                fine_maes.append(self.validation_results['finetuned']['range_metrics'][range_name]['mae'])
            else:
                orig_maes.append(0)
                fine_maes.append(0)
        
        x = np.arange(len(ranges))
        width = 0.35
        
        ax3.bar(x - width/2, orig_maes, width, label='Original Model', alpha=0.8)
        ax3.bar(x + width/2, fine_maes, width, label='Fine-tuned Model', alpha=0.8)
        
        ax3.set_xlabel('Bulk Modulus Range')
        ax3.set_ylabel('MAE (GPa)')
        ax3.set_title('Performance by Bulk Modulus Range')
        ax3.set_xticks(x)
        ax3.set_xticklabels(ranges, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Absolute error vs target value
        ax4 = axes[1, 1]
        orig_abs_errors = np.abs(orig_errors)
        fine_abs_errors = np.abs(fine_errors)
        
        ax4.scatter(orig_targets, orig_abs_errors, alpha=0.6, label='Original Model', s=30)
        ax4.scatter(fine_targets, fine_abs_errors, alpha=0.6, label='Fine-tuned Model', s=30)
        ax4.set_xlabel('True Bulk Modulus (GPa)')
        ax4.set_ylabel('Absolute Error (GPa)')
        ax4.set_title('Absolute Error vs Target Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("outputs/bulk_modulus_validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plots saved to {output_dir / 'model_comparison.png'}")
    
    def save_validation_results(self) -> None:
        """Save validation results"""
        output_dir = Path("outputs/bulk_modulus_validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_dir / "validation_results.json", 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Save summary
        if 'original' in self.validation_results and 'finetuned' in self.validation_results:
            summary = {
                'validation_date': pd.Timestamp.now().isoformat(),
                'original_model': {
                    'mae': self.validation_results['original']['overall_mae'],
                    'rmse': self.validation_results['original']['overall_rmse'],
                    'r2': self.validation_results['original']['overall_r2']
                },
                'finetuned_model': {
                    'mae': self.validation_results['finetuned']['overall_mae'],
                    'rmse': self.validation_results['finetuned']['overall_rmse'],
                    'r2': self.validation_results['finetuned']['overall_r2']
                },
                'improvements': {
                    'mae_improvement_percent': ((self.validation_results['original']['overall_mae'] - 
                                               self.validation_results['finetuned']['overall_mae']) / 
                                              self.validation_results['original']['overall_mae']) * 100,
                    'r2_improvement': (self.validation_results['finetuned']['overall_r2'] - 
                                     self.validation_results['original']['overall_r2'])
                }
            }
            
            with open(output_dir / "validation_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
        
        # Create comparison plots
        self.create_comparison_plots()
        
        logger.info(f"Validation results saved to {output_dir}")
    
    def cleanup(self) -> None:
        """Clean up temporary files"""
        if self.temp_data_dir.exists():
            import shutil
            shutil.rmtree(self.temp_data_dir)
            logger.info("Temporary files cleaned up")


def main():
    """Main function"""
    print("🔍 CGCNN Bulk Modulus Model Validation")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        Path("env/property_predictions/bulk-moduli.pth.tar"),
        Path("outputs/bulk_modulus_finetuned/best_bulk_modulus_model.pth.tar"),
        Path("bulk_modulus_data/obelix_bulk_modulus_high.csv")
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure you have:")
        print("1. Run 'python extract_mp_bulk_modulus.py'")
        print("2. Run 'python finetune_cgcnn_bulk_modulus.py'")
        return
    
    # Create validator
    validator = BulkModulusModelValidator()
    
    try:
        # Run validation
        validator.compare_models()
        
        print("\n🎉 Validation completed successfully!")
        print("📁 Results saved to: outputs/bulk_modulus_validation/")
        print("📊 Check validation_summary.json for key metrics")
        print("📈 Check model_comparison.png for visual comparison")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        validator.cleanup()


if __name__ == "__main__":
    main()