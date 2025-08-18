#!/usr/bin/env python3
"""
Complete Training Pipeline for Enhanced CGCNN Ionic Conductivity Prediction

This script orchestrates the complete training process:
1. Download CIF files from Materials Project
2. Prepare data for Enhanced CGCNN
3. Train the Enhanced CGCNN model
4. Evaluate results
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CGCNNTrainingPipeline:
    """Complete training pipeline for Enhanced CGCNN"""
    
    def __init__(self, config_path: str = "cgcnn_config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # Setup paths
        self.data_dir = Path("data")
        self.cif_dir = self.data_dir / "cifs"
        self.output_dir = Path(self.config.get("output_dir", "outputs/cgcnn_ionic_conductivity"))
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.cif_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline status
        self.status = {
            'cif_download': False,
            'data_preparation': False,
            'model_training': False,
            'evaluation': False
        }
    
    def load_config(self) -> dict:
        """Load training configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        else:
            # Default configuration
            config = {
                "data_path": "data/ionic_conductivity_dataset.pkl",
                "prop_name": "ionic_conductivity",
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "hidden_dim": 128,
                "num_layers": 4,
                "dropout": 0.1,
                "batch_size": 32,
                "learning_rate": 0.001,
                "num_epochs": 200,
                "patience": 20,
                "conductivity_threshold": 1e-12,
                "output_dir": "outputs/cgcnn_ionic_conductivity",
                "device": "auto"
            }
            logger.info("Using default configuration")
        
        return config
    
    def step_1_download_cifs(self) -> bool:
        """Step 1: Download CIF files from Materials Project"""
        logger.info("=" * 60)
        logger.info("STEP 1: DOWNLOADING CIF FILES")
        logger.info("=" * 60)
        
        try:
            # Check if CIF files already exist
            existing_cifs = list(self.cif_dir.glob("*.cif"))
            if len(existing_cifs) > 100:  # Reasonable threshold
                logger.info(f"Found {len(existing_cifs)} existing CIF files, skipping download")
                self.status['cif_download'] = True
                return True
            
            # Run CIF downloader
            logger.info("Starting CIF download from Materials Project...")
            result = subprocess.run([
                sys.executable, "download_cifs_for_cgcnn.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("CIF download completed successfully")
                logger.info(result.stdout)
                self.status['cif_download'] = True
                return True
            else:
                logger.error("CIF download failed")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"Error in CIF download step: {e}")
            return False
    
    def step_2_prepare_data(self) -> bool:
        """Step 2: Prepare data for Enhanced CGCNN"""
        logger.info("=" * 60)
        logger.info("STEP 2: PREPARING DATA FOR ENHANCED CGCNN")
        logger.info("=" * 60)
        
        try:
            # Check if dataset already exists
            dataset_path = Path(self.config["data_path"])
            if dataset_path.exists():
                logger.info(f"Dataset already exists at {dataset_path}, skipping preparation")
                self.status['data_preparation'] = True
                return True
            
            # Run data preparation
            logger.info("Starting data preparation...")
            result = subprocess.run([
                sys.executable, "prepare_cgcnn_data.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Data preparation completed successfully")
                logger.info(result.stdout)
                self.status['data_preparation'] = True
                return True
            else:
                logger.error("Data preparation failed")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"Error in data preparation step: {e}")
            return False
    
    def step_3_train_model(self) -> bool:
        """Step 3: Train Enhanced CGCNN model"""
        logger.info("=" * 60)
        logger.info("STEP 3: TRAINING ENHANCED CGCNN MODEL")
        logger.info("=" * 60)
        
        try:
            # Check if model already exists
            model_path = self.output_dir / "best_model.pt"
            if model_path.exists():
                logger.info(f"Trained model already exists at {model_path}")
                response = input("Do you want to retrain? (y/N): ").strip().lower()
                if response != 'y':
                    self.status['model_training'] = True
                    return True
            
            # Run Enhanced CGCNN training
            logger.info("Starting Enhanced CGCNN training...")
            cmd = [
                sys.executable, "enhanced_cgcnn_ionic_conductivity.py",
                "--config", str(self.config_path),
                "--data_path", self.config["data_path"],
                "--output_dir", str(self.output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                logger.info("Model training completed successfully")
                self.status['model_training'] = True
                return True
            else:
                logger.error("Model training failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in model training step: {e}")
            return False
    
    def step_4_evaluate_results(self) -> bool:
        """Step 4: Evaluate training results"""
        logger.info("=" * 60)
        logger.info("STEP 4: EVALUATING RESULTS")
        logger.info("=" * 60)
        
        try:
            # Check for training results
            results_files = [
                self.output_dir / "test_results.json",
                self.output_dir / "training_history.json",
                self.output_dir / "best_model.pt"
            ]
            
            missing_files = [f for f in results_files if not f.exists()]
            if missing_files:
                logger.error(f"Missing result files: {missing_files}")
                return False
            
            # Load and display results
            with open(self.output_dir / "test_results.json", 'r') as f:
                test_results = json.load(f)
            
            logger.info("TRAINING RESULTS:")
            logger.info("-" * 40)
            for metric, value in test_results.items():
                if isinstance(value, float):
                    logger.info(f"{metric}: {value:.6f}")
                else:
                    logger.info(f"{metric}: {value}")
            
            # Load training history
            with open(self.output_dir / "training_history.json", 'r') as f:
                history = json.load(f)
            
            if 'train_losses' in history and 'val_losses' in history:
                final_train_loss = history['train_losses'][-1]
                final_val_loss = history['val_losses'][-1]
                logger.info(f"Final training loss: {final_train_loss:.6f}")
                logger.info(f"Final validation loss: {final_val_loss:.6f}")
            
            self.status['evaluation'] = True
            return True
            
        except Exception as e:
            logger.error(f"Error in evaluation step: {e}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete training pipeline"""
        logger.info("🚀 STARTING ENHANCED CGCNN TRAINING PIPELINE")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        # Step 1: Download CIF files
        if not self.step_1_download_cifs():
            logger.error("❌ Pipeline failed at Step 1: CIF Download")
            return False
        
        # Step 2: Prepare data
        if not self.step_2_prepare_data():
            logger.error("❌ Pipeline failed at Step 2: Data Preparation")
            return False
        
        # Step 3: Train model
        if not self.step_3_train_model():
            logger.error("❌ Pipeline failed at Step 3: Model Training")
            return False
        
        # Step 4: Evaluate results
        if not self.step_4_evaluate_results():
            logger.error("❌ Pipeline failed at Step 4: Evaluation")
            return False
        
        # Success!
        logger.info("=" * 60)
        logger.info("🎉 ENHANCED CGCNN TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📁 Results saved to: {self.output_dir}")
        logger.info(f"🤖 Best model: {self.output_dir / 'best_model.pt'}")
        logger.info(f"📊 Test results: {self.output_dir / 'test_results.json'}")
        logger.info("=" * 60)
        
        return True
    
    def print_status(self) -> None:
        """Print current pipeline status"""
        logger.info("PIPELINE STATUS:")
        logger.info("-" * 30)
        for step, completed in self.status.items():
            status = "✅" if completed else "❌"
            logger.info(f"{status} {step.replace('_', ' ').title()}")


def main():
    """Main function"""
    print("🧪 Enhanced CGCNN for Ionic Conductivity Prediction")
    print("=" * 60)
    
    # Create pipeline
    pipeline = CGCNNTrainingPipeline()
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("Your Enhanced CGCNN model is ready for ionic conductivity prediction!")
    else:
        print("\n❌ Training pipeline failed.")
        print("Check the logs above for error details.")
        pipeline.print_status()


if __name__ == "__main__":
    main()