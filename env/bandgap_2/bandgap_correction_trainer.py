import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add the modnet repository to the path if cloned locally
# Uncomment if you clone the repo to your current directory
# sys.path.append('modnet')

try:
    from modnet.preprocessing import MODData
    from modnet.models import MODNetModel
    from modnet.featurizers import MODFeaturizer
    from pymatgen.core import Structure
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install modnet: pip install modnet")
    print("And ensure pymatgen is installed: pip install pymatgen")
    sys.exit(1)

class BandgapCorrectionTrainer:
    def __init__(self, data_path, cif_directory=None):
        """
        Initialize the bandgap correction trainer
        
        Args:
            data_path (str): Path to the CSV file with bandgap data
            cif_directory (str): Directory containing CIF files (optional)
        """
        self.data_path = data_path
        self.cif_directory = cif_directory
        self.data = None
        self.moddata = None
        self.model = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the JARVIS bandgap data"""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        
        # Remove rows with missing HSE bandgaps (our target)
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=['hse_bandgap'])
        print(f"Removed {initial_count - len(self.data)} rows with missing HSE bandgaps")
        
        # Remove rows with failed CIF files (but keep them for now since we'll create dummy structures)
        failed_cifs = self.data['cif_path'].str.contains('FAILED', na=False).sum()
        print(f"Found {failed_cifs} failed CIF files - will create dummy structures from formulas")
        print(f"Remaining samples after filtering: {len(self.data)}")
        
        # Create compositions from formulas (simpler approach without CIF files)
        print("Creating compositions from formulas...")
        from pymatgen.core import Composition
        compositions = []
        valid_indices = []
        
        for idx, row in self.data.iterrows():
            try:
                # Create composition from formula
                comp = Composition(row['formula'])
                compositions.append(comp)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"Error processing formula {row['formula']}: {e}")
                continue
        
        # Filter data to only valid compositions
        self.data = self.data.loc[valid_indices].reset_index(drop=True)
        print(f"Successfully processed {len(compositions)} compositions")
        
        return compositions
    
    def create_moddata(self, compositions):
        """Create MODData object for training"""
        print("Creating MODData object...")
        
        # Prepare target data - we want to predict HSE from PBE
        targets = self.data['hse_bandgap'].values
        target_names = ['hse_bandgap']
        
        # Create materials IDs
        materials_ids = [f"material_{i}" for i in range(len(compositions))]
        
        # Create MODData object
        self.moddata = MODData(
            materials=compositions,
            targets=targets,
            target_names=target_names,
            structure_ids=materials_ids
        )
        
        # Add PBE bandgap as an additional feature
        # This is crucial - we want the model to learn the correction based on PBE values
        pbe_features = pd.DataFrame({
            'pbe_bandgap': self.data['pbe_bandgap'].values
        }, index=materials_ids)
        
        # Featurize structures
        print("Featurizing structures...")
        featurizer = MODFeaturizer()
        self.moddata.featurize(featurizer)
        
        # Add PBE bandgap as custom feature
        for mat_id, pbe_bg in zip(materials_ids, self.data['pbe_bandgap'].values):
            if mat_id in self.moddata.df_featurized.index:
                self.moddata.df_featurized.loc[mat_id, 'pbe_bandgap'] = pbe_bg
        
        print(f"Features shape: {self.moddata.df_featurized.shape}")
        print(f"Targets shape: {self.moddata.df_targets.shape}")
        
        return self.moddata
    
    def prepare_training_data(self):
        """Prepare and split data for training"""
        print("Preparing training data...")
        
        # Feature selection and preprocessing
        self.moddata.feature_selection(n=300)  # Select top 300 features
        
        # Split data
        train_indices, test_indices = train_test_split(
            range(len(self.moddata.df_targets)), 
            test_size=0.2, 
            random_state=42
        )
        
        return train_indices, test_indices
    
    def train_model(self, train_indices, test_indices):
        """Train the MODNet model"""
        print("Training MODNet model...")
        
        # Initialize model
        self.model = MODNetModel(
            targets=['hse_bandgap'],
            weights={'hse_bandgap': 1.0},
            num_neurons=[[128], [64], [32]],  # 3-layer architecture
            num_classes={'hse_bandgap': 1},
            n_feat=len(self.moddata.get_optimal_descriptors())
        )
        
        # Train the model
        self.model.fit(
            self.moddata,
            train_indices=train_indices,
            val_fraction=0.2,
            lr=0.001,
            epochs=500,
            batch_size=32,
            verbose=1
        )
        
        return self.model
    
    def evaluate_model(self, test_indices):
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        # Make predictions
        predictions = self.model.predict(self.moddata, return_prob=False)
        
        # Get test predictions and targets
        test_predictions = predictions.loc[self.moddata.df_targets.index[test_indices], 'hse_bandgap']
        test_targets = self.moddata.df_targets.loc[self.moddata.df_targets.index[test_indices], 'hse_bandgap']
        
        # Calculate metrics
        mae = mean_absolute_error(test_targets, test_predictions)
        r2 = r2_score(test_targets, test_predictions)
        
        print(f"Test MAE: {mae:.4f} eV")
        print(f"Test R²: {r2:.4f}")
        
        # Plot results
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(test_targets, test_predictions, alpha=0.6)
        plt.plot([test_targets.min(), test_targets.max()], 
                 [test_targets.min(), test_targets.max()], 'r--')
        plt.xlabel('True Correction (eV)')
        plt.ylabel('Predicted Correction (eV)')
        plt.title(f'Bandgap Correction Prediction\nMAE: {mae:.4f} eV, R²: {r2:.4f}')
        
        plt.subplot(1, 2, 2)
        residuals = test_predictions - test_targets
        plt.scatter(test_predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Correction (eV)')
        plt.ylabel('Residuals (eV)')
        plt.title('Residuals Plot')
        
        plt.tight_layout()
        plt.savefig('bandgap_correction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return mae, r2, predictions
    
    def save_model(self, filename='modnet_bandgap_correction.pkl'):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filename)
            print(f"Model saved as {filename}")
        else:
            print("No model to save. Train the model first.")
    
    def predict_corrections(self, new_structures, pbe_bandgaps):
        """
        Predict bandgap corrections for new materials
        
        Args:
            new_structures: List of pymatgen Structure objects
            pbe_bandgaps: List of PBE bandgap values
        
        Returns:
            Array of predicted HSE bandgaps
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Create MODData for new structures
        materials_ids = [f"new_material_{i}" for i in range(len(new_structures))]
        
        new_moddata = MODData(
            structures=new_structures,
            structure_ids=materials_ids
        )
        
        # Featurize
        featurizer = MODFeaturizer()
        new_moddata.featurize(featurizer)
        
        # Add PBE bandgaps as features
        for mat_id, pbe_bg in zip(materials_ids, pbe_bandgaps):
            if mat_id in new_moddata.df_featurized.index:
                new_moddata.df_featurized.loc[mat_id, 'pbe_bandgap'] = pbe_bg
        
        # Make predictions
        predictions = self.model.predict(new_moddata, return_prob=False)
        hse_predictions = predictions['hse_bandgap'].values
        
        return hse_predictions

def main():
    # Configuration
    data_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\bandgap_2\jarvis_paired_data\jarvis_paired_bandgaps.csv"
    
    # Optional: specify CIF directory if you have the actual crystal structures
    # cif_directory = r"C:\path\to\your\cif\files"
    cif_directory = None
    
    # Initialize trainer
    trainer = BandgapCorrectionTrainer(data_path, cif_directory)
    
    try:
        # Load and preprocess data
        compositions = trainer.load_and_preprocess_data()
        
        # Create MODData
        moddata = trainer.create_moddata(compositions)
        
        # Prepare training data
        train_indices, test_indices = trainer.prepare_training_data()
        
        # Train model
        model = trainer.train_model(train_indices, test_indices)
        
        # Evaluate model
        mae, r2, predictions = trainer.evaluate_model(test_indices)
        
        # Save model
        trainer.save_model('modnet_pbe_to_hse_correction.pkl')
        
        print("\nTraining completed successfully!")
        print(f"Model performance - MAE: {mae:.4f} eV, R²: {r2:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()