"""
Fully optimized ML predictor that completely bypasses the original cgcnn_predict functions
This ensures models are loaded only once and cached properly.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import your ML prediction modules
from env.sei_predictor import SEIPredictor
from env.cei_predictor import CEIPredictor
from env.cgcnn_bandgap_ionic_cond_bulk_moduli.cgcnn_pretrained.cgcnn.model import CrystalGraphConvNet
from env.cgcnn_bandgap_ionic_cond_bulk_moduli.cgcnn_pretrained.cgcnn.data import CIFData, collate_pool
from env.cgcnn_bandgap_ionic_cond_bulk_moduli.main import Normalizer


class FullyOptimizedMLPredictor:
    """Fully optimized ML predictor with complete model caching - no reloading"""
    
    def __init__(self):
        # Configuration paths
        self.dataset_root = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\CIF_OBELiX"
        self.bandgap_model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\cgcnn_pretrained\band-gap.pth.tar"
        self.bulk_model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\cgcnn_pretrained\bulk-moduli.pth.tar"
        self.finetuned_model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\checkpoint.pth.tar"
        
        # Cached models - these will be loaded ONCE
        self._sei_predictor = None
        self._cei_predictor = None
        self._bandgap_model = None
        self._bulk_model = None
        self._finetuned_model = None
        self._finetuned_normalizer = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dataset cache
        self._dataset = None
        self._cif_filenames = None
        
        # Model loading status
        self._models_loaded = {
            'sei': False,
            'cei': False,
            'bandgap': False,
            'bulk': False,
            'ionic': False
        }
        
        print(f"FullyOptimizedMLPredictor initialized with device: {self._device}")
    
    def _load_dataset_once(self):
        """Load dataset information once and cache it"""
        if self._dataset is None or self._cif_filenames is None:
            print("Loading dataset information (one-time setup)...")
            cifs_folder = os.path.join(self.dataset_root, "cifs")
            self._dataset = CIFData(cifs_folder)
            
            # Read CIF filenames
            id_prop_path = os.path.join(cifs_folder, "id_prop.csv")
            id_prop_df = pd.read_csv(id_prop_path)
            cif_ids = id_prop_df.iloc[:, 0].tolist()
            self._cif_filenames = [cid + ".cif" for cid in cif_ids]
            print(f"Dataset loaded with {len(self._cif_filenames)} CIF files")
        
        return self._dataset, self._cif_filenames
    
    def _load_cgcnn_model_once(self, model_path: str, model_name: str):
        """Load a CGCNN model ONCE with proper architecture"""
        print(f"🔄 Loading {model_name} model (ONE TIME ONLY)...")
        
        # Ensure dataset is loaded first
        self._load_dataset_once()
        
        checkpoint = torch.load(model_path, map_location=self._device)
        
        # Get model architecture from sample
        sample = [self._dataset[0]]
        input_data, _, _ = collate_pool(sample)
        
        orig_atom_fea_len = input_data[0].shape[-1]
        nbr_fea_len = input_data[1].shape[-1]
        
        model = CrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=64,
            n_conv=3,
            h_fea_len=128,
            n_h=1,
            classification=False
        ).to(self._device)
        
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        print(f"✅ {model_name} model loaded and cached successfully!")
        return model, checkpoint
    
    def get_sei_predictor(self):
        """Get SEI predictor (loaded once)"""
        if not self._models_loaded['sei']:
            print("🔄 Loading SEI predictor (ONE TIME ONLY)...")
            self._sei_predictor = SEIPredictor()
            self._models_loaded['sei'] = True
            print("✅ SEI predictor loaded and cached!")
        return self._sei_predictor
    
    def get_cei_predictor(self):
        """Get CEI predictor (loaded once)"""
        if not self._models_loaded['cei']:
            print("🔄 Loading CEI predictor (ONE TIME ONLY)...")
            self._cei_predictor = CEIPredictor()
            self._models_loaded['cei'] = True
            print("✅ CEI predictor loaded and cached!")
        return self._cei_predictor
    
    def get_bandgap_model(self):
        """Get bandgap model (loaded once)"""
        if not self._models_loaded['bandgap']:
            self._bandgap_model, _ = self._load_cgcnn_model_once(
                self.bandgap_model_path, "Bandgap"
            )
            self._models_loaded['bandgap'] = True
        return self._bandgap_model
    
    def get_bulk_model(self):
        """Get bulk modulus model (loaded once)"""
        if not self._models_loaded['bulk']:
            self._bulk_model, _ = self._load_cgcnn_model_once(
                self.bulk_model_path, "Bulk Modulus"
            )
            self._models_loaded['bulk'] = True
        return self._bulk_model
    
    def get_finetuned_model(self):
        """Get finetuned ionic conductivity model (loaded once)"""
        if not self._models_loaded['ionic']:
            self._finetuned_model, checkpoint = self._load_cgcnn_model_once(
                self.finetuned_model_path, "Ionic Conductivity"
            )
            
            # Load normalizer if available
            if 'normalizer' in checkpoint:
                self._finetuned_normalizer = Normalizer(torch.tensor([0.0]))
                self._finetuned_normalizer.load_state_dict(checkpoint['normalizer'])
                print("✅ Normalizer loaded for ionic conductivity model")
            
            self._models_loaded['ionic'] = True
        
        return self._finetuned_model, self._finetuned_normalizer
    
    def predict_cgcnn_property(self, model, cif_file_path: str, normalizer=None):
        """Predict property using cached CGCNN model"""
        dataset, cif_filenames = self._load_dataset_once()
        
        cif_basename = os.path.basename(cif_file_path)
        sample_index = None
        for idx, fname in enumerate(cif_filenames):
            if fname == cif_basename:
                sample_index = idx
                break
        
        if sample_index is None:
            raise ValueError(f"CIF file {cif_file_path} not found in dataset")
        
        # Prepare sample
        sample = [dataset[sample_index]]
        input_data, targets, cif_ids_result = collate_pool(sample)
        
        input_vars = (
            input_data[0].to(self._device),
            input_data[1].to(self._device),
            input_data[2].to(self._device),
            input_data[3],  # crystal_atom_idx (stays on CPU)
        )
        
        with torch.no_grad():
            output = model(*input_vars)
            pred = output.cpu().numpy().flatten()[0]
        
        # Denormalize if normalizer available
        if normalizer is not None:
            pred_tensor = torch.tensor([pred])
            pred = normalizer.denorm(pred_tensor).item()
        
        return pred
    
    def predict_single_cif(self, cif_file_path: str, verbose: bool = False) -> Dict[str, Any]:
        """Run all predictions with fully cached models - NO RELOADING"""
        
        results = {
            "composition": self._extract_composition_from_cif(cif_file_path),
            "bandgap": 0.0,
            "sei_score": 0.0,
            "cei_score": 0.0,
            "ionic_conductivity": 0.0,
            "bulk_modulus": 0.0,
            "prediction_status": {
                "sei": "failed",
                "cei": "failed", 
                "bandgap": "failed",
                "bulk_modulus": "failed",
                "ionic_conductivity": "failed"
            }
        }
        
        if verbose:
            print(f"Processing CIF: {os.path.basename(cif_file_path)}")
        
        # SEI Prediction (using cached model)
        try:
            sei_predictor = self.get_sei_predictor()
            sei_results = sei_predictor.predict_from_cif(cif_file_path)
            if sei_results is not None and 'sei_score' in sei_results:
                results["sei_score"] = float(sei_results['sei_score'])
                results["prediction_status"]["sei"] = "success"
                if verbose:
                    print(f"  SEI Score: {results['sei_score']:.3f}")
        except Exception as e:
            if verbose:
                print(f"  SEI prediction failed: {e}")
        
        # CEI Prediction (using cached model)
        try:
            cei_predictor = self.get_cei_predictor()
            cei_results = cei_predictor.predict_from_cif(cif_file_path)
            if cei_results is not None and 'cei_score' in cei_results:
                results["cei_score"] = float(cei_results['cei_score'])
                results["prediction_status"]["cei"] = "success"
                if verbose:
                    print(f"  CEI Score: {results['cei_score']:.3f}")
        except Exception as e:
            if verbose:
                print(f"  CEI prediction failed: {e}")
        
        # Bandgap Prediction (using cached model - NO cgcnn_predict.main!)
        try:
            bandgap_model = self.get_bandgap_model()
            bandgap_pred = self.predict_cgcnn_property(bandgap_model, cif_file_path)
            results["bandgap"] = float(bandgap_pred)
            results["prediction_status"]["bandgap"] = "success"
            if verbose:
                print(f"  Bandgap: {results['bandgap']:.3f} eV")
        except Exception as e:
            if verbose:
                print(f"  Bandgap prediction failed: {e}")
        
        # Bulk Modulus Prediction (using cached model - NO cgcnn_predict.main!)
        try:
            bulk_model = self.get_bulk_model()
            bulk_pred = self.predict_cgcnn_property(bulk_model, cif_file_path)
            results["bulk_modulus"] = float(bulk_pred)
            results["prediction_status"]["bulk_modulus"] = "success"
            if verbose:
                print(f"  Bulk Modulus: {results['bulk_modulus']:.1f} GPa")
        except Exception as e:
            if verbose:
                print(f"  Bulk modulus prediction failed: {e}")
        
        # Ionic Conductivity Prediction (using cached model)
        try:
            finetuned_model, normalizer = self.get_finetuned_model()
            ic_pred = self.predict_cgcnn_property(finetuned_model, cif_file_path, normalizer)
            results["ionic_conductivity"] = float(ic_pred)
            results["prediction_status"]["ionic_conductivity"] = "success"
            if verbose:
                print(f"  Ionic Conductivity: {results['ionic_conductivity']:.2e} S/cm")
        except Exception as e:
            if verbose:
                print(f"  Ionic conductivity prediction failed: {e}")
        
        return results
    
    def _extract_composition_from_cif(self, cif_file_path: str) -> str:
        """Extract composition from CIF file"""
        try:
            with open(cif_file_path, 'r') as f:
                lines = f.readlines()
            
            # Look for data_ line which often contains composition info
            for line in lines:
                if line.startswith('data_'):
                    composition = line.replace('data_', '').strip()
                    if composition:
                        return composition
            
            # Fallback: use filename
            return os.path.splitext(os.path.basename(cif_file_path))[0]
        except:
            return os.path.splitext(os.path.basename(cif_file_path))[0]
    
    def print_model_status(self):
        """Print which models are loaded"""
        print("\n🔍 Model Loading Status:")
        for model_name, loaded in self._models_loaded.items():
            status = "✅ LOADED" if loaded else "❌ NOT LOADED"
            print(f"  {model_name.upper()}: {status}")


# Global instance for caching - CRITICAL for performance
_global_fully_optimized_predictor = None

def get_fully_optimized_predictor():
    """Get the global fully optimized predictor instance"""
    global _global_fully_optimized_predictor
    if _global_fully_optimized_predictor is None:
        _global_fully_optimized_predictor = FullyOptimizedMLPredictor()
    return _global_fully_optimized_predictor

def predict_single_cif_fully_optimized(cif_file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Fully optimized prediction function - models loaded ONCE and cached forever"""
    predictor = get_fully_optimized_predictor()
    return predictor.predict_single_cif(cif_file_path, verbose)

def print_predictor_status():
    """Print the current model loading status"""
    predictor = get_fully_optimized_predictor()
    predictor.print_model_status()