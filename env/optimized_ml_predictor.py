"""
Optimized ML predictor with model caching for the Pareto GA
This avoids reloading models for every prediction, significantly improving performance.
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
from env.cgcnn_bandgap_ionic_cond_bulk_moduli.cgcnn_pretrained import cgcnn_predict
from env.cgcnn_bandgap_ionic_cond_bulk_moduli.cgcnn_pretrained.cgcnn.model import CrystalGraphConvNet
from env.cgcnn_bandgap_ionic_cond_bulk_moduli.cgcnn_pretrained.cgcnn.data import CIFData, collate_pool
from env.cgcnn_bandgap_ionic_cond_bulk_moduli.main import Normalizer


class OptimizedMLPredictor:
    """Optimized ML predictor with model caching"""
    
    def __init__(self):
        # Configuration paths
        self.dataset_root = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\CIF_OBELiX"
        self.bandgap_model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\cgcnn_pretrained\band-gap.pth.tar"
        self.bulk_model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\cgcnn_pretrained\bulk-moduli.pth.tar"
        self.finetuned_model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\checkpoint.pth.tar"
        
        # Cached models
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
        
        print(f"OptimizedMLPredictor initialized with device: {self._device}")
    
    def _get_sei_predictor(self):
        """Get cached SEI predictor"""
        if self._sei_predictor is None:
            print("Loading SEI predictor...")
            self._sei_predictor = SEIPredictor()
        return self._sei_predictor
    
    def _get_cei_predictor(self):
        """Get cached CEI predictor"""
        if self._cei_predictor is None:
            print("Loading CEI predictor...")
            self._cei_predictor = CEIPredictor()
        return self._cei_predictor
    
    def _load_cgcnn_model(self, model_path: str):
        """Load a CGCNN model with proper architecture"""
        print(f"Loading CGCNN model from {os.path.basename(model_path)}...")
        checkpoint = torch.load(model_path, map_location=self._device)
        
        # Get model architecture parameters from a sample
        if self._dataset is None:
            cifs_folder = os.path.join(self.dataset_root, "cifs")
            self._dataset = CIFData(cifs_folder)
        
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
        print(f"Successfully loaded {os.path.basename(model_path)}")
        
        return model, checkpoint
    
    def _get_bandgap_model(self):
        """Get cached bandgap model"""
        if self._bandgap_model is None:
            self._bandgap_model, _ = self._load_cgcnn_model(self.bandgap_model_path)
        return self._bandgap_model
    
    def _get_bulk_model(self):
        """Get cached bulk modulus model"""
        if self._bulk_model is None:
            self._bulk_model, _ = self._load_cgcnn_model(self.bulk_model_path)
        return self._bulk_model
    
    def _get_finetuned_model(self):
        """Get cached finetuned model with normalizer"""
        if self._finetuned_model is None:
            self._finetuned_model, checkpoint = self._load_cgcnn_model(self.finetuned_model_path)
            
            # Load normalizer if available
            if 'normalizer' in checkpoint:
                self._finetuned_normalizer = Normalizer(torch.tensor([0.0]))
                self._finetuned_normalizer.load_state_dict(checkpoint['normalizer'])
            
        return self._finetuned_model, self._finetuned_normalizer
    
    def _get_dataset_info(self):
        """Get cached dataset information"""
        if self._dataset is None or self._cif_filenames is None:
            cifs_folder = os.path.join(self.dataset_root, "cifs")
            self._dataset = CIFData(cifs_folder)
            
            # Read CIF filenames
            id_prop_path = os.path.join(cifs_folder, "id_prop.csv")
            id_prop_df = pd.read_csv(id_prop_path)
            cif_ids = id_prop_df.iloc[:, 0].tolist()
            self._cif_filenames = [cid + ".cif" for cid in cif_ids]
        
        return self._dataset, self._cif_filenames
    
    def _predict_cgcnn_property(self, model, cif_file_path: str, normalizer=None):
        """Generic CGCNN prediction function"""
        dataset, cif_filenames = self._get_dataset_info()
        
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
        """Run all predictions on a single CIF file with cached models"""
        
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
        
        # SEI Prediction
        try:
            sei_predictor = self._get_sei_predictor()
            sei_results = sei_predictor.predict_from_cif(cif_file_path)
            if sei_results is not None and 'sei_score' in sei_results:
                results["sei_score"] = float(sei_results['sei_score'])
                results["prediction_status"]["sei"] = "success"
                if verbose:
                    print(f"  SEI Score: {results['sei_score']:.3f}")
        except Exception as e:
            if verbose:
                print(f"  SEI prediction failed: {e}")
        
        # CEI Prediction
        try:
            cei_predictor = self._get_cei_predictor()
            cei_results = cei_predictor.predict_from_cif(cif_file_path)
            if cei_results is not None and 'cei_score' in cei_results:
                results["cei_score"] = float(cei_results['cei_score'])
                results["prediction_status"]["cei"] = "success"
                if verbose:
                    print(f"  CEI Score: {results['cei_score']:.3f}")
        except Exception as e:
            if verbose:
                print(f"  CEI prediction failed: {e}")
        
        # Bandgap Prediction
        try:
            bandgap_model = self._get_bandgap_model()
            bandgap_pred = self._predict_cgcnn_property(bandgap_model, cif_file_path)
            results["bandgap"] = float(bandgap_pred)
            results["prediction_status"]["bandgap"] = "success"
            if verbose:
                print(f"  Bandgap: {results['bandgap']:.3f} eV")
        except Exception as e:
            if verbose:
                print(f"  Bandgap prediction failed: {e}")
        
        # Bulk Modulus Prediction
        try:
            bulk_model = self._get_bulk_model()
            bulk_pred = self._predict_cgcnn_property(bulk_model, cif_file_path)
            results["bulk_modulus"] = float(bulk_pred)
            results["prediction_status"]["bulk_modulus"] = "success"
            if verbose:
                print(f"  Bulk Modulus: {results['bulk_modulus']:.1f} GPa")
        except Exception as e:
            if verbose:
                print(f"  Bulk modulus prediction failed: {e}")
        
        # Ionic Conductivity Prediction
        try:
            finetuned_model, normalizer = self._get_finetuned_model()
            ic_pred = self._predict_cgcnn_property(finetuned_model, cif_file_path, normalizer)
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


# Global instance for caching
_global_predictor = None

def get_optimized_predictor():
    """Get the global optimized predictor instance"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = OptimizedMLPredictor()
    return _global_predictor

def predict_single_cif_optimized(cif_file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Optimized prediction function that uses cached models"""
    predictor = get_optimized_predictor()
    return predictor.predict_single_cif(cif_file_path, verbose)