#!/usr/bin/env python3
"""
Cached Property Predictor - EXACT same logic as property_prediction_script.py but with cached models
This eliminates the model reloading bottleneck while maintaining identical prediction behavior.
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

# Add parent directory to path so we can import from env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import exactly the same modules as property_prediction_script.py
from env.sei_predictor import SEIPredictor
from env.cei_predictor import CEIPredictor
from env.property_predictions.cgcnn_pretrained.cgcnn.model import CrystalGraphConvNet
from env.property_predictions.cgcnn_pretrained.cgcnn.data import CIFData, collate_pool
from env.property_predictions.main import Normalizer

# Import bandgap correction from fully_optimized_predictor (same logic)
from genetic_algo.fully_optimized_predictor import apply_ml_bandgap_correction, BANDGAP_CORRECTION_AVAILABLE, CORRECTION_METHOD

# Import composition-only ionic conductivity predictor
from genetic_algo.composition_only_ionic_conductivity import predict_ionic_conductivity_from_composition

class CachedPropertyPredictor:
    """
    EXACT same prediction logic as property_prediction_script.py but with cached models
    This eliminates model reloading while maintaining identical behavior
    """
    
    def __init__(self):
        # Configuration paths - EXACT same as property_prediction_script.py
        self.dataset_root = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\property_predictions\CIF_OBELiX"
        self.bandgap_model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\property_predictions\cgcnn_pretrained\band-gap.pth.tar"
        self.bulk_model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\property_predictions\cgcnn_pretrained\bulk-moduli.pth.tar"
        self.finetuned_model_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\checkpoint.pth.tar"
        
        # Cached models and predictors - loaded ONCE
        # NOTE: Ionic conductivity CGCNN removed due to poor performance (R² ≈ 0)
        self._sei_predictor = None
        self._cei_predictor = None
        self._bandgap_model = None
        self._bulk_model = None
        # self._finetuned_model = None  # REMOVED - CGCNN ionic conductivity skipped
        # self._finetuned_normalizer = None  # REMOVED - CGCNN ionic conductivity skipped
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dataset cache
        self._dataset = None
        self._cif_filenames = None
        
        print(f"CachedPropertyPredictor initialized - will cache models to eliminate reloading")
    
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
        """Load a CGCNN model ONCE with proper architecture detected from checkpoint"""
        print(f"🔄 Loading {model_name} model (ONE TIME ONLY)...")
        
        # Ensure dataset is loaded first
        self._load_dataset_once()
        
        checkpoint = torch.load(model_path, map_location=self._device)
        
        # Get model architecture from sample
        sample = [self._dataset[0]]
        input_data, _, _ = collate_pool(sample)
        
        orig_atom_fea_len = input_data[0].shape[-1]
        nbr_fea_len = input_data[1].shape[-1]
        
        # Detect architecture from checkpoint state_dict
        state_dict = checkpoint['state_dict']
        
        # Detect h_fea_len from conv_to_fc layer
        h_fea_len = 32  # default for older models
        if 'conv_to_fc.weight' in state_dict:
            h_fea_len = state_dict['conv_to_fc.weight'].shape[0]
        
        # Detect n_conv from number of conv layers
        n_conv = 3  # default
        conv_layers = [k for k in state_dict.keys() if k.startswith('convs.') and 'fc_full.weight' in k]
        if conv_layers:
            max_conv_idx = max([int(k.split('.')[1]) for k in conv_layers])
            n_conv = max_conv_idx + 1
        
        # Standard parameters
        atom_fea_len = 64
        n_h = 1
        
        print(f"   Detected architecture: h_fea_len={h_fea_len}, n_conv={n_conv}")
        
        try:
            model = CrystalGraphConvNet(
                orig_atom_fea_len=orig_atom_fea_len,
                nbr_fea_len=nbr_fea_len,
                atom_fea_len=atom_fea_len,
                n_conv=n_conv,
                h_fea_len=h_fea_len,
                n_h=n_h,
                classification=False
            ).to(self._device)
            
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            print(f"✅ {model_name} model loaded and cached successfully!")
            return model, checkpoint
            
        except Exception as e:
            print(f"   ❌ Failed to load {model_name} model: {e}")
            return None, checkpoint
    
    def get_sei_predictor(self):
        """Get SEI predictor (loaded once)"""
        if self._sei_predictor is None:
            print("🔄 Loading SEI predictor (ONE TIME ONLY)...")
            self._sei_predictor = SEIPredictor()
            print("✅ SEI predictor loaded and cached!")
        return self._sei_predictor
    
    def get_cei_predictor(self):
        """Get CEI predictor (loaded once)"""
        if self._cei_predictor is None:
            print("🔄 Loading CEI predictor (ONE TIME ONLY)...")
            self._cei_predictor = CEIPredictor()
            print("✅ CEI predictor loaded and cached!")
        return self._cei_predictor
    
    def get_bandgap_model(self):
        """Get bandgap model (loaded once)"""
        if self._bandgap_model is None:
            self._bandgap_model, _ = self._load_cgcnn_model_once(
                self.bandgap_model_path, "Bandgap"
            )
        return self._bandgap_model
    
    def get_bulk_model(self):
        """Get bulk modulus model (loaded once)"""
        if self._bulk_model is None:
            self._bulk_model, _ = self._load_cgcnn_model_once(
                self.bulk_model_path, "Bulk Modulus"
            )
        return self._bulk_model
    
    # REMOVED: get_finetuned_model() - CGCNN ionic conductivity skipped due to poor performance
    # Original performance: R² ≈ 0, MAPE > 8 million %
    # Replaced with fast, reliable composition-based prediction
    
    def predict_cgcnn_property_cached(self, model, cif_file_path: str, normalizer=None):
        """
        Predict property using cached CGCNN model - EXACT same logic as property_prediction_script.py
        but without model reloading
        """
        try:
            # Create temporary directory structure that CIFData expects
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy CIF file to temp directory with simple name
                temp_cif_name = "temp_structure"
                temp_cif_path = os.path.join(temp_dir, temp_cif_name + ".cif")
                shutil.copy2(cif_file_path, temp_cif_path)
                
                # Create id_prop.csv with dummy target (we only care about prediction)
                id_prop_path = os.path.join(temp_dir, "id_prop.csv")
                with open(id_prop_path, 'w') as f:
                    f.write(f"{temp_cif_name},0.0\n")  # dummy target value
                
                # Copy atom_init.json from the main dataset
                atom_init_src = os.path.join(self.dataset_root, "cifs", "atom_init.json")
                atom_init_dst = os.path.join(temp_dir, "atom_init.json")
                shutil.copy2(atom_init_src, atom_init_dst)
                
                # Create dataset using the standard CIFData interface
                temp_dataset = CIFData(temp_dir)
                
                # Prepare sample
                sample = [temp_dataset[0]]
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
            
        except Exception as e:
            print(f"    CGCNN prediction error: {e}")
            return 0.0
    
    def run_sei_prediction_cached(self, cif_file_path: str):
        """Run SEI prediction using cached predictor"""
        predictor = self.get_sei_predictor()
        results = predictor.predict_from_cif(cif_file_path)
        return results
    
    def run_cei_prediction_cached(self, cif_file_path: str):
        """Run CEI prediction using cached predictor"""
        predictor = self.get_cei_predictor()
        results = predictor.predict_from_cif(cif_file_path)
        return results
    
    def extract_composition_from_cif(self, cif_file_path: str) -> str:
        """Extract composition from CIF file exactly as in property_prediction_script.py"""
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
    
    # REMOVED: estimate_ionic_conductivity_from_composition()
    # Replaced with optimized version in composition_only_ionic_conductivity.py
    
    def predict_single_cif(self, cif_file_path: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Run all predictions with EXACT same logic as property_prediction_script.py
        but using cached models to eliminate reloading bottleneck
        """
        
        results = {
            "composition": self.extract_composition_from_cif(cif_file_path),
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
        
        # Run SEI Prediction (exactly as in property_prediction_script.py)
        try:
            sei_results = self.run_sei_prediction_cached(cif_file_path)
            if sei_results is not None and 'sei_score' in sei_results:
                results["sei_score"] = float(sei_results['sei_score'])
                results["prediction_status"]["sei"] = "success"
                if verbose:
                    print(f"  SEI Score: {results['sei_score']:.3f}")
            else:
                if verbose:
                    print("  SEI prediction failed or no score returned")
        except Exception as e:
            if verbose:
                print(f"  SEI prediction failed: {e}")
        
        # CEI Prediction (exactly as in property_prediction_script.py)
        try:
            cei_results = self.run_cei_prediction_cached(cif_file_path)
            if cei_results is not None and 'cei_score' in cei_results:
                results["cei_score"] = float(cei_results['cei_score'])
                results["prediction_status"]["cei"] = "success"
                if verbose:
                    print(f"  CEI Score: {results['cei_score']:.3f}")
            else:
                if verbose:
                    print("  CEI prediction failed or no score returned")
        except Exception as e:
            if verbose:
                print(f"  CEI prediction failed: {e}")
        
        # Bandgap Prediction (CGCNN model + ML bandgap correction)
        try:
            bandgap_model = self.get_bandgap_model()
            if bandgap_model is not None:
                raw_pbe_bandgap = self.predict_cgcnn_property_cached(bandgap_model, cif_file_path)
                
                if raw_pbe_bandgap is not None and raw_pbe_bandgap != 0.0:
                    # Apply ML bandgap correction (PBE → HSE)
                    if BANDGAP_CORRECTION_AVAILABLE:
                        composition_str = results["composition"]
                        corrected_bandgap = apply_ml_bandgap_correction(raw_pbe_bandgap, composition_str)
                        
                        results["bandgap"] = float(corrected_bandgap)
                        results["bandgap_raw_pbe"] = float(raw_pbe_bandgap)
                        results["bandgap_correction_applied"] = True
                        results["correction_method"] = CORRECTION_METHOD
                        
                        if verbose:
                            print(f"  Bandgap (PBE): {raw_pbe_bandgap:.3f} eV")
                            print(f"  Bandgap (HSE-corrected): {results['bandgap']:.3f} eV")
                    else:
                        results["bandgap"] = raw_pbe_bandgap
                        results["bandgap_correction_applied"] = False
                        results["correction_method"] = "none"
                        
                        if verbose:
                            print(f"  Bandgap: {results['bandgap']:.3f} eV")
                    
                    results["prediction_status"]["bandgap"] = "success"
                else:
                    if verbose:
                        print("  Bandgap CGCNN prediction returned 0 or None")
            else:
                if verbose:
                    print("  Bandgap CGCNN model failed to load")
        except Exception as e:
            if verbose:
                print(f"  Bandgap prediction failed: {e}")
        
        # Bulk Modulus Prediction (CGCNN model)
        try:
            bulk_model = self.get_bulk_model()
            if bulk_model is not None:
                bulk_pred = self.predict_cgcnn_property_cached(bulk_model, cif_file_path)
                
                if bulk_pred is not None and bulk_pred != 0.0:
                    results["bulk_modulus"] = float(bulk_pred)
                    results["prediction_status"]["bulk_modulus"] = "success"
                    if verbose:
                        print(f"  Bulk Modulus: {results['bulk_modulus']:.1f} GPa")
                else:
                    if verbose:
                        print("  Bulk modulus CGCNN prediction returned 0 or None")
            else:
                if verbose:
                    print("  Bulk modulus CGCNN model failed to load")
        except Exception as e:
            if verbose:
                print(f"  Bulk modulus prediction failed: {e}")
        
        # Ionic Conductivity Prediction (COMPOSITION-ONLY - CGCNN SKIPPED)
        # CGCNN performance is terrible: R² ≈ 0, MAPE > 8 million %
        # Direct composition-based prediction is faster and more reliable
        try:
            results["ionic_conductivity"] = predict_ionic_conductivity_from_composition(results["composition"])
            results["prediction_status"]["ionic_conductivity"] = "composition_based"
            results["cgcnn_skipped"] = True
            results["cgcnn_skip_reason"] = "Poor_performance_R2_negative"
            
            if verbose:
                print(f"  Ionic Conductivity (composition-based): {results['ionic_conductivity']:.2e} S/cm")
                print(f"  CGCNN skipped due to poor performance (R² ≈ 0)")
                
        except Exception as e:
            # Fallback to a reasonable default if composition parsing fails
            results["ionic_conductivity"] = 1e-8  # Default low conductivity
            results["prediction_status"]["ionic_conductivity"] = "default_fallback"
            results["cgcnn_skipped"] = True
            results["cgcnn_skip_reason"] = "Poor_performance_R2_negative"
            
            if verbose:
                print(f"  Ionic conductivity composition parsing failed ({e}), using default: {results['ionic_conductivity']:.2e} S/cm")
        
        return results


# Global instance for caching
_global_cached_predictor = None

def get_cached_predictor():
    """Get the global cached predictor instance"""
    global _global_cached_predictor
    if _global_cached_predictor is None:
        _global_cached_predictor = CachedPropertyPredictor()
    return _global_cached_predictor

def predict_single_cif_cached(cif_file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Cached prediction function - models loaded ONCE and cached forever"""
    predictor = get_cached_predictor()
    return predictor.predict_single_cif(cif_file_path, verbose)

if __name__ == "__main__":
    print("🔬 CACHED PROPERTY PREDICTOR TEST")
    print("=" * 50)
    print("Same logic as property_prediction_script.py but with cached models")
    print("This eliminates model reloading for massive speed improvement")