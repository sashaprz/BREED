import os
import sys
import torch
import pandas as pd
import numpy as np
import json
import argparse
from typing import List, Dict, Any


from env.sei_predictor import SEIPredictor
from env.cei_predictor import CEIPredictor
from env.property_predictions.cgcnn_pretrained import cgcnn_predict


from env.property_predictions.cgcnn_pretrained.cgcnn.model import CrystalGraphConvNet
from env.property_predictions.cgcnn_pretrained.cgcnn.data import CIFData, collate_pool
from env.property_predictions.main import Normalizer


def run_sei_prediction(cif_file_path: str):
    predictor = SEIPredictor()
    results = predictor.predict_from_cif(cif_file_path)
    return results


def run_cei_prediction(cif_file_path: str):
    predictor = CEIPredictor()
    results = predictor.predict_from_cif(cif_file_path)
    return results


def run_cgcnn_prediction(model_checkpoint: str, cif_file_path: str):
    """Run CGCNN prediction on a single CIF file"""
    try:
        results = cgcnn_predict.main([model_checkpoint, cif_file_path])
        return results
    except Exception as e:
        print(f"Error running CGCNN prediction: {e}")
        return None


def run_finetuned_cgcnn_prediction(checkpoint_path: str, dataset_root: str, cif_file_path: str):
    """
    dataset_root: path to CIF_OBELiX folder
    CIF files and id_prop.csv are in dataset_root/cifs/
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Point to the cifs subfolder inside dataset_root
    cifs_folder = os.path.join(dataset_root, "cifs")

    # Read id_prop.csv inside the cifs folder to get the list of CIF ids
    id_prop_path = os.path.join(cifs_folder, "id_prop.csv")
    id_prop_df = pd.read_csv(id_prop_path)
    # Assuming first column of id_prop.csv has CIF ids without ".cif"
    cif_ids = id_prop_df.iloc[:, 0].tolist()
    # Append ".cif" to create CIF filenames
    cif_filenames = [cid + ".cif" for cid in cif_ids]

    cif_basename = os.path.basename(cif_file_path)
    sample_index = None
    for idx, fname in enumerate(cif_filenames):
        if fname == cif_basename:
            sample_index = idx
            break

    if sample_index is None:
        raise ValueError(f"CIF file {cif_file_path} not found in dataset folder {cifs_folder}")

    # Load dataset with CIFData pointing at cifs folder (where CIF files live)
    dataset = CIFData(cifs_folder)

    # Prepare single sample batch
    sample = [dataset[sample_index]]
    input_data, targets, cif_ids_result = collate_pool(sample)

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
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load Normalizer state for denormalization, if available
    normalizer = None
    if 'normalizer' in checkpoint:
        normalizer = Normalizer(torch.tensor([0.0]))
        normalizer.load_state_dict(checkpoint['normalizer'])

    input_vars = (
        input_data[0].to(device),
        input_data[1].to(device),
        input_data[2].to(device),
        input_data[3],  # crystal_atom_idx (list of tensors, stays on CPU)
    )

    with torch.no_grad():
        output = model(*input_vars)
        pred = output.cpu().numpy().flatten()[0]

    # Denormalize prediction if normalizer is available
    if normalizer is not None:
        pred_tensor = torch.tensor([pred])
        pred_denorm = normalizer.denorm(pred_tensor).item()
    else:
        pred_denorm = pred

    # Uncomment below if you used log or log10 transform on target during training
    # pred_final = np.exp(pred_denorm)        # For natural log
    # pred_final = 10 ** pred_denorm           # For log10
    # Otherwise, use pred_denorm directly
    pred_final = pred_denorm

    results = {
        'cif_ids': cif_ids_result,
        'predictions': [pred_final],
        'mae': checkpoint.get('best_mae_error', None),
    }
    return results


def extract_composition_from_cif(cif_file_path: str) -> str:
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


def predict_single_cif(cif_file_path: str, verbose:bool = True) -> Dict[str, Any]:
    """Run all predictions on a single CIF file and return consolidated results"""
    
    # Configuration paths - adjust these to your actual paths
    dataset_root = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\CIF_OBELiX"
    bandgap_model = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\cgcnn_pretrained\band-gap.pth.tar"
    bulk_model = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\cgcnn_pretrained\bulk-moduli.pth.tar"
    finetuned_model = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\checkpoint.pth.tar"
    
    results = {
        "composition": extract_composition_from_cif(cif_file_path),
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
    
    # Run SEI Prediction (with appropriate printing control)
    try:
        sei_results = run_sei_prediction(cif_file_path)
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
    
    # Similarly for CEI, Bandgap, Bulk Modulus, Ionic Conductivity:
    # Wrap all prints related to them inside `if verbose:` blocks
    
    # CEI Prediction
    try:
        cei_results = run_cei_prediction(cif_file_path)
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
    
    # Bandgap Prediction
    try:
        bandgap_results = run_cgcnn_prediction(bandgap_model, cif_file_path)
        if bandgap_results is not None and 'predictions' in bandgap_results and len(bandgap_results['predictions']) > 0:
            results["bandgap"] = float(bandgap_results['predictions'][0])
            results["prediction_status"]["bandgap"] = "success"
            if verbose:
                print(f"  Bandgap: {results['bandgap']:.3f} eV")
        else:
            if verbose:
                print("  Bandgap prediction failed or no predictions returned")
    except Exception as e:
        if verbose:
            print(f"  Bandgap prediction failed: {e}")
    
    # Bulk Modulus Prediction
    try:
        bulk_results = run_cgcnn_prediction(bulk_model, cif_file_path)
        if bulk_results is not None and 'predictions' in bulk_results and len(bulk_results['predictions']) > 0:
            results["bulk_modulus"] = float(bulk_results['predictions'][0])
            results["prediction_status"]["bulk_modulus"] = "success"
            if verbose:
                print(f"  Bulk Modulus: {results['bulk_modulus']:.1f} GPa")
        else:
            if verbose:
                print("  Bulk modulus prediction failed or no predictions returned")
    except Exception as e:
        if verbose:
            print(f"  Bulk modulus prediction failed: {e}")
    
    # Ionic Conductivity Prediction (Fine-tuned model)
    try:
        finetuned_results = run_finetuned_cgcnn_prediction(finetuned_model, dataset_root, cif_file_path)
        if finetuned_results is not None and 'predictions' in finetuned_results and len(finetuned_results['predictions']) > 0:
            results["ionic_conductivity"] = float(finetuned_results['predictions'][0])
            results["prediction_status"]["ionic_conductivity"] = "success"
            if verbose:
                print(f"  Ionic Conductivity: {results['ionic_conductivity']:.2e} S/cm")
        else:
            if verbose:
                print("  Ionic conductivity prediction failed or no predictions returned")
    except Exception as e:
        if verbose:
            print(f"  Ionic conductivity prediction failed: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run ML property predictions on CIF files')
    parser.add_argument('--cif_files', nargs='+', required=True, 
                       help='List of CIF files to process')
    parser.add_argument('--output', required=True,
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    print(f"Running ML predictions on {len(args.cif_files)} CIF files...")
    print(f"Output will be saved to: {args.output}")
    
    # Process all CIF files
    all_results = {}
    
    for i, cif_file in enumerate(args.cif_files, 1):
        print(f"\n[{i}/{len(args.cif_files)}] Processing: {cif_file}")
        
        if not os.path.isfile(cif_file):
            print(f"  Warning: CIF file not found: {cif_file}")
            # Create default failed result
            all_results[os.path.basename(cif_file)] = {
                "composition": "unknown",
                "bandgap": 0.0,
                "sei_score": 0.0,
                "cei_score": 0.0,
                "ionic_conductivity": 0.0,
                "bulk_modulus": 0.0,
                "prediction_status": {
                    "sei": "file_not_found",
                    "cei": "file_not_found",
                    "bandgap": "file_not_found",
                    "bulk_modulus": "file_not_found",
                    "ionic_conductivity": "file_not_found"
                }
            }
            continue
        
        try:
            result = predict_single_cif(cif_file)
            # Use basename as key to match GA expectations
            all_results[os.path.basename(cif_file)] = result
        except Exception as e:
            print(f"  Error processing {cif_file}: {e}")
            # Create default failed result
            all_results[os.path.basename(cif_file)] = {
                "composition": extract_composition_from_cif(cif_file),
                "bandgap": 0.0,
                "sei_score": 0.0,
                "cei_score": 0.0,
                "ionic_conductivity": 0.0,
                "bulk_modulus": 0.0,
                "prediction_status": {
                    "sei": "error",
                    "cei": "error",
                    "bandgap": "error",
                    "bulk_modulus": "error",
                    "ionic_conductivity": "error"
                }
            }
    
    # Save results to JSON file
    try:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
        
        # Print summary
        successful_predictions = 0
        total_properties = 0
        for result in all_results.values():
            if 'prediction_status' in result:
                for prop, status in result['prediction_status'].items():
                    total_properties += 1
                    if status == "success":
                        successful_predictions += 1
        
        print(f"Summary: {successful_predictions}/{total_properties} property predictions successful")
        print(f"Success rate: {100*successful_predictions/total_properties if total_properties > 0 else 0:.1f}%")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if being called with command line arguments (from GA)
    if len(sys.argv) > 1 and '--cif_files' in sys.argv:
        main()
    else:
        # Original standalone mode for testing
        print("Running in standalone test mode...")
        cif_path = r"C:\Users\Sasha\repos\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_bulk_moduli\CIF_OBELiX\cifs\test_CIF.cif"
        
        if not os.path.isfile(cif_path):
            print(f"Error: CIF file not found at {cif_path}")
            sys.exit(1)
        
        result = predict_single_cif(cif_path)
        
        print("\n=== Final Prediction Summary ===\n")
        print(f"Composition: {result['composition']}")
        print(f"SEI Score: {result['sei_score']:.3f}")
        print(f"CEI Score: {result['cei_score']:.3f}")
        print(f"Bandgap: {result['bandgap']:.3f} eV")
        print(f"Bulk Modulus: {result['bulk_modulus']:.1f} GPa")
        print(f"Ionic Conductivity: {result['ionic_conductivity']:.2e} S/cm")
        print(f"\nPrediction Status: {result['prediction_status']}")