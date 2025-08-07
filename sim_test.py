import os
import sys
import torch

from env.sei_predictor import SEIPredictor
from env.cei_predictor import CEIPredictor
from env.cgcnn_bandgap_ionic_cond_shear_moduli.cgcnn_pretrained import cgcnn_predict

from env.cgcnn_bandgap_ionic_cond_shear_moduli.cgcnn_pretrained.cgcnn.model import CrystalGraphConvNet
from env.cgcnn_bandgap_ionic_cond_shear_moduli.cgcnn_pretrained.cgcnn.data import CIFData, collate_pool
from env.cgcnn_bandgap_ionic_cond_shear_moduli.main import Normalizer

print("Running main_rl.py:", __file__)


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


def run_finetuned_cgcnn_prediction(checkpoint_path: str, cif_file_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = CIFData(cif_file_path)  # Load CIF and extract features

    sample = [dataset[0]]  # Single sample batch
    input_data, targets, cif_ids = collate_pool(sample)

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

    input_vars = (
        input_data[0].to(device),
        input_data[1].to(device),
        input_data[2].to(device),
        input_data[3],  # crystal_atom_idx (list of tensors, stays on CPU)
    )

    with torch.no_grad():
        output = model(*input_vars)
        pred = output.cpu().numpy().flatten()[0]

    results = {
        'cif_ids': cif_ids,
        'predictions': [pred],
        'mae': checkpoint.get('best_mae_error', None),
    }
    return results


def format_cgcnn_results(results, model_name):
    if results is None:
        return f"{model_name} prediction failed"

    if not results.get('predictions'):
        return f"No predictions available for {model_name}"

    prediction = results['predictions'][0]
    cif_id = results['cif_ids'][0]

    output = f"CIF ID: {cif_id}\n"
    output += f"Prediction: {prediction:.4f}\n"

    if 'mae' in results and results['mae'] is not None:
        output += f"Model MAE: {results['mae']:.4f}\n"
    if 'auc' in results:
        output += f"Model AUC: {results['auc']:.4f}\n"

    return output


if __name__ == "__main__":
    # Use test_CIF.cif in the same dir as the script by default
    cif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_CIF.cif")

    if not os.path.isfile(cif_path):
        print(f"Error: CIF file not found at {cif_path}")
        sys.exit(1)

    # SEI Prediction
    print("=== SEI Prediction ===")
    try:
        sei_results = run_sei_prediction(cif_path)
        print("SEI Score:", sei_results.get('sei_score', 'N/A'))
        if 'overall_properties' in sei_results:
            print("Overall SEI Properties:")
            for prop, val in sei_results['overall_properties'].items():
                print(f"  {prop}: {val:.3f}")
    except Exception as e:
        print(f"SEI prediction failed: {e}")
    print()

    # CEI Prediction
    print("=== CEI Prediction ===")
    try:
        cei_results = run_cei_prediction(cif_path)
        print("CEI Score:", cei_results.get('cei_score', 'N/A'))
        if 'overall_properties' in cei_results:
            print("Overall CEI Properties:")
            for prop, val in cei_results['overall_properties'].items():
                print(f"  {prop}: {val:.3f}")
    except Exception as e:
        print(f"CEI prediction failed: {e}")
    print()

    # CGCNN Bandgap Prediction (pretrained)
    print("=== CGCNN Bandgap Prediction ===")
    try:
        bandgap_results = run_cgcnn_prediction(
            r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_pretrained\band-gap.pth.tar",
            cif_path
        )
        print(format_cgcnn_results(bandgap_results, "Bandgap"))
    except Exception as e:
        print(f"CGCNN Bandgap prediction failed: {e}")
    print()

    # CGCNN Shear Moduli Prediction (pretrained)
    print("=== CGCNN Shear Moduli Prediction ===")
    try:
        shear_results = run_cgcnn_prediction(
            r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_pretrained\shear-moduli.pth.tar",
            cif_path
        )
        print(format_cgcnn_results(shear_results, "Shear Moduli"))
    except Exception as e:
        print(f"CGCNN Shear Moduli prediction failed: {e}")
    print()

    # CGCNN Fine-tuned Model Prediction
    print("=== CGCNN Fine-tuned Model Prediction ===")
    try:
        finetuned_checkpoint_path = r"C:\Users\Sasha\OneDrive\vscode\fr8\RL-electrolyte-design\env\cgcnn_bandgap_ionic_cond_shear_moduli\model_best.pth.tar"
        finetuned_results = run_finetuned_cgcnn_prediction(finetuned_checkpoint_path, cif_path)
        print(format_cgcnn_results(finetuned_results, "Fine-tuned CGCNN"))
    except Exception as e:
        print(f"Fine-tuned CGCNN prediction failed: {e}")
    print()
