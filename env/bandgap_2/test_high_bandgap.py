#!/usr/bin/env python3
"""
Test script to demonstrate improved bandgap model performance on high bandgap materials
"""

import pickle
import pandas as pd
import numpy as np
from pymatgen.core import Composition

def load_model():
    """Load the improved bandgap correction model"""
    try:
        with open('improved_bandgap_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print("✅ Successfully loaded improved bandgap model")
        print(f"📊 Model info: {model_data['training_samples']} training samples, {len(model_data['feature_names'])} features")
        print(f"🎯 High bandgap performance: {model_data.get('high_bg_performance', 'N/A')}")
        return model_data, model_data['feature_names']
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return None, None

def get_composition_features(formula):
    """Extract composition features from chemical formula"""
    try:
        comp = Composition(formula)
        features = {}
        
        # Basic composition features
        features['num_elements'] = len(comp.elements)
        features['total_atoms'] = sum(comp.values())
        
        # Electronegativity features
        electronegativities = [elem.X for elem in comp.elements if elem.X is not None]
        if electronegativities:
            features['avg_electronegativity'] = np.mean(electronegativities)
            features['max_electronegativity'] = max(electronegativities)
            features['min_electronegativity'] = min(electronegativities)
            features['electronegativity_range'] = max(electronegativities) - min(electronegativities)
        else:
            features['avg_electronegativity'] = 0
            features['max_electronegativity'] = 0
            features['min_electronegativity'] = 0
            features['electronegativity_range'] = 0
        
        # Atomic mass features
        atomic_masses = [elem.atomic_mass * comp[elem] for elem in comp.elements]
        features['avg_atomic_mass'] = np.mean(atomic_masses)
        features['total_mass'] = sum(atomic_masses)
        
        # Element presence indicators (common wide bandgap elements)
        wide_bandgap_elements = ['O', 'N', 'F', 'Cl', 'S', 'Se', 'Te']
        for elem_symbol in wide_bandgap_elements:
            features[f'has_{elem_symbol}'] = 1 if elem_symbol in [str(e) for e in comp.elements] else 0
        
        # Oxide and nitride indicators
        features['is_oxide'] = 1 if 'O' in [str(e) for e in comp.elements] else 0
        features['is_nitride'] = 1 if 'N' in [str(e) for e in comp.elements] else 0
        
        return features
    except Exception as e:
        print(f"Error processing composition {formula}: {e}")
        return None

def predict_bandgap(model_data, feature_names, formula, pbe_bandgap):
    """Predict HSE bandgap from PBE bandgap and formula using ensemble model"""
    # Get composition features
    comp_features = get_composition_features(formula)
    if comp_features is None:
        return None
    
    # Create feature vector
    features = {
        'pbe_bandgap': pbe_bandgap,
        'pbe_squared': pbe_bandgap**2,
        'pbe_cubed': pbe_bandgap**3,
        'log_pbe': np.log1p(pbe_bandgap),  # log(1 + pbe) to handle zeros
        **comp_features
    }
    
    # Ensure all required features are present
    feature_vector = []
    for feature_name in feature_names:
        if feature_name in features:
            feature_vector.append(features[feature_name])
        else:
            feature_vector.append(0)  # Default value for missing features
    
    # Scale features
    X_scaled = model_data['scaler'].transform([feature_vector])
    
    # Make predictions with both models
    rf_pred = model_data['rf_model'].predict(X_scaled)[0]
    gb_pred = model_data['gb_model'].predict(X_scaled)[0]
    
    # Ensemble prediction using weights
    weights = model_data['ensemble_weights']
    prediction = weights[0] * rf_pred + weights[1] * gb_pred
    
    return prediction

def test_high_bandgap_materials():
    """Test the model on various high bandgap materials"""
    
    # Load the model
    model_data, feature_names = load_model()
    if model_data is None:
        return
    
    print(f"Model features: {len(feature_names)} features")
    print("Feature names:", feature_names[:10], "..." if len(feature_names) > 10 else "")
    print()
    
    # Test materials with known high bandgaps
    test_materials = [
        # Wide bandgap semiconductors and insulators
        {"formula": "BN", "pbe_bandgap": 4.5, "expected_hse": "~6.0", "material_type": "Boron Nitride"},
        {"formula": "AlN", "pbe_bandgap": 4.2, "expected_hse": "~6.2", "material_type": "Aluminum Nitride"},
        {"formula": "GaN", "pbe_bandgap": 2.2, "expected_hse": "~3.4", "material_type": "Gallium Nitride"},
        {"formula": "ZnO", "pbe_bandgap": 0.8, "expected_hse": "~3.4", "material_type": "Zinc Oxide"},
        {"formula": "TiO2", "pbe_bandgap": 2.0, "expected_hse": "~3.2", "material_type": "Titanium Dioxide"},
        {"formula": "SiC", "pbe_bandgap": 1.4, "expected_hse": "~2.4", "material_type": "Silicon Carbide"},
        {"formula": "Al2O3", "pbe_bandgap": 6.2, "expected_hse": "~8.8", "material_type": "Aluminum Oxide"},
        {"formula": "MgO", "pbe_bandgap": 4.9, "expected_hse": "~7.8", "material_type": "Magnesium Oxide"},
        {"formula": "CaF2", "pbe_bandgap": 8.1, "expected_hse": "~12.1", "material_type": "Calcium Fluoride"},
        {"formula": "LiF", "pbe_bandgap": 9.0, "expected_hse": "~14.2", "material_type": "Lithium Fluoride"},
    ]
    
    print("🔬 Testing High Bandgap Materials")
    print("=" * 80)
    print(f"{'Material':<20} {'Formula':<10} {'PBE (eV)':<10} {'Predicted HSE':<15} {'Expected HSE':<15} {'Correction':<12}")
    print("-" * 80)
    
    for material in test_materials:
        formula = material["formula"]
        pbe_bg = material["pbe_bandgap"]
        expected = material["expected_hse"]
        mat_type = material["material_type"]
        
        # Make prediction
        predicted_hse = predict_bandgap(model_data, feature_names, formula, pbe_bg)
        
        if predicted_hse is not None:
            correction = predicted_hse - pbe_bg
            print(f"{mat_type:<20} {formula:<10} {pbe_bg:<10.1f} {predicted_hse:<15.2f} {expected:<15} {correction:<12.2f}")
        else:
            print(f"{mat_type:<20} {formula:<10} {pbe_bg:<10.1f} {'ERROR':<15} {expected:<15} {'N/A':<12}")
    
    print("-" * 80)
    print()
    
    # Test some edge cases
    print("🧪 Testing Edge Cases")
    print("=" * 50)
    
    edge_cases = [
        {"formula": "Diamond", "pbe_bandgap": 4.1, "note": "Carbon diamond structure"},
        {"formula": "Si", "pbe_bandgap": 0.6, "note": "Silicon (narrow gap)"},
        {"formula": "Ge", "pbe_bandgap": 0.0, "note": "Germanium (zero gap in PBE)"},
    ]
    
    for case in edge_cases:
        try:
            # For diamond, use C formula
            formula = "C" if case["formula"] == "Diamond" else case["formula"]
            predicted = predict_bandgap(model_data, feature_names, formula, case["pbe_bandgap"])
            if predicted is not None:
                print(f"{case['formula']:<15} PBE: {case['pbe_bandgap']:.1f} eV → HSE: {predicted:.2f} eV ({case['note']})")
            else:
                print(f"{case['formula']:<15} Error in prediction ({case['note']})")
        except Exception as e:
            print(f"{case['formula']:<15} Error: {e}")
    
    print()
    print("📊 Model Performance Notes:")
    print("- The improved model uses weighted training for high bandgap materials (>3 eV)")
    print("- Enhanced features include pbe_cubed, log_pbe, and oxide/nitride indicators")
    print("- Sample weights: 3x for >3 eV, 2x for 2.5-3 eV materials")
    print("- Expected improvements in wide bandgap semiconductor predictions")

if __name__ == "__main__":
    test_high_bandgap_materials()