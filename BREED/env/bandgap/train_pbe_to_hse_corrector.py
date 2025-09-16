#!/usr/bin/env python3
"""
JARVIS HSE Correction Model Trainer
Adapts the existing improved_bandgap_model.py architecture to train on JARVIS HSE data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import sys
from datetime import datetime

def create_features(data):
    """Create features from the JARVIS data - adapted from improved_bandgap_model.py"""
    print("Creating features...")
    
    # Start with PBE bandgap as main feature
    features = pd.DataFrame({
        'pbe_bandgap': data['pbe_bandgap']
    })
    
    # Add composition-based features if available
    if 'formula' in data.columns:
        from pymatgen.core import Composition
        
        compositions = []
        for idx, formula in enumerate(data['formula']):
            try:
                if pd.notna(formula):
                    comp = Composition(formula)
                    compositions.append(comp)
                else:
                    compositions.append(None)
            except:
                compositions.append(None)
        
        # Extract composition features
        features['n_elements'] = [len(comp.elements) if comp else 2 for comp in compositions]
        features['total_atoms'] = [comp.num_atoms if comp else 2 for comp in compositions]
        
        # Electronegativity features
        electronegativities = []
        atomic_masses = []
        for comp in compositions:
            if comp:
                try:
                    en_values = [el.X for el in comp.elements if el.X and not np.isnan(el.X)]
                    mass_values = [el.atomic_mass for el in comp.elements]
                    
                    if en_values:
                        electronegativities.append(np.mean(en_values))
                        atomic_masses.append(np.mean(mass_values))
                    else:
                        electronegativities.append(2.0)  # Default
                        atomic_masses.append(50.0)  # Default
                except:
                    electronegativities.append(2.0)
                    atomic_masses.append(50.0)
            else:
                electronegativities.append(2.0)
                atomic_masses.append(50.0)
        
        features['avg_electronegativity'] = electronegativities
        features['avg_atomic_mass'] = atomic_masses
        
        # Element presence features
        features['has_O'] = data['formula'].str.contains('O', na=False).astype(int)
        features['has_N'] = data['formula'].str.contains('N', na=False).astype(int)
        features['has_C'] = data['formula'].str.contains('C', na=False).astype(int)
        features['has_Si'] = data['formula'].str.contains('Si', na=False).astype(int)
        features['has_Al'] = data['formula'].str.contains('Al', na=False).astype(int)
        features['has_Ti'] = data['formula'].str.contains('Ti', na=False).astype(int)
        features['has_Fe'] = data['formula'].str.contains('Fe', na=False).astype(int)
        features['has_F'] = data['formula'].str.contains('F', na=False).astype(int)  # NEW: Fluorides
        features['has_H'] = data['formula'].str.contains('H', na=False).astype(int)  # NEW: Hydrides
        
        # Material type features
        features['is_oxide'] = data['formula'].str.contains('O', na=False).astype(int)
        features['is_nitride'] = data['formula'].str.contains('N', na=False).astype(int)
        features['is_carbide'] = data['formula'].str.contains('C', na=False).astype(int)
        features['is_fluoride'] = data['formula'].str.contains('F', na=False).astype(int)  # NEW
        features['is_hydride'] = data['formula'].str.contains('H', na=False).astype(int)  # NEW
        
        # Derived features - enhanced for high bandgaps
        features['pbe_squared'] = features['pbe_bandgap'] ** 2
        features['pbe_cubed'] = features['pbe_bandgap'] ** 3
        features['pbe_sqrt'] = np.sqrt(np.abs(features['pbe_bandgap']))
        features['log_pbe'] = np.log1p(features['pbe_bandgap'])
        features['en_pbe_product'] = features['avg_electronegativity'] * features['pbe_bandgap']
        features['en_squared'] = features['avg_electronegativity'] ** 2
        
        # JARVIS-specific features
        if 'dimensionality' in data.columns:
            features['is_2d'] = (data['dimensionality'] == '2D').astype(int)
            features['is_3d'] = (data['dimensionality'] == '3D').astype(int)
    
    print(f"Created {features.shape[1]} features")
    return features

def create_sample_weights(y_train, high_bg_threshold=3.0, weight_multiplier=3.0):
    """Create sample weights to emphasize high bandgap materials"""
    weights = np.ones(len(y_train))
    
    # Give higher weight to wide bandgap materials
    high_bg_mask = y_train > high_bg_threshold
    weights[high_bg_mask] = weight_multiplier
    
    # Gradual weighting for materials near the threshold
    medium_bg_mask = (y_train > 2.5) & (y_train <= high_bg_threshold)
    weights[medium_bg_mask] = 2.0
    
    print(f"Sample weights: {np.sum(weights == 1.0)} normal, {np.sum(weights == 2.0)} medium, {np.sum(weights == weight_multiplier)} high")
    return weights

def train_jarvis_hse_model():
    """Train HSE correction model using JARVIS data with your existing architecture"""
    
    print("🚀 Training JARVIS HSE Correction Model")
    print("=" * 60)
    print("Using your existing improved_bandgap_model.py architecture")
    print("Training on 7,483 materials with real HSE data from JARVIS!")
    print("-" * 60)
    
    # Load JARVIS data
    print("Loading JARVIS HSE data...")
    data = pd.read_csv("jarvis_hse_training_data_20250823_175723.csv")
    data = data.dropna(subset=['hse_bandgap', 'pbe_bandgap'])
    print(f"Loaded {len(data)} samples with HSE data")
    
    # Analyze bandgap distribution
    high_bg_count = np.sum(data['hse_bandgap'] > 3.0)
    very_high_bg_count = np.sum(data['hse_bandgap'] > 5.0)
    print(f"High bandgap materials (>3 eV): {high_bg_count} ({high_bg_count/len(data)*100:.1f}%)")
    print(f"Very high bandgap materials (>5 eV): {very_high_bg_count} ({very_high_bg_count/len(data)*100:.1f}%)")
    print(f"HSE correction range: {data['bandgap_correction'].min():.3f} to {data['bandgap_correction'].max():.3f} eV")
    print(f"Mean HSE correction: {data['bandgap_correction'].mean():.3f} ± {data['bandgap_correction'].std():.3f} eV")
    
    # Create features
    X = create_features(data)
    y = data['hse_bandgap'].values
    
    # Handle NaN values
    X = X.fillna(X.mean())
    
    # Split data with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=pd.cut(y, bins=10, labels=False, duplicates='drop')
        )
    except:
        # Fallback if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create sample weights for training (emphasize high bandgap materials)
    sample_weights = create_sample_weights(y_train, high_bg_threshold=3.0, weight_multiplier=3.0)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training improved models with weighted samples...")
    
    # Model 1: Random Forest with sample weights
    rf_model = RandomForestRegressor(
        n_estimators=300,  # Increased for better performance
        max_depth=20,      # Deeper for complex patterns
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # Model 2: Gradient Boosting with oversampling for high bandgap materials
    high_bg_indices = np.where(y_train > 3.0)[0]
    very_high_bg_indices = np.where(y_train > 5.0)[0]
    
    # Oversample high and very high bandgap materials
    oversample_indices = np.concatenate([
        np.tile(high_bg_indices, 2),  # Duplicate high BG samples
        np.tile(very_high_bg_indices, 3)  # Triple very high BG samples
    ])
    
    X_train_oversampled = np.vstack([X_train_scaled, X_train_scaled[oversample_indices]])
    y_train_oversampled = np.hstack([y_train, y_train[oversample_indices]])
    
    gb_model = HistGradientBoostingRegressor(
        max_iter=300,      # Increased iterations
        max_depth=10,      # Deeper trees
        learning_rate=0.05, # Lower learning rate for better convergence
        random_state=42
    )
    gb_model.fit(X_train_oversampled, y_train_oversampled)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    ensemble_pred = 0.6 * rf_pred + 0.4 * gb_pred
    
    # Calculate overall metrics
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    # Calculate metrics for high bandgap materials specifically
    high_bg_test_mask = y_test > 3.0
    very_high_bg_test_mask = y_test > 5.0
    
    if np.sum(high_bg_test_mask) > 0:
        high_bg_mae = mean_absolute_error(y_test[high_bg_test_mask], ensemble_pred[high_bg_test_mask])
        high_bg_r2 = r2_score(y_test[high_bg_test_mask], ensemble_pred[high_bg_test_mask])
        print(f"\nHigh bandgap (>3 eV) performance:")
        print(f"High BG samples in test: {np.sum(high_bg_test_mask)}")
        print(f"High BG MAE: {high_bg_mae:.4f} eV")
        print(f"High BG R²: {high_bg_r2:.4f}")
    
    if np.sum(very_high_bg_test_mask) > 0:
        very_high_bg_mae = mean_absolute_error(y_test[very_high_bg_test_mask], ensemble_pred[very_high_bg_test_mask])
        very_high_bg_r2 = r2_score(y_test[very_high_bg_test_mask], ensemble_pred[very_high_bg_test_mask])
        print(f"\nVery high bandgap (>5 eV) performance:")
        print(f"Very high BG samples in test: {np.sum(very_high_bg_test_mask)}")
        print(f"Very high BG MAE: {very_high_bg_mae:.4f} eV")
        print(f"Very high BG R²: {very_high_bg_r2:.4f}")
    
    print(f"\nOverall Model Performance:")
    print(f"Random Forest - MAE: {rf_mae:.4f} eV, R²: {rf_r2:.4f}")
    print(f"Gradient Boosting - MAE: {gb_mae:.4f} eV, R²: {gb_r2:.4f}")
    print(f"Ensemble - MAE: {ensemble_mae:.4f} eV, R²: {ensemble_r2:.4f}")
    
    # Enhanced plotting
    plt.figure(figsize=(15, 10))
    
    # Main prediction plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, ensemble_pred, alpha=0.6, c=y_test, cmap='viridis')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True HSE Bandgap (eV)')
    plt.ylabel('Predicted HSE Bandgap (eV)')
    plt.title(f'JARVIS HSE Correction Model\nMAE: {ensemble_mae:.4f} eV, R²: {ensemble_r2:.4f}')
    plt.colorbar(label='True Bandgap (eV)')
    
    # High bandgap focus plot
    plt.subplot(2, 2, 2)
    high_mask = y_test > 2.5
    if np.sum(high_mask) > 0:
        plt.scatter(y_test[high_mask], ensemble_pred[high_mask], alpha=0.7, color='red', label='High BG (>2.5 eV)')
        plt.scatter(y_test[~high_mask], ensemble_pred[~high_mask], alpha=0.3, color='blue', label='Low-Med BG (≤2.5 eV)')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('True HSE Bandgap (eV)')
        plt.ylabel('Predicted HSE Bandgap (eV)')
        plt.title('High vs Low Bandgap Performance')
        plt.legend()
    
    # Residuals plot
    plt.subplot(2, 2, 3)
    residuals = ensemble_pred - y_test
    plt.scatter(ensemble_pred, residuals, alpha=0.6, c=y_test, cmap='viridis')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted HSE Bandgap (eV)')
    plt.ylabel('Residuals (eV)')
    plt.title('Residuals Plot')
    plt.colorbar(label='True Bandgap (eV)')
    
    # Error distribution
    plt.subplot(2, 2, 4)
    plt.hist(np.abs(residuals), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error (eV)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.axvline(ensemble_mae, color='red', linestyle='--', label=f'MAE: {ensemble_mae:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('jarvis_hse_model_results.png', dpi=300, bbox_inches='tight')
    print("Results plot saved as 'jarvis_hse_model_results.png'")
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_data = {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'ensemble_weights': [0.6, 0.4],
        'performance': {
            'rf_mae': rf_mae, 'rf_r2': rf_r2,
            'gb_mae': gb_mae, 'gb_r2': gb_r2,
            'ensemble_mae': ensemble_mae, 'ensemble_r2': ensemble_r2
        },
        'high_bg_performance': {
            'high_bg_mae': high_bg_mae if 'high_bg_mae' in locals() else None,
            'high_bg_r2': high_bg_r2 if 'high_bg_r2' in locals() else None,
            'high_bg_samples': np.sum(high_bg_test_mask) if 'high_bg_test_mask' in locals() else 0,
            'very_high_bg_mae': very_high_bg_mae if 'very_high_bg_mae' in locals() else None,
            'very_high_bg_r2': very_high_bg_r2 if 'very_high_bg_r2' in locals() else None,
            'very_high_bg_samples': np.sum(very_high_bg_test_mask) if 'very_high_bg_test_mask' in locals() else 0
        },
        'training_samples': len(data),
        'feature_importance': dict(zip(X.columns, rf_model.feature_importances_)),
        'model_type': 'jarvis_hse_correction',
        'data_source': 'jarvis',
        'timestamp': timestamp
    }
    
    model_filename = f'jarvis_hse_correction_model_{timestamp}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ JARVIS HSE correction model saved as '{model_filename}'")
    print(f"✅ File size: {len(pickle.dumps(model_data)) / 1024 / 1024:.1f} MB")
    print(f"✅ Trained on {len(data)} materials with real HSE data!")
    
    # Create prediction function
    prediction_script = f'''#!/usr/bin/env python3
"""
JARVIS HSE Bandgap Prediction Function
Generated: {datetime.now().isoformat()}
Trained on {len(data)} materials from JARVIS database
"""

import pickle
import numpy as np
import pandas as pd

def predict_hse_bandgap(pbe_bandgap: float, formula: str = None) -> float:
    """
    Predict HSE bandgap from PBE bandgap using JARVIS-trained model
    
    Args:
        pbe_bandgap: PBE bandgap in eV
        formula: Chemical formula (optional, improves accuracy)
        
    Returns:
        HSE bandgap in eV
    """
    # Load model
    with open('{model_filename}', 'rb') as f:
        model_data = pickle.load(f)
    
    rf_model = model_data['rf_model']
    gb_model = model_data['gb_model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Create features (simplified version)
    features = {{'pbe_bandgap': pbe_bandgap}}
    
    if formula:
        try:
            from pymatgen.core import Composition
            comp = Composition(formula)
            
            features['n_elements'] = len(comp.elements)
            features['total_atoms'] = comp.num_atoms
            
            # Electronegativity
            en_values = [el.X for el in comp.elements if el.X and not np.isnan(el.X)]
            features['avg_electronegativity'] = np.mean(en_values) if en_values else 2.0
            
            # Mass
            mass_values = [el.atomic_mass for el in comp.elements]
            features['avg_atomic_mass'] = np.mean(mass_values) if mass_values else 50.0
            
            # Element presence
            formula_str = str(formula)
            features['has_O'] = int('O' in formula_str)
            features['has_N'] = int('N' in formula_str)
            features['has_C'] = int('C' in formula_str)
            features['has_Si'] = int('Si' in formula_str)
            features['has_Al'] = int('Al' in formula_str)
            features['has_Ti'] = int('Ti' in formula_str)
            features['has_Fe'] = int('Fe' in formula_str)
            features['has_F'] = int('F' in formula_str)
            features['has_H'] = int('H' in formula_str)
            
            # Material types
            features['is_oxide'] = features['has_O']
            features['is_nitride'] = features['has_N']
            features['is_carbide'] = features['has_C']
            features['is_fluoride'] = features['has_F']
            features['is_hydride'] = features['has_H']
            
        except:
            # Default values if composition parsing fails
            for key in ['n_elements', 'total_atoms', 'avg_electronegativity', 'avg_atomic_mass',
                       'has_O', 'has_N', 'has_C', 'has_Si', 'has_Al', 'has_Ti', 'has_Fe', 'has_F', 'has_H',
                       'is_oxide', 'is_nitride', 'is_carbide', 'is_fluoride', 'is_hydride']:
                features[key] = 0
            features['avg_electronegativity'] = 2.0
            features['avg_atomic_mass'] = 50.0
    else:
        # Default values when no formula provided
        for key in feature_names:
            if key not in features:
                features[key] = 0
        features['avg_electronegativity'] = 2.0
        features['avg_atomic_mass'] = 50.0
    
    # Derived features
    features['pbe_squared'] = pbe_bandgap ** 2
    features['pbe_cubed'] = pbe_bandgap ** 3
    features['pbe_sqrt'] = np.sqrt(abs(pbe_bandgap))
    features['log_pbe'] = np.log1p(pbe_bandgap)
    features['en_pbe_product'] = features['avg_electronegativity'] * pbe_bandgap
    features['en_squared'] = features['avg_electronegativity'] ** 2
    
    # Create feature vector
    X = np.array([[features.get(name, 0) for name in feature_names]])
    X_scaled = scaler.transform(X)
    
    # Make ensemble prediction
    rf_pred = rf_model.predict(X_scaled)[0]
    gb_pred = gb_model.predict(X_scaled)[0]
    hse_bandgap = 0.6 * rf_pred + 0.4 * gb_pred
    
    return max(0.0, hse_bandgap)

def predict_hse_correction(pbe_bandgap: float, formula: str = None) -> float:
    """Predict HSE correction (HSE - PBE)"""
    hse_bandgap = predict_hse_bandgap(pbe_bandgap, formula)
    return hse_bandgap - pbe_bandgap

if __name__ == "__main__":
    # Test predictions
    test_cases = [
        (0.5, "Si"),
        (1.0, "GaAs"), 
        (1.5, "GaN"),
        (2.0, "ZnO"),
        (2.5, "TiO2"),
        (3.0, "AlN")
    ]
    
    print("JARVIS HSE Bandgap Predictions:")
    print("PBE (eV)  Formula  HSE (eV)  Correction (eV)")
    print("-" * 45)
    
    for pbe, formula in test_cases:
        hse = predict_hse_bandgap(pbe, formula)
        corr = predict_hse_correction(pbe, formula)
        print(f"{{pbe:6.2f}}    {{formula:7s}}  {{hse:6.2f}}    {{corr:+6.2f}}")
'''
    
    script_filename = f"predict_jarvis_hse_{timestamp}.py"
    with open(script_filename, 'w') as f:
        f.write(prediction_script)
    
    print(f"🎯 Created prediction script: {script_filename}")
    
    return model_data, model_filename, script_filename

if __name__ == "__main__":
    model_data, model_file, script_file = train_jarvis_hse_model()