#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import joblib
import sys
import os
from datetime import datetime

# Print version information for debugging
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {__import__('sklearn').__version__}")
print(f"Joblib version: {joblib.__version__}")
print("-" * 50)

def create_advanced_features(data):
    """Create advanced features from the data"""
    print("Creating advanced features...")
    
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
                    en_values = [el.X for el in comp.elements if el.X]
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
        
        # Derived features
        features['pbe_squared'] = features['pbe_bandgap'] ** 2
        features['pbe_sqrt'] = np.sqrt(np.abs(features['pbe_bandgap']))
        features['en_pbe_product'] = features['avg_electronegativity'] * features['pbe_bandgap']
    
    print(f"Created {features.shape[1]} features")
    return features

def train_ensemble_model(data_path):
    """Train an ensemble of models for bandgap correction"""
    
    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)
    data = data.dropna(subset=['hse_bandgap', 'pbe_bandgap'])
    print(f"Loaded {len(data)} samples")
    
    # Create features
    X = create_advanced_features(data)
    y = data['hse_bandgap'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Handle NaN values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training ensemble models...")
    
    # Model 1: Random Forest (uses all CPU cores)
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    # Model 2: Histogram Gradient Boosting (modern, NaN-tolerant)
    gb_model = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        verbose=1
    )
    
    # Train models
    print("Training Random Forest...")
    rf_model.fit(X_train_scaled, y_train)
    
    print("Training Gradient Boosting...")
    gb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    
    # Ensemble prediction (weighted average)
    ensemble_pred = 0.6 * rf_pred + 0.4 * gb_pred
    
    # Calculate metrics
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    print(f"\nModel Performance:")
    print(f"Random Forest - MAE: {rf_mae:.4f} eV, R²: {rf_r2:.4f}")
    print(f"Gradient Boosting - MAE: {gb_mae:.4f} eV, R²: {gb_r2:.4f}")
    print(f"Ensemble - MAE: {ensemble_mae:.4f} eV, R²: {ensemble_r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, rf_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True HSE Bandgap (eV)')
    plt.ylabel('Predicted HSE Bandgap (eV)')
    plt.title(f'Random Forest\nMAE: {rf_mae:.4f} eV, R²: {rf_r2:.4f}')
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, gb_pred, alpha=0.6, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True HSE Bandgap (eV)')
    plt.ylabel('Predicted HSE Bandgap (eV)')
    plt.title(f'Gradient Boosting\nMAE: {gb_mae:.4f} eV, R²: {gb_r2:.4f}')
    
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, ensemble_pred, alpha=0.6, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True HSE Bandgap (eV)')
    plt.ylabel('Predicted HSE Bandgap (eV)')
    plt.title(f'Ensemble Model\nMAE: {ensemble_mae:.4f} eV, R²: {ensemble_r2:.4f}')
    
    plt.tight_layout()
    plt.savefig('fast_bandgap_results.png', dpi=300, bbox_inches='tight')
    print("Results plot saved as 'fast_bandgap_results.png'")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf_model.feature_importances_,
        'gb_importance': gb_model.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    print("\nFeature Importance (Random Forest):")
    print(importance[['feature', 'rf_importance']].head(10))
    
    # Save models with enhanced metadata and multiple formats
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
        'training_samples': len(data),
        'feature_importance': dict(zip(X.columns, rf_model.feature_importances_)),
        'metadata': {
            'created_date': datetime.now().isoformat(),
            'numpy_version': np.__version__,
            'sklearn_version': __import__('sklearn').__version__,
            'pandas_version': pd.__version__,
            'python_version': sys.version,
            'training_script': 'fast_bandgap_trainer.py'
        }
    }
    
    # Save with multiple methods for compatibility
    print("Saving model with pickle (highest protocol)...")
    with open('fast_bandgap_model.pkl', 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Saving model with joblib...")
    joblib.dump(model_data, 'fast_bandgap_model_joblib.pkl', compress=3)
    
    print("Saving model with pickle protocol 4...")
    with open('fast_bandgap_model_v4.pkl', 'wb') as f:
        pickle.dump(model_data, f, protocol=4)
    
    # Get file sizes
    pkl_size = os.path.getsize('fast_bandgap_model.pkl') / 1024 / 1024
    joblib_size = os.path.getsize('fast_bandgap_model_joblib.pkl') / 1024 / 1024
    v4_size = os.path.getsize('fast_bandgap_model_v4.pkl') / 1024 / 1024
    
    print(f"\n✅ MODELS SAVED SUCCESSFULLY!")
    print(f"📁 fast_bandgap_model.pkl (pickle): {pkl_size:.1f} MB")
    print(f"📁 fast_bandgap_model_joblib.pkl (joblib): {joblib_size:.1f} MB")
    print(f"📁 fast_bandgap_model_v4.pkl (pickle v4): {v4_size:.1f} MB")
    print(f"🎯 Use the joblib version for maximum compatibility!")
    
    return model_data

def predict_bandgap(pbe_bandgap, formula=None, model_file='fast_bandgap_model_joblib.pkl'):
    """Use the trained ensemble to predict HSE bandgap"""
    
    # Try joblib first, then fallback to pickle
    try:
        if model_file.endswith('_joblib.pkl'):
            model_data = joblib.load(model_file)
        else:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading model {model_file}: {e}")
        # Try alternative formats
        if 'joblib' in model_file:
            alt_file = model_file.replace('_joblib.pkl', '.pkl')
        else:
            alt_file = model_file.replace('.pkl', '_joblib.pkl')
        
        try:
            if alt_file.endswith('_joblib.pkl'):
                model_data = joblib.load(alt_file)
            else:
                with open(alt_file, 'rb') as f:
                    model_data = pickle.load(f)
            print(f"Successfully loaded alternative model: {alt_file}")
        except:
            raise Exception(f"Could not load any model format for {model_file}")
    
    rf_model = model_data['rf_model']
    gb_model = model_data['gb_model']
    scaler = model_data['scaler']
    weights = model_data['ensemble_weights']
    
    # Create features (simplified for single prediction)
    features = pd.DataFrame({
        'pbe_bandgap': [pbe_bandgap],
        'n_elements': [2],  # Default
        'total_atoms': [2],  # Default
        'avg_electronegativity': [2.0],  # Default
        'avg_atomic_mass': [50.0],  # Default
        'has_O': [1 if formula and 'O' in formula else 0],
        'has_N': [1 if formula and 'N' in formula else 0],
        'has_C': [1 if formula and 'C' in formula else 0],
        'has_Si': [1 if formula and 'Si' in formula else 0],
        'has_Al': [1 if formula and 'Al' in formula else 0],
        'has_Ti': [1 if formula and 'Ti' in formula else 0],
        'has_Fe': [1 if formula and 'Fe' in formula else 0],
        'pbe_squared': [pbe_bandgap ** 2],
        'pbe_sqrt': [np.sqrt(abs(pbe_bandgap))],
        'en_pbe_product': [2.0 * pbe_bandgap]
    })
    
    # Scale and predict
    X_scaled = scaler.transform(features)
    rf_pred = rf_model.predict(X_scaled)[0]
    gb_pred = gb_model.predict(X_scaled)[0]
    
    ensemble_pred = weights[0] * rf_pred + weights[1] * gb_pred
    
    return ensemble_pred

def main():
    data_path = "jarvis_paired_data/jarvis_paired_bandgaps.csv"
    
    try:
        print("=== Fast Bandgap Correction Training ===")
        print("Using optimized sklearn models with full CPU utilization")
        
        model_data = train_ensemble_model(data_path)
        
        print("\n=== Testing Predictions ===")
        test_cases = [
            (1.5, "SiO2"),
            (2.0, "TiO2"), 
            (0.5, "GaAs"),
            (3.0, "Al2O3"),
            (1.0, "Si")
        ]
        
        for pbe_bg, formula in test_cases:
            hse_pred = predict_bandgap(pbe_bg, formula)
            print(f"Formula: {formula:6s}, PBE: {pbe_bg:.1f} eV → Predicted HSE: {hse_pred:.3f} eV")
        
        perf = model_data['performance']
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Best model performance - MAE: {perf['ensemble_mae']:.4f} eV, R²: {perf['ensemble_r2']:.4f}")
        print(f"✓ Model saved as 'fast_bandgap_model.pkl'")
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()