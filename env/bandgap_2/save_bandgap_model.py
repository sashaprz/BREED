#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
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

def create_features(data):
    """Create features from the data"""
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
        
        # Derived features
        features['pbe_squared'] = features['pbe_bandgap'] ** 2
        features['pbe_sqrt'] = np.sqrt(np.abs(features['pbe_bandgap']))
        features['en_pbe_product'] = features['avg_electronegativity'] * features['pbe_bandgap']
    
    print(f"Created {features.shape[1]} features")
    return features

def train_and_save_model():
    """Train and save the bandgap correction model"""
    
    # Load data
    print("Loading data...")
    data = pd.read_csv("jarvis_paired_data/jarvis_paired_bandgaps.csv")
    data = data.dropna(subset=['hse_bandgap', 'pbe_bandgap'])
    print(f"Loaded {len(data)} samples")
    
    # Create features
    X = create_features(data)
    y = data['hse_bandgap'].values
    
    # Handle NaN values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training models...")
    
    # Model 1: Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Model 2: Gradient Boosting
    gb_model = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
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
    
    # Save model with multiple methods for maximum compatibility
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
            'training_script': 'save_bandgap_model.py'
        }
    }
    
    # Method 1: Save with current pickle (highest protocol)
    print("Saving model with pickle (highest protocol)...")
    with open('bandgap_correction_model.pkl', 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Method 2: Save with joblib (often more robust for sklearn models)
    print("Saving model with joblib...")
    joblib.dump(model_data, 'bandgap_correction_model_joblib.pkl', compress=3)
    
    # Method 3: Save with protocol 4 for broader compatibility
    print("Saving model with pickle protocol 4...")
    with open('bandgap_correction_model_v4.pkl', 'wb') as f:
        pickle.dump(model_data, f, protocol=4)
    
    # Get file sizes
    pkl_size = os.path.getsize('bandgap_correction_model.pkl') / 1024 / 1024
    joblib_size = os.path.getsize('bandgap_correction_model_joblib.pkl') / 1024 / 1024
    v4_size = os.path.getsize('bandgap_correction_model_v4.pkl') / 1024 / 1024
    
    print(f"\n✅ MODEL SAVED SUCCESSFULLY!")
    print(f"📁 bandgap_correction_model.pkl (pickle): {pkl_size:.1f} MB")
    print(f"📁 bandgap_correction_model_joblib.pkl (joblib): {joblib_size:.1f} MB")
    print(f"📁 bandgap_correction_model_v4.pkl (pickle v4): {v4_size:.1f} MB")
    print(f"🎯 Use the joblib version for maximum compatibility!")
    
    # Test loading immediately to verify
    print("\n🧪 Testing model loading...")
    try:
        test_model = joblib.load('bandgap_correction_model_joblib.pkl')
        print("✅ Joblib model loads successfully!")
        
        # Quick prediction test
        test_features = pd.DataFrame({
            'pbe_bandgap': [0.005],
            'n_elements': [3], 'total_atoms': [12], 'avg_electronegativity': [2.5], 'avg_atomic_mass': [45.0],
            'has_O': [1], 'has_N': [0], 'has_C': [0], 'has_Si': [0],
            'has_Al': [0], 'has_Ti': [0], 'has_Fe': [0],
            'pbe_squared': [0.005 ** 2], 'pbe_sqrt': [np.sqrt(0.005)], 'en_pbe_product': [2.5 * 0.005]
        })
        
        X_test = test_model['scaler'].transform(test_features)
        rf_pred = test_model['rf_model'].predict(X_test)[0]
        gb_pred = test_model['gb_model'].predict(X_test)[0]
        final_pred = test_model['ensemble_weights'][0] * rf_pred + test_model['ensemble_weights'][1] * gb_pred
        
        print(f"✅ Prediction test: 0.005 eV → {final_pred:.4f} eV ({final_pred/0.005:.1f}x correction)")
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
    
    return model_data

if __name__ == "__main__":
    train_and_save_model()