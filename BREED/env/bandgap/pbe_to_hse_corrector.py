#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import sys

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
        
        # NEW: Features for wide bandgap materials
        features['is_oxide'] = data['formula'].str.contains('O', na=False).astype(int)
        features['is_nitride'] = data['formula'].str.contains('N', na=False).astype(int)
        features['is_carbide'] = data['formula'].str.contains('C', na=False).astype(int)
        
        # Derived features - enhanced for high bandgaps
        features['pbe_squared'] = features['pbe_bandgap'] ** 2
        features['pbe_cubed'] = features['pbe_bandgap'] ** 3  # NEW: Better for high values
        features['pbe_sqrt'] = np.sqrt(np.abs(features['pbe_bandgap']))
        features['log_pbe'] = np.log1p(features['pbe_bandgap'])  # NEW: Log scaling
        features['en_pbe_product'] = features['avg_electronegativity'] * features['pbe_bandgap']
        features['en_squared'] = features['avg_electronegativity'] ** 2  # NEW
    
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

def train_improved_model():
    """Train an improved bandgap correction model with weighted training"""
    
    # Load data
    print("Loading data...")
    data = pd.read_csv("env/bandgap/paired_bandgap_dataset.csv")
    # Map column names to expected format
    if 'es_bandgap' in data.columns:
        data['hse_bandgap'] = data['es_bandgap']
    data = data.dropna(subset=['hse_bandgap', 'pbe_bandgap'])
    print(f"Loaded {len(data)} samples")
    
    # Analyze bandgap distribution
    high_bg_count = np.sum(data['hse_bandgap'] > 3.0)
    print(f"High bandgap materials (>3 eV): {high_bg_count} ({high_bg_count/len(data)*100:.1f}%)")
    
    # Create features
    X = create_features(data)
    y = data['hse_bandgap'].values
    
    # Handle NaN values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)  # Stratified split
    )
    
    # Create sample weights for training
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
    
    # Model 2: Gradient Boosting (doesn't support sample_weight directly)
    # So we'll use a different approach - oversample high bandgap materials
    high_bg_indices = np.where(y_train > 3.0)[0]
    oversample_indices = np.tile(high_bg_indices, 2)  # Duplicate high BG samples
    
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
    if np.sum(high_bg_test_mask) > 0:
        high_bg_mae = mean_absolute_error(y_test[high_bg_test_mask], ensemble_pred[high_bg_test_mask])
        high_bg_r2 = r2_score(y_test[high_bg_test_mask], ensemble_pred[high_bg_test_mask])
        print(f"\nHigh bandgap (>3 eV) performance:")
        print(f"High BG samples in test: {np.sum(high_bg_test_mask)}")
        print(f"High BG MAE: {high_bg_mae:.4f} eV")
        print(f"High BG R²: {high_bg_r2:.4f}")
    
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
    plt.title(f'Improved Ensemble Model\nMAE: {ensemble_mae:.4f} eV, R²: {ensemble_r2:.4f}')
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
    plt.savefig('improved_bandgap_results.png', dpi=300, bbox_inches='tight')
    print("Enhanced results plot saved as 'improved_bandgap_results.png'")
    
    # Save improved model
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
            'high_bg_samples': np.sum(high_bg_test_mask) if 'high_bg_test_mask' in locals() else 0
        },
        'training_samples': len(data),
        'feature_importance': dict(zip(X.columns, rf_model.feature_importances_)),
        'model_type': 'improved_weighted'
    }
    
    with open('improved_bandgap_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Improved model saved as 'improved_bandgap_model.pkl'")
    print(f"✓ File size: {len(pickle.dumps(model_data)) / 1024 / 1024:.1f} MB")
    print(f"✓ Enhanced for high bandgap materials!")
    
    return model_data

if __name__ == "__main__":
    train_improved_model()