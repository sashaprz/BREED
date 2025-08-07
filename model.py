import pandas as pd
import numpy as np
import json
import re
import math
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import pickle

#loading and formatting data
obelix = pd.read_csv(r'C:\Users\Sasha\OneDrive\Documents\Projects\FR8\ionic_conductivity_data\OBELiX_data.csv').drop(['ID', 'DOI', 'CIF'], axis=1) #removing useless rows

liverpool_original = pd.read_csv(r'C:\Users\Sasha\OneDrive\Documents\Projects\FR8\ionic_conductivity_data\liverpool_data.csv', skiprows=4).drop(['log_target', 'ID', 'source'], axis=1) #removing useless rows

liverpool = liverpool_original.rename(
    columns={
        'target': 'Ionic conductivity (S cm-1)', #checking inside the raw data, every instance has a specifier column that indicates it's ionic conductivity
        'composition': 'Composition', #standardizing names to match obelix
        'ChemicalFamily': 'Chemical Family',
        'family': 'Structural Family'
    }
)

#removed the vscode data section because there were multiple instances where the values were just WAY off what was expected for that composition, and it was screwing up the model
"""
with open('/content/drive/My Drive/Projects/FR8/IonicalConductivityDatabase.json', 'r') as f:
  vscode_original = json.load(f)
  vscode_1 = pd.json_normalize(vscode_original) #flattening the data because it's a bunch of embedded dictionaries + then i can work with it as a pandas dataframe
  vscode_2 = vscode_1[['BatteryConductivity.value', 'BatteryConductivity.compound.Compound.names', 'BatteryConductivity.units']] #this double bracket thing selects only specific columnns
  vscode = vscode_2.rename(
      columns={
          'BatteryConductivity.value' : 'Ionic conductivity (S cm-1)',
          'BatteryConductivity.compound.Compound.names': 'Composition',
          'BatteryConductivity.units': 'Units'#this renames the columns to make them consistent
      }
  )
vscode['Composition'] = vscode['Composition'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else x) #keeping only the second element in the composition list as that is the actual chemical formula

def extract_formula(lst): #this is going to find the actual formula from the big list of info somewhat related to composition in the dataset
  if not isinstance(lst, list):
    lst = [lst] #if its not a list make it one
  for item in lst:
    if (
        isinstance(item, str)
        and re.match(r'[A-Z][A-Za-z0-9().\-]*$', item) #if it has the capital letter pattern
        and re.search(r'\d', item) # if it has at least one number
    ):
        return item
  return None #returning none will equal NaN later

def extract_first_number(val): #removing the square brackets on the ionic conductivity values
  # If it's a list, return the first element
  if isinstance(val, list):
      return val[0] if len(val) > 0 else None
  # If it's a string that looks like a list, try to eval it safely
  if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
      try:
          import ast
          lst = ast.literal_eval(val)
          return lst[0] if isinstance(lst, list) and len(lst) > 0 else None
      except Exception:
          return None
  # Otherwise, try to convert to float
  try:
      return float(val)
  except Exception:
      return None

def convert_to_s_per_cm(row):
    value = row['Ionic conductivity (S cm-1)']
    units = row['Units']
    # Extract the first number if it's a list
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None or units is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    # Only call .lower() if units is a string
    if not isinstance(units, str):
        return None
    units = units.lower()
    if 'millisiemens' in units:
        value = value / 1000
    elif 'microsiemens' in units:
        value = value / 1_000_000
    elif 'siemens' in units:
        pass
    else:
        return None
    return value

vscode['Ionic conductivity (S cm-1)'] = vscode.apply(convert_to_s_per_cm, axis=1)
vscode['Ionic conductivity (S cm-1)'] = vscode['Ionic conductivity (S cm-1)'].apply(extract_first_number)
vscode['Composition'] = vscode['Composition'].apply(extract_formula)
vscode = vscode.dropna() #dropping the rows with NaN values, ie the ones that don't have a valid chemical formula
"""

#making all 3 sets have the same columns
all_columns = set(obelix.columns) | set(liverpool.columns) #| set(vscode.columns) #a set makes sure all the values are unique so we dont have duplicate columns. | is the union operator, it combines two or more sets into one

obelix = obelix.reindex(columns=all_columns)
liverpool = liverpool.reindex(columns=all_columns)
#vscode = vscode.reindex(columns=all_columns)

obelix.loc[:, 'temperature'] = 25.0 #setting all the Obelix ones to room temp as I know that's how they were measured

obelix['temperature'] = 25.0

column_order = ['Composition', 'Ionic Conductivity (S cm-1)', 'temperature', 'Chemical Family', 'Family', 'Space group number','a', 'b', 'c', 'alpha', 'beta', 'gamma']
for df in[obelix, liverpool]:
  df = df.reindex(columns = column_order)

data = pd.concat([obelix, liverpool], axis=0, ignore_index=True)


#extracting additional features
def get_li_concentration_per_formula_unit(Composition):
  """
  Extracts the number of Li atoms per formula unit from a chemical formula string.
  If no subscript is present after 'Li', returns 1.
  If 'Li' is not found, returns 0.
  """
  match = re.search(r'Li([0-9\.]*)', Composition) #this is looking for a pattern of Li followed by any number 0 to 9, where the star means that this pattern must occur 0 or more times. ie it will still work if the Li doesn't have a subscript
  if match:
    num = match.group(1) #the first thing in the re.search is the first group. in this case the second group (index 1) is the number
    if num == '':
      return 1 #if no subscript, there is only 1 Li atom/unit
    else:
      return float(num)
  return 0 #no Li found

def classify_crystal_system(
    a, b, c, alpha, beta, gamma,
    length_tol=0.00001, angle_tol=0.1
):
    """
    defines the crystal system based on the lattice parameters
    """
    # Convert all to float
    a, b, c, alpha, beta, gamma = map(float, [a, b, c, alpha, beta, gamma])

    # Separate helpers for lengths and angles
    def eq_length(x, y): return math.isclose(x, y, abs_tol=length_tol)
    def eq_angle(x, y):  return math.isclose(x, y, abs_tol=angle_tol)

    # Cubic: all sides equal, all angles 90
    if eq_length(a, b) and eq_length(b, c) and \
       eq_angle(alpha, 90) and eq_angle(beta, 90) and eq_angle(gamma, 90):
        return "cubic"
    # Tetragonal: a = b ≠ c, all angles 90
    elif eq_length(a, b) and not eq_length(b, c) and \
         eq_angle(alpha, 90) and eq_angle(beta, 90) and eq_angle(gamma, 90):
        return "tetragonal"
    # Orthorhombic: a ≠ b ≠ c, all angles 90
    elif not eq_length(a, b) and not eq_length(b, c) and not eq_length(a, c) and \
         eq_angle(alpha, 90) and eq_angle(beta, 90) and eq_angle(gamma, 90):
        return "orthorhombic"
    # Hexagonal: a = b ≠ c, alpha = beta = 90, gamma = 120
    elif eq_length(a, b) and not eq_length(b, c) and \
         eq_angle(alpha, 90) and eq_angle(beta, 90) and eq_angle(gamma, 120):
        return "hexagonal"
    # Trigonal (Rhombohedral): a = b = c, alpha = beta = gamma ≠ 90
    elif eq_length(a, b) and eq_length(b, c) and \
         eq_angle(alpha, beta) and eq_angle(beta, gamma) and not eq_angle(alpha, 90):
        return "trigonal"
    # Monoclinic: a ≠ b ≠ c, alpha = gamma = 90, beta ≠ 90
    elif not eq_length(a, b) and not eq_length(b, c) and not eq_length(a, c) and \
         eq_angle(alpha, 90) and not eq_angle(beta, 90) and eq_angle(gamma, 90):
        return "monoclinic"
    # Triclinic: a ≠ b ≠ c, all angles ≠ 90
    elif not eq_length(a, b) and not eq_length(b, c) and not eq_length(a, c) and \
         not eq_angle(alpha, 90) and not eq_angle(beta, 90) and not eq_angle(gamma, 90):
        return "triclinic"
    else:
        return "unknown"

def calculate_unit_cell_volume(a, b, c, alpha, beta, gamma):
  """
  Calculate the volume of a unit cell with right angles (α=β=γ=90°).

  Parameters:
        a (float): Length of unit cell edge a
        b (float): Length of unit cell edge b
        c (float): Length of unit cell edge c

    Returns:
        float: Volume of the unit cell
  """
  a, b, c, alpha, beta, gamma = map(float, [a, b , c, alpha, beta, gamma]) #covnverting to floats so i can do math
  alpha, beta, gamma = map(math.radians, [alpha, beta, gamma]) #converting to radians for math.cos

  #standard equation for calculating volume, this is because not all crystal structures have a unit cell with angles 90 degrees. this is volume of a parallelepiped, which is made of 3 vectors (a b and c)
  unit_cell_volume = a * b * c * math.sqrt(1-math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2 + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma))

  return unit_cell_volume

def parse_composition_series(composition_series):
    """
    Parse a pandas Series of chemical composition strings into dictionaries.
    """
    def _parse_single_composition(composition_string):
        """Helper function to parse a single composition string."""
        # Handle NaN or None values
        if pd.isna(composition_string) or composition_string is None:
            return {}

        # Pattern to match element symbol followed by optional number
        pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        matches = re.findall(pattern, str(composition_string))
        composition_dict = {}

        for element, amount in matches:
            # If no amount is specified, default to 1.0
            if amount == '':
                amount = 1.0
            else:
                try:
                    amount = float(amount)
                except ValueError:
                    # Skip invalid amounts
                    continue

            composition_dict[element] = amount
        return composition_dict
    return composition_series.apply(_parse_single_composition)

def convert_to_percentages_key_elements(composition_dict, key_elements):
    """
    Convert a composition dictionary to percentages for key elements only.
    """
    # Handle empty dictionaries
    if not composition_dict:
        return {element: 0.0 for element in key_elements}

    # Calculate total for normalization
    total = sum(composition_dict.values())
    if total == 0:
        return {element: 0.0 for element in key_elements}

    # Only include key elements in the result
    result = {}
    for element in key_elements:
        if element in composition_dict:
            result[element] = round((composition_dict[element] / total) * 100, 2)
        else:
            result[element] = 0.0

    return result

def convert_series_to_percentages_key_elements(composition_series, key_elements):
    """
    Convert a pandas Series of composition dictionaries to percentages for key elements only.
    """
    return composition_series.apply(lambda comp: convert_to_percentages_key_elements(comp, key_elements))

def add_key_element_columns_to_dataframe(data, percentage_compositions):
    """
    Add key element percentage columns to the dataframe.
    """
    # Create DataFrame with key elements
    element_data = []
    for composition_dict in percentage_compositions:
        element_data.append(composition_dict)

    element_df = pd.DataFrame(element_data)

    # Add percentage suffix to column names for clarity
    element_df.columns = [f'{element}_pct' for element in element_df.columns]

    # Concatenate with original dataframe
    result_df = pd.concat([data, element_df], axis=1)

    return result_df, list(element_df.columns)

# Define key elements for solid state electrolyte ionic conductivity prediction
key_elements = [
    'Li',  # Mobile ion - critical for Li-ion conductivity
    'F',   # Common anion in fluoride-based electrolytes
    'O',   # Common in oxide electrolytes
    'S',   # Sulfide electrolytes often have high conductivity
    'C',   # Carbon-based frameworks
    'N',   # Nitride electrolytes
    'Ti',  # Common framework cation
    'Zr',  # NASICON-type frameworks
    'Al',  # Common dopant/framework element
    'P',   # Phosphate-based electrolytes (NASICON, etc.)
    'Si',  # Silicon-based frameworks
    'Ge',  # Similar to Si, used in some electrolytes
    'La',  # LLZO and other garnet-type electrolytes
    'Cl',  # Chloride electrolytes
    'Br',  # Bromide electrolytes
    'I'    # Iodide electrolytes
]




# Usage in your main code:
data['[Li+]/formula unit'] = data['Composition'].apply(get_li_concentration_per_formula_unit)
data['Crystal System'] = data.apply(
    lambda row: classify_crystal_system(row['a'], row['b'], row['c'], row['alpha'], row['beta'], row['gamma']),
    axis=1)
data['Unit cell volume (Angstroms cubed)'] = data.apply(
    lambda row: calculate_unit_cell_volume(row['a'], row['b'], row['c'], row['alpha'], row['beta'], row['gamma']),
    axis=1)
#data = data[data['Ionic conductivity (S cm-1)'] <= 0.01] #removing all the stuff greater than 10 as that is probably a mistake

# Add key element columns (matching your original structure):
nan_compositions = data['Composition'].isna().sum()
parsed_compositions = parse_composition_series(data['Composition'])
empty_compositions = sum(1 for comp in parsed_compositions if not comp)
percentage_compositions = convert_series_to_percentages_key_elements(parsed_compositions, key_elements)
data, elements_added = add_key_element_columns_to_dataframe(data, percentage_compositions)

data = data.dropna(subset=["Ionic conductivity (S cm-1)"])
data = data[np.isfinite(data["Ionic conductivity (S cm-1)"])]

#print(data[data['Ionic conductivity (S cm-1)'] > 1][['Composition', 'Ionic conductivity (S cm-1)']])

# MAJOR IMPROVEMENT 1: Better log transformation
def safe_log_transform(values, base=10, epsilon=1e-20):
    """
    Safely apply log transformation to ionic conductivity values
    """
    # Add small epsilon to avoid log(0) and handle negative values
    safe_values = np.maximum(values, epsilon)
    return np.log10(safe_values) if base == 10 else np.log(safe_values)

def inverse_log_transform(log_values, base=10, epsilon=1e-20):
    """
    Convert log-transformed values back to original scale
    """
    if base == 10:
        return np.power(10, log_values)
    else:
        return np.exp(log_values)

# MAJOR IMPROVEMENT 2: Add physics-based features
def add_physics_features(data):
    """
    Add physics-based features that are relevant for ionic conductivity
    """
    # Ionic radii approximations (in Angstroms) - you'd want to get these from a proper database
    ionic_radii = {
        'Li': 0.76, 'Na': 1.02, 'K': 1.38, 'Rb': 1.52, 'Cs': 1.67,
        'F': 1.33, 'Cl': 1.81, 'Br': 1.96, 'I': 2.20,
        'O': 1.40, 'S': 1.84, 'Se': 1.98, 'Te': 2.21,
        'Al': 0.535, 'Ti': 0.605, 'Zr': 0.72, 'P': 0.44, 'Si': 0.40
    }
    
    # Calculate weighted average ionic radius
    def calc_weighted_ionic_radius(row):
        total_weighted_radius = 0
        total_atoms = 0
        
        for element in ionic_radii.keys():
            if f'{element}_pct' in row.index and row[f'{element}_pct'] > 0:
                weight = row[f'{element}_pct'] / 100.0
                total_weighted_radius += ionic_radii[element] * weight
                total_atoms += weight
        
        return total_weighted_radius / total_atoms if total_atoms > 0 else 0
    
    # Add new physics-based features
    data['weighted_ionic_radius'] = data.apply(calc_weighted_ionic_radius, axis=1)
    
    # Lattice parameter ratios (crystal structure descriptors)
    data['b_a_ratio'] = data['b'] / data['a']
    data['c_a_ratio'] = data['c'] / data['a']
    data['c_b_ratio'] = data['c'] / data['b']
    
    # Li concentration normalized by unit cell volume
    data['li_density'] = data['[Li+]/formula unit'] / data['Unit cell volume (Angstroms cubed)']
    
    # Temperature effects (Arrhenius-like behavior)
    data['inverse_temperature'] = 1000.0 / (data['temperature'] + 273.15)  # 1000/T for numerical stability
    
    # Electronegativity differences (approximate, you'd want Pauling electronegativity values)
    electronegativity = {
        'Li': 0.98, 'F': 3.98, 'O': 3.44, 'S': 2.58, 'Cl': 3.16,
        'P': 2.19, 'Si': 1.90, 'Al': 1.61, 'Ti': 1.54, 'Zr': 1.33
    }
    
    def calc_electronegativity_range(row):
        electronegativities = []
        for element in electronegativity.keys():
            if f'{element}_pct' in row.index and row[f'{element}_pct'] > 0:
                electronegativities.append(electronegativity[element])
        
        if len(electronegativities) >= 2:
            return max(electronegativities) - min(electronegativities)
        return 0
    
    data['electronegativity_range'] = data.apply(calc_electronegativity_range, axis=1)
    
    # Structural complexity (number of different elements)
    def count_elements(row):
        count = 0
        for element in key_elements:
            if f'{element}_pct' in row.index and row[f'{element}_pct'] > 0:
                count += 1
        return count
    
    data['element_count'] = data.apply(count_elements, axis=1)
    
    return data

# MAJOR IMPROVEMENT 3: Data stratification for better training
def stratified_split(data, target_column, n_bins=10, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data while maintaining distribution across conductivity ranges
    """
    # Create bins based on log-transformed target
    log_target = safe_log_transform(data[target_column])
    data['temp_bins'] = pd.cut(log_target, bins=n_bins, duplicates='drop')
    
    # Stratified split
    train_val, test = train_test_split(
        data, test_size=test_size, random_state=random_state, 
        stratify=data['temp_bins']
    )
    
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), random_state=random_state,
        stratify=train_val['temp_bins']
    )
    
    # Remove temporary column
    for df in [train, val, test]:
        df.drop('temp_bins', axis=1, inplace=True)
    
    return train, val, test

# MAJOR IMPROVEMENT 4: Multiple models ensemble
def train_multiple_models(X_train, y_train, X_val, y_val):
    """
    Train multiple models and return them for ensemble
    """
    models = {}
    
    # XGBoost with better hyperparameters
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=100,
        eval_metric='rmse'
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    models['xgb'] = xgb_model
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=100,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    models['lgb'] = lgb_model
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['rf'] = rf_model
    
    return models

def ensemble_predict(models, X, weights=None):
    """
    Make ensemble predictions from multiple models
    """
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    predictions = []
    for model_name, model in models.items():
        pred = model.predict(X)
        predictions.append(pred)
    
    # Weighted average
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    return ensemble_pred

# APPLY IMPROVEMENTS TO YOUR MAIN CODE
# Add physics features
data = add_physics_features(data)

# Better data cleaning - remove extreme outliers
Q1 = data['Ionic conductivity (S cm-1)'].quantile(0.01)
Q3 = data['Ionic conductivity (S cm-1)'].quantile(0.99)
data = data[(data['Ionic conductivity (S cm-1)'] >= Q1) & (data['Ionic conductivity (S cm-1)'] <= Q3)]

# Remove rows with missing target
data = data.dropna(subset=["Ionic conductivity (S cm-1)"])
data = data[np.isfinite(data["Ionic conductivity (S cm-1)"])]

# MAJOR CHANGE: Use stratified split and log transform target
train_set, val_set, test_set = stratified_split(data, 'Ionic conductivity (S cm-1)')

# Prepare features and targets
def prepare_data(dataset):
    properties = dataset.drop(["Ionic conductivity (S cm-1)"], axis=1)
    labels = dataset["Ionic conductivity (S cm-1)"].copy()
    labels_log = safe_log_transform(labels)  # LOG TRANSFORM HERE
    return properties, labels, labels_log

train_properties, train_labels, train_labels_log = prepare_data(train_set)
val_properties, val_labels, val_labels_log = prepare_data(val_set)
test_properties, test_labels, test_labels_log = prepare_data(test_set)

# Updated feature lists with new physics features
element_features = [f"{el}_pct" for el in key_elements]
physics_features = [
    'weighted_ionic_radius', 'b_a_ratio', 'c_a_ratio', 'c_b_ratio',
    'li_density', 'inverse_temperature', 'electronegativity_range', 'element_count'
]
num_features = [
    'Space group number', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 
    'temperature', '[Li+]/formula unit', 'Unit cell volume (Angstroms cubed)'
] + element_features + physics_features

# Updated preprocessing pipeline
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

cat_features = ['Chemical Family', 'Structural Family', 'Crystal System']
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown='ignore')
)

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Fit preprocessor and transform data
preprocessor.fit(train_properties)
X_train_prepared = preprocessor.transform(train_properties)
X_val_prepared = preprocessor.transform(val_properties)
X_test_prepared = preprocessor.transform(test_properties)

# Train ensemble of models
models = train_multiple_models(X_train_prepared, train_labels_log, X_val_prepared, val_labels_log)

# Make predictions with ensemble
train_pred_log = ensemble_predict(models, X_train_prepared)
val_pred_log = ensemble_predict(models, X_val_prepared)
test_pred_log = ensemble_predict(models, X_test_prepared)

# Convert back to original scale
train_pred = inverse_log_transform(train_pred_log)
val_pred = inverse_log_transform(val_pred_log)
test_pred = inverse_log_transform(test_pred_log)

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_true_log, y_pred_log):
    # Original scale metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Log scale metrics (often more meaningful for this type of data)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)
    
    return {
        'RMSE': rmse, 'MAE': mae, 'R2': r2,
        'RMSE_log': rmse_log, 'MAE_log': mae_log, 'R2_log': r2_log
    }

"""
# Print results
train_metrics = calculate_metrics(train_labels, train_pred, train_labels_log, train_pred_log)
val_metrics = calculate_metrics(val_labels, val_pred, val_labels_log, val_pred_log)
test_metrics = calculate_metrics(test_labels, test_pred, test_labels_log, test_pred_log)


print("=== ENSEMBLE MODEL RESULTS ===")
print(f"Train - RMSE: {train_metrics['RMSE']:.6f}, R2: {train_metrics['R2']:.4f}, R2_log: {train_metrics['R2_log']:.4f}")
print(f"Val   - RMSE: {val_metrics['RMSE']:.6f}, R2: {val_metrics['R2']:.4f}, R2_log: {val_metrics['R2_log']:.4f}")
print(f"Test  - RMSE: {test_metrics['RMSE']:.6f}, R2: {test_metrics['R2']:.4f}, R2_log: {test_metrics['R2_log']:.4f}")

# Better visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Original scale predictions
ax1.scatter(test_labels, test_pred, alpha=0.6, s=50)
ax1.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'r--', lw=2)
ax1.set_xlabel('True Ionic Conductivity (S/cm)')
ax1.set_ylabel('Predicted Ionic Conductivity (S/cm)')
ax1.set_title('Predicted vs True (Original Scale)')
ax1.set_xscale('log')
ax1.set_yscale('log')

# Log scale predictions
ax2.scatter(test_labels_log, test_pred_log, alpha=0.6, s=50)
ax2.plot([test_labels_log.min(), test_labels_log.max()], [test_labels_log.min(), test_labels_log.max()], 'r--', lw=2)
ax2.set_xlabel('True log10(Ionic Conductivity)')
ax2.set_ylabel('Predicted log10(Ionic Conductivity)')
ax2.set_title('Predicted vs True (Log Scale)')

# Distribution of predictions
ax3.hist(test_pred, bins=50, alpha=0.7, label='Predictions', density=True)
ax3.hist(test_labels, bins=50, alpha=0.7, label='True Values', density=True)
ax3.set_xlabel('Ionic Conductivity (S/cm)')
ax3.set_ylabel('Density')
ax3.set_title('Distribution Comparison')
ax3.set_xscale('log')
ax3.legend()

# Feature importance from XGBoost
feature_names = preprocessor.get_feature_names_out()
importances = models['xgb'].feature_importances_
top_indices = np.argsort(importances)[-20:]

ax4.barh(range(len(top_indices)), importances[top_indices])
ax4.set_yticks(range(len(top_indices)))
ax4.set_yticklabels([feature_names[i] for i in top_indices])
ax4.set_xlabel('Feature Importance')
ax4.set_title('Top 20 Most Important Features')

plt.tight_layout()
plt.show()

# Sample predictions table
output_table = pd.DataFrame({
    "True Value": test_labels[:10],
    "Predicted Value": test_pred[:10],
    "Log10 True": test_labels_log[:10],
    "Log10 Predicted": test_pred_log[:10],
    "Absolute Error": np.abs(test_labels[:10] - test_pred[:10])
})

print("\n=== SAMPLE PREDICTIONS ===")
print(output_table)
"""

to_save = {
    'models': models,
    'preprocessor': preprocessor,
    'features': num_features,
    'cat_features': cat_features
}

with open(r'C:\Users\Sasha\OneDrive\Documents\Projects\FR8\model.pkl', 'wb') as f:
    pickle.dump(to_save, f)

