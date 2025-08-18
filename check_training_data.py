#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Read the id_prop.csv to see the range of ionic conductivity values
id_prop_path = r'C:\Users\Sasha\repos\RL-electrolyte-design\env\property_predictions\CIF_OBELiX\cifs\id_prop.csv'
df = pd.read_csv(id_prop_path, header=None, names=['cif_id', 'ionic_conductivity'])

print('Ionic conductivity values in training data:')
print(f'Min: {df["ionic_conductivity"].min():.2e}')
print(f'Max: {df["ionic_conductivity"].max():.2e}')
print(f'Mean: {df["ionic_conductivity"].mean():.2e}')
print(f'Median: {df["ionic_conductivity"].median():.2e}')

print(f'\nFirst 10 values:')
for i in range(10):
    val = df.iloc[i]['ionic_conductivity']
    print(f'{df.iloc[i]["cif_id"]}: {val:.2e}')

# Check if any values are negative
negative_count = (df['ionic_conductivity'] < 0).sum()
print(f'\nNegative values in training data: {negative_count}/{len(df)}')
print(f'Range: {df["ionic_conductivity"].min():.6f} to {df["ionic_conductivity"].max():.6f}')