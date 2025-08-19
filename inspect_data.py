#!/usr/bin/env python3
"""
Quick script to inspect the data structure in the pickle file
"""
import pickle

# Load and inspect the data
with open('data/ionic_conductivity_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Data type: {type(data)}")
print(f"Data length: {len(data)}")

if isinstance(data, list) and len(data) > 0:
    print(f"First item type: {type(data[0])}")
    if isinstance(data[0], dict):
        print(f"Keys in first item: {list(data[0].keys())}")
        for key, value in data[0].items():
            print(f"  {key}: {type(value)} - {str(value)[:100]}...")