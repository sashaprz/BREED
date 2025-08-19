#!/usr/bin/env python3
"""
Script to analyze the range of ionic conductivity values in the obelix dataset
"""
import pickle
import numpy as np

def analyze_ionic_conductivity():
    """Analyze the ionic conductivity values in the obelix dataset"""
    
    # Load the dataset
    print("Loading obelix dataset...")
    with open('data/ionic_conductivity_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"Dataset contains {len(data)} entries")
    
    # Extract ionic conductivity values
    conductivity_values = []
    for entry in data:
        if 'ionic_conductivity' in entry:
            conductivity_values.append(entry['ionic_conductivity'])
    
    # Convert to numpy array for analysis
    conductivity_array = np.array(conductivity_values)
    
    # Basic statistics
    print("\n" + "="*60)
    print("IONIC CONDUCTIVITY ANALYSIS - OBELIX DATASET")
    print("="*60)
    
    print(f"Total number of entries: {len(conductivity_values)}")
    print(f"Data type: {type(conductivity_array[0])}")
    
    print(f"\nBasic Statistics:")
    print(f"  Minimum value: {np.min(conductivity_array):.2e} S/cm")
    print(f"  Maximum value: {np.max(conductivity_array):.2e} S/cm")
    print(f"  Mean value: {np.mean(conductivity_array):.2e} S/cm")
    print(f"  Median value: {np.median(conductivity_array):.2e} S/cm")
    print(f"  Standard deviation: {np.std(conductivity_array):.2e} S/cm")
    
    # Range analysis
    print(f"\nRange Analysis:")
    print(f"  Range (max - min): {np.max(conductivity_array) - np.min(conductivity_array):.2e} S/cm")
    print(f"  Log10 range: {np.log10(np.max(conductivity_array)) - np.log10(np.min(conductivity_array)):.2f} orders of magnitude")
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(conductivity_array, p)
        print(f"  {p:2d}th percentile: {value:.2e} S/cm")
    
    # Distribution analysis
    print(f"\nDistribution Analysis:")
    
    # Count values in different ranges
    ranges = [
        (0, 1e-12, "≤ 1e-12"),
        (1e-12, 1e-10, "1e-12 to 1e-10"),
        (1e-10, 1e-8, "1e-10 to 1e-8"),
        (1e-8, 1e-6, "1e-8 to 1e-6"),
        (1e-6, 1e-4, "1e-6 to 1e-4"),
        (1e-4, 1e-2, "1e-4 to 1e-2"),
        (1e-2, 1, "1e-2 to 1"),
        (1, float('inf'), "> 1")
    ]
    
    for min_val, max_val, label in ranges:
        count = np.sum((conductivity_array > min_val) & (conductivity_array <= max_val))
        percentage = (count / len(conductivity_array)) * 100
        if count > 0:
            print(f"  {label:15s}: {count:4d} entries ({percentage:5.1f}%)")
    
    # Check for zero or negative values
    zero_count = np.sum(conductivity_array == 0)
    negative_count = np.sum(conductivity_array < 0)
    
    if zero_count > 0:
        print(f"\nWarning: Found {zero_count} entries with zero conductivity")
    if negative_count > 0:
        print(f"Warning: Found {negative_count} entries with negative conductivity")
    
    # Show some example entries
    print(f"\nExample entries (first 10):")
    for i in range(min(10, len(data))):
        entry = data[i]
        print(f"  {entry['material_id']:15s}: {entry['ionic_conductivity']:.2e} S/cm")
    
    # Create sorted list for easier viewing
    material_conductivity = [(entry['material_id'], entry['ionic_conductivity']) for entry in data]
    material_conductivity.sort(key=lambda x: x[1])  # Sort by conductivity
    
    print(f"\nTop 10 highest conductivity materials:")
    for material_id, conductivity in material_conductivity[-10:]:
        print(f"  {material_id:20s}: {conductivity:.2e} S/cm")
    
    print(f"\nTop 10 lowest conductivity materials:")
    for material_id, conductivity in material_conductivity[:10]:
        print(f"  {material_id:20s}: {conductivity:.2e} S/cm")
    
    # Save summary to CSV
    output_file = 'obelix_conductivity_analysis.csv'
    with open(output_file, 'w') as f:
        f.write("material_id,ionic_conductivity\n")
        for material_id, conductivity in material_conductivity:
            f.write(f"{material_id},{conductivity:.6e}\n")
    print(f"\nFull analysis saved to: {output_file}")
    
    return {
        'min': np.min(conductivity_array),
        'max': np.max(conductivity_array),
        'mean': np.mean(conductivity_array),
        'median': np.median(conductivity_array),
        'std': np.std(conductivity_array),
        'count': len(conductivity_array),
        'log_range': np.log10(np.max(conductivity_array)) - np.log10(np.min(conductivity_array))
    }

if __name__ == "__main__":
    results = analyze_ionic_conductivity()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"The obelix dataset contains {results['count']} entries with ionic conductivity values")
    print(f"ranging from {results['min']:.2e} to {results['max']:.2e} S/cm")
    print(f"spanning {results['log_range']:.1f} orders of magnitude.")