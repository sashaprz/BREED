#!/usr/bin/env python3
"""
Generate large synthetic paired PBE/HSE bandgap dataset for solid-state electrolytes
"""

import csv
import json
import os
import sys
import random
from pathlib import Path

def create_large_synthetic_dataset(num_entries=5000):
    """Create large synthetic paired PBE/HSE bandgap dataset"""
    print(f"🔧 Generating {num_entries} synthetic paired bandgap entries...")
    
    # Define material class templates with correction factors based on literature
    material_templates = {
        "phosphate": {
            "correction_range": (0.8, 1.0),
            "pbe_range": (1.5, 4.0),
            "elements": {
                "Li": (1, 6), "Na": (1, 4), "K": (1, 2),
                "Ti": (1, 3), "Zr": (1, 2), "Al": (1, 2), "Ga": (1, 2),
                "P": (1, 4), "O": (8, 16)
            }
        },
        "garnet": {
            "correction_range": (1.0, 1.4),
            "pbe_range": (3.0, 6.0),
            "elements": {
                "Li": (5, 8), "Na": (3, 6),
                "La": (2, 4), "Y": (2, 4), "Gd": (1, 3), "Lu": (1, 3),
                "Zr": (1, 3), "Hf": (1, 2), "Ta": (1, 2),
                "O": (10, 14)
            }
        },
        "argyrodite": {
            "correction_range": (0.6, 0.8),
            "pbe_range": (2.5, 4.5),
            "elements": {
                "Li": (4, 8), "Na": (3, 6),
                "P": (1, 2), "As": (1, 2), "Sb": (1, 2),
                "S": (4, 8), "Se": (3, 6),
                "Cl": (1, 3), "Br": (1, 3), "I": (1, 2)
            }
        },
        "halide": {
            "correction_range": (0.8, 1.1),
            "pbe_range": (3.5, 6.5),
            "elements": {
                "Li": (1, 4), "Na": (1, 3),
                "Y": (1, 2), "In": (1, 2), "Sc": (1, 2), "Er": (1, 2),
                "Zr": (1, 2), "Hf": (1, 2),
                "Cl": (3, 8), "Br": (3, 7), "I": (2, 6), "F": (2, 5)
            }
        },
        "oxide": {
            "correction_range": (0.9, 1.3),
            "pbe_range": (4.0, 7.0),
            "elements": {
                "Li": (1, 6), "Na": (1, 4),
                "Al": (1, 3), "Ga": (1, 2), "In": (1, 2),
                "Si": (1, 2), "Ge": (1, 2), "Sn": (1, 2),
                "Ti": (1, 2), "Zr": (1, 2), "Nb": (1, 2),
                "O": (3, 12)
            }
        },
        "perovskite": {
            "correction_range": (0.9, 1.2),
            "pbe_range": (2.8, 5.5),
            "elements": {
                "Li": (1, 4), "Na": (1, 3),
                "La": (1, 4), "Sr": (1, 3), "Ba": (1, 2), "Ca": (1, 2),
                "Ti": (1, 2), "Zr": (1, 2), "Nb": (1, 3), "Ta": (1, 2),
                "O": (6, 15)
            }
        },
        "chalcogenide": {
            "correction_range": (0.5, 0.8),
            "pbe_range": (1.8, 4.2),
            "elements": {
                "Li": (1, 6), "Na": (1, 4),
                "Mg": (1, 2), "Ca": (1, 2), "Zn": (1, 2),
                "Al": (1, 2), "Ga": (1, 2), "In": (1, 2),
                "S": (1, 8), "Se": (1, 6), "Te": (1, 4)
            }
        },
        "thiophosphate": {
            "correction_range": (0.6, 0.9),
            "pbe_range": (2.2, 4.8),
            "elements": {
                "Li": (4, 12), "Na": (3, 8),
                "P": (1, 3), "As": (1, 2),
                "Ge": (1, 2), "Sn": (1, 2), "Si": (1, 2),
                "S": (6, 16), "Se": (4, 12)
            }
        },
        "nitride": {
            "correction_range": (0.7, 1.0),
            "pbe_range": (2.0, 5.0),
            "elements": {
                "Li": (1, 6), "Na": (1, 4),
                "Al": (1, 2), "Ga": (1, 2), "In": (1, 2),
                "Si": (1, 2), "Ge": (1, 2),
                "N": (1, 6), "O": (0, 4)
            }
        },
        "fluoride": {
            "correction_range": (1.1, 1.5),
            "pbe_range": (5.0, 8.0),
            "elements": {
                "Li": (1, 4), "Na": (1, 3),
                "Mg": (1, 2), "Ca": (1, 2), "Ba": (1, 2),
                "Al": (1, 2), "Y": (1, 2), "La": (1, 2),
                "F": (2, 8), "O": (0, 3)
            }
        },
        "borohydride": {
            "correction_range": (0.8, 1.1),
            "pbe_range": (3.2, 6.0),
            "elements": {
                "Li": (1, 4), "Na": (1, 3), "Mg": (1, 2),
                "Ca": (1, 2), "Al": (1, 2), "Zn": (1, 2),
                "B": (1, 4), "H": (4, 16)
            }
        },
        "antiperovskite": {
            "correction_range": (0.7, 1.0),
            "pbe_range": (2.5, 5.0),
            "elements": {
                "Li": (2, 6), "Na": (2, 4),
                "O": (1, 2), "N": (1, 2), "S": (1, 2),
                "Cl": (1, 2), "Br": (1, 2)
            }
        }
    }
    
    synthetic_data = []
    
    # Progress tracking
    progress_interval = max(1, num_entries // 20)
    
    for i in range(num_entries):
        if i % progress_interval == 0:
            progress = (i / num_entries) * 100
            print(f"   Progress: {progress:.1f}% ({i}/{num_entries})")
        
        # Select random material class
        mat_class = random.choice(list(material_templates.keys()))
        template = material_templates[mat_class]
        
        # Generate composition
        composition = {}
        elements = template["elements"]
        
        # Select 2-5 elements for the composition
        num_elements = random.randint(2, min(5, len(elements)))
        selected_elements = random.sample(list(elements.keys()), num_elements)
        
        for element in selected_elements:
            min_count, max_count = elements[element]
            composition[element] = random.randint(min_count, max_count)
        
        # Generate formula string
        formula_parts = []
        for element in sorted(composition.keys()):
            count = composition[element]
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")
        formula = "".join(formula_parts)
        
        # Generate PBE bandgap with some noise
        pbe_min, pbe_max = template["pbe_range"]
        pbe_bandgap = round(random.uniform(pbe_min, pbe_max), 3)
        
        # Generate HSE correction with realistic variation
        corr_min, corr_max = template["correction_range"]
        correction = round(random.uniform(corr_min, corr_max), 3)
        hse_bandgap = round(pbe_bandgap + correction, 3)
        
        # Add some additional properties for realism
        density = round(random.uniform(2.0, 8.0), 2)  # g/cm³
        formation_energy = round(random.uniform(-3.0, 0.5), 3)  # eV/atom
        
        synthetic_data.append({
            "material_id": f"synthetic_{i+1:06d}",
            "formula": formula,
            "pbe_bandgap": pbe_bandgap,
            "hse_bandgap": hse_bandgap,
            "material_class": mat_class,
            "correction": correction,
            "density": density,
            "formation_energy": formation_energy
        })
    
    print(f"   ✅ Generated {len(synthetic_data)} synthetic entries")
    return synthetic_data

def save_large_dataset(synthetic_data, output_dir="large_paired_dataset"):
    """Save large paired bandgap dataset to files"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"💾 Saving {len(synthetic_data)} entries to {output_dir}/...")
    
    # Save CSV
    csv_file = output_path / "large_paired_bandgaps.csv"
    fieldnames = ["material_id", "formula", "pbe_bandgap", "hse_bandgap", 
                  "material_class", "correction", "density", "formation_energy"]
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(synthetic_data)
    
    # Save JSON (might be large)
    json_file = output_path / "large_paired_bandgaps.json"
    with open(json_file, "w") as f:
        json.dump(synthetic_data, f, indent=2)
    
    # Save summary statistics
    stats_file = output_path / "dataset_statistics.json"
    stats = analyze_dataset(synthetic_data)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"   📄 CSV: {csv_file}")
    print(f"   📄 JSON: {json_file}")
    print(f"   📊 Stats: {stats_file}")
    
    return csv_file, json_file, stats_file

def analyze_dataset(synthetic_data):
    """Analyze the synthetic dataset"""
    print(f"\n📊 Analyzing {len(synthetic_data)} entries...")
    
    # Calculate correction statistics
    corrections = [entry["correction"] for entry in synthetic_data]
    pbe_gaps = [entry["pbe_bandgap"] for entry in synthetic_data]
    hse_gaps = [entry["hse_bandgap"] for entry in synthetic_data]
    
    import statistics
    
    stats = {
        "total_entries": len(synthetic_data),
        "corrections": {
            "mean": round(statistics.mean(corrections), 3),
            "median": round(statistics.median(corrections), 3),
            "min": round(min(corrections), 3),
            "max": round(max(corrections), 3),
            "std_dev": round(statistics.stdev(corrections), 3)
        },
        "pbe_bandgaps": {
            "mean": round(statistics.mean(pbe_gaps), 3),
            "median": round(statistics.median(pbe_gaps), 3),
            "min": round(min(pbe_gaps), 3),
            "max": round(max(pbe_gaps), 3),
            "std_dev": round(statistics.stdev(pbe_gaps), 3)
        },
        "hse_bandgaps": {
            "mean": round(statistics.mean(hse_gaps), 3),
            "median": round(statistics.median(hse_gaps), 3),
            "min": round(min(hse_gaps), 3),
            "max": round(max(hse_gaps), 3),
            "std_dev": round(statistics.stdev(hse_gaps), 3)
        }
    }
    
    # Material class distribution
    class_counts = {}
    for entry in synthetic_data:
        mat_class = entry["material_class"]
        class_counts[mat_class] = class_counts.get(mat_class, 0) + 1
    
    stats["material_classes"] = class_counts
    
    print(f"   📈 Bandgap corrections (HSE - PBE):")
    print(f"      Mean: {stats['corrections']['mean']:.3f} eV")
    print(f"      Range: {stats['corrections']['min']:.3f} - {stats['corrections']['max']:.3f} eV")
    print(f"      Std Dev: {stats['corrections']['std_dev']:.3f} eV")
    
    print(f"   🏷️ Material class distribution:")
    for mat_class, count in sorted(class_counts.items()):
        percentage = (count / len(synthetic_data)) * 100
        print(f"      {mat_class}: {count} ({percentage:.1f}%)")
    
    return stats

def main():
    """Main function to generate large paired bandgap dataset"""
    
    print("🚀 Generating Large Synthetic Paired PBE/HSE Bandgap Dataset")
    print("=" * 60)
    
    # Get number of entries from command line or use default
    num_entries = 5000  # Default
    if len(sys.argv) > 1:
        try:
            num_entries = int(sys.argv[1])
            if num_entries <= 0:
                raise ValueError("Number must be positive")
        except ValueError as e:
            print(f"❌ Invalid number: {e}")
            print("Usage: python generate_large_paired_dataset.py [number_of_entries]")
            print("Using default: 5000 entries")
            num_entries = 5000
    
    print(f"📊 Generating {num_entries} synthetic entries...")
    
    # Generate synthetic data
    synthetic_data = create_large_synthetic_dataset(num_entries)
    
    # Save dataset
    csv_file, json_file, stats_file = save_large_dataset(synthetic_data)
    
    # Analyze dataset
    stats = analyze_dataset(synthetic_data)
    
    print(f"\n🎉 Large paired bandgap dataset generation completed!")
    print(f"   📁 Output directory: large_paired_dataset/")
    print(f"   📊 Total entries: {len(synthetic_data)}")
    print(f"   💾 Files created: CSV, JSON, and statistics")
    
    return synthetic_data, stats

if __name__ == "__main__":
    results = main()