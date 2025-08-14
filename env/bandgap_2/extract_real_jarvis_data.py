#!/usr/bin/env python3
"""
Extract real paired PBE/HSE bandgap data from JARVIS-DFT dataset.
This script downloads the JARVIS-DFT 3D dataset and extracts materials
with both PBE-like and high-fidelity (HSE/mBJ/GW) bandgap calculations.
"""

import csv
import json
import os
import sys
from pathlib import Path
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms

def download_jarvis_data():
    """Download JARVIS-DFT 3D dataset"""
    print("📥 Downloading JARVIS-DFT 3D dataset...")
    print("   This may take several minutes for the first download...")
    
    try:
        dft_data = data('dft_3d')
        print(f"   ✅ Downloaded {len(dft_data)} entries from JARVIS-DFT")
        return dft_data
    except Exception as e:
        print(f"   ❌ Error downloading JARVIS data: {e}")
        return None

def extract_paired_bandgaps(dft_data, max_entries=None):
    """Extract materials with paired PBE and high-fidelity bandgaps"""
    
    print("🔍 Filtering for paired PBE/high-fidelity bandgap data...")
    
    paired_entries = []
    processed = 0
    
    # Create output folder for CIFs
    cif_dir = Path("jarvis_structures_cif")
    cif_dir.mkdir(exist_ok=True)
    
    for i, entry in enumerate(dft_data):
        processed += 1
        
        # Progress update
        if processed % 1000 == 0:
            print(f"   Processed {processed}/{len(dft_data)} entries... Found {len(paired_entries)} paired entries")
        
        # Stop if max_entries reached
        if max_entries and len(paired_entries) >= max_entries:
            print(f"   Reached maximum of {max_entries} paired entries")
            break
        
        try:
            # Get PBE-like bandgap (OptB88vdW is a PBE-based functional)
            pbe_gap = entry.get("optb88vdw_bandgap", None)
            
            # Try to get high-fidelity bandgaps
            hse_gap = entry.get("hse_gap", None)
            mbj_gap = entry.get("mbj_bandgap", None)  # meta-GGA TBmBJ
            gw_gap = entry.get("gw_bandgap", None)    # GW approximation
            
            # Choose the first available high-fidelity gap
            high_gap = None
            gap_type = None
            
            if hse_gap is not None:
                high_gap = hse_gap
                gap_type = "HSE"
            elif mbj_gap is not None:
                high_gap = mbj_gap
                gap_type = "mBJ"
            elif gw_gap is not None:
                high_gap = gw_gap
                gap_type = "GW"
            
            # Only proceed if we have both PBE and high-fidelity gaps
            if pbe_gap is not None and high_gap is not None:
                
                # Get crystal structure
                atoms = Atoms.from_dict(entry["atoms"])
                formula = atoms.composition.reduced_formula
                
                # Generate CIF file
                cif_str = atoms.write_cif()
                cif_filename = f"{entry['jid']}.cif"
                cif_path = cif_dir / cif_filename
                
                # Save CIF file
                with open(cif_path, "w") as f:
                    f.write(cif_str)
                
                # Calculate correction
                correction = high_gap - pbe_gap
                
                paired_entries.append({
                    "material_id": entry["jid"],
                    "formula": formula,
                    "cif_path": str(cif_path),
                    "pbe_bandgap": round(pbe_gap, 3),
                    "hse_bandgap": round(high_gap, 3),
                    "gap_type": gap_type,
                    "correction": round(correction, 3),
                    "formation_energy": entry.get("formation_energy_peratom", None),
                    "total_energy": entry.get("total_energy", None),
                    "spacegroup": entry.get("spg_symbol", None)
                })
                
        except Exception as e:
            # Skip problematic entries
            continue
    
    print(f"✅ Found {len(paired_entries)} materials with paired bandgap data")
    return paired_entries

def analyze_bandgap_data(paired_entries):
    """Analyze the extracted paired bandgap data"""
    
    print(f"\n📊 Analysis of {len(paired_entries)} paired entries:")
    
    # Bandgap statistics
    pbe_gaps = [entry["pbe_bandgap"] for entry in paired_entries]
    hse_gaps = [entry["hse_bandgap"] for entry in paired_entries]
    corrections = [entry["correction"] for entry in paired_entries]
    
    import statistics
    
    print(f"   📈 PBE Bandgaps:")
    print(f"      Mean: {statistics.mean(pbe_gaps):.3f} eV")
    print(f"      Range: {min(pbe_gaps):.3f} - {max(pbe_gaps):.3f} eV")
    
    print(f"   📈 High-fidelity Bandgaps:")
    print(f"      Mean: {statistics.mean(hse_gaps):.3f} eV")
    print(f"      Range: {min(hse_gaps):.3f} - {max(hse_gaps):.3f} eV")
    
    print(f"   📈 Bandgap Corrections (High - PBE):")
    print(f"      Mean: {statistics.mean(corrections):.3f} eV")
    print(f"      Median: {statistics.median(corrections):.3f} eV")
    print(f"      Range: {min(corrections):.3f} - {max(corrections):.3f} eV")
    print(f"      Std Dev: {statistics.stdev(corrections):.3f} eV")
    
    # Gap type distribution
    gap_types = {}
    for entry in paired_entries:
        gap_type = entry["gap_type"]
        gap_types[gap_type] = gap_types.get(gap_type, 0) + 1
    
    print(f"   🏷️ High-fidelity method distribution:")
    for gap_type, count in sorted(gap_types.items()):
        percentage = (count / len(paired_entries)) * 100
        print(f"      {gap_type}: {count} ({percentage:.1f}%)")
    
    # Element analysis
    all_elements = set()
    for entry in paired_entries:
        # Extract elements from formula
        formula = entry["formula"]
        import re
        elements = re.findall(r'[A-Z][a-z]?', formula)
        all_elements.update(elements)
    
    print(f"   🧪 Unique elements found: {len(all_elements)}")
    print(f"      Elements: {', '.join(sorted(all_elements))}")

def save_paired_data(paired_entries, output_dir="jarvis_paired_data"):
    """Save paired bandgap data to files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save CSV
    csv_file = output_path / "jarvis_paired_bandgaps.csv"
    fieldnames = [
        "material_id", "formula", "cif_path", "pbe_bandgap", 
        "hse_bandgap", "gap_type", "correction", "formation_energy", 
        "total_energy", "spacegroup"
    ]
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_entries)
    
    # Save JSON
    json_file = output_path / "jarvis_paired_bandgaps.json"
    with open(json_file, "w") as f:
        json.dump(paired_entries, f, indent=2)
    
    # Save statistics
    stats_file = output_path / "extraction_statistics.json"
    
    import statistics
    corrections = [entry["correction"] for entry in paired_entries]
    pbe_gaps = [entry["pbe_bandgap"] for entry in paired_entries]
    hse_gaps = [entry["hse_bandgap"] for entry in paired_entries]
    
    gap_types = {}
    for entry in paired_entries:
        gap_type = entry["gap_type"]
        gap_types[gap_type] = gap_types.get(gap_type, 0) + 1
    
    stats = {
        "total_entries": len(paired_entries),
        "pbe_bandgap_stats": {
            "mean": round(statistics.mean(pbe_gaps), 3),
            "min": round(min(pbe_gaps), 3),
            "max": round(max(pbe_gaps), 3),
            "std_dev": round(statistics.stdev(pbe_gaps), 3)
        },
        "hse_bandgap_stats": {
            "mean": round(statistics.mean(hse_gaps), 3),
            "min": round(min(hse_gaps), 3),
            "max": round(max(hse_gaps), 3),
            "std_dev": round(statistics.stdev(hse_gaps), 3)
        },
        "correction_stats": {
            "mean": round(statistics.mean(corrections), 3),
            "median": round(statistics.median(corrections), 3),
            "min": round(min(corrections), 3),
            "max": round(max(corrections), 3),
            "std_dev": round(statistics.stdev(corrections), 3)
        },
        "gap_type_distribution": gap_types
    }
    
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n💾 Saved data to:")
    print(f"   📄 CSV: {csv_file}")
    print(f"   📄 JSON: {json_file}")
    print(f"   📊 Statistics: {stats_file}")
    print(f"   📁 CIF structures: jarvis_structures_cif/")
    
    return csv_file, json_file, stats_file

def main():
    """Main function to extract JARVIS paired bandgap data"""
    
    print("🚀 JARVIS Paired Bandgap Data Extraction")
    print("=" * 50)
    
    # Parse command line arguments
    max_entries = None
    if len(sys.argv) > 1:
        try:
            max_entries = int(sys.argv[1])
            print(f"   Limiting extraction to {max_entries} paired entries")
        except ValueError:
            print(f"   Invalid number: {sys.argv[1]}, extracting all available data")
    
    # Download JARVIS data
    dft_data = download_jarvis_data()
    if not dft_data:
        print("❌ Failed to download JARVIS data")
        return None
    
    # Extract paired bandgap data
    paired_entries = extract_paired_bandgaps(dft_data, max_entries)
    
    if not paired_entries:
        print("❌ No paired bandgap data found")
        return None
    
    # Analyze the data
    analyze_bandgap_data(paired_entries)
    
    # Save the data
    csv_file, json_file, stats_file = save_paired_data(paired_entries)
    
    print(f"\n🎉 JARVIS paired bandgap data extraction completed!")
    print(f"   📊 Total paired entries: {len(paired_entries)}")
    print(f"   📁 Output directory: jarvis_paired_data/")
    print(f"   🧪 CIF structures: jarvis_structures_cif/")
    
    return paired_entries

if __name__ == "__main__":
    results = main()