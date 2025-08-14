#!/usr/bin/env python3
"""
Test the exact same logic on the first few entries to debug the issue
"""

from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from pathlib import Path

def test_first_entries():
    """Test the exact extraction logic on first entries"""
    
    print("🔍 Loading JARVIS-DFT 3D dataset...")
    dft_data = data('dft_3d')
    print(f"   ✅ Loaded {len(dft_data)} entries")
    
    print("\n🧪 Testing extraction logic on first 20 entries:")
    
    paired_entries = []
    
    def clean_value(val):
        if val is None or val == "na" or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    
    for i, entry in enumerate(dft_data[:20]):
        try:
            # Get PBE-like bandgap (OptB88vdW is a PBE-based functional)
            pbe_gap = entry.get("optb88vdw_bandgap", None)
            
            # Try to get high-fidelity bandgaps
            hse_gap = entry.get("hse_gap", None)
            mbj_gap = entry.get("mbj_bandgap", None)  # meta-GGA TBmBJ
            gw_gap = entry.get("gw_bandgap", None)    # GW approximation
            
            # Convert "na" strings to None and ensure numeric values
            pbe_gap = clean_value(pbe_gap)
            hse_gap = clean_value(hse_gap)
            mbj_gap = clean_value(mbj_gap)
            gw_gap = clean_value(gw_gap)
            
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
            
            jid = entry.get('jid', 'unknown')
            formula = entry.get('formula', 'unknown')
            
            print(f"   Entry {i+1}: {jid} ({formula})")
            print(f"      Raw PBE: {repr(entry.get('optb88vdw_bandgap'))}")
            print(f"      Raw HSE: {repr(entry.get('hse_gap'))}")
            print(f"      Raw mBJ: {repr(entry.get('mbj_bandgap'))}")
            print(f"      Cleaned PBE: {pbe_gap}")
            print(f"      Cleaned HSE: {hse_gap}")
            print(f"      Cleaned mBJ: {mbj_gap}")
            print(f"      High gap: {high_gap} ({gap_type})")
            
            # Only proceed if we have both PBE and high-fidelity gaps
            if pbe_gap is not None and high_gap is not None:
                print(f"      ✅ PAIRED DATA FOUND!")
                
                # Get crystal structure
                atoms = Atoms.from_dict(entry["atoms"])
                formula = atoms.composition.reduced_formula
                
                # Calculate correction
                correction = high_gap - pbe_gap
                
                paired_entries.append({
                    "material_id": entry["jid"],
                    "formula": formula,
                    "pbe_bandgap": round(pbe_gap, 3),
                    "hse_bandgap": round(high_gap, 3),
                    "gap_type": gap_type,
                    "correction": round(correction, 3),
                })
            else:
                print(f"      ❌ No paired data (PBE: {pbe_gap is not None}, High: {high_gap is not None})")
            
            print()
                
        except Exception as e:
            print(f"      ❌ Error processing entry: {e}")
            continue
    
    print(f"📊 Results:")
    print(f"   Found {len(paired_entries)} paired entries in first 20")
    
    for entry in paired_entries:
        print(f"   - {entry['material_id']}: PBE={entry['pbe_bandgap']}, {entry['gap_type']}={entry['hse_bandgap']}")

if __name__ == "__main__":
    test_first_entries()