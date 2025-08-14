#!/usr/bin/env python3
"""
Debug specific entries to understand the exact values and find paired data
"""

from jarvis.db.figshare import data

def debug_specific_entries():
    """Look at specific entries to understand the data structure"""
    
    print("🔍 Loading JARVIS-DFT 3D dataset...")
    dft_data = data('dft_3d')
    print(f"   ✅ Loaded {len(dft_data)} entries")
    
    # Let's look at the first 50 entries in detail
    print("\n🔍 Examining first 50 entries for paired data:")
    
    paired_count = 0
    pbe_available = 0
    hse_available = 0
    mbj_available = 0
    
    def clean_value(val):
        if val is None or val == "na" or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    
    for i, entry in enumerate(dft_data[:50]):
        jid = entry.get('jid', 'unknown')
        formula = entry.get('formula', 'unknown')
        
        # Get bandgap values
        pbe_gap = clean_value(entry.get("optb88vdw_bandgap"))
        hse_gap = clean_value(entry.get("hse_gap"))
        mbj_gap = clean_value(entry.get("mbj_bandgap"))
        
        # Count availability
        if pbe_gap is not None:
            pbe_available += 1
        if hse_gap is not None:
            hse_available += 1
        if mbj_gap is not None:
            mbj_available += 1
        
        # Check for paired data
        has_pbe = pbe_gap is not None
        has_high = (hse_gap is not None) or (mbj_gap is not None)
        
        if has_pbe and has_high:
            paired_count += 1
            high_type = "HSE" if hse_gap is not None else "mBJ"
            high_val = hse_gap if hse_gap is not None else mbj_gap
            print(f"   ✅ PAIRED: {jid} ({formula})")
            print(f"      PBE: {pbe_gap}, {high_type}: {high_val}")
        
        # Show some examples of what we're seeing
        if i < 10:
            print(f"   Entry {i+1}: {jid} ({formula})")
            print(f"      PBE: {pbe_gap}, HSE: {hse_gap}, mBJ: {mbj_gap}")
    
    print(f"\n📊 Summary of first 50 entries:")
    print(f"   PBE available: {pbe_available}/50 ({pbe_available/50*100:.1f}%)")
    print(f"   HSE available: {hse_available}/50 ({hse_available/50*100:.1f}%)")
    print(f"   mBJ available: {mbj_available}/50 ({mbj_available/50*100:.1f}%)")
    print(f"   Paired entries: {paired_count}/50 ({paired_count/50*100:.1f}%)")
    
    # Let's also check a larger sample to see if paired data exists later
    print(f"\n🔍 Checking every 1000th entry for paired data:")
    
    sample_indices = range(0, len(dft_data), 1000)
    sample_paired = 0
    
    for idx in sample_indices:
        if idx >= len(dft_data):
            break
            
        entry = dft_data[idx]
        jid = entry.get('jid', 'unknown')
        
        pbe_gap = clean_value(entry.get("optb88vdw_bandgap"))
        hse_gap = clean_value(entry.get("hse_gap"))
        mbj_gap = clean_value(entry.get("mbj_bandgap"))
        
        has_pbe = pbe_gap is not None
        has_high = (hse_gap is not None) or (mbj_gap is not None)
        
        if has_pbe and has_high:
            sample_paired += 1
            high_type = "HSE" if hse_gap is not None else "mBJ"
            high_val = hse_gap if hse_gap is not None else mbj_gap
            print(f"   ✅ Index {idx}: {jid} - PBE: {pbe_gap}, {high_type}: {high_val}")
    
    print(f"\n📊 Sample results:")
    print(f"   Sampled {len(list(sample_indices))} entries")
    print(f"   Found {sample_paired} paired entries in sample")
    
    if sample_paired == 0:
        print("\n⚠️  No paired data found in sample. Let's check raw values:")
        # Show some raw values to understand the data better
        for i in range(5):
            entry = dft_data[i]
            print(f"   Entry {i}: {entry.get('jid')}")
            print(f"      Raw optb88vdw_bandgap: {repr(entry.get('optb88vdw_bandgap'))}")
            print(f"      Raw hse_gap: {repr(entry.get('hse_gap'))}")
            print(f"      Raw mbj_bandgap: {repr(entry.get('mbj_bandgap'))}")

if __name__ == "__main__":
    debug_specific_entries()