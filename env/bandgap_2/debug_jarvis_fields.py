#!/usr/bin/env python3
"""
Debug script to explore JARVIS dataset field names and find available bandgap data
"""

import json
from jarvis.db.figshare import data

def explore_jarvis_fields():
    """Explore the actual field names in JARVIS dataset"""
    
    print("🔍 Loading JARVIS-DFT 3D dataset...")
    dft_data = data('dft_3d')
    print(f"   ✅ Loaded {len(dft_data)} entries")
    
    # Look at first few entries to understand structure
    print("\n📋 Exploring first 5 entries for available fields:")
    
    all_fields = set()
    bandgap_fields = set()
    
    for i, entry in enumerate(dft_data[:5]):
        print(f"\n--- Entry {i+1} (JID: {entry.get('jid', 'unknown')}) ---")
        
        # Collect all fields
        fields = list(entry.keys())
        all_fields.update(fields)
        
        # Look for bandgap-related fields
        bg_fields = [f for f in fields if 'gap' in f.lower() or 'bandgap' in f.lower()]
        bandgap_fields.update(bg_fields)
        
        print(f"   Total fields: {len(fields)}")
        print(f"   Bandgap-related fields: {bg_fields}")
        
        # Show values for bandgap fields
        for field in bg_fields:
            value = entry.get(field)
            print(f"      {field}: {value}")
    
    print(f"\n📊 Summary across first 5 entries:")
    print(f"   Total unique fields: {len(all_fields)}")
    print(f"   Bandgap-related fields found: {sorted(bandgap_fields)}")
    
    # Now let's sample more entries to find ones with bandgap data
    print(f"\n🔍 Sampling 100 entries to find bandgap data availability:")
    
    field_counts = {}
    sample_entries = []
    
    for i, entry in enumerate(dft_data[:100]):
        for field in bandgap_fields:
            value = entry.get(field)
            if value is not None:
                if field not in field_counts:
                    field_counts[field] = 0
                field_counts[field] += 1
                
                # Save some sample entries with data
                if len(sample_entries) < 3 and field in ['optb88vdw_bandgap', 'hse_gap', 'mbj_bandgap']:
                    sample_entries.append({
                        'jid': entry.get('jid'),
                        'field': field,
                        'value': value,
                        'formula': entry.get('formula', 'unknown')
                    })
    
    print(f"\n📈 Bandgap field availability in first 100 entries:")
    for field, count in sorted(field_counts.items()):
        percentage = (count / 100) * 100
        print(f"   {field}: {count}/100 ({percentage:.1f}%)")
    
    print(f"\n📝 Sample entries with bandgap data:")
    for sample in sample_entries:
        print(f"   {sample['jid']} ({sample['formula']}): {sample['field']} = {sample['value']}")
    
    # Let's also check what the actual field names are by looking at all fields
    print(f"\n📋 All available fields (first 20):")
    sorted_fields = sorted(all_fields)
    for field in sorted_fields[:20]:
        print(f"   {field}")
    
    if len(sorted_fields) > 20:
        print(f"   ... and {len(sorted_fields) - 20} more fields")
    
    return bandgap_fields, field_counts

if __name__ == "__main__":
    explore_jarvis_fields()