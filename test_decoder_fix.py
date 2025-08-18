#!/usr/bin/env python3
"""
Test script to isolate and fix the decoder scale_file issue
"""

import torch
import sys
import os
from pathlib import Path

# Add CDVAE to path
sys.path.append(r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE')

# Set required environment variables
os.environ['PROJECT_ROOT'] = r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE'

try:
    from cdvae.pl_modules.decoder import GemNetTDecoder
    print("✅ Decoder import successful")
    
    # Test 1: Create decoder with scale_file=None (this should fail)
    print("🔧 Testing decoder with scale_file=None...")
    try:
        decoder = GemNetTDecoder(
            hidden_dim=128,
            latent_dim=256,
            max_neighbors=20,
            radius=6.0,
            scale_file=None
        )
        print("✅ Decoder created successfully with scale_file=None")
    except Exception as e:
        print(f"❌ Decoder failed with scale_file=None: {e}")
    
    # Test 2: Create decoder without scale_file parameter
    print("🔧 Testing decoder without scale_file parameter...")
    try:
        decoder = GemNetTDecoder(
            hidden_dim=128,
            latent_dim=256,
            max_neighbors=20,
            radius=6.0
        )
        print("✅ Decoder created successfully without scale_file")
    except Exception as e:
        print(f"❌ Decoder failed without scale_file: {e}")
    
    # Test 3: Create decoder with valid scale_file path
    print("🔧 Testing decoder with valid scale_file...")
    scale_file_path = r'C:\Users\Sasha\repos\RL-electrolyte-design\generator\CDVAE\cdvae\pl_modules\gemnet\gemnet-dT.json'
    try:
        decoder = GemNetTDecoder(
            hidden_dim=128,
            latent_dim=256,
            max_neighbors=20,
            radius=6.0,
            scale_file=scale_file_path
        )
        print("✅ Decoder created successfully with valid scale_file")
    except Exception as e:
        print(f"❌ Decoder failed with valid scale_file: {e}")

except ImportError as e:
    print(f"❌ Import failed: {e}")