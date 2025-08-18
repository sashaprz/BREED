"""
Test script for OBELiX data preparation functionality

This script creates sample data and tests the preparation pipeline to ensure
everything works correctly before processing real datasets.
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Import the preparation script
from prepare_obelix_data import OBELiXDataPreparator

def create_sample_cif_content(material_id: str, composition: str) -> str:
    """Create a simple CIF file content for testing"""
    
    # Simple CIF templates for different compositions
    cif_templates = {
        'Li2O': f"""
data_{material_id}
_chemical_formula_sum 'Li2 O1'
_cell_length_a 4.611
_cell_length_b 4.611
_cell_length_c 4.611
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_IT_number 225
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 0.25 0.25 0.25
O1 0.0 0.0 0.0
""",
        'LiCoO2': f"""
data_{material_id}
_chemical_formula_sum 'Li1 Co1 O2'
_cell_length_a 2.816
_cell_length_b 2.816
_cell_length_c 14.051
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 120.0
_space_group_IT_number 166
_symmetry_space_group_name_H-M 'R -3 m'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 0.0 0.0 0.0
Co1 0.0 0.0 0.5
O1 0.0 0.0 0.26
""",
        'Li3PO4': f"""
data_{material_id}
_chemical_formula_sum 'Li3 P1 O4'
_cell_length_a 10.493
_cell_length_b 6.115
_cell_length_c 4.856
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_IT_number 62
_symmetry_space_group_name_H-M 'P n m a'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 0.0 0.0 0.0
Li2 0.333 0.25 0.333
Li3 0.667 0.75 0.667
P1 0.095 0.25 0.418
O1 0.097 0.25 0.743
O2 0.118 0.047 0.282
O3 0.165 0.25 0.286
O4 0.045 0.25 0.286
"""
    }
    
    # Return template or create a generic one
    if composition in cif_templates:
        return cif_templates[composition].strip()
    else:
        return f"""
data_{material_id}
_chemical_formula_sum '{composition}'
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
X1 0.0 0.0 0.0
""".strip()

def create_sample_data(temp_dir: str) -> tuple:
    """Create sample OBELiX Excel file and CIF files for testing"""
    
    # Sample OBELiX data
    sample_data = {
        'ID': ['sample_001', 'sample_002', 'sample_003', 'sample_004', 'sample_005'],
        'Composition': ['Li2O', 'LiCoO2', 'Li3PO4', 'NaCl', 'MgO'],
        'Ionic conductivity (S cm-1)': [1.5e-5, 3.2e-7, 8.1e-6, 1.2e-8, 5.5e-9],
        'Space group number': [225, 166, 62, 225, 225],
        'Temperature (K)': [300, 300, 300, 300, 300],
        'Reference': ['Test1', 'Test2', 'Test3', 'Test4', 'Test5']
    }
    
    # Add some low-quality data to test filtering
    sample_data['ID'].extend(['bad_001', 'bad_002'])
    sample_data['Composition'].extend(['BadComp1', 'BadComp2'])
    sample_data['Ionic conductivity (S cm-1)'].extend([1e-15, np.nan])  # Below threshold and NaN
    sample_data['Space group number'].extend([1, 1])
    sample_data['Temperature (K)'].extend([300, 300])
    sample_data['Reference'].extend(['BadTest1', 'BadTest2'])
    
    # Create Excel file
    df = pd.DataFrame(sample_data)
    excel_path = os.path.join(temp_dir, 'sample_obelix.xlsx')
    df.to_excel(excel_path, index=False)
    
    # Create CIF files directory
    cif_dir = os.path.join(temp_dir, 'cifs')
    os.makedirs(cif_dir, exist_ok=True)
    
    # Create CIF files (only for the good samples)
    cif_files = []
    for i, (material_id, composition) in enumerate(zip(sample_data['ID'][:5], sample_data['Composition'][:5])):
        cif_content = create_sample_cif_content(material_id, composition)
        cif_path = os.path.join(cif_dir, f'{material_id}.cif')
        
        with open(cif_path, 'w') as f:
            f.write(cif_content)
        
        cif_files.append(cif_path)
    
    return excel_path, cif_dir, cif_files

def test_basic_functionality():
    """Test basic data preparation functionality"""
    print("=== Testing Basic Functionality ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create sample data
            excel_path, cif_dir, cif_files = create_sample_data(temp_dir)
            print(f"✅ Created sample data in {temp_dir}")
            print(f"   - Excel file: {excel_path}")
            print(f"   - CIF directory: {cif_dir}")
            print(f"   - CIF files: {len(cif_files)}")
            
            # Configure preparator
            config = {
                'output_dir': os.path.join(temp_dir, 'output'),
                'conductivity_threshold': 1e-12,
                'conductivity_column': 'Ionic conductivity (S cm-1)',
                'matching_strategy': 'id'
            }
            
            # Initialize preparator
            preparator = OBELiXDataPreparator(config)
            print("✅ Initialized OBELiXDataPreparator")
            
            # Test Excel loading
            df = preparator.load_obelix_excel(excel_path)
            print(f"✅ Loaded Excel file: {len(preparator.obelix_data)} valid entries")
            
            # Test CIF parsing
            cif_metadata = preparator.parse_cif_files(cif_dir)
            print(f"✅ Parsed CIF files: {len(cif_metadata)} valid structures")
            
            # Test matching
            matches = preparator.match_cifs_to_obelix('id')
            print(f"✅ Matched CIF-OBELiX pairs: {len(matches)}")
            
            # Test dataset creation
            dataset = preparator.create_enhanced_cgcnn_dataset()
            print(f"✅ Created Enhanced CGCNN dataset: {len(dataset)} entries")
            
            # Test saving
            output_path = os.path.join(temp_dir, 'output', 'test_dataset.pkl')
            preparator.save_dataset(output_path, 'pickle')
            print(f"✅ Saved dataset to {output_path}")
            
            # Test CGCNN format saving
            cgcnn_dir = os.path.join(temp_dir, 'output', 'cgcnn_format')
            preparator.save_cgcnn_format(cgcnn_dir)
            print(f"✅ Saved CGCNN format to {cgcnn_dir}")
            
            # Print statistics
            preparator.print_statistics()
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_matching_strategies():
    """Test different matching strategies"""
    print("\n=== Testing Matching Strategies ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create sample data
            excel_path, cif_dir, cif_files = create_sample_data(temp_dir)
            
            strategies = ['id', 'composition_spacegroup', 'composition_only']
            results = {}
            
            for strategy in strategies:
                print(f"\nTesting strategy: {strategy}")
                
                config = {
                    'output_dir': os.path.join(temp_dir, f'output_{strategy}'),
                    'conductivity_threshold': 1e-12,
                    'matching_strategy': strategy
                }
                
                preparator = OBELiXDataPreparator(config)
                preparator.load_obelix_excel(excel_path)
                preparator.parse_cif_files(cif_dir)
                matches = preparator.match_cifs_to_obelix(strategy)
                
                results[strategy] = len(matches)
                print(f"  Matches found: {len(matches)}")
            
            print(f"\n✅ Strategy comparison:")
            for strategy, count in results.items():
                print(f"  {strategy}: {count} matches")
            
            return True
            
        except Exception as e:
            print(f"❌ Matching strategy test failed: {e}")
            return False

def test_data_filtering():
    """Test data filtering functionality"""
    print("\n=== Testing Data Filtering ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create sample data with various quality levels
            excel_path, cif_dir, cif_files = create_sample_data(temp_dir)
            
            # Test different thresholds
            thresholds = [1e-15, 1e-12, 1e-8, 1e-6]
            
            for threshold in thresholds:
                print(f"\nTesting threshold: {threshold}")
                
                config = {
                    'output_dir': os.path.join(temp_dir, f'output_thresh_{threshold}'),
                    'conductivity_threshold': threshold,
                    'matching_strategy': 'id'
                }
                
                preparator = OBELiXDataPreparator(config)
                preparator.load_obelix_excel(excel_path)
                
                print(f"  Valid entries after filtering: {len(preparator.obelix_data)}")
            
            print("✅ Data filtering test completed")
            return True
            
        except Exception as e:
            print(f"❌ Data filtering test failed: {e}")
            return False

def test_output_formats():
    """Test different output formats"""
    print("\n=== Testing Output Formats ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create sample data
            excel_path, cif_dir, cif_files = create_sample_data(temp_dir)
            
            config = {
                'output_dir': os.path.join(temp_dir, 'output_formats'),
                'conductivity_threshold': 1e-12,
                'matching_strategy': 'id'
            }
            
            preparator = OBELiXDataPreparator(config)
            preparator.load_obelix_excel(excel_path)
            preparator.parse_cif_files(cif_dir)
            preparator.match_cifs_to_obelix('id')
            preparator.create_enhanced_cgcnn_dataset()
            
            # Test different output formats
            formats = ['pickle', 'csv', 'json']
            
            for fmt in formats:
                output_path = os.path.join(temp_dir, 'output_formats', f'dataset.{fmt}')
                preparator.save_dataset(output_path, fmt)
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"✅ {fmt.upper()} format: {output_path} ({file_size} bytes)")
                else:
                    print(f"❌ {fmt.upper()} format: File not created")
            
            return True
            
        except Exception as e:
            print(f"❌ Output format test failed: {e}")
            return False

def test_error_handling():
    """Test error handling for various edge cases"""
    print("\n=== Testing Error Handling ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            config = {
                'output_dir': os.path.join(temp_dir, 'error_test'),
                'conductivity_threshold': 1e-12
            }
            
            preparator = OBELiXDataPreparator(config)
            
            # Test 1: Non-existent Excel file
            try:
                preparator.load_obelix_excel('nonexistent.xlsx')
                print("❌ Should have failed for non-existent Excel file")
            except FileNotFoundError:
                print("✅ Correctly handled non-existent Excel file")
            
            # Test 2: Non-existent CIF directory
            try:
                preparator.parse_cif_files('nonexistent_dir')
                print("❌ Should have failed for non-existent CIF directory")
            except FileNotFoundError:
                print("✅ Correctly handled non-existent CIF directory")
            
            # Test 3: Empty data
            preparator.obelix_data = []
            preparator.cif_metadata = {}
            matches = preparator.match_cifs_to_obelix('id')
            if len(matches) == 0:
                print("✅ Correctly handled empty data")
            else:
                print("❌ Should have returned empty matches for empty data")
            
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False

def run_all_tests():
    """Run all tests"""
    print("OBELiX Data Preparation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Matching Strategies", test_matching_strategies),
        ("Data Filtering", test_data_filtering),
        ("Output Formats", test_output_formats),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The OBELiX data preparation script is ready to use.")
    else:
        print("⚠️  Some tests failed. Please review the issues before using with real data.")
    
    return passed == total

if __name__ == "__main__":
    # Check if pandas is available for creating Excel files
    try:
        import pandas as pd
        run_all_tests()
    except ImportError:
        print("❌ pandas is required for testing. Please install it with: pip install pandas openpyxl")
        sys.exit(1)