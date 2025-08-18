"""
Example script showing how to use the OBELiX data preparation script

This demonstrates different ways to prepare the OBELiX dataset for Enhanced CGCNN training.
"""

import os
import json
from prepare_obelix_data import OBELiXDataPreparator

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Configuration
    config = {
        'output_dir': 'data/obelix_basic',
        'conductivity_threshold': 1e-12,
        'conductivity_column': 'Ionic conductivity (S cm-1)',
        'matching_strategy': 'composition_spacegroup'
    }
    
    # Initialize preparator
    preparator = OBELiXDataPreparator(config)
    
    # Example paths (adjust these to your actual paths)
    obelix_excel = "data/OBELiX_data.xlsx"
    cif_folder = "data/cifs"
    
    if os.path.exists(obelix_excel) and os.path.exists(cif_folder):
        try:
            # Load and process data
            preparator.load_obelix_excel(obelix_excel)
            preparator.parse_cif_files(cif_folder)
            preparator.match_cifs_to_obelix('composition_spacegroup')
            preparator.create_enhanced_cgcnn_dataset()
            
            # Save in multiple formats
            preparator.save_dataset('data/obelix_basic/dataset.pkl', 'pickle')
            preparator.save_dataset('data/obelix_basic/dataset.csv', 'csv')
            preparator.save_cgcnn_format('data/obelix_basic/cgcnn_format')
            
            # Print statistics
            preparator.print_statistics()
            
        except Exception as e:
            print(f"Error in basic usage: {e}")
    else:
        print("Example data files not found. Please adjust paths in the script.")

def example_advanced_usage():
    """Advanced usage with custom configuration"""
    print("\n=== Advanced Usage Example ===")
    
    # Load configuration from file
    config_file = "obelix_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'output_dir': 'data/obelix_advanced',
            'conductivity_threshold': 1e-10,  # Stricter threshold
            'conductivity_column': 'Ionic conductivity (S cm-1)',
            'matching_strategy': 'composition_spacegroup'
        }
    
    preparator = OBELiXDataPreparator(config)
    
    # Example with different matching strategies
    strategies = ['id', 'composition_spacegroup', 'composition_only']
    
    for strategy in strategies:
        print(f"\nTrying matching strategy: {strategy}")
        
        # Reset for each strategy
        preparator.matched_pairs = []
        preparator.processed_data = []
        
        try:
            # You would load your actual data here
            # preparator.load_obelix_excel("your_obelix_file.xlsx")
            # preparator.parse_cif_files("your_cif_folder")
            
            # For demonstration, we'll just show the method calls
            print(f"Would match using strategy: {strategy}")
            # preparator.match_cifs_to_obelix(strategy)
            # preparator.create_enhanced_cgcnn_dataset()
            
        except Exception as e:
            print(f"Error with strategy {strategy}: {e}")

def example_filtering_and_validation():
    """Example showing data filtering and validation"""
    print("\n=== Filtering and Validation Example ===")
    
    config = {
        'output_dir': 'data/obelix_filtered',
        'conductivity_threshold': 1e-8,  # Higher threshold for better quality data
        'conductivity_column': 'Ionic conductivity (S cm-1)',
        'matching_strategy': 'composition_spacegroup'
    }
    
    preparator = OBELiXDataPreparator(config)
    
    # Example of how you might validate and filter data
    print("Configuration for high-quality dataset:")
    print(f"- Conductivity threshold: {config['conductivity_threshold']}")
    print(f"- Matching strategy: {config['matching_strategy']}")
    print("- This will create a smaller but higher quality dataset")

def example_command_line_usage():
    """Show command line usage examples"""
    print("\n=== Command Line Usage Examples ===")
    
    examples = [
        # Basic usage
        "python prepare_obelix_data.py --obelix_excel data/OBELiX_data.xlsx --cif_folder data/cifs",
        
        # With custom output directory
        "python prepare_obelix_data.py --obelix_excel data/OBELiX_data.xlsx --cif_folder data/cifs --output_dir results/my_dataset",
        
        # With different matching strategy
        "python prepare_obelix_data.py --obelix_excel data/OBELiX_data.xlsx --cif_folder data/cifs --matching_strategy id",
        
        # With custom threshold and CGCNN format
        "python prepare_obelix_data.py --obelix_excel data/OBELiX_data.xlsx --cif_folder data/cifs --conductivity_threshold 1e-10 --save_cgcnn_format",
        
        # Using configuration file
        "python prepare_obelix_data.py --obelix_excel data/OBELiX_data.xlsx --cif_folder data/cifs --config_file obelix_config.json",
        
        # Multiple output formats
        "python prepare_obelix_data.py --obelix_excel data/OBELiX_data.xlsx --cif_folder data/cifs --output_format csv --save_cgcnn_format"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")

def create_sample_data():
    """Create sample data for testing (if needed)"""
    print("\n=== Creating Sample Data ===")
    
    # This would create sample OBELiX Excel file and CIF files for testing
    # In practice, you would use your actual OBELiX dataset
    
    sample_data = {
        'ID': ['sample1', 'sample2', 'sample3'],
        'Composition': ['Li2O', 'LiCoO2', 'Li3PO4'],
        'Ionic conductivity (S cm-1)': [1e-5, 1e-7, 1e-6],
        'Space group number': [225, 166, 62]
    }
    
    # Create sample directory
    os.makedirs('data/sample', exist_ok=True)
    
    # Save sample Excel file
    import pandas as pd
    df = pd.DataFrame(sample_data)
    df.to_excel('data/sample/sample_obelix.xlsx', index=False)
    
    print("Sample OBELiX Excel file created at: data/sample/sample_obelix.xlsx")
    print("Note: You'll need actual CIF files to complete the preparation process")

def main():
    """Main function to run examples"""
    print("OBELiX Data Preparation Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_advanced_usage()
    example_filtering_and_validation()
    example_command_line_usage()
    
    # Optionally create sample data
    create_sample = input("\nCreate sample data for testing? (y/n): ").lower().strip()
    if create_sample == 'y':
        try:
            create_sample_data()
        except ImportError:
            print("pandas not available for creating sample Excel file")
        except Exception as e:
            print(f"Error creating sample data: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use the data preparation script:")
    print("1. Adjust the paths in the examples above to point to your actual data")
    print("2. Run the script with your OBELiX Excel file and CIF folder")
    print("3. The prepared dataset will be ready for Enhanced CGCNN training")

if __name__ == "__main__":
    main()