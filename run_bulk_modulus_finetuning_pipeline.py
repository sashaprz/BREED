#!/usr/bin/env python3
"""
Complete Bulk Modulus Fine-Tuning Pipeline

This script orchestrates the complete fine-tuning process:
1. Extract bulk modulus values from Materials Project
2. Fine-tune the CGCNN model on high bulk modulus materials
3. Validate the fine-tuned model performance
4. Generate comprehensive reports
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BulkModulusFineTuningPipeline:
    """Complete pipeline for bulk modulus model fine-tuning"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.pipeline_status = {
            'extraction': False,
            'fine_tuning': False,
            'validation': False,
            'report_generation': False
        }
        
        # Output directory
        self.output_dir = Path("outputs/bulk_modulus_pipeline")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🚀 Bulk Modulus Fine-Tuning Pipeline Initialized")
        logger.info(f"Pipeline started at: {self.start_time}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are available"""
        logger.info("🔍 Checking prerequisites...")
        
        required_files = [
            "env/property_predictions/bulk-moduli.pth.tar",
            "env/property_predictions/CIF_OBELiX/cifs"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error("❌ Missing required files/directories:")
            for f in missing_files:
                logger.error(f"   - {f}")
            return False
        
        # Check if Materials Project API is available
        try:
            from mp_api.client import MPRester
            logger.info("✅ Materials Project API available")
        except ImportError:
            logger.error("❌ Materials Project API not available. Install with: pip install mp-api")
            return False
        
        # Check if PyTorch is available
        try:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"✅ PyTorch available (device: {device})")
        except ImportError:
            logger.error("❌ PyTorch not available")
            return False
        
        logger.info("✅ All prerequisites satisfied")
        return True
    
    def step_1_extract_bulk_modulus_data(self) -> bool:
        """Step 1: Extract bulk modulus data from Materials Project"""
        logger.info("=" * 60)
        logger.info("STEP 1: EXTRACTING BULK MODULUS DATA FROM MATERIALS PROJECT")
        logger.info("=" * 60)
        
        try:
            # Check if data already exists
            data_file = Path("bulk_modulus_data/obelix_bulk_modulus_high.csv")
            if data_file.exists():
                logger.info("Bulk modulus data already exists, skipping extraction")
                self.pipeline_status['extraction'] = True
                return True
            
            # Run extraction script
            logger.info("Starting Materials Project bulk modulus extraction...")
            result = subprocess.run([
                sys.executable, "extract_mp_bulk_modulus.py"
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info("✅ Bulk modulus extraction completed successfully")
                logger.info("Extraction output:")
                logger.info(result.stdout)
                self.pipeline_status['extraction'] = True
                return True
            else:
                logger.error("❌ Bulk modulus extraction failed")
                logger.error("Error output:")
                logger.error(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Bulk modulus extraction timed out (>1 hour)")
            return False
        except Exception as e:
            logger.error(f"❌ Error in bulk modulus extraction: {e}")
            return False
    
    def step_2_fine_tune_model(self) -> bool:
        """Step 2: Fine-tune the CGCNN bulk modulus model"""
        logger.info("=" * 60)
        logger.info("STEP 2: FINE-TUNING CGCNN BULK MODULUS MODEL")
        logger.info("=" * 60)
        
        try:
            # Check if fine-tuned model already exists
            model_file = Path("outputs/bulk_modulus_finetuned/best_bulk_modulus_model.pth.tar")
            if model_file.exists():
                logger.info("Fine-tuned model already exists")
                response = input("Do you want to retrain? (y/N): ").strip().lower()
                if response != 'y':
                    self.pipeline_status['fine_tuning'] = True
                    return True
            
            # Run fine-tuning script
            logger.info("Starting CGCNN bulk modulus fine-tuning...")
            result = subprocess.run([
                sys.executable, "finetune_cgcnn_bulk_modulus.py"
            ], capture_output=False, text=True)  # Show output in real-time
            
            if result.returncode == 0:
                logger.info("✅ Fine-tuning completed successfully")
                self.pipeline_status['fine_tuning'] = True
                return True
            else:
                logger.error("❌ Fine-tuning failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in fine-tuning: {e}")
            return False
    
    def step_3_validate_model(self) -> bool:
        """Step 3: Validate the fine-tuned model"""
        logger.info("=" * 60)
        logger.info("STEP 3: VALIDATING FINE-TUNED MODEL")
        logger.info("=" * 60)
        
        try:
            # Run validation script
            logger.info("Starting model validation...")
            result = subprocess.run([
                sys.executable, "validate_bulk_modulus_model.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Model validation completed successfully")
                logger.info("Validation output:")
                logger.info(result.stdout)
                self.pipeline_status['validation'] = True
                return True
            else:
                logger.error("❌ Model validation failed")
                logger.error("Error output:")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in validation: {e}")
            return False
    
    def step_4_generate_report(self) -> bool:
        """Step 4: Generate comprehensive pipeline report"""
        logger.info("=" * 60)
        logger.info("STEP 4: GENERATING PIPELINE REPORT")
        logger.info("=" * 60)
        
        try:
            end_time = datetime.now()
            duration = end_time - self.start_time
            
            # Collect results from all steps
            report = {
                'pipeline_info': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration.total_seconds(),
                    'duration_formatted': str(duration),
                    'status': self.pipeline_status
                },
                'extraction_results': self.load_extraction_results(),
                'fine_tuning_results': self.load_fine_tuning_results(),
                'validation_results': self.load_validation_results()
            }
            
            # Save comprehensive report
            report_file = self.output_dir / "pipeline_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate summary report
            self.generate_summary_report(report)
            
            logger.info(f"✅ Pipeline report generated: {report_file}")
            self.pipeline_status['report_generation'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
            return False
    
    def load_extraction_results(self) -> Dict[str, Any]:
        """Load extraction results"""
        try:
            summary_file = Path("bulk_modulus_data/extraction_summary.json")
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load extraction results: {e}")
        return {}
    
    def load_fine_tuning_results(self) -> Dict[str, Any]:
        """Load fine-tuning results"""
        try:
            results_file = Path("outputs/bulk_modulus_finetuned/training_history.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load fine-tuning results: {e}")
        return {}
    
    def load_validation_results(self) -> Dict[str, Any]:
        """Load validation results"""
        try:
            results_file = Path("outputs/bulk_modulus_validation/validation_summary.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load validation results: {e}")
        return {}
    
    def generate_summary_report(self, report: Dict[str, Any]) -> None:
        """Generate human-readable summary report"""
        summary_file = self.output_dir / "PIPELINE_SUMMARY.md"
        
        with open(summary_file, 'w') as f:
            f.write("# CGCNN Bulk Modulus Fine-Tuning Pipeline Report\n\n")
            
            # Pipeline info
            f.write("## Pipeline Information\n\n")
            f.write(f"- **Start Time**: {report['pipeline_info']['start_time']}\n")
            f.write(f"- **End Time**: {report['pipeline_info']['end_time']}\n")
            f.write(f"- **Duration**: {report['pipeline_info']['duration_formatted']}\n")
            f.write(f"- **Status**: {'✅ SUCCESS' if all(report['pipeline_info']['status'].values()) else '❌ PARTIAL'}\n\n")
            
            # Step status
            f.write("## Step Status\n\n")
            for step, status in report['pipeline_info']['status'].items():
                status_icon = "✅" if status else "❌"
                f.write(f"- {status_icon} **{step.replace('_', ' ').title()}**\n")
            f.write("\n")
            
            # Extraction results
            if report['extraction_results']:
                f.write("## Data Extraction Results\n\n")
                ext_results = report['extraction_results']
                f.write(f"- **Total OBELiX structures processed**: {ext_results.get('total_obelix_structures', 'N/A')}\n")
                f.write(f"- **Successful matches**: {ext_results.get('successful_matches', 'N/A')}\n")
                f.write(f"- **High bulk modulus materials**: {ext_results.get('high_bulk_modulus_materials', 'N/A')}\n")
                
                if 'bulk_modulus_stats' in ext_results:
                    stats = ext_results['bulk_modulus_stats']
                    f.write(f"- **Bulk modulus range**: {stats.get('min', 'N/A'):.1f} - {stats.get('max', 'N/A'):.1f} GPa\n")
                    f.write(f"- **Mean bulk modulus**: {stats.get('mean', 'N/A'):.1f} ± {stats.get('std', 'N/A'):.1f} GPa\n")
                f.write("\n")
            
            # Validation results
            if report['validation_results']:
                f.write("## Model Performance Comparison\n\n")
                val_results = report['validation_results']
                
                if 'original_model' in val_results and 'finetuned_model' in val_results:
                    orig = val_results['original_model']
                    fine = val_results['finetuned_model']
                    
                    f.write("### Original Model\n")
                    f.write(f"- **MAE**: {orig.get('mae', 'N/A'):.2f} GPa\n")
                    f.write(f"- **RMSE**: {orig.get('rmse', 'N/A'):.2f} GPa\n")
                    f.write(f"- **R²**: {orig.get('r2', 'N/A'):.3f}\n\n")
                    
                    f.write("### Fine-tuned Model\n")
                    f.write(f"- **MAE**: {fine.get('mae', 'N/A'):.2f} GPa\n")
                    f.write(f"- **RMSE**: {fine.get('rmse', 'N/A'):.2f} GPa\n")
                    f.write(f"- **R²**: {fine.get('r2', 'N/A'):.3f}\n\n")
                    
                    if 'improvements' in val_results:
                        imp = val_results['improvements']
                        f.write("### Improvements\n")
                        f.write(f"- **MAE Improvement**: {imp.get('mae_improvement_percent', 'N/A'):+.1f}%\n")
                        f.write(f"- **R² Improvement**: {imp.get('r2_improvement', 'N/A'):+.3f}\n\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            if all(report['pipeline_info']['status'].values()):
                f.write("🎉 **Pipeline completed successfully!**\n\n")
                
                if report['validation_results'] and 'improvements' in report['validation_results']:
                    imp = report['validation_results']['improvements']
                    mae_imp = imp.get('mae_improvement_percent', 0)
                    r2_imp = imp.get('r2_improvement', 0)
                    
                    if mae_imp > 0 and r2_imp > 0:
                        f.write("✅ **Fine-tuning was successful** - both MAE and R² improved on high bulk modulus materials.\n")
                    elif mae_imp > 0 or r2_imp > 0:
                        f.write("⚠️ **Fine-tuning showed mixed results** - some metrics improved.\n")
                    else:
                        f.write("❌ **Fine-tuning was not effective** - consider adjusting hyperparameters.\n")
                else:
                    f.write("⚠️ **Pipeline completed but validation results are incomplete.**\n")
            else:
                f.write("❌ **Pipeline completed with errors.** Check individual step logs for details.\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("- `bulk_modulus_data/` - Extracted Materials Project data\n")
            f.write("- `outputs/bulk_modulus_finetuned/` - Fine-tuned model and training history\n")
            f.write("- `outputs/bulk_modulus_validation/` - Validation results and comparison plots\n")
            f.write("- `outputs/bulk_modulus_pipeline/` - Pipeline reports and summaries\n")
        
        logger.info(f"✅ Summary report generated: {summary_file}")
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete fine-tuning pipeline"""
        logger.info("🚀 STARTING COMPLETE BULK MODULUS FINE-TUNING PIPELINE")
        logger.info("=" * 70)
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites not satisfied. Exiting.")
            return False
        
        # Step 1: Extract bulk modulus data
        if not self.step_1_extract_bulk_modulus_data():
            logger.error("❌ Pipeline failed at Step 1: Data Extraction")
            return False
        
        # Step 2: Fine-tune model
        if not self.step_2_fine_tune_model():
            logger.error("❌ Pipeline failed at Step 2: Fine-tuning")
            return False
        
        # Step 3: Validate model
        if not self.step_3_validate_model():
            logger.error("❌ Pipeline failed at Step 3: Validation")
            return False
        
        # Step 4: Generate report
        if not self.step_4_generate_report():
            logger.error("❌ Pipeline failed at Step 4: Report Generation")
            return False
        
        # Success!
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        logger.info("=" * 70)
        logger.info("🎉 BULK MODULUS FINE-TUNING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"⏱️  Total Duration: {duration}")
        logger.info(f"📁 Results Directory: {self.output_dir}")
        logger.info(f"📊 Summary Report: {self.output_dir / 'PIPELINE_SUMMARY.md'}")
        logger.info(f"🤖 Fine-tuned Model: outputs/bulk_modulus_finetuned/best_bulk_modulus_model.pth.tar")
        logger.info("=" * 70)
        
        return True


def main():
    """Main function"""
    print("🔧 CGCNN Bulk Modulus Fine-Tuning Pipeline")
    print("=" * 60)
    print("This pipeline will:")
    print("1. Extract bulk modulus data from Materials Project")
    print("2. Fine-tune CGCNN model on high bulk modulus materials")
    print("3. Validate model performance improvements")
    print("4. Generate comprehensive reports")
    print("=" * 60)
    
    # Confirm execution
    response = input("Do you want to proceed? (y/N): ").strip().lower()
    if response != 'y':
        print("Pipeline cancelled.")
        return
    
    # Create and run pipeline
    pipeline = BulkModulusFineTuningPipeline()
    
    try:
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            print("Check the generated reports for detailed results.")
        else:
            print("\n❌ Pipeline failed. Check the logs for details.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()