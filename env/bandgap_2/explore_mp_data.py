#!/usr/bin/env python3
"""
Materials Project Data Explorer
Discovers what high-fidelity bandgap data is available and how to access it
"""

import logging
from mp_api.client import MPRester
import json
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MPDataExplorer:
    """Explore Materials Project API to find high-fidelity bandgap data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.mpr = MPRester(api_key)
    
    def explore_available_endpoints(self):
        """Explore what endpoints are available in the API"""
        logger.info("=== Exploring Available API Endpoints ===")
        
        # Safely check what's available in the MPRester object
        endpoints = []
        known_endpoints = [
            'materials', 'summary', 'thermo', 'electronic_structure',
            'phonon', 'elasticity', 'piezo', 'dielectric', 'magnetism',
            'oxidation_states', 'xas', 'grain_boundary', 'fermi'
        ]
        
        for attr in known_endpoints:
            try:
                if hasattr(self.mpr, attr):
                    endpoint_obj = getattr(self.mpr, attr)
                    if hasattr(endpoint_obj, 'search'):
                        endpoints.append(attr)
                        logger.info(f"  ✅ Found endpoint: {attr}")
                    else:
                        logger.info(f"  ❌ No search method: {attr}")
                else:
                    logger.info(f"  ❌ Endpoint not found: {attr}")
            except Exception as e:
                logger.warning(f"  ⚠️  Error checking {attr}: {e}")
        
        # Also check materials sub-endpoints
        if hasattr(self.mpr, 'materials'):
            materials_endpoints = []
            materials_attrs = ['summary', 'core', 'absorption', 'alloys', 'electronic_structure']
            for attr in materials_attrs:
                try:
                    if hasattr(self.mpr.materials, attr):
                        sub_endpoint = getattr(self.mpr.materials, attr)
                        if hasattr(sub_endpoint, 'search'):
                            materials_endpoints.append(f"materials.{attr}")
                            logger.info(f"  ✅ Found materials sub-endpoint: materials.{attr}")
                except Exception as e:
                    logger.warning(f"  ⚠️  Error checking materials.{attr}: {e}")
            endpoints.extend(materials_endpoints)
        
        logger.info(f"Total available endpoints with search methods: {len(endpoints)}")
        return endpoints
    
    def explore_summary_fields(self):
        """Explore what fields are available in summary data"""
        logger.info("=== Exploring Summary Data Fields ===")
        
        try:
            # Get a small sample to see available fields
            sample = self.mpr.materials.summary.search(
                band_gap=(1.0, 2.0),
                chunk_size=5,
                num_chunks=1
            )
            
            if sample:
                first_material = sample[0]
                logger.info(f"Sample material: {first_material.material_id}")
                
                # Get all available attributes
                available_fields = []
                for attr in dir(first_material):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(first_material, attr)
                            if not callable(value):
                                available_fields.append((attr, type(value).__name__, str(value)[:100]))
                        except:
                            available_fields.append((attr, "unknown", "error accessing"))
                
                logger.info("Available fields in summary data:")
                for field, field_type, sample_value in available_fields:
                    logger.info(f"  {field} ({field_type}): {sample_value}")
                
                return available_fields
            else:
                logger.warning("No sample materials found")
                return []
                
        except Exception as e:
            logger.error(f"Error exploring summary fields: {e}")
            return []
    
    def explore_electronic_structure_data(self):
        """Explore electronic structure endpoint for bandgap data"""
        logger.info("=== Exploring Electronic Structure Data ===")
        
        try:
            # Check multiple possible electronic structure endpoints
            es_endpoints = [
                ('electronic_structure', lambda: self.mpr.electronic_structure),
                ('materials.electronic_structure', lambda: self.mpr.materials.electronic_structure),
                ('materials.core', lambda: self.mpr.materials.core)
            ]
            
            for endpoint_name, endpoint_getter in es_endpoints:
                try:
                    logger.info(f"Trying endpoint: {endpoint_name}")
                    endpoint = endpoint_getter()
                    
                    if hasattr(endpoint, 'search'):
                        logger.info(f"  ✅ {endpoint_name} has search method")
                        
                        # Try to search for materials
                        try:
                            es_data = endpoint.search(
                                band_gap=(1.0, 2.0),
                                chunk_size=3,
                                num_chunks=1
                            )
                            
                            if es_data:
                                logger.info(f"  ✅ Found {len(es_data)} materials with {endpoint_name}")
                                first_es = es_data[0]
                                
                                # Explore fields
                                es_fields = []
                                for attr in dir(first_es):
                                    if not attr.startswith('_'):
                                        try:
                                            value = getattr(first_es, attr)
                                            if not callable(value):
                                                es_fields.append((attr, type(value).__name__, str(value)[:100]))
                                        except:
                                            es_fields.append((attr, "unknown", "error accessing"))
                                
                                logger.info(f"Fields in {endpoint_name}:")
                                for field, field_type, sample_value in es_fields:
                                    logger.info(f"  {field} ({field_type}): {sample_value}")
                                
                                return es_fields
                            else:
                                logger.info(f"  ❌ No data found with {endpoint_name}")
                                
                        except Exception as e:
                            logger.warning(f"  ❌ Error searching {endpoint_name}: {e}")
                    else:
                        logger.info(f"  ❌ {endpoint_name} has no search method")
                        
                except Exception as e:
                    logger.warning(f"  ❌ Error accessing {endpoint_name}: {e}")
            
            logger.info("No electronic structure data found in any endpoint")
            return []
                
        except Exception as e:
            logger.error(f"Error exploring electronic structure: {e}")
            return []
    
    def explore_thermo_data(self):
        """Explore thermodynamics endpoint"""
        logger.info("=== Exploring Thermodynamics Data ===")
        
        try:
            if hasattr(self.mpr, 'thermo'):
                logger.info("Thermodynamics endpoint found!")
                
                # Try to get some thermo data
                try:
                    thermo_data = self.mpr.thermo.search(
                        formula="Li2O",
                        chunk_size=5,
                        num_chunks=1
                    )
                    
                    if thermo_data:
                        logger.info(f"Found {len(thermo_data)} thermodynamics entries")
                        first_thermo = thermo_data[0]
                        
                        # Explore fields
                        thermo_fields = []
                        for attr in dir(first_thermo):
                            if not attr.startswith('_'):
                                try:
                                    value = getattr(first_thermo, attr)
                                    if not callable(value):
                                        thermo_fields.append((attr, type(value).__name__, str(value)[:100]))
                                except:
                                    thermo_fields.append((attr, "unknown", "error accessing"))
                        
                        logger.info("Thermodynamics fields:")
                        for field, field_type, sample_value in thermo_fields:
                            logger.info(f"  {field} ({field_type}): {sample_value}")
                        
                        return thermo_fields
                    else:
                        logger.info("No thermodynamics data found")
                        return []
                        
                except Exception as e:
                    logger.warning(f"Error searching thermodynamics: {e}")
                    return []
            else:
                logger.info("No thermodynamics endpoint found")
                return []
                
        except Exception as e:
            logger.error(f"Error exploring thermodynamics: {e}")
            return []
    
    def search_for_hse_data(self):
        """Search specifically for materials that might have HSE data"""
        logger.info("=== Searching for HSE/High-Fidelity Data ===")
        
        # Try different approaches to find HSE data
        approaches = [
            ("Summary with all fields", self._search_summary_all_fields),
            ("Electronic structure search", self._search_electronic_structure),
            ("Direct material lookup", self._search_direct_materials),
            ("Search by calculation type", self._search_by_calc_type)
        ]
        
        results = {}
        for approach_name, search_func in approaches:
            logger.info(f"Trying approach: {approach_name}")
            try:
                result = search_func()
                results[approach_name] = result
                if result:
                    logger.info(f"  ✅ Found data with {approach_name}")
                else:
                    logger.info(f"  ❌ No data found with {approach_name}")
            except Exception as e:
                logger.warning(f"  ❌ Error with {approach_name}: {e}")
                results[approach_name] = None
        
        return results
    
    def _search_summary_all_fields(self):
        """Try to get summary data with all possible fields"""
        try:
            # Start with basic fields and add more if they work
            basic_fields = ["material_id", "formula_pretty", "band_gap"]
            extended_fields = ["structure", "symmetry", "energy_above_hull", "formation_energy_per_atom"]
            experimental_fields = ["theoretical", "experimental", "electronic_structure", "bandstructure", "dos"]
            
            # Try different field combinations
            field_sets = [
                ("basic", basic_fields),
                ("extended", basic_fields + extended_fields),
                ("experimental", basic_fields + extended_fields + experimental_fields)
            ]
            
            for set_name, fields in field_sets:
                try:
                    logger.info(f"  Trying {set_name} field set: {fields}")
                    materials = self.mpr.materials.summary.search(
                        band_gap=(1.0, 2.0),
                        fields=fields,
                        chunk_size=5,
                        num_chunks=1
                    )
                    
                    if materials:
                        logger.info(f"  ✅ Success with {set_name} fields: {len(materials)} materials")
                        
                        # Check first material for bandgap-related data
                        first_mat = materials[0]
                        bandgap_fields = []
                        
                        for attr in dir(first_mat):
                            if not attr.startswith('_') and ('band' in attr.lower() or 'gap' in attr.lower() or 'electronic' in attr.lower()):
                                try:
                                    value = getattr(first_mat, attr)
                                    if not callable(value) and value is not None:
                                        bandgap_fields.append((attr, value))
                                        logger.info(f"    Found bandgap-related field: {attr} = {value}")
                                except:
                                    pass
                        
                        if bandgap_fields:
                            logger.info(f"  ✅ Found {len(bandgap_fields)} bandgap-related fields")
                        else:
                            logger.info("  ❌ No additional bandgap fields found")
                        
                        return materials
                    else:
                        logger.info(f"  ❌ No materials found with {set_name} fields")
                        
                except Exception as e:
                    logger.warning(f"  ❌ Error with {set_name} fields: {e}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Summary all fields search failed: {e}")
            return None
    
    def _search_electronic_structure(self):
        """Try electronic structure endpoint"""
        try:
            if hasattr(self.mpr, 'electronic_structure'):
                return self.mpr.electronic_structure.search(
                    band_gap=(1.0, 2.0),
                    chunk_size=10,
                    num_chunks=1
                )
            return None
        except Exception as e:
            logger.warning(f"Electronic structure search failed: {e}")
            return None
    
    def _search_direct_materials(self):
        """Try direct materials endpoint"""
        try:
            if hasattr(self.mpr, 'materials'):
                # Try different sub-endpoints
                if hasattr(self.mpr.materials, 'core'):
                    return self.mpr.materials.core.search(
                        band_gap=(1.0, 2.0),
                        chunk_size=10,
                        num_chunks=1
                    )
            return None
        except Exception as e:
            logger.warning(f"Direct materials search failed: {e}")
            return None
    
    def _search_by_calc_type(self):
        """Try searching by calculation type"""
        try:
            # This might not work, but worth trying
            materials = self.mpr.materials.summary.search(
                band_gap=(1.0, 2.0),
                chunk_size=10,
                num_chunks=1
            )
            
            # Look for materials that might have multiple calculations
            if materials:
                for mat in materials:
                    # Try to get detailed data for this material
                    try:
                        detailed = self.mpr.get_data_by_id(mat.material_id)
                        if detailed:
                            logger.info(f"Detailed data for {mat.material_id}: {type(detailed)}")
                            return detailed
                    except Exception as e:
                        logger.warning(f"Could not get detailed data for {mat.material_id}: {e}")
            
            return materials
        except Exception as e:
            logger.warning(f"Calc type search failed: {e}")
            return None
    
    def generate_report(self):
        """Generate a comprehensive report of findings"""
        logger.info("=== Generating Comprehensive Report ===")
        
        report = {
            "endpoints": self.explore_available_endpoints(),
            "summary_fields": self.explore_summary_fields(),
            "electronic_structure": self.explore_electronic_structure_data(),
            "thermodynamics": self.explore_thermo_data(),
            "hse_search_results": self.search_for_hse_data()
        }
        
        # Save report to file
        with open("mp_data_exploration_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Report saved to mp_data_exploration_report.json")
        
        # Print summary
        logger.info("=== SUMMARY ===")
        logger.info(f"Available endpoints: {len(report['endpoints'])}")
        logger.info(f"Summary fields found: {len(report['summary_fields'])}")
        logger.info(f"Electronic structure available: {'Yes' if report['electronic_structure'] else 'No'}")
        logger.info(f"Thermodynamics available: {'Yes' if report['thermodynamics'] else 'No'}")
        
        # Check for high-fidelity data
        hse_found = any(result for result in report['hse_search_results'].values() if result)
        logger.info(f"High-fidelity data found: {'Yes' if hse_found else 'No'}")
        
        return report

def main():
    """Main exploration function"""
    API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"
    
    logger.info("Starting Materials Project data exploration...")
    
    explorer = MPDataExplorer(API_KEY)
    
    try:
        report = explorer.generate_report()
        
        logger.info("=== RECOMMENDATIONS ===")
        
        # Analyze results and provide recommendations
        if report['hse_search_results']:
            successful_approaches = [k for k, v in report['hse_search_results'].items() if v]
            if successful_approaches:
                logger.info(f"✅ Found data using: {', '.join(successful_approaches)}")
                logger.info("Recommendation: Update extraction script to use these approaches")
            else:
                logger.info("❌ No high-fidelity bandgap data found in Materials Project")
                logger.info("Recommendations:")
                logger.info("  1. Use JARVIS-DFT database (has more HSE data)")
                logger.info("  2. Apply known correction factors to PBE bandgaps")
                logger.info("  3. Search literature for experimental values")
                logger.info("  4. Use Materials Project for structures only")
        
        logger.info("Exploration completed successfully!")
        
    except Exception as e:
        logger.error(f"Exploration failed: {e}")
        raise

if __name__ == "__main__":
    main()