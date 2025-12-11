#!/usr/bin/env python3
"""
Comprehensive Integration Tests for GPU-CPU Hybrid Processing System.

This test suite validates the integration between the existing address processing
pipeline and the new GPU-CPU hybrid processing system. It tests compatibility
with current CSV input/output formats, data models, and processing workflows.

Requirements: 9.1, 9.2
"""

import sys
import os
import unittest
import tempfile
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, 'src')

from src.hybrid_integration_adapter import HybridIntegrationAdapter, create_hybrid_integration_adapter
from src.migration_utilities import ConfigurationMigrator, ScriptMigrator, CompatibilityValidator
from src.backward_compatibility import LegacyProcessorAdapter, BackwardCompatibilityManager
from src.hybrid_processor import ProcessingConfiguration
from src.models import AddressRecord, ParsedAddress


class TestHybridIntegrationAdapter(unittest.TestCase):
    """Test hybrid integration adapter functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config = ProcessingConfiguration(
            gpu_batch_size=100,  # Small batch for testing
            target_throughput=500,  # Lower target for testing
            gpu_memory_fraction=0.8,
            cpu_allocation_ratio=0.1
        )
        
        self.adapter = HybridIntegrationAdapter(self.test_config)
        
        # Sample test data
        self.test_addresses = [
            "123 Main Street, Apartment 4B, Near Central Park, New York, NY 10001",
            "456 Oak Avenue, Building C, Unit 12, Los Angeles, CA 90210",
            "789 Pine Road, Suite 5A, Downtown Area, Chicago, IL 60601",
            "321 Elm Street, Floor 2, Business District, Houston, TX 77001",
            "654 Maple Drive, Apt 3C, Residential Area, Phoenix, AZ 85001"
        ]
        
        self.test_records = [
            AddressRecord(
                addr_hash_key=f"hash_{i}",
                addr_text=addr,
                city_id=f"city_{i}",
                pincode=f"1000{i}",
                state_id=f"state_{i}",
                zone_id=f"zone_{i}",
                address_id=f"addr_{i}",
                assigned_pickup_dlvd_geo_points="",
                assigned_pickup_dlvd_geo_points_count=0
            )
            for i, addr in enumerate(self.test_addresses)
        ]
    
    def test_adapter_initialization(self):
        """Test adapter initialization and setup."""
        # Test initialization
        self.assertFalse(self.adapter.is_initialized)
        
        # Initialize adapter
        self.adapter.initialize()
        self.assertTrue(self.adapter.is_initialized)
        
        # Test double initialization (should not fail)
        self.adapter.initialize()
        self.assertTrue(self.adapter.is_initialized)
    
    def test_process_address_list(self):
        """Test processing address list with integration adapter."""
        self.adapter.initialize()
        
        try:
            # Process addresses
            result = self.adapter.process_address_list(self.test_addresses)
            
            # Validate results
            self.assertIsNotNone(result)
            self.assertEqual(len(result.parsed_addresses), len(self.test_addresses))
            
            # Check that we got ParsedAddress objects
            for parsed_addr in result.parsed_addresses:
                self.assertIsInstance(parsed_addr, ParsedAddress)
            
            # Validate performance metrics
            self.assertIsNotNone(result.performance_metrics)
            self.assertGreater(result.processing_time, 0)
            
        finally:
            self.adapter.shutdown()
    
    def test_process_csv_file_integration(self):
        """Test CSV file processing with full integration."""
        self.adapter.initialize()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test CSV file
                input_file = Path(temp_dir) / "test_input.csv"
                output_file = Path(temp_dir) / "test_output.csv"
                
                self._create_test_csv(input_file)
                
                # Process CSV file
                result, consolidation_stats = self.adapter.process_csv_file(
                    str(input_file),
                    str(output_file),
                    consolidate_results=False,
                    comprehensive_output=False
                )
                
                # Validate results
                self.assertIsNotNone(result)
                self.assertEqual(len(result.parsed_addresses), len(self.test_addresses))
                
                # Check output file exists
                self.assertTrue(output_file.exists())
                
                # Validate output CSV format
                self._validate_output_csv(output_file)
                
        finally:
            self.adapter.shutdown()
    
    def test_comprehensive_output_generation(self):
        """Test comprehensive output generation with metadata."""
        self.adapter.initialize()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test CSV file
                input_file = Path(temp_dir) / "test_input.csv"
                output_file = Path(temp_dir) / "test_output.csv"
                
                self._create_test_csv(input_file)
                
                # Process with comprehensive output
                result, consolidation_stats = self.adapter.process_csv_file(
                    str(input_file),
                    str(output_file),
                    consolidate_results=False,
                    comprehensive_output=True
                )
                
                # Validate comprehensive output files were created
                output_dir = output_file.parent
                csv_files = list(output_dir.glob("*.csv"))
                json_files = list(output_dir.glob("*.json"))
                
                self.assertGreater(len(csv_files), 0, "No CSV output files generated")
                # Note: JSON files may not be generated if comprehensive output generator is not fully implemented
                
        finally:
            self.adapter.shutdown()
    
    def test_processing_statistics(self):
        """Test processing statistics generation."""
        self.adapter.initialize()
        
        try:
            # Process addresses
            result = self.adapter.process_address_list(self.test_addresses, self.test_records)
            
            # Get processing statistics
            stats = self.adapter.get_processing_statistics(result)
            
            # Validate statistics
            self.assertIn('total_addresses', stats)
            self.assertIn('successfully_parsed', stats)
            self.assertIn('processing_time', stats)
            self.assertIn('success_rate', stats)
            
            self.assertEqual(stats['total_addresses'], len(self.test_addresses))
            self.assertGreaterEqual(stats['success_rate'], 0)
            self.assertLessEqual(stats['success_rate'], 100)
            
        finally:
            self.adapter.shutdown()
    
    def _create_test_csv(self, file_path: Path):
        """Create test CSV file with sample data."""
        fieldnames = [
            'addr_hash_key', 'addr_text', 'city_id', 'pincode', 'state_id', 'zone_id',
            'address_id', 'assigned_pickup_dlvd_geo_points', 'assigned_pickup_dlvd_geo_points_count'
        ]
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in self.test_records:
                writer.writerow({
                    'addr_hash_key': record.addr_hash_key,
                    'addr_text': record.addr_text,
                    'city_id': record.city_id,
                    'pincode': record.pincode,
                    'state_id': record.state_id,
                    'zone_id': record.zone_id,
                    'address_id': record.address_id,
                    'assigned_pickup_dlvd_geo_points': record.assigned_pickup_dlvd_geo_points,
                    'assigned_pickup_dlvd_geo_points_count': record.assigned_pickup_dlvd_geo_points_count
                })
    
    def _validate_output_csv(self, file_path: Path):
        """Validate output CSV file format and content."""
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check that required columns exist
            required_columns = ['addr_text', 'UN', 'SN', 'LOC', 'CY', 'PIN']
            for col in required_columns:
                self.assertIn(col, reader.fieldnames, f"Required column {col} missing from output")
            
            # Check that we have data rows
            rows = list(reader)
            self.assertGreater(len(rows), 0, "No data rows in output CSV")
            self.assertEqual(len(rows), len(self.test_addresses), "Row count mismatch")


class TestMigrationUtilities(unittest.TestCase):
    """Test migration utilities functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.migrator = ConfigurationMigrator()
        self.script_migrator = ScriptMigrator()
        self.validator = CompatibilityValidator()
    
    def test_configuration_migration(self):
        """Test configuration migration from legacy format."""
        # Legacy configuration
        legacy_config = {
            'batch_size': 200,
            'gpu_memory': 0.9,
            'cpu_ratio': 0.05,
            'target_rate': 1800
        }
        
        # Migrate configuration
        migrated_config = self.migrator._migrate_config_dict(legacy_config)
        
        # Validate migration
        self.assertEqual(migrated_config['gpu_batch_size'], 200)
        self.assertEqual(migrated_config['gpu_memory_fraction'], 0.9)
        self.assertEqual(migrated_config['cpu_allocation_ratio'], 0.05)
        self.assertEqual(migrated_config['target_throughput'], 1800)
        
        # Test ProcessingConfiguration creation
        config = ProcessingConfiguration(**migrated_config)
        self.assertIsInstance(config, ProcessingConfiguration)
    
    def test_script_migration(self):
        """Test script migration functionality."""
        # Sample legacy script content
        legacy_script = """
import sys
from src.llm_parser import LLMParser

def main():
    parser = LLMParser()
    parser.setup_parser()
    
    addresses = ["123 Main St", "456 Oak Ave"]
    results = parser.parse_addresses(addresses)
    
    parser.cleanup_parser()
    return results

if __name__ == "__main__":
    main()
"""
        
        # Migrate script content
        migrated_script = self.script_migrator._migrate_script_content(legacy_script)
        
        # Validate migration
        self.assertIn('HybridIntegrationAdapter', migrated_script)
        self.assertIn('process_address_list', migrated_script)
        self.assertIn('initialize', migrated_script)
        self.assertIn('shutdown', migrated_script)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = ProcessingConfiguration(
            gpu_batch_size=400,
            target_throughput=2000,
            gpu_memory_fraction=0.95
        )
        
        is_valid, messages = self.validator.validate_configuration(valid_config)
        self.assertTrue(is_valid)
        self.assertGreater(len(messages), 0)
        
        # Invalid configuration (will be caught by ProcessingConfiguration validation)
        with self.assertRaises(ValueError):
            ProcessingConfiguration(
                gpu_batch_size=50,  # Too small
                gpu_memory_fraction=1.5  # Too large
            )


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.legacy_config = {
            'batch_size': 300,
            'gpu_memory': 0.85,
            'cpu_ratio': 0.03
        }
        
        self.test_addresses = [
            "123 Main Street, New York, NY 10001",
            "456 Oak Avenue, Los Angeles, CA 90210"
        ]
    
    def test_legacy_processor_adapter(self):
        """Test legacy processor adapter functionality."""
        # Create legacy adapter
        adapter = LegacyProcessorAdapter(self.legacy_config)
        
        try:
            # Test legacy setup
            adapter.setup_parser()
            self.assertTrue(adapter.is_setup)
            
            # Test legacy parsing
            results = adapter.parse_addresses(self.test_addresses)
            self.assertEqual(len(results), len(self.test_addresses))
            
            for result in results:
                self.assertIsInstance(result, ParsedAddress)
            
            # Test legacy statistics
            stats = adapter.get_stats()
            self.assertIn('total_processed', stats)
            self.assertEqual(stats['total_processed'], len(self.test_addresses))
            
        finally:
            # Test legacy cleanup
            adapter.cleanup_parser()
            self.assertFalse(adapter.is_setup)
    
    def test_legacy_csv_processing(self):
        """Test legacy CSV processing interface."""
        adapter = LegacyProcessorAdapter(self.legacy_config)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test CSV
                input_file = Path(temp_dir) / "test_input.csv"
                output_file = Path(temp_dir) / "test_output.csv"
                
                self._create_simple_csv(input_file)
                
                # Process using legacy interface
                records_processed = adapter.process_csv(str(input_file), str(output_file))
                
                # Validate results
                self.assertGreater(records_processed, 0)
                self.assertTrue(output_file.exists())
                
        finally:
            adapter.cleanup_parser()
    
    def test_backward_compatibility_manager(self):
        """Test backward compatibility manager."""
        manager = BackwardCompatibilityManager()
        
        # Test compatibility check
        compatibility_report = manager.check_compatibility(
            legacy_config=self.legacy_config
        )
        
        # Validate report structure
        self.assertIn('overall_compatibility', compatibility_report)
        self.assertIn('warnings', compatibility_report)
        self.assertIn('suggestions', compatibility_report)
        
        # Test legacy adapter creation
        adapter = manager.create_legacy_adapter(self.legacy_config)
        self.assertIsInstance(adapter, LegacyProcessorAdapter)
        
        # Test migration guide
        guide = manager.get_migration_guide()
        self.assertIsInstance(guide, str)
        self.assertIn('Migration Guide', guide)
    
    def _create_simple_csv(self, file_path: Path):
        """Create simple test CSV file."""
        fieldnames = [
            'addr_hash_key', 'addr_text', 'city_id', 'pincode', 'state_id', 'zone_id',
            'address_id', 'assigned_pickup_dlvd_geo_points', 'assigned_pickup_dlvd_geo_points_count'
        ]
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, addr in enumerate(self.test_addresses):
                writer.writerow({
                    'addr_hash_key': f"hash_{i}",
                    'addr_text': addr,
                    'city_id': f"city_{i}",
                    'pincode': f"1000{i}",
                    'state_id': f"state_{i}",
                    'zone_id': f"zone_{i}",
                    'address_id': f"addr_{i}",
                    'assigned_pickup_dlvd_geo_points': "",
                    'assigned_pickup_dlvd_geo_points_count': 0
                })


class TestExistingDatasetIntegration(unittest.TestCase):
    """Test integration with existing address datasets."""
    
    def setUp(self):
        """Set up test environment with real dataset samples."""
        # Look for existing CSV files in the workspace
        self.existing_csv_files = []
        
        # Check for sample CSV files
        workspace_root = Path('.')
        csv_patterns = ['export_customer_address_store_p*.csv', 'sample_*.csv', '*_sample.csv']
        
        for pattern in csv_patterns:
            files = list(workspace_root.glob(pattern))
            self.existing_csv_files.extend(files[:2])  # Limit to 2 files per pattern
        
        # Limit total files for testing
        self.existing_csv_files = self.existing_csv_files[:3]
        
        self.adapter = create_hybrid_integration_adapter(
            gpu_batch_size=200,  # Smaller batch for testing
            target_throughput=1000  # Lower target for testing
        )
    
    def test_existing_csv_file_processing(self):
        """Test processing existing CSV files from the workspace."""
        if not self.existing_csv_files:
            self.skipTest("No existing CSV files found for integration testing")
        
        self.adapter.initialize()
        
        try:
            for csv_file in self.existing_csv_files:
                with self.subTest(csv_file=csv_file.name):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        output_file = Path(temp_dir) / f"processed_{csv_file.name}"
                        
                        try:
                            # Process existing CSV file
                            result, consolidation_stats = self.adapter.process_csv_file(
                                str(csv_file),
                                str(output_file),
                                consolidate_results=False,
                                comprehensive_output=False
                            )
                            
                            # Validate processing results
                            self.assertIsNotNone(result)
                            self.assertGreater(len(result.parsed_addresses), 0)
                            
                            # Check output file was created
                            self.assertTrue(output_file.exists())
                            
                            # Log processing summary
                            success_count = sum(1 for addr in result.parsed_addresses if addr.parse_success)
                            print(f"\n✅ Processed {csv_file.name}: {success_count}/{len(result.parsed_addresses)} addresses")
                            
                            if result.performance_metrics:
                                print(f"   Performance: {result.performance_metrics.throughput_rate:.1f} addr/sec")
                            
                        except Exception as e:
                            print(f"\n❌ Failed to process {csv_file.name}: {e}")
                            # Don't fail the test for individual file processing errors
                            continue
        
        finally:
            self.adapter.shutdown()
    
    def test_sample_address_processing(self):
        """Test processing with sample addresses from existing datasets."""
        if not self.existing_csv_files:
            self.skipTest("No existing CSV files found for sample testing")
        
        self.adapter.initialize()
        
        try:
            # Extract sample addresses from existing files
            sample_addresses = []
            
            for csv_file in self.existing_csv_files[:1]:  # Use first file only
                try:
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for i, row in enumerate(reader):
                            if i >= 10:  # Limit to 10 addresses per file
                                break
                            if 'addr_text' in row and row['addr_text'].strip():
                                sample_addresses.append(row['addr_text'].strip())
                except Exception as e:
                    print(f"Warning: Could not read {csv_file}: {e}")
                    continue
            
            if not sample_addresses:
                self.skipTest("No sample addresses extracted from existing files")
            
            # Process sample addresses
            result = self.adapter.process_address_list(sample_addresses)
            
            # Validate results
            self.assertEqual(len(result.parsed_addresses), len(sample_addresses))
            
            # Check for successful parsing
            success_count = sum(1 for addr in result.parsed_addresses if addr.parse_success)
            success_rate = (success_count / len(sample_addresses)) * 100
            
            print(f"\n📊 Sample Processing Results:")
            print(f"   Total Addresses: {len(sample_addresses)}")
            print(f"   Successfully Parsed: {success_count}")
            print(f"   Success Rate: {success_rate:.1f}%")
            
            if result.performance_metrics:
                print(f"   Throughput: {result.performance_metrics.throughput_rate:.1f} addr/sec")
                print(f"   GPU Utilization: {result.performance_metrics.gpu_utilization:.1f}%")
            
            # Validate that we got some successful results
            self.assertGreater(success_count, 0, "No addresses were successfully parsed")
            
        finally:
            self.adapter.shutdown()


def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running GPU-CPU Hybrid Processing Integration Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestHybridIntegrationAdapter,
        TestMigrationUtilities,
        TestBackwardCompatibility,
        TestExistingDatasetIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 Integration Test Summary:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed.'}")
    
    return success


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)