#!/usr/bin/env python3
"""
Basic Integration Test for GPU-CPU Hybrid Processing System.

Quick validation test for the integration components without heavy GPU processing.
"""

import sys
import os
import tempfile
import csv
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all integration modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        from src.hybrid_integration_adapter import HybridIntegrationAdapter, create_hybrid_integration_adapter
        print("✅ HybridIntegrationAdapter imported successfully")
        
        from src.migration_utilities import ConfigurationMigrator, ScriptMigrator, CompatibilityValidator
        print("✅ Migration utilities imported successfully")
        
        from src.backward_compatibility import LegacyProcessorAdapter, BackwardCompatibilityManager
        print("✅ Backward compatibility modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_configuration_migration():
    """Test configuration migration functionality."""
    print("\n🧪 Testing configuration migration...")
    
    try:
        from src.migration_utilities import ConfigurationMigrator
        
        migrator = ConfigurationMigrator()
        
        # Test legacy configuration migration
        legacy_config = {
            'batch_size': 200,
            'gpu_memory': 0.9,
            'cpu_ratio': 0.05,
            'target_rate': 1800
        }
        
        migrated_config = migrator._migrate_config_dict(legacy_config)
        
        # Validate migration
        assert migrated_config['gpu_batch_size'] == 200
        assert migrated_config['gpu_memory_fraction'] == 0.9
        assert migrated_config['cpu_allocation_ratio'] == 0.05
        assert migrated_config['target_throughput'] == 1800
        
        print("✅ Configuration migration working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration migration failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility functionality."""
    print("\n🧪 Testing backward compatibility...")
    
    try:
        from src.backward_compatibility import LegacyProcessorAdapter
        
        # Create legacy adapter with simple config
        legacy_config = {
            'batch_size': 100,
            'gpu_memory': 0.8,
            'cpu_ratio': 0.1
        }
        
        adapter = LegacyProcessorAdapter(legacy_config)
        
        # Test configuration migration
        assert adapter.migrated_config['gpu_batch_size'] == 100
        assert adapter.migrated_config['gpu_memory_fraction'] == 0.8
        assert adapter.migrated_config['cpu_allocation_ratio'] == 0.1
        
        # Test stats method
        stats = adapter.get_stats()
        assert 'total_processed' in stats
        assert 'processor_type' in stats
        
        print("✅ Backward compatibility working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")
        return False


def test_csv_format_compatibility():
    """Test CSV format compatibility."""
    print("\n🧪 Testing CSV format compatibility...")
    
    try:
        from src.models import AddressRecord
        from src.csv_reader import CSVReader
        
        # Create test CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'addr_hash_key', 'addr_text', 'city_id', 'pincode', 'state_id', 'zone_id',
                'address_id', 'assigned_pickup_dlvd_geo_points', 'assigned_pickup_dlvd_geo_points_count'
            ])
            
            # Write test data
            writer.writerow([
                'hash_1', '123 Main Street, New York, NY 10001', 'city_1', '10001',
                'state_1', 'zone_1', 'addr_1', '', '0'
            ])
            
            csv_file = f.name
        
        try:
            # Test CSV reading
            required_columns = [
                'addr_hash_key', 'addr_text', 'city_id', 'pincode', 
                'state_id', 'zone_id', 'address_id', 
                'assigned_pickup_dlvd_geo_points', 'assigned_pickup_dlvd_geo_points_count'
            ]
            
            csv_reader = CSVReader(csv_file, required_columns)
            
            # Validate columns
            is_valid, missing_columns = csv_reader.validate_columns()
            assert is_valid, f"CSV validation failed: {missing_columns}"
            
            # Read records
            records = list(csv_reader.read())
            assert len(records) == 1
            
            record = records[0]
            assert isinstance(record, AddressRecord)
            assert record.addr_text == '123 Main Street, New York, NY 10001'
            assert record.pincode == '10001'
            
            print("✅ CSV format compatibility working correctly")
            return True
            
        finally:
            # Cleanup
            os.unlink(csv_file)
        
    except Exception as e:
        print(f"❌ CSV format compatibility failed: {e}")
        return False


def test_data_model_compatibility():
    """Test data model compatibility."""
    print("\n🧪 Testing data model compatibility...")
    
    try:
        from src.models import AddressRecord, ParsedAddress
        
        # Test AddressRecord creation
        record = AddressRecord(
            addr_hash_key="test_hash",
            addr_text="123 Test Street, Test City, TC 12345",
            city_id="test_city",
            pincode="12345",
            state_id="test_state",
            zone_id="test_zone",
            address_id="test_addr",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0
        )
        
        assert record.addr_text == "123 Test Street, Test City, TC 12345"
        assert record.pincode == "12345"
        
        # Test ParsedAddress creation
        parsed = ParsedAddress(
            unit_number="123",
            society_name="Test Society",
            locality="Test Locality",
            city="Test City",
            pin_code="12345",
            parse_success=True
        )
        
        assert parsed.unit_number == "123"
        assert parsed.society_name == "Test Society"
        assert parsed.parse_success == True
        
        print("✅ Data model compatibility working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Data model compatibility failed: {e}")
        return False


def test_integration_adapter_creation():
    """Test integration adapter creation without initialization."""
    print("\n🧪 Testing integration adapter creation...")
    
    try:
        from src.hybrid_integration_adapter import create_hybrid_integration_adapter
        from src.hybrid_processor import ProcessingConfiguration
        
        # Test adapter creation with default parameters
        adapter = create_hybrid_integration_adapter()
        
        assert adapter is not None
        assert isinstance(adapter.config, ProcessingConfiguration)
        assert not adapter.is_initialized
        
        # Test adapter creation with custom parameters
        custom_adapter = create_hybrid_integration_adapter(
            gpu_batch_size=300,
            target_throughput=1500,
            gpu_memory_fraction=0.85
        )
        
        assert custom_adapter.config.gpu_batch_size == 300
        assert custom_adapter.config.target_throughput == 1500
        assert custom_adapter.config.gpu_memory_fraction == 0.85
        
        print("✅ Integration adapter creation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Integration adapter creation failed: {e}")
        return False


def main():
    """Run basic integration tests."""
    print("🎯 BASIC INTEGRATION VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration Migration", test_configuration_migration),
        ("Backward Compatibility", test_backward_compatibility),
        ("CSV Format Compatibility", test_csv_format_compatibility),
        ("Data Model Compatibility", test_data_model_compatibility),
        ("Integration Adapter Creation", test_integration_adapter_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 BASIC INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {successful}/{total} tests passed")
    
    if successful == total:
        print("🎉 All basic integration tests passed!")
        print("📋 The integration components are working correctly.")
        print("💡 You can now run full integration tests or use the hybrid processing system.")
    else:
        print("⚠️ Some basic tests failed. Check the output above for details.")
    
    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)