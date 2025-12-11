#!/usr/bin/env python3
"""
Example: GPU-CPU Hybrid Processing Integration with Existing Pipeline

This example demonstrates how to integrate the new GPU-CPU hybrid processing
system with existing address processing workflows. It shows compatibility
with current CSV formats, data models, and processing patterns.

Requirements: 9.1, 9.2
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_integration_adapter import HybridIntegrationAdapter, create_hybrid_integration_adapter
from hybrid_processor import ProcessingConfiguration
from backward_compatibility import create_legacy_processor, load_legacy_configuration
from migration_utilities import migrate_existing_configuration


def example_basic_integration():
    """Example: Basic integration with hybrid processing."""
    print("🚀 Example: Basic Hybrid Processing Integration")
    print("-" * 50)
    
    # Create hybrid integration adapter with default settings
    adapter = create_hybrid_integration_adapter(
        gpu_batch_size=400,
        target_throughput=2000,
        gpu_memory_fraction=0.95,
        cpu_allocation_ratio=0.02
    )
    
    # Sample addresses for processing
    addresses = [
        "123 Main Street, Apartment 4B, Near Central Park, New York, NY 10001",
        "456 Oak Avenue, Building C, Unit 12, Los Angeles, CA 90210",
        "789 Pine Road, Suite 5A, Downtown Area, Chicago, IL 60601",
        "321 Elm Street, Floor 2, Business District, Houston, TX 77001",
        "654 Maple Drive, Apt 3C, Residential Area, Phoenix, AZ 85001"
    ]
    
    try:
        # Initialize hybrid processing
        print("📋 Initializing hybrid processing system...")
        adapter.initialize()
        
        # Process addresses
        print(f"🔄 Processing {len(addresses)} addresses...")
        result = adapter.process_address_list(addresses)
        
        # Display results
        success_count = sum(1 for addr in result.parsed_addresses if addr.parse_success)
        print(f"✅ Processing completed: {success_count}/{len(addresses)} addresses parsed")
        
        if result.performance_metrics:
            print(f"📈 Performance: {result.performance_metrics.throughput_rate:.1f} addr/sec")
            print(f"🎯 GPU Utilization: {result.performance_metrics.gpu_utilization:.1f}%")
        
        # Show sample parsed results
        print("\n📄 Sample Parsed Results:")
        for i, parsed_addr in enumerate(result.parsed_addresses[:3]):
            if parsed_addr.parse_success:
                print(f"   {i+1}. Society: {parsed_addr.society_name}")
                print(f"      Locality: {parsed_addr.locality}")
                print(f"      City: {parsed_addr.city}")
                print(f"      PIN: {parsed_addr.pin_code}")
            else:
                print(f"   {i+1}. Parse failed: {parsed_addr.parse_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False
        
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        adapter.shutdown()


def example_csv_file_integration():
    """Example: CSV file processing with hybrid integration."""
    print("\n🚀 Example: CSV File Processing Integration")
    print("-" * 50)
    
    # Look for existing CSV files to process
    csv_files = list(Path('.').glob('export_customer_address_store_p*.csv'))
    if not csv_files:
        csv_files = list(Path('.').glob('sample_*.csv'))
    
    if not csv_files:
        print("⚠️ No CSV files found for processing example")
        return True
    
    # Use the first available CSV file
    input_file = csv_files[0]
    output_file = Path('examples') / f'hybrid_processed_{input_file.name}'
    
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output file: {output_file}")
    
    # Create adapter with optimized settings for file processing
    adapter = create_hybrid_integration_adapter(
        gpu_batch_size=600,
        target_throughput=2500,
        gpu_memory_fraction=0.95
    )
    
    try:
        # Initialize processing
        adapter.initialize()
        
        # Process CSV file with comprehensive output
        print("🔄 Processing CSV file with hybrid system...")
        result, consolidation_stats = adapter.process_csv_file(
            str(input_file),
            str(output_file),
            consolidate_results=True,
            comprehensive_output=True
        )
        
        # Display processing results
        success_count = sum(1 for addr in result.parsed_addresses if addr.parse_success)
        print(f"✅ CSV processing completed:")
        print(f"   Total addresses: {len(result.parsed_addresses)}")
        print(f"   Successfully parsed: {success_count}")
        print(f"   Success rate: {(success_count/len(result.parsed_addresses)*100):.1f}%")
        print(f"   Processing time: {result.processing_time:.2f} seconds")
        
        if result.performance_metrics:
            print(f"   Throughput: {result.performance_metrics.throughput_rate:.1f} addr/sec")
            print(f"   GPU utilization: {result.performance_metrics.gpu_utilization:.1f}%")
        
        if consolidation_stats:
            print(f"   Consolidated groups: {consolidation_stats.total_groups}")
            print(f"   Average group size: {consolidation_stats.avg_records_per_group:.1f}")
        
        # Show optimization suggestions
        if result.optimization_suggestions:
            print("\n💡 Optimization Suggestions:")
            for suggestion in result.optimization_suggestions[:3]:
                print(f"   - {suggestion}")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV processing failed: {e}")
        return False
        
    finally:
        adapter.shutdown()


def example_legacy_compatibility():
    """Example: Backward compatibility with legacy processing."""
    print("\n🚀 Example: Legacy Compatibility")
    print("-" * 50)
    
    # Legacy configuration (old format)
    legacy_config = {
        'batch_size': 300,
        'gpu_memory': 0.9,
        'cpu_ratio': 0.03,
        'target_rate': 1800
    }
    
    print("📋 Using legacy configuration format:")
    for key, value in legacy_config.items():
        print(f"   {key}: {value}")
    
    # Create legacy processor adapter
    legacy_processor = create_legacy_processor(legacy_config)
    
    # Sample addresses
    addresses = [
        "123 Legacy Street, Old Town, NY 10001",
        "456 Compatibility Avenue, Backward City, CA 90210"
    ]
    
    try:
        # Use legacy interface
        print("\n🔄 Using legacy processing interface...")
        legacy_processor.setup_parser()
        
        results = legacy_processor.parse_addresses(addresses)
        
        # Display results in legacy format
        print(f"✅ Legacy processing completed: {len(results)} results")
        
        for i, result in enumerate(results):
            if result.parse_success:
                print(f"   Address {i+1}: Parsed successfully")
            else:
                print(f"   Address {i+1}: Parse failed - {result.parse_error}")
        
        # Get legacy statistics
        stats = legacy_processor.get_stats()
        print(f"\n📊 Legacy Statistics:")
        print(f"   Total processed: {stats['total_processed']}")
        print(f"   Processor type: {stats['processor_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy processing failed: {e}")
        return False
        
    finally:
        legacy_processor.cleanup_parser()


def example_configuration_migration():
    """Example: Configuration migration from legacy format."""
    print("\n🚀 Example: Configuration Migration")
    print("-" * 50)
    
    # Create sample legacy configuration file
    legacy_config_content = """
# Legacy configuration file
batch_size=250
gpu_memory=0.85
cpu_ratio=0.04
target_rate=1500
queue_size=8
"""
    
    legacy_config_file = Path('examples') / 'legacy_config.txt'
    migrated_config_file = Path('examples') / 'hybrid_config.json'
    
    try:
        # Create legacy config file
        legacy_config_file.parent.mkdir(exist_ok=True)
        with open(legacy_config_file, 'w') as f:
            f.write(legacy_config_content)
        
        print(f"📁 Created legacy config: {legacy_config_file}")
        
        # Migrate configuration
        print("🔄 Migrating configuration to hybrid format...")
        migrated_config = migrate_existing_configuration(
            str(legacy_config_file),
            str(migrated_config_file)
        )
        
        print(f"✅ Configuration migrated to: {migrated_config_file}")
        print("📋 Migrated configuration:")
        print(f"   GPU batch size: {migrated_config.gpu_batch_size}")
        print(f"   GPU memory fraction: {migrated_config.gpu_memory_fraction}")
        print(f"   CPU allocation ratio: {migrated_config.cpu_allocation_ratio}")
        print(f"   Target throughput: {migrated_config.target_throughput}")
        print(f"   GPU queue size: {migrated_config.gpu_queue_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration migration failed: {e}")
        return False
        
    finally:
        # Cleanup
        if legacy_config_file.exists():
            legacy_config_file.unlink()


def example_performance_comparison():
    """Example: Performance comparison between processing methods."""
    print("\n🚀 Example: Performance Comparison")
    print("-" * 50)
    
    # Test addresses
    test_addresses = [
        f"Address {i}, Street {i%10}, City {i%5}, State {i%3}, PIN {10000+i}"
        for i in range(50)  # 50 addresses for comparison
    ]
    
    print(f"🧪 Testing with {len(test_addresses)} addresses")
    
    # Test different configurations
    configs = [
        ("Conservative", {"gpu_batch_size": 200, "target_throughput": 1000}),
        ("Balanced", {"gpu_batch_size": 400, "target_throughput": 2000}),
        ("Aggressive", {"gpu_batch_size": 800, "target_throughput": 3000})
    ]
    
    results = []
    
    for config_name, config_params in configs:
        print(f"\n🔧 Testing {config_name} configuration...")
        
        adapter = create_hybrid_integration_adapter(**config_params)
        
        try:
            adapter.initialize()
            
            import time
            start_time = time.time()
            
            result = adapter.process_address_list(test_addresses)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            success_count = sum(1 for addr in result.parsed_addresses if addr.parse_success)
            throughput = len(test_addresses) / processing_time if processing_time > 0 else 0
            
            results.append({
                'config': config_name,
                'processing_time': processing_time,
                'throughput': throughput,
                'success_count': success_count,
                'gpu_utilization': result.performance_metrics.gpu_utilization if result.performance_metrics else 0
            })
            
            print(f"   ⏱️ Processing time: {processing_time:.2f}s")
            print(f"   🚀 Throughput: {throughput:.1f} addr/sec")
            print(f"   ✅ Success rate: {(success_count/len(test_addresses)*100):.1f}%")
            
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
            
        finally:
            adapter.shutdown()
    
    # Display comparison
    if results:
        print(f"\n📊 Performance Comparison Summary:")
        print(f"{'Configuration':<12} {'Time (s)':<10} {'Throughput':<12} {'Success %':<10}")
        print("-" * 50)
        
        for result in results:
            success_rate = (result['success_count'] / len(test_addresses)) * 100
            print(f"{result['config']:<12} {result['processing_time']:<10.2f} "
                  f"{result['throughput']:<12.1f} {success_rate:<10.1f}")
    
    return True


def main():
    """Run all integration examples."""
    print("🎯 GPU-CPU HYBRID PROCESSING INTEGRATION EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("Basic Integration", example_basic_integration),
        ("CSV File Processing", example_csv_file_integration),
        ("Legacy Compatibility", example_legacy_compatibility),
        ("Configuration Migration", example_configuration_migration),
        ("Performance Comparison", example_performance_comparison)
    ]
    
    results = []
    
    for example_name, example_func in examples:
        try:
            print(f"\n🎯 Running: {example_name}")
            success = example_func()
            results.append((example_name, success))
            
            if success:
                print(f"✅ {example_name} completed successfully")
            else:
                print(f"❌ {example_name} failed")
                
        except Exception as e:
            print(f"💥 {example_name} crashed: {e}")
            results.append((example_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INTEGRATION EXAMPLES SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for example_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {example_name}")
    
    print(f"\n🎯 Overall: {successful}/{total} examples completed successfully")
    
    if successful == total:
        print("🎉 All integration examples passed! The hybrid processing system is ready for use.")
    else:
        print("⚠️ Some examples failed. Check the output above for details.")
    
    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)