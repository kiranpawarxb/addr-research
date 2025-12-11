#!/usr/bin/env python3
"""Integration test for hybrid processing setup.

Tests the complete setup of the GPU-CPU hybrid processing architecture
including configuration loading, processor initialization, and basic processing.
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration
from hybrid_config import load_hybrid_config
from hybrid_logging import setup_hybrid_logging


def test_hybrid_setup():
    """Test complete hybrid processing setup."""
    print("=== Hybrid Processing Setup Integration Test ===")
    
    # Test 1: Configuration loading
    print("\n1. Testing configuration loading...")
    try:
        config = load_hybrid_config(
            config_path='config/config.yaml',
            hybrid_config_path='config/hybrid_config.yaml'
        )
        print(f"✓ Configuration loaded successfully")
        print(f"  - GPU batch size: {config.processing_config.gpu_batch_size}")
        print(f"  - CPU allocation: {config.processing_config.cpu_allocation_ratio:.1%}")
        print(f"  - Target throughput: {config.processing_config.target_throughput}")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
    
    # Test 2: Logging setup
    print("\n2. Testing logging setup...")
    try:
        logger = setup_hybrid_logging(level='INFO')
        logger.info("Test log message")
        print("✓ Logging setup successful")
    except Exception as e:
        print(f"✗ Logging setup failed: {e}")
        return False
    
    # Test 3: Processor initialization
    print("\n3. Testing processor initialization...")
    try:
        processor = GPUCPUHybridProcessor(config.processing_config)
        print(f"✓ Processor created successfully")
        print(f"  - Initialized: {processor.is_initialized}")
        print(f"  - Processing active: {processor.processing_active}")
    except Exception as e:
        print(f"✗ Processor initialization failed: {e}")
        return False
    
    # Test 4: Workload distribution
    print("\n4. Testing workload distribution...")
    try:
        test_addresses = [f"Address {i}" for i in range(100)]
        gpu_addrs, cpu_addrs = processor.distribute_workload(test_addresses)
        gpu_ratio = len(gpu_addrs) / len(test_addresses) * 100
        cpu_ratio = len(cpu_addrs) / len(test_addresses) * 100
        
        print(f"✓ Workload distribution successful")
        print(f"  - GPU addresses: {len(gpu_addrs)} ({gpu_ratio:.1f}%)")
        print(f"  - CPU addresses: {len(cpu_addrs)} ({cpu_ratio:.1f}%)")
        
        # Verify GPU gets majority (95%+)
        if gpu_ratio >= 95.0:
            print("✓ GPU allocation meets requirements (95%+)")
        else:
            print(f"✗ GPU allocation below requirements: {gpu_ratio:.1f}% < 95%")
            return False
            
    except Exception as e:
        print(f"✗ Workload distribution failed: {e}")
        return False
    
    # Test 5: Configuration validation
    print("\n5. Testing configuration validation...")
    try:
        # Test valid configuration
        valid_config = ProcessingConfiguration(
            gpu_batch_size=500,
            gpu_memory_fraction=0.90,
            target_throughput=1500
        )
        print("✓ Valid configuration accepted")
        
        # Test invalid configuration (should raise ValueError)
        try:
            invalid_config = ProcessingConfiguration(gpu_batch_size=50)  # Too small
            print("✗ Invalid configuration was accepted (should have failed)")
            return False
        except ValueError:
            print("✓ Invalid configuration properly rejected")
            
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False
    
    # Test 6: Performance metrics
    print("\n6. Testing performance metrics...")
    try:
        metrics = processor.monitor_performance()
        print(f"✓ Performance metrics retrieved")
        print(f"  - GPU utilization: {metrics.gpu_utilization:.1f}%")
        print(f"  - Throughput rate: {metrics.throughput_rate:.1f} addr/sec")
        print(f"  - Total processed: {metrics.total_processed}")
    except Exception as e:
        print(f"✗ Performance metrics failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    print("✓ Core hybrid processing architecture is ready")
    print("✓ Configuration management is working")
    print("✓ Logging system is operational")
    print("✓ Workload distribution meets requirements")
    print("✓ Performance monitoring is functional")
    
    return True


if __name__ == "__main__":
    success = test_hybrid_setup()
    sys.exit(0 if success else 1)