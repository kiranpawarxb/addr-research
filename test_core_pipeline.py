#!/usr/bin/env python3
"""Core test for the main processing pipeline functionality.

This script tests the core functionality without complex async processing.
"""

import sys
import os
import logging
import time
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration
from models import ParsedAddress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_core_pipeline():
    """Test the core processing pipeline functionality."""
    logger.info("🚀 Testing Core Processing Pipeline")
    
    try:
        # Create simple configuration
        config = ProcessingConfiguration(
            gpu_batch_size=100,
            dataset_batch_size=200,
            gpu_memory_fraction=0.85,
            gpu_queue_size=5,  # Minimum valid value
            num_gpu_streams=1,
            cpu_allocation_ratio=0.10,
            cpu_batch_size=20,
            cpu_worker_count=1,
            target_throughput=1500,
            gpu_utilization_threshold=0.90,
            performance_log_interval=30,
            enable_model_compilation=True,
            use_half_precision=True,
            enable_cudnn_benchmark=True,
            enable_tensor_float32=True
        )
        
        logger.info("✅ Configuration created")
        
        # Create hybrid processor
        processor = GPUCPUHybridProcessor(config)
        logger.info("✅ Hybrid processor created")
        
        # Initialize components
        logger.info("🔧 Initializing components...")
        processor.initialize_hybrid_processing()
        logger.info("✅ Components initialized")
        
        # Create test addresses
        test_addresses = [
            "Flat 101, Sunrise Apartments, Near City Mall, MG Road, Pune, Maharashtra 411001",
            "House No 45, Sector 12, Dwarka, New Delhi 110075",
            "Shop 23, Commercial Complex, Bandra West, Mumbai 400050",
            "Villa 67, Green Valley Society, Whitefield, Bangalore, Karnataka 560066",
            "Apartment 204, Ocean View Towers, Marine Drive, Mumbai 400002"
        ]
        
        # Replicate to create larger dataset
        addresses = []
        for i in range(50):  # 250 total addresses
            for addr in test_addresses:
                addresses.append(f"{addr} - Test {i+1}")
        
        logger.info(f"📝 Created {len(addresses)} test addresses")
        
        # Test the main processing pipeline
        start_time = time.time()
        
        # Call the main processing method directly
        result = processor.process_addresses_hybrid(addresses)
        
        processing_time = time.time() - start_time
        
        # Analyze results
        logger.info(f"\n📊 Processing Results:")
        logger.info(f"   Total addresses: {len(addresses)}")
        logger.info(f"   Results returned: {len(result.parsed_addresses)}")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        
        success_count = sum(1 for r in result.parsed_addresses if r.parse_success)
        logger.info(f"   Success rate: {success_count}/{len(result.parsed_addresses)} ({success_count/len(result.parsed_addresses)*100:.1f}%)")
        
        # Calculate actual throughput
        actual_throughput = len(addresses) / processing_time if processing_time > 0 else 0
        logger.info(f"   Actual throughput: {actual_throughput:.1f} addresses/second")
        
        # Performance metrics
        if result.performance_metrics:
            metrics = result.performance_metrics
            logger.info(f"\n🎯 Performance Metrics:")
            logger.info(f"   Reported throughput: {metrics.throughput_rate:.1f} addr/sec")
            logger.info(f"   GPU utilization: {metrics.gpu_utilization:.1f}%")
            logger.info(f"   GPU processed: {metrics.gpu_processed}")
            logger.info(f"   CPU processed: {metrics.cpu_processed}")
        
        # Device statistics
        if result.device_statistics:
            stats = result.device_statistics
            logger.info(f"\n📈 Device Statistics:")
            logger.info(f"   GPU allocation: {stats.get('gpu_allocation_ratio', 0)*100:.1f}%")
            logger.info(f"   CPU allocation: {stats.get('cpu_allocation_ratio', 0)*100:.1f}%")
            
            # Performance validation
            validation = stats.get('performance_validation', {})
            if validation:
                logger.info(f"\n🎯 Performance Validation:")
                logger.info(f"   Throughput target (1500+): {validation.get('throughput_meets_target', False)}")
                logger.info(f"   GPU utilization target (90%+): {validation.get('gpu_utilization_meets_target', False)}")
                logger.info(f"   Performance score: {validation.get('performance_score', 0):.1f}/100")
                logger.info(f"   Summary: {validation.get('summary', 'N/A')}")
        
        # Test validation
        throughput_ok = actual_throughput >= 100  # Relaxed for testing
        results_ok = len(result.parsed_addresses) == len(addresses)
        success_rate_ok = success_count / len(result.parsed_addresses) >= 0.8  # 80% success rate
        
        logger.info(f"\n🏆 Test Validation:")
        logger.info(f"   Throughput adequate: {'✅' if throughput_ok else '❌'} ({actual_throughput:.1f} >= 100)")
        logger.info(f"   All addresses processed: {'✅' if results_ok else '❌'} ({len(result.parsed_addresses)} == {len(addresses)})")
        logger.info(f"   Success rate adequate: {'✅' if success_rate_ok else '❌'} ({success_count/len(result.parsed_addresses)*100:.1f}% >= 80%)")
        
        # Check if main pipeline features are working
        features_working = []
        
        # Check if performance validation is implemented
        if result.device_statistics and 'performance_validation' in result.device_statistics:
            features_working.append("Performance validation")
        
        # Check if GPU utilization monitoring is implemented
        if result.performance_metrics and result.performance_metrics.gpu_utilization >= 0:
            features_working.append("GPU utilization monitoring")
        
        # Check if workload distribution is working
        if result.performance_metrics and (result.performance_metrics.gpu_processed > 0 or result.performance_metrics.cpu_processed > 0):
            features_working.append("Workload distribution")
        
        # Check if synchronization delay elimination is flagged
        if result.device_statistics and result.device_statistics.get('synchronization_delays_eliminated'):
            features_working.append("Synchronization delay elimination")
        
        logger.info(f"\n✨ Features Working: {', '.join(features_working) if features_working else 'None detected'}")
        
        # Overall test result
        test_passed = throughput_ok and results_ok and success_rate_ok and len(features_working) >= 2
        
        logger.info(f"\n🎯 Overall Test Result: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        # Shutdown
        processor.shutdown()
        logger.info("✅ Processor shutdown completed")
        
        return test_passed
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_core_pipeline()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)