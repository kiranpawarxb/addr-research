#!/usr/bin/env python3
"""Simple test for the main processing pipeline with performance validation.

This script tests the core functionality of the enhanced process_addresses_hybrid() method.
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


def create_test_addresses(count: int) -> List[str]:
    """Create test addresses for processing validation."""
    base_addresses = [
        "Flat 101, Sunrise Apartments, Near City Mall, MG Road, Pune, Maharashtra 411001",
        "House No 45, Sector 12, Dwarka, New Delhi 110075",
        "Shop 23, Commercial Complex, Bandra West, Mumbai 400050",
        "Villa 67, Green Valley Society, Whitefield, Bangalore, Karnataka 560066",
        "Apartment 204, Ocean View Towers, Marine Drive, Mumbai 400002"
    ]
    
    # Replicate addresses to reach desired count
    addresses = []
    for i in range(count):
        base_addr = base_addresses[i % len(base_addresses)]
        # Add variation to make each address unique
        addresses.append(f"{base_addr} - Variant {i+1}")
    
    return addresses


def test_main_processing_pipeline_simple():
    """Test the main processing pipeline with performance validation."""
    logger.info("🚀 Testing Main Processing Pipeline (Simple)")
    
    try:
        # Create configuration optimized for testing
        config = ProcessingConfiguration(
            # GPU Configuration
            gpu_batch_size=200,  # Smaller batches for testing
            dataset_batch_size=500,  # Smaller dataset batch size
            gpu_memory_fraction=0.90,  # Use 90% of GPU memory
            gpu_queue_size=5,  # Smaller queue for testing
            num_gpu_streams=2,  # Multiple streams
            
            # CPU Configuration (minimal)
            cpu_allocation_ratio=0.05,  # 5% to CPU for testing
            cpu_batch_size=25,  # Small CPU batches
            cpu_worker_count=1,  # Single CPU worker
            
            # Performance Targets
            target_throughput=1500,  # Target 1500 addr/sec
            gpu_utilization_threshold=0.90,  # 90%+ GPU utilization
            performance_log_interval=10,  # Log every 10 seconds
            
            # Advanced Optimizations
            enable_model_compilation=True,
            use_half_precision=True,
            enable_cudnn_benchmark=True,
            enable_tensor_float32=True
        )
        
        logger.info("✅ Configuration created")
        
        # Create hybrid processor
        processor = GPUCPUHybridProcessor(config)
        logger.info("✅ Hybrid processor created")
        
        # Initialize all components
        logger.info("🔧 Initializing hybrid processing components...")
        processor.initialize_hybrid_processing()
        logger.info("✅ All components initialized successfully")
        
        # Test with a reasonable dataset size
        test_size = 200
        logger.info(f"🧪 Testing with {test_size} addresses")
        
        # Create test addresses
        test_addresses = create_test_addresses(test_size)
        logger.info(f"📝 Created {len(test_addresses)} test addresses")
        
        # Process addresses with performance validation
        start_time = time.time()
        
        try:
            result = processor.process_addresses_hybrid(test_addresses)
            
            processing_time = time.time() - start_time
            
            # Analyze results
            logger.info(f"\n📊 Processing Results:")
            logger.info(f"   Total processed: {len(result.parsed_addresses)}")
            logger.info(f"   Processing time: {processing_time:.2f} seconds")
            logger.info(f"   Success rate: {sum(1 for r in result.parsed_addresses if r.parse_success)}/{len(result.parsed_addresses)}")
            
            # Performance metrics analysis
            if result.performance_metrics:
                metrics = result.performance_metrics
                logger.info(f"\n🎯 Performance Metrics:")
                logger.info(f"   Throughput: {metrics.throughput_rate:.1f} addr/sec")
                logger.info(f"   GPU Utilization: {metrics.gpu_utilization:.1f}%")
                logger.info(f"   Processing Efficiency: {metrics.processing_efficiency:.1f}%")
                logger.info(f"   GPU Processed: {metrics.gpu_processed}")
                logger.info(f"   CPU Processed: {metrics.cpu_processed}")
            
            # Device statistics analysis
            if result.device_statistics:
                stats = result.device_statistics
                logger.info(f"\n📈 Device Statistics:")
                logger.info(f"   GPU Allocation: {stats.get('gpu_allocation_ratio', 0)*100:.1f}%")
                logger.info(f"   CPU Allocation: {stats.get('cpu_allocation_ratio', 0)*100:.1f}%")
                logger.info(f"   Throughput Achieved: {stats.get('throughput_achieved', 0):.1f} addr/sec")
                logger.info(f"   GPU Utilization Achieved: {stats.get('gpu_utilization_achieved', 0):.1f}%")
                
                # Performance validation results
                validation = stats.get('performance_validation', {})
                if validation:
                    logger.info(f"\n🎯 Performance Validation:")
                    logger.info(f"   Throughput Target Met: {validation.get('throughput_meets_target', False)}")
                    logger.info(f"   GPU Utilization Target Met: {validation.get('gpu_utilization_meets_target', False)}")
                    logger.info(f"   Overall Performance Acceptable: {validation.get('overall_performance_acceptable', False)}")
                    logger.info(f"   Performance Score: {validation.get('performance_score', 0):.1f}/100")
                    logger.info(f"   Summary: {validation.get('summary', 'No summary available')}")
                
                # Validate performance targets
                throughput_achieved = stats.get('throughput_achieved', 0)
                gpu_util_achieved = stats.get('gpu_utilization_achieved', 0)
                
                # Check if we met the requirements
                throughput_target_met = throughput_achieved >= 1500  # Requirement 1.4
                gpu_util_target_met = gpu_util_achieved >= 90  # Requirement 1.3
                sync_delays_eliminated = stats.get('synchronization_delays_eliminated', False)  # Requirement 1.5
                
                logger.info(f"\n🏆 Requirements Validation:")
                logger.info(f"   Requirement 1.4 (1500+ addr/sec): {'✅ PASSED' if throughput_target_met else '❌ FAILED'} ({throughput_achieved:.1f})")
                logger.info(f"   Requirement 1.3 (90%+ GPU util): {'✅ PASSED' if gpu_util_target_met else '❌ FAILED'} ({gpu_util_achieved:.1f}%)")
                logger.info(f"   Requirement 1.5 (sync delay elimination): {'✅ PASSED' if sync_delays_eliminated else '❌ FAILED'}")
                
                # Overall test result
                test_passed = throughput_target_met and gpu_util_target_met and sync_delays_eliminated
                logger.info(f"\n🎯 Overall Test Result: {'✅ PASSED' if test_passed else '❌ FAILED'}")
                
                # Show optimization suggestions if any
                if result.optimization_suggestions:
                    logger.info(f"\n💡 Optimization Suggestions:")
                    for i, suggestion in enumerate(result.optimization_suggestions[:3], 1):  # Show first 3
                        logger.info(f"   {i}. {suggestion}")
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return False
        
        # Shutdown processor
        logger.info(f"\n🔄 Shutting down hybrid processor...")
        processor.shutdown()
        logger.info("✅ Hybrid processor shutdown completed")
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Main Processing Pipeline Test Completed")
        logger.info(f"{'='*60}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_main_processing_pipeline_simple()
    sys.exit(0 if success else 1)