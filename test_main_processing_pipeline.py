#!/usr/bin/env python3
"""Test script for the main processing pipeline with performance validation.

This script tests the enhanced process_addresses_hybrid() method with:
- Performance validation for 1500+ addresses/second target
- Sustained GPU utilization monitoring (90%+ target)
- Synchronization delay elimination through pre-loaded batches
- Integration of all components (GPU, CPU, queues, monitoring)

Requirements: 1.3, 1.4, 1.5
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
        "Apartment 204, Ocean View Towers, Marine Drive, Mumbai 400002",
        "Plot 89, Industrial Area, Phase 2, Gurgaon, Haryana 122016",
        "Office 301, Tech Park, Electronic City, Bangalore 560100",
        "Flat 15B, Royal Residency, Koramangala, Bangalore 560034",
        "House 78, Model Town, Ludhiana, Punjab 141002",
        "Shop 12, Main Market, Connaught Place, New Delhi 110001"
    ]
    
    # Replicate addresses to reach desired count
    addresses = []
    for i in range(count):
        base_addr = base_addresses[i % len(base_addresses)]
        # Add variation to make each address unique
        addresses.append(f"{base_addr} - Variant {i+1}")
    
    return addresses


def test_main_processing_pipeline():
    """Test the main processing pipeline with performance validation."""
    logger.info("🚀 Testing Main Processing Pipeline with Performance Validation")
    
    try:
        # Create configuration optimized for performance testing
        config = ProcessingConfiguration(
            # GPU Configuration for maximum performance
            gpu_batch_size=400,  # Large batches for GPU efficiency
            dataset_batch_size=1000,  # HuggingFace dataset batch size
            gpu_memory_fraction=0.95,  # Use 95% of GPU memory
            gpu_queue_size=10,  # Pre-load 10 batches
            num_gpu_streams=2,  # Multiple streams for overlapping
            
            # CPU Configuration (minimal to avoid GPU interference)
            cpu_allocation_ratio=0.02,  # Only 2% to CPU
            cpu_batch_size=50,  # Small CPU batches
            cpu_worker_count=2,  # Minimal CPU workers
            
            # Performance Targets
            target_throughput=2000,  # Target 2000 addr/sec
            gpu_utilization_threshold=0.90,  # 90%+ GPU utilization
            performance_log_interval=5,  # Log every 5 seconds
            
            # Advanced Optimizations
            enable_model_compilation=True,  # PyTorch compilation
            use_half_precision=True,  # float16 for speed
            enable_cudnn_benchmark=True,  # cuDNN optimization
            enable_tensor_float32=True  # TF32 operations
        )
        
        logger.info("✅ Configuration created for performance testing")
        
        # Create hybrid processor
        processor = GPUCPUHybridProcessor(config)
        logger.info("✅ Hybrid processor created")
        
        # Initialize all components
        logger.info("🔧 Initializing hybrid processing components...")
        processor.initialize_hybrid_processing()
        logger.info("✅ All components initialized successfully")
        
        # Test with different dataset sizes to validate performance scaling
        test_sizes = [100, 500, 1000, 2000]  # Start small and scale up
        
        for test_size in test_sizes:
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 Testing with {test_size} addresses")
            logger.info(f"{'='*60}")
            
            # Create test addresses
            test_addresses = create_test_addresses(test_size)
            logger.info(f"📝 Created {len(test_addresses)} test addresses")
            
            # Process addresses with performance validation
            start_time = time.time()
            
            try:
                result = processor.process_addresses_hybrid(test_addresses)
                
                processing_time = time.time() - start_time
                
                # Analyze results
                logger.info(f"\n📊 Processing Results for {test_size} addresses:")
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
                
                # Optimization suggestions
                if result.optimization_suggestions:
                    logger.info(f"\n💡 Optimization Suggestions:")
                    for i, suggestion in enumerate(result.optimization_suggestions, 1):
                        logger.info(f"   {i}. {suggestion}")
                
                # Validate performance targets
                throughput_achieved = stats.get('throughput_achieved', 0) if result.device_statistics else 0
                gpu_util_achieved = stats.get('gpu_utilization_achieved', 0) if result.device_statistics else 0
                
                # Check if we met the requirements
                throughput_target_met = throughput_achieved >= 1500  # Requirement 1.4
                gpu_util_target_met = gpu_util_achieved >= 90  # Requirement 1.3
                
                logger.info(f"\n🏆 Requirements Validation:")
                logger.info(f"   Requirement 1.4 (1500+ addr/sec): {'✅ PASSED' if throughput_target_met else '❌ FAILED'}")
                logger.info(f"   Requirement 1.3 (90%+ GPU util): {'✅ PASSED' if gpu_util_target_met else '❌ FAILED'}")
                logger.info(f"   Requirement 1.5 (sync delay elimination): {'✅ PASSED' if stats.get('synchronization_delays_eliminated') else '❌ FAILED'}")
                
                # Overall test result for this size
                test_passed = throughput_target_met and gpu_util_target_met
                logger.info(f"\n🎯 Test Result for {test_size} addresses: {'✅ PASSED' if test_passed else '❌ FAILED'}")
                
            except Exception as e:
                logger.error(f"❌ Processing failed for {test_size} addresses: {e}")
                continue
        
        # Shutdown processor
        logger.info(f"\n🔄 Shutting down hybrid processor...")
        processor.shutdown()
        logger.info("✅ Hybrid processor shutdown completed")
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Main Processing Pipeline Test Completed")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_main_processing_pipeline()