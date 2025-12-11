#!/usr/bin/env python3
"""Test script for DatasetGPUProcessor implementation.

This script tests the DatasetGPUProcessor with a small sample of addresses
to verify that all the advanced optimizations are working correctly.
"""

import logging
import sys
import time
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dataset_gpu_processor():
    """Test DatasetGPUProcessor with sample addresses."""
    try:
        # Import required modules
        from src.hybrid_processor import ProcessingConfiguration
        from src.dataset_gpu_processor import DatasetGPUProcessor
        
        logger.info("🧪 Testing DatasetGPUProcessor implementation...")
        
        # Create test configuration
        config = ProcessingConfiguration(
            gpu_batch_size=100,
            dataset_batch_size=200,
            gpu_memory_fraction=0.90,  # Use 90% for testing
            num_gpu_streams=2,
            enable_model_compilation=False,  # Disable for testing to avoid Triton issues
            use_half_precision=True,
            enable_cudnn_benchmark=True,
            enable_tensor_float32=True
        )
        
        # Sample addresses for testing
        test_addresses = [
            "Flat 301, Nisarg Complex, Near Daisy Mall, Sector 15, Pune, Maharashtra 411001",
            "Shop 12, Chembers Society, Road No 5, Banjara Hills, Hyderabad, Telangana 500034",
            "Unit 205, Green Valley Apartments, Behind City Center Mall, Whitefield, Bangalore 560066",
            "House 45, Phase 2, DLF City, Gurgaon, Haryana 122002",
            "Office 801, Tech Tower, IT Park, Hinjewadi, Pune, Maharashtra 411057"
        ]
        
        logger.info(f"Testing with {len(test_addresses)} sample addresses")
        
        # Initialize processor
        processor = DatasetGPUProcessor(config)
        
        # Setup GPU pipeline
        logger.info("Setting up dataset GPU pipeline...")
        setup_success = processor.setup_dataset_gpu_pipeline()
        
        if not setup_success:
            logger.error("❌ Failed to setup GPU pipeline")
            return False
        
        logger.info("✅ GPU pipeline setup successful")
        
        # Test dataset batching processing
        logger.info("Testing dataset batching processing...")
        start_time = time.time()
        
        results = processor.process_with_dataset_batching(test_addresses)
        
        processing_time = time.time() - start_time
        
        # Analyze results
        success_count = sum(1 for r in results if r.parse_success)
        failed_count = len(results) - success_count
        
        logger.info(f"📊 Processing Results:")
        logger.info(f"  Total addresses: {len(test_addresses)}")
        logger.info(f"  Successful: {success_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Processing time: {processing_time:.2f}s")
        logger.info(f"  Throughput: {len(test_addresses)/processing_time:.1f} addr/sec")
        
        # Print sample results
        logger.info("📋 Sample Results:")
        for i, (addr, result) in enumerate(zip(test_addresses[:3], results[:3])):
            logger.info(f"  Address {i+1}: {addr[:50]}...")
            if result.parse_success:
                logger.info(f"    Society: {result.society_name}")
                logger.info(f"    City: {result.city}")
                logger.info(f"    PIN: {result.pin_code}")
            else:
                logger.info(f"    Error: {result.parse_error}")
        
        # Get processing statistics
        stats = processor.get_processing_statistics()
        logger.info(f"📈 Processing Statistics:")
        logger.info(f"  Model compiled: {stats['model_compiled']}")
        logger.info(f"  GPU streams: {stats['gpu_streams']}")
        logger.info(f"  GPU utilization: {stats['current_gpu_utilization']:.1f}%")
        logger.info(f"  Total processed: {stats['total_processed']}")
        
        # Cleanup
        processor.shutdown()
        
        # Determine success
        if success_count >= len(test_addresses) * 0.8:  # 80% success rate
            logger.info("✅ Test PASSED - DatasetGPUProcessor working correctly")
            return True
        else:
            logger.error("❌ Test FAILED - Low success rate")
            return False
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure all required packages are installed:")
        logger.error("  pip install torch transformers datasets")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting DatasetGPUProcessor test...")
    
    success = test_dataset_gpu_processor()
    
    if success:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()