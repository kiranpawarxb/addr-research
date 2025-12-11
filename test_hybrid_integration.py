"""Test integration of AsynchronousQueueManager with HybridProcessor.

Tests the integration between AsynchronousQueueManager and GPUCPUHybridProcessor
to ensure proper initialization and coordination.
"""

import logging
from src.hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hybrid_processor_queue_manager_integration():
    """Test that HybridProcessor properly initializes AsynchronousQueueManager."""
    logger.info("🧪 Testing HybridProcessor + AsynchronousQueueManager integration...")
    
    config = ProcessingConfiguration(
        gpu_queue_size=8,
        gpu_batch_size=200,
        num_gpu_streams=2
    )
    
    # Create hybrid processor
    hybrid_processor = GPUCPUHybridProcessor(config)
    
    try:
        # Initialize hybrid processing (this should initialize the queue manager)
        hybrid_processor.initialize_hybrid_processing()
        
        # Verify queue manager was initialized
        assert hybrid_processor.queue_manager is not None, "Queue manager should be initialized"
        assert hybrid_processor.queue_manager.is_initialized, "Queue manager should be initialized"
        
        # Check queue manager configuration
        assert hybrid_processor.queue_manager.queue_size == 8, "Queue size should match configuration"
        
        # Get queue status
        status = hybrid_processor.queue_manager.get_queue_status()
        assert status.max_queue_size == 8, "Max queue size should match"
        assert status.input_queue_size == 0, "Input queue should be empty initially"
        assert status.output_queue_size == 0, "Output queue should be empty initially"
        
        logger.info("✅ Queue manager integration successful")
        logger.info(f"  Queue size: {status.max_queue_size}")
        logger.info(f"  Input queue: {status.input_queue_size}")
        logger.info(f"  Output queue: {status.output_queue_size}")
        
    finally:
        # Cleanup
        hybrid_processor.shutdown()
    
    logger.info("✅ HybridProcessor + AsynchronousQueueManager integration test passed")


def test_hybrid_processor_initialization_components():
    """Test that all hybrid processor components initialize correctly."""
    logger.info("🧪 Testing HybridProcessor component initialization...")
    
    config = ProcessingConfiguration(
        gpu_queue_size=6,
        gpu_batch_size=150,
        num_gpu_streams=2
    )
    
    hybrid_processor = GPUCPUHybridProcessor(config)
    
    try:
        # Test initialization
        hybrid_processor.initialize_hybrid_processing()
        
        # Verify all components are initialized
        assert hybrid_processor.is_initialized, "Hybrid processor should be initialized"
        assert hybrid_processor.dataset_gpu_processor is not None, "GPU processor should be initialized"
        assert hybrid_processor.queue_manager is not None, "Queue manager should be initialized"
        
        # Test performance monitoring
        metrics = hybrid_processor.monitor_performance()
        assert metrics is not None, "Should return performance metrics"
        
        logger.info("✅ All components initialized successfully")
        
    finally:
        # Cleanup
        hybrid_processor.shutdown()
    
    logger.info("✅ HybridProcessor component initialization test passed")


if __name__ == "__main__":
    logger.info("🚀 Starting HybridProcessor integration tests...")
    
    try:
        test_hybrid_processor_queue_manager_integration()
        test_hybrid_processor_initialization_components()
        
        logger.info("🎉 All HybridProcessor integration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        raise