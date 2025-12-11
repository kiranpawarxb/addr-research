"""Test AsynchronousQueueManager implementation.

Tests the AsynchronousQueueManager class for sustained GPU utilization
through continuous batch feeding and asynchronous processing.
"""

import time
import logging
from typing import List
from src.asynchronous_queue_manager import AsynchronousQueueManager, QueueStatus
from src.hybrid_processor import ProcessingConfiguration
from src.models import ParsedAddress

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mock_gpu_processing_function(batch: List[str]) -> List[ParsedAddress]:
    """Mock GPU processing function for testing.
    
    Simulates GPU processing by creating ParsedAddress objects
    with a small delay to simulate processing time.
    
    Args:
        batch: List of address strings to process
        
    Returns:
        List of ParsedAddress objects
    """
    # Simulate GPU processing time
    time.sleep(0.01 * len(batch))  # 10ms per address
    
    results = []
    for i, address in enumerate(batch):
        parsed = ParsedAddress(
            city=f"TestCity{i}",
            state="TestState",
            country="India",
            pin_code="123456",
            note=f"Processed by mock GPU function: {address[:50]}",
            parse_success=True
        )
        results.append(parsed)
    
    return results


def test_queue_initialization():
    """Test GPU queue initialization."""
    logger.info("🧪 Testing queue initialization...")
    
    config = ProcessingConfiguration(
        gpu_queue_size=5,
        gpu_batch_size=100,
        num_gpu_streams=2
    )
    
    queue_manager = AsynchronousQueueManager(config)
    
    # Test initialization
    assert queue_manager.initialize_gpu_queues(), "Queue initialization should succeed"
    assert queue_manager.is_initialized, "Queue manager should be initialized"
    
    # Test queue status
    status = queue_manager.get_queue_status()
    assert isinstance(status, QueueStatus), "Should return QueueStatus object"
    assert status.max_queue_size == 5, "Max queue size should match configuration"
    
    # Cleanup
    queue_manager.shutdown()
    
    logger.info("✅ Queue initialization test passed")


def test_data_feeder():
    """Test data feeder functionality."""
    logger.info("🧪 Testing data feeder...")
    
    config = ProcessingConfiguration(
        gpu_queue_size=10,
        gpu_batch_size=100,  # Must be 100-1000
        num_gpu_streams=2
    )
    
    queue_manager = AsynchronousQueueManager(config)
    queue_manager.initialize_gpu_queues()
    
    # Create test addresses
    test_addresses = [f"Test Address {i}" for i in range(200)]
    
    # Start data feeder
    assert queue_manager.start_data_feeder(test_addresses), "Data feeder should start successfully"
    
    # Wait for some batches to be queued
    time.sleep(0.5)
    
    # Check queue status
    status = queue_manager.get_queue_status()
    assert status.input_queue_size > 0, "Input queue should have batches"
    
    logger.info(f"Input queue size: {status.input_queue_size}")
    
    # Cleanup
    queue_manager.shutdown()
    
    logger.info("✅ Data feeder test passed")


def test_gpu_workers():
    """Test GPU workers functionality."""
    logger.info("🧪 Testing GPU workers...")
    
    config = ProcessingConfiguration(
        gpu_queue_size=8,
        gpu_batch_size=100,  # Must be 100-1000
        num_gpu_streams=2
    )
    
    queue_manager = AsynchronousQueueManager(config)
    queue_manager.initialize_gpu_queues()
    
    # Create test addresses
    test_addresses = [f"Test Address {i}" for i in range(100)]
    
    # Start data feeder
    queue_manager.start_data_feeder(test_addresses)
    
    # Start GPU workers
    assert queue_manager.start_gpu_workers(
        processing_function=mock_gpu_processing_function
    ), "GPU workers should start successfully"
    
    # Wait for processing
    time.sleep(2.0)
    
    # Check status
    status = queue_manager.get_queue_status()
    logger.info(f"Active workers: {status.active_workers}")
    logger.info(f"Total processed: {status.total_processed}")
    logger.info(f"Processing rate: {status.processing_rate:.1f} addr/sec")
    
    assert status.active_workers > 0, "Should have active workers"
    
    # Cleanup
    queue_manager.shutdown()
    
    logger.info("✅ GPU workers test passed")


def test_result_collection():
    """Test result collection functionality."""
    logger.info("🧪 Testing result collection...")
    
    config = ProcessingConfiguration(
        gpu_queue_size=6,
        gpu_batch_size=120,  # Must be 100-1000
        num_gpu_streams=2
    )
    
    queue_manager = AsynchronousQueueManager(config)
    queue_manager.initialize_gpu_queues()
    
    # Create test addresses
    test_addresses = [f"Test Address {i}" for i in range(90)]
    
    # Start processing pipeline
    queue_manager.start_data_feeder(test_addresses)
    queue_manager.start_gpu_workers(processing_function=mock_gpu_processing_function)
    
    # Collect results
    results = queue_manager.collect_results(timeout=5.0)
    
    logger.info(f"Collected {len(results)} results")
    
    # Verify results
    assert len(results) > 0, "Should collect some results"
    
    for result in results[:5]:  # Check first 5 results
        assert isinstance(result, ParsedAddress), "Results should be ParsedAddress objects"
        assert result.parse_success, "Mock processing should succeed"
        assert result.city.startswith("TestCity"), "Should have test city name"
    
    # Cleanup
    queue_manager.shutdown()
    
    logger.info("✅ Result collection test passed")


def test_full_pipeline():
    """Test complete asynchronous processing pipeline."""
    logger.info("🧪 Testing full asynchronous pipeline...")
    
    config = ProcessingConfiguration(
        gpu_queue_size=10,
        gpu_batch_size=150,  # Must be 100-1000
        num_gpu_streams=3
    )
    
    queue_manager = AsynchronousQueueManager(config)
    queue_manager.initialize_gpu_queues()
    
    # Create larger test dataset
    test_addresses = [f"Full Pipeline Test Address {i}" for i in range(400)]
    
    start_time = time.time()
    
    # Start complete pipeline
    queue_manager.start_data_feeder(test_addresses)
    queue_manager.start_gpu_workers(processing_function=mock_gpu_processing_function)
    
    # Monitor processing
    for i in range(10):  # Monitor for 10 seconds max
        status = queue_manager.get_queue_status()
        logger.info(f"Status check {i+1}: "
                   f"Input={status.input_queue_size}, "
                   f"Output={status.output_queue_size}, "
                   f"Processed={status.total_processed}, "
                   f"Rate={status.processing_rate:.1f} addr/sec")
        
        if status.total_processed >= len(test_addresses):
            break
        
        time.sleep(1.0)
    
    # Collect all results
    results = queue_manager.collect_results(timeout=10.0)
    
    total_time = time.time() - start_time
    throughput = len(results) / total_time
    
    logger.info(f"Pipeline completed in {total_time:.2f}s")
    logger.info(f"Processed {len(results)}/{len(test_addresses)} addresses")
    logger.info(f"Throughput: {throughput:.1f} addresses/second")
    
    # Verify results
    assert len(results) > 0, "Should process some addresses"
    success_count = sum(1 for r in results if r.parse_success)
    success_rate = success_count / len(results) * 100
    
    logger.info(f"Success rate: {success_rate:.1f}%")
    assert success_rate > 90, "Should have high success rate"
    
    # Get final statistics
    stats = queue_manager.get_processing_statistics()
    logger.info(f"Final statistics: {stats}")
    
    # Cleanup
    queue_manager.shutdown()
    
    logger.info("✅ Full pipeline test passed")


def test_queue_status_monitoring():
    """Test queue status monitoring functionality."""
    logger.info("🧪 Testing queue status monitoring...")
    
    config = ProcessingConfiguration(
        gpu_queue_size=5,
        gpu_batch_size=100,  # Must be 100-1000
        num_gpu_streams=2
    )
    
    queue_manager = AsynchronousQueueManager(config)
    queue_manager.initialize_gpu_queues()
    
    # Test initial status
    status = queue_manager.get_queue_status()
    assert status.input_queue_size == 0, "Initial input queue should be empty"
    assert status.output_queue_size == 0, "Initial output queue should be empty"
    assert status.active_workers == 0, "No workers should be active initially"
    
    # Start processing
    test_addresses = [f"Status Test Address {i}" for i in range(60)]
    queue_manager.start_data_feeder(test_addresses)
    queue_manager.start_gpu_workers(processing_function=mock_gpu_processing_function)
    
    # Monitor status changes
    for i in range(5):
        status = queue_manager.get_queue_status()
        logger.info(f"Status {i+1}: Input={status.input_queue_size}, "
                   f"Output={status.output_queue_size}, "
                   f"Workers={status.active_workers}, "
                   f"Utilization={status.queue_utilization:.2f}")
        
        if i == 0:
            assert status.active_workers > 0, "Should have active workers"
        
        time.sleep(0.5)
    
    # Cleanup
    queue_manager.shutdown()
    
    logger.info("✅ Queue status monitoring test passed")


if __name__ == "__main__":
    logger.info("🚀 Starting AsynchronousQueueManager tests...")
    
    try:
        test_queue_initialization()
        test_data_feeder()
        test_gpu_workers()
        test_result_collection()
        test_full_pipeline()
        test_queue_status_monitoring()
        
        logger.info("🎉 All AsynchronousQueueManager tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise