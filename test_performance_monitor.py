#!/usr/bin/env python3
"""
Test script for PerformanceMonitor functionality.
Tests GPU utilization tracking, throughput calculation, queue monitoring, and report generation.
"""

import time
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.performance_monitor import PerformanceMonitor, QueueStatus, GPUStats, PerformanceReport
from src.hybrid_processor import ProcessingConfiguration, PerformanceMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_performance_monitor_initialization():
    """Test PerformanceMonitor initialization with configuration."""
    print("🔧 Testing PerformanceMonitor Initialization")
    print("=" * 60)
    
    # Create test configuration
    config = ProcessingConfiguration(
        gpu_batch_size=400,
        target_throughput=2000,
        performance_log_interval=5,
        gpu_utilization_threshold=0.90
    )
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(config)
    
    print(f"✅ PerformanceMonitor initialized successfully")
    print(f"   - Target throughput: {config.target_throughput} addr/sec")
    print(f"   - Log interval: {config.performance_log_interval} seconds")
    print(f"   - GPU utilization threshold: {config.gpu_utilization_threshold*100:.0f}%")
    print()
    
    return monitor

def test_gpu_utilization_tracking(monitor):
    """Test GPU utilization tracking functionality."""
    print("📊 Testing GPU Utilization Tracking")
    print("=" * 60)
    
    # Track GPU utilization
    gpu_util = monitor.track_gpu_utilization()
    
    print(f"Current GPU Utilization: {gpu_util:.1f}%")
    
    # Get current metrics
    metrics = monitor.get_current_metrics()
    print(f"GPU Memory Usage: {metrics.memory_usage:.1f}%")
    
    # Test multiple samples
    print("\nTaking 5 GPU utilization samples...")
    for i in range(5):
        util = monitor.track_gpu_utilization()
        print(f"  Sample {i+1}: {util:.1f}%")
        time.sleep(1)
    
    print("✅ GPU utilization tracking completed")
    print()

def test_throughput_calculation(monitor):
    """Test throughput rate calculation."""
    print("⚡ Testing Throughput Rate Calculation")
    print("=" * 60)
    
    # Simulate processing addresses
    addresses_batches = [100, 150, 200, 175, 125]
    
    print("Simulating address processing...")
    for i, batch_size in enumerate(addresses_batches):
        # Calculate throughput
        throughput = monitor.calculate_throughput_rate(batch_size)
        
        print(f"  Batch {i+1}: {batch_size} addresses -> {throughput:.1f} addr/sec")
        time.sleep(2)  # Simulate processing time
    
    # Get final metrics
    metrics = monitor.get_current_metrics()
    print(f"\nFinal Metrics:")
    print(f"  Total Processed: {metrics.total_processed}")
    print(f"  Current Throughput: {metrics.throughput_rate:.1f} addr/sec")
    
    print("✅ Throughput calculation completed")
    print()

def test_queue_monitoring(monitor):
    """Test queue status monitoring."""
    print("📋 Testing Queue Status Monitoring")
    print("=" * 60)
    
    # Create mock queue manager for testing
    class MockQueueManager:
        def __init__(self):
            self.input_size = 8
            self.output_size = 2
            
        def get_queue_status(self):
            return {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'max_size': 10,
                'active_workers': 2,
                'data_feeder_active': True,
                'result_collector_active': True
            }
    
    # Set mock queue manager
    mock_queue_manager = MockQueueManager()
    monitor.set_queue_manager(mock_queue_manager)
    
    # Monitor queue status
    queue_status = monitor.monitor_queue_status()
    
    print(f"Queue Status:")
    print(f"  Input Queue: {queue_status.input_queue_size}/{queue_status.max_queue_size}")
    print(f"  Output Queue: {queue_status.output_queue_size}")
    print(f"  Queue Utilization: {queue_status.queue_utilization_percent:.1f}%")
    print(f"  Active Workers: {queue_status.processing_workers_active}")
    print(f"  Data Feeder Active: {queue_status.data_feeder_active}")
    print(f"  Bottleneck Detected: {queue_status.bottleneck_detected}")
    
    if queue_status.bottleneck_detected:
        print(f"  Bottleneck Location: {queue_status.bottleneck_location}")
    
    print("✅ Queue monitoring completed")
    print()

def test_performance_report_generation(monitor):
    """Test comprehensive performance report generation."""
    print("📈 Testing Performance Report Generation")
    print("=" * 60)
    
    # Let monitor collect some data
    print("Collecting performance data for 10 seconds...")
    
    # Simulate some processing activity
    for i in range(10):
        monitor.track_gpu_utilization()
        monitor.calculate_throughput_rate(50)  # 50 addresses per second
        monitor.monitor_queue_status()
        time.sleep(1)
    
    # Generate performance report
    report = monitor.generate_performance_report()
    
    print("\n📊 PERFORMANCE REPORT")
    print("=" * 60)
    print(report.generate_summary())
    
    print(f"\nReport Details:")
    print(f"  Performance Score: {report.performance_score:.1f}/100")
    print(f"  Monitoring Duration: {report.monitoring_duration:.1f} seconds")
    print(f"  GPU Statistics Available: {len(report.gpu_stats)} GPUs")
    
    if report.performance_warnings:
        print(f"  Performance Warnings: {len(report.performance_warnings)}")
    
    if report.optimization_suggestions:
        print(f"  Optimization Suggestions: {len(report.optimization_suggestions)}")
    
    print("✅ Performance report generation completed")
    print()

def test_real_time_monitoring(monitor):
    """Test real-time monitoring with logging."""
    print("🔄 Testing Real-Time Monitoring")
    print("=" * 60)
    
    print("Starting real-time monitoring for 15 seconds...")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate processing activity
    for i in range(15):
        # Simulate processing some addresses
        addresses_processed = 80 + (i * 10)  # Varying load
        monitor.log_performance_update(addresses_processed)
        
        print(f"  Second {i+1}: Processed {addresses_processed} addresses")
        time.sleep(1)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("✅ Real-time monitoring completed")
    print()

def test_performance_thresholds(monitor):
    """Test performance threshold warnings."""
    print("⚠️  Testing Performance Threshold Warnings")
    print("=" * 60)
    
    # Get current configuration
    config = monitor.config
    
    print(f"Performance Thresholds:")
    print(f"  Target Throughput: {config.target_throughput} addr/sec")
    print(f"  GPU Utilization Threshold: {config.gpu_utilization_threshold*100:.0f}%")
    
    # Test with low performance values
    print("\nTesting with low performance values...")
    
    # Simulate low throughput
    low_throughput = monitor.calculate_throughput_rate(50)  # Low throughput
    print(f"  Simulated Low Throughput: {low_throughput:.1f} addr/sec")
    
    # Generate report to see warnings
    report = monitor.generate_performance_report()
    
    if report.performance_warnings:
        print(f"\n⚠️  Performance Warnings Generated:")
        for warning in report.performance_warnings:
            print(f"    - {warning}")
    
    if report.optimization_suggestions:
        print(f"\n💡 Optimization Suggestions:")
        for suggestion in report.optimization_suggestions:
            print(f"    - {suggestion}")
    
    print("✅ Performance threshold testing completed")
    print()

def main():
    """Main test function."""
    print("🚀 PERFORMANCE MONITOR TEST SUITE")
    print("=" * 80)
    print()
    
    try:
        # Initialize performance monitor
        monitor = test_performance_monitor_initialization()
        
        # Run all tests
        test_gpu_utilization_tracking(monitor)
        test_throughput_calculation(monitor)
        test_queue_monitoring(monitor)
        test_performance_report_generation(monitor)
        test_real_time_monitoring(monitor)
        test_performance_thresholds(monitor)
        
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Final performance summary
        final_metrics = monitor.get_current_metrics()
        print(f"\nFinal Performance Summary:")
        print(f"  Total Addresses Processed: {final_metrics.total_processed}")
        print(f"  Final Throughput Rate: {final_metrics.throughput_rate:.1f} addr/sec")
        print(f"  Final GPU Utilization: {final_metrics.gpu_utilization:.1f}%")
        print(f"  Processing Efficiency: {final_metrics.processing_efficiency:.1f}%")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)