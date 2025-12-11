#!/usr/bin/env python3
"""
Integration test for PerformanceMonitor with GPUCPUHybridProcessor.
Tests the integration between performance monitoring and hybrid processing.
"""

import sys
import os
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_performance_monitor_integration():
    """Test PerformanceMonitor integration with GPUCPUHybridProcessor."""
    print("🔗 Testing PerformanceMonitor Integration")
    print("=" * 60)
    
    # Create test configuration
    config = ProcessingConfiguration(
        gpu_batch_size=200,
        target_throughput=1500,
        performance_log_interval=3,
        gpu_utilization_threshold=0.85
    )
    
    # Initialize hybrid processor
    processor = GPUCPUHybridProcessor(config)
    
    print("✅ GPUCPUHybridProcessor initialized")
    
    try:
        # Initialize hybrid processing (this should initialize PerformanceMonitor)
        processor.initialize_hybrid_processing()
        print("✅ Hybrid processing initialized with PerformanceMonitor")
        
        # Check that performance monitor is available
        if processor.performance_monitor:
            print("✅ PerformanceMonitor successfully integrated")
            
            # Get current metrics
            metrics = processor.monitor_performance()
            print(f"   - Current GPU Utilization: {metrics.gpu_utilization:.1f}%")
            print(f"   - Current Throughput: {metrics.throughput_rate:.1f} addr/sec")
            print(f"   - Total Processed: {metrics.total_processed}")
            
            # Test performance report generation
            if hasattr(processor.performance_monitor, 'generate_performance_report'):
                report = processor.performance_monitor.generate_performance_report()
                print(f"   - Performance Report Generated: Score {report.performance_score:.1f}/100")
            
        else:
            print("❌ PerformanceMonitor not properly integrated")
            return False
        
        print("✅ Integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            processor.shutdown()
            print("✅ Processor shutdown completed")
        except Exception as e:
            print(f"⚠️  Shutdown warning: {e}")

def main():
    """Main test function."""
    print("🚀 PERFORMANCE MONITOR INTEGRATION TEST")
    print("=" * 80)
    print()
    
    success = test_performance_monitor_integration()
    
    if success:
        print("\n🎉 INTEGRATION TEST PASSED!")
        return 0
    else:
        print("\n❌ INTEGRATION TEST FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)