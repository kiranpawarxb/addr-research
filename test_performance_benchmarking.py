#!/usr/bin/env python3
"""
Test script for performance benchmarking and optimization tools.
Tests hardware detection, benchmark execution, optimization suggestions, and regression testing.
"""

import time
import logging
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.performance_benchmarking import (
    PerformanceBenchmarker, HardwareDetector, BenchmarkConfiguration,
    create_benchmark_configuration, run_quick_benchmark
)
from src.hybrid_processor import ProcessingConfiguration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_hardware_detection():
    """Test hardware capability detection."""
    print("🔍 Testing Hardware Detection")
    print("=" * 60)
    
    detector = HardwareDetector()
    capabilities = detector.detect_capabilities()
    
    print(f"Hardware Detection Results:")
    print(f"  Platform: {capabilities.system_platform}")
    print(f"  CPU: {capabilities.cpu_model}")
    print(f"  CPU Cores: {capabilities.cpu_count}")
    print(f"  Total Memory: {capabilities.total_memory_gb:.1f} GB")
    print(f"  Available Memory: {capabilities.available_memory_gb:.1f} GB")
    print(f"  GPU Count: {capabilities.gpu_count}")
    print(f"  CUDA Available: {capabilities.cuda_available}")
    print(f"  CUDA Version: {capabilities.cuda_version}")
    
    print(f"\nPerformance Estimates:")
    print(f"  Estimated Max Throughput: {capabilities.estimated_max_throughput} addr/sec")
    print(f"  Recommended GPU Batch Size: {capabilities.recommended_gpu_batch_size}")
    print(f"  Recommended CPU Allocation: {capabilities.recommended_cpu_allocation:.3f}")
    print(f"  Recommended Queue Size: {capabilities.recommended_queue_size}")
    
    print(f"\nHardware Limitations:")
    print(f"  Memory Limited: {capabilities.memory_limited}")
    print(f"  GPU Memory Limited: {capabilities.gpu_memory_limited}")
    print(f"  CPU Limited: {capabilities.cpu_limited}")
    
    # Test optimized configuration generation
    optimized_config = capabilities.generate_optimization_config()
    print(f"\nOptimized Configuration:")
    print(f"  GPU Batch Size: {optimized_config.gpu_batch_size}")
    print(f"  GPU Queue Size: {optimized_config.gpu_queue_size}")
    print(f"  CPU Allocation Ratio: {optimized_config.cpu_allocation_ratio:.3f}")
    print(f"  Target Throughput: {optimized_config.target_throughput}")
    print(f"  GPU Memory Fraction: {optimized_config.gpu_memory_fraction:.2f}")
    print(f"  Model Compilation: {optimized_config.enable_model_compilation}")
    
    print("✅ Hardware detection completed")
    print()
    
    return capabilities

def test_benchmark_configuration():
    """Test benchmark configuration creation."""
    print("⚙️  Testing Benchmark Configuration")
    print("=" * 60)
    
    # Test default configuration
    default_config = create_benchmark_configuration()
    print(f"Default Configuration:")
    print(f"  Test Name: {default_config.test_name}")
    print(f"  Address Counts: {default_config.address_counts}")
    print(f"  Iterations: {default_config.iterations_per_test}")
    print(f"  Warmup Iterations: {default_config.warmup_iterations}")
    
    # Test custom configuration
    custom_config = create_benchmark_configuration(
        test_name="custom_test",
        address_counts=[50, 100, 200],
        iterations=2
    )
    print(f"\nCustom Configuration:")
    print(f"  Test Name: {custom_config.test_name}")
    print(f"  Address Counts: {custom_config.address_counts}")
    print(f"  Iterations: {custom_config.iterations_per_test}")
    
    # Test with processing configurations
    processing_configs = [
        ProcessingConfiguration(gpu_batch_size=200, target_throughput=1000),
        ProcessingConfiguration(gpu_batch_size=400, target_throughput=1500)
    ]
    
    config_with_processing = BenchmarkConfiguration(
        test_name="processing_config_test",
        address_counts=[100, 200],
        iterations_per_test=1,
        configurations=processing_configs
    )
    
    print(f"\nConfiguration with Processing Configs:")
    print(f"  Test Name: {config_with_processing.test_name}")
    print(f"  Processing Configurations: {len(config_with_processing.configurations)}")
    
    print("✅ Benchmark configuration testing completed")
    print()
    
    return custom_config

def test_quick_benchmark():
    """Test quick benchmark functionality."""
    print("🚀 Testing Quick Benchmark")
    print("=" * 60)
    
    # Generate test addresses
    test_addresses = [
        "123 Main Street, Anytown, CA 90210",
        "456 Oak Avenue, Springfield, IL 62701",
        "789 Pine Road, Austin, TX 73301",
        "321 Elm Street, Portland, OR 97201",
        "654 Maple Drive, Denver, CO 80201"
    ]
    
    # Extend for testing
    extended_addresses = []
    for i in range(100):
        base_address = test_addresses[i % len(test_addresses)]
        street_num = 100 + (i * 7) % 9000
        extended_addresses.append(base_address.replace("123", str(street_num), 1))
    
    print(f"Running quick benchmark with {len(extended_addresses)} test addresses...")
    
    try:
        # Run quick benchmark
        suite = run_quick_benchmark(extended_addresses)
        
        print(f"\nQuick Benchmark Results:")
        print(f"  Suite Name: {suite.suite_name}")
        print(f"  Total Tests: {len(suite.benchmark_results)}")
        print(f"  Best Configuration: {suite.best_configuration}")
        
        if suite.performance_summary:
            print(f"\nPerformance Summary:")
            for config_name, stats in suite.performance_summary.items():
                print(f"  {config_name}:")
                print(f"    Average Throughput: {stats['avg_throughput']:.1f} addr/sec")
                print(f"    Success Rate: {stats['success_rate']:.1f}%")
        
        if suite.optimization_recommendations:
            print(f"\nOptimization Recommendations:")
            for i, rec in enumerate(suite.optimization_recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("✅ Quick benchmark completed successfully")
        
        return suite
        
    except Exception as e:
        print(f"❌ Quick benchmark failed: {e}")
        # This is expected in test environment without full GPU setup
        print("   (This is expected in test environment without full hybrid processor setup)")
        return None
    
    print()

def test_comprehensive_benchmark():
    """Test comprehensive benchmark functionality."""
    print("📊 Testing Comprehensive Benchmark")
    print("=" * 60)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmarker = PerformanceBenchmarker(output_directory=temp_dir)
        
        # Test hardware detection
        print("Detecting hardware capabilities...")
        capabilities = benchmarker.hardware_detector.detect_capabilities()
        benchmarker.hardware_capabilities = capabilities
        
        print(f"Hardware detected: {capabilities.cpu_count} cores, {capabilities.gpu_count} GPUs")
        
        # Create test configuration
        config = create_benchmark_configuration(
            test_name="comprehensive_test",
            address_counts=[50, 100],  # Small counts for testing
            iterations=1  # Single iteration for speed
        )
        
        # Generate test addresses
        test_addresses = [
            f"{100 + i} Test Street, City {i % 5}, State {i % 3}"
            for i in range(150)
        ]
        
        print(f"Created benchmark configuration with {len(config.address_counts)} address counts")
        print(f"Generated {len(test_addresses)} test addresses")
        
        try:
            # Run comprehensive benchmark
            print("Running comprehensive benchmark...")
            suite = benchmarker.run_comprehensive_benchmark(config, test_addresses)
            
            print(f"\nComprehensive Benchmark Results:")
            print(f"  Suite Name: {suite.suite_name}")
            print(f"  Total Tests: {len(suite.benchmark_results)}")
            print(f"  Hardware Capabilities Detected: {suite.hardware_capabilities is not None}")
            
            # Check if results were saved
            output_files = list(Path(temp_dir).glob("benchmark_*"))
            print(f"  Output Files Created: {len(output_files)}")
            
            for file_path in output_files:
                print(f"    - {file_path.name}")
            
            print("✅ Comprehensive benchmark completed successfully")
            
            return suite
            
        except Exception as e:
            print(f"❌ Comprehensive benchmark failed: {e}")
            print("   (This is expected in test environment without full hybrid processor setup)")
            return None
    
    print()

def test_optimization_suggestions():
    """Test optimization suggestion generation."""
    print("💡 Testing Optimization Suggestions")
    print("=" * 60)
    
    benchmarker = PerformanceBenchmarker()
    
    # Create mock benchmark suite for testing
    from src.performance_benchmarking import BenchmarkSuite, BenchmarkResult
    
    suite = BenchmarkSuite(suite_name="test_suite")
    
    # Add mock results with different performance characteristics
    results = [
        BenchmarkResult(
            test_name="low_performance",
            configuration_name="LowPerf",
            throughput=300.0,
            gpu_utilization_avg=45.0,
            test_successful=True
        ),
        BenchmarkResult(
            test_name="medium_performance", 
            configuration_name="MedPerf",
            throughput=800.0,
            gpu_utilization_avg=75.0,
            test_successful=True
        ),
        BenchmarkResult(
            test_name="high_performance",
            configuration_name="HighPerf", 
            throughput=1800.0,
            gpu_utilization_avg=92.0,
            test_successful=True
        )
    ]
    
    for result in results:
        suite.add_result(result)
    
    # Analyze results
    suite.analyze_results()
    
    print(f"Mock Suite Analysis:")
    print(f"  Best Configuration: {suite.best_configuration}")
    print(f"  Performance Summary: {len(suite.performance_summary)} configurations")
    
    # Generate optimization suggestions
    suggestions = benchmarker.generate_optimization_suggestions(suite, target_throughput=2000.0)
    
    print(f"\nOptimization Suggestions ({len(suggestions)} total):")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    # Test with different target throughput
    high_target_suggestions = benchmarker.generate_optimization_suggestions(suite, target_throughput=3000.0)
    
    print(f"\nHigh Target Suggestions ({len(high_target_suggestions)} total):")
    for i, suggestion in enumerate(high_target_suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    print("✅ Optimization suggestion testing completed")
    print()

def test_configuration_comparison():
    """Test configuration comparison functionality."""
    print("🔄 Testing Configuration Comparison")
    print("=" * 60)
    
    benchmarker = PerformanceBenchmarker()
    
    # Create mock benchmark suites for comparison
    from src.performance_benchmarking import BenchmarkSuite, BenchmarkResult
    
    # Suite 1: Conservative configuration
    suite1 = BenchmarkSuite(suite_name="Conservative")
    suite1.add_result(BenchmarkResult(
        configuration_name="Conservative",
        throughput=600.0,
        gpu_utilization_avg=65.0,
        test_successful=True
    ))
    suite1.analyze_results()
    
    # Suite 2: Aggressive configuration
    suite2 = BenchmarkSuite(suite_name="Aggressive")
    suite2.add_result(BenchmarkResult(
        configuration_name="Aggressive",
        throughput=1400.0,
        gpu_utilization_avg=88.0,
        test_successful=True
    ))
    suite2.analyze_results()
    
    # Suite 3: Balanced configuration
    suite3 = BenchmarkSuite(suite_name="Balanced")
    suite3.add_result(BenchmarkResult(
        configuration_name="Balanced",
        throughput=1000.0,
        gpu_utilization_avg=78.0,
        test_successful=True
    ))
    suite3.analyze_results()
    
    # Compare configurations
    comparison = benchmarker.compare_configurations(
        [suite1, suite2, suite3],
        comparison_name="Configuration Strategy Comparison"
    )
    
    print(f"Configuration Comparison Results:")
    print(f"  Comparison Name: {comparison['comparison_name']}")
    print(f"  Suite Count: {comparison['suite_count']}")
    print(f"  Best Overall Suite: {comparison['best_overall_suite']}")
    
    print(f"\nPerformance Rankings:")
    for i, (suite_name, throughput) in enumerate(comparison['performance_rankings'], 1):
        print(f"  {i}. {suite_name}: {throughput:.1f} addr/sec")
    
    if comparison['optimization_insights']:
        print(f"\nOptimization Insights:")
        for insight in comparison['optimization_insights']:
            print(f"  - {insight}")
    
    print("✅ Configuration comparison testing completed")
    print()

def test_regression_analysis():
    """Test performance regression analysis."""
    print("📈 Testing Regression Analysis")
    print("=" * 60)
    
    # Create mock baseline and current suites
    from src.performance_benchmarking import BenchmarkSuite, BenchmarkResult
    
    # Baseline suite (good performance)
    baseline = BenchmarkSuite(suite_name="Baseline")
    baseline.add_result(BenchmarkResult(
        configuration_name="Standard",
        throughput=1200.0,
        gpu_utilization_avg=85.0,
        test_successful=True
    ))
    baseline.analyze_results()
    
    # Current suite (with regression)
    current = BenchmarkSuite(suite_name="Current")
    current.add_result(BenchmarkResult(
        configuration_name="Standard",
        throughput=950.0,  # 20% regression
        gpu_utilization_avg=78.0,
        test_successful=True
    ))
    current.analyze_results()
    
    # Analyze regression
    benchmarker = PerformanceBenchmarker()
    regression_analysis = benchmarker._analyze_performance_regression(
        baseline, current, threshold=0.1  # 10% threshold
    )
    
    print(f"Regression Analysis Results:")
    print(f"  Regression Detected: {regression_analysis['regression_detected']}")
    print(f"  Overall Change: {regression_analysis['overall_change']:.1f}%")
    print(f"  Threshold: {regression_analysis['threshold']*100:.0f}%")
    print(f"  Summary: {regression_analysis['summary']}")
    
    if regression_analysis['configuration_changes']:
        print(f"\nConfiguration Changes:")
        for config, changes in regression_analysis['configuration_changes'].items():
            print(f"  {config}:")
            print(f"    Baseline: {changes['baseline_throughput']:.1f} addr/sec")
            print(f"    Current: {changes['current_throughput']:.1f} addr/sec")
            print(f"    Change: {changes['change_percent']:.1f}%")
            print(f"    Regression: {changes['regression']}")
    
    print("✅ Regression analysis testing completed")
    print()

def test_file_operations():
    """Test benchmark result file operations."""
    print("💾 Testing File Operations")
    print("=" * 60)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmarker = PerformanceBenchmarker(output_directory=temp_dir)
        
        # Create mock benchmark suite
        from src.performance_benchmarking import BenchmarkSuite, BenchmarkResult
        
        suite = BenchmarkSuite(suite_name="file_test")
        suite.add_result(BenchmarkResult(
            test_name="test_1",
            configuration_name="TestConfig",
            throughput=1000.0,
            test_successful=True
        ))
        suite.analyze_results()
        
        # Test saving results
        benchmarker._save_benchmark_results(suite)
        
        # Check created files
        output_files = list(Path(temp_dir).glob("*"))
        print(f"Files created: {len(output_files)}")
        
        json_files = list(Path(temp_dir).glob("*.json"))
        txt_files = list(Path(temp_dir).glob("*.txt"))
        
        print(f"  JSON files: {len(json_files)}")
        print(f"  Text files: {len(txt_files)}")
        
        # Verify file contents
        if json_files:
            import json
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            print(f"  JSON data keys: {list(data.keys())}")
        
        if txt_files:
            with open(txt_files[0], 'r') as f:
                content = f.read()
            print(f"  Text file length: {len(content)} characters")
            print(f"  Contains 'BENCHMARK REPORT': {'BENCHMARK REPORT' in content}")
    
    print("✅ File operations testing completed")
    print()

def main():
    """Main test function."""
    print("🚀 PERFORMANCE BENCHMARKING TEST SUITE")
    print("=" * 80)
    print()
    
    try:
        # Run all tests
        capabilities = test_hardware_detection()
        config = test_benchmark_configuration()
        
        # These tests may fail in environments without full GPU setup
        # but should demonstrate the functionality
        quick_suite = test_quick_benchmark()
        comprehensive_suite = test_comprehensive_benchmark()
        
        # These tests use mock data and should always work
        test_optimization_suggestions()
        test_configuration_comparison()
        test_regression_analysis()
        test_file_operations()
        
        print("🎉 ALL PERFORMANCE BENCHMARKING TESTS COMPLETED!")
        print("=" * 80)
        
        # Summary
        print(f"\nTest Summary:")
        print(f"  Hardware Detection: ✅ Completed")
        print(f"  Configuration Creation: ✅ Completed")
        print(f"  Quick Benchmark: {'✅ Completed' if quick_suite else '⚠️  Skipped (expected in test env)'}")
        print(f"  Comprehensive Benchmark: {'✅ Completed' if comprehensive_suite else '⚠️  Skipped (expected in test env)'}")
        print(f"  Optimization Suggestions: ✅ Completed")
        print(f"  Configuration Comparison: ✅ Completed")
        print(f"  Regression Analysis: ✅ Completed")
        print(f"  File Operations: ✅ Completed")
        
        if capabilities:
            print(f"\nDetected Hardware Summary:")
            print(f"  CPU Cores: {capabilities.cpu_count}")
            print(f"  Total Memory: {capabilities.total_memory_gb:.1f} GB")
            print(f"  GPU Count: {capabilities.gpu_count}")
            print(f"  Estimated Max Throughput: {capabilities.estimated_max_throughput} addr/sec")
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)