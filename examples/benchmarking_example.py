#!/usr/bin/env python3
"""
Example usage of performance benchmarking and optimization tools.

This script demonstrates how to use the performance benchmarking system
to detect hardware capabilities, run benchmarks, analyze results, and
generate optimization recommendations.

Requirements: 5.4, 5.5, 6.4
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.performance_benchmarking import (
    PerformanceBenchmarker, HardwareDetector, BenchmarkConfiguration,
    create_benchmark_configuration, run_quick_benchmark
)
from src.hybrid_processor import ProcessingConfiguration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def example_hardware_detection():
    """Example: Hardware capability detection and optimization."""
    print("🔍 EXAMPLE: Hardware Detection and Optimization")
    print("=" * 60)
    
    # Create hardware detector
    detector = HardwareDetector()
    
    # Detect hardware capabilities
    print("Detecting hardware capabilities...")
    capabilities = detector.detect_capabilities()
    
    # Display hardware information
    print(f"\n📊 Hardware Summary:")
    print(f"  CPU: {capabilities.cpu_model} ({capabilities.cpu_count} cores)")
    print(f"  Memory: {capabilities.total_memory_gb:.1f} GB total")
    print(f"  GPUs: {capabilities.gpu_count} devices")
    print(f"  CUDA: {capabilities.cuda_available}")
    
    # Show performance estimates
    print(f"\n⚡ Performance Estimates:")
    print(f"  Estimated Max Throughput: {capabilities.estimated_max_throughput} addr/sec")
    print(f"  Recommended GPU Batch Size: {capabilities.recommended_gpu_batch_size}")
    print(f"  Recommended CPU Allocation: {capabilities.recommended_cpu_allocation:.3f}")
    
    # Generate optimized configuration
    optimized_config = capabilities.generate_optimization_config()
    
    print(f"\n🔧 Optimized Configuration:")
    print(f"  GPU Batch Size: {optimized_config.gpu_batch_size}")
    print(f"  GPU Queue Size: {optimized_config.gpu_queue_size}")
    print(f"  Target Throughput: {optimized_config.target_throughput}")
    print(f"  Model Compilation: {optimized_config.enable_model_compilation}")
    
    return capabilities, optimized_config

def example_quick_benchmark():
    """Example: Quick performance benchmark."""
    print("\n🚀 EXAMPLE: Quick Benchmark")
    print("=" * 60)
    
    # Generate sample test addresses
    test_addresses = [
        "123 Main Street, Anytown, CA 90210",
        "456 Oak Avenue, Springfield, IL 62701", 
        "789 Pine Road, Austin, TX 73301",
        "321 Elm Street, Portland, OR 97201",
        "654 Maple Drive, Denver, CO 80201"
    ]
    
    # Extend for more comprehensive testing
    extended_addresses = []
    for i in range(200):
        base_address = test_addresses[i % len(test_addresses)]
        street_num = 100 + (i * 7) % 9000
        extended_addresses.append(base_address.replace("123", str(street_num), 1))
    
    print(f"Running quick benchmark with {len(extended_addresses)} test addresses...")
    
    try:
        # Run quick benchmark
        suite = run_quick_benchmark(extended_addresses)
        
        print(f"\n📈 Quick Benchmark Results:")
        print(f"  Total Tests: {len(suite.benchmark_results)}")
        print(f"  Best Configuration: {suite.best_configuration}")
        
        if suite.performance_summary:
            for config_name, stats in suite.performance_summary.items():
                print(f"\n  {config_name}:")
                print(f"    Average Throughput: {stats['avg_throughput']:.1f} addr/sec")
                print(f"    Success Rate: {stats['success_rate']:.1f}%")
        
        if suite.optimization_recommendations:
            print(f"\n💡 Optimization Recommendations:")
            for i, rec in enumerate(suite.optimization_recommendations, 1):
                print(f"  {i}. {rec}")
        
        return suite
        
    except Exception as e:
        print(f"❌ Quick benchmark failed: {e}")
        print("   (This is expected in environments without full GPU setup)")
        return None

def example_comprehensive_benchmark():
    """Example: Comprehensive benchmark with multiple configurations."""
    print("\n📊 EXAMPLE: Comprehensive Benchmark")
    print("=" * 60)
    
    # Create benchmarker with output directory
    output_dir = "example_benchmark_results"
    benchmarker = PerformanceBenchmarker(output_directory=output_dir)
    
    # Create multiple test configurations
    configurations = [
        ProcessingConfiguration(
            gpu_batch_size=200,
            target_throughput=1000,
            cpu_allocation_ratio=0.03
        ),
        ProcessingConfiguration(
            gpu_batch_size=400,
            target_throughput=1500,
            cpu_allocation_ratio=0.02,
            enable_model_compilation=True
        ),
        ProcessingConfiguration(
            gpu_batch_size=600,
            target_throughput=2000,
            cpu_allocation_ratio=0.01,
            enable_model_compilation=True,
            use_half_precision=True
        )
    ]
    
    # Create benchmark configuration
    config = BenchmarkConfiguration(
        test_name="comprehensive_example",
        address_counts=[100, 300, 500],  # Smaller counts for example
        iterations_per_test=2,  # Fewer iterations for speed
        configurations=configurations,
        save_detailed_results=True
    )
    
    # Generate test addresses
    test_addresses = [
        f"{100 + i} Example Street, Test City {i % 3}, State {i % 2}"
        for i in range(600)
    ]
    
    print(f"Running comprehensive benchmark...")
    print(f"  Configurations: {len(configurations)}")
    print(f"  Address Counts: {config.address_counts}")
    print(f"  Iterations: {config.iterations_per_test}")
    
    try:
        # Run comprehensive benchmark
        suite = benchmarker.run_comprehensive_benchmark(config, test_addresses)
        
        print(f"\n📈 Comprehensive Benchmark Results:")
        print(f"  Total Tests: {len(suite.benchmark_results)}")
        print(f"  Best Configuration: {suite.best_configuration}")
        
        if suite.performance_summary:
            print(f"\n📊 Performance Summary:")
            for config_name, stats in suite.performance_summary.items():
                print(f"  {config_name}:")
                print(f"    Avg Throughput: {stats['avg_throughput']:.1f} ± {stats['throughput_std']:.1f}")
                print(f"    Peak Throughput: {stats['max_throughput']:.1f}")
                print(f"    Success Rate: {stats['success_rate']:.1f}%")
        
        # Generate optimization suggestions
        suggestions = benchmarker.generate_optimization_suggestions(suite, target_throughput=2000.0)
        
        if suggestions:
            print(f"\n💡 Optimization Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        print(f"\n💾 Results saved to: {output_dir}")
        
        return suite
        
    except Exception as e:
        print(f"❌ Comprehensive benchmark failed: {e}")
        print("   (This is expected in environments without full GPU setup)")
        return None

def example_configuration_comparison():
    """Example: Compare different benchmark configurations."""
    print("\n🔄 EXAMPLE: Configuration Comparison")
    print("=" * 60)
    
    # Create mock benchmark suites for demonstration
    from src.performance_benchmarking import BenchmarkSuite, BenchmarkResult
    
    # Conservative configuration suite
    conservative_suite = BenchmarkSuite(suite_name="Conservative")
    conservative_suite.add_result(BenchmarkResult(
        configuration_name="Conservative",
        throughput=800.0,
        gpu_utilization_avg=70.0,
        test_successful=True
    ))
    conservative_suite.analyze_results()
    
    # Aggressive configuration suite
    aggressive_suite = BenchmarkSuite(suite_name="Aggressive")
    aggressive_suite.add_result(BenchmarkResult(
        configuration_name="Aggressive", 
        throughput=1600.0,
        gpu_utilization_avg=92.0,
        test_successful=True
    ))
    aggressive_suite.analyze_results()
    
    # Balanced configuration suite
    balanced_suite = BenchmarkSuite(suite_name="Balanced")
    balanced_suite.add_result(BenchmarkResult(
        configuration_name="Balanced",
        throughput=1200.0,
        gpu_utilization_avg=82.0,
        test_successful=True
    ))
    balanced_suite.analyze_results()
    
    # Compare configurations
    benchmarker = PerformanceBenchmarker()
    comparison = benchmarker.compare_configurations(
        [conservative_suite, aggressive_suite, balanced_suite],
        comparison_name="Configuration Strategy Analysis"
    )
    
    print(f"Configuration Comparison Results:")
    print(f"  Suites Compared: {comparison['suite_count']}")
    print(f"  Best Overall: {comparison['best_overall_suite']}")
    
    print(f"\n🏆 Performance Rankings:")
    for i, (suite_name, throughput) in enumerate(comparison['performance_rankings'], 1):
        print(f"  {i}. {suite_name}: {throughput:.1f} addr/sec")
    
    if comparison['optimization_insights']:
        print(f"\n💡 Optimization Insights:")
        for insight in comparison['optimization_insights']:
            print(f"  - {insight}")
    
    return comparison

def example_regression_testing():
    """Example: Performance regression testing."""
    print("\n📈 EXAMPLE: Regression Testing")
    print("=" * 60)
    
    # Create mock baseline and current results for demonstration
    from src.performance_benchmarking import BenchmarkSuite, BenchmarkResult
    
    # Baseline suite (good performance)
    baseline_suite = BenchmarkSuite(suite_name="Baseline_v1.0")
    baseline_suite.add_result(BenchmarkResult(
        configuration_name="Standard",
        throughput=1400.0,
        gpu_utilization_avg=88.0,
        test_successful=True
    ))
    baseline_suite.analyze_results()
    
    # Current suite (with slight regression)
    current_suite = BenchmarkSuite(suite_name="Current_v1.1")
    current_suite.add_result(BenchmarkResult(
        configuration_name="Standard",
        throughput=1250.0,  # ~11% regression
        gpu_utilization_avg=82.0,
        test_successful=True
    ))
    current_suite.analyze_results()
    
    # Analyze regression
    benchmarker = PerformanceBenchmarker()
    regression_analysis = benchmarker._analyze_performance_regression(
        baseline_suite, current_suite, threshold=0.1  # 10% threshold
    )
    
    print(f"Regression Analysis Results:")
    print(f"  Regression Detected: {regression_analysis['regression_detected']}")
    print(f"  Overall Change: {regression_analysis['overall_change']:.1f}%")
    print(f"  Threshold: {regression_analysis['threshold']*100:.0f}%")
    print(f"  Summary: {regression_analysis['summary']}")
    
    if regression_analysis['configuration_changes']:
        print(f"\n📊 Detailed Changes:")
        for config, changes in regression_analysis['configuration_changes'].items():
            print(f"  {config}:")
            print(f"    Baseline: {changes['baseline_throughput']:.1f} addr/sec")
            print(f"    Current: {changes['current_throughput']:.1f} addr/sec")
            print(f"    Change: {changes['change_percent']:.1f}%")
            print(f"    Regression: {'❌ Yes' if changes['regression'] else '✅ No'}")
    
    return regression_analysis

def example_optimization_suggestions():
    """Example: Generate optimization suggestions."""
    print("\n💡 EXAMPLE: Optimization Suggestions")
    print("=" * 60)
    
    # Create mock benchmark suite with various performance levels
    from src.performance_benchmarking import BenchmarkSuite, BenchmarkResult
    
    suite = BenchmarkSuite(suite_name="Optimization_Analysis")
    
    # Add results with different performance characteristics
    results = [
        BenchmarkResult(
            configuration_name="LowPerformance",
            throughput=400.0,
            gpu_utilization_avg=35.0,
            test_successful=True
        ),
        BenchmarkResult(
            configuration_name="MediumPerformance",
            throughput=900.0,
            gpu_utilization_avg=68.0,
            test_successful=True
        ),
        BenchmarkResult(
            configuration_name="HighPerformance",
            throughput=1700.0,
            gpu_utilization_avg=91.0,
            test_successful=True
        )
    ]
    
    for result in results:
        suite.add_result(result)
    
    suite.analyze_results()
    
    # Generate optimization suggestions
    benchmarker = PerformanceBenchmarker()
    
    # Test with different target throughputs
    targets = [1500.0, 2000.0, 2500.0]
    
    for target in targets:
        suggestions = benchmarker.generate_optimization_suggestions(suite, target)
        
        print(f"\nOptimization Suggestions (Target: {target:.0f} addr/sec):")
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        else:
            print("  No specific suggestions - performance targets met!")
    
    return suite

def main():
    """Main example function."""
    print("🚀 PERFORMANCE BENCHMARKING EXAMPLES")
    print("=" * 80)
    
    try:
        # Example 1: Hardware Detection
        capabilities, optimized_config = example_hardware_detection()
        
        # Example 2: Quick Benchmark
        quick_suite = example_quick_benchmark()
        
        # Example 3: Comprehensive Benchmark
        comprehensive_suite = example_comprehensive_benchmark()
        
        # Example 4: Configuration Comparison
        comparison = example_configuration_comparison()
        
        # Example 5: Regression Testing
        regression_analysis = example_regression_testing()
        
        # Example 6: Optimization Suggestions
        optimization_suite = example_optimization_suggestions()
        
        print("\n🎉 ALL EXAMPLES COMPLETED!")
        print("=" * 80)
        
        # Summary
        print(f"\nExample Summary:")
        print(f"  Hardware Detection: ✅ Completed")
        print(f"  Quick Benchmark: {'✅ Completed' if quick_suite else '⚠️  Skipped (expected)'}")
        print(f"  Comprehensive Benchmark: {'✅ Completed' if comprehensive_suite else '⚠️  Skipped (expected)'}")
        print(f"  Configuration Comparison: ✅ Completed")
        print(f"  Regression Testing: ✅ Completed")
        print(f"  Optimization Suggestions: ✅ Completed")
        
        print(f"\nKey Insights:")
        if capabilities:
            print(f"  - Detected {capabilities.cpu_count} CPU cores, {capabilities.gpu_count} GPUs")
            print(f"  - Estimated max throughput: {capabilities.estimated_max_throughput} addr/sec")
        
        if comparison:
            print(f"  - Best configuration strategy: {comparison['best_overall_suite']}")
        
        if regression_analysis:
            status = "detected" if regression_analysis['regression_detected'] else "not detected"
            print(f"  - Performance regression: {status}")
        
        print(f"\n📚 Next Steps:")
        print(f"  1. Use 'python src/benchmark_cli.py detect' to analyze your hardware")
        print(f"  2. Run 'python src/benchmark_cli.py quick' for a quick performance test")
        print(f"  3. Use 'python src/benchmark_cli.py benchmark' for comprehensive analysis")
        print(f"  4. Set up regression testing with baseline results")
        
    except Exception as e:
        print(f"❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)