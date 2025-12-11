#!/usr/bin/env python3
"""Command-line interface for performance benchmarking and optimization tools.

Provides easy access to hardware detection, benchmark execution, optimization analysis,
and performance regression testing through a comprehensive CLI interface.

Requirements: 5.4, 5.5, 6.4
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Optional

try:
    from .performance_benchmarking import (
        PerformanceBenchmarker, HardwareDetector, BenchmarkConfiguration,
        create_benchmark_configuration, run_quick_benchmark
    )
    from .hybrid_processor import ProcessingConfiguration
    from .hybrid_config import load_hybrid_config
except ImportError:
    # Fallback for direct execution
    from performance_benchmarking import (
        PerformanceBenchmarker, HardwareDetector, BenchmarkConfiguration,
        create_benchmark_configuration, run_quick_benchmark
    )
    from hybrid_processor import ProcessingConfiguration
    from hybrid_config import load_hybrid_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_detect_hardware(args):
    """Command: Detect hardware capabilities."""
    print("🔍 Hardware Detection")
    print("=" * 50)
    
    detector = HardwareDetector()
    capabilities = detector.detect_capabilities()
    
    print(f"\nSystem Information:")
    print(f"  Platform: {capabilities.system_platform}")
    print(f"  CPU Model: {capabilities.cpu_model}")
    print(f"  CPU Cores: {capabilities.cpu_count}")
    print(f"  Total Memory: {capabilities.total_memory_gb:.1f} GB")
    print(f"  Available Memory: {capabilities.available_memory_gb:.1f} GB")
    
    print(f"\nGPU Information:")
    print(f"  GPU Count: {capabilities.gpu_count}")
    print(f"  CUDA Available: {capabilities.cuda_available}")
    print(f"  CUDA Version: {capabilities.cuda_version}")
    
    if capabilities.gpu_devices:
        for i, gpu in enumerate(capabilities.gpu_devices):
            print(f"  GPU {i}: {gpu.name} ({gpu.memory_total_mb:.0f} MB)")
    
    print(f"\nPerformance Estimates:")
    print(f"  Estimated Max Throughput: {capabilities.estimated_max_throughput} addr/sec")
    print(f"  Recommended GPU Batch Size: {capabilities.recommended_gpu_batch_size}")
    print(f"  Recommended CPU Allocation: {capabilities.recommended_cpu_allocation:.3f}")
    print(f"  Recommended Queue Size: {capabilities.recommended_queue_size}")
    
    print(f"\nHardware Limitations:")
    limitations = []
    if capabilities.memory_limited:
        limitations.append("Memory Limited")
    if capabilities.gpu_memory_limited:
        limitations.append("GPU Memory Limited")
    if capabilities.cpu_limited:
        limitations.append("CPU Limited")
    
    if limitations:
        print(f"  {', '.join(limitations)}")
    else:
        print(f"  No significant limitations detected")
    
    # Generate optimized configuration
    optimized_config = capabilities.generate_optimization_config()
    
    print(f"\nOptimized Configuration:")
    print(f"  GPU Batch Size: {optimized_config.gpu_batch_size}")
    print(f"  GPU Queue Size: {optimized_config.gpu_queue_size}")
    print(f"  CPU Allocation Ratio: {optimized_config.cpu_allocation_ratio:.3f}")
    print(f"  Target Throughput: {optimized_config.target_throughput}")
    print(f"  GPU Memory Fraction: {optimized_config.gpu_memory_fraction:.2f}")
    print(f"  Model Compilation: {optimized_config.enable_model_compilation}")
    print(f"  Half Precision: {optimized_config.use_half_precision}")
    
    # Save configuration if requested
    if args.save_config:
        config_data = {
            'gpu': {
                'batch_size': optimized_config.gpu_batch_size,
                'queue_size': optimized_config.gpu_queue_size,
                'memory_fraction': optimized_config.gpu_memory_fraction,
                'num_streams': optimized_config.num_gpu_streams,
                'enable_compilation': optimized_config.enable_model_compilation,
                'use_half_precision': optimized_config.use_half_precision,
                'enable_cudnn_benchmark': optimized_config.enable_cudnn_benchmark,
                'enable_tensor_float32': optimized_config.enable_tensor_float32
            },
            'cpu': {
                'allocation_ratio': optimized_config.cpu_allocation_ratio,
                'batch_size': optimized_config.cpu_batch_size,
                'worker_count': optimized_config.cpu_worker_count
            },
            'performance': {
                'target_throughput': optimized_config.target_throughput,
                'gpu_utilization_threshold': optimized_config.gpu_utilization_threshold
            }
        }
        
        config_path = Path(args.save_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\n💾 Optimized configuration saved to: {config_path}")


def cmd_quick_benchmark(args):
    """Command: Run quick benchmark."""
    print("🚀 Quick Benchmark")
    print("=" * 50)
    
    # Generate test addresses if not provided
    test_addresses = None
    if args.addresses_file:
        try:
            with open(args.addresses_file, 'r') as f:
                test_addresses = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(test_addresses)} addresses from {args.addresses_file}")
        except Exception as e:
            print(f"❌ Failed to load addresses from {args.addresses_file}: {e}")
            return 1
    
    try:
        suite = run_quick_benchmark(test_addresses)
        
        print(f"\n📊 Quick Benchmark Results:")
        print(f"  Suite Name: {suite.suite_name}")
        print(f"  Total Tests: {len(suite.benchmark_results)}")
        print(f"  Best Configuration: {suite.best_configuration}")
        
        if suite.performance_summary:
            print(f"\n📈 Performance Summary:")
            for config_name, stats in suite.performance_summary.items():
                print(f"  {config_name}:")
                print(f"    Average Throughput: {stats['avg_throughput']:.1f} addr/sec")
                print(f"    Peak Throughput: {stats['max_throughput']:.1f} addr/sec")
                print(f"    Average GPU Utilization: {stats['avg_gpu_utilization']:.1f}%")
                print(f"    Success Rate: {stats['success_rate']:.1f}%")
        
        if suite.optimization_recommendations:
            print(f"\n💡 Optimization Recommendations:")
            for i, rec in enumerate(suite.optimization_recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Generate and display report
        report = suite.generate_report()
        print(f"\n📄 Full Report:")
        print(report)
        
        return 0
        
    except Exception as e:
        print(f"❌ Quick benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_comprehensive_benchmark(args):
    """Command: Run comprehensive benchmark."""
    print("📊 Comprehensive Benchmark")
    print("=" * 50)
    
    # Create benchmarker
    benchmarker = PerformanceBenchmarker(output_directory=args.output_dir)
    
    # Load test addresses
    test_addresses = None
    if args.addresses_file:
        try:
            with open(args.addresses_file, 'r') as f:
                test_addresses = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(test_addresses)} addresses from {args.addresses_file}")
        except Exception as e:
            print(f"❌ Failed to load addresses: {e}")
            return 1
    
    # Create benchmark configuration
    address_counts = args.address_counts or [100, 500, 1000, 2000]
    
    config = create_benchmark_configuration(
        test_name=args.test_name,
        address_counts=address_counts,
        iterations=args.iterations
    )
    
    print(f"Benchmark Configuration:")
    print(f"  Test Name: {config.test_name}")
    print(f"  Address Counts: {config.address_counts}")
    print(f"  Iterations per Test: {config.iterations_per_test}")
    print(f"  Output Directory: {args.output_dir}")
    
    try:
        suite = benchmarker.run_comprehensive_benchmark(config, test_addresses)
        
        print(f"\n📊 Comprehensive Benchmark Results:")
        print(f"  Suite Name: {suite.suite_name}")
        print(f"  Total Tests: {len(suite.benchmark_results)}")
        print(f"  Best Configuration: {suite.best_configuration}")
        
        if suite.performance_summary:
            print(f"\n📈 Performance Summary:")
            for config_name, stats in suite.performance_summary.items():
                print(f"  {config_name}:")
                print(f"    Average Throughput: {stats['avg_throughput']:.1f} ± {stats['throughput_std']:.1f} addr/sec")
                print(f"    Peak Throughput: {stats['max_throughput']:.1f} addr/sec")
                print(f"    Average GPU Utilization: {stats['avg_gpu_utilization']:.1f}%")
                print(f"    Success Rate: {stats['success_rate']:.1f}%")
                print(f"    Tests Run: {stats['test_count']}")
        
        # Generate optimization suggestions
        suggestions = benchmarker.generate_optimization_suggestions(suite, args.target_throughput)
        
        if suggestions:
            print(f"\n💡 Optimization Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        print(f"\n📄 Full report saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Comprehensive benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_regression_test(args):
    """Command: Run regression test."""
    print("📈 Performance Regression Test")
    print("=" * 50)
    
    # Load baseline results
    try:
        with open(args.baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Reconstruct baseline suite (simplified)
        from performance_benchmarking import BenchmarkSuite
        baseline_suite = BenchmarkSuite(suite_name=baseline_data['suite_name'])
        baseline_suite.performance_summary = baseline_data.get('performance_summary', {})
        
        print(f"Loaded baseline: {baseline_suite.suite_name}")
        print(f"Baseline configurations: {len(baseline_suite.performance_summary)}")
        
    except Exception as e:
        print(f"❌ Failed to load baseline file: {e}")
        return 1
    
    # Create current benchmark configuration
    benchmarker = PerformanceBenchmarker(output_directory=args.output_dir)
    
    address_counts = args.address_counts or [100, 500, 1000]
    
    current_config = create_benchmark_configuration(
        test_name=f"regression_test_{args.test_name}",
        address_counts=address_counts,
        iterations=args.iterations
    )
    
    try:
        regression_analysis = benchmarker.run_regression_test(
            baseline_suite, current_config, args.threshold
        )
        
        print(f"\n📈 Regression Test Results:")
        print(f"  Regression Detected: {regression_analysis['regression_detected']}")
        print(f"  Overall Performance Change: {regression_analysis['overall_change']:.1f}%")
        print(f"  Regression Threshold: {regression_analysis['threshold']*100:.0f}%")
        print(f"  Summary: {regression_analysis['summary']}")
        
        if regression_analysis['configuration_changes']:
            print(f"\n📊 Configuration Changes:")
            for config, changes in regression_analysis['configuration_changes'].items():
                print(f"  {config}:")
                print(f"    Baseline: {changes['baseline_throughput']:.1f} addr/sec")
                print(f"    Current: {changes['current_throughput']:.1f} addr/sec")
                print(f"    Change: {changes['change_percent']:.1f}%")
                print(f"    Regression: {'❌ Yes' if changes['regression'] else '✅ No'}")
        
        # Return appropriate exit code
        return 1 if regression_analysis['regression_detected'] else 0
        
    except Exception as e:
        print(f"❌ Regression test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_analyze_results(args):
    """Command: Analyze existing benchmark results."""
    print("🔍 Benchmark Results Analysis")
    print("=" * 50)
    
    result_files = []
    
    # Collect result files
    if args.result_files:
        result_files = args.result_files
    elif args.results_dir:
        results_path = Path(args.results_dir)
        result_files = list(results_path.glob("benchmark_*.json"))
    
    if not result_files:
        print("❌ No result files found")
        return 1
    
    print(f"Analyzing {len(result_files)} result files...")
    
    # Load and analyze results
    suites = []
    benchmarker = PerformanceBenchmarker()
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct suite (simplified)
            from performance_benchmarking import BenchmarkSuite
            suite = BenchmarkSuite(suite_name=data['suite_name'])
            suite.performance_summary = data.get('performance_summary', {})
            suite.best_configuration = data.get('best_configuration')
            suite.optimization_recommendations = data.get('optimization_recommendations', [])
            
            suites.append(suite)
            print(f"  ✅ Loaded: {suite.suite_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to load {file_path}: {e}")
    
    if not suites:
        print("❌ No valid result files loaded")
        return 1
    
    # Compare configurations
    comparison = benchmarker.compare_configurations(suites, "Multi-Suite Analysis")
    
    print(f"\n📊 Multi-Suite Analysis Results:")
    print(f"  Suites Analyzed: {comparison['suite_count']}")
    print(f"  Best Overall Suite: {comparison['best_overall_suite']}")
    
    if comparison['performance_rankings']:
        print(f"\n🏆 Performance Rankings:")
        for i, (suite_name, throughput) in enumerate(comparison['performance_rankings'], 1):
            print(f"  {i}. {suite_name}: {throughput:.1f} addr/sec")
    
    if comparison['optimization_insights']:
        print(f"\n💡 Optimization Insights:")
        for insight in comparison['optimization_insights']:
            print(f"  - {insight}")
    
    # Generate cross-suite optimization suggestions
    if suites:
        best_suite = suites[0]  # Assume first is best for simplicity
        suggestions = benchmarker.generate_optimization_suggestions(best_suite, args.target_throughput)
        
        if suggestions:
            print(f"\n🔧 Optimization Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Performance Benchmarking and Optimization Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect hardware and save optimized config
  python benchmark_cli.py detect --save-config config/optimized.json
  
  # Run quick benchmark
  python benchmark_cli.py quick --addresses-file test_addresses.txt
  
  # Run comprehensive benchmark
  python benchmark_cli.py benchmark --test-name production_test --iterations 5
  
  # Run regression test
  python benchmark_cli.py regression --baseline baseline_results.json --threshold 0.1
  
  # Analyze existing results
  python benchmark_cli.py analyze --results-dir benchmark_results/
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Hardware detection command
    detect_parser = subparsers.add_parser('detect', help='Detect hardware capabilities')
    detect_parser.add_argument('--save-config', metavar='FILE',
                              help='Save optimized configuration to file')
    
    # Quick benchmark command
    quick_parser = subparsers.add_parser('quick', help='Run quick benchmark')
    quick_parser.add_argument('--addresses-file', metavar='FILE',
                             help='File containing test addresses (one per line)')
    
    # Comprehensive benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run comprehensive benchmark')
    benchmark_parser.add_argument('--test-name', default='comprehensive_benchmark',
                                 help='Name for the benchmark test')
    benchmark_parser.add_argument('--addresses-file', metavar='FILE',
                                 help='File containing test addresses')
    benchmark_parser.add_argument('--address-counts', type=int, nargs='+',
                                 help='List of address counts to test')
    benchmark_parser.add_argument('--iterations', type=int, default=3,
                                 help='Number of iterations per test')
    benchmark_parser.add_argument('--output-dir', default='benchmark_results',
                                 help='Output directory for results')
    benchmark_parser.add_argument('--target-throughput', type=float, default=2000.0,
                                 help='Target throughput for optimization suggestions')
    
    # Regression test command
    regression_parser = subparsers.add_parser('regression', help='Run regression test')
    regression_parser.add_argument('--baseline-file', required=True, metavar='FILE',
                                  help='Baseline benchmark results file')
    regression_parser.add_argument('--test-name', default='regression_test',
                                  help='Name for the regression test')
    regression_parser.add_argument('--address-counts', type=int, nargs='+',
                                  help='List of address counts to test')
    regression_parser.add_argument('--iterations', type=int, default=2,
                                  help='Number of iterations per test')
    regression_parser.add_argument('--threshold', type=float, default=0.1,
                                  help='Regression threshold (0.1 = 10%)')
    regression_parser.add_argument('--output-dir', default='regression_results',
                                  help='Output directory for results')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze benchmark results')
    analyze_group = analyze_parser.add_mutually_exclusive_group(required=True)
    analyze_group.add_argument('--result-files', nargs='+', metavar='FILE',
                              help='Specific result files to analyze')
    analyze_group.add_argument('--results-dir', metavar='DIR',
                              help='Directory containing result files')
    analyze_parser.add_argument('--target-throughput', type=float, default=2000.0,
                               help='Target throughput for optimization suggestions')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    try:
        if args.command == 'detect':
            return cmd_detect_hardware(args)
        elif args.command == 'quick':
            return cmd_quick_benchmark(args)
        elif args.command == 'benchmark':
            return cmd_comprehensive_benchmark(args)
        elif args.command == 'regression':
            return cmd_regression_test(args)
        elif args.command == 'analyze':
            return cmd_analyze_results(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())