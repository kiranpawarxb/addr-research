"""Performance benchmarking and optimization tools for GPU-CPU hybrid processing.

This module provides comprehensive performance benchmarking utilities for different hardware
configurations, optimization suggestion generation, performance regression testing,
hardware capability detection, and performance comparison analysis.

Requirements: 5.4, 5.5, 6.4
"""

import logging
import time
import json
import os
import subprocess
import platform
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, Future
import statistics
from pathlib import Path

try:
    from .models import ParsedAddress
    from .hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration, PerformanceMetrics
    from .performance_monitor import PerformanceReport, GPUStats
    from .hybrid_config import HybridProcessingConfig
except ImportError:
    # Fallback for direct execution
    from models import ParsedAddress
    from hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration, PerformanceMetrics
    from performance_monitor import PerformanceReport, GPUStats
    from hybrid_config import HybridProcessingConfig


@dataclass
class HardwareCapabilities:
    """Hardware capability detection and configuration.
    
    Detects system hardware capabilities and provides automatic configuration
    recommendations for optimal performance.
    
    Requirements: 5.4, 5.5
    """
    # System Information
    system_platform: str = ""
    cpu_count: int = 0
    cpu_model: str = ""
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    
    # GPU Information
    gpu_count: int = 0
    gpu_devices: List[GPUStats] = field(default_factory=list)
    cuda_available: bool = False
    cuda_version: str = ""
    
    # Performance Capabilities
    estimated_max_throughput: int = 0
    recommended_gpu_batch_size: int = 400
    recommended_cpu_allocation: float = 0.02
    recommended_queue_size: int = 10
    
    # Hardware Limitations
    memory_limited: bool = False
    gpu_memory_limited: bool = False
    cpu_limited: bool = False
    
    def __post_init__(self):
        """Calculate derived capabilities and recommendations."""
        self._calculate_performance_estimates()
        self._detect_limitations()
    
    def _calculate_performance_estimates(self):
        """Calculate estimated performance based on hardware."""
        # Base throughput estimation
        base_throughput = 500  # Conservative baseline
        
        # GPU performance multiplier
        if self.gpu_count > 0 and self.cuda_available:
            # Estimate based on GPU memory and count
            gpu_memory_total = sum(gpu.memory_total_mb for gpu in self.gpu_devices)
            if gpu_memory_total > 8000:  # 8GB+ GPU
                base_throughput *= 4
            elif gpu_memory_total > 4000:  # 4GB+ GPU
                base_throughput *= 2.5
            else:  # Lower-end GPU
                base_throughput *= 1.5
            
            # Multi-GPU bonus
            if self.gpu_count > 1:
                base_throughput *= min(self.gpu_count * 0.7, 2.0)
        
        # CPU performance multiplier
        if self.cpu_count >= 16:
            base_throughput *= 1.3
        elif self.cpu_count >= 8:
            base_throughput *= 1.1
        
        # Memory performance factor
        if self.total_memory_gb >= 32:
            base_throughput *= 1.2
        elif self.total_memory_gb >= 16:
            base_throughput *= 1.1
        
        self.estimated_max_throughput = int(base_throughput)
        
        # Adjust recommendations based on capabilities
        if self.gpu_count > 0:
            gpu_memory_per_device = sum(gpu.memory_total_mb for gpu in self.gpu_devices) / self.gpu_count
            if gpu_memory_per_device > 8000:
                self.recommended_gpu_batch_size = 600
                self.recommended_queue_size = 15
            elif gpu_memory_per_device > 4000:
                self.recommended_gpu_batch_size = 400
                self.recommended_queue_size = 10
            else:
                self.recommended_gpu_batch_size = 200
                self.recommended_queue_size = 8
        
        # CPU allocation based on core count
        if self.cpu_count >= 16:
            self.recommended_cpu_allocation = 0.01  # Lower CPU allocation for high-core systems
        elif self.cpu_count >= 8:
            self.recommended_cpu_allocation = 0.02
        else:
            self.recommended_cpu_allocation = 0.03
    
    def _detect_limitations(self):
        """Detect hardware limitations that may affect performance."""
        # Memory limitations
        if self.total_memory_gb < 8:
            self.memory_limited = True
        
        # GPU memory limitations
        if self.gpu_devices:
            min_gpu_memory = min(gpu.memory_total_mb for gpu in self.gpu_devices)
            if min_gpu_memory < 4000:  # Less than 4GB GPU memory
                self.gpu_memory_limited = True
        
        # CPU limitations
        if self.cpu_count < 4:
            self.cpu_limited = True
    
    def generate_optimization_config(self) -> ProcessingConfiguration:
        """Generate optimized ProcessingConfiguration based on hardware capabilities.
        
        Returns:
            ProcessingConfiguration optimized for detected hardware
        """
        return ProcessingConfiguration(
            gpu_batch_size=self.recommended_gpu_batch_size,
            gpu_queue_size=self.recommended_queue_size,
            cpu_allocation_ratio=self.recommended_cpu_allocation,
            target_throughput=min(self.estimated_max_throughput, 3000),  # Cap at 3000
            gpu_memory_fraction=0.90 if self.gpu_memory_limited else 0.95,
            num_gpu_streams=min(self.gpu_count * 2, 4) if self.gpu_count > 0 else 2,
            cpu_worker_count=max(2, self.cpu_count // 4) if not self.cpu_limited else 1,
            enable_model_compilation=not self.memory_limited,
            use_half_precision=True,  # Always enable for memory efficiency
            enable_cudnn_benchmark=self.cuda_available,
            enable_tensor_float32=self.cuda_available
        )


@dataclass
class BenchmarkConfiguration:
    """Configuration for performance benchmarking tests.
    
    Defines parameters for comprehensive performance benchmarking
    across different scenarios and hardware configurations.
    """
    # Test Parameters
    test_name: str = "default"
    address_counts: List[int] = field(default_factory=lambda: [100, 500, 1000, 2000])
    iterations_per_test: int = 3
    warmup_iterations: int = 1
    
    # Processing Configurations to Test
    configurations: List[ProcessingConfiguration] = field(default_factory=list)
    
    # Hardware Scenarios
    test_gpu_only: bool = True
    test_cpu_only: bool = True
    test_hybrid: bool = True
    
    # Performance Thresholds
    min_throughput_threshold: float = 500.0
    target_throughput_threshold: float = 1500.0
    max_acceptable_latency: float = 10.0  # seconds
    
    # Output Configuration
    save_detailed_results: bool = True
    generate_comparison_charts: bool = False  # Requires matplotlib
    output_directory: str = "benchmark_results"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test run.
    
    Contains comprehensive performance data from a benchmark execution
    including timing, throughput, resource utilization, and error information.
    """
    # Test Identification
    test_name: str = ""
    configuration_name: str = ""
    address_count: int = 0
    iteration: int = 0
    
    # Performance Metrics
    processing_time: float = 0.0
    throughput: float = 0.0
    gpu_utilization_avg: float = 0.0
    gpu_utilization_peak: float = 0.0
    cpu_utilization_avg: float = 0.0
    memory_usage_peak_gb: float = 0.0
    
    # Processing Results
    addresses_processed: int = 0
    success_rate: float = 100.0
    error_count: int = 0
    
    # Hardware Information
    gpu_devices_used: int = 0
    cpu_cores_used: int = 0
    
    # Detailed Metrics
    performance_report: Optional[PerformanceReport] = None
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Status
    test_successful: bool = True
    error_message: str = ""


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results with analysis and comparisons.
    
    Aggregates multiple benchmark results and provides comprehensive
    analysis, performance comparisons, and optimization recommendations.
    """
    # Suite Information
    suite_name: str = ""
    execution_timestamp: float = field(default_factory=time.time)
    hardware_capabilities: Optional[HardwareCapabilities] = None
    
    # Results
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    
    # Analysis
    best_configuration: Optional[str] = None
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    regression_analysis: Dict[str, Any] = field(default_factory=dict)
    optimization_recommendations: List[str] = field(default_factory=list)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to the suite."""
        self.benchmark_results.append(result)
    
    def analyze_results(self):
        """Analyze all benchmark results and generate insights."""
        if not self.benchmark_results:
            return
        
        # Group results by configuration
        config_results = {}
        for result in self.benchmark_results:
            if result.configuration_name not in config_results:
                config_results[result.configuration_name] = []
            config_results[result.configuration_name].append(result)
        
        # Calculate performance statistics for each configuration
        config_stats = {}
        for config_name, results in config_results.items():
            successful_results = [r for r in results if r.test_successful]
            if successful_results:
                throughputs = [r.throughput for r in successful_results]
                gpu_utils = [r.gpu_utilization_avg for r in successful_results]
                
                config_stats[config_name] = {
                    'avg_throughput': statistics.mean(throughputs),
                    'max_throughput': max(throughputs),
                    'min_throughput': min(throughputs),
                    'throughput_std': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                    'avg_gpu_utilization': statistics.mean(gpu_utils) if gpu_utils else 0,
                    'success_rate': len(successful_results) / len(results) * 100,
                    'test_count': len(results)
                }
        
        self.performance_summary = config_stats
        
        # Find best configuration
        if config_stats:
            best_config = max(config_stats.keys(), 
                            key=lambda k: config_stats[k]['avg_throughput'])
            self.best_configuration = best_config
        
        # Generate optimization recommendations
        self._generate_optimization_recommendations(config_stats)
    
    def _generate_optimization_recommendations(self, config_stats: Dict[str, Any]):
        """Generate optimization recommendations based on benchmark results."""
        recommendations = []
        
        if not config_stats:
            recommendations.append("No successful benchmark results to analyze")
            self.optimization_recommendations = recommendations
            return
        
        # Find performance patterns
        best_throughput = max(stats['avg_throughput'] for stats in config_stats.values())
        best_gpu_util = max(stats['avg_gpu_utilization'] for stats in config_stats.values())
        
        # Throughput recommendations
        if best_throughput < 1000:
            recommendations.append(
                "Low throughput detected. Consider: enabling GPU acceleration, "
                "increasing batch sizes, or optimizing model compilation."
            )
        elif best_throughput < 1500:
            recommendations.append(
                "Moderate throughput achieved. Consider: fine-tuning batch sizes, "
                "optimizing queue management, or enabling advanced GPU features."
            )
        
        # GPU utilization recommendations
        if best_gpu_util < 70:
            recommendations.append(
                "Low GPU utilization detected. Consider: increasing batch sizes, "
                "reducing CPU allocation ratio, or optimizing data feeding."
            )
        elif best_gpu_util > 95:
            recommendations.append(
                "Very high GPU utilization detected. Monitor for thermal throttling "
                "and consider optimizing memory usage."
            )
        
        # Configuration-specific recommendations
        for config_name, stats in config_stats.items():
            if stats['success_rate'] < 90:
                recommendations.append(
                    f"Configuration '{config_name}' has low success rate ({stats['success_rate']:.1f}%). "
                    f"Check for memory limitations or processing errors."
                )
        
        self.optimization_recommendations = recommendations
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.benchmark_results:
            return "No benchmark results available."
        
        report = f"""
PERFORMANCE BENCHMARK REPORT
============================
Suite: {self.suite_name}
Executed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.execution_timestamp))}
Total Tests: {len(self.benchmark_results)}

HARDWARE CONFIGURATION:
"""
        
        if self.hardware_capabilities:
            hw = self.hardware_capabilities
            report += f"""
- Platform: {hw.system_platform}
- CPU: {hw.cpu_model} ({hw.cpu_count} cores)
- Memory: {hw.total_memory_gb:.1f} GB total, {hw.available_memory_gb:.1f} GB available
- GPUs: {hw.gpu_count} devices, CUDA: {hw.cuda_available}
- Estimated Max Throughput: {hw.estimated_max_throughput} addr/sec
"""
        
        report += f"\nPERFORMACE SUMMARY:\n"
        
        if self.performance_summary:
            for config_name, stats in self.performance_summary.items():
                report += f"""
{config_name}:
  - Average Throughput: {stats['avg_throughput']:.1f} ± {stats['throughput_std']:.1f} addr/sec
  - Peak Throughput: {stats['max_throughput']:.1f} addr/sec
  - Average GPU Utilization: {stats['avg_gpu_utilization']:.1f}%
  - Success Rate: {stats['success_rate']:.1f}%
  - Tests Run: {stats['test_count']}
"""
        
        if self.best_configuration:
            report += f"\nBEST CONFIGURATION: {self.best_configuration}\n"
        
        if self.optimization_recommendations:
            report += f"\nOPTIMIZATION RECOMMENDATIONS:\n"
            for i, rec in enumerate(self.optimization_recommendations, 1):
                report += f"{i}. {rec}\n"
        
        return report.strip()


class HardwareDetector:
    """Hardware capability detection and automatic configuration.
    
    Detects system hardware capabilities including CPU, memory, and GPU
    specifications to provide automatic configuration recommendations.
    
    Requirements: 5.4, 5.5
    """
    
    def __init__(self):
        """Initialize hardware detector."""
        self.logger = logging.getLogger(__name__)
    
    def detect_capabilities(self) -> HardwareCapabilities:
        """Detect comprehensive hardware capabilities.
        
        Returns:
            HardwareCapabilities with detected system information
        """
        self.logger.info("🔍 Detecting hardware capabilities...")
        
        capabilities = HardwareCapabilities()
        
        # System information
        capabilities.system_platform = platform.platform()
        capabilities.cpu_count = psutil.cpu_count(logical=True)
        
        # CPU model detection
        try:
            if platform.system() == "Windows":
                capabilities.cpu_model = self._get_windows_cpu_model()
            else:
                capabilities.cpu_model = self._get_linux_cpu_model()
        except Exception as e:
            self.logger.debug(f"Could not detect CPU model: {e}")
            capabilities.cpu_model = "Unknown CPU"
        
        # Memory information
        memory_info = psutil.virtual_memory()
        capabilities.total_memory_gb = memory_info.total / (1024**3)
        capabilities.available_memory_gb = memory_info.available / (1024**3)
        
        # GPU detection
        capabilities.gpu_devices = self._detect_gpu_devices()
        capabilities.gpu_count = len(capabilities.gpu_devices)
        capabilities.cuda_available = self._check_cuda_availability()
        capabilities.cuda_version = self._get_cuda_version()
        
        self.logger.info(f"✅ Hardware detection completed:")
        self.logger.info(f"   CPU: {capabilities.cpu_model} ({capabilities.cpu_count} cores)")
        self.logger.info(f"   Memory: {capabilities.total_memory_gb:.1f} GB")
        self.logger.info(f"   GPUs: {capabilities.gpu_count} devices")
        self.logger.info(f"   CUDA: {capabilities.cuda_available}")
        
        return capabilities
    
    def _get_windows_cpu_model(self) -> str:
        """Get CPU model on Windows."""
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
            return cpu_name.strip()
        except Exception:
            return "Unknown Windows CPU"
    
    def _get_linux_cpu_model(self) -> str:
        """Get CPU model on Linux."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        except Exception:
            pass
        return "Unknown Linux CPU"
    
    def _detect_gpu_devices(self) -> List[GPUStats]:
        """Detect available GPU devices using nvidia-smi."""
        gpu_devices = []
        
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,temperature.gpu,power.limit',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            try:
                                gpu_devices.append(GPUStats(
                                    gpu_id=int(parts[0]) if parts[0] != 'N/A' else 0,
                                    name=parts[1],
                                    memory_total_mb=float(parts[2]) if parts[2] != 'N/A' else 0.0,
                                    temperature_c=float(parts[3]) if parts[3] != 'N/A' else 0.0,
                                    power_limit_w=float(parts[4]) if parts[4] != 'N/A' else 0.0
                                ))
                            except (ValueError, IndexError):
                                continue
        
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            self.logger.debug(f"GPU detection failed: {e}")
        
        return gpu_devices
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_cuda_version(self) -> str:
        """Get CUDA version if available."""
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        return line.strip()
        except Exception:
            pass
        return "Unknown"


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking system.
    
    Provides performance benchmarking utilities for different hardware configurations,
    optimization suggestion generation, and performance regression testing capabilities.
    
    Requirements: 5.4, 5.5, 6.4
    """
    
    def __init__(self, output_directory: str = "benchmark_results"):
        """Initialize performance benchmarker.
        
        Args:
            output_directory: Directory to save benchmark results
        """
        self.logger = logging.getLogger(__name__)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Hardware detection
        self.hardware_detector = HardwareDetector()
        self.hardware_capabilities = None
        
        # Benchmark history for regression testing
        self.benchmark_history: List[BenchmarkSuite] = []
    
    def run_comprehensive_benchmark(self, 
                                  config: BenchmarkConfiguration,
                                  test_addresses: Optional[List[str]] = None) -> BenchmarkSuite:
        """Run comprehensive performance benchmark across multiple configurations.
        
        Args:
            config: Benchmark configuration with test parameters
            test_addresses: Optional list of test addresses (generated if not provided)
            
        Returns:
            BenchmarkSuite with complete results and analysis
            
        Requirements: 5.4, 5.5
        """
        self.logger.info(f"🚀 Starting comprehensive benchmark: {config.test_name}")
        
        # Detect hardware capabilities
        if not self.hardware_capabilities:
            self.hardware_capabilities = self.hardware_detector.detect_capabilities()
        
        # Create benchmark suite
        suite = BenchmarkSuite(
            suite_name=config.test_name,
            hardware_capabilities=self.hardware_capabilities
        )
        
        # Generate test addresses if not provided
        if not test_addresses:
            test_addresses = self._generate_test_addresses()
        
        # Generate configurations if not provided
        if not config.configurations:
            config.configurations = self._generate_test_configurations()
        
        # Run benchmarks for each configuration and address count
        total_tests = len(config.configurations) * len(config.address_counts) * config.iterations_per_test
        test_count = 0
        
        for configuration in config.configurations:
            config_name = self._get_configuration_name(configuration)
            self.logger.info(f"📊 Testing configuration: {config_name}")
            
            for address_count in config.address_counts:
                # Select test addresses for this run
                test_subset = test_addresses[:address_count]
                
                for iteration in range(config.iterations_per_test):
                    test_count += 1
                    self.logger.info(f"   Test {test_count}/{total_tests}: "
                                   f"{address_count} addresses, iteration {iteration + 1}")
                    
                    # Run single benchmark
                    result = self._run_single_benchmark(
                        configuration, test_subset, config_name, 
                        address_count, iteration, config.warmup_iterations > 0
                    )
                    
                    suite.add_result(result)
                    
                    # Log immediate results
                    if result.test_successful:
                        self.logger.info(f"     ✅ {result.throughput:.1f} addr/sec, "
                                       f"GPU: {result.gpu_utilization_avg:.1f}%")
                    else:
                        self.logger.warning(f"     ❌ Test failed: {result.error_message}")
        
        # Analyze results
        suite.analyze_results()
        
        # Save results
        if config.save_detailed_results:
            self._save_benchmark_results(suite)
        
        # Add to history for regression testing
        self.benchmark_history.append(suite)
        
        self.logger.info(f"✅ Benchmark completed: {config.test_name}")
        if suite.best_configuration:
            best_stats = suite.performance_summary[suite.best_configuration]
            self.logger.info(f"🏆 Best configuration: {suite.best_configuration} "
                           f"({best_stats['avg_throughput']:.1f} addr/sec)")
        
        return suite
    
    def run_regression_test(self, 
                           baseline_suite: BenchmarkSuite,
                           current_config: BenchmarkConfiguration,
                           regression_threshold: float = 0.1) -> Dict[str, Any]:
        """Run performance regression testing against baseline results.
        
        Args:
            baseline_suite: Baseline benchmark suite for comparison
            current_config: Current benchmark configuration to test
            regression_threshold: Acceptable performance regression (0.1 = 10%)
            
        Returns:
            Dictionary with regression analysis results
            
        Requirements: 6.4
        """
        self.logger.info("🔍 Running performance regression test...")
        
        # Run current benchmark
        current_suite = self.run_comprehensive_benchmark(current_config)
        
        # Compare with baseline
        regression_analysis = self._analyze_performance_regression(
            baseline_suite, current_suite, regression_threshold
        )
        
        # Update suite with regression analysis
        current_suite.regression_analysis = regression_analysis
        
        self.logger.info(f"📈 Regression test completed:")
        self.logger.info(f"   Performance change: {regression_analysis['overall_change']:.1f}%")
        self.logger.info(f"   Regression detected: {regression_analysis['regression_detected']}")
        
        return regression_analysis
    
    def generate_optimization_suggestions(self, 
                                        suite: BenchmarkSuite,
                                        target_throughput: float = 2000.0) -> List[str]:
        """Generate optimization suggestions based on benchmark results.
        
        Args:
            suite: Benchmark suite with performance results
            target_throughput: Target throughput for optimization
            
        Returns:
            List of optimization suggestions
            
        Requirements: 5.5
        """
        suggestions = []
        
        if not suite.performance_summary:
            suggestions.append("No benchmark results available for analysis")
            return suggestions
        
        # Analyze best performing configuration
        if suite.best_configuration:
            best_stats = suite.performance_summary[suite.best_configuration]
            best_throughput = best_stats['avg_throughput']
            
            # Throughput analysis
            if best_throughput < target_throughput * 0.5:
                suggestions.append(
                    f"Throughput significantly below target ({best_throughput:.1f} vs {target_throughput:.1f}). "
                    f"Consider: hardware upgrade, enabling GPU acceleration, or optimizing batch processing."
                )
            elif best_throughput < target_throughput * 0.8:
                suggestions.append(
                    f"Throughput below target ({best_throughput:.1f} vs {target_throughput:.1f}). "
                    f"Consider: increasing batch sizes, optimizing GPU utilization, or enabling model compilation."
                )
            
            # GPU utilization analysis
            best_gpu_util = best_stats['avg_gpu_utilization']
            if best_gpu_util < 70:
                suggestions.append(
                    f"Low GPU utilization ({best_gpu_util:.1f}%). "
                    f"Consider: increasing batch sizes, reducing CPU allocation, or optimizing data feeding."
                )
            elif best_gpu_util > 95:
                suggestions.append(
                    f"Very high GPU utilization ({best_gpu_util:.1f}%). "
                    f"Monitor for thermal throttling and consider memory optimization."
                )
        
        # Hardware-specific suggestions
        if self.hardware_capabilities:
            hw = self.hardware_capabilities
            
            if hw.memory_limited:
                suggestions.append(
                    "System memory limited. Consider: reducing batch sizes, "
                    "enabling memory optimization, or upgrading system RAM."
                )
            
            if hw.gpu_memory_limited:
                suggestions.append(
                    "GPU memory limited. Consider: enabling half-precision processing, "
                    "reducing GPU batch sizes, or using gradient checkpointing."
                )
            
            if hw.cpu_limited:
                suggestions.append(
                    "CPU limited system. Consider: reducing CPU allocation ratio, "
                    "optimizing GPU processing, or upgrading CPU."
                )
        
        # Configuration comparison suggestions
        if len(suite.performance_summary) > 1:
            throughputs = [(name, stats['avg_throughput']) 
                          for name, stats in suite.performance_summary.items()]
            throughputs.sort(key=lambda x: x[1], reverse=True)
            
            best_config, best_perf = throughputs[0]
            worst_config, worst_perf = throughputs[-1]
            
            if best_perf > worst_perf * 1.5:
                suggestions.append(
                    f"Significant performance variation detected. "
                    f"'{best_config}' performs {best_perf/worst_perf:.1f}x better than '{worst_config}'. "
                    f"Focus on optimizing high-performing configurations."
                )
        
        return suggestions
    
    def compare_configurations(self, 
                             suites: List[BenchmarkSuite],
                             comparison_name: str = "Configuration Comparison") -> Dict[str, Any]:
        """Compare performance across multiple benchmark suites.
        
        Args:
            suites: List of benchmark suites to compare
            comparison_name: Name for the comparison analysis
            
        Returns:
            Dictionary with detailed comparison analysis
            
        Requirements: 5.4, 5.5
        """
        self.logger.info(f"📊 Comparing {len(suites)} benchmark suites...")
        
        comparison = {
            'comparison_name': comparison_name,
            'suite_count': len(suites),
            'suite_summaries': [],
            'best_overall_suite': None,
            'performance_rankings': [],
            'optimization_insights': []
        }
        
        # Analyze each suite
        for i, suite in enumerate(suites):
            if not suite.performance_summary:
                continue
            
            # Get best configuration from each suite
            best_config = suite.best_configuration
            if best_config and best_config in suite.performance_summary:
                best_stats = suite.performance_summary[best_config]
                
                suite_summary = {
                    'suite_name': suite.suite_name,
                    'best_configuration': best_config,
                    'best_throughput': best_stats['avg_throughput'],
                    'best_gpu_utilization': best_stats['avg_gpu_utilization'],
                    'success_rate': best_stats['success_rate']
                }
                
                comparison['suite_summaries'].append(suite_summary)
        
        # Rank suites by performance
        if comparison['suite_summaries']:
            comparison['suite_summaries'].sort(
                key=lambda x: x['best_throughput'], reverse=True
            )
            
            comparison['best_overall_suite'] = comparison['suite_summaries'][0]['suite_name']
            comparison['performance_rankings'] = [
                (summary['suite_name'], summary['best_throughput'])
                for summary in comparison['suite_summaries']
            ]
        
        # Generate optimization insights
        if len(comparison['suite_summaries']) > 1:
            best = comparison['suite_summaries'][0]
            worst = comparison['suite_summaries'][-1]
            
            performance_gap = best['best_throughput'] - worst['best_throughput']
            if performance_gap > 500:  # Significant gap
                comparison['optimization_insights'].append(
                    f"Significant performance gap detected: {performance_gap:.1f} addr/sec "
                    f"between best ({best['suite_name']}) and worst ({worst['suite_name']}) configurations."
                )
        
        self.logger.info(f"✅ Configuration comparison completed")
        if comparison['best_overall_suite']:
            best_perf = comparison['suite_summaries'][0]['best_throughput']
            self.logger.info(f"🏆 Best overall: {comparison['best_overall_suite']} "
                           f"({best_perf:.1f} addr/sec)")
        
        return comparison
    
    # Private helper methods
    
    def _generate_test_addresses(self) -> List[str]:
        """Generate test addresses for benchmarking."""
        test_addresses = [
            "123 Main Street, Anytown, CA 90210",
            "456 Oak Avenue, Springfield, IL 62701",
            "789 Pine Road, Austin, TX 73301",
            "321 Elm Street, Portland, OR 97201",
            "654 Maple Drive, Denver, CO 80201",
            "987 Cedar Lane, Miami, FL 33101",
            "147 Birch Court, Seattle, WA 98101",
            "258 Willow Way, Boston, MA 02101",
            "369 Aspen Circle, Phoenix, AZ 85001",
            "741 Spruce Street, Atlanta, GA 30301"
        ]
        
        # Extend list to ensure we have enough addresses
        extended_addresses = []
        for i in range(500):  # Generate 500 test addresses
            base_address = test_addresses[i % len(test_addresses)]
            # Vary the address slightly
            street_num = 100 + (i * 7) % 9000
            extended_addresses.append(base_address.replace("123", str(street_num), 1))
        
        return extended_addresses
    
    def _generate_test_configurations(self) -> List[ProcessingConfiguration]:
        """Generate test configurations for benchmarking."""
        configurations = []
        
        # Base configuration
        base_config = ProcessingConfiguration()
        configurations.append(base_config)
        
        # High-performance configuration
        if self.hardware_capabilities and self.hardware_capabilities.gpu_count > 0:
            high_perf_config = ProcessingConfiguration(
                gpu_batch_size=600,
                gpu_queue_size=15,
                num_gpu_streams=4,
                target_throughput=2500,
                enable_model_compilation=True,
                use_half_precision=True
            )
            configurations.append(high_perf_config)
        
        # Memory-optimized configuration
        memory_opt_config = ProcessingConfiguration(
            gpu_batch_size=200,
            gpu_memory_fraction=0.85,
            gpu_queue_size=8,
            cpu_allocation_ratio=0.03,
            use_half_precision=True
        )
        configurations.append(memory_opt_config)
        
        # CPU-focused configuration
        cpu_focused_config = ProcessingConfiguration(
            gpu_batch_size=100,
            cpu_allocation_ratio=0.1,
            cpu_worker_count=4,
            target_throughput=800
        )
        configurations.append(cpu_focused_config)
        
        return configurations
    
    def _get_configuration_name(self, config: ProcessingConfiguration) -> str:
        """Generate descriptive name for configuration."""
        name_parts = []
        
        if config.gpu_batch_size >= 500:
            name_parts.append("HighBatch")
        elif config.gpu_batch_size <= 200:
            name_parts.append("LowBatch")
        else:
            name_parts.append("MedBatch")
        
        if config.cpu_allocation_ratio >= 0.05:
            name_parts.append("HighCPU")
        elif config.cpu_allocation_ratio <= 0.02:
            name_parts.append("LowCPU")
        
        if config.enable_model_compilation:
            name_parts.append("Compiled")
        
        if config.use_half_precision:
            name_parts.append("FP16")
        
        return "_".join(name_parts) if name_parts else "Default"
    
    def _run_single_benchmark(self, 
                            config: ProcessingConfiguration,
                            test_addresses: List[str],
                            config_name: str,
                            address_count: int,
                            iteration: int,
                            warmup: bool = False) -> BenchmarkResult:
        """Run a single benchmark test."""
        result = BenchmarkResult(
            test_name=f"benchmark_{config_name}",
            configuration_name=config_name,
            address_count=address_count,
            iteration=iteration
        )
        
        try:
            # Create hybrid processor
            processor = GPUCPUHybridProcessor(config)
            
            # Initialize processor
            processor.initialize_hybrid_processing()
            
            # Warmup run if requested
            if warmup and iteration == 0:
                warmup_addresses = test_addresses[:min(50, len(test_addresses))]
                processor.process_addresses_hybrid(warmup_addresses)
                time.sleep(1)  # Brief pause after warmup
            
            # Record start time and system state
            start_time = time.time()
            start_memory = psutil.virtual_memory().used / (1024**3)
            
            # Run processing
            processing_result = processor.process_addresses_hybrid(test_addresses)
            
            # Record end time and calculate metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update result with performance data
            result.processing_time = processing_time
            result.throughput = len(test_addresses) / processing_time if processing_time > 0 else 0
            result.addresses_processed = len(processing_result.parsed_addresses)
            result.success_rate = (result.addresses_processed / len(test_addresses)) * 100 if test_addresses else 100
            result.error_count = processing_result.error_count
            
            # Get performance metrics
            if processing_result.performance_metrics:
                result.gpu_utilization_avg = processing_result.performance_metrics.gpu_utilization
                result.gpu_utilization_peak = processing_result.performance_metrics.gpu_utilization  # Simplified
                result.cpu_utilization_avg = processing_result.performance_metrics.cpu_utilization
            
            # Memory usage
            end_memory = psutil.virtual_memory().used / (1024**3)
            result.memory_usage_peak_gb = max(start_memory, end_memory)
            
            # Hardware usage
            result.gpu_devices_used = 1 if config.gpu_batch_size > 0 else 0
            result.cpu_cores_used = config.cpu_worker_count
            
            # Optimization suggestions
            result.optimization_suggestions = processing_result.optimization_suggestions.copy()
            
            result.test_successful = True
            
            # Cleanup
            processor.shutdown()
            
        except Exception as e:
            result.test_successful = False
            result.error_message = str(e)
            self.logger.error(f"Benchmark test failed: {e}")
        
        return result
    
    def _analyze_performance_regression(self, 
                                      baseline: BenchmarkSuite,
                                      current: BenchmarkSuite,
                                      threshold: float) -> Dict[str, Any]:
        """Analyze performance regression between baseline and current results."""
        analysis = {
            'regression_detected': False,
            'overall_change': 0.0,
            'configuration_changes': {},
            'threshold': threshold,
            'summary': ""
        }
        
        if not baseline.performance_summary or not current.performance_summary:
            analysis['summary'] = "Insufficient data for regression analysis"
            return analysis
        
        # Compare common configurations
        common_configs = set(baseline.performance_summary.keys()) & set(current.performance_summary.keys())
        
        if not common_configs:
            analysis['summary'] = "No common configurations found for comparison"
            return analysis
        
        total_change = 0.0
        regression_count = 0
        
        for config_name in common_configs:
            baseline_perf = baseline.performance_summary[config_name]['avg_throughput']
            current_perf = current.performance_summary[config_name]['avg_throughput']
            
            change = (current_perf - baseline_perf) / baseline_perf if baseline_perf > 0 else 0
            total_change += change
            
            analysis['configuration_changes'][config_name] = {
                'baseline_throughput': baseline_perf,
                'current_throughput': current_perf,
                'change_percent': change * 100,
                'regression': change < -threshold
            }
            
            if change < -threshold:
                regression_count += 1
        
        # Calculate overall metrics
        analysis['overall_change'] = (total_change / len(common_configs)) * 100 if common_configs else 0
        analysis['regression_detected'] = regression_count > 0
        
        # Generate summary
        if analysis['regression_detected']:
            analysis['summary'] = f"Performance regression detected in {regression_count}/{len(common_configs)} configurations"
        else:
            analysis['summary'] = f"No significant regression detected (change: {analysis['overall_change']:.1f}%)"
        
        return analysis
    
    def _save_benchmark_results(self, suite: BenchmarkSuite):
        """Save benchmark results to file."""
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(suite.execution_timestamp))
        filename = f"benchmark_{suite.suite_name}_{timestamp}.json"
        filepath = self.output_directory / filename
        
        # Convert suite to serializable format
        suite_data = {
            'suite_name': suite.suite_name,
            'execution_timestamp': suite.execution_timestamp,
            'hardware_capabilities': self.hardware_capabilities.__dict__ if self.hardware_capabilities else None,
            'benchmark_results': [result.__dict__ for result in suite.benchmark_results],
            'performance_summary': suite.performance_summary,
            'best_configuration': suite.best_configuration,
            'optimization_recommendations': suite.optimization_recommendations,
            'regression_analysis': suite.regression_analysis
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(suite_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Benchmark results saved to: {filepath}")
            
            # Also save human-readable report
            report_filename = f"benchmark_report_{suite.suite_name}_{timestamp}.txt"
            report_filepath = self.output_directory / report_filename
            
            with open(report_filepath, 'w') as f:
                f.write(suite.generate_report())
            
            self.logger.info(f"📄 Benchmark report saved to: {report_filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save benchmark results: {e}")


def create_benchmark_configuration(
    test_name: str = "default_benchmark",
    address_counts: Optional[List[int]] = None,
    iterations: int = 3,
    configurations: Optional[List[ProcessingConfiguration]] = None
) -> BenchmarkConfiguration:
    """Create a benchmark configuration with sensible defaults.
    
    Args:
        test_name: Name for the benchmark test
        address_counts: List of address counts to test
        iterations: Number of iterations per test
        configurations: List of processing configurations to test
        
    Returns:
        BenchmarkConfiguration ready for use
    """
    if address_counts is None:
        address_counts = [100, 500, 1000, 2000]
    
    config = BenchmarkConfiguration(
        test_name=test_name,
        address_counts=address_counts,
        iterations_per_test=iterations,
        configurations=configurations or []
    )
    
    return config


def run_quick_benchmark(test_addresses: Optional[List[str]] = None) -> BenchmarkSuite:
    """Run a quick performance benchmark with default settings.
    
    Args:
        test_addresses: Optional test addresses (generated if not provided)
        
    Returns:
        BenchmarkSuite with results
    """
    benchmarker = PerformanceBenchmarker()
    
    config = create_benchmark_configuration(
        test_name="quick_benchmark",
        address_counts=[100, 500],
        iterations=2
    )
    
    return benchmarker.run_comprehensive_benchmark(config, test_addresses)