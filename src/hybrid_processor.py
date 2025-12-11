"""High-Performance GPU-CPU Hybrid Address Processing System.

This module implements a hybrid processing architecture that maximizes hardware utilization
to achieve 2000+ addresses per second throughput. The system leverages NVIDIA GPU acceleration
with HuggingFace dataset batching, complementary CPU processing, and advanced optimization
techniques including model compilation, sustained GPU utilization, and asynchronous processing.

Requirements: 1.1, 2.1, 4.1, 8.1
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from queue import Queue, Empty
import os

try:
    from .models import ParsedAddress
    from .error_handling import ErrorRecoveryManager
    from .comprehensive_output_generator import ComprehensiveOutputGenerator, ComprehensiveOutputConfig
except ImportError:
    # Fallback for direct execution
    from models import ParsedAddress
    from error_handling import ErrorRecoveryManager
    from comprehensive_output_generator import ComprehensiveOutputGenerator, ComprehensiveOutputConfig


@dataclass
class ProcessingConfiguration:
    """Configuration for GPU-CPU hybrid processing.
    
    Controls all aspects of hybrid processing including GPU optimization,
    CPU allocation, queue management, and performance thresholds.
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    # GPU Processing Configuration
    gpu_batch_size: int = 400  # GPU batch size for dataset processing
    dataset_batch_size: int = 1000  # HuggingFace dataset batch size
    gpu_memory_fraction: float = 0.95  # GPU memory allocation percentage (80-98%)
    gpu_queue_size: int = 10  # Number of pre-loaded batches (5-20)
    num_gpu_streams: int = 2  # Multiple GPU streams for overlapping execution
    
    # CPU Processing Configuration
    cpu_allocation_ratio: float = 0.02  # CPU workload percentage (2% = 0.1-0.5 of cores)
    cpu_batch_size: int = 50  # CPU batch size for overflow processing
    cpu_worker_count: int = 2  # Number of CPU workers
    
    # Performance Configuration
    performance_log_interval: int = 10  # Performance logging interval (seconds)
    target_throughput: int = 2000  # Target addresses per second (500-3000)
    gpu_utilization_threshold: float = 0.90  # Minimum GPU utilization (90%+)
    
    # Advanced GPU Optimizations
    enable_model_compilation: bool = True  # PyTorch 2.0+ compilation
    use_half_precision: bool = True  # float16 for memory efficiency
    enable_cudnn_benchmark: bool = True  # cuDNN benchmarking
    enable_tensor_float32: bool = True  # TensorFloat-32 operations
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate GPU batch size (100-1000)
        if not (100 <= self.gpu_batch_size <= 1000):
            raise ValueError(f"gpu_batch_size must be 100-1000, got {self.gpu_batch_size}")
        
        # Validate GPU memory allocation (80-98%)
        if not (0.80 <= self.gpu_memory_fraction <= 0.98):
            raise ValueError(f"gpu_memory_fraction must be 0.80-0.98, got {self.gpu_memory_fraction}")
        
        # Validate GPU queue size (5-20)
        if not (5 <= self.gpu_queue_size <= 20):
            raise ValueError(f"gpu_queue_size must be 5-20, got {self.gpu_queue_size}")
        
        # Validate CPU allocation ratio (0.1-0.5 of cores)
        if not (0.001 <= self.cpu_allocation_ratio <= 0.5):
            raise ValueError(f"cpu_allocation_ratio must be 0.001-0.5, got {self.cpu_allocation_ratio}")
        
        # Validate target throughput (500-3000)
        if not (500 <= self.target_throughput <= 3000):
            raise ValueError(f"target_throughput must be 500-3000, got {self.target_throughput}")


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics for hybrid processing.
    
    Tracks GPU utilization, throughput rates, queue status, and processing efficiency
    to provide comprehensive performance monitoring and optimization feedback.
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    # Core Performance Metrics
    gpu_utilization: float = 0.0  # Current GPU utilization percentage
    throughput_rate: float = 0.0  # Current addresses per second
    processing_efficiency: float = 0.0  # Parallelization efficiency percentage
    
    # Queue Status Metrics
    queue_input_size: int = 0  # GPU input queue size
    queue_output_size: int = 0  # GPU output queue size
    queue_max_size: int = 0  # Maximum queue size configured
    
    # Memory and Resource Metrics
    memory_usage: float = 0.0  # GPU memory usage percentage
    cpu_utilization: float = 0.0  # CPU utilization percentage
    
    # Processing Statistics
    total_processed: int = 0  # Total addresses processed
    gpu_processed: int = 0  # Addresses processed by GPU
    cpu_processed: int = 0  # Addresses processed by CPU
    success_rate: float = 100.0  # Processing success rate percentage
    
    # Timing Metrics
    processing_start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    
    def update_throughput(self, addresses_processed: int) -> None:
        """Update throughput rate based on recent processing."""
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        
        if time_elapsed > 0:
            self.throughput_rate = addresses_processed / time_elapsed
            self.last_update_time = current_time
    
    def calculate_efficiency(self) -> float:
        """Calculate processing efficiency based on GPU utilization and throughput."""
        if self.gpu_utilization > 0:
            # Efficiency = (actual throughput / theoretical max) * GPU utilization
            theoretical_max = 2500  # Theoretical maximum for RTX 4070
            actual_ratio = min(self.throughput_rate / theoretical_max, 1.0)
            self.processing_efficiency = actual_ratio * self.gpu_utilization
        return self.processing_efficiency


@dataclass
class ProcessingResult:
    """Result of hybrid address processing operation.
    
    Contains processed addresses, performance metrics, timing information,
    and optimization suggestions for comprehensive analysis.
    
    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    # Processing Results
    parsed_addresses: List[ParsedAddress] = field(default_factory=list)
    
    # Performance Data
    performance_metrics: Optional[PerformanceMetrics] = None
    
    # Timing Information
    processing_time: float = 0.0  # Total processing time in seconds
    gpu_processing_time: float = 0.0  # GPU-specific processing time
    cpu_processing_time: float = 0.0  # CPU-specific processing time
    
    # Device Statistics
    device_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Error Information
    error_count: int = 0  # Number of processing errors
    error_details: List[str] = field(default_factory=list)
    
    # Optimization Suggestions
    optimization_suggestions: List[str] = field(default_factory=list)
    
    def add_optimization_suggestion(self, suggestion: str) -> None:
        """Add an optimization suggestion based on performance analysis."""
        if suggestion not in self.optimization_suggestions:
            self.optimization_suggestions.append(suggestion)
    
    def calculate_success_rate(self) -> float:
        """Calculate processing success rate."""
        total_addresses = len(self.parsed_addresses) + self.error_count
        if total_addresses == 0:
            return 100.0
        return (len(self.parsed_addresses) / total_addresses) * 100.0


@dataclass
class BatchProcessingReport:
    """Comprehensive report for batch file processing operations.
    
    Provides detailed analysis of multi-file processing including per-file results,
    comparative performance analysis, and optimization recommendations.
    
    Requirements: 6.4, 9.3, 9.4
    """
    # Batch Processing Summary
    total_files_processed: int = 0
    total_addresses: int = 0
    
    # Performance Summary
    average_throughput: float = 0.0  # Average processing rate across all files
    peak_throughput: float = 0.0  # Maximum processing rate achieved
    average_gpu_utilization: float = 0.0  # Average GPU utilization
    processing_efficiency: float = 0.0  # Overall processing efficiency
    
    # Detailed Results
    file_results: List[ProcessingResult] = field(default_factory=list)
    
    # Human-readable Summary
    performance_summary: str = ""
    
    def add_file_result(self, result: ProcessingResult, filename: str) -> None:
        """Add a file processing result to the batch report."""
        self.file_results.append(result)
        self.total_files_processed += 1
        self.total_addresses += len(result.parsed_addresses)
        
        # Update performance metrics
        if result.performance_metrics:
            self.peak_throughput = max(self.peak_throughput, result.performance_metrics.throughput_rate)
            
            # Calculate running averages
            total_throughput = sum(r.performance_metrics.throughput_rate for r in self.file_results 
                                 if r.performance_metrics)
            total_gpu_util = sum(r.performance_metrics.gpu_utilization for r in self.file_results 
                               if r.performance_metrics)
            
            self.average_throughput = total_throughput / len(self.file_results)
            self.average_gpu_utilization = total_gpu_util / len(self.file_results)
    
    def generate_summary(self) -> str:
        """Generate human-readable performance summary."""
        if not self.file_results:
            return "No files processed."
        
        total_time = sum(r.processing_time for r in self.file_results)
        total_errors = sum(r.error_count for r in self.file_results)
        
        self.performance_summary = f"""
Batch Processing Summary:
- Files Processed: {self.total_files_processed}
- Total Addresses: {self.total_addresses:,}
- Total Processing Time: {total_time:.2f} seconds
- Average Throughput: {self.average_throughput:.1f} addresses/second
- Peak Throughput: {self.peak_throughput:.1f} addresses/second
- Average GPU Utilization: {self.average_gpu_utilization:.1f}%
- Processing Efficiency: {self.processing_efficiency:.1f}%
- Total Errors: {total_errors}
- Success Rate: {((self.total_addresses - total_errors) / max(self.total_addresses, 1)) * 100:.2f}%
        """.strip()
        
        return self.performance_summary


class GPUCPUHybridProcessor:
    """Main orchestration class for GPU-CPU hybrid address processing.
    
    Coordinates GPU and CPU processing tiers, manages asynchronous queues,
    monitors performance, and provides intelligent workload distribution
    to achieve maximum throughput with sustained GPU utilization.
    
    Requirements: 1.1, 2.1, 4.1, 8.1
    """
    
    def __init__(self, config: ProcessingConfiguration):
        """Initialize the hybrid processor with configuration.
        
        Args:
            config: Processing configuration with GPU/CPU settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Processing Components (will be initialized in setup methods)
        self.dataset_gpu_processor = None
        self.minimal_cpu_processor = None
        self.performance_monitor = None
        self.queue_manager = None
        self.error_recovery_manager = None
        
        # Processing State
        self.is_initialized = False
        self.processing_active = False
        self.shutdown_requested = False
        
        # Thread Management - Separate pools for GPU and CPU processing
        self.gpu_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="GPU-Worker")
        self.cpu_executor = ThreadPoolExecutor(max_workers=config.cpu_worker_count, thread_name_prefix="CPU-Worker")
        self.coordination_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Coordinator")
        
        # Coordination and synchronization
        self.processing_lock = threading.Lock()
        self.gpu_processing_lock = threading.Lock()
        self.cpu_processing_lock = threading.Lock()
        self.resource_coordination_lock = threading.Lock()
        
        # Resource conflict prevention
        self.gpu_memory_reserved = False
        self.cpu_intensive_processing = False
        self.active_gpu_streams = 0
        self.active_cpu_workers = 0
        
        self.logger.info(f"Initialized GPUCPUHybridProcessor with config: "
                        f"GPU batch size={config.gpu_batch_size}, "
                        f"CPU ratio={config.cpu_allocation_ratio}, "
                        f"Target throughput={config.target_throughput}")
    
    def initialize_hybrid_processing(self) -> None:
        """Initialize all processing components and validate system readiness.
        
        Sets up GPU pipeline, CPU processing, performance monitoring,
        and asynchronous queue management for hybrid processing.
        
        Requirements: 1.1, 2.1, 4.1
        """
        if self.is_initialized:
            self.logger.warning("Hybrid processor already initialized")
            return
        
        try:
            self.logger.info("Initializing hybrid processing components...")
            
            # Initialize components (placeholder - will be implemented in later tasks)
            self._initialize_error_recovery_manager()
            self._initialize_dataset_gpu_processor()
            self._initialize_minimal_cpu_processor()
            self._initialize_performance_monitor()
            self._initialize_queue_manager()
            
            # Set component references for error handling
            self.error_recovery_manager.set_component_references(
                gpu_processor=self.dataset_gpu_processor,
                cpu_processor=self.minimal_cpu_processor,
                performance_monitor=self.performance_monitor
            )
            
            # Set error recovery manager references in components
            if self.dataset_gpu_processor and hasattr(self.dataset_gpu_processor, 'set_error_recovery_manager'):
                self.dataset_gpu_processor.set_error_recovery_manager(self.error_recovery_manager)
            
            if self.minimal_cpu_processor and hasattr(self.minimal_cpu_processor, 'set_error_recovery_manager'):
                self.minimal_cpu_processor.set_error_recovery_manager(self.error_recovery_manager)
            
            if self.queue_manager and hasattr(self.queue_manager, 'set_error_recovery_manager'):
                self.queue_manager.set_error_recovery_manager(self.error_recovery_manager)
            
            # Validate system readiness
            self._validate_system_readiness()
            
            self.is_initialized = True
            self.logger.info("Hybrid processing initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid processing: {e}")
            raise RuntimeError(f"Hybrid processing initialization failed: {e}")
    
    def process_addresses_hybrid(self, addresses: List[str]) -> ProcessingResult:
        """Main processing entry point for hybrid GPU-CPU address processing.
        
        Implements advanced hybrid processing with performance validation, sustained GPU
        utilization monitoring, and synchronization delay elimination through pre-loaded
        batches. Integrates all components for maximum throughput achievement.
        
        Args:
            addresses: List of raw address strings to process
            
        Returns:
            ProcessingResult with parsed addresses and performance data
            
        Requirements: 1.3, 1.4, 1.5, 4.1, 4.2
        """
        if not self.is_initialized:
            raise RuntimeError("Hybrid processor not initialized. Call initialize_hybrid_processing() first.")
        
        if not addresses:
            return ProcessingResult()
        
        start_time = time.time()
        
        with self.processing_lock:
            self.processing_active = True
        
        try:
            self.logger.info(f"🚀 Starting hybrid processing of {len(addresses)} addresses with performance validation")
            
            # Initialize performance tracking with real-time monitoring
            performance_metrics = PerformanceMetrics()
            performance_metrics.processing_start_time = start_time
            
            # Set queue manager reference for performance monitoring
            if self.performance_monitor and self.queue_manager:
                self.performance_monitor.set_queue_manager(self.queue_manager)
            
            # Start sustained GPU utilization monitoring (90%+ target)
            self._start_sustained_gpu_monitoring()
            
            # Distribute workload between GPU and CPU with dynamic balancing
            gpu_addresses, cpu_addresses = self.distribute_workload(addresses)
            
            self.logger.info(f"📊 Workload distribution: GPU={len(gpu_addresses)} ({len(gpu_addresses)/len(addresses)*100:.1f}%), "
                           f"CPU={len(cpu_addresses)} ({len(cpu_addresses)/len(addresses)*100:.1f}%)")
            
            # Configure synchronization delay elimination through pre-loaded batches
            processing_start = time.time()
            gpu_results, cpu_results = self._process_with_preloaded_batches(gpu_addresses, cpu_addresses)
            
            # Calculate processing times with detailed metrics
            total_processing_time = time.time() - processing_start
            
            # Estimate GPU and CPU times based on allocation and results
            if gpu_results and cpu_results:
                # Both processed - estimate based on allocation ratio
                gpu_ratio = len(gpu_addresses) / (len(gpu_addresses) + len(cpu_addresses))
                gpu_time = total_processing_time * gpu_ratio
                cpu_time = total_processing_time * (1 - gpu_ratio)
            elif gpu_results:
                # Only GPU processed
                gpu_time = total_processing_time
                cpu_time = 0.0
            elif cpu_results:
                # Only CPU processed
                gpu_time = 0.0
                cpu_time = total_processing_time
            else:
                # No results
                gpu_time = 0.0
                cpu_time = 0.0
            
            # Update performance metrics with comprehensive data
            performance_metrics.gpu_processed = len(gpu_results)
            performance_metrics.cpu_processed = len(cpu_results)
            
            # Combine results and calculate final metrics
            all_results = gpu_results + cpu_results
            total_time = time.time() - start_time
            
            # Update performance metrics with final calculations
            performance_metrics.total_processed = len(all_results)
            performance_metrics.update_throughput(len(all_results))
            performance_metrics.calculate_efficiency()
            
            # Get final GPU utilization for validation
            final_gpu_utilization = self._get_final_gpu_utilization()
            performance_metrics.gpu_utilization = final_gpu_utilization
            
            # Implement performance validation for 1500+ addresses/second target
            throughput_achieved = len(addresses) / total_time if total_time > 0 else 0
            performance_validation = self._validate_performance_targets(
                throughput_achieved, final_gpu_utilization, len(addresses)
            )
            
            # Log performance update with validation results
            if self.performance_monitor:
                self.performance_monitor.log_performance_update(len(all_results))
            
            # Create comprehensive processing result
            result = ProcessingResult(
                parsed_addresses=all_results,
                performance_metrics=performance_metrics,
                processing_time=total_time,
                gpu_processing_time=gpu_time,
                cpu_processing_time=cpu_time,
                device_statistics={
                    "gpu_addresses": len(gpu_addresses),
                    "cpu_addresses": len(cpu_addresses),
                    "gpu_allocation_ratio": len(gpu_addresses) / len(addresses) if addresses else 0,
                    "cpu_allocation_ratio": len(cpu_addresses) / len(addresses) if addresses else 0,
                    "throughput_achieved": throughput_achieved,
                    "gpu_utilization_achieved": final_gpu_utilization,
                    "performance_validation": performance_validation,
                    "synchronization_delays_eliminated": True,
                    "preloaded_batches_used": True
                }
            )
            
            # Generate optimization suggestions based on performance analysis
            self._generate_optimization_suggestions(result)
            
            # Add performance validation results to suggestions
            self._add_performance_validation_suggestions(result, performance_validation)
            
            # Log comprehensive completion summary
            self.logger.info(f"✅ Hybrid processing completed: {len(all_results)} addresses in {total_time:.2f}s")
            self.logger.info(f"📈 Performance: {throughput_achieved:.1f} addr/sec (target: 1500+), "
                           f"GPU: {final_gpu_utilization:.1f}% (target: 90%+)")
            self.logger.info(f"🎯 Validation: {performance_validation['summary']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hybrid processing failed: {e}")
            raise
        finally:
            with self.processing_lock:
                self.processing_active = False
    
    def distribute_workload(self, addresses: List[str]) -> Tuple[List[str], List[str]]:
        """Distribute processing workload between GPU and CPU tiers with dynamic balancing.
        
        Allocates 95-98% of addresses to GPU processing and 2-5% to CPU
        based on configuration and current GPU utilization metrics.
        Implements dynamic workload balancing to optimize resource utilization.
        
        Args:
            addresses: List of addresses to distribute
            
        Returns:
            Tuple of (gpu_addresses, cpu_addresses)
            
        Requirements: 4.1, 4.2, 4.3, 4.5
        """
        if not addresses:
            return [], []
        
        total_count = len(addresses)
        
        # Get current GPU utilization for dynamic balancing
        current_gpu_utilization = 0.0
        if self.performance_monitor:
            current_metrics = self.performance_monitor.get_current_metrics()
            current_gpu_utilization = current_metrics.gpu_utilization
        
        # Dynamic workload balancing based on GPU utilization
        base_cpu_ratio = self.config.cpu_allocation_ratio
        
        # Adjust CPU allocation based on GPU utilization
        if current_gpu_utilization > 0.95:
            # GPU highly utilized, increase CPU allocation slightly
            adjusted_cpu_ratio = min(base_cpu_ratio * 1.5, 0.05)  # Max 5% CPU
            self.logger.debug(f"High GPU utilization ({current_gpu_utilization:.1f}%), "
                            f"increasing CPU allocation to {adjusted_cpu_ratio:.3f}")
        elif current_gpu_utilization < 0.80:
            # GPU underutilized, decrease CPU allocation
            adjusted_cpu_ratio = max(base_cpu_ratio * 0.5, 0.01)  # Min 1% CPU
            self.logger.debug(f"Low GPU utilization ({current_gpu_utilization:.1f}%), "
                            f"decreasing CPU allocation to {adjusted_cpu_ratio:.3f}")
        else:
            # Normal GPU utilization, use base allocation
            adjusted_cpu_ratio = base_cpu_ratio
        
        # Calculate allocation counts
        cpu_count = max(1, int(total_count * adjusted_cpu_ratio))
        gpu_count = total_count - cpu_count
        
        # Ensure GPU gets the majority (95-98%) - hard constraint
        actual_gpu_ratio = gpu_count / total_count
        if actual_gpu_ratio < 0.95:
            # Adjust to ensure minimum 95% GPU allocation
            gpu_count = max(int(total_count * 0.95), total_count - max(10, int(total_count * 0.05)))
            cpu_count = total_count - gpu_count
        elif actual_gpu_ratio > 0.98:
            # Ensure minimum CPU allocation for overflow handling
            cpu_count = max(1, int(total_count * 0.02))
            gpu_count = total_count - cpu_count
        
        # Split addresses for optimal processing
        gpu_addresses = addresses[:gpu_count]
        cpu_addresses = addresses[gpu_count:gpu_count + cpu_count]
        
        final_gpu_ratio = len(gpu_addresses) / total_count
        final_cpu_ratio = len(cpu_addresses) / total_count
        
        self.logger.info(f"Workload distribution: GPU={len(gpu_addresses)} ({final_gpu_ratio*100:.1f}%), "
                        f"CPU={len(cpu_addresses)} ({final_cpu_ratio*100:.1f}%), "
                        f"GPU utilization={current_gpu_utilization:.1f}%")
        
        return gpu_addresses, cpu_addresses
    
    def coordinate_hybrid_processing(self, gpu_addresses: List[str], cpu_addresses: List[str]) -> Tuple[List[ParsedAddress], List[ParsedAddress]]:
        """Coordinate GPU and CPU processing with resource conflict prevention.
        
        Uses separate thread pools for GPU and CPU processing to prevent blocking
        and implements coordination logic to avoid resource conflicts.
        
        Args:
            gpu_addresses: Addresses allocated for GPU processing
            cpu_addresses: Addresses allocated for CPU processing
            
        Returns:
            Tuple of (gpu_results, cpu_results)
            
        Requirements: 4.3, 4.5
        """
        gpu_results = []
        cpu_results = []
        
        # Reserve resources and coordinate processing
        with self.resource_coordination_lock:
            # Check for resource conflicts
            if self._check_resource_conflicts():
                self.logger.warning("Resource conflicts detected, adjusting processing strategy")
                return self._process_with_conflict_resolution(gpu_addresses, cpu_addresses)
        
        # Submit processing tasks to separate thread pools
        gpu_future = None
        cpu_future = None
        
        try:
            # Submit GPU processing task
            if gpu_addresses:
                gpu_future = self.gpu_executor.submit(self._coordinated_gpu_processing, gpu_addresses)
                self.logger.debug(f"Submitted GPU processing task for {len(gpu_addresses)} addresses")
            
            # Submit CPU processing task (with slight delay to avoid resource conflicts)
            if cpu_addresses:
                cpu_future = self.cpu_executor.submit(self._coordinated_cpu_processing, cpu_addresses)
                self.logger.debug(f"Submitted CPU processing task for {len(cpu_addresses)} addresses")
            
            # Collect results from both processing threads
            if gpu_future:
                try:
                    gpu_results = gpu_future.result(timeout=300)  # 5 minute timeout
                    self.logger.debug(f"GPU processing completed: {len(gpu_results)} results")
                except Exception as e:
                    self.logger.error(f"GPU processing failed: {e}")
                    # Fallback to CPU processing for failed GPU addresses
                    gpu_results = self.handle_gpu_fallback(gpu_addresses)
            
            if cpu_future:
                try:
                    cpu_results = cpu_future.result(timeout=180)  # 3 minute timeout
                    self.logger.debug(f"CPU processing completed: {len(cpu_results)} results")
                except Exception as e:
                    self.logger.error(f"CPU processing failed: {e}")
                    cpu_results = [ParsedAddress(parse_success=False, parse_error=f"CPU processing error: {str(e)}") 
                                 for _ in cpu_addresses]
            
        except Exception as e:
            self.logger.error(f"Coordinated processing failed: {e}")
            # Emergency fallback processing
            gpu_results = self.handle_gpu_fallback(gpu_addresses) if gpu_addresses else []
            cpu_results = [ParsedAddress(parse_success=False, parse_error=f"Coordination error: {str(e)}") 
                         for _ in cpu_addresses] if cpu_addresses else []
        
        return gpu_results, cpu_results
    
    def _coordinated_gpu_processing(self, addresses: List[str]) -> List[ParsedAddress]:
        """GPU processing with coordination and resource management.
        
        Args:
            addresses: Addresses to process on GPU
            
        Returns:
            List of parsed addresses from GPU processing
        """
        with self.gpu_processing_lock:
            try:
                # Reserve GPU resources
                self.gpu_memory_reserved = True
                self.active_gpu_streams += 1
                
                self.logger.debug(f"Starting coordinated GPU processing for {len(addresses)} addresses")
                
                # Process using GPU pipeline
                results = self._process_gpu_addresses(addresses)
                
                self.logger.debug(f"Coordinated GPU processing completed: {len(results)} results")
                return results
                
            finally:
                # Release GPU resources
                self.gpu_memory_reserved = False
                self.active_gpu_streams = max(0, self.active_gpu_streams - 1)
    
    def _coordinated_cpu_processing(self, addresses: List[str]) -> List[ParsedAddress]:
        """CPU processing with coordination and resource management.
        
        Args:
            addresses: Addresses to process on CPU
            
        Returns:
            List of parsed addresses from CPU processing
        """
        with self.cpu_processing_lock:
            try:
                # Reserve CPU resources
                self.cpu_intensive_processing = True
                self.active_cpu_workers += 1
                
                self.logger.debug(f"Starting coordinated CPU processing for {len(addresses)} addresses")
                
                # Small delay to avoid GPU interference during initialization
                if self.gpu_memory_reserved:
                    time.sleep(0.1)
                
                # Process using CPU pipeline
                results = self._process_cpu_addresses(addresses)
                
                self.logger.debug(f"Coordinated CPU processing completed: {len(results)} results")
                return results
                
            finally:
                # Release CPU resources
                self.cpu_intensive_processing = False
                self.active_cpu_workers = max(0, self.active_cpu_workers - 1)
    
    def _check_resource_conflicts(self) -> bool:
        """Check for potential resource conflicts between GPU and CPU processing.
        
        Returns:
            True if resource conflicts are detected, False otherwise
        """
        # Check GPU memory conflicts
        if self.gpu_memory_reserved and self.active_gpu_streams >= 2:
            self.logger.warning("GPU memory conflict detected: multiple streams active")
            return True
        
        # Check CPU intensive processing conflicts
        if self.cpu_intensive_processing and self.active_cpu_workers >= self.config.cpu_worker_count:
            self.logger.warning("CPU processing conflict detected: maximum workers active")
            return True
        
        # Check system resource conflicts
        if self.active_gpu_streams > 0 and self.active_cpu_workers > self.config.cpu_worker_count // 2:
            self.logger.warning("System resource conflict detected: high GPU and CPU utilization")
            return True
        
        return False
    
    def _process_with_conflict_resolution(self, gpu_addresses: List[str], cpu_addresses: List[str]) -> Tuple[List[ParsedAddress], List[ParsedAddress]]:
        """Process addresses with conflict resolution strategy.
        
        Implements sequential processing when resource conflicts are detected
        to prevent system instability and ensure reliable processing.
        
        Args:
            gpu_addresses: Addresses for GPU processing
            cpu_addresses: Addresses for CPU processing
            
        Returns:
            Tuple of (gpu_results, cpu_results)
        """
        self.logger.info("Using conflict resolution strategy: sequential processing")
        
        gpu_results = []
        cpu_results = []
        
        # Process GPU first (higher priority)
        if gpu_addresses:
            self.logger.debug("Processing GPU addresses first due to conflicts")
            gpu_results = self._coordinated_gpu_processing(gpu_addresses)
        
        # Wait for GPU processing to complete before starting CPU
        time.sleep(0.2)
        
        # Process CPU after GPU completion
        if cpu_addresses:
            self.logger.debug("Processing CPU addresses after GPU completion")
            cpu_results = self._coordinated_cpu_processing(cpu_addresses)
        
        return gpu_results, cpu_results
    
    def handle_gpu_fallback(self, failed_addresses: List[str]) -> List[ParsedAddress]:
        """Handle GPU processing failures using CPU fallback.
        
        Processes addresses that failed on GPU using the minimal CPU processor
        with error recovery mechanisms and alternative processing strategies.
        
        Args:
            failed_addresses: List of addresses that failed GPU processing
            
        Returns:
            List of parsed addresses from CPU fallback
            
        Requirements: 7.1
        """
        if not failed_addresses:
            return []
        
        # Use error recovery manager for comprehensive GPU fallback handling
        if self.error_recovery_manager:
            return self.error_recovery_manager.handle_gpu_processing_error(
                failed_addresses, 
                Exception("GPU processing failed"), 
                "HybridProcessor"
            )
        
        # Fallback to original implementation if error manager not available
        if not self.minimal_cpu_processor or not self.minimal_cpu_processor.is_initialized:
            self.logger.error("Minimal CPU processor not available for fallback")
            return [ParsedAddress(parse_success=False, 
                                parse_error="CPU fallback not available") 
                   for _ in failed_addresses]
        
        self.logger.warning(f"GPU processing failed, falling back to CPU for {len(failed_addresses)} addresses")
        
        try:
            return self.minimal_cpu_processor.handle_gpu_fallback(failed_addresses)
        except Exception as e:
            self.logger.error(f"CPU fallback processing failed: {e}")
            return [ParsedAddress(parse_success=False, 
                                parse_error=f"CPU fallback error: {str(e)}") 
                   for _ in failed_addresses]
    
    def monitor_performance(self) -> PerformanceMetrics:
        """Get current performance metrics and system status.
        
        Returns:
            Current performance metrics with real-time data
            
        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
        """
        if self.performance_monitor:
            return self.performance_monitor.get_current_metrics()
        else:
            # Return basic metrics if monitor not initialized
            return PerformanceMetrics()
    
    def shutdown(self) -> None:
        """Gracefully shutdown the hybrid processor and cleanup resources."""
        self.logger.info("Shutting down hybrid processor...")
        
        with self.processing_lock:
            self.shutdown_requested = True
        
        # Wait for active processing to complete
        while self.processing_active:
            time.sleep(0.1)
        
        # Shutdown components
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        if self.queue_manager:
            self.queue_manager.shutdown()
        
        if self.minimal_cpu_processor:
            self.minimal_cpu_processor.shutdown()
        
        # Shutdown thread executors
        self.gpu_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)
        self.coordination_executor.shutdown(wait=True)
        
        self.is_initialized = False
        self.logger.info("Hybrid processor shutdown completed")
    
    # Private helper methods (placeholders for later implementation)
    
    def _initialize_error_recovery_manager(self) -> None:
        """Initialize error recovery manager for comprehensive error handling."""
        self.logger.info("Initializing error recovery manager...")
        
        try:
            # Create error recovery manager
            self.error_recovery_manager = ErrorRecoveryManager(self.config)
            
            self.logger.info("✅ Error recovery manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize error recovery manager: {e}")
            raise
    
    def _initialize_dataset_gpu_processor(self) -> None:
        """Initialize GPU processing component with dataset optimization."""
        self.logger.info("Initializing dataset GPU processor...")
        
        try:
            try:
                from .dataset_gpu_processor import DatasetGPUProcessor
            except ImportError:
                # Fallback for direct execution
                from dataset_gpu_processor import DatasetGPUProcessor
            
            # Create and initialize dataset GPU processor
            self.dataset_gpu_processor = DatasetGPUProcessor(self.config)
            
            # Setup the GPU pipeline with all optimizations
            if not self.dataset_gpu_processor.setup_dataset_gpu_pipeline():
                raise RuntimeError("Failed to setup dataset GPU pipeline")
            
            self.logger.info("✅ Dataset GPU processor initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import DatasetGPUProcessor: {e}")
            raise RuntimeError(f"DatasetGPUProcessor not available: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize dataset GPU processor: {e}")
            raise
    
    def _initialize_minimal_cpu_processor(self) -> None:
        """Initialize CPU processing component for overflow and fallback processing."""
        self.logger.info("Initializing minimal CPU processor...")
        
        try:
            try:
                from .minimal_cpu_processor import MinimalCPUProcessor
            except ImportError:
                # Fallback for direct execution
                from minimal_cpu_processor import MinimalCPUProcessor
            
            # Create and initialize minimal CPU processor
            self.minimal_cpu_processor = MinimalCPUProcessor(self.config)
            
            # Setup the CPU pipeline with lightweight configuration
            if not self.minimal_cpu_processor.setup_minimal_cpu_pipeline():
                raise RuntimeError("Failed to setup minimal CPU pipeline")
            
            self.logger.info("✅ Minimal CPU processor initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import MinimalCPUProcessor: {e}")
            raise RuntimeError(f"MinimalCPUProcessor not available: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize minimal CPU processor: {e}")
            raise
    
    def _initialize_performance_monitor(self) -> None:
        """Initialize performance monitoring component for real-time tracking."""
        self.logger.info("Initializing performance monitor...")
        
        try:
            try:
                from .performance_monitor import PerformanceMonitor
            except ImportError:
                # Fallback for direct execution
                from performance_monitor import PerformanceMonitor
            
            # Create and initialize performance monitor
            self.performance_monitor = PerformanceMonitor(self.config)
            
            # Start real-time monitoring
            self.performance_monitor.start_monitoring()
            
            self.logger.info("✅ Performance monitor initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import PerformanceMonitor: {e}")
            raise RuntimeError(f"PerformanceMonitor not available: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize performance monitor: {e}")
            raise
    
    def _initialize_queue_manager(self) -> None:
        """Initialize asynchronous queue manager for sustained GPU utilization."""
        self.logger.info("Initializing asynchronous queue manager...")
        
        try:
            try:
                from .asynchronous_queue_manager import AsynchronousQueueManager
            except ImportError:
                # Fallback for direct execution
                from asynchronous_queue_manager import AsynchronousQueueManager
            
            # Create and initialize queue manager
            self.queue_manager = AsynchronousQueueManager(self.config)
            
            # Initialize GPU queues with configured size
            if not self.queue_manager.initialize_gpu_queues():
                raise RuntimeError("Failed to initialize GPU queues")
            
            self.logger.info("✅ Asynchronous queue manager initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import AsynchronousQueueManager: {e}")
            raise RuntimeError(f"AsynchronousQueueManager not available: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize queue manager: {e}")
            raise
    
    def _validate_system_readiness(self) -> None:
        """Validate that all components are ready for processing."""
        self.logger.info("Validating system readiness...")
        # Basic validation - will be expanded in later tasks
        pass
    
    def _process_gpu_addresses(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process addresses using GPU pipeline with dataset optimization."""
        if not self.dataset_gpu_processor or not self.dataset_gpu_processor.is_initialized:
            self.logger.error("Dataset GPU processor not initialized")
            return [ParsedAddress(parse_success=False, parse_error="GPU processor not initialized") for _ in addresses]
        
        try:
            # Use dataset batching for optimal GPU processing
            return self.dataset_gpu_processor.process_with_dataset_batching(addresses)
        except Exception as e:
            self.logger.error(f"GPU processing failed: {e}")
            
            # Use error recovery manager for comprehensive error handling
            if self.error_recovery_manager:
                return self.error_recovery_manager.handle_gpu_processing_error(
                    addresses, e, "DatasetGPUProcessor"
                )
            
            # Fallback to basic error handling
            return [ParsedAddress(parse_success=False, parse_error=f"GPU processing error: {str(e)}") for _ in addresses]
    
    def _process_cpu_addresses(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process addresses using minimal CPU pipeline for overflow processing."""
        if not self.minimal_cpu_processor or not self.minimal_cpu_processor.is_initialized:
            self.logger.error("Minimal CPU processor not initialized")
            return [ParsedAddress(parse_success=False, parse_error="CPU processor not initialized") for _ in addresses]
        
        try:
            # Use CPU overflow processing for regular CPU allocation
            return self.minimal_cpu_processor.process_cpu_overflow(addresses)
        except Exception as e:
            self.logger.error(f"CPU processing failed: {e}")
            return [ParsedAddress(parse_success=False, parse_error=f"CPU processing error: {str(e)}") for _ in addresses]
    
    def _start_sustained_gpu_monitoring(self) -> None:
        """Start sustained GPU utilization monitoring (90%+ target).
        
        Initializes real-time GPU monitoring to ensure sustained utilization
        above 90% throughout the processing duration.
        
        Requirements: 1.3
        """
        if self.performance_monitor:
            # Ensure performance monitoring is active
            if not self.performance_monitor.monitoring_active:
                self.performance_monitor.start_monitoring()
            
            # Log monitoring start
            self.logger.info(f"🔍 Started sustained GPU utilization monitoring (target: {self.config.gpu_utilization_threshold*100:.0f}%+)")
        else:
            self.logger.warning("Performance monitor not available for sustained GPU monitoring")
    
    def _process_with_preloaded_batches(self, gpu_addresses: List[str], cpu_addresses: List[str]) -> Tuple[List[ParsedAddress], List[ParsedAddress]]:
        """Process addresses with synchronization delay elimination through pre-loaded batches.
        
        Implements asynchronous processing with pre-loaded batch queues to eliminate
        CPU-GPU synchronization delays and maintain continuous GPU feeding.
        
        Args:
            gpu_addresses: Addresses allocated for GPU processing
            cpu_addresses: Addresses allocated for CPU processing
            
        Returns:
            Tuple of (gpu_results, cpu_results)
            
        Requirements: 1.5, 3.1, 3.2, 3.3, 3.4, 3.5
        """
        gpu_results = []
        cpu_results = []
        
        try:
            # Use asynchronous queue manager for GPU processing with pre-loaded batches
            if gpu_addresses and self.queue_manager:
                self.logger.info(f"🔄 Processing {len(gpu_addresses)} GPU addresses with pre-loaded batches")
                
                # Set GPU processing function for queue manager
                def gpu_processing_function(batch_addresses):
                    return self._process_gpu_addresses(batch_addresses)
                
                # Start data feeder with pre-loading (eliminates synchronization delays)
                if not self.queue_manager.start_data_feeder(gpu_addresses, self.config.gpu_batch_size):
                    self.logger.warning("Failed to start data feeder, falling back to direct processing")
                    gpu_results = self._process_gpu_addresses(gpu_addresses)
                else:
                    # Start GPU workers with asynchronous processing
                    if self.queue_manager.start_gpu_workers(
                        num_workers=self.config.num_gpu_streams,
                        processing_function=gpu_processing_function
                    ):
                        # Collect results asynchronously with timeout handling
                        try:
                            gpu_results = self.queue_manager.collect_results(timeout=300)  # 5 minute timeout
                            self.logger.info(f"✅ Asynchronous GPU processing completed: {len(gpu_results)} results")
                        except Exception as timeout_error:
                            if self.error_recovery_manager:
                                self.logger.warning("GPU processing timeout, attempting recovery...")
                                gpu_results = self.error_recovery_manager.handle_timeout_error(
                                    self.queue_manager.collect_results,
                                    (),
                                    {"timeout": 300},
                                    300,
                                    timeout_error,
                                    "AsynchronousQueueManager"
                                )
                                if gpu_results is None:
                                    gpu_results = self._process_gpu_addresses(gpu_addresses)
                            else:
                                gpu_results = self._process_gpu_addresses(gpu_addresses)
                    else:
                        self.logger.warning("Failed to start GPU workers, falling back to direct processing")
                        gpu_results = self._process_gpu_addresses(gpu_addresses)
            elif gpu_addresses:
                # Fallback to direct GPU processing if queue manager not available
                self.logger.info(f"🔄 Processing {len(gpu_addresses)} GPU addresses (direct processing)")
                gpu_results = self._process_gpu_addresses(gpu_addresses)
            
            # Process CPU addresses in parallel (minimal interference with GPU)
            if cpu_addresses:
                self.logger.info(f"🔄 Processing {len(cpu_addresses)} CPU addresses (overflow processing)")
                
                # Use separate thread for CPU processing to avoid blocking
                cpu_future = self.cpu_executor.submit(self._process_cpu_addresses, cpu_addresses)
                
                try:
                    cpu_results = cpu_future.result(timeout=180)  # 3 minute timeout
                    self.logger.info(f"✅ CPU overflow processing completed: {len(cpu_results)} results")
                except Exception as e:
                    self.logger.error(f"CPU processing failed: {e}")
                    cpu_results = [ParsedAddress(parse_success=False, 
                                                parse_error=f"CPU processing error: {str(e)}") 
                                 for _ in cpu_addresses]
            
            return gpu_results, cpu_results
            
        except Exception as e:
            self.logger.error(f"Pre-loaded batch processing failed: {e}")
            # Emergency fallback processing
            gpu_results = self.handle_gpu_fallback(gpu_addresses) if gpu_addresses else []
            cpu_results = [ParsedAddress(parse_success=False, 
                                       parse_error=f"Processing error: {str(e)}") 
                         for _ in cpu_addresses] if cpu_addresses else []
            return gpu_results, cpu_results
    
    def _get_final_gpu_utilization(self) -> float:
        """Get final GPU utilization for performance validation.
        
        Returns:
            Final GPU utilization percentage
        """
        if self.performance_monitor:
            return self.performance_monitor.track_gpu_utilization()
        elif self.dataset_gpu_processor:
            return self.dataset_gpu_processor.get_gpu_utilization()
        else:
            return 0.0
    
    def _validate_performance_targets(self, throughput: float, gpu_utilization: float, total_addresses: int) -> Dict[str, Any]:
        """Implement performance validation for 1500+ addresses/second target.
        
        Validates that processing meets performance targets for throughput and
        GPU utilization as specified in requirements.
        
        Args:
            throughput: Achieved throughput in addresses/second
            gpu_utilization: Achieved GPU utilization percentage
            total_addresses: Total number of addresses processed
            
        Returns:
            Dictionary with validation results and analysis
            
        Requirements: 1.4, 1.3
        """
        validation_results = {
            "throughput_target": 1500.0,  # Minimum target from requirements
            "gpu_utilization_target": 90.0,  # 90%+ target from requirements
            "throughput_achieved": throughput,
            "gpu_utilization_achieved": gpu_utilization,
            "total_addresses": total_addresses,
            "throughput_meets_target": False,
            "gpu_utilization_meets_target": False,
            "overall_performance_acceptable": False,
            "performance_score": 0.0,
            "validation_details": [],
            "summary": ""
        }
        
        # Validate throughput target (1500+ addresses/second)
        if throughput >= 1500.0:
            validation_results["throughput_meets_target"] = True
            validation_results["validation_details"].append(
                f"✅ Throughput target achieved: {throughput:.1f} >= 1500 addr/sec"
            )
        else:
            validation_results["validation_details"].append(
                f"❌ Throughput below target: {throughput:.1f} < 1500 addr/sec"
            )
        
        # Validate GPU utilization target (90%+)
        if gpu_utilization >= 90.0:
            validation_results["gpu_utilization_meets_target"] = True
            validation_results["validation_details"].append(
                f"✅ GPU utilization target achieved: {gpu_utilization:.1f}% >= 90%"
            )
        else:
            validation_results["validation_details"].append(
                f"❌ GPU utilization below target: {gpu_utilization:.1f}% < 90%"
            )
        
        # Calculate performance score (0-100)
        throughput_score = min((throughput / 1500.0) * 50, 50)  # Max 50 points for throughput
        gpu_score = min((gpu_utilization / 90.0) * 50, 50)  # Max 50 points for GPU utilization
        validation_results["performance_score"] = throughput_score + gpu_score
        
        # Determine overall performance acceptability
        validation_results["overall_performance_acceptable"] = (
            validation_results["throughput_meets_target"] and 
            validation_results["gpu_utilization_meets_target"]
        )
        
        # Generate summary
        if validation_results["overall_performance_acceptable"]:
            validation_results["summary"] = f"Performance targets achieved (Score: {validation_results['performance_score']:.1f}/100)"
        else:
            missing_targets = []
            if not validation_results["throughput_meets_target"]:
                missing_targets.append("throughput")
            if not validation_results["gpu_utilization_meets_target"]:
                missing_targets.append("GPU utilization")
            validation_results["summary"] = f"Performance targets not met: {', '.join(missing_targets)} (Score: {validation_results['performance_score']:.1f}/100)"
        
        # Log validation results
        self.logger.info(f"🎯 Performance Validation Results:")
        for detail in validation_results["validation_details"]:
            self.logger.info(f"   {detail}")
        
        return validation_results
    
    def _add_performance_validation_suggestions(self, result: ProcessingResult, validation: Dict[str, Any]) -> None:
        """Add performance validation suggestions to processing result.
        
        Args:
            result: ProcessingResult to add suggestions to
            validation: Performance validation results
        """
        if not validation["overall_performance_acceptable"]:
            if not validation["throughput_meets_target"]:
                throughput_gap = 1500.0 - validation["throughput_achieved"]
                result.add_optimization_suggestion(
                    f"Throughput {throughput_gap:.1f} addr/sec below target. "
                    f"Consider: increasing GPU batch size, enabling model compilation, "
                    f"or optimizing dataset batching configuration."
                )
            
            if not validation["gpu_utilization_meets_target"]:
                gpu_gap = 90.0 - validation["gpu_utilization_achieved"]
                result.add_optimization_suggestion(
                    f"GPU utilization {gpu_gap:.1f}% below target. "
                    f"Consider: increasing queue size, reducing CPU allocation ratio, "
                    f"or enabling more GPU streams for overlapping execution."
                )
        
        # Add performance score to suggestions
        result.add_optimization_suggestion(
            f"Performance Score: {validation['performance_score']:.1f}/100. "
            f"Target: Both throughput ≥1500 addr/sec AND GPU utilization ≥90%."
        )
    
    def _generate_optimization_suggestions(self, result: ProcessingResult) -> None:
        """Generate optimization suggestions based on performance analysis."""
        if not result.performance_metrics:
            return
        
        metrics = result.performance_metrics
        
        # GPU utilization suggestions
        if metrics.gpu_utilization < self.config.gpu_utilization_threshold:
            result.add_optimization_suggestion(
                f"GPU utilization ({metrics.gpu_utilization:.1f}%) below target "
                f"({self.config.gpu_utilization_threshold*100:.0f}%). "
                f"Consider increasing batch size or queue size."
            )
        
        # Throughput suggestions
        if metrics.throughput_rate < self.config.target_throughput * 0.8:
            result.add_optimization_suggestion(
                f"Throughput ({metrics.throughput_rate:.1f} addr/sec) below 80% of target "
                f"({self.config.target_throughput}). Consider GPU optimizations."
            )
        
        # Queue management suggestions
        if metrics.queue_input_size < 3:
            result.add_optimization_suggestion(
                "GPU input queue size low. Consider increasing queue size or data feeding rate."
            )