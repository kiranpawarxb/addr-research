"""Performance monitoring for GPU-CPU hybrid address processing.

This module provides real-time performance tracking and optimization feedback
for the hybrid processing system. It monitors GPU utilization, throughput rates,
queue status, and processing efficiency to ensure sustained high-performance operation.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import logging
import time
import threading
import subprocess
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from queue import Queue, Empty
import psutil
import os

try:
    from .models import ParsedAddress
    from .hybrid_processor import PerformanceMetrics, ProcessingConfiguration
except ImportError:
    # Fallback for direct execution
    from models import ParsedAddress
    from hybrid_processor import PerformanceMetrics, ProcessingConfiguration


@dataclass
class QueueStatus:
    """Status information for GPU processing queues.
    
    Tracks queue sizes, processing pipeline status, and bottleneck detection
    for asynchronous processing optimization.
    
    Requirements: 5.3
    """
    input_queue_size: int = 0
    output_queue_size: int = 0
    max_queue_size: int = 0
    processing_workers_active: int = 0
    data_feeder_active: bool = False
    result_collector_active: bool = False
    queue_utilization_percent: float = 0.0
    bottleneck_detected: bool = False
    bottleneck_location: str = ""
    
    def __post_init__(self):
        """Calculate queue utilization and detect bottlenecks."""
        if self.max_queue_size > 0:
            self.queue_utilization_percent = (self.input_queue_size / self.max_queue_size) * 100
        
        # Detect bottlenecks
        if self.input_queue_size == 0 and self.processing_workers_active > 0:
            self.bottleneck_detected = True
            self.bottleneck_location = "data_feeding"
        elif self.output_queue_size >= self.max_queue_size * 0.8:
            self.bottleneck_detected = True
            self.bottleneck_location = "result_collection"


@dataclass
class GPUStats:
    """NVIDIA GPU statistics from nvidia-smi.
    
    Contains real-time GPU utilization, memory usage, temperature,
    and device information for performance monitoring.
    
    Requirements: 5.1
    """
    gpu_id: int = 0
    name: str = ""
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_utilization_percent: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.memory_total_mb > 0:
            self.memory_utilization_percent = (self.memory_used_mb / self.memory_total_mb) * 100


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report.
    
    Provides detailed performance metrics, efficiency analysis,
    optimization suggestions, and comparative performance data.
    
    Requirements: 5.4, 5.5
    """
    # Report Metadata
    report_timestamp: float = field(default_factory=time.time)
    monitoring_duration: float = 0.0
    
    # Performance Summary
    average_throughput: float = 0.0
    peak_throughput: float = 0.0
    average_gpu_utilization: float = 0.0
    peak_gpu_utilization: float = 0.0
    processing_efficiency: float = 0.0
    
    # Resource Utilization
    gpu_stats: List[GPUStats] = field(default_factory=list)
    cpu_utilization: float = 0.0
    memory_usage_gb: float = 0.0
    
    # Queue Performance
    queue_performance: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization Analysis
    performance_warnings: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Comparative Analysis
    target_throughput: int = 2000
    target_gpu_utilization: float = 90.0
    performance_score: float = 0.0
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        # Throughput score (40% weight)
        throughput_score = min((self.average_throughput / self.target_throughput) * 100, 100) * 0.4
        
        # GPU utilization score (40% weight)
        gpu_score = min((self.average_gpu_utilization / self.target_gpu_utilization) * 100, 100) * 0.4
        
        # Efficiency score (20% weight)
        efficiency_score = self.processing_efficiency * 0.2
        
        self.performance_score = throughput_score + gpu_score + efficiency_score
        return self.performance_score
    
    def generate_summary(self) -> str:
        """Generate human-readable performance summary."""
        self.calculate_performance_score()
        
        summary = f"""
Performance Report Summary:
==========================
Monitoring Duration: {self.monitoring_duration:.1f} seconds
Performance Score: {self.performance_score:.1f}/100

Throughput Performance:
- Average: {self.average_throughput:.1f} addresses/second
- Peak: {self.peak_throughput:.1f} addresses/second
- Target: {self.target_throughput} addresses/second
- Achievement: {(self.average_throughput/self.target_throughput)*100:.1f}%

GPU Utilization:
- Average: {self.average_gpu_utilization:.1f}%
- Peak: {self.peak_gpu_utilization:.1f}%
- Target: {self.target_gpu_utilization:.1f}%
- Achievement: {(self.average_gpu_utilization/self.target_gpu_utilization)*100:.1f}%

Processing Efficiency: {self.processing_efficiency:.1f}%

Resource Usage:
- CPU Utilization: {self.cpu_utilization:.1f}%
- System Memory: {self.memory_usage_gb:.1f} GB
        """.strip()
        
        if self.gpu_stats:
            summary += "\n\nGPU Details:"
            for gpu in self.gpu_stats:
                summary += f"\n- GPU {gpu.gpu_id} ({gpu.name}): {gpu.utilization_percent:.1f}% util, "
                summary += f"{gpu.memory_utilization_percent:.1f}% memory, {gpu.temperature_c:.0f}°C"
        
        if self.performance_warnings:
            summary += "\n\nPerformance Warnings:"
            for warning in self.performance_warnings:
                summary += f"\n⚠️  {warning}"
        
        if self.optimization_suggestions:
            summary += "\n\nOptimization Suggestions:"
            for suggestion in self.optimization_suggestions:
                summary += f"\n💡 {suggestion}"
        
        return summary


class PerformanceMonitor:
    """Real-time performance monitoring for GPU-CPU hybrid processing.
    
    Provides comprehensive performance tracking including GPU utilization monitoring,
    throughput rate calculation, queue status tracking, and performance reporting
    with optimization feedback.
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    
    def __init__(self, config: ProcessingConfiguration):
        """Initialize the performance monitor with configuration.
        
        Args:
            config: Processing configuration with performance thresholds
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance Tracking State
        self.current_metrics = PerformanceMetrics()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.shutdown_requested = False
        
        # Performance History
        self.throughput_history: List[float] = []
        self.gpu_utilization_history: List[float] = []
        self.performance_samples: List[PerformanceMetrics] = []
        
        # Queue Monitoring
        self.queue_status = QueueStatus()
        self.queue_manager = None  # Will be set by hybrid processor
        
        # Timing and Statistics
        self.monitoring_start_time = time.time()
        self.last_log_time = time.time()
        self.addresses_processed_since_last_log = 0
        
        # Thread Safety
        self.metrics_lock = threading.Lock()
        
        self.logger.info(f"Initialized PerformanceMonitor with {config.performance_log_interval}s intervals")
    
    def start_monitoring(self) -> None:
        """Start real-time performance monitoring with configured intervals.
        
        Begins continuous monitoring of GPU utilization, throughput rates,
        and queue status with logging every 10 seconds.
        
        Requirements: 5.1, 5.2, 5.3
        """
        if self.monitoring_active:
            self.logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_start_time = time.time()
        self.shutdown_requested = False
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Started real-time performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring and cleanup resources."""
        if not self.monitoring_active:
            return
        
        self.shutdown_requested = True
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Stopped performance monitoring")
    
    def track_gpu_utilization(self) -> float:
        """Track real-time GPU utilization using nvidia-smi integration.
        
        Queries NVIDIA GPU statistics and updates current metrics
        with utilization, memory usage, and temperature data.
        
        Returns:
            Current GPU utilization percentage
            
        Requirements: 5.1
        """
        try:
            gpu_stats = self._get_nvidia_gpu_stats()
            
            if gpu_stats:
                # Use primary GPU (GPU 0) for main metrics
                primary_gpu = gpu_stats[0]
                
                with self.metrics_lock:
                    self.current_metrics.gpu_utilization = primary_gpu.utilization_percent
                    self.current_metrics.memory_usage = primary_gpu.memory_utilization_percent
                
                # Update GPU utilization history
                self.gpu_utilization_history.append(primary_gpu.utilization_percent)
                
                # Keep only recent history (last 100 samples)
                if len(self.gpu_utilization_history) > 100:
                    self.gpu_utilization_history = self.gpu_utilization_history[-100:]
                
                return primary_gpu.utilization_percent
            else:
                self.logger.warning("No GPU statistics available")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Failed to track GPU utilization: {e}")
            return 0.0
    
    def calculate_throughput_rate(self, addresses_processed: int = 0) -> float:
        """Calculate current processing throughput rate.
        
        Computes addresses per second based on recent processing activity
        and updates throughput history for trend analysis.
        
        Args:
            addresses_processed: Number of addresses processed since last calculation
            
        Returns:
            Current throughput rate in addresses per second
            
        Requirements: 5.2
        """
        current_time = time.time()
        
        # Update addresses processed counter
        if addresses_processed > 0:
            self.addresses_processed_since_last_log += addresses_processed
        
        # Calculate throughput since last update
        time_elapsed = current_time - self.current_metrics.last_update_time
        
        if time_elapsed >= 1.0:  # Update every second minimum
            if self.addresses_processed_since_last_log > 0:
                throughput = self.addresses_processed_since_last_log / time_elapsed
                
                with self.metrics_lock:
                    self.current_metrics.throughput_rate = throughput
                    self.current_metrics.last_update_time = current_time
                    self.current_metrics.total_processed += self.addresses_processed_since_last_log
                
                # Update throughput history
                self.throughput_history.append(throughput)
                
                # Keep only recent history (last 100 samples)
                if len(self.throughput_history) > 100:
                    self.throughput_history = self.throughput_history[-100:]
                
                # Reset counter
                self.addresses_processed_since_last_log = 0
                
                return throughput
        
        return self.current_metrics.throughput_rate
    
    def monitor_queue_status(self) -> QueueStatus:
        """Monitor GPU queue sizes and processing pipeline status.
        
        Tracks input/output queue sizes, worker activity, and detects
        bottlenecks in the asynchronous processing pipeline.
        
        Returns:
            Current queue status with bottleneck detection
            
        Requirements: 5.3
        """
        try:
            if self.queue_manager:
                # Get queue information from queue manager
                queue_info = self.queue_manager.get_queue_status()
                
                # Handle both QueueStatus object and dictionary
                if hasattr(queue_info, 'input_queue_size'):
                    # QueueStatus object
                    self.queue_status = queue_info
                else:
                    # Dictionary format
                    self.queue_status = QueueStatus(
                        input_queue_size=queue_info.get('input_size', 0),
                        output_queue_size=queue_info.get('output_size', 0),
                        max_queue_size=queue_info.get('max_size', self.config.gpu_queue_size),
                        processing_workers_active=queue_info.get('active_workers', 0),
                        data_feeder_active=queue_info.get('data_feeder_active', False),
                        result_collector_active=queue_info.get('result_collector_active', False)
                    )
                
                # Update current metrics
                with self.metrics_lock:
                    self.current_metrics.queue_input_size = self.queue_status.input_queue_size
                    self.current_metrics.queue_output_size = self.queue_status.output_queue_size
                    self.current_metrics.queue_max_size = self.queue_status.max_queue_size
            
            return self.queue_status
            
        except Exception as e:
            self.logger.error(f"Failed to monitor queue status: {e}")
            return QueueStatus()
    
    def generate_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance analysis report.
        
        Creates detailed performance report with metrics analysis,
        optimization suggestions, and comparative performance data.
        
        Returns:
            Comprehensive performance report with analysis
            
        Requirements: 5.4, 5.5
        """
        try:
            current_time = time.time()
            monitoring_duration = current_time - self.monitoring_start_time
            
            # Calculate performance statistics
            avg_throughput = sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0.0
            peak_throughput = max(self.throughput_history) if self.throughput_history else 0.0
            
            avg_gpu_util = sum(self.gpu_utilization_history) / len(self.gpu_utilization_history) if self.gpu_utilization_history else 0.0
            peak_gpu_util = max(self.gpu_utilization_history) if self.gpu_utilization_history else 0.0
            
            # Get current system stats
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.used / (1024**3)
            
            # Get GPU statistics
            gpu_stats = self._get_nvidia_gpu_stats()
            
            # Calculate processing efficiency
            efficiency = self.current_metrics.calculate_efficiency()
            
            # Create performance report
            report = PerformanceReport(
                monitoring_duration=monitoring_duration,
                average_throughput=avg_throughput,
                peak_throughput=peak_throughput,
                average_gpu_utilization=avg_gpu_util,
                peak_gpu_utilization=peak_gpu_util,
                processing_efficiency=efficiency,
                gpu_stats=gpu_stats,
                cpu_utilization=cpu_percent,
                memory_usage_gb=memory_gb,
                queue_performance={
                    'input_queue_avg': self.queue_status.input_queue_size,
                    'output_queue_avg': self.queue_status.output_queue_size,
                    'bottlenecks_detected': self.queue_status.bottleneck_detected,
                    'bottleneck_location': self.queue_status.bottleneck_location
                },
                target_throughput=self.config.target_throughput,
                target_gpu_utilization=self.config.gpu_utilization_threshold * 100
            )
            
            # Generate performance warnings and suggestions
            self._analyze_performance_issues(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return PerformanceReport()
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot.
        
        Returns:
            Current performance metrics with real-time data
        """
        with self.metrics_lock:
            # Update efficiency calculation
            self.current_metrics.calculate_efficiency()
            return self.current_metrics
    
    def set_queue_manager(self, queue_manager) -> None:
        """Set the queue manager for queue status monitoring.
        
        Args:
            queue_manager: AsynchronousQueueManager instance
        """
        self.queue_manager = queue_manager
        self.logger.debug("Queue manager set for performance monitoring")
    
    def log_performance_update(self, addresses_processed: int = 0) -> None:
        """Log performance update if interval has elapsed.
        
        Args:
            addresses_processed: Number of addresses processed since last log
        """
        current_time = time.time()
        
        if current_time - self.last_log_time >= self.config.performance_log_interval:
            # Update metrics
            gpu_util = self.track_gpu_utilization()
            throughput = self.calculate_throughput_rate(addresses_processed)
            queue_status = self.monitor_queue_status()
            
            # Log performance summary
            self.logger.info(
                f"Performance Update: "
                f"GPU={gpu_util:.1f}%, "
                f"Throughput={throughput:.1f} addr/sec, "
                f"Queue={queue_status.input_queue_size}/{queue_status.max_queue_size}, "
                f"Total={self.current_metrics.total_processed}"
            )
            
            # Check for performance warnings
            if gpu_util < self.config.gpu_utilization_threshold * 100:
                self.logger.warning(f"GPU utilization ({gpu_util:.1f}%) below target "
                                  f"({self.config.gpu_utilization_threshold*100:.0f}%)")
            
            if throughput < self.config.target_throughput * 0.8:
                self.logger.warning(f"Throughput ({throughput:.1f}) below 80% of target "
                                  f"({self.config.target_throughput})")
            
            self.last_log_time = current_time
    
    # Private helper methods
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        self.logger.info("Performance monitoring loop started")
        
        while not self.shutdown_requested and self.monitoring_active:
            try:
                # Update all metrics
                self.track_gpu_utilization()
                self.calculate_throughput_rate()
                self.monitor_queue_status()
                
                # Store performance sample
                with self.metrics_lock:
                    sample = PerformanceMetrics(
                        gpu_utilization=self.current_metrics.gpu_utilization,
                        throughput_rate=self.current_metrics.throughput_rate,
                        processing_efficiency=self.current_metrics.processing_efficiency,
                        queue_input_size=self.current_metrics.queue_input_size,
                        queue_output_size=self.current_metrics.queue_output_size,
                        memory_usage=self.current_metrics.memory_usage,
                        total_processed=self.current_metrics.total_processed
                    )
                
                self.performance_samples.append(sample)
                
                # Keep only recent samples (last 1000)
                if len(self.performance_samples) > 1000:
                    self.performance_samples = self.performance_samples[-1000:]
                
                # Sleep for monitoring interval
                time.sleep(1.0)  # Monitor every second, log every config.performance_log_interval
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
        
        self.logger.info("Performance monitoring loop stopped")
    
    def _get_nvidia_gpu_stats(self) -> List[GPUStats]:
        """Get NVIDIA GPU statistics using nvidia-smi."""
        try:
            # Run nvidia-smi with comprehensive query
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_stats = []
                
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 8:
                            try:
                                gpu_stats.append(GPUStats(
                                    gpu_id=int(parts[0]) if parts[0] != 'N/A' else 0,
                                    name=parts[1],
                                    utilization_percent=float(parts[2]) if parts[2] != 'N/A' else 0.0,
                                    memory_used_mb=float(parts[3]) if parts[3] != 'N/A' else 0.0,
                                    memory_total_mb=float(parts[4]) if parts[4] != 'N/A' else 0.0,
                                    temperature_c=float(parts[5]) if parts[5] != 'N/A' else 0.0,
                                    power_draw_w=float(parts[6]) if parts[6] != 'N/A' else 0.0,
                                    power_limit_w=float(parts[7]) if parts[7] != 'N/A' else 0.0
                                ))
                            except (ValueError, IndexError) as e:
                                self.logger.debug(f"Failed to parse GPU stats line: {line}, error: {e}")
                                continue
                
                return gpu_stats
            else:
                self.logger.debug(f"nvidia-smi error: {result.stderr}")
                return []
                
        except subprocess.TimeoutExpired:
            self.logger.warning("nvidia-smi timeout")
            return []
        except FileNotFoundError:
            self.logger.warning("nvidia-smi not found - GPU monitoring disabled")
            return []
        except Exception as e:
            self.logger.error(f"Error getting GPU stats: {e}")
            return []
    
    def _analyze_performance_issues(self, report: PerformanceReport) -> None:
        """Analyze performance data and generate warnings/suggestions."""
        # GPU utilization analysis
        if report.average_gpu_utilization < report.target_gpu_utilization:
            report.performance_warnings.append(
                f"GPU utilization ({report.average_gpu_utilization:.1f}%) below target "
                f"({report.target_gpu_utilization:.1f}%)"
            )
            
            if report.average_gpu_utilization < 50:
                report.optimization_suggestions.append(
                    "Increase GPU batch size or reduce CPU allocation ratio to improve GPU utilization"
                )
            elif report.average_gpu_utilization < 70:
                report.optimization_suggestions.append(
                    "Consider increasing GPU queue size or enabling more GPU streams"
                )
        
        # Throughput analysis
        if report.average_throughput < report.target_throughput * 0.8:
            report.performance_warnings.append(
                f"Throughput ({report.average_throughput:.1f}) below 80% of target "
                f"({report.target_throughput})"
            )
            
            if report.average_throughput < report.target_throughput * 0.5:
                report.optimization_suggestions.append(
                    "Enable model compilation and half-precision processing for better throughput"
                )
            else:
                report.optimization_suggestions.append(
                    "Consider increasing dataset batch size or optimizing GPU memory allocation"
                )
        
        # Queue performance analysis
        if report.queue_performance.get('bottlenecks_detected', False):
            bottleneck_location = report.queue_performance.get('bottleneck_location', 'unknown')
            report.performance_warnings.append(f"Processing bottleneck detected: {bottleneck_location}")
            
            if bottleneck_location == 'data_feeding':
                report.optimization_suggestions.append(
                    "Increase data feeder thread count or optimize batch preparation"
                )
            elif bottleneck_location == 'result_collection':
                report.optimization_suggestions.append(
                    "Increase result collector thread count or optimize output processing"
                )
        
        # Memory analysis
        if report.gpu_stats:
            for gpu in report.gpu_stats:
                if gpu.memory_utilization_percent > 95:
                    report.performance_warnings.append(
                        f"GPU {gpu.gpu_id} memory usage very high ({gpu.memory_utilization_percent:.1f}%)"
                    )
                    report.optimization_suggestions.append(
                        "Reduce GPU batch size or enable gradient checkpointing to reduce memory usage"
                    )
                elif gpu.memory_utilization_percent < 70:
                    report.optimization_suggestions.append(
                        f"GPU {gpu.gpu_id} memory underutilized - consider increasing batch size"
                    )
        
        # Temperature analysis
        if report.gpu_stats:
            for gpu in report.gpu_stats:
                if gpu.temperature_c > 80:
                    report.performance_warnings.append(
                        f"GPU {gpu.gpu_id} temperature high ({gpu.temperature_c:.0f}°C)"
                    )
                    report.optimization_suggestions.append(
                        "Monitor GPU cooling and consider reducing processing intensity if temperature remains high"
                    )