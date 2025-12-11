"""Logging configuration for GPU-CPU Hybrid Processing System.

Provides specialized logging setup for hybrid processing with performance monitoring,
GPU utilization tracking, and detailed processing metrics logging.

Requirements: 8.1
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional
from datetime import datetime


class HybridProcessingFormatter(logging.Formatter):
    """Custom formatter for hybrid processing logs with performance context."""
    
    def __init__(self):
        super().__init__()
        self.base_format = "[{asctime}] {levelname:8} | {name:25} | {message}"
        self.performance_format = "[{asctime}] {levelname:8} | PERF | {message}"
    
    def format(self, record):
        """Format log record with appropriate style based on logger name."""
        if 'performance' in record.name.lower() or 'gpu' in record.name.lower():
            formatter = logging.Formatter(self.performance_format, style='{')
        else:
            formatter = logging.Formatter(self.base_format, style='{')
        
        return formatter.format(record)


def setup_hybrid_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_performance: bool = True,
    disable_progress: bool = False,
    max_log_size_mb: int = 50,
    backup_count: int = 5
) -> logging.Logger:
    """Set up comprehensive logging for hybrid processing system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, uses default naming
        enable_performance: Enable detailed performance logging
        disable_progress: Disable progress output to console
        max_log_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger for hybrid processing
        
    Requirements: 8.1
    """
    # Create main logger
    logger = logging.getLogger('hybrid_processor')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create custom formatter
    formatter = HybridProcessingFormatter()
    
    # Console handler with color support (unless progress is disabled)
    if not disable_progress:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"hybrid_processing_{timestamp}.log"
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
    os.makedirs(log_dir, exist_ok=True)
    
    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_log_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG to file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Performance logger for detailed metrics
    if enable_performance:
        perf_logger = logging.getLogger('hybrid_processor.performance')
        perf_logger.setLevel(logging.INFO)
        
        # Separate performance log file
        perf_log_file = log_file.replace('.log', '_performance.log')
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=max_log_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(formatter)
        perf_logger.addHandler(perf_handler)
        
        # Also log performance to console at INFO level (unless progress is disabled)
        if not disable_progress:
            perf_console = logging.StreamHandler(sys.stdout)
            perf_console.setLevel(logging.INFO)
            perf_console.setFormatter(formatter)
            perf_logger.addHandler(perf_console)
    
    logger.info(f"Hybrid processing logging initialized: level={level}, file={log_file}")
    
    return logger


def get_hybrid_logger(name: str) -> logging.Logger:
    """Get a logger for hybrid processing components.
    
    Args:
        name: Logger name (will be prefixed with 'hybrid_processor.')
        
    Returns:
        Configured logger instance
    """
    full_name = f"hybrid_processor.{name}" if not name.startswith('hybrid_processor') else name
    return logging.getLogger(full_name)


def log_performance_metrics(
    logger: logging.Logger,
    throughput: float,
    gpu_utilization: float,
    queue_size: int,
    processing_time: float,
    addresses_count: int
) -> None:
    """Log performance metrics in a standardized format.
    
    Args:
        logger: Logger instance to use
        throughput: Current throughput in addresses per second
        gpu_utilization: GPU utilization percentage
        queue_size: Current GPU queue size
        processing_time: Processing time in seconds
        addresses_count: Number of addresses processed
    """
    logger.info(
        f"METRICS | Throughput: {throughput:.1f} addr/sec | "
        f"GPU: {gpu_utilization:.1f}% | Queue: {queue_size} | "
        f"Time: {processing_time:.2f}s | Count: {addresses_count}"
    )


def log_gpu_optimization_status(
    logger: logging.Logger,
    model_compiled: bool,
    half_precision: bool,
    memory_fraction: float,
    num_streams: int
) -> None:
    """Log GPU optimization configuration status.
    
    Args:
        logger: Logger instance to use
        model_compiled: Whether model compilation is enabled
        half_precision: Whether half-precision is enabled
        memory_fraction: GPU memory allocation fraction
        num_streams: Number of GPU streams configured
    """
    logger.info(
        f"GPU_CONFIG | Compiled: {model_compiled} | "
        f"FP16: {half_precision} | Memory: {memory_fraction:.1%} | "
        f"Streams: {num_streams}"
    )


def log_workload_distribution(
    logger: logging.Logger,
    gpu_addresses: int,
    cpu_addresses: int,
    total_addresses: int
) -> None:
    """Log workload distribution between GPU and CPU.
    
    Args:
        logger: Logger instance to use
        gpu_addresses: Number of addresses allocated to GPU
        cpu_addresses: Number of addresses allocated to CPU
        total_addresses: Total number of addresses
    """
    gpu_ratio = (gpu_addresses / total_addresses * 100) if total_addresses > 0 else 0
    cpu_ratio = (cpu_addresses / total_addresses * 100) if total_addresses > 0 else 0
    
    logger.info(
        f"WORKLOAD | GPU: {gpu_addresses} ({gpu_ratio:.1f}%) | "
        f"CPU: {cpu_addresses} ({cpu_ratio:.1f}%) | "
        f"Total: {total_addresses}"
    )