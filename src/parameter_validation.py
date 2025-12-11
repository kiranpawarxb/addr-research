"""Parameter validation for GPU-CPU Hybrid Processing System CLI.

Provides comprehensive validation functions for all CLI parameters according to
system requirements and hardware constraints.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


class ParameterValidationError(Exception):
    """Exception raised for parameter validation errors."""
    pass


def validate_gpu_parameters(
    batch_size: Optional[int] = None,
    memory_fraction: Optional[float] = None,
    queue_size: Optional[int] = None,
    num_streams: Optional[int] = None,
    cuda_device: Optional[int] = None
) -> Dict[str, Any]:
    """Validate GPU configuration parameters.
    
    Args:
        batch_size: GPU batch size for dataset processing
        memory_fraction: GPU memory allocation fraction
        queue_size: GPU queue size for pre-loaded batches
        num_streams: Number of GPU streams
        cuda_device: CUDA device ID
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ParameterValidationError: If parameters are invalid
        
    Requirements: 8.1, 8.2
    """
    validated = {}
    
    # Validate GPU batch size (100-1000)
    if batch_size is not None:
        if not isinstance(batch_size, int) or not (100 <= batch_size <= 1000):
            raise ParameterValidationError(
                f"GPU batch size must be an integer between 100 and 1000, got {batch_size}"
            )
        validated['batch_size'] = batch_size
    
    # Validate GPU memory fraction (80-98%)
    if memory_fraction is not None:
        if not isinstance(memory_fraction, (int, float)) or not (0.80 <= memory_fraction <= 0.98):
            raise ParameterValidationError(
                f"GPU memory fraction must be between 0.80 and 0.98, got {memory_fraction}"
            )
        validated['memory_fraction'] = float(memory_fraction)
    
    # Validate GPU queue size (5-20)
    if queue_size is not None:
        if not isinstance(queue_size, int) or not (5 <= queue_size <= 20):
            raise ParameterValidationError(
                f"GPU queue size must be an integer between 5 and 20, got {queue_size}"
            )
        validated['queue_size'] = queue_size
    
    # Validate number of GPU streams (1-8)
    if num_streams is not None:
        if not isinstance(num_streams, int) or not (1 <= num_streams <= 8):
            raise ParameterValidationError(
                f"Number of GPU streams must be an integer between 1 and 8, got {num_streams}"
            )
        validated['num_streams'] = num_streams
    
    # Validate CUDA device ID (0-7)
    if cuda_device is not None:
        if not isinstance(cuda_device, int) or cuda_device < 0:
            raise ParameterValidationError(
                f"CUDA device ID must be a non-negative integer, got {cuda_device}"
            )
        validated['cuda_device'] = cuda_device
    
    return validated


def validate_cpu_parameters(
    allocation_ratio: Optional[float] = None,
    batch_size: Optional[int] = None,
    worker_count: Optional[int] = None,
    max_memory_gb: Optional[float] = None,
    core_limit: Optional[int] = None
) -> Dict[str, Any]:
    """Validate CPU configuration parameters.
    
    Args:
        allocation_ratio: CPU allocation ratio (0.001-0.5)
        batch_size: CPU batch size for overflow processing
        worker_count: Number of CPU workers
        max_memory_gb: Maximum system memory usage in GB
        core_limit: CPU core limit
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ParameterValidationError: If parameters are invalid
        
    Requirements: 8.3, 8.4
    """
    validated = {}
    
    # Validate CPU allocation ratio (0.1-0.5 of cores, translated to 0.001-0.5)
    if allocation_ratio is not None:
        if not isinstance(allocation_ratio, (int, float)) or not (0.001 <= allocation_ratio <= 0.5):
            raise ParameterValidationError(
                f"CPU allocation ratio must be between 0.001 and 0.5, got {allocation_ratio}"
            )
        validated['allocation_ratio'] = float(allocation_ratio)
    
    # Validate CPU batch size (10-500)
    if batch_size is not None:
        if not isinstance(batch_size, int) or not (10 <= batch_size <= 500):
            raise ParameterValidationError(
                f"CPU batch size must be an integer between 10 and 500, got {batch_size}"
            )
        validated['batch_size'] = batch_size
    
    # Validate CPU worker count (1-32)
    if worker_count is not None:
        if not isinstance(worker_count, int) or not (1 <= worker_count <= 32):
            raise ParameterValidationError(
                f"CPU worker count must be an integer between 1 and 32, got {worker_count}"
            )
        validated['worker_count'] = worker_count
    
    # Validate maximum memory usage (1.0-128.0 GB)
    if max_memory_gb is not None:
        if not isinstance(max_memory_gb, (int, float)) or not (1.0 <= max_memory_gb <= 128.0):
            raise ParameterValidationError(
                f"Maximum memory usage must be between 1.0 and 128.0 GB, got {max_memory_gb}"
            )
        validated['max_memory_gb'] = float(max_memory_gb)
    
    # Validate CPU core limit (1-64)
    if core_limit is not None:
        if not isinstance(core_limit, int) or not (1 <= core_limit <= 64):
            raise ParameterValidationError(
                f"CPU core limit must be an integer between 1 and 64, got {core_limit}"
            )
        validated['core_limit'] = core_limit
    
    return validated


def validate_performance_parameters(
    target_throughput: Optional[int] = None,
    gpu_threshold: Optional[float] = None,
    log_interval: Optional[int] = None,
    monitoring_interval: Optional[int] = None
) -> Dict[str, Any]:
    """Validate performance configuration parameters.
    
    Args:
        target_throughput: Target processing rate in addresses/second
        gpu_threshold: Minimum GPU utilization threshold
        log_interval: Performance logging interval in seconds
        monitoring_interval: GPU monitoring interval in seconds
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ParameterValidationError: If parameters are invalid
        
    Requirements: 8.5
    """
    validated = {}
    
    # Validate target throughput (500-3000 addresses/second)
    if target_throughput is not None:
        if not isinstance(target_throughput, int) or not (500 <= target_throughput <= 3000):
            raise ParameterValidationError(
                f"Target throughput must be an integer between 500 and 3000, got {target_throughput}"
            )
        validated['target_throughput'] = target_throughput
    
    # Validate GPU utilization threshold (0.0-1.0)
    if gpu_threshold is not None:
        if not isinstance(gpu_threshold, (int, float)) or not (0.0 <= gpu_threshold <= 1.0):
            raise ParameterValidationError(
                f"GPU utilization threshold must be between 0.0 and 1.0, got {gpu_threshold}"
            )
        validated['gpu_threshold'] = float(gpu_threshold)
    
    # Validate performance logging interval (1-300 seconds)
    if log_interval is not None:
        if not isinstance(log_interval, int) or not (1 <= log_interval <= 300):
            raise ParameterValidationError(
                f"Performance logging interval must be between 1 and 300 seconds, got {log_interval}"
            )
        validated['log_interval'] = log_interval
    
    # Validate GPU monitoring interval (1-60 seconds)
    if monitoring_interval is not None:
        if not isinstance(monitoring_interval, int) or not (1 <= monitoring_interval <= 60):
            raise ParameterValidationError(
                f"GPU monitoring interval must be between 1 and 60 seconds, got {monitoring_interval}"
            )
        validated['monitoring_interval'] = monitoring_interval
    
    return validated


def validate_file_paths(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    batch_input_pattern: Optional[str] = None,
    batch_output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """Validate file and directory paths.
    
    Args:
        input_path: Single input file path
        output_path: Single output file path
        batch_input_pattern: Batch input glob pattern
        batch_output_dir: Batch output directory
        config_path: Configuration file path
        log_file: Log file path
        
    Returns:
        Dictionary of validated paths
        
    Raises:
        ParameterValidationError: If paths are invalid
    """
    validated = {}
    
    # Validate input file path
    if input_path is not None:
        input_file = Path(input_path)
        if not input_file.exists():
            raise ParameterValidationError(f"Input file does not exist: {input_path}")
        if not input_file.is_file():
            raise ParameterValidationError(f"Input path is not a file: {input_path}")
        if not input_file.suffix.lower() == '.csv':
            raise ParameterValidationError(f"Input file must be a CSV file: {input_path}")
        validated['input_path'] = str(input_file.resolve())
    
    # Validate output file path
    if output_path is not None:
        output_file = Path(output_path)
        output_dir = output_file.parent
        
        # Check if output directory exists or can be created
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ParameterValidationError(f"Cannot create output directory {output_dir}: {e}")
        
        # Check write permissions
        if not os.access(output_dir, os.W_OK):
            raise ParameterValidationError(f"No write permission for output directory: {output_dir}")
        
        validated['output_path'] = str(output_file.resolve())
    
    # Validate batch input pattern
    if batch_input_pattern is not None:
        import glob
        matching_files = glob.glob(batch_input_pattern)
        if not matching_files:
            raise ParameterValidationError(f"No files match batch input pattern: {batch_input_pattern}")
        
        # Validate that all matching files are CSV files
        non_csv_files = [f for f in matching_files if not f.lower().endswith('.csv')]
        if non_csv_files:
            raise ParameterValidationError(f"Non-CSV files found in batch input: {non_csv_files[:3]}")
        
        validated['batch_input_pattern'] = batch_input_pattern
        validated['matching_files'] = matching_files
    
    # Validate batch output directory
    if batch_output_dir is not None:
        output_dir = Path(batch_output_dir)
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ParameterValidationError(f"Cannot create batch output directory {output_dir}: {e}")
        
        if not os.access(output_dir, os.W_OK):
            raise ParameterValidationError(f"No write permission for batch output directory: {output_dir}")
        
        validated['batch_output_dir'] = str(output_dir.resolve())
    
    # Validate configuration file path
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise ParameterValidationError(f"Configuration file does not exist: {config_path}")
        if not config_file.is_file():
            raise ParameterValidationError(f"Configuration path is not a file: {config_path}")
        validated['config_path'] = str(config_file.resolve())
    
    # Validate log file path
    if log_file is not None:
        log_path = Path(log_file)
        log_dir = log_path.parent
        
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ParameterValidationError(f"Cannot create log directory {log_dir}: {e}")
        
        if not os.access(log_dir, os.W_OK):
            raise ParameterValidationError(f"No write permission for log directory: {log_dir}")
        
        validated['log_file'] = str(log_path.resolve())
    
    return validated


def validate_parameter_combinations(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    batch_input: Optional[str] = None,
    batch_output: Optional[str] = None,
    gpu_batch_size: Optional[int] = None,
    cpu_allocation_ratio: Optional[float] = None
) -> None:
    """Validate parameter combinations and dependencies.
    
    Args:
        input_path: Single input file path
        output_path: Single output file path
        batch_input: Batch input pattern
        batch_output: Batch output directory
        gpu_batch_size: GPU batch size
        cpu_allocation_ratio: CPU allocation ratio
        
    Raises:
        ParameterValidationError: If parameter combinations are invalid
    """
    # Validate input/output combinations
    if input_path and batch_input:
        raise ParameterValidationError("Cannot specify both single input and batch input")
    
    if output_path and batch_output:
        raise ParameterValidationError("Cannot specify both single output and batch output")
    
    if input_path and not output_path:
        raise ParameterValidationError("Output path is required when input path is specified")
    
    if batch_input and not batch_output:
        raise ParameterValidationError("Batch output directory is required when batch input is specified")
    
    if not input_path and not batch_input:
        raise ParameterValidationError("Must specify either single input or batch input")
    
    # Validate GPU/CPU balance
    if gpu_batch_size and cpu_allocation_ratio:
        # Warn if CPU allocation is too high for large GPU batches
        if gpu_batch_size > 600 and cpu_allocation_ratio > 0.05:
            logging.warning(
                f"High CPU allocation ({cpu_allocation_ratio:.1%}) with large GPU batch size "
                f"({gpu_batch_size}) may reduce overall performance"
            )


def validate_all_parameters(
    gpu_params: Dict[str, Any],
    cpu_params: Dict[str, Any],
    performance_params: Dict[str, Any],
    file_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate all parameter categories together.
    
    Args:
        gpu_params: GPU configuration parameters
        cpu_params: CPU configuration parameters
        performance_params: Performance configuration parameters
        file_params: File path parameters
        
    Returns:
        Dictionary of all validated parameters
        
    Raises:
        ParameterValidationError: If any parameters are invalid
    """
    validated = {}
    
    # Validate each category
    validated.update(validate_gpu_parameters(**gpu_params))
    validated.update(validate_cpu_parameters(**cpu_params))
    validated.update(validate_performance_parameters(**performance_params))
    validated.update(validate_file_paths(**file_params))
    
    # Validate parameter combinations
    validate_parameter_combinations(
        input_path=file_params.get('input_path'),
        output_path=file_params.get('output_path'),
        batch_input=file_params.get('batch_input_pattern'),
        batch_output=file_params.get('batch_output_dir'),
        gpu_batch_size=gpu_params.get('batch_size'),
        cpu_allocation_ratio=cpu_params.get('allocation_ratio')
    )
    
    return validated


def get_parameter_recommendations(
    gpu_memory_gb: Optional[float] = None,
    cpu_cores: Optional[int] = None,
    target_throughput: Optional[int] = None
) -> Dict[str, Any]:
    """Get parameter recommendations based on hardware capabilities.
    
    Args:
        gpu_memory_gb: Available GPU memory in GB
        cpu_cores: Number of CPU cores
        target_throughput: Target processing throughput
        
    Returns:
        Dictionary of recommended parameters
    """
    recommendations = {}
    
    # GPU recommendations based on memory
    if gpu_memory_gb is not None:
        if gpu_memory_gb >= 12:
            recommendations.update({
                'gpu_batch_size': 800,
                'gpu_memory_fraction': 0.95,
                'gpu_queue_size': 15,
                'num_gpu_streams': 3
            })
        elif gpu_memory_gb >= 8:
            recommendations.update({
                'gpu_batch_size': 600,
                'gpu_memory_fraction': 0.90,
                'gpu_queue_size': 12,
                'num_gpu_streams': 2
            })
        else:
            recommendations.update({
                'gpu_batch_size': 400,
                'gpu_memory_fraction': 0.85,
                'gpu_queue_size': 8,
                'num_gpu_streams': 2
            })
    
    # CPU recommendations based on cores
    if cpu_cores is not None:
        if cpu_cores >= 16:
            recommendations.update({
                'cpu_allocation_ratio': 0.02,
                'cpu_worker_count': 4,
                'cpu_batch_size': 75
            })
        elif cpu_cores >= 8:
            recommendations.update({
                'cpu_allocation_ratio': 0.03,
                'cpu_worker_count': 3,
                'cpu_batch_size': 60
            })
        else:
            recommendations.update({
                'cpu_allocation_ratio': 0.05,
                'cpu_worker_count': 2,
                'cpu_batch_size': 50
            })
    
    # Performance recommendations based on target
    if target_throughput is not None:
        if target_throughput >= 2500:
            recommendations.update({
                'performance_log_interval': 5,
                'gpu_utilization_threshold': 0.95
            })
        elif target_throughput >= 2000:
            recommendations.update({
                'performance_log_interval': 10,
                'gpu_utilization_threshold': 0.90
            })
        else:
            recommendations.update({
                'performance_log_interval': 15,
                'gpu_utilization_threshold': 0.85
            })
    
    return recommendations