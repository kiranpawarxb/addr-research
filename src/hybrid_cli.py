"""Command-line interface for GPU-CPU Hybrid Address Processing System.

This module provides a comprehensive CLI for running high-performance hybrid processing
with configurable GPU optimization, CPU allocation, performance tuning, and batch processing.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

try:
    from .hybrid_config import HybridConfigLoader, HybridProcessingConfig, create_default_hybrid_config
    from .hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration
    from .batch_file_processor import BatchFileProcessor
    from .hybrid_logging import setup_hybrid_logging
    from .models import ParsedAddress
    from .parameter_validation import (
        validate_gpu_parameters, validate_cpu_parameters, validate_performance_parameters,
        validate_file_paths, validate_parameter_combinations, validate_all_parameters,
        get_parameter_recommendations, ParameterValidationError
    )
except ImportError:
    # Fallback for direct execution
    from hybrid_config import HybridConfigLoader, HybridProcessingConfig, create_default_hybrid_config
    from hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration
    from batch_file_processor import BatchFileProcessor
    from hybrid_logging import setup_hybrid_logging
    from models import ParsedAddress
    from parameter_validation import (
        validate_gpu_parameters, validate_cpu_parameters, validate_performance_parameters,
        validate_file_paths, validate_parameter_combinations, validate_all_parameters,
        get_parameter_recommendations, ParameterValidationError
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser for hybrid processing CLI.
    
    Returns:
        Configured ArgumentParser with all hybrid processing options
    """
    parser = argparse.ArgumentParser(
        prog='hybrid-address-processor',
        description='GPU-CPU Hybrid Address Processing System - High-performance address parsing with 2000+ addresses/second throughput',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing with default configuration
  python -m src.hybrid_cli --input addresses.csv --output results.csv
  
  # Batch processing multiple files
  python -m src.hybrid_cli --batch-input "*.csv" --batch-output results/
  
  # Custom GPU configuration for maximum performance
  python -m src.hybrid_cli --input data.csv --gpu-batch-size 800 --gpu-memory 0.98 --target-throughput 2500
  
  # CPU-heavy configuration for systems with limited GPU memory
  python -m src.hybrid_cli --input data.csv --cpu-ratio 0.1 --gpu-memory 0.85 --cpu-workers 4
  
  # Performance monitoring with detailed logging
  python -m src.hybrid_cli --input data.csv --verbose --performance-log --gpu-monitoring
  
  # Create default configuration files
  python -m src.hybrid_cli --create-config
  
  # Resume interrupted batch processing
  python -m src.hybrid_cli --batch-input "*.csv" --batch-output results/ --resume
        """
    )
    
    # Configuration Management
    config_group = parser.add_argument_group('Configuration Management')
    config_group.add_argument(
        '-c', '--config',
        type=str,
        default='config/config.yaml',
        help='Path to base configuration YAML file (default: config/config.yaml)'
    )
    config_group.add_argument(
        '--hybrid-config',
        type=str,
        help='Path to hybrid-specific configuration YAML file (default: config/hybrid_config.yaml)'
    )
    config_group.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration files and exit'
    )
    
    # Input/Output Configuration
    io_group = parser.add_argument_group('Input/Output Configuration')
    io_group.add_argument(
        '-i', '--input',
        type=str,
        help='Path to input CSV file for single file processing'
    )
    io_group.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output CSV file for single file processing'
    )
    io_group.add_argument(
        '--batch-input',
        type=str,
        help='Glob pattern for batch input files (e.g., "data/*.csv")'
    )
    io_group.add_argument(
        '--batch-output',
        type=str,
        help='Directory for batch output files'
    )
    io_group.add_argument(
        '--resume',
        action='store_true',
        help='Resume interrupted batch processing (skip already processed files)'
    )
    
    # GPU Configuration (Requirements 8.1, 8.2)
    gpu_group = parser.add_argument_group('GPU Configuration')
    gpu_group.add_argument(
        '--gpu-batch-size',
        type=int,
        metavar='SIZE',
        help='GPU batch size for dataset processing (100-1000, default: 400)'
    )
    gpu_group.add_argument(
        '--dataset-batch-size',
        type=int,
        metavar='SIZE',
        help='HuggingFace dataset batch size (default: 1000)'
    )
    gpu_group.add_argument(
        '--gpu-memory',
        type=float,
        metavar='FRACTION',
        help='GPU memory allocation fraction (0.80-0.98, default: 0.95)'
    )
    gpu_group.add_argument(
        '--gpu-queue-size',
        type=int,
        metavar='SIZE',
        help='GPU queue size for pre-loaded batches (5-20, default: 10)'
    )
    gpu_group.add_argument(
        '--gpu-streams',
        type=int,
        metavar='COUNT',
        help='Number of GPU streams for overlapping execution (default: 2)'
    )
    gpu_group.add_argument(
        '--cuda-device',
        type=int,
        metavar='ID',
        help='CUDA device ID to use (default: 0)'
    )
    gpu_group.add_argument(
        '--disable-compilation',
        action='store_true',
        help='Disable PyTorch model compilation (reduces performance)'
    )
    gpu_group.add_argument(
        '--disable-half-precision',
        action='store_true',
        help='Disable half-precision (float16) processing'
    )
    
    # CPU Configuration (Requirements 8.3, 8.4)
    cpu_group = parser.add_argument_group('CPU Configuration')
    cpu_group.add_argument(
        '--cpu-ratio',
        type=float,
        metavar='RATIO',
        help='CPU allocation ratio (0.001-0.5, default: 0.02 = 2%%)'
    )
    cpu_group.add_argument(
        '--cpu-batch-size',
        type=int,
        metavar='SIZE',
        help='CPU batch size for overflow processing (default: 50)'
    )
    cpu_group.add_argument(
        '--cpu-workers',
        type=int,
        metavar='COUNT',
        help='Number of CPU workers (default: 2)'
    )
    cpu_group.add_argument(
        '--max-memory',
        type=float,
        metavar='GB',
        help='Maximum system memory usage in GB (default: 16.0)'
    )
    cpu_group.add_argument(
        '--cpu-cores',
        type=int,
        metavar='COUNT',
        help='Limit CPU cores (default: auto-detect)'
    )
    
    # Performance Configuration (Requirements 8.5)
    perf_group = parser.add_argument_group('Performance Configuration')
    perf_group.add_argument(
        '--target-throughput',
        type=int,
        metavar='RATE',
        help='Target processing rate in addresses/second (500-3000, default: 2000)'
    )
    perf_group.add_argument(
        '--gpu-threshold',
        type=float,
        metavar='PERCENT',
        help='Minimum GPU utilization threshold (0.0-1.0, default: 0.90)'
    )
    perf_group.add_argument(
        '--performance-log',
        action='store_true',
        help='Enable detailed performance logging'
    )
    perf_group.add_argument(
        '--performance-interval',
        type=int,
        metavar='SECONDS',
        help='Performance logging interval in seconds (default: 10)'
    )
    
    # Monitoring and Logging
    monitor_group = parser.add_argument_group('Monitoring and Logging')
    monitor_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    monitor_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output (WARNING level only)'
    )
    monitor_group.add_argument(
        '--log-file',
        type=str,
        metavar='PATH',
        help='Path to log file (overrides config file setting)'
    )
    monitor_group.add_argument(
        '--gpu-monitoring',
        action='store_true',
        help='Enable real-time GPU monitoring via nvidia-smi'
    )
    monitor_group.add_argument(
        '--monitoring-interval',
        type=int,
        metavar='SECONDS',
        help='GPU monitoring interval in seconds (default: 5)'
    )
    monitor_group.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars and status updates'
    )
    
    # Advanced Options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and show processing plan without executing'
    )
    advanced_group.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark with current configuration'
    )
    advanced_group.add_argument(
        '--optimize-config',
        action='store_true',
        help='Auto-optimize configuration based on hardware capabilities'
    )
    advanced_group.add_argument(
        '--export-config',
        type=str,
        metavar='PATH',
        help='Export current configuration to specified file'
    )
    
    # Version and Help
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0 - GPU-CPU Hybrid Address Processing System'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments and parameter combinations.
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        ParameterValidationError: If arguments are invalid or incompatible
    """
    # Check for special modes that don't require full validation
    if args.create_config or args.benchmark or args.export_config:
        return
    
    # Validate logging combinations first
    if args.verbose and args.quiet:
        raise ParameterValidationError("Cannot specify both --verbose and --quiet")
    
    # Prepare parameter dictionaries for validation
    gpu_params = {
        'batch_size': args.gpu_batch_size,
        'memory_fraction': args.gpu_memory,
        'queue_size': args.gpu_queue_size,
        'num_streams': args.gpu_streams,
        'cuda_device': args.cuda_device
    }
    
    cpu_params = {
        'allocation_ratio': args.cpu_ratio,
        'batch_size': args.cpu_batch_size,
        'worker_count': args.cpu_workers,
        'max_memory_gb': args.max_memory,
        'core_limit': args.cpu_cores
    }
    
    performance_params = {
        'target_throughput': args.target_throughput,
        'gpu_threshold': args.gpu_threshold,
        'log_interval': args.performance_interval,
        'monitoring_interval': args.monitoring_interval
    }
    
    file_params = {
        'input_path': args.input,
        'output_path': args.output,
        'batch_input_pattern': args.batch_input,
        'batch_output_dir': args.batch_output,
        'config_path': args.config if hasattr(args, 'config') else None,
        'log_file': args.log_file
    }
    
    # Validate all parameters
    try:
        validate_all_parameters(gpu_params, cpu_params, performance_params, file_params)
    except ParameterValidationError as e:
        raise ParameterValidationError(f"Parameter validation failed: {e}")


def load_and_override_config(args: argparse.Namespace) -> HybridProcessingConfig:
    """Load configuration and apply command-line overrides.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        HybridProcessingConfig with overrides applied
        
    Raises:
        ConfigurationError: If configuration cannot be loaded
    """
    # Load base configuration
    loader = HybridConfigLoader(args.config, args.hybrid_config)
    config = loader.load()
    
    # Apply GPU configuration overrides
    if args.gpu_batch_size is not None:
        config.processing_config.gpu_batch_size = args.gpu_batch_size
        logging.info(f"GPU batch size overridden to: {args.gpu_batch_size}")
    
    if args.dataset_batch_size is not None:
        config.processing_config.dataset_batch_size = args.dataset_batch_size
        logging.info(f"Dataset batch size overridden to: {args.dataset_batch_size}")
    
    if args.gpu_memory is not None:
        config.processing_config.gpu_memory_fraction = args.gpu_memory
        logging.info(f"GPU memory fraction overridden to: {args.gpu_memory}")
    
    if args.gpu_queue_size is not None:
        config.processing_config.gpu_queue_size = args.gpu_queue_size
        logging.info(f"GPU queue size overridden to: {args.gpu_queue_size}")
    
    if args.gpu_streams is not None:
        config.processing_config.num_gpu_streams = args.gpu_streams
        logging.info(f"GPU streams overridden to: {args.gpu_streams}")
    
    if args.cuda_device is not None:
        config.cuda_device_id = args.cuda_device
        logging.info(f"CUDA device overridden to: {args.cuda_device}")
    
    if args.disable_compilation:
        config.processing_config.enable_model_compilation = False
        logging.info("Model compilation disabled")
    
    if args.disable_half_precision:
        config.processing_config.use_half_precision = False
        logging.info("Half-precision processing disabled")
    
    # Apply CPU configuration overrides
    if args.cpu_ratio is not None:
        config.processing_config.cpu_allocation_ratio = args.cpu_ratio
        logging.info(f"CPU allocation ratio overridden to: {args.cpu_ratio}")
    
    if args.cpu_batch_size is not None:
        config.processing_config.cpu_batch_size = args.cpu_batch_size
        logging.info(f"CPU batch size overridden to: {args.cpu_batch_size}")
    
    if args.cpu_workers is not None:
        config.processing_config.cpu_worker_count = args.cpu_workers
        logging.info(f"CPU worker count overridden to: {args.cpu_workers}")
    
    if args.max_memory is not None:
        config.max_memory_usage_gb = args.max_memory
        logging.info(f"Max memory usage overridden to: {args.max_memory} GB")
    
    if args.cpu_cores is not None:
        config.cpu_core_limit = args.cpu_cores
        logging.info(f"CPU core limit overridden to: {args.cpu_cores}")
    
    # Apply performance configuration overrides
    if args.target_throughput is not None:
        config.processing_config.target_throughput = args.target_throughput
        logging.info(f"Target throughput overridden to: {args.target_throughput}")
    
    if args.gpu_threshold is not None:
        config.processing_config.gpu_utilization_threshold = args.gpu_threshold
        logging.info(f"GPU utilization threshold overridden to: {args.gpu_threshold}")
    
    if args.performance_interval is not None:
        config.processing_config.performance_log_interval = args.performance_interval
        logging.info(f"Performance log interval overridden to: {args.performance_interval}")
    
    # Apply monitoring configuration overrides
    if args.gpu_monitoring:
        config.enable_gpu_monitoring = True
        logging.info("GPU monitoring enabled")
    
    if args.monitoring_interval is not None:
        config.gpu_monitoring_interval = args.monitoring_interval
        logging.info(f"GPU monitoring interval overridden to: {args.monitoring_interval}")
    
    # Apply logging configuration overrides
    if args.verbose:
        config.hybrid_log_level = "DEBUG"
    elif args.quiet:
        config.hybrid_log_level = "WARNING"
    
    if args.log_file:
        config.hybrid_log_file = args.log_file
    
    if args.performance_log:
        config.enable_performance_logging = True
        logging.info("Performance logging enabled")
    
    return config


def setup_logging_from_config(config: HybridProcessingConfig, no_progress: bool = False) -> None:
    """Setup logging based on hybrid configuration.
    
    Args:
        config: Hybrid processing configuration
        no_progress: Whether to disable progress output
    """
    setup_hybrid_logging(
        level=config.hybrid_log_level,
        log_file=config.hybrid_log_file,
        enable_performance=config.enable_performance_logging,
        disable_progress=no_progress
    )


def create_configuration_files(args: argparse.Namespace) -> None:
    """Create default configuration files.
    
    Args:
        args: Parsed command-line arguments
    """
    print("Creating default configuration files...")
    
    # Create base config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Create hybrid configuration file
    hybrid_config_path = config_dir / "hybrid_config.yaml"
    create_default_hybrid_config(str(hybrid_config_path))
    print(f"Created hybrid configuration: {hybrid_config_path}")
    
    # Create base configuration if it doesn't exist
    base_config_path = config_dir / "config.yaml"
    if not base_config_path.exists():
        base_config = {
            'input': {
                'file_path': 'data/addresses.csv',
                'encoding': 'utf-8'
            },
            'output': {
                'file_path': 'results/parsed_addresses.csv',
                'include_statistics': True
            },
            'logging': {
                'level': 'INFO',
                'file_path': None
            }
        }
        
        with open(base_config_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False, indent=2)
        print(f"Created base configuration: {base_config_path}")
    
    print("\nConfiguration files created successfully!")
    print(f"Edit {hybrid_config_path} to customize GPU-CPU hybrid processing settings.")
    print(f"Edit {base_config_path} to customize general system settings.")


def export_configuration(config: HybridProcessingConfig, export_path: str) -> None:
    """Export current configuration to file.
    
    Args:
        config: Current hybrid processing configuration
        export_path: Path to export configuration file
    """
    export_config = {
        'gpu': {
            'batch_size': config.processing_config.gpu_batch_size,
            'dataset_batch_size': config.processing_config.dataset_batch_size,
            'memory_fraction': config.processing_config.gpu_memory_fraction,
            'queue_size': config.processing_config.gpu_queue_size,
            'num_streams': config.processing_config.num_gpu_streams,
            'enable_compilation': config.processing_config.enable_model_compilation,
            'use_half_precision': config.processing_config.use_half_precision,
            'enable_cudnn_benchmark': config.processing_config.enable_cudnn_benchmark,
            'enable_tensor_float32': config.processing_config.enable_tensor_float32,
            'device_id': config.cuda_device_id,
            'enable_monitoring': config.enable_gpu_monitoring,
            'monitoring_interval': config.gpu_monitoring_interval
        },
        'cpu': {
            'allocation_ratio': config.processing_config.cpu_allocation_ratio,
            'batch_size': config.processing_config.cpu_batch_size,
            'worker_count': config.processing_config.cpu_worker_count
        },
        'performance': {
            'log_interval': config.processing_config.performance_log_interval,
            'target_throughput': config.processing_config.target_throughput,
            'gpu_utilization_threshold': config.processing_config.gpu_utilization_threshold
        },
        'logging': {
            'level': config.hybrid_log_level,
            'file_path': config.hybrid_log_file,
            'enable_performance': config.enable_performance_logging
        },
        'system': {
            'max_memory_gb': config.max_memory_usage_gb,
            'cpu_core_limit': config.cpu_core_limit
        }
    }
    
    # Ensure export directory exists
    export_dir = os.path.dirname(export_path)
    if export_dir:  # Only create directory if path has a directory component
        os.makedirs(export_dir, exist_ok=True)
    
    with open(export_path, 'w') as f:
        yaml.dump(export_config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration exported to: {export_path}")


def validate_paths(args: argparse.Namespace) -> None:
    """Validate input and output paths.
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        FileNotFoundError: If input files don't exist
        PermissionError: If output directories are not writable
    """
    # Validate single file input
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        if not input_path.is_file():
            raise ValueError(f"Input path is not a file: {args.input}")
    
    # Validate batch input pattern
    if args.batch_input:
        import glob
        matching_files = glob.glob(args.batch_input)
        if not matching_files:
            raise FileNotFoundError(f"No files match batch input pattern: {args.batch_input}")
    
    # Validate output directories
    if args.output:
        output_path = Path(args.output)
        output_dir = output_path.parent
        if output_dir and not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created output directory: {output_dir}")
            except Exception as e:
                raise PermissionError(f"Cannot create output directory {output_dir}: {e}")
    
    if args.batch_output:
        batch_output_dir = Path(args.batch_output)
        if not batch_output_dir.exists():
            try:
                batch_output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created batch output directory: {batch_output_dir}")
            except Exception as e:
                raise PermissionError(f"Cannot create batch output directory {batch_output_dir}: {e}")


def run_single_file_processing(args: argparse.Namespace, config: HybridProcessingConfig) -> int:
    """Run single file processing.
    
    Args:
        args: Parsed command-line arguments
        config: Hybrid processing configuration
        
    Returns:
        Exit code (0 for success)
    """
    logging.info("=" * 70)
    logging.info("GPU-CPU Hybrid Address Processing - Single File Mode")
    logging.info("=" * 70)
    logging.info(f"Input file: {args.input}")
    logging.info(f"Output file: {args.output}")
    logging.info(f"Target throughput: {config.processing_config.target_throughput} addresses/second")
    logging.info(f"GPU batch size: {config.processing_config.gpu_batch_size}")
    logging.info(f"GPU memory fraction: {config.processing_config.gpu_memory_fraction}")
    
    if args.dry_run:
        logging.info("DRY RUN MODE - Configuration validated successfully")
        return 0
    
    # Create hybrid processor
    processor = GPUCPUHybridProcessor(config.processing_config)
    
    # Initialize processing
    processor.initialize_hybrid_processing()
    
    # Read input addresses
    import pandas as pd
    df = pd.read_csv(args.input)
    addresses = df['address'].tolist() if 'address' in df.columns else df.iloc[:, 0].tolist()
    
    logging.info(f"Processing {len(addresses)} addresses...")
    
    # Process addresses
    start_time = time.time()
    results = processor.process_addresses_hybrid(addresses)
    processing_time = time.time() - start_time
    
    # Calculate performance metrics
    throughput = len(addresses) / processing_time
    logging.info(f"Processing completed in {processing_time:.2f} seconds")
    logging.info(f"Throughput: {throughput:.1f} addresses/second")
    
    # Save results
    output_df = pd.DataFrame([{
        'original_address': addr,
        'parsed_address': str(result)
    } for addr, result in zip(addresses, results)])
    
    output_df.to_csv(args.output, index=False)
    logging.info(f"Results saved to: {args.output}")
    
    return 0


def run_batch_processing(args: argparse.Namespace, config: HybridProcessingConfig) -> int:
    """Run batch file processing.
    
    Args:
        args: Parsed command-line arguments
        config: Hybrid processing configuration
        
    Returns:
        Exit code (0 for success)
    """
    logging.info("=" * 70)
    logging.info("GPU-CPU Hybrid Address Processing - Batch Mode")
    logging.info("=" * 70)
    logging.info(f"Input pattern: {args.batch_input}")
    logging.info(f"Output directory: {args.batch_output}")
    logging.info(f"Resume mode: {args.resume}")
    
    if args.dry_run:
        import glob
        matching_files = glob.glob(args.batch_input)
        logging.info(f"Would process {len(matching_files)} files:")
        for file_path in matching_files:
            logging.info(f"  - {file_path}")
        logging.info("DRY RUN MODE - Configuration validated successfully")
        return 0
    
    # Create batch processor
    batch_processor = BatchFileProcessor(config)
    
    # Run batch processing
    batch_processor.process_batch(
        input_pattern=args.batch_input,
        output_directory=args.batch_output,
        resume=args.resume
    )
    
    return 0


def run_benchmark(config: HybridProcessingConfig) -> int:
    """Run performance benchmark.
    
    Args:
        config: Hybrid processing configuration
        
    Returns:
        Exit code (0 for success)
    """
    logging.info("=" * 70)
    logging.info("GPU-CPU Hybrid Processing - Benchmark Mode")
    logging.info("=" * 70)
    
    # Create test addresses
    test_addresses = [
        f"Test Address {i}, City {i % 10}, State {i % 5}, PIN {100000 + i}"
        for i in range(1000)
    ]
    
    logging.info(f"Running benchmark with {len(test_addresses)} test addresses...")
    
    # Create hybrid processor
    processor = GPUCPUHybridProcessor(config.processing_config)
    processor.initialize_hybrid_processing()
    
    # Run benchmark
    start_time = time.time()
    results = processor.process_addresses_hybrid(test_addresses)
    processing_time = time.time() - start_time
    
    # Calculate and display results
    throughput = len(test_addresses) / processing_time
    
    logging.info("=" * 70)
    logging.info("BENCHMARK RESULTS")
    logging.info("=" * 70)
    logging.info(f"Addresses processed: {len(test_addresses)}")
    logging.info(f"Processing time: {processing_time:.2f} seconds")
    logging.info(f"Throughput: {throughput:.1f} addresses/second")
    logging.info(f"Target throughput: {config.processing_config.target_throughput} addresses/second")
    
    if throughput >= config.processing_config.target_throughput:
        logging.info("✓ BENCHMARK PASSED - Target throughput achieved")
    else:
        logging.warning("⚠ BENCHMARK FAILED - Target throughput not achieved")
        logging.info("Consider optimizing configuration or hardware")
    
    return 0


def main() -> int:
    """Main entry point for the hybrid processing CLI.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Handle configuration creation
        if args.create_config:
            create_configuration_files(args)
            return 0
        
        # Validate arguments
        validate_arguments(args)
        
        # Load and override configuration
        config = load_and_override_config(args)
        
        # Setup logging
        setup_logging_from_config(config, args.no_progress)
        
        # Export configuration if requested
        if args.export_config:
            export_configuration(config, args.export_config)
            return 0
        
        # Validate paths
        validate_paths(args)
        
        # Run appropriate processing mode
        if args.benchmark:
            return run_benchmark(config)
        elif args.input:
            return run_single_file_processing(args, config)
        elif args.batch_input:
            return run_batch_processing(args, config)
        else:
            parser.print_help()
            return 1
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.", file=sys.stderr)
        logging.warning("Operation cancelled by user")
        return 130  # Standard exit code for SIGINT
        
    except ParameterValidationError as e:
        print(f"\nParameter Validation Error: {e}", file=sys.stderr)
        logging.error(f"Parameter validation error: {e}")
        return 4
        
    except Exception as e:
        print(f"\nUnexpected Error: {e}", file=sys.stderr)
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 5


if __name__ == '__main__':
    sys.exit(main())