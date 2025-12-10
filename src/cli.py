"""Command-line interface for the Address Consolidation System.

This module provides a CLI for running the address consolidation pipeline
with configurable options for input, output, and logging.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.config_loader import ConfigLoader, ConfigurationError, Config
from src.pipeline import AddressConsolidationPipeline


def setup_logging(level: str, log_file: Optional[str] = None) -> None:
    """Configure logging with console and optional file handlers.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging to file: {log_file}")
        except Exception as e:
            logging.warning(f"Could not create log file {log_file}: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog='address-consolidation',
        description='Address Consolidation System - Parse and consolidate Indian addresses using LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config file
  python -m src.cli
  
  # Run with custom config file
  python -m src.cli --config my_config.yaml
  
  # Override input and output files
  python -m src.cli --input addresses.csv --output results.csv
  
  # Enable debug logging
  python -m src.cli --verbose --log-file debug.log
  
  # Run with all options
  python -m src.cli --config config.yaml --input data.csv --output out.csv --verbose
        """
    )
    
    # Configuration file
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration YAML file (default: config/config.yaml)'
    )
    
    # Input file override
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Path to input CSV file (overrides config file setting)'
    )
    
    # Output file override
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output CSV file (overrides config file setting)'
    )
    
    # Logging options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output (WARNING level only)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (overrides config file setting)'
    )
    
    # Statistics option
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Disable statistics display at the end'
    )
    
    # Version
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    return parser.parse_args()


def load_and_override_config(args: argparse.Namespace) -> Config:
    """Load configuration and apply command-line overrides.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Config object with overrides applied
        
    Raises:
        ConfigurationError: If configuration cannot be loaded
    """
    # Load base configuration
    try:
        config_loader = ConfigLoader(args.config)
        config = config_loader.load()
    except ConfigurationError as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        raise
    
    # Apply command-line overrides
    if args.input:
        config.input.file_path = args.input
        logging.info(f"Input file overridden to: {args.input}")
    
    if args.output:
        config.output.file_path = args.output
        logging.info(f"Output file overridden to: {args.output}")
    
    if args.no_stats:
        config.output.include_statistics = False
        logging.info("Statistics display disabled")
    
    # Override logging configuration
    if args.verbose:
        config.logging.level = "DEBUG"
    elif args.quiet:
        config.logging.level = "WARNING"
    
    if args.log_file:
        config.logging.file_path = args.log_file
    
    return config


def validate_paths(config: Config) -> None:
    """Validate that required paths exist and are accessible.
    
    Args:
        config: Configuration object
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        PermissionError: If output directory is not writable
    """
    # Check input file exists
    input_path = Path(config.input.file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {config.input.file_path}")
    
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {config.input.file_path}")
    
    # Check output directory is writable
    output_path = Path(config.output.file_path)
    output_dir = output_path.parent
    
    if output_dir and not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        except Exception as e:
            raise PermissionError(f"Cannot create output directory {output_dir}: {e}")
    
    # Test write permissions
    try:
        test_file = output_dir / '.write_test'
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"Output directory is not writable {output_dir}: {e}")


def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load and override configuration
        config = load_and_override_config(args)
        
        # Setup logging
        setup_logging(config.logging.level, config.logging.file_path)
        
        # Log startup
        logging.info("=" * 70)
        logging.info("Address Consolidation System - Starting")
        logging.info("=" * 70)
        logging.info(f"Configuration file: {args.config}")
        logging.info(f"Input file: {config.input.file_path}")
        logging.info(f"Output file: {config.output.file_path}")
        logging.info(f"Log level: {config.logging.level}")
        
        # Validate paths
        validate_paths(config)
        
        # Create and run pipeline
        pipeline = AddressConsolidationPipeline(config)
        pipeline.run()
        
        # Success
        logging.info("=" * 70)
        logging.info("Address Consolidation System - Completed Successfully")
        logging.info("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.", file=sys.stderr)
        logging.warning("Operation cancelled by user")
        return 130  # Standard exit code for SIGINT
        
    except ConfigurationError as e:
        print(f"\nConfiguration Error: {e}", file=sys.stderr)
        logging.error(f"Configuration error: {e}")
        return 1
        
    except FileNotFoundError as e:
        print(f"\nFile Not Found: {e}", file=sys.stderr)
        logging.error(f"File not found: {e}")
        return 2
        
    except PermissionError as e:
        print(f"\nPermission Error: {e}", file=sys.stderr)
        logging.error(f"Permission error: {e}")
        return 3
        
    except ValueError as e:
        print(f"\nValidation Error: {e}", file=sys.stderr)
        logging.error(f"Validation error: {e}")
        return 4
        
    except Exception as e:
        print(f"\nUnexpected Error: {e}", file=sys.stderr)
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 5


if __name__ == '__main__':
    sys.exit(main())
