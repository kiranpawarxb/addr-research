#!/usr/bin/env python3
"""
GPU-CPU Hybrid Address Processing CLI Examples

This script demonstrates various usage patterns for the hybrid processing CLI
with different configuration scenarios and performance optimization strategies.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> None:
    """Run a CLI command and display the description.
    
    Args:
        cmd: Command to execute
        description: Description of what the command does
    """
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    # Note: In a real scenario, you would uncomment the following line
    # subprocess.run(cmd, shell=True, check=True)
    print("(Command execution disabled in example script)")


def main():
    """Demonstrate various CLI usage patterns."""
    
    print("GPU-CPU Hybrid Address Processing CLI Examples")
    print("=" * 60)
    
    # Basic Examples
    run_command(
        "python -m src.hybrid_main --input addresses.csv --output results.csv",
        "Basic single file processing with default settings"
    )
    
    run_command(
        "python -m src.hybrid_main --batch-input 'data/*.csv' --batch-output results/",
        "Batch processing all CSV files in data directory"
    )
    
    # Configuration Management
    run_command(
        "python -m src.hybrid_main --create-config",
        "Create default configuration files"
    )
    
    run_command(
        "python -m src.hybrid_main --config custom.yaml --hybrid-config hybrid.yaml --input data.csv --output results.csv",
        "Use custom configuration files"
    )
    
    # Performance Optimization Examples
    run_command(
        "python -m src.hybrid_main --input large_dataset.csv --output results.csv --gpu-batch-size 800 --gpu-memory 0.98 --target-throughput 2500",
        "Maximum performance configuration for high-end GPU"
    )
    
    run_command(
        "python -m src.hybrid_main --input data.csv --output results.csv --gpu-memory 0.85 --cpu-ratio 0.05 --cpu-workers 4",
        "Memory-constrained configuration with increased CPU usage"
    )
    
    run_command(
        "python -m src.hybrid_main --input data.csv --output results.csv --cpu-ratio 0.15 --cpu-workers 8 --gpu-batch-size 300",
        "CPU-heavy configuration for systems with limited GPU"
    )
    
    # Monitoring and Debugging Examples
    run_command(
        "python -m src.hybrid_main --input data.csv --output results.csv --verbose --performance-log --gpu-monitoring",
        "Comprehensive monitoring and logging"
    )
    
    run_command(
        "python -m src.hybrid_main --benchmark",
        "Run performance benchmark with current configuration"
    )
    
    run_command(
        "python -m src.hybrid_main --input data.csv --output results.csv --dry-run",
        "Validate configuration without processing"
    )
    
    # Advanced Configuration Examples
    run_command(
        "python -m src.hybrid_main --input data.csv --gpu-batch-size 600 --gpu-streams 3 --gpu-queue-size 15 --export-config optimized.yaml",
        "Export optimized configuration for reuse"
    )
    
    run_command(
        "python -m src.hybrid_main --batch-input 'export_*.csv' --batch-output processed/ --resume --performance-interval 5",
        "Resume interrupted batch processing with frequent performance logging"
    )
    
    # Specialized Hardware Examples
    run_command(
        "python -m src.hybrid_main --input data.csv --output results.csv --cuda-device 1 --gpu-memory 0.90 --disable-half-precision",
        "Use specific GPU device with full precision"
    )
    
    run_command(
        "python -m src.hybrid_main --input data.csv --output results.csv --disable-compilation --gpu-batch-size 400 --max-memory 32.0",
        "Disable optimizations for compatibility with increased memory limit"
    )
    
    # Error Handling and Recovery Examples
    run_command(
        "python -m src.hybrid_main --batch-input 'data/*.csv' --batch-output results/ --cpu-ratio 0.10 --log-file error_recovery.log",
        "Batch processing with increased CPU fallback and detailed logging"
    )
    
    print(f"\n{'='*60}")
    print("Example Usage Patterns Complete")
    print("=" * 60)
    print("\nTo run these examples:")
    print("1. Ensure you have input CSV files in the specified locations")
    print("2. Uncomment the subprocess.run() line in the run_command() function")
    print("3. Run this script: python examples/hybrid_cli_examples.py")
    print("\nFor more information, see: docs/hybrid_cli_guide.md")


if __name__ == "__main__":
    main()