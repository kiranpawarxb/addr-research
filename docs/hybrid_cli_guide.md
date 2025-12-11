# GPU-CPU Hybrid Address Processing CLI Guide

## Overview

The GPU-CPU Hybrid Address Processing System provides a high-performance command-line interface for processing addresses at 2000+ addresses per second using optimized GPU acceleration with CPU fallback.

## Quick Start

### Basic Usage

```bash
# Process a single CSV file
python -m src.hybrid_main --input addresses.csv --output results.csv

# Batch process multiple files
python -m src.hybrid_main --batch-input "data/*.csv" --batch-output results/

# Create default configuration files
python -m src.hybrid_main --create-config
```

### Performance Optimization

```bash
# Maximum performance configuration
python -m src.hybrid_main --input data.csv --gpu-batch-size 800 --gpu-memory 0.98 --target-throughput 2500

# Memory-constrained configuration
python -m src.hybrid_main --input data.csv --gpu-memory 0.85 --cpu-ratio 0.05
```

## Configuration Management

### Configuration Files

The system uses two configuration files:
- `config/config.yaml`: Base system configuration
- `config/hybrid_config.yaml`: GPU-CPU hybrid specific settings

### Creating Configuration Files

```bash
python -m src.hybrid_main --create-config
```

### Using Custom Configuration

```bash
python -m src.hybrid_main --config my_config.yaml --hybrid-config my_hybrid.yaml
```

## Command-Line Options

### Input/Output Options

- `--input PATH`: Single input CSV file
- `--output PATH`: Single output CSV file  
- `--batch-input PATTERN`: Glob pattern for batch input (e.g., "*.csv")
- `--batch-output DIR`: Directory for batch output files
- `--resume`: Resume interrupted batch processing

### GPU Configuration

- `--gpu-batch-size SIZE`: GPU batch size (100-1000, default: 400)
- `--dataset-batch-size SIZE`: HuggingFace dataset batch size (default: 1000)
- `--gpu-memory FRACTION`: GPU memory allocation (0.80-0.98, default: 0.95)
- `--gpu-queue-size SIZE`: Pre-loaded batch queue size (5-20, default: 10)
- `--gpu-streams COUNT`: Number of GPU streams (default: 2)
- `--cuda-device ID`: CUDA device ID (default: 0)
- `--disable-compilation`: Disable PyTorch model compilation
- `--disable-half-precision`: Disable float16 processing

### CPU Configuration

- `--cpu-ratio RATIO`: CPU allocation ratio (0.001-0.5, default: 0.02)
- `--cpu-batch-size SIZE`: CPU batch size (default: 50)
- `--cpu-workers COUNT`: Number of CPU workers (default: 2)
- `--max-memory GB`: Maximum system memory usage (default: 16.0)
- `--cpu-cores COUNT`: Limit CPU cores (default: auto-detect)

### Performance Configuration

- `--target-throughput RATE`: Target addresses/second (500-3000, default: 2000)
- `--gpu-threshold PERCENT`: Minimum GPU utilization (0.0-1.0, default: 0.90)
- `--performance-log`: Enable detailed performance logging
- `--performance-interval SECONDS`: Performance logging interval (default: 10)

### Monitoring and Logging

- `--verbose`: Enable DEBUG logging
- `--quiet`: Enable WARNING level only
- `--log-file PATH`: Custom log file path
- `--gpu-monitoring`: Enable real-time GPU monitoring
- `--monitoring-interval SECONDS`: GPU monitoring interval (default: 5)
- `--no-progress`: Disable progress bars

### Advanced Options

- `--dry-run`: Validate configuration without processing
- `--benchmark`: Run performance benchmark
- `--optimize-config`: Auto-optimize configuration
- `--export-config PATH`: Export current configuration

## Examples

### Single File Processing

```bash
# Basic processing
python -m src.hybrid_main --input addresses.csv --output results.csv

# High-performance processing
python -m src.hybrid_main \
  --input large_dataset.csv \
  --output results.csv \
  --gpu-batch-size 800 \
  --gpu-memory 0.98 \
  --target-throughput 2500 \
  --verbose

# Memory-constrained processing
python -m src.hybrid_main \
  --input addresses.csv \
  --output results.csv \
  --gpu-memory 0.85 \
  --cpu-ratio 0.05 \
  --cpu-workers 4
```
### Batch Processing

```bash
# Process all CSV files in a directory
python -m src.hybrid_main \
  --batch-input "data/*.csv" \
  --batch-output results/ \
  --performance-log

# Resume interrupted batch processing
python -m src.hybrid_main \
  --batch-input "data/*.csv" \
  --batch-output results/ \
  --resume

# Batch processing with custom configuration
python -m src.hybrid_main \
  --batch-input "export_*.csv" \
  --batch-output processed/ \
  --gpu-batch-size 600 \
  --target-throughput 2200
```

### Performance Monitoring

```bash
# Enable comprehensive monitoring
python -m src.hybrid_main \
  --input data.csv \
  --output results.csv \
  --gpu-monitoring \
  --performance-log \
  --verbose \
  --log-file detailed.log

# Run performance benchmark
python -m src.hybrid_main --benchmark

# Validate configuration without processing
python -m src.hybrid_main \
  --input data.csv \
  --output results.csv \
  --dry-run
```

### Configuration Management

```bash
# Export current configuration
python -m src.hybrid_main \
  --gpu-batch-size 800 \
  --gpu-memory 0.98 \
  --export-config optimized_config.yaml

# Use exported configuration
python -m src.hybrid_main \
  --hybrid-config optimized_config.yaml \
  --input data.csv \
  --output results.csv
```

## Performance Tuning

### GPU Optimization

For maximum GPU performance:
- Increase `--gpu-batch-size` to 800-1000
- Set `--gpu-memory` to 0.98 (if memory allows)
- Use `--gpu-streams 3` for high-end GPUs
- Keep `--cpu-ratio` low (0.01-0.02)

### Memory-Constrained Systems

For systems with limited GPU memory:
- Reduce `--gpu-batch-size` to 200-400
- Set `--gpu-memory` to 0.80-0.85
- Increase `--cpu-ratio` to 0.05-0.10
- Add more `--cpu-workers`

### CPU-Heavy Processing

For systems with powerful CPUs but limited GPU:
- Set `--cpu-ratio` to 0.10-0.20
- Increase `--cpu-workers` to match CPU cores
- Reduce `--gpu-batch-size` to 200-300
- Set `--gpu-memory` to 0.80

## Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   ```bash
   # Reduce GPU batch size and memory allocation
   --gpu-batch-size 200 --gpu-memory 0.80
   ```

2. **Low Throughput**
   ```bash
   # Enable monitoring to identify bottlenecks
   --gpu-monitoring --performance-log --verbose
   ```

3. **Configuration Errors**
   ```bash
   # Validate configuration
   --dry-run
   ```

### Performance Diagnostics

```bash
# Run benchmark to test current configuration
python -m src.hybrid_main --benchmark

# Monitor GPU utilization during processing
python -m src.hybrid_main \
  --input data.csv \
  --output results.csv \
  --gpu-monitoring \
  --monitoring-interval 2
```

## Requirements Validation

The CLI validates all parameters according to system requirements:

- GPU batch size: 100-1000 addresses (Requirement 8.1)
- GPU memory allocation: 80-98% (Requirement 8.2)  
- GPU queue size: 5-20 batches (Requirement 8.3)
- CPU worker ratio: 0.1-0.5 of cores (Requirement 8.4)
- Target throughput: 500-3000 addresses/second (Requirement 8.5)

## Exit Codes

- 0: Success
- 1: General error
- 2: Configuration error
- 3: File not found
- 4: Permission error
- 5: Validation error
- 130: User cancellation (Ctrl+C)