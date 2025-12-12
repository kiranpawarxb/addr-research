# Sustained GPU Maximizer

A high-performance address parsing system that maintains 90%+ GPU utilization through asynchronous batch processing and continuous GPU feeding.

## Features

- **Sustained GPU Utilization**: Maintains 90%+ NVIDIA GPU utilization
- **Asynchronous Processing**: Pre-loaded data batches eliminate CPU-GPU sync delays
- **Multiple GPU Streams**: Overlapping execution with continuous feeding
- **Real-time Monitoring**: GPU utilization and processing rate monitoring
- **Batch Processing**: Processes multiple CSV files automatically
- **Smart File Detection**: Automatically skips already processed files

## Architecture

The system uses a sophisticated multi-threaded architecture:

1. **Data Feeder Thread**: Pre-processes and queues batches for GPU consumption
2. **GPU Worker Threads**: Multiple streams process batches continuously
3. **Progress Monitor**: Real-time logging of GPU utilization and processing rates
4. **Result Collector**: Asynchronously collects and organizes results

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- NVIDIA drivers and CUDA toolkit
- Required Python packages (see requirements.txt)

## Installation

1. Clone or copy this project to your desired location
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure NVIDIA drivers and CUDA are properly installed
4. Verify GPU access:
   ```bash
   nvidia-smi
   ```

## Usage

### Basic Usage

Place your CSV files in the same directory as the script and run:

```bash
python sustained_gpu_maximizer.py
```

The system will automatically:
- Detect CSV files matching the pattern `export_customer_address_store_p*.csv`
- Skip already processed files
- Process files with sustained GPU utilization
- Save results with timestamps

### CSV File Format

Your CSV files should contain an address column with one of these names:
- `addr_text`
- `address`
- `full_address`
- `Address`
- `addr`

### Output

For each processed file, the system generates:
- **Results CSV**: `{filename}_sustained_gpu_{timestamp}.csv`
- **Log File**: `sustained_gpu_{timestamp}.log`
- **Performance Metrics**: Processing speed, GPU utilization, success rates

## Configuration

Key parameters can be adjusted in the `SustainedGPUMaximizer` class:

```python
# Sustained GPU settings
self.nvidia_batch_size = 200      # Batch size for GPU processing
self.gpu_queue_size = 10          # Number of pre-loaded batches
self.num_gpu_streams = 2          # Parallel GPU streams
```

## Performance Optimization

The system is optimized for:
- **High Throughput**: 200+ addresses/second on RTX 4070
- **Sustained Utilization**: 90%+ GPU utilization maintained
- **Memory Efficiency**: Optimized batch sizes and queue management
- **Minimal CPU Overhead**: Reduced CPU usage to avoid GPU interference

## Monitoring

Real-time monitoring includes:
- Processing rate (addresses/second)
- GPU utilization percentage
- Queue status and batch completion
- ETA calculations
- Success/failure rates

## File Processing Logic

The system intelligently manages file processing:
1. Scans for CSV files matching the pattern
2. Checks for existing output files to avoid reprocessing
3. Processes files in numerical order (p5, p6, p7, etc.)
4. Generates comprehensive batch summaries

## Error Handling

Robust error handling includes:
- GPU availability checks
- CUDA memory management
- Timeout handling for stuck processes
- Graceful degradation to CPU processing
- Detailed error logging

## Output Format

Results are saved with the following columns:
- `id`: Sequential record ID
- `original_address`: Input address text
- `unit_number`: Extracted unit/flat number
- `society_name`: Building/society name
- `landmark`: Nearby landmarks
- `road`: Street/road name
- `sub_locality`: Sub-locality/area
- `locality`: Locality/neighborhood
- `city`: City name
- `district`: District name
- `state`: State name
- `country`: Country (default: India)
- `pin_code`: PIN/postal code
- `parse_success`: Success flag
- `parse_error`: Error message (if failed)
- `note`: Processing notes

## Troubleshooting

### GPU Not Detected
- Verify NVIDIA drivers: `nvidia-smi`
- Check CUDA installation: `nvcc --version`
- Ensure PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Issues
- Reduce `nvidia_batch_size` if GPU memory is insufficient
- Adjust `gpu_queue_size` to manage memory usage
- Monitor GPU memory with `nvidia-smi`

### Performance Issues
- Ensure adequate GPU cooling
- Check for background processes using GPU
- Verify sufficient system RAM
- Monitor CPU usage to avoid bottlenecks

## License

This project is provided as-is for address parsing and consolidation tasks.