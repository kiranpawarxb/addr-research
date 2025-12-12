# Sustained GPU Maximizer - Project Structure

## Directory Structure

```
sustained-gpu-maximizer/
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── models.py                      # Data models (ParsedAddress, etc.)
│   └── ultimate_multi_device_parser.py # Multi-device parser base class
├── sustained_gpu_maximizer.py         # Main application script
├── run_example.py                     # Example usage script
├── test_setup.py                      # Setup verification script
├── run.bat                           # Windows batch runner
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup script
├── README.md                         # Comprehensive documentation
├── PROJECT_STRUCTURE.md              # This file
└── .gitignore                        # Git ignore patterns
```

## File Descriptions

### Core Application Files

- **`sustained_gpu_maximizer.py`**: Main application that implements the sustained GPU processing logic with asynchronous batch processing, multiple GPU streams, and continuous feeding.

- **`src/ultimate_multi_device_parser.py`**: Base parser class that handles multi-device processing including NVIDIA GPU, Intel GPU/OpenVINO, and CPU cores.

- **`src/models.py`**: Data models including `ParsedAddress`, `AddressRecord`, and other structured data classes.

### Utility Scripts

- **`run_example.py`**: Demonstrates how to use the system with sample data. Creates test CSV files and shows basic usage patterns.

- **`test_setup.py`**: Comprehensive setup verification that tests imports, GPU availability, model downloads, and basic functionality.

- **`run.bat`**: Windows batch script that provides a menu-driven interface for running different components.

### Configuration Files

- **`requirements.txt`**: All Python dependencies including PyTorch, Transformers, OpenVINO, and other required packages.

- **`setup.py`**: Package installation script for pip-based installation.

- **`.gitignore`**: Excludes temporary files, logs, data files, and environment-specific files from version control.

## Key Features Implemented

### 1. Sustained GPU Processing
- Asynchronous batch processing with GPU queuing
- Multiple GPU streams (default: 2) for overlapping execution
- Pre-loaded data batches to eliminate CPU-GPU sync delays
- Continuous GPU feeding without idle time

### 2. Performance Monitoring
- Real-time GPU utilization monitoring via nvidia-smi
- Processing rate tracking (addresses/second)
- Queue status and batch completion logging
- ETA calculations and progress reporting

### 3. Smart File Management
- Automatic detection of CSV files matching patterns
- Skip already processed files to avoid duplication
- Timestamped output files for tracking
- Comprehensive logging with rotation

### 4. Error Handling & Robustness
- GPU availability checks and fallback to CPU
- Memory management and CUDA optimization
- Timeout handling for stuck processes
- Graceful degradation and error recovery

## Usage Patterns

### 1. Quick Start
```bash
# Test setup
python test_setup.py

# Run with sample data
python run_example.py

# Process your files
python sustained_gpu_maximizer.py
```

### 2. Windows Users
```cmd
# Use the batch runner
run.bat
```

### 3. Custom Integration
```python
from sustained_gpu_maximizer import SustainedGPUMaximizer

processor = SustainedGPUMaximizer(
    batch_size=200,
    use_nvidia_gpu=True,
    use_intel_gpu=False,
    use_all_cpu_cores=False
)

result = processor.process_single_file_sustained("your_file.csv")
```

## Dependencies

### Core Dependencies
- `pandas>=2.0.0` - Data manipulation and CSV handling
- `torch>=2.0.0` - PyTorch for GPU processing
- `transformers>=4.30.0` - Hugging Face transformers for NER models
- `openvino>=2023.0.0` - Intel OpenVINO for Intel GPU support

### Optional Dependencies
- `psutil>=5.9.0` - System monitoring
- `tqdm>=4.66.0` - Progress bars
- `pytest>=7.4.0` - Testing framework

## Hardware Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA-compatible drivers

### Recommended Requirements
- Python 3.10+
- 16GB+ RAM
- NVIDIA RTX 4070 or better
- Latest NVIDIA drivers and CUDA toolkit

## Performance Characteristics

### Expected Performance (RTX 4070)
- **Throughput**: 200+ addresses/second
- **GPU Utilization**: 90%+ sustained
- **Batch Processing**: 200 addresses per batch
- **Memory Usage**: ~6GB GPU VRAM

### Scalability
- Automatically adjusts to available GPU memory
- Supports multiple GPU streams for higher utilization
- CPU fallback for systems without GPU
- Configurable batch sizes for different hardware

## Troubleshooting

### Common Issues
1. **GPU Not Detected**: Check NVIDIA drivers and CUDA installation
2. **Memory Errors**: Reduce batch_size parameter
3. **Slow Performance**: Verify GPU utilization with nvidia-smi
4. **Import Errors**: Install requirements with pip install -r requirements.txt

### Debug Mode
Enable detailed logging by modifying the logging level in setup_logging():
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Extension Points

The system is designed for easy extension:

1. **Custom Parsers**: Extend `UltimateMultiDeviceParser` for different models
2. **Output Formats**: Modify `save_results()` for different output formats
3. **Monitoring**: Add custom metrics in the progress logging
4. **Preprocessing**: Extend `_clean_address_text()` for domain-specific cleaning

## License & Support

This project is provided as-is for address parsing and consolidation tasks. For issues or questions, refer to the comprehensive logging and error messages provided by the system.