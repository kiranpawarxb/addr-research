#!/usr/bin/env python3
"""
Command-line interface for GPU-CPU Hybrid Batch File Processing.

This script provides a command-line interface for processing multiple CSV files
using the GPU-CPU hybrid processing system with smart resume capabilities.

Usage:
    python batch_process_hybrid.py --patterns "*.csv" --output-dir batch_output
    python batch_process_hybrid.py --resume batch_20231201_120000
    python batch_process_hybrid.py --patterns "p*.csv" --no-skip --continue-on-error

Requirements: 6.1, 6.2, 6.3, 6.5
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from src.batch_file_processor import BatchFileProcessor
from src.hybrid_processor import ProcessingConfiguration


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    else:
        # Default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(f'batch_processing_{timestamp}.log', encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def create_processing_config(args) -> ProcessingConfiguration:
    """Create processing configuration from command line arguments."""
    return ProcessingConfiguration(
        gpu_batch_size=args.gpu_batch_size,
        dataset_batch_size=args.dataset_batch_size,
        gpu_memory_fraction=args.gpu_memory_fraction,
        gpu_queue_size=args.gpu_queue_size,
        num_gpu_streams=args.num_gpu_streams,
        cpu_allocation_ratio=args.cpu_allocation_ratio,
        cpu_batch_size=args.cpu_batch_size,
        target_throughput=args.target_throughput,
        gpu_utilization_threshold=args.gpu_utilization_threshold
    )


def main():
    """Main CLI function for batch file processing."""
    
    parser = argparse.ArgumentParser(
        description="GPU-CPU Hybrid Batch File Processing with Smart Resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CSV files in current directory
  python batch_process_hybrid.py --patterns "*.csv"
  
  # Process specific file patterns with custom output directory
  python batch_process_hybrid.py --patterns "export_*.csv" "sample_*.csv" --output-dir results
  
  # Resume interrupted batch processing
  python batch_process_hybrid.py --resume batch_20231201_120000
  
  # Process with custom GPU settings
  python batch_process_hybrid.py --patterns "*.csv" --gpu-batch-size 800 --target-throughput 2500
  
  # Process without skipping already processed files
  python batch_process_hybrid.py --patterns "*.csv" --no-skip --continue-on-error
        """
    )
    
    # File Processing Arguments
    parser.add_argument(
        '--patterns', 
        nargs='+', 
        default=['*.csv'],
        help='File patterns to process (default: *.csv)'
    )
    
    parser.add_argument(
        '--output-dir', 
        default='batch_output',
        help='Output directory for results and state files (default: batch_output)'
    )
    
    parser.add_argument(
        '--resume', 
        help='Resume processing from specific batch ID'
    )
    
    parser.add_argument(
        '--no-skip', 
        action='store_true',
        help='Do not skip already processed files'
    )
    
    parser.add_argument(
        '--continue-on-error', 
        action='store_true',
        default=True,
        help='Continue processing after individual file failures (default: True)'
    )
    
    # GPU Processing Configuration
    gpu_group = parser.add_argument_group('GPU Processing Options')
    gpu_group.add_argument(
        '--gpu-batch-size', 
        type=int, 
        default=400,
        help='GPU batch size for dataset processing (100-1000, default: 400)'
    )
    
    gpu_group.add_argument(
        '--dataset-batch-size', 
        type=int, 
        default=1000,
        help='HuggingFace dataset batch size (default: 1000)'
    )
    
    gpu_group.add_argument(
        '--gpu-memory-fraction', 
        type=float, 
        default=0.95,
        help='GPU memory allocation fraction (0.80-0.98, default: 0.95)'
    )
    
    gpu_group.add_argument(
        '--gpu-queue-size', 
        type=int, 
        default=10,
        help='Number of pre-loaded GPU batches (5-20, default: 10)'
    )
    
    gpu_group.add_argument(
        '--num-gpu-streams', 
        type=int, 
        default=2,
        help='Number of GPU streams for overlapping execution (default: 2)'
    )
    
    # CPU Processing Configuration
    cpu_group = parser.add_argument_group('CPU Processing Options')
    cpu_group.add_argument(
        '--cpu-allocation-ratio', 
        type=float, 
        default=0.02,
        help='CPU workload allocation ratio (0.001-0.5, default: 0.02)'
    )
    
    cpu_group.add_argument(
        '--cpu-batch-size', 
        type=int, 
        default=50,
        help='CPU batch size for overflow processing (default: 50)'
    )
    
    # Performance Configuration
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument(
        '--target-throughput', 
        type=int, 
        default=2000,
        help='Target processing throughput in addresses/second (500-3000, default: 2000)'
    )
    
    perf_group.add_argument(
        '--gpu-utilization-threshold', 
        type=float, 
        default=0.90,
        help='Minimum GPU utilization threshold (default: 0.90)'
    )
    
    # Logging Configuration
    log_group = parser.add_argument_group('Logging Options')
    log_group.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    log_group.add_argument(
        '--log-file', 
        help='Custom log file path (default: auto-generated with timestamp)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    print("🚀 GPU-CPU HYBRID BATCH FILE PROCESSOR")
    print("=" * 60)
    
    # Display configuration
    print("🔧 Processing Configuration:")
    print(f"   File Patterns: {args.patterns}")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   GPU Batch Size: {args.gpu_batch_size}")
    print(f"   Target Throughput: {args.target_throughput} addresses/second")
    print(f"   GPU Utilization Target: {args.gpu_utilization_threshold*100:.0f}%")
    print(f"   CPU Allocation: {args.cpu_allocation_ratio*100:.1f}%")
    print(f"   Skip Processed Files: {not args.no_skip}")
    print(f"   Continue on Error: {args.continue_on_error}")
    
    if args.resume:
        print(f"   Resume Batch ID: {args.resume}")
    
    print()
    
    try:
        # Create processing configuration
        config = create_processing_config(args)
        
        # Initialize batch processor
        processor = BatchFileProcessor(config, args.output_dir)
        
        # Process files
        print("🚀 Starting batch processing...")
        
        batch_report = processor.process_files_batch(
            file_patterns=args.patterns,
            resume_batch_id=args.resume,
            skip_processed=not args.no_skip,
            continue_on_error=args.continue_on_error
        )
        
        # Display results
        print("\n🎉 BATCH PROCESSING COMPLETED!")
        print("=" * 60)
        
        print("📊 Processing Summary:")
        print(f"   Files Processed: {batch_report.total_files_processed}")
        print(f"   Total Addresses: {batch_report.total_addresses:,}")
        print(f"   Average Throughput: {batch_report.average_throughput:.1f} addresses/second")
        print(f"   Peak Throughput: {batch_report.peak_throughput:.1f} addresses/second")
        print(f"   Average GPU Utilization: {batch_report.average_gpu_utilization:.1f}%")
        print(f"   Processing Efficiency: {batch_report.processing_efficiency:.1f}%")
        
        print(f"\n📁 Results saved to: {args.output_dir}")
        
        # Show output files
        results_dir = Path(args.output_dir) / "results"
        if results_dir.exists():
            output_files = list(results_dir.glob("*.csv"))
            if output_files:
                print(f"\n📄 Output Files ({len(output_files)}):")
                for output_file in sorted(output_files)[-5:]:  # Show last 5 files
                    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                    print(f"   - {output_file.name} ({file_size:.1f} MB)")
                
                if len(output_files) > 5:
                    print(f"   ... and {len(output_files) - 5} more files")
        
        print(f"\n📋 Performance Summary:")
        print(batch_report.performance_summary)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        print("💾 Processing state has been saved for resume capability")
        return False
        
    except Exception as e:
        print(f"\n❌ Batch processing failed: {e}")
        logging.error(f"Batch processing failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)