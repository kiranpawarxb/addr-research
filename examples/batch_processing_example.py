#!/usr/bin/env python3
"""
Example usage of BatchFileProcessor with smart resume capabilities.

This example demonstrates how to use the BatchFileProcessor for processing
multiple CSV files with automatic detection and skipping of processed files,
error handling, and resume functionality.

Requirements: 6.1, 6.2, 6.3, 6.5
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from batch_file_processor import BatchFileProcessor
from hybrid_processor import ProcessingConfiguration


def setup_logging():
    """Set up logging for batch processing example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('batch_processing_example.log', encoding='utf-8')
        ]
    )


def main():
    """Main example function demonstrating batch file processing."""
    
    setup_logging()
    
    print("🚀 BATCH FILE PROCESSING EXAMPLE")
    print("=" * 50)
    
    # Configure processing parameters
    config = ProcessingConfiguration(
        gpu_batch_size=400,  # Large batches for GPU efficiency
        dataset_batch_size=1000,  # HuggingFace dataset batch size
        gpu_memory_fraction=0.95,  # Use 95% of GPU memory
        gpu_queue_size=10,  # Pre-load 10 batches
        num_gpu_streams=2,  # Multiple GPU streams
        cpu_allocation_ratio=0.02,  # 2% CPU allocation
        target_throughput=2000,  # Target 2000 addresses/second
        gpu_utilization_threshold=0.90  # 90%+ GPU utilization
    )
    
    print("🔧 Configuration:")
    print(f"   GPU Batch Size: {config.gpu_batch_size}")
    print(f"   Target Throughput: {config.target_throughput} addresses/second")
    print(f"   GPU Utilization Target: {config.gpu_utilization_threshold*100:.0f}%")
    print(f"   CPU Allocation: {config.cpu_allocation_ratio*100:.1f}%")
    print()
    
    # Initialize batch processor
    output_dir = "batch_processing_output"
    processor = BatchFileProcessor(config, output_dir)
    
    print(f"📁 Output Directory: {output_dir}")
    print()
    
    # Define file patterns to process
    file_patterns = [
        "export_customer_address_store_p*.csv",  # Process all P files
        "sample_*.csv",  # Process sample files
        # Add more patterns as needed
    ]
    
    print("📂 File Patterns:")
    for pattern in file_patterns:
        print(f"   - {pattern}")
    print()
    
    try:
        # Example 1: New batch processing
        print("🚀 Example 1: New Batch Processing")
        print("-" * 40)
        
        batch_report = processor.process_files_batch(
            file_patterns=file_patterns,
            skip_processed=True,  # Skip already processed files
            continue_on_error=True  # Continue processing after errors
        )
        
        print("📊 Batch Processing Results:")
        print(f"   Files Processed: {batch_report.total_files_processed}")
        print(f"   Total Addresses: {batch_report.total_addresses:,}")
        print(f"   Average Throughput: {batch_report.average_throughput:.1f} addr/sec")
        print(f"   Peak Throughput: {batch_report.peak_throughput:.1f} addr/sec")
        print(f"   Average GPU Utilization: {batch_report.average_gpu_utilization:.1f}%")
        print()
        
        # Example 2: Resume interrupted processing
        print("🔄 Example 2: Resume Interrupted Processing")
        print("-" * 40)
        
        # Simulate resuming from a specific batch ID
        # In practice, you would get this from a previous interrupted run
        batch_id_to_resume = "batch_20231201_120000"  # Example batch ID
        
        try:
            resumed_report = processor.process_files_batch(
                file_patterns=file_patterns,
                resume_batch_id=batch_id_to_resume,  # Resume from specific batch
                skip_processed=True,
                continue_on_error=True
            )
            
            print("📊 Resumed Processing Results:")
            print(f"   Files Processed: {resumed_report.total_files_processed}")
            print(f"   Total Addresses: {resumed_report.total_addresses:,}")
            
        except Exception as e:
            print(f"ℹ️ No batch to resume (this is normal for first run): {e}")
        
        print()
        
        # Example 3: Processing specific files with custom settings
        print("🎯 Example 3: Custom Processing Settings")
        print("-" * 40)
        
        # Custom configuration for specific requirements
        custom_config = ProcessingConfiguration(
            gpu_batch_size=200,  # Smaller batches for memory-constrained systems
            target_throughput=1000,  # Lower throughput target
            cpu_allocation_ratio=0.05,  # 5% CPU allocation
            gpu_utilization_threshold=0.85  # 85% GPU utilization target
        )
        
        custom_processor = BatchFileProcessor(custom_config, "custom_batch_output")
        
        # Process only specific file patterns
        specific_patterns = ["export_customer_address_store_p1*.csv"]
        
        custom_report = custom_processor.process_files_batch(
            file_patterns=specific_patterns,
            skip_processed=False,  # Process all files (don't skip)
            continue_on_error=True
        )
        
        print("📊 Custom Processing Results:")
        print(f"   Files Processed: {custom_report.total_files_processed}")
        print(f"   Processing Efficiency: {custom_report.processing_efficiency:.1f}%")
        
        print()
        print("🎉 Batch processing examples completed successfully!")
        
        # Show output directory structure
        print("\n📁 Output Directory Structure:")
        output_path = Path(output_dir)
        if output_path.exists():
            for item in output_path.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(output_path)
                    print(f"   {relative_path}")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)