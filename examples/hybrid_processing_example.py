#!/usr/bin/env python3
"""Example usage of GPU-CPU Hybrid Processing System.

This example demonstrates how to set up and use the hybrid processor
for high-performance address processing with GPU acceleration.

Requirements: 1.1, 2.1, 4.1, 8.1
"""

import sys
import os
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration
from hybrid_config import load_hybrid_config
from hybrid_logging import setup_hybrid_logging


def main():
    """Main example function demonstrating hybrid processing setup."""
    
    # Set up logging
    logger = setup_hybrid_logging(
        log_level="INFO",
        enable_performance_logging=True
    )
    
    logger.info("Starting GPU-CPU Hybrid Processing Example")
    
    try:
        # Load configuration
        logger.info("Loading hybrid processing configuration...")
        config = load_hybrid_config(
            config_path="../config/config.yaml",
            hybrid_config_path="../config/hybrid_config.yaml"
        )
        
        logger.info(f"Configuration loaded successfully:")
        logger.info(f"  - GPU batch size: {config.processing_config.gpu_batch_size}")
        logger.info(f"  - CPU allocation: {config.processing_config.cpu_allocation_ratio:.1%}")
        logger.info(f"  - Target throughput: {config.processing_config.target_throughput} addr/sec")
        
        # Create hybrid processor
        logger.info("Initializing hybrid processor...")
        processor = GPUCPUHybridProcessor(config.processing_config)
        
        # Initialize processing components
        logger.info("Setting up processing components...")
        processor.initialize_hybrid_processing()
        
        # Example addresses for processing
        sample_addresses = [
            "123 Main Street, Apartment 4B, Near City Mall, Bangalore, Karnataka 560001",
            "Plot No 45, Prestige Lakeside, Whitefield Road, Bangalore 560066",
            "House No 67, Brigade Gateway, Rajaji Nagar, Bangalore, Karnataka 560010",
            "Flat 302, Sobha City, Thanisandra Main Road, Bangalore 560077",
            "Villa 12, Embassy Springs, Devanahalli, Bangalore, Karnataka 562110"
        ] * 100  # Repeat to simulate larger dataset
        
        logger.info(f"Processing {len(sample_addresses)} sample addresses...")
        
        # Process addresses using hybrid system
        result = processor.process_addresses_hybrid(sample_addresses)
        
        # Display results
        logger.info("Processing completed successfully!")
        logger.info(f"Results:")
        logger.info(f"  - Addresses processed: {len(result.parsed_addresses)}")
        logger.info(f"  - Processing time: {result.processing_time:.2f} seconds")
        logger.info(f"  - GPU processing time: {result.gpu_processing_time:.2f} seconds")
        logger.info(f"  - CPU processing time: {result.cpu_processing_time:.2f} seconds")
        
        if result.performance_metrics:
            metrics = result.performance_metrics
            logger.info(f"Performance Metrics:")
            logger.info(f"  - Throughput: {metrics.throughput_rate:.1f} addresses/second")
            logger.info(f"  - GPU utilization: {metrics.gpu_utilization:.1f}%")
            logger.info(f"  - Processing efficiency: {metrics.processing_efficiency:.1f}%")
            logger.info(f"  - GPU processed: {metrics.gpu_processed}")
            logger.info(f"  - CPU processed: {metrics.cpu_processed}")
        
        # Display device statistics
        if result.device_statistics:
            stats = result.device_statistics
            logger.info(f"Device Statistics:")
            logger.info(f"  - GPU allocation: {stats.get('gpu_allocation_ratio', 0):.1%}")
            logger.info(f"  - CPU allocation: {stats.get('cpu_allocation_ratio', 0):.1%}")
        
        # Display optimization suggestions
        if result.optimization_suggestions:
            logger.info("Optimization Suggestions:")
            for suggestion in result.optimization_suggestions:
                logger.info(f"  - {suggestion}")
        
        # Shutdown processor
        logger.info("Shutting down hybrid processor...")
        processor.shutdown()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()