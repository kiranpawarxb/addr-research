"""Hybrid Integration Adapter for GPU-CPU Hybrid Processing System.

This module provides integration between the existing address processing pipeline
and the new GPU-CPU hybrid processing system. It ensures compatibility with
current CSV input/output formats, ParsedAddress and AddressRecord models,
and provides seamless migration from existing processing scripts.

Requirements: 9.1, 9.2
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import csv

try:
    from .models import ParsedAddress, AddressRecord
    from .csv_reader import CSVReader
    from .output_writer import OutputWriter
    from .hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration, ProcessingResult
    from .consolidation_engine import ConsolidationEngine
    from .models import ConsolidatedGroup, ConsolidationStats
except ImportError:
    # Fallback for direct execution
    from models import ParsedAddress, AddressRecord
    from csv_reader import CSVReader
    from output_writer import OutputWriter
    from hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration, ProcessingResult
    from consolidation_engine import ConsolidationEngine
    from models import ConsolidatedGroup, ConsolidationStats


logger = logging.getLogger(__name__)


class HybridIntegrationAdapter:
    """Integration adapter for GPU-CPU hybrid processing with existing pipeline.
    
    Provides seamless integration between the existing address processing pipeline
    and the new GPU-CPU hybrid processing system. Maintains compatibility with
    current CSV formats, data models, and processing workflows.
    
    Requirements: 9.1, 9.2
    """
    
    def __init__(self, config: ProcessingConfiguration):
        """Initialize the integration adapter.
        
        Args:
            config: Processing configuration for hybrid processing
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize hybrid processor
        self.hybrid_processor = GPUCPUHybridProcessor(config)
        
        # Initialize consolidation engine for backward compatibility
        self.consolidation_engine = None
        
        # Track processing state
        self.is_initialized = False
        
        self.logger.info("Initialized HybridIntegrationAdapter")
    
    def initialize(self) -> None:
        """Initialize the hybrid processing system and consolidation engine."""
        if self.is_initialized:
            self.logger.warning("Integration adapter already initialized")
            return
        
        try:
            # Initialize hybrid processor
            self.hybrid_processor.initialize_hybrid_processing()
            
            # Initialize consolidation engine for backward compatibility
            self.consolidation_engine = ConsolidationEngine()
            
            self.is_initialized = True
            self.logger.info("✅ Hybrid integration adapter initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid integration adapter: {e}")
            raise RuntimeError(f"Integration adapter initialization failed: {e}")
    
    def process_csv_file(
        self,
        input_file: str,
        output_file: str,
        required_columns: Optional[List[str]] = None,
        consolidate_results: bool = True,
        comprehensive_output: bool = True
    ) -> Tuple[ProcessingResult, Optional[ConsolidationStats]]:
        """Process a CSV file using hybrid processing with full integration.
        
        Provides complete integration with existing CSV processing pipeline,
        including reading, hybrid processing, consolidation, and output generation.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            required_columns: Required CSV columns (uses defaults if None)
            consolidate_results: Whether to perform address consolidation
            comprehensive_output: Whether to generate comprehensive output
            
        Returns:
            Tuple of (ProcessingResult, ConsolidationStats or None)
            
        Requirements: 9.1, 9.2
        """
        if not self.is_initialized:
            raise RuntimeError("Integration adapter not initialized. Call initialize() first.")
        
        self.logger.info(f"🚀 Processing CSV file with hybrid integration: {input_file}")
        
        # Use default required columns if not specified
        if required_columns is None:
            required_columns = [
                'addr_hash_key', 'addr_text', 'city_id', 'pincode', 
                'state_id', 'zone_id', 'address_id', 
                'assigned_pickup_dlvd_geo_points', 'assigned_pickup_dlvd_geo_points_count'
            ]
        
        try:
            # Step 1: Read CSV file using existing CSV reader
            self.logger.info("📖 Reading CSV file...")
            csv_reader = CSVReader(input_file, required_columns)
            
            # Validate CSV structure
            is_valid, missing_columns = csv_reader.validate_columns()
            if not is_valid:
                raise ValueError(f"CSV validation failed. Missing columns: {missing_columns}")
            
            # Read all records
            address_records = list(csv_reader.read())
            self.logger.info(f"📊 Loaded {len(address_records)} address records from CSV")
            
            # Step 2: Extract address strings for hybrid processing
            address_strings = [record.addr_text for record in address_records]
            
            # Step 3: Process addresses using hybrid GPU-CPU processing
            self.logger.info("🔄 Processing addresses with hybrid GPU-CPU system...")
            processing_result = self.hybrid_processor.process_addresses_hybrid(address_strings)
            
            # Step 4: Combine original records with parsed results
            combined_records = self._combine_records_with_results(
                address_records, processing_result.parsed_addresses
            )
            
            # Step 5: Perform consolidation if requested (backward compatibility)
            consolidation_stats = None
            if consolidate_results and self.consolidation_engine:
                self.logger.info("🔗 Performing address consolidation...")
                consolidated_groups = self.consolidation_engine.consolidate_addresses(combined_records)
                consolidation_stats = self.consolidation_engine.calculate_stats(consolidated_groups)
                
                # Write consolidated output
                output_writer = OutputWriter(output_file, comprehensive_output)
                if comprehensive_output:
                    # Generate comprehensive output with all metadata
                    output_files = output_writer.write_comprehensive_output(
                        processing_result, address_records
                    )
                    self.logger.info(f"📁 Comprehensive output generated: {len(output_files)} files")
                else:
                    # Write standard consolidated CSV
                    records_written = output_writer.write(consolidated_groups)
                    self.logger.info(f"📄 Wrote {records_written} consolidated records to {output_file}")
            else:
                # Step 6: Write results without consolidation
                self.logger.info("📝 Writing results without consolidation...")
                self._write_hybrid_results_csv(
                    combined_records, output_file, processing_result, comprehensive_output
                )
            
            # Log completion summary
            success_count = sum(1 for addr in processing_result.parsed_addresses if addr.parse_success)
            self.logger.info(f"✅ CSV processing completed: {success_count}/{len(address_strings)} addresses parsed successfully")
            
            if processing_result.performance_metrics:
                self.logger.info(f"📈 Performance: {processing_result.performance_metrics.throughput_rate:.1f} addr/sec, "
                               f"GPU: {processing_result.performance_metrics.gpu_utilization:.1f}%")
            
            return processing_result, consolidation_stats
            
        except Exception as e:
            self.logger.error(f"CSV file processing failed: {e}")
            raise
    
    def process_address_list(
        self,
        addresses: List[str],
        original_records: Optional[List[AddressRecord]] = None
    ) -> ProcessingResult:
        """Process a list of address strings using hybrid processing.
        
        Provides direct integration for processing address lists while maintaining
        compatibility with existing data models and processing patterns.
        
        Args:
            addresses: List of address strings to process
            original_records: Optional original AddressRecord objects for metadata
            
        Returns:
            ProcessingResult with parsed addresses and performance data
            
        Requirements: 9.1, 9.2
        """
        if not self.is_initialized:
            raise RuntimeError("Integration adapter not initialized. Call initialize() first.")
        
        if not addresses:
            return ProcessingResult()
        
        self.logger.info(f"🔄 Processing {len(addresses)} addresses with hybrid integration")
        
        try:
            # Process addresses using hybrid GPU-CPU processing
            processing_result = self.hybrid_processor.process_addresses_hybrid(addresses)
            
            # Add original record metadata if provided
            if original_records and len(original_records) == len(addresses):
                self._enrich_results_with_metadata(processing_result, original_records)
            
            # Log processing summary
            success_count = sum(1 for addr in processing_result.parsed_addresses if addr.parse_success)
            self.logger.info(f"✅ Address processing completed: {success_count}/{len(addresses)} addresses parsed successfully")
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Address list processing failed: {e}")
            raise
    
    def create_consolidated_groups(
        self,
        combined_records: List[Tuple[AddressRecord, ParsedAddress]]
    ) -> List[ConsolidatedGroup]:
        """Create consolidated groups from combined records for backward compatibility.
        
        Args:
            combined_records: List of (AddressRecord, ParsedAddress) tuples
            
        Returns:
            List of ConsolidatedGroup objects
        """
        if not self.consolidation_engine:
            raise RuntimeError("Consolidation engine not initialized")
        
        return self.consolidation_engine.consolidate_addresses(combined_records)
    
    def get_processing_statistics(self, processing_result: ProcessingResult) -> Dict[str, Any]:
        """Get comprehensive processing statistics for integration reporting.
        
        Args:
            processing_result: Result from hybrid processing
            
        Returns:
            Dictionary with comprehensive processing statistics
            
        Requirements: 9.3, 9.4
        """
        stats = {
            "total_addresses": len(processing_result.parsed_addresses),
            "successfully_parsed": sum(1 for addr in processing_result.parsed_addresses if addr.parse_success),
            "failed_to_parse": sum(1 for addr in processing_result.parsed_addresses if not addr.parse_success),
            "processing_time": processing_result.processing_time,
            "gpu_processing_time": processing_result.gpu_processing_time,
            "cpu_processing_time": processing_result.cpu_processing_time,
            "error_count": processing_result.error_count,
            "optimization_suggestions": processing_result.optimization_suggestions
        }
        
        # Add performance metrics if available
        if processing_result.performance_metrics:
            stats.update({
                "throughput_rate": processing_result.performance_metrics.throughput_rate,
                "gpu_utilization": processing_result.performance_metrics.gpu_utilization,
                "cpu_utilization": processing_result.performance_metrics.cpu_utilization,
                "processing_efficiency": processing_result.performance_metrics.processing_efficiency,
                "gpu_processed": processing_result.performance_metrics.gpu_processed,
                "cpu_processed": processing_result.performance_metrics.cpu_processed
            })
        
        # Add device statistics
        stats.update(processing_result.device_statistics)
        
        # Calculate success rate
        if stats["total_addresses"] > 0:
            stats["success_rate"] = (stats["successfully_parsed"] / stats["total_addresses"]) * 100.0
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def shutdown(self) -> None:
        """Gracefully shutdown the integration adapter and cleanup resources."""
        self.logger.info("Shutting down hybrid integration adapter...")
        
        if self.hybrid_processor:
            self.hybrid_processor.shutdown()
        
        self.is_initialized = False
        self.logger.info("Hybrid integration adapter shutdown completed")
    
    # Private helper methods
    
    def _combine_records_with_results(
        self,
        address_records: List[AddressRecord],
        parsed_addresses: List[ParsedAddress]
    ) -> List[Tuple[AddressRecord, ParsedAddress]]:
        """Combine original address records with parsed results.
        
        Args:
            address_records: Original AddressRecord objects
            parsed_addresses: Parsed address results from hybrid processing
            
        Returns:
            List of (AddressRecord, ParsedAddress) tuples
        """
        if len(address_records) != len(parsed_addresses):
            self.logger.warning(
                f"Record count mismatch: {len(address_records)} records vs "
                f"{len(parsed_addresses)} results. Padding with empty results."
            )
            
            # Pad with empty results if needed
            while len(parsed_addresses) < len(address_records):
                parsed_addresses.append(ParsedAddress(
                    parse_success=False,
                    parse_error="No result from hybrid processing"
                ))
        
        return list(zip(address_records, parsed_addresses))
    
    def _enrich_results_with_metadata(
        self,
        processing_result: ProcessingResult,
        original_records: List[AddressRecord]
    ) -> None:
        """Enrich processing results with metadata from original records.
        
        Args:
            processing_result: ProcessingResult to enrich
            original_records: Original AddressRecord objects with metadata
        """
        # Add original record information to device statistics
        processing_result.device_statistics.update({
            "original_records_count": len(original_records),
            "input_source": "AddressRecord_objects",
            "has_original_metadata": True
        })
        
        # Extract unique cities and pincodes for analysis
        unique_cities = set(record.city_id for record in original_records if record.city_id)
        unique_pincodes = set(record.pincode for record in original_records if record.pincode)
        
        processing_result.device_statistics.update({
            "unique_cities": len(unique_cities),
            "unique_pincodes": len(unique_pincodes),
            "geographic_diversity": len(unique_cities) + len(unique_pincodes)
        })
    
    def _write_hybrid_results_csv(
        self,
        combined_records: List[Tuple[AddressRecord, ParsedAddress]],
        output_file: str,
        processing_result: ProcessingResult,
        comprehensive_output: bool
    ) -> None:
        """Write hybrid processing results to CSV format.
        
        Args:
            combined_records: Combined address records and parsed results
            output_file: Output CSV file path
            processing_result: Processing result with performance data
            comprehensive_output: Whether to generate comprehensive output
        """
        if comprehensive_output:
            # Use comprehensive output generator
            output_writer = OutputWriter(output_file, comprehensive_output=True)
            original_records = [record for record, _ in combined_records]
            output_files = output_writer.write_comprehensive_output(
                processing_result, original_records
            )
            self.logger.info(f"📁 Comprehensive output generated: {len(output_files)} files")
        else:
            # Write standard CSV with hybrid results
            self._write_standard_hybrid_csv(combined_records, output_file, processing_result)
    
    def _write_standard_hybrid_csv(
        self,
        combined_records: List[Tuple[AddressRecord, ParsedAddress]],
        output_file: str,
        processing_result: ProcessingResult
    ) -> None:
        """Write standard CSV output with hybrid processing results.
        
        Args:
            combined_records: Combined address records and parsed results
            output_file: Output CSV file path
            processing_result: Processing result with performance data
        """
        if not combined_records:
            self.logger.warning("No records to write to CSV")
            return
        
        # Prepare output directory
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define output columns
        fieldnames = [
            # Original columns
            'addr_hash_key', 'addr_text', 'city_id', 'pincode', 'state_id', 'zone_id',
            'address_id', 'assigned_pickup_dlvd_geo_points', 'assigned_pickup_dlvd_geo_points_count',
            # Parsed address fields
            'UN', 'SN', 'LN', 'RD', 'SL', 'LOC', 'CY', 'DIS', 'ST', 'CN', 'PIN', 'Note',
            # Processing metadata
            'parse_success', 'parse_error', 'processing_method', 'processing_time'
        ]
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for record, parsed in combined_records:
                    row = {
                        # Original fields
                        'addr_hash_key': record.addr_hash_key,
                        'addr_text': record.addr_text,
                        'city_id': record.city_id,
                        'pincode': record.pincode,
                        'state_id': record.state_id,
                        'zone_id': record.zone_id,
                        'address_id': record.address_id,
                        'assigned_pickup_dlvd_geo_points': record.assigned_pickup_dlvd_geo_points,
                        'assigned_pickup_dlvd_geo_points_count': record.assigned_pickup_dlvd_geo_points_count,
                        
                        # Parsed fields
                        'UN': parsed.unit_number,
                        'SN': parsed.society_name,
                        'LN': parsed.landmark,
                        'RD': parsed.road,
                        'SL': parsed.sub_locality,
                        'LOC': parsed.locality,
                        'CY': parsed.city,
                        'DIS': parsed.district,
                        'ST': parsed.state,
                        'CN': parsed.country,
                        'PIN': parsed.pin_code,
                        'Note': parsed.note,
                        
                        # Processing metadata
                        'parse_success': parsed.parse_success,
                        'parse_error': parsed.parse_error or '',
                        'processing_method': 'hybrid_gpu_cpu',
                        'processing_time': processing_result.processing_time
                    }
                    
                    writer.writerow(row)
            
            self.logger.info(f"📄 Wrote {len(combined_records)} records to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to write CSV output: {e}")
            raise


def create_hybrid_integration_adapter(
    gpu_batch_size: int = 400,
    target_throughput: int = 2000,
    gpu_memory_fraction: float = 0.95,
    cpu_allocation_ratio: float = 0.02
) -> HybridIntegrationAdapter:
    """Create a hybrid integration adapter with default configuration.
    
    Convenience function for creating an integration adapter with commonly used
    configuration parameters for seamless integration with existing workflows.
    
    Args:
        gpu_batch_size: GPU batch size for processing (default: 400)
        target_throughput: Target processing throughput (default: 2000)
        gpu_memory_fraction: GPU memory allocation fraction (default: 0.95)
        cpu_allocation_ratio: CPU allocation ratio (default: 0.02)
        
    Returns:
        Configured HybridIntegrationAdapter instance
    """
    config = ProcessingConfiguration(
        gpu_batch_size=gpu_batch_size,
        target_throughput=target_throughput,
        gpu_memory_fraction=gpu_memory_fraction,
        cpu_allocation_ratio=cpu_allocation_ratio
    )
    
    return HybridIntegrationAdapter(config)