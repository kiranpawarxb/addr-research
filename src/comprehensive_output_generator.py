"""Comprehensive Output Generation and Reporting for GPU-CPU Hybrid Processing.

This module implements comprehensive output generation with all parsed address fields,
processing metadata, performance metrics, batch processing reports, and detailed
error information as specified in Requirements 9.1, 9.2, 9.3, 9.4, 9.5.
"""

import logging
import os
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

try:
    from .models import ParsedAddress, AddressRecord
except ImportError:
    # Fallback for direct execution
    from models import ParsedAddress, AddressRecord

# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from .hybrid_processor import ProcessingResult, BatchProcessingReport, PerformanceMetrics
        from .performance_monitor import PerformanceReport, GPUStats
    except ImportError:
        from hybrid_processor import ProcessingResult, BatchProcessingReport, PerformanceMetrics
        from performance_monitor import PerformanceReport, GPUStats


@dataclass
class OutputMetadata:
    """Comprehensive metadata for output generation.
    
    Contains all processing metadata including timestamps, device information,
    performance metrics, and error details for comprehensive output generation.
    
    Requirements: 9.1, 9.2
    """
    # Processing Information
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_start_time: float = field(default_factory=time.time)
    processing_end_time: Optional[float] = None
    total_processing_time: float = 0.0
    
    # Device Information
    gpu_device_name: str = ""
    gpu_device_id: int = 0
    cpu_cores_used: int = 0
    system_memory_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    
    # Processing Configuration
    gpu_batch_size: int = 0
    cpu_batch_size: int = 0
    gpu_allocation_ratio: float = 0.0
    cpu_allocation_ratio: float = 0.0
    target_throughput: int = 0
    
    # Performance Metrics
    actual_throughput: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0
    processing_efficiency: float = 0.0
    success_rate: float = 0.0
    
    # Processing Statistics
    total_addresses: int = 0
    successfully_parsed: int = 0
    failed_to_parse: int = 0
    gpu_processed: int = 0
    cpu_processed: int = 0
    
    # Error Information
    error_count: int = 0
    error_details: List[str] = field(default_factory=list)
    warning_count: int = 0
    warning_details: List[str] = field(default_factory=list)
    
    # Optimization Information
    optimization_suggestions: List[str] = field(default_factory=list)
    performance_warnings: List[str] = field(default_factory=list)
    
    def finalize_processing(self, end_time: Optional[float] = None) -> None:
        """Finalize processing metadata with end time and calculations."""
        self.processing_end_time = end_time or time.time()
        self.total_processing_time = self.processing_end_time - self.processing_start_time
        
        # Calculate success rate
        if self.total_addresses > 0:
            self.success_rate = (self.successfully_parsed / self.total_addresses) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ComprehensiveOutputConfig:
    """Configuration for comprehensive output generation.
    
    Controls output format, metadata inclusion, and reporting options
    for comprehensive output generation.
    """
    # Output Format Options
    include_metadata_header: bool = True
    include_performance_summary: bool = True
    include_device_information: bool = True
    include_error_details: bool = True
    include_optimization_suggestions: bool = True
    
    # File Output Options
    generate_json_metadata: bool = True
    generate_performance_report: bool = True
    generate_error_report: bool = True
    
    # CSV Output Options
    include_processing_timestamps: bool = True
    include_device_columns: bool = True
    include_performance_columns: bool = True
    
    # Batch Processing Options
    generate_batch_summary: bool = True
    generate_comparative_analysis: bool = True
    include_per_file_metrics: bool = True


class ComprehensiveOutputGenerator:
    """Comprehensive output generation with all parsed address fields and metadata.
    
    Implements comprehensive output generation including all parsed address fields,
    processing timestamps, device information, performance metrics, batch processing
    reports, and detailed error information.
    
    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    
    def __init__(self, config: ComprehensiveOutputConfig = None):
        """Initialize comprehensive output generator.
        
        Args:
            config: Configuration for output generation options
        """
        self.config = config or ComprehensiveOutputConfig()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initialized ComprehensiveOutputGenerator")
    
    def generate_comprehensive_output(
        self,
        parsed_addresses: List[ParsedAddress],
        original_records: List[AddressRecord],
        processing_result: 'ProcessingResult',
        output_path: str,
        metadata: Optional[OutputMetadata] = None
    ) -> Dict[str, str]:
        """Generate comprehensive output with all parsed address fields and metadata.
        
        Creates comprehensive output files including CSV with all parsed fields,
        JSON metadata, performance reports, and error analysis.
        
        Args:
            parsed_addresses: List of parsed address results
            original_records: Original address records from input
            processing_result: Complete processing result with metrics
            output_path: Base path for output files
            metadata: Optional processing metadata
            
        Returns:
            Dictionary mapping output type to file path
            
        Requirements: 9.1, 9.2
        """
        self.logger.info(f"Generating comprehensive output for {len(parsed_addresses)} addresses")
        
        # Create output metadata if not provided
        if metadata is None:
            metadata = self._create_output_metadata(processing_result, len(parsed_addresses))
        
        # Finalize metadata
        metadata.finalize_processing()
        
        output_files = {}
        
        try:
            # Generate main CSV output with all fields and metadata
            csv_path = self._generate_comprehensive_csv(
                parsed_addresses, original_records, processing_result, output_path, metadata
            )
            output_files['csv'] = csv_path
            
            # Generate JSON metadata file
            if self.config.generate_json_metadata:
                json_path = self._generate_json_metadata(metadata, output_path)
                output_files['metadata'] = json_path
            
            # Generate performance report
            if self.config.generate_performance_report and processing_result.performance_metrics:
                perf_path = self._generate_performance_report(processing_result, output_path, metadata)
                output_files['performance'] = perf_path
            
            # Generate error report if errors occurred
            if self.config.generate_error_report and (metadata.error_count > 0 or processing_result.error_count > 0):
                error_path = self._generate_error_report(processing_result, output_path, metadata)
                output_files['errors'] = error_path
            
            self.logger.info(f"✅ Comprehensive output generated: {len(output_files)} files created")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive output: {e}")
            raise
    
    def generate_batch_processing_report(
        self,
        batch_report: 'BatchProcessingReport',
        output_dir: str,
        batch_id: str = None
    ) -> Dict[str, str]:
        """Generate comprehensive batch processing reports with comparative analysis.
        
        Creates detailed batch processing reports including per-file results,
        comparative performance analysis, and optimization recommendations.
        
        Args:
            batch_report: Batch processing report with file results
            output_dir: Directory for batch report outputs
            batch_id: Optional batch identifier
            
        Returns:
            Dictionary mapping report type to file path
            
        Requirements: 9.3, 9.4
        """
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Generating batch processing report: {batch_id}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        report_files = {}
        
        try:
            # Generate batch summary report
            if self.config.generate_batch_summary:
                summary_path = self._generate_batch_summary_report(batch_report, output_dir, batch_id)
                report_files['summary'] = summary_path
            
            # Generate comparative performance analysis
            if self.config.generate_comparative_analysis:
                analysis_path = self._generate_comparative_analysis(batch_report, output_dir, batch_id)
                report_files['analysis'] = analysis_path
            
            # Generate per-file metrics report
            if self.config.include_per_file_metrics:
                metrics_path = self._generate_per_file_metrics(batch_report, output_dir, batch_id)
                report_files['metrics'] = metrics_path
            
            # Generate batch performance JSON
            json_path = self._generate_batch_json_report(batch_report, output_dir, batch_id)
            report_files['json'] = json_path
            
            self.logger.info(f"✅ Batch processing report generated: {len(report_files)} files created")
            
            return report_files
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch processing report: {e}")
            raise
    
    # Private helper methods for comprehensive output generation
    
    def _create_output_metadata(self, processing_result: 'ProcessingResult', total_addresses: int) -> OutputMetadata:
        """Create output metadata from processing result."""
        metadata = OutputMetadata()
        
        # Basic processing information
        metadata.total_addresses = total_addresses
        metadata.successfully_parsed = len(processing_result.parsed_addresses)
        metadata.failed_to_parse = processing_result.error_count
        metadata.error_count = processing_result.error_count
        metadata.error_details = processing_result.error_details.copy()
        metadata.optimization_suggestions = processing_result.optimization_suggestions.copy()
        
        # Performance metrics
        if processing_result.performance_metrics:
            perf = processing_result.performance_metrics
            metadata.actual_throughput = perf.throughput_rate
            metadata.gpu_utilization_percent = perf.gpu_utilization
            metadata.cpu_utilization_percent = perf.cpu_utilization
            metadata.processing_efficiency = perf.processing_efficiency
            metadata.gpu_processed = perf.gpu_processed
            metadata.cpu_processed = perf.cpu_processed
        
        # Device statistics
        if processing_result.device_statistics:
            stats = processing_result.device_statistics
            metadata.gpu_allocation_ratio = stats.get('gpu_allocation_ratio', 0.0)
            metadata.cpu_allocation_ratio = stats.get('cpu_allocation_ratio', 0.0)
        
        # Processing times
        metadata.total_processing_time = processing_result.processing_time
        
        return metadata
    
    def _generate_comprehensive_csv(
        self,
        parsed_addresses: List[ParsedAddress],
        original_records: List[AddressRecord],
        processing_result: 'ProcessingResult',
        output_path: str,
        metadata: OutputMetadata
    ) -> str:
        """Generate comprehensive CSV with all parsed fields and metadata."""
        # Create CSV filename
        base_path = Path(output_path)
        csv_path = base_path.with_suffix('.csv')
        
        self.logger.info(f"Generating comprehensive CSV: {csv_path}")
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Write metadata header if configured
                if self.config.include_metadata_header:
                    self._write_csv_metadata_header(csvfile, metadata)
                
                # Prepare CSV data
                rows = self._prepare_csv_rows(parsed_addresses, original_records, metadata)
                
                if rows:
                    # Get fieldnames from first row
                    fieldnames = list(rows[0].keys())
                    
                    # Write CSV data
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                    
                    self.logger.info(f"✅ CSV generated: {len(rows)} rows written")
                else:
                    self.logger.warning("No data rows to write to CSV")
            
            return str(csv_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive CSV: {e}")
            raise
    
    def _write_csv_metadata_header(self, csvfile, metadata: OutputMetadata) -> None:
        """Write comprehensive metadata header to CSV file."""
        csvfile.write("# COMPREHENSIVE PROCESSING METADATA\n")
        csvfile.write(f"# Processing Timestamp: {metadata.processing_timestamp}\n")
        csvfile.write(f"# Total Processing Time: {metadata.total_processing_time:.2f} seconds\n")
        csvfile.write(f"# Total Addresses: {metadata.total_addresses:,}\n")
        csvfile.write(f"# Successfully Parsed: {metadata.successfully_parsed:,}\n")
        csvfile.write(f"# Failed to Parse: {metadata.failed_to_parse:,}\n")
        csvfile.write(f"# Success Rate: {metadata.success_rate:.1f}%\n")
        
        # Performance metrics
        csvfile.write(f"# Actual Throughput: {metadata.actual_throughput:.1f} addresses/second\n")
        csvfile.write(f"# GPU Utilization: {metadata.gpu_utilization_percent:.1f}%\n")
        csvfile.write(f"# Processing Efficiency: {metadata.processing_efficiency:.1f}%\n")
        
        # Device information
        if self.config.include_device_information:
            csvfile.write(f"# GPU Device: {metadata.gpu_device_name}\n")
            csvfile.write(f"# GPU Memory Used: {metadata.gpu_memory_used_gb:.1f} GB\n")
            csvfile.write(f"# CPU Cores Used: {metadata.cpu_cores_used}\n")
        
        # Processing allocation
        csvfile.write(f"# GPU Processed: {metadata.gpu_processed:,} ({metadata.gpu_allocation_ratio*100:.1f}%)\n")
        csvfile.write(f"# CPU Processed: {metadata.cpu_processed:,} ({metadata.cpu_allocation_ratio*100:.1f}%)\n")
        
        # Error information
        if metadata.error_count > 0:
            csvfile.write(f"# Error Count: {metadata.error_count}\n")
            if metadata.error_details:
                csvfile.write("# Error Details:\n")
                for error in metadata.error_details[:5]:  # Limit to first 5 errors
                    csvfile.write(f"#   - {error}\n")
        
        # Optimization suggestions
        if metadata.optimization_suggestions:
            csvfile.write("# Optimization Suggestions:\n")
            for suggestion in metadata.optimization_suggestions[:3]:  # Limit to first 3
                csvfile.write(f"#   - {suggestion}\n")
        
        csvfile.write("#\n")
        csvfile.write("# CSV DATA BEGINS BELOW\n")
        csvfile.write("#\n")
    
    def _prepare_csv_rows(
        self,
        parsed_addresses: List[ParsedAddress],
        original_records: List[AddressRecord],
        metadata: OutputMetadata
    ) -> List[Dict[str, Any]]:
        """Prepare CSV rows with all parsed fields and processing metadata."""
        rows = []
        
        # Ensure we have matching counts
        min_count = min(len(parsed_addresses), len(original_records))
        
        for i in range(min_count):
            parsed = parsed_addresses[i]
            original = original_records[i]
            
            # Start with original record data
            row = dict(original.raw_data) if original.raw_data else {}
            
            # Add standard original fields
            row.update({
                'addr_hash_key': original.addr_hash_key,
                'addr_text': original.addr_text,
                'city_id': original.city_id,
                'pincode': original.pincode,
                'state_id': original.state_id,
                'zone_id': original.zone_id,
                'address_id': original.address_id,
                'assigned_pickup_dlvd_geo_points': original.assigned_pickup_dlvd_geo_points,
                'assigned_pickup_dlvd_geo_points_count': original.assigned_pickup_dlvd_geo_points_count,
            })
            
            # Add all parsed address fields
            row.update({
                'parsed_unit_number': parsed.unit_number,
                'parsed_society_name': parsed.society_name,
                'parsed_landmark': parsed.landmark,
                'parsed_road': parsed.road,
                'parsed_sub_locality': parsed.sub_locality,
                'parsed_locality': parsed.locality,
                'parsed_city': parsed.city,
                'parsed_district': parsed.district,
                'parsed_state': parsed.state,
                'parsed_country': parsed.country,
                'parsed_pin_code': parsed.pin_code,
                'parsed_note': parsed.note,
                'parse_success': parsed.parse_success,
                'parse_error': parsed.parse_error or "",
            })
            
            # Add processing metadata columns if configured
            if self.config.include_processing_timestamps:
                row.update({
                    'processing_timestamp': metadata.processing_timestamp,
                    'processing_time_seconds': metadata.total_processing_time,
                })
            
            if self.config.include_device_columns:
                row.update({
                    'processed_by_device': 'GPU' if i < metadata.gpu_processed else 'CPU',
                    'gpu_device_name': metadata.gpu_device_name,
                })
            
            if self.config.include_performance_columns:
                row.update({
                    'throughput_rate': metadata.actual_throughput,
                    'gpu_utilization_percent': metadata.gpu_utilization_percent,
                    'processing_efficiency': metadata.processing_efficiency,
                })
            
            rows.append(row)
        
        return rows
    
    def _generate_json_metadata(self, metadata: OutputMetadata, output_path: str) -> str:
        """Generate comprehensive JSON metadata file."""
        json_path = Path(output_path).with_suffix('.metadata.json')
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"✅ JSON metadata generated: {json_path}")
            return str(json_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON metadata: {e}")
            raise
    
    def _generate_performance_report(
        self,
        processing_result: 'ProcessingResult',
        output_path: str,
        metadata: OutputMetadata
    ) -> str:
        """Generate detailed performance report."""
        perf_path = Path(output_path).with_suffix('.performance.txt')
        
        try:
            with open(perf_path, 'w', encoding='utf-8') as f:
                f.write("COMPREHENSIVE PERFORMANCE REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Processing Summary
                f.write("PROCESSING SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Addresses: {metadata.total_addresses:,}\n")
                f.write(f"Successfully Parsed: {metadata.successfully_parsed:,}\n")
                f.write(f"Failed to Parse: {metadata.failed_to_parse:,}\n")
                f.write(f"Success Rate: {metadata.success_rate:.1f}%\n")
                f.write(f"Total Processing Time: {metadata.total_processing_time:.2f} seconds\n\n")
                
                # Performance Metrics
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Throughput Rate: {metadata.actual_throughput:.1f} addresses/second\n")
                f.write(f"Target Throughput: {metadata.target_throughput} addresses/second\n")
                f.write(f"Throughput Achievement: {(metadata.actual_throughput/max(metadata.target_throughput,1))*100:.1f}%\n")
                f.write(f"GPU Utilization: {metadata.gpu_utilization_percent:.1f}%\n")
                f.write(f"CPU Utilization: {metadata.cpu_utilization_percent:.1f}%\n")
                f.write(f"Processing Efficiency: {metadata.processing_efficiency:.1f}%\n\n")
                
                # Device Information
                f.write("DEVICE INFORMATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"GPU Device: {metadata.gpu_device_name}\n")
                f.write(f"GPU Memory Used: {metadata.gpu_memory_used_gb:.1f} GB\n")
                f.write(f"CPU Cores Used: {metadata.cpu_cores_used}\n")
                f.write(f"System Memory: {metadata.system_memory_gb:.1f} GB\n\n")
                
                # Processing Distribution
                f.write("PROCESSING DISTRIBUTION\n")
                f.write("-" * 25 + "\n")
                f.write(f"GPU Processed: {metadata.gpu_processed:,} ({metadata.gpu_allocation_ratio*100:.1f}%)\n")
                f.write(f"CPU Processed: {metadata.cpu_processed:,} ({metadata.cpu_allocation_ratio*100:.1f}%)\n")
                f.write(f"GPU Batch Size: {metadata.gpu_batch_size}\n")
                f.write(f"CPU Batch Size: {metadata.cpu_batch_size}\n\n")
                
                # Optimization Suggestions
                if metadata.optimization_suggestions:
                    f.write("OPTIMIZATION SUGGESTIONS\n")
                    f.write("-" * 25 + "\n")
                    for i, suggestion in enumerate(metadata.optimization_suggestions, 1):
                        f.write(f"{i}. {suggestion}\n")
                    f.write("\n")
                
                # Performance Warnings
                if metadata.performance_warnings:
                    f.write("PERFORMANCE WARNINGS\n")
                    f.write("-" * 22 + "\n")
                    for warning in metadata.performance_warnings:
                        f.write(f"⚠️  {warning}\n")
            
            self.logger.info(f"✅ Performance report generated: {perf_path}")
            return str(perf_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            raise
    
    def _generate_error_report(
        self,
        processing_result: 'ProcessingResult',
        output_path: str,
        metadata: OutputMetadata
    ) -> str:
        """Generate detailed error report with processing context."""
        error_path = Path(output_path).with_suffix('.errors.txt')
        
        try:
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write("COMPREHENSIVE ERROR REPORT\n")
                f.write("=" * 40 + "\n\n")
                
                # Error Summary
                f.write("ERROR SUMMARY\n")
                f.write("-" * 15 + "\n")
                f.write(f"Total Errors: {metadata.error_count}\n")
                f.write(f"Error Rate: {(metadata.error_count/max(metadata.total_addresses,1))*100:.1f}%\n")
                f.write(f"Processing Context: GPU-CPU Hybrid Processing\n")
                f.write(f"Processing Timestamp: {metadata.processing_timestamp}\n\n")
                
                # Detailed Error Information
                if metadata.error_details:
                    f.write("DETAILED ERROR INFORMATION\n")
                    f.write("-" * 30 + "\n")
                    for i, error in enumerate(metadata.error_details, 1):
                        f.write(f"Error {i}: {error}\n")
                    f.write("\n")
                
                # Processing Result Errors
                if processing_result.error_details:
                    f.write("PROCESSING RESULT ERRORS\n")
                    f.write("-" * 26 + "\n")
                    for i, error in enumerate(processing_result.error_details, 1):
                        f.write(f"Processing Error {i}: {error}\n")
                    f.write("\n")
                
                # Error Analysis and Recommendations
                f.write("ERROR ANALYSIS AND RECOMMENDATIONS\n")
                f.write("-" * 38 + "\n")
                
                if metadata.error_count > metadata.total_addresses * 0.1:
                    f.write("⚠️  High error rate detected (>10%)\n")
                    f.write("   Recommendations:\n")
                    f.write("   - Check input data quality\n")
                    f.write("   - Verify model configuration\n")
                    f.write("   - Consider adjusting batch sizes\n\n")
                
                if metadata.gpu_utilization_percent < 50:
                    f.write("⚠️  Low GPU utilization during processing\n")
                    f.write("   This may indicate GPU processing errors\n")
                    f.write("   Recommendations:\n")
                    f.write("   - Check GPU memory availability\n")
                    f.write("   - Verify CUDA installation\n")
                    f.write("   - Consider CPU-only processing\n\n")
            
            self.logger.info(f"✅ Error report generated: {error_path}")
            return str(error_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate error report: {e}")
            raise
    
    def _generate_batch_summary_report(
        self,
        batch_report: 'BatchProcessingReport',
        output_dir: Path,
        batch_id: str
    ) -> str:
        """Generate comprehensive batch summary report."""
        summary_path = output_dir / f"{batch_id}_batch_summary.txt"
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("COMPREHENSIVE BATCH PROCESSING SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                # Batch Overview
                f.write("BATCH OVERVIEW\n")
                f.write("-" * 15 + "\n")
                f.write(f"Batch ID: {batch_id}\n")
                f.write(f"Processing Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Total Files Processed: {batch_report.total_files_processed}\n")
                f.write(f"Total Addresses: {batch_report.total_addresses:,}\n\n")
                
                # Performance Summary
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Throughput: {batch_report.average_throughput:.1f} addresses/second\n")
                f.write(f"Peak Throughput: {batch_report.peak_throughput:.1f} addresses/second\n")
                f.write(f"Average GPU Utilization: {batch_report.average_gpu_utilization:.1f}%\n")
                f.write(f"Processing Efficiency: {batch_report.processing_efficiency:.1f}%\n\n")
                
                # File Processing Results
                if batch_report.file_results:
                    f.write("FILE PROCESSING RESULTS\n")
                    f.write("-" * 25 + "\n")
                    
                    total_time = sum(r.processing_time for r in batch_report.file_results)
                    total_errors = sum(r.error_count for r in batch_report.file_results)
                    
                    f.write(f"Total Processing Time: {total_time:.2f} seconds\n")
                    f.write(f"Total Errors: {total_errors}\n")
                    f.write(f"Average Processing Time per File: {total_time/len(batch_report.file_results):.2f} seconds\n")
                    f.write(f"Success Rate: {((batch_report.total_addresses - total_errors) / max(batch_report.total_addresses, 1)) * 100:.2f}%\n\n")
                
                # Performance Analysis
                f.write("PERFORMANCE ANALYSIS\n")
                f.write("-" * 21 + "\n")
                
                if batch_report.average_throughput >= 1500:
                    f.write("✅ Throughput target achieved (≥1500 addr/sec)\n")
                else:
                    f.write("❌ Throughput below target (<1500 addr/sec)\n")
                    f.write(f"   Gap: {1500 - batch_report.average_throughput:.1f} addr/sec\n")
                
                if batch_report.average_gpu_utilization >= 90:
                    f.write("✅ GPU utilization target achieved (≥90%)\n")
                else:
                    f.write("❌ GPU utilization below target (<90%)\n")
                    f.write(f"   Gap: {90 - batch_report.average_gpu_utilization:.1f}%\n")
                
                f.write(f"\nOverall Performance Score: {self._calculate_batch_performance_score(batch_report):.1f}/100\n\n")
                
                # Recommendations
                f.write("OPTIMIZATION RECOMMENDATIONS\n")
                f.write("-" * 30 + "\n")
                recommendations = self._generate_batch_recommendations(batch_report)
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            
            self.logger.info(f"✅ Batch summary report generated: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch summary report: {e}")
            raise
    
    def _generate_comparative_analysis(
        self,
        batch_report: 'BatchProcessingReport',
        output_dir: Path,
        batch_id: str
    ) -> str:
        """Generate comparative performance analysis across files."""
        analysis_path = output_dir / f"{batch_id}_comparative_analysis.csv"
        
        try:
            with open(analysis_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'file_index', 'addresses_processed', 'processing_time', 'throughput_rate',
                    'gpu_utilization', 'cpu_utilization', 'success_rate', 'error_count',
                    'gpu_processing_time', 'cpu_processing_time', 'efficiency_score',
                    'performance_relative_to_average', 'optimization_needed'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Calculate averages for comparison
                avg_throughput = batch_report.average_throughput
                avg_gpu_util = batch_report.average_gpu_utilization
                
                for i, result in enumerate(batch_report.file_results):
                    # Calculate metrics
                    addresses = len(result.parsed_addresses)
                    throughput = addresses / result.processing_time if result.processing_time > 0 else 0
                    success_rate = result.calculate_success_rate()
                    
                    # Performance metrics
                    gpu_util = result.performance_metrics.gpu_utilization if result.performance_metrics else 0
                    cpu_util = result.performance_metrics.cpu_utilization if result.performance_metrics else 0
                    
                    # Relative performance
                    relative_perf = (throughput / max(avg_throughput, 1)) * 100
                    
                    # Efficiency score
                    efficiency = self._calculate_file_efficiency_score(result, throughput, gpu_util)
                    
                    # Optimization needed
                    needs_optimization = throughput < avg_throughput * 0.8 or gpu_util < 70
                    
                    row = {
                        'file_index': i + 1,
                        'addresses_processed': addresses,
                        'processing_time': result.processing_time,
                        'throughput_rate': throughput,
                        'gpu_utilization': gpu_util,
                        'cpu_utilization': cpu_util,
                        'success_rate': success_rate,
                        'error_count': result.error_count,
                        'gpu_processing_time': result.gpu_processing_time,
                        'cpu_processing_time': result.cpu_processing_time,
                        'efficiency_score': efficiency,
                        'performance_relative_to_average': relative_perf,
                        'optimization_needed': needs_optimization
                    }
                    
                    writer.writerow(row)
            
            self.logger.info(f"✅ Comparative analysis generated: {analysis_path}")
            return str(analysis_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparative analysis: {e}")
            raise
    
    def _generate_per_file_metrics(
        self,
        batch_report: 'BatchProcessingReport',
        output_dir: Path,
        batch_id: str
    ) -> str:
        """Generate detailed per-file metrics report."""
        metrics_path = output_dir / f"{batch_id}_per_file_metrics.json"
        
        try:
            per_file_data = []
            
            for i, result in enumerate(batch_report.file_results):
                file_metrics = {
                    'file_index': i + 1,
                    'processing_summary': {
                        'addresses_processed': len(result.parsed_addresses),
                        'processing_time': result.processing_time,
                        'gpu_processing_time': result.gpu_processing_time,
                        'cpu_processing_time': result.cpu_processing_time,
                        'success_rate': result.calculate_success_rate(),
                        'error_count': result.error_count
                    },
                    'performance_metrics': {},
                    'device_statistics': result.device_statistics,
                    'optimization_suggestions': result.optimization_suggestions,
                    'error_details': result.error_details
                }
                
                # Add performance metrics if available
                if result.performance_metrics:
                    file_metrics['performance_metrics'] = {
                        'throughput_rate': result.performance_metrics.throughput_rate,
                        'gpu_utilization': result.performance_metrics.gpu_utilization,
                        'cpu_utilization': result.performance_metrics.cpu_utilization,
                        'processing_efficiency': result.performance_metrics.processing_efficiency,
                        'memory_usage': result.performance_metrics.memory_usage,
                        'gpu_processed': result.performance_metrics.gpu_processed,
                        'cpu_processed': result.performance_metrics.cpu_processed
                    }
                
                per_file_data.append(file_metrics)
            
            # Write JSON report
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'batch_id': batch_id,
                    'generation_timestamp': datetime.now().isoformat(),
                    'total_files': len(batch_report.file_results),
                    'batch_summary': {
                        'total_addresses': batch_report.total_addresses,
                        'average_throughput': batch_report.average_throughput,
                        'peak_throughput': batch_report.peak_throughput,
                        'average_gpu_utilization': batch_report.average_gpu_utilization,
                        'processing_efficiency': batch_report.processing_efficiency
                    },
                    'per_file_metrics': per_file_data
                }, f, indent=2, default=str)
            
            self.logger.info(f"✅ Per-file metrics generated: {metrics_path}")
            return str(metrics_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate per-file metrics: {e}")
            raise
    
    def _generate_batch_json_report(
        self,
        batch_report: 'BatchProcessingReport',
        output_dir: Path,
        batch_id: str
    ) -> str:
        """Generate comprehensive batch report in JSON format."""
        json_path = output_dir / f"{batch_id}_batch_report.json"
        
        try:
            # Convert batch report to dictionary
            batch_data = {
                'batch_id': batch_id,
                'generation_timestamp': datetime.now().isoformat(),
                'batch_summary': {
                    'total_files_processed': batch_report.total_files_processed,
                    'total_addresses': batch_report.total_addresses,
                    'average_throughput': batch_report.average_throughput,
                    'peak_throughput': batch_report.peak_throughput,
                    'average_gpu_utilization': batch_report.average_gpu_utilization,
                    'processing_efficiency': batch_report.processing_efficiency,
                    'performance_summary': batch_report.performance_summary
                },
                'performance_analysis': {
                    'throughput_target_met': batch_report.average_throughput >= 1500,
                    'gpu_utilization_target_met': batch_report.average_gpu_utilization >= 90,
                    'performance_score': self._calculate_batch_performance_score(batch_report),
                    'optimization_recommendations': self._generate_batch_recommendations(batch_report)
                },
                'file_processing_details': []
            }
            
            # Add file processing details
            for i, result in enumerate(batch_report.file_results):
                file_detail = {
                    'file_index': i + 1,
                    'addresses_processed': len(result.parsed_addresses),
                    'processing_time': result.processing_time,
                    'success_rate': result.calculate_success_rate(),
                    'error_count': result.error_count,
                    'device_statistics': result.device_statistics,
                    'optimization_suggestions': result.optimization_suggestions
                }
                
                if result.performance_metrics:
                    file_detail['performance_metrics'] = {
                        'throughput_rate': result.performance_metrics.throughput_rate,
                        'gpu_utilization': result.performance_metrics.gpu_utilization,
                        'processing_efficiency': result.performance_metrics.processing_efficiency
                    }
                
                batch_data['file_processing_details'].append(file_detail)
            
            # Write JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Batch JSON report generated: {json_path}")
            return str(json_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch JSON report: {e}")
            raise
    
    # Helper methods for analysis and scoring
    
    def _calculate_batch_performance_score(self, batch_report: 'BatchProcessingReport') -> float:
        """Calculate overall performance score for batch processing (0-100)."""
        # Throughput score (50% weight)
        throughput_score = min((batch_report.average_throughput / 1500) * 50, 50)
        
        # GPU utilization score (30% weight)
        gpu_score = min((batch_report.average_gpu_utilization / 90) * 30, 30)
        
        # Efficiency score (20% weight)
        efficiency_score = batch_report.processing_efficiency * 0.2
        
        return throughput_score + gpu_score + efficiency_score
    
    def _calculate_file_efficiency_score(
        self,
        result: 'ProcessingResult',
        throughput: float,
        gpu_utilization: float
    ) -> float:
        """Calculate efficiency score for individual file processing."""
        # Base efficiency from processing result
        base_efficiency = result.performance_metrics.processing_efficiency if result.performance_metrics else 0
        
        # Throughput efficiency (target: 1500 addr/sec)
        throughput_efficiency = min((throughput / 1500) * 100, 100)
        
        # GPU utilization efficiency (target: 90%)
        gpu_efficiency = min((gpu_utilization / 90) * 100, 100)
        
        # Combined efficiency score
        return (base_efficiency + throughput_efficiency + gpu_efficiency) / 3
    
    def _generate_batch_recommendations(self, batch_report: 'BatchProcessingReport') -> List[str]:
        """Generate optimization recommendations for batch processing."""
        recommendations = []
        
        # Throughput recommendations
        if batch_report.average_throughput < 1500:
            gap = 1500 - batch_report.average_throughput
            recommendations.append(
                f"Increase throughput by {gap:.1f} addr/sec. Consider: larger GPU batch sizes, "
                f"model compilation, or dataset optimization."
            )
        
        # GPU utilization recommendations
        if batch_report.average_gpu_utilization < 90:
            gap = 90 - batch_report.average_gpu_utilization
            recommendations.append(
                f"Improve GPU utilization by {gap:.1f}%. Consider: increasing queue size, "
                f"reducing CPU allocation, or enabling more GPU streams."
            )
        
        # Efficiency recommendations
        if batch_report.processing_efficiency < 80:
            recommendations.append(
                "Improve processing efficiency. Consider: optimizing memory allocation, "
                "enabling advanced GPU optimizations, or tuning batch sizes."
            )
        
        # Consistency recommendations
        if batch_report.file_results:
            throughputs = [
                len(r.parsed_addresses) / r.processing_time 
                for r in batch_report.file_results 
                if r.processing_time > 0
            ]
            
            if throughputs:
                throughput_std = (sum((t - batch_report.average_throughput) ** 2 for t in throughputs) / len(throughputs)) ** 0.5
                cv = throughput_std / batch_report.average_throughput if batch_report.average_throughput > 0 else 0
                
                if cv > 0.3:  # High coefficient of variation
                    recommendations.append(
                        "High performance variability detected across files. Consider: "
                        "consistent batch sizes, file size normalization, or system resource monitoring."
                    )
        
        # Default recommendation if none specific
        if not recommendations:
            recommendations.append(
                "Performance targets achieved. Consider monitoring for consistency "
                "and exploring advanced optimization techniques for further improvements."
            )
        
        return recommendations


# Utility functions for integration with existing components

def create_comprehensive_output_from_processing_result(
    processing_result: 'ProcessingResult',
    original_records: List[AddressRecord],
    output_path: str,
    config: ComprehensiveOutputConfig = None
) -> Dict[str, str]:
    """Utility function to create comprehensive output from processing result.
    
    Args:
        processing_result: Complete processing result with parsed addresses
        original_records: Original address records from input
        output_path: Base path for output files
        config: Optional output configuration
        
    Returns:
        Dictionary mapping output type to file path
    """
    generator = ComprehensiveOutputGenerator(config)
    
    return generator.generate_comprehensive_output(
        processing_result.parsed_addresses,
        original_records,
        processing_result,
        output_path
    )


def create_batch_processing_report(
    batch_report: 'BatchProcessingReport',
    output_dir: str,
    batch_id: str = None,
    config: ComprehensiveOutputConfig = None
) -> Dict[str, str]:
    """Utility function to create batch processing report.
    
    Args:
        batch_report: Batch processing report with file results
        output_dir: Directory for batch report outputs
        batch_id: Optional batch identifier
        config: Optional output configuration
        
    Returns:
        Dictionary mapping report type to file path
    """
    generator = ComprehensiveOutputGenerator(config)
    
    return generator.generate_batch_processing_report(
        batch_report,
        output_dir,
        batch_id
    )