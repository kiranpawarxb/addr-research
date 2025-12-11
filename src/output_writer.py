"""Output Writer component for the Address Consolidation System."""

import csv
import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from .models import ConsolidatedGroup, AddressRecord, ParsedAddress
    from .comprehensive_output_generator import (
        ComprehensiveOutputGenerator, 
        ComprehensiveOutputConfig,
        create_comprehensive_output_from_processing_result
    )
    from .hybrid_processor import ProcessingResult
except ImportError:
    # Fallback for direct execution
    from models import ConsolidatedGroup, AddressRecord, ParsedAddress
    from comprehensive_output_generator import (
        ComprehensiveOutputGenerator, 
        ComprehensiveOutputConfig,
        create_comprehensive_output_from_processing_result
    )
    from hybrid_processor import ProcessingResult


logger = logging.getLogger(__name__)


class OutputWriter:
    """Writes consolidated results to CSV with all parsed components.
    
    Handles CSV export with original columns plus parsed fields and group identifiers.
    Implements error handling for file write operations and character escaping.
    Enhanced with comprehensive output generation capabilities.
    """
    
    def __init__(self, output_path: str, comprehensive_output: bool = False):
        """Initialize writer with output file path.
        
        Args:
            output_path: Path where the output CSV file will be written
            comprehensive_output: Whether to generate comprehensive output with metadata
        """
        self.output_path = output_path
        self.comprehensive_output = comprehensive_output
        self.comprehensive_generator = None
        
        if comprehensive_output:
            self.comprehensive_generator = ComprehensiveOutputGenerator()
        
    def write(self, consolidated_groups: List[ConsolidatedGroup]) -> int:
        """Write consolidated groups to CSV.
        
        Creates a CSV file with all original columns plus parsed address fields
        and group identifiers. Handles file write errors gracefully.
        
        Args:
            consolidated_groups: List of ConsolidatedGroup objects to write
            
        Returns:
            Total number of records written
            
        Raises:
            PermissionError: If file cannot be written due to permissions
            OSError: If disk is full or other I/O error occurs
        """
        if not consolidated_groups:
            logger.warning("No consolidated groups to write")
            return 0
        
        logger.info(f"Starting output write to: {self.output_path}")
        logger.info(f"Writing {len(consolidated_groups)} consolidated groups...")
        
        # Check if output directory exists and is writable
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                logger.info(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
            except PermissionError as e:
                logger.error(f"Permission denied creating directory: {output_dir}")
                raise PermissionError(f"Cannot create output directory: {output_dir}") from e
        
        # Check write permissions
        if output_dir:
            if not os.access(output_dir, os.W_OK):
                logger.error(f"No write permission for directory: {output_dir}")
                raise PermissionError(f"No write permission for directory: {output_dir}")
        
        total_records = 0
        rows_to_write = []
        
        logger.debug("Building output rows from consolidated groups...")
        
        # Collect all rows from all groups
        for group in consolidated_groups:
            for address_record, parsed_address in group.records:
                row = self._build_output_row(address_record, parsed_address, group.group_id)
                rows_to_write.append(row)
                total_records += 1
        
        if not rows_to_write:
            logger.warning("No records to write")
            return 0
        
        logger.info(f"Prepared {total_records} rows for output")
        
        # Determine column order: original columns + parsed fields + group_id + location_identifier
        fieldnames = self._get_fieldnames(rows_to_write[0])
        logger.debug(f"Output CSV will have {len(fieldnames)} columns")
        
        try:
            # Write to CSV file
            logger.debug(f"Writing CSV file to: {self.output_path}")
            with open(self.output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_to_write)
            
            logger.info(f"Successfully wrote {total_records} records to {self.output_path}")
            return total_records
            
        except PermissionError as e:
            logger.error(f"Permission denied writing to file: {self.output_path}")
            raise PermissionError(f"Cannot write to file: {self.output_path}") from e
        except OSError as e:
            # Handle disk full and other I/O errors
            if e.errno == 28:  # ENOSPC - No space left on device
                logger.error(f"Disk full - cannot write to: {self.output_path}")
                # Attempt to clean up partial file
                self._cleanup_partial_file()
                raise OSError(f"Disk full - cannot write to: {self.output_path}") from e
            else:
                logger.error(f"I/O error writing to file: {self.output_path} - {e}")
                raise OSError(f"Error writing to file: {self.output_path}") from e
    
    def _build_output_row(
        self,
        record: AddressRecord,
        parsed: ParsedAddress,
        group_id: str
    ) -> Dict[str, Any]:
        """Build a single output row with all fields.
        
        Combines original CSV columns, parsed address fields, and group identifiers.
        Escapes invalid characters in data.
        
        Args:
            record: Original AddressRecord
            parsed: ParsedAddress with extracted fields
            group_id: Unique identifier for the consolidated group
            
        Returns:
            Dictionary representing a single CSV row
        """
        # Start with original columns from raw_data
        row = dict(record.raw_data)
        
        # Add known fields from AddressRecord (in case they're not in raw_data)
        row.update({
            'addr_hash_key': self._escape_value(record.addr_hash_key),
            'addr_text': self._escape_value(record.addr_text),
            'city_id': self._escape_value(record.city_id),
            'pincode': self._escape_value(record.pincode),
            'state_id': self._escape_value(record.state_id),
            'zone_id': self._escape_value(record.zone_id),
            'address_id': self._escape_value(record.address_id),
            'assigned_pickup_dlvd_geo_points': self._escape_value(record.assigned_pickup_dlvd_geo_points),
            'assigned_pickup_dlvd_geo_points_count': record.assigned_pickup_dlvd_geo_points_count,
        })
        
        # Add parsed address fields with standard column names
        row.update({
            'UN': self._escape_value(parsed.unit_number),
            'SN': self._escape_value(parsed.society_name),
            'LN': self._escape_value(parsed.landmark),
            'RD': self._escape_value(parsed.road),
            'SL': self._escape_value(parsed.sub_locality),
            'LOC': self._escape_value(parsed.locality),
            'CY': self._escape_value(parsed.city),
            'DIS': self._escape_value(parsed.district),
            'ST': self._escape_value(parsed.state),
            'CN': self._escape_value(parsed.country),
            'PIN': self._escape_value(parsed.pin_code),
            'Note': self._escape_value(parsed.note),
        })
        
        # Add group identifier
        row['group_id'] = self._escape_value(group_id)
        
        # Add location identifier (combination of society name and PIN)
        location_identifier = f"{parsed.society_name}_{parsed.pin_code}" if parsed.society_name else f"UNMATCHED_{parsed.pin_code}"
        row['location_identifier'] = self._escape_value(location_identifier)
        
        return row
    
    def _escape_value(self, value: Any) -> str:
        """Escape invalid characters in output data.
        
        Handles None values, converts to string, and escapes problematic characters
        that could break CSV formatting.
        
        Args:
            value: Value to escape
            
        Returns:
            Escaped string value
        """
        if value is None:
            return ""
        
        # Convert to string
        str_value = str(value)
        
        # Replace null bytes which can cause issues in CSV
        str_value = str_value.replace('\x00', '')
        
        # Replace other control characters (except newlines and tabs which CSV handles)
        # Keep \n and \t as they're valid in CSV fields
        for i in range(32):
            if i not in (9, 10, 13):  # Keep tab, newline, carriage return
                str_value = str_value.replace(chr(i), '')
        
        return str_value
    
    def _get_fieldnames(self, sample_row: Dict[str, Any]) -> List[str]:
        """Determine the column order for the output CSV.
        
        Orders columns as: original columns, parsed fields, group_id, location_identifier
        
        Args:
            sample_row: A sample row to extract field names from
            
        Returns:
            Ordered list of field names
        """
        # Define the order of parsed fields
        parsed_fields = ['UN', 'SN', 'LN', 'RD', 'SL', 'LOC', 'CY', 'DIS', 'ST', 'CN', 'PIN', 'Note']
        group_fields = ['group_id', 'location_identifier']
        
        # Get all original columns (excluding parsed and group fields)
        original_columns = [
            key for key in sample_row.keys()
            if key not in parsed_fields and key not in group_fields
        ]
        
        # Combine in order: original + parsed + group
        fieldnames = original_columns + parsed_fields + group_fields
        
        return fieldnames
    
    def _cleanup_partial_file(self):
        """Attempt to clean up a partially written file.
        
        Called when disk full error occurs to remove incomplete output.
        """
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
                logger.info(f"Cleaned up partial file: {self.output_path}")
        except Exception as e:
            logger.warning(f"Could not clean up partial file: {e}")
    
    def write_comprehensive_output(
        self,
        processing_result: ProcessingResult,
        original_records: List[AddressRecord],
        config: Optional[ComprehensiveOutputConfig] = None
    ) -> Dict[str, str]:
        """Write comprehensive output with all parsed fields and metadata.
        
        Generates comprehensive output including CSV with all parsed fields,
        JSON metadata, performance reports, and error analysis.
        
        Args:
            processing_result: Complete processing result with parsed addresses
            original_records: Original address records from input
            config: Optional configuration for output generation
            
        Returns:
            Dictionary mapping output type to file path
            
        Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
        """
        if not self.comprehensive_generator:
            self.comprehensive_generator = ComprehensiveOutputGenerator(config)
        
        logger.info(f"Generating comprehensive output for {len(processing_result.parsed_addresses)} addresses")
        
        try:
            # Use the comprehensive output generator
            output_files = self.comprehensive_generator.generate_comprehensive_output(
                processing_result.parsed_addresses,
                original_records,
                processing_result,
                self.output_path
            )
            
            logger.info(f"✅ Comprehensive output generated: {len(output_files)} files created")
            
            return output_files
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive output: {e}")
            raise
    
    def write_batch_processing_report(
        self,
        batch_report: 'BatchProcessingReport',
        output_dir: str,
        batch_id: str = None,
        config: Optional[ComprehensiveOutputConfig] = None
    ) -> Dict[str, str]:
        """Write comprehensive batch processing report.
        
        Generates detailed batch processing reports including per-file results,
        comparative performance analysis, and optimization recommendations.
        
        Args:
            batch_report: Batch processing report with file results
            output_dir: Directory for batch report outputs
            batch_id: Optional batch identifier
            config: Optional configuration for output generation
            
        Returns:
            Dictionary mapping report type to file path
            
        Requirements: 9.3, 9.4
        """
        if not self.comprehensive_generator:
            self.comprehensive_generator = ComprehensiveOutputGenerator(config)
        
        logger.info(f"Generating batch processing report for {batch_report.total_files_processed} files")
        
        try:
            # Use the comprehensive output generator
            report_files = self.comprehensive_generator.generate_batch_processing_report(
                batch_report,
                output_dir,
                batch_id
            )
            
            logger.info(f"✅ Batch processing report generated: {len(report_files)} files created")
            
            return report_files
            
        except Exception as e:
            logger.error(f"Failed to generate batch processing report: {e}")
            raise
