"""CSV Reader component for the Address Consolidation System."""

import csv
import logging
from pathlib import Path
from typing import Iterator, List, Tuple

from src.models import AddressRecord


logger = logging.getLogger(__name__)


class CSVReader:
    """Reads and validates CSV files containing address data.
    
    Streams large CSV files using iterators to avoid memory issues.
    Validates required columns and handles malformed rows gracefully.
    """
    
    def __init__(self, file_path: str, required_columns: List[str]):
        """Initialize reader with file path and required columns.
        
        Args:
            file_path: Path to the CSV file to read
            required_columns: List of column names that must be present
        """
        self.file_path = Path(file_path)
        self.required_columns = required_columns
        self._total_loaded = 0
        self._malformed_count = 0
        
    def validate_columns(self) -> Tuple[bool, List[str]]:
        """Validate that required columns exist in the CSV file.
        
        Returns:
            Tuple of (is_valid, missing_columns)
            - is_valid: True if all required columns are present
            - missing_columns: List of column names that are missing
        """
        # Check if file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        
        # Check if file is empty
        if self.file_path.stat().st_size == 0:
            raise ValueError(f"CSV file is empty: {self.file_path}")
        
        # Read header and check for required columns
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                
                if fieldnames is None:
                    raise ValueError(f"CSV file has no header row: {self.file_path}")
                
                missing_columns = [col for col in self.required_columns if col not in fieldnames]
                
                if missing_columns:
                    return False, missing_columns
                
                return True, []
                
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
    
    def read(self) -> Iterator[AddressRecord]:
        """Read CSV file and yield AddressRecord objects.
        
        Streams the file line by line to handle large files efficiently.
        Skips malformed rows and logs warnings with row numbers.
        
        Yields:
            AddressRecord objects for each valid row
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If required columns are missing or file is empty
        """
        # Validate columns first
        is_valid, missing_columns = self.validate_columns()
        if not is_valid:
            raise ValueError(
                f"CSV file is missing required columns: {', '.join(missing_columns)}"
            )
        
        # Reset counters
        self._total_loaded = 0
        self._malformed_count = 0
        
        logger.info(f"Starting CSV read from: {self.file_path}")
        
        # Stream the CSV file
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    # Create AddressRecord from row
                    record = self._parse_row(row)
                    self._total_loaded += 1
                    
                    # Log progress every 1000 records
                    if self._total_loaded % 1000 == 0:
                        logger.debug(f"Loaded {self._total_loaded} records so far...")
                    
                    yield record
                    
                except (ValueError, KeyError, TypeError) as e:
                    # Log malformed row and continue (Requirement 1.4)
                    self._malformed_count += 1
                    logger.warning(
                        f"Skipping malformed row {row_num}: {e}. Row data: {row}"
                    )
                    continue
        
        # Log summary
        logger.info(
            f"CSV loading complete. Loaded {self._total_loaded} records, "
            f"skipped {self._malformed_count} malformed rows."
        )
    
    def _parse_row(self, row: dict) -> AddressRecord:
        """Parse a CSV row into an AddressRecord.
        
        Args:
            row: Dictionary representing a CSV row
            
        Returns:
            AddressRecord object
            
        Raises:
            ValueError: If required fields are missing or invalid
            KeyError: If required columns are not in the row
        """
        # Extract required fields
        addr_hash_key = row.get('addr_hash_key', '').strip()
        addr_text = row.get('addr_text', '').strip()
        city_id = row.get('city_id', '').strip()
        pincode = row.get('pincode', '').strip()
        state_id = row.get('state_id', '').strip()
        zone_id = row.get('zone_id', '').strip()
        address_id = row.get('address_id', '').strip()
        assigned_pickup_dlvd_geo_points = row.get('assigned_pickup_dlvd_geo_points', '').strip()
        
        # Parse count field (may be empty or non-numeric)
        count_str = row.get('assigned_pickup_dlvd_geo_points_count', '0').strip()
        try:
            assigned_pickup_dlvd_geo_points_count = int(count_str) if count_str else 0
        except ValueError:
            assigned_pickup_dlvd_geo_points_count = 0
        
        # Validate that we have the minimum required data
        if not addr_text:
            raise ValueError("addr_text is required but empty or missing")
        if not pincode:
            raise ValueError("pincode is required but empty or missing")
        if not city_id:
            raise ValueError("city_id is required but empty or missing")
        
        # Create AddressRecord with all original data preserved
        record = AddressRecord(
            addr_hash_key=addr_hash_key,
            addr_text=addr_text,
            city_id=city_id,
            pincode=pincode,
            state_id=state_id,
            zone_id=zone_id,
            address_id=address_id,
            assigned_pickup_dlvd_geo_points=assigned_pickup_dlvd_geo_points,
            assigned_pickup_dlvd_geo_points_count=assigned_pickup_dlvd_geo_points_count,
            raw_data=dict(row)  # Preserve all original columns
        )
        
        return record
    
    def get_total_loaded(self) -> int:
        """Get the total number of successfully loaded records.
        
        Returns:
            Count of loaded records
        """
        return self._total_loaded
    
    def get_malformed_count(self) -> int:
        """Get the count of malformed rows that were skipped.
        
        Returns:
            Count of malformed rows
        """
        return self._malformed_count
