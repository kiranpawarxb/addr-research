"""Data models for the Address Consolidation System."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AddressRecord:
    """Raw address record from CSV file.
    
    Represents a single row in the CSV file containing address information.
    Preserves all original columns for data integrity.
    """
    addr_hash_key: str
    addr_text: str
    city_id: str
    pincode: str
    state_id: str
    zone_id: str
    address_id: str
    assigned_pickup_dlvd_geo_points: str
    assigned_pickup_dlvd_geo_points_count: int
    raw_data: Dict[str, Any] = field(default_factory=dict)  # Preserve all original columns
    
    def __post_init__(self):
        """Validate required fields are not empty."""
        if not self.addr_text:
            raise ValueError("addr_text cannot be empty")
        if not self.pincode:
            raise ValueError("pincode cannot be empty")
        if not self.city_id:
            raise ValueError("city_id cannot be empty")


@dataclass
class ParsedAddress:
    """Structured address components extracted by LLM.
    
    Contains all 12 standardized address fields extracted from raw address text.
    Tracks parsing success/failure status.
    """
    unit_number: str = ""  # UN
    society_name: str = ""  # SN
    landmark: str = ""  # LN
    road: str = ""  # RD
    sub_locality: str = ""  # SL
    locality: str = ""  # LOC
    city: str = ""  # CY
    district: str = ""  # DIS
    state: str = ""  # ST
    country: str = ""  # CN
    pin_code: str = ""  # PIN
    note: str = ""  # Note
    parse_success: bool = False
    parse_error: Optional[str] = None
    
    def __post_init__(self):
        """Normalize empty fields to empty string or 'NA'."""
        # Convert None values to empty strings
        for field_name in ['unit_number', 'society_name', 'landmark', 'road', 
                          'sub_locality', 'locality', 'city', 'district', 
                          'state', 'country', 'pin_code', 'note']:
            value = getattr(self, field_name)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                setattr(self, field_name, "")


@dataclass
class ConsolidatedGroup:
    """A group of addresses belonging to the same location.
    
    Groups addresses that share the same Society Name and PIN code.
    Each group has a unique identifier for tracking.
    """
    group_id: str
    society_name: str
    pin_code: str
    records: List[Tuple[AddressRecord, ParsedAddress]] = field(default_factory=list)
    record_count: int = 0
    
    def __post_init__(self):
        """Ensure record_count matches the actual number of records."""
        if self.record_count == 0 and self.records:
            self.record_count = len(self.records)
    
    def add_record(self, address_record: AddressRecord, parsed_address: ParsedAddress):
        """Add a record to this consolidated group."""
        self.records.append((address_record, parsed_address))
        self.record_count = len(self.records)


@dataclass
class ConsolidationStats:
    """Statistics about the consolidation process.
    
    Provides comprehensive metrics about grouping quality and parsing success.
    """
    total_records: int
    total_groups: int
    avg_records_per_group: float
    largest_group_size: int
    largest_group_name: str
    match_percentage: float
    parse_success_count: int
    parse_failure_count: int
    unmatched_count: int = 0
    
    def __post_init__(self):
        """Validate statistics consistency."""
        if self.total_records < 0:
            raise ValueError("total_records cannot be negative")
        if self.total_groups < 0:
            raise ValueError("total_groups cannot be negative")
        if self.parse_success_count + self.parse_failure_count != self.total_records:
            raise ValueError(
                f"Parse counts ({self.parse_success_count} + {self.parse_failure_count}) "
                f"must equal total records ({self.total_records})"
            )
        if self.total_groups > 0:
            expected_avg = self.total_records / self.total_groups
            # Allow small floating point differences
            if abs(self.avg_records_per_group - expected_avg) > 0.01:
                raise ValueError(
                    f"Average records per group ({self.avg_records_per_group}) "
                    f"does not match calculated value ({expected_avg})"
                )
        if not 0 <= self.match_percentage <= 100:
            raise ValueError("match_percentage must be between 0 and 100")