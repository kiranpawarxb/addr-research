"""Tests for data models."""

import pytest
from src.models import AddressRecord, ParsedAddress, ConsolidatedGroup, ConsolidationStats


class TestAddressRecord:
    """Tests for AddressRecord dataclass."""
    
    def test_create_valid_address_record(self):
        """Test creating a valid AddressRecord."""
        record = AddressRecord(
            addr_hash_key="hash123",
            addr_text="123 Main St, Mumbai 400001",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=5,
            raw_data={"extra": "data"}
        )
        assert record.addr_text == "123 Main St, Mumbai 400001"
        assert record.pincode == "400001"
        assert record.raw_data == {"extra": "data"}
    
    def test_address_record_validation_empty_addr_text(self):
        """Test that empty addr_text raises ValueError."""
        with pytest.raises(ValueError, match="addr_text cannot be empty"):
            AddressRecord(
                addr_hash_key="hash123",
                addr_text="",
                city_id="city1",
                pincode="400001",
                state_id="state1",
                zone_id="zone1",
                address_id="addr1",
                assigned_pickup_dlvd_geo_points="point1",
                assigned_pickup_dlvd_geo_points_count=5
            )


class TestParsedAddress:
    """Tests for ParsedAddress dataclass."""
    
    def test_create_parsed_address_with_all_fields(self):
        """Test creating a ParsedAddress with all fields populated."""
        parsed = ParsedAddress(
            unit_number="101",
            society_name="Green Valley Apartments",
            landmark="Near City Mall",
            road="MG Road",
            sub_locality="Bandra West",
            locality="Bandra",
            city="Mumbai",
            district="Mumbai Suburban",
            state="Maharashtra",
            country="India",
            pin_code="400050",
            note="Verified address",
            parse_success=True
        )
        assert parsed.society_name == "Green Valley Apartments"
        assert parsed.pin_code == "400050"
        assert parsed.parse_success is True
    
    def test_parsed_address_defaults_to_empty_strings(self):
        """Test that ParsedAddress fields default to empty strings."""
        parsed = ParsedAddress()
        assert parsed.unit_number == ""
        assert parsed.society_name == ""
        assert parsed.parse_success is False
        assert parsed.parse_error is None


class TestConsolidatedGroup:
    """Tests for ConsolidatedGroup dataclass."""
    
    def test_create_consolidated_group(self):
        """Test creating a ConsolidatedGroup."""
        group = ConsolidatedGroup(
            group_id="group1",
            society_name="Green Valley",
            pin_code="400050"
        )
        assert group.group_id == "group1"
        assert group.record_count == 0
        assert len(group.records) == 0
    
    def test_add_record_to_group(self):
        """Test adding records to a consolidated group."""
        group = ConsolidatedGroup(
            group_id="group1",
            society_name="Green Valley",
            pin_code="400050"
        )
        
        record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Test address",
            city_id="city1",
            pincode="400050",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1
        )
        
        parsed = ParsedAddress(society_name="Green Valley", pin_code="400050")
        
        group.add_record(record, parsed)
        assert group.record_count == 1
        assert len(group.records) == 1


class TestConsolidationStats:
    """Tests for ConsolidationStats dataclass."""
    
    def test_create_valid_stats(self):
        """Test creating valid ConsolidationStats."""
        stats = ConsolidationStats(
            total_records=100,
            total_groups=10,
            avg_records_per_group=10.0,
            largest_group_size=25,
            largest_group_name="Green Valley",
            match_percentage=95.0,
            parse_success_count=95,
            parse_failure_count=5,
            unmatched_count=5
        )
        assert stats.total_records == 100
        assert stats.total_groups == 10
        assert stats.avg_records_per_group == 10.0
    
    def test_stats_validation_parse_counts_mismatch(self):
        """Test that mismatched parse counts raise ValueError."""
        with pytest.raises(ValueError, match="Parse counts.*must equal total records"):
            ConsolidationStats(
                total_records=100,
                total_groups=10,
                avg_records_per_group=10.0,
                largest_group_size=25,
                largest_group_name="Green Valley",
                match_percentage=95.0,
                parse_success_count=90,  # 90 + 5 = 95, not 100
                parse_failure_count=5
            )
    
    def test_stats_validation_invalid_match_percentage(self):
        """Test that invalid match percentage raises ValueError."""
        with pytest.raises(ValueError, match="match_percentage must be between 0 and 100"):
            ConsolidationStats(
                total_records=100,
                total_groups=10,
                avg_records_per_group=10.0,
                largest_group_size=25,
                largest_group_name="Green Valley",
                match_percentage=150.0,  # Invalid: > 100
                parse_success_count=95,
                parse_failure_count=5
            )
