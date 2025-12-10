"""Tests for the Output Writer component."""

import csv
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from src.output_writer import OutputWriter
from src.models import ConsolidatedGroup, AddressRecord, ParsedAddress


class TestOutputWriter:
    """Test suite for OutputWriter class."""
    
    def test_write_single_group(self, tmp_path):
        """Test writing a single consolidated group to CSV."""
        output_path = tmp_path / "output.csv"
        writer = OutputWriter(str(output_path))
        
        # Create test data
        address_record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="123 Main St, Mumbai 400001",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1,
            raw_data={
                "addr_hash_key": "hash1",
                "addr_text": "123 Main St, Mumbai 400001",
                "city_id": "city1",
                "pincode": "400001"
            }
        )
        
        parsed_address = ParsedAddress(
            unit_number="123",
            society_name="Main Society",
            landmark="Near Park",
            road="Main St",
            sub_locality="Andheri",
            locality="West",
            city="Mumbai",
            district="Mumbai",
            state="Maharashtra",
            country="India",
            pin_code="400001",
            note="Test note",
            parse_success=True
        )
        
        group = ConsolidatedGroup(
            group_id="group1",
            society_name="Main Society",
            pin_code="400001",
            records=[(address_record, parsed_address)],
            record_count=1
        )
        
        # Write to CSV
        record_count = writer.write([group])
        
        # Verify
        assert record_count == 1
        assert output_path.exists()
        
        # Read and verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) == 1
            row = rows[0]
            
            # Check original columns
            assert row['addr_hash_key'] == "hash1"
            assert row['addr_text'] == "123 Main St, Mumbai 400001"
            assert row['pincode'] == "400001"
            
            # Check parsed fields
            assert row['UN'] == "123"
            assert row['SN'] == "Main Society"
            assert row['LN'] == "Near Park"
            assert row['RD'] == "Main St"
            assert row['SL'] == "Andheri"
            assert row['LOC'] == "West"
            assert row['CY'] == "Mumbai"
            assert row['DIS'] == "Mumbai"
            assert row['ST'] == "Maharashtra"
            assert row['CN'] == "India"
            assert row['PIN'] == "400001"
            assert row['Note'] == "Test note"
            
            # Check group fields
            assert row['group_id'] == "group1"
            assert row['location_identifier'] == "Main Society_400001"
    
    def test_write_multiple_groups(self, tmp_path):
        """Test writing multiple consolidated groups."""
        output_path = tmp_path / "output.csv"
        writer = OutputWriter(str(output_path))
        
        # Create two groups with records
        groups = []
        for i in range(2):
            address_record = AddressRecord(
                addr_hash_key=f"hash{i}",
                addr_text=f"Address {i}",
                city_id=f"city{i}",
                pincode=f"40000{i}",
                state_id=f"state{i}",
                zone_id=f"zone{i}",
                address_id=f"addr{i}",
                assigned_pickup_dlvd_geo_points=f"point{i}",
                assigned_pickup_dlvd_geo_points_count=i,
                raw_data={"addr_hash_key": f"hash{i}"}
            )
            
            parsed_address = ParsedAddress(
                society_name=f"Society {i}",
                pin_code=f"40000{i}",
                parse_success=True
            )
            
            group = ConsolidatedGroup(
                group_id=f"group{i}",
                society_name=f"Society {i}",
                pin_code=f"40000{i}",
                records=[(address_record, parsed_address)],
                record_count=1
            )
            groups.append(group)
        
        # Write to CSV
        record_count = writer.write(groups)
        
        # Verify
        assert record_count == 2
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
    
    def test_write_empty_groups(self, tmp_path):
        """Test writing empty groups list."""
        output_path = tmp_path / "output.csv"
        writer = OutputWriter(str(output_path))
        
        record_count = writer.write([])
        
        assert record_count == 0
        assert not output_path.exists()
    
    def test_write_preserves_all_original_columns(self, tmp_path):
        """Test that all original CSV columns are preserved in output."""
        output_path = tmp_path / "output.csv"
        writer = OutputWriter(str(output_path))
        
        # Create record with extra columns in raw_data
        address_record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Address",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1,
            raw_data={
                "addr_hash_key": "hash1",
                "custom_field_1": "value1",
                "custom_field_2": "value2",
                "extra_column": "extra_value"
            }
        )
        
        parsed_address = ParsedAddress(society_name="Society", pin_code="400001")
        
        group = ConsolidatedGroup(
            group_id="group1",
            society_name="Society",
            pin_code="400001",
            records=[(address_record, parsed_address)]
        )
        
        writer.write([group])
        
        # Verify all columns are present
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            row = rows[0]
            
            assert 'custom_field_1' in row
            assert 'custom_field_2' in row
            assert 'extra_column' in row
            assert row['custom_field_1'] == "value1"
            assert row['extra_column'] == "extra_value"
    
    def test_write_handles_empty_parsed_fields(self, tmp_path):
        """Test writing records with empty parsed fields."""
        output_path = tmp_path / "output.csv"
        writer = OutputWriter(str(output_path))
        
        address_record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Address",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1,
            raw_data={}
        )
        
        # ParsedAddress with all empty fields
        parsed_address = ParsedAddress(parse_success=False)
        
        group = ConsolidatedGroup(
            group_id="group1",
            society_name="",
            pin_code="",
            records=[(address_record, parsed_address)]
        )
        
        record_count = writer.write([group])
        
        assert record_count == 1
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            row = rows[0]
            
            # All parsed fields should be empty strings
            assert row['UN'] == ""
            assert row['SN'] == ""
            assert row['PIN'] == ""
    
    def test_write_handles_unmatched_group(self, tmp_path):
        """Test writing unmatched group with special group ID."""
        output_path = tmp_path / "output.csv"
        writer = OutputWriter(str(output_path))
        
        address_record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Address",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1,
            raw_data={}
        )
        
        parsed_address = ParsedAddress(society_name="", pin_code="400001")
        
        group = ConsolidatedGroup(
            group_id="UNMATCHED",
            society_name="",
            pin_code="",
            records=[(address_record, parsed_address)]
        )
        
        writer.write([group])
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            row = rows[0]
            
            assert row['group_id'] == "UNMATCHED"
            assert row['location_identifier'] == "UNMATCHED_400001"
    
    def test_escape_null_bytes(self, tmp_path):
        """Test that null bytes are escaped in output."""
        output_path = tmp_path / "output.csv"
        writer = OutputWriter(str(output_path))
        
        address_record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Address\x00with\x00nulls",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1,
            raw_data={}
        )
        
        parsed_address = ParsedAddress(
            society_name="Society\x00Name",
            pin_code="400001"
        )
        
        group = ConsolidatedGroup(
            group_id="group1",
            society_name="Society Name",
            pin_code="400001",
            records=[(address_record, parsed_address)]
        )
        
        writer.write([group])
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            row = rows[0]
            
            # Null bytes should be removed
            assert '\x00' not in row['addr_text']
            assert '\x00' not in row['SN']
            assert row['addr_text'] == "Addresswithnulls"
            assert row['SN'] == "SocietyName"
    
    def test_escape_control_characters(self, tmp_path):
        """Test that control characters are escaped."""
        output_path = tmp_path / "output.csv"
        writer = OutputWriter(str(output_path))
        
        address_record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Address\x01\x02\x03",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1,
            raw_data={}
        )
        
        parsed_address = ParsedAddress(society_name="Society", pin_code="400001")
        
        group = ConsolidatedGroup(
            group_id="group1",
            society_name="Society",
            pin_code="400001",
            records=[(address_record, parsed_address)]
        )
        
        writer.write([group])
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Control characters should be removed
            assert '\x01' not in content
            assert '\x02' not in content
            assert '\x03' not in content
    
    def test_preserve_valid_whitespace(self, tmp_path):
        """Test that tabs and newlines are preserved in CSV fields."""
        output_path = tmp_path / "output.csv"
        writer = OutputWriter(str(output_path))
        
        address_record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Address\twith\ttabs\nand\nnewlines",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1,
            raw_data={}
        )
        
        parsed_address = ParsedAddress(society_name="Society", pin_code="400001")
        
        group = ConsolidatedGroup(
            group_id="group1",
            society_name="Society",
            pin_code="400001",
            records=[(address_record, parsed_address)]
        )
        
        writer.write([group])
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            row = rows[0]
            
            # Tabs and newlines should be preserved
            assert '\t' in row['addr_text']
            assert '\n' in row['addr_text']
    
    def test_permission_error_on_write(self, tmp_path):
        """Test handling of permission denied error."""
        import sys
        
        # Skip on Windows as chmod doesn't work the same way
        if sys.platform == 'win32':
            pytest.skip("Permission test not applicable on Windows")
        
        output_path = tmp_path / "readonly" / "output.csv"
        
        # Create readonly directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        writer = OutputWriter(str(output_path))
        
        address_record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Address",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1,
            raw_data={}
        )
        
        parsed_address = ParsedAddress(society_name="Society", pin_code="400001")
        
        group = ConsolidatedGroup(
            group_id="group1",
            society_name="Society",
            pin_code="400001",
            records=[(address_record, parsed_address)]
        )
        
        # Should raise PermissionError
        with pytest.raises(PermissionError):
            writer.write([group])
        
        # Cleanup
        readonly_dir.chmod(0o755)
    
    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_path = tmp_path / "new_dir" / "subdir" / "output.csv"
        writer = OutputWriter(str(output_path))
        
        address_record = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Address",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1,
            raw_data={}
        )
        
        parsed_address = ParsedAddress(society_name="Society", pin_code="400001")
        
        group = ConsolidatedGroup(
            group_id="group1",
            society_name="Society",
            pin_code="400001",
            records=[(address_record, parsed_address)]
        )
        
        writer.write([group])
        
        # Verify directory was created
        assert output_path.parent.exists()
        assert output_path.exists()
    
    def test_location_identifier_format(self, tmp_path):
        """Test location_identifier format for matched and unmatched groups."""
        output_path = tmp_path / "output.csv"
        writer = OutputWriter(str(output_path))
        
        # Matched group
        address_record1 = AddressRecord(
            addr_hash_key="hash1",
            addr_text="Address 1",
            city_id="city1",
            pincode="400001",
            state_id="state1",
            zone_id="zone1",
            address_id="addr1",
            assigned_pickup_dlvd_geo_points="point1",
            assigned_pickup_dlvd_geo_points_count=1,
            raw_data={}
        )
        
        parsed_address1 = ParsedAddress(
            society_name="Test Society",
            pin_code="400001"
        )
        
        # Unmatched group
        address_record2 = AddressRecord(
            addr_hash_key="hash2",
            addr_text="Address 2",
            city_id="city2",
            pincode="400002",
            state_id="state2",
            zone_id="zone2",
            address_id="addr2",
            assigned_pickup_dlvd_geo_points="point2",
            assigned_pickup_dlvd_geo_points_count=2,
            raw_data={}
        )
        
        parsed_address2 = ParsedAddress(
            society_name="",
            pin_code="400002"
        )
        
        groups = [
            ConsolidatedGroup(
                group_id="group1",
                society_name="Test Society",
                pin_code="400001",
                records=[(address_record1, parsed_address1)]
            ),
            ConsolidatedGroup(
                group_id="UNMATCHED",
                society_name="",
                pin_code="",
                records=[(address_record2, parsed_address2)]
            )
        ]
        
        writer.write(groups)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Matched group should have "SocietyName_PIN" format
            assert rows[0]['location_identifier'] == "Test Society_400001"
            
            # Unmatched group should have "UNMATCHED_PIN" format
            assert rows[1]['location_identifier'] == "UNMATCHED_400002"
