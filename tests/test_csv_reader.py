"""Tests for the CSV Reader component."""

import csv
import logging
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from src.csv_reader import CSVReader
from src.models import AddressRecord


class TestCSVReader:
    """Test suite for CSVReader class."""
    
    def test_read_valid_csv(self, tmp_path):
        """Test reading a valid CSV file with all required columns."""
        # Create a temporary CSV file
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'addr_hash_key', 'addr_text', 'city_id', 'pincode', 'state_id',
                'zone_id', 'address_id', 'assigned_pickup_dlvd_geo_points',
                'assigned_pickup_dlvd_geo_points_count'
            ])
            writer.writeheader()
            writer.writerow({
                'addr_hash_key': 'abc123',
                'addr_text': '123 Main St, Mumbai',
                'city_id': 'MUM',
                'pincode': '400001',
                'state_id': 'MH',
                'zone_id': '1',
                'address_id': 'addr1',
                'assigned_pickup_dlvd_geo_points': 'POINT(1 2)',
                'assigned_pickup_dlvd_geo_points_count': '1'
            })
            writer.writerow({
                'addr_hash_key': 'def456',
                'addr_text': '456 Park Ave, Delhi',
                'city_id': 'DEL',
                'pincode': '110001',
                'state_id': 'DL',
                'zone_id': '2',
                'address_id': 'addr2',
                'assigned_pickup_dlvd_geo_points': 'POINT(3 4)',
                'assigned_pickup_dlvd_geo_points_count': '2'
            })
        
        # Read the CSV
        reader = CSVReader(str(csv_file), ['addr_text', 'pincode', 'city_id'])
        records = list(reader.read())
        
        # Verify results
        assert len(records) == 2
        assert reader.get_total_loaded() == 2
        assert reader.get_malformed_count() == 0
        
        # Check first record
        assert records[0].addr_hash_key == 'abc123'
        assert records[0].addr_text == '123 Main St, Mumbai'
        assert records[0].city_id == 'MUM'
        assert records[0].pincode == '400001'
        assert records[0].assigned_pickup_dlvd_geo_points_count == 1
        
        # Check second record
        assert records[1].addr_hash_key == 'def456'
        assert records[1].addr_text == '456 Park Ave, Delhi'
        assert records[1].city_id == 'DEL'
        assert records[1].pincode == '110001'
    
    def test_validate_columns_success(self, tmp_path):
        """Test column validation with all required columns present."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'addr_hash_key', 'addr_text', 'city_id', 'pincode'
            ])
            writer.writeheader()
        
        reader = CSVReader(str(csv_file), ['addr_text', 'pincode', 'city_id'])
        is_valid, missing = reader.validate_columns()
        
        assert is_valid is True
        assert missing == []
    
    def test_validate_columns_missing(self, tmp_path):
        """Test column validation with missing required columns."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['addr_text', 'city_id'])
            writer.writeheader()
        
        reader = CSVReader(str(csv_file), ['addr_text', 'pincode', 'city_id'])
        is_valid, missing = reader.validate_columns()
        
        assert is_valid is False
        assert 'pincode' in missing
    
    def test_file_not_found(self):
        """Test error handling when CSV file doesn't exist."""
        reader = CSVReader('nonexistent.csv', ['addr_text'])
        
        with pytest.raises(FileNotFoundError) as exc_info:
            reader.validate_columns()
        
        assert 'nonexistent.csv' in str(exc_info.value)
    
    def test_empty_file(self, tmp_path):
        """Test error handling for empty CSV file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.touch()  # Create empty file
        
        reader = CSVReader(str(csv_file), ['addr_text'])
        
        with pytest.raises(ValueError) as exc_info:
            reader.validate_columns()
        
        assert 'empty' in str(exc_info.value).lower()
    
    def test_malformed_row_skipped(self, tmp_path, caplog):
        """Test that malformed rows are skipped and logged."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'addr_hash_key', 'addr_text', 'city_id', 'pincode', 'state_id',
                'zone_id', 'address_id', 'assigned_pickup_dlvd_geo_points',
                'assigned_pickup_dlvd_geo_points_count'
            ])
            writer.writeheader()
            # Valid row
            writer.writerow({
                'addr_hash_key': 'abc123',
                'addr_text': '123 Main St',
                'city_id': 'MUM',
                'pincode': '400001',
                'state_id': 'MH',
                'zone_id': '1',
                'address_id': 'addr1',
                'assigned_pickup_dlvd_geo_points': 'POINT(1 2)',
                'assigned_pickup_dlvd_geo_points_count': '1'
            })
            # Malformed row - missing addr_text
            writer.writerow({
                'addr_hash_key': 'def456',
                'addr_text': '',  # Empty addr_text
                'city_id': 'DEL',
                'pincode': '110001',
                'state_id': 'DL',
                'zone_id': '2',
                'address_id': 'addr2',
                'assigned_pickup_dlvd_geo_points': 'POINT(3 4)',
                'assigned_pickup_dlvd_geo_points_count': '2'
            })
            # Another valid row
            writer.writerow({
                'addr_hash_key': 'ghi789',
                'addr_text': '789 Oak Rd',
                'city_id': 'BLR',
                'pincode': '560001',
                'state_id': 'KA',
                'zone_id': '3',
                'address_id': 'addr3',
                'assigned_pickup_dlvd_geo_points': 'POINT(5 6)',
                'assigned_pickup_dlvd_geo_points_count': '1'
            })
        
        # Read with logging enabled
        with caplog.at_level(logging.WARNING):
            reader = CSVReader(str(csv_file), ['addr_text', 'pincode', 'city_id'])
            records = list(reader.read())
        
        # Verify results
        assert len(records) == 2  # Only valid rows
        assert reader.get_total_loaded() == 2
        assert reader.get_malformed_count() == 1
        
        # Check that warning was logged
        assert any('malformed' in record.message.lower() for record in caplog.records)
        assert any('row 3' in record.message.lower() for record in caplog.records)
    
    def test_missing_required_columns_raises_error(self, tmp_path):
        """Test that reading fails when required columns are missing."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['addr_text', 'city_id'])
            writer.writeheader()
            writer.writerow({'addr_text': '123 Main St', 'city_id': 'MUM'})
        
        reader = CSVReader(str(csv_file), ['addr_text', 'pincode', 'city_id'])
        
        with pytest.raises(ValueError) as exc_info:
            list(reader.read())
        
        assert 'missing required columns' in str(exc_info.value).lower()
        assert 'pincode' in str(exc_info.value)
    
    def test_raw_data_preserved(self, tmp_path):
        """Test that all original CSV columns are preserved in raw_data."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'addr_hash_key', 'addr_text', 'city_id', 'pincode', 'state_id',
                'zone_id', 'address_id', 'assigned_pickup_dlvd_geo_points',
                'assigned_pickup_dlvd_geo_points_count', 'extra_column'
            ])
            writer.writeheader()
            writer.writerow({
                'addr_hash_key': 'abc123',
                'addr_text': '123 Main St',
                'city_id': 'MUM',
                'pincode': '400001',
                'state_id': 'MH',
                'zone_id': '1',
                'address_id': 'addr1',
                'assigned_pickup_dlvd_geo_points': 'POINT(1 2)',
                'assigned_pickup_dlvd_geo_points_count': '1',
                'extra_column': 'extra_value'
            })
        
        reader = CSVReader(str(csv_file), ['addr_text', 'pincode', 'city_id'])
        records = list(reader.read())
        
        assert len(records) == 1
        assert 'extra_column' in records[0].raw_data
        assert records[0].raw_data['extra_column'] == 'extra_value'
    
    def test_empty_pincode_raises_error(self, tmp_path):
        """Test that empty pincode causes row to be skipped."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'addr_hash_key', 'addr_text', 'city_id', 'pincode', 'state_id',
                'zone_id', 'address_id', 'assigned_pickup_dlvd_geo_points',
                'assigned_pickup_dlvd_geo_points_count'
            ])
            writer.writeheader()
            writer.writerow({
                'addr_hash_key': 'abc123',
                'addr_text': '123 Main St',
                'city_id': 'MUM',
                'pincode': '',  # Empty pincode
                'state_id': 'MH',
                'zone_id': '1',
                'address_id': 'addr1',
                'assigned_pickup_dlvd_geo_points': 'POINT(1 2)',
                'assigned_pickup_dlvd_geo_points_count': '1'
            })
        
        reader = CSVReader(str(csv_file), ['addr_text', 'pincode', 'city_id'])
        records = list(reader.read())
        
        assert len(records) == 0
        assert reader.get_malformed_count() == 1
    
    def test_non_numeric_count_defaults_to_zero(self, tmp_path):
        """Test that non-numeric count values default to 0."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'addr_hash_key', 'addr_text', 'city_id', 'pincode', 'state_id',
                'zone_id', 'address_id', 'assigned_pickup_dlvd_geo_points',
                'assigned_pickup_dlvd_geo_points_count'
            ])
            writer.writeheader()
            writer.writerow({
                'addr_hash_key': 'abc123',
                'addr_text': '123 Main St',
                'city_id': 'MUM',
                'pincode': '400001',
                'state_id': 'MH',
                'zone_id': '1',
                'address_id': 'addr1',
                'assigned_pickup_dlvd_geo_points': 'POINT(1 2)',
                'assigned_pickup_dlvd_geo_points_count': 'invalid'
            })
        
        reader = CSVReader(str(csv_file), ['addr_text', 'pincode', 'city_id'])
        records = list(reader.read())
        
        assert len(records) == 1
        assert records[0].assigned_pickup_dlvd_geo_points_count == 0
    
    def test_streaming_large_file(self, tmp_path):
        """Test that large files are streamed efficiently (iterator pattern)."""
        csv_file = tmp_path / "large.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'addr_hash_key', 'addr_text', 'city_id', 'pincode', 'state_id',
                'zone_id', 'address_id', 'assigned_pickup_dlvd_geo_points',
                'assigned_pickup_dlvd_geo_points_count'
            ])
            writer.writeheader()
            for i in range(1000):
                writer.writerow({
                    'addr_hash_key': f'hash{i}',
                    'addr_text': f'{i} Street Name',
                    'city_id': 'MUM',
                    'pincode': '400001',
                    'state_id': 'MH',
                    'zone_id': '1',
                    'address_id': f'addr{i}',
                    'assigned_pickup_dlvd_geo_points': 'POINT(1 2)',
                    'assigned_pickup_dlvd_geo_points_count': '1'
                })
        
        reader = CSVReader(str(csv_file), ['addr_text', 'pincode', 'city_id'])
        
        # Read only first 10 records to verify streaming works
        count = 0
        for record in reader.read():
            count += 1
            if count == 10:
                break
        
        assert count == 10
