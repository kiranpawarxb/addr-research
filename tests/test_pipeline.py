"""Tests for the main pipeline orchestrator."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path

from src.pipeline import AddressConsolidationPipeline
from src.models import AddressRecord, ParsedAddress, ConsolidatedGroup, ConsolidationStats
from src.config_loader import Config, InputConfig, LLMConfig, ConsolidationConfig, OutputConfig, LoggingConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return Config(
        input=InputConfig(
            file_path="test_input.csv",
            required_columns=["addr_text", "pincode", "city_id"]
        ),
        llm=LLMConfig(
            api_endpoint="https://api.test.com",
            api_key="test_key",
            model="gpt-4",
            max_retries=3,
            timeout_seconds=30,
            batch_size=10
        ),
        consolidation=ConsolidationConfig(
            fuzzy_matching=True,
            similarity_threshold=0.85,
            normalize_society_names=True
        ),
        output=OutputConfig(
            file_path="test_output.csv",
            include_statistics=True
        ),
        logging=LoggingConfig(
            level="INFO",
            file_path="test.log"
        )
    )


@pytest.fixture
def sample_address_record():
    """Create a sample address record for testing."""
    return AddressRecord(
        addr_hash_key="hash123",
        addr_text="Flat 101, Test Society, Test Area, Mumbai 400001",
        city_id="city1",
        pincode="400001",
        state_id="state1",
        zone_id="zone1",
        address_id="addr1",
        assigned_pickup_dlvd_geo_points="",
        assigned_pickup_dlvd_geo_points_count=0,
        raw_data={}
    )


@pytest.fixture
def sample_parsed_address():
    """Create a sample parsed address for testing."""
    return ParsedAddress(
        unit_number="Flat 101",
        society_name="Test Society",
        landmark="",
        road="",
        sub_locality="Test Area",
        locality="Test Area",
        city="Mumbai",
        district="Mumbai",
        state="Maharashtra",
        country="India",
        pin_code="400001",
        note="",
        parse_success=True,
        parse_error=None
    )


class TestAddressConsolidationPipeline:
    """Test suite for AddressConsolidationPipeline."""
    
    def test_pipeline_initialization(self, mock_config):
        """Test that pipeline initializes all components correctly."""
        pipeline = AddressConsolidationPipeline(mock_config)
        
        assert pipeline.config == mock_config
        assert pipeline.csv_reader is not None
        assert pipeline.llm_parser is not None
        assert pipeline.consolidation_engine is not None
        assert pipeline.output_writer is not None
        assert pipeline.statistics_reporter is not None
        assert pipeline._total_records == 0
        assert pipeline._failed_parses == 0
    
    @patch('src.pipeline.tqdm')
    @patch.object(AddressConsolidationPipeline, '_display_statistics')
    @patch.object(AddressConsolidationPipeline, '_write_output')
    @patch.object(AddressConsolidationPipeline, '_consolidate_addresses')
    @patch.object(AddressConsolidationPipeline, '_parse_addresses')
    @patch.object(AddressConsolidationPipeline, '_read_csv')
    def test_run_complete_pipeline(
        self,
        mock_read_csv,
        mock_parse_addresses,
        mock_consolidate,
        mock_write_output,
        mock_display_stats,
        mock_tqdm,
        mock_config,
        sample_address_record,
        sample_parsed_address
    ):
        """Test that run() executes all pipeline stages in order."""
        # Setup mocks
        mock_read_csv.return_value = [sample_address_record]
        mock_parse_addresses.return_value = [(sample_address_record, sample_parsed_address)]
        mock_consolidate.return_value = [Mock(spec=ConsolidatedGroup)]
        
        # Run pipeline
        pipeline = AddressConsolidationPipeline(mock_config)
        pipeline.run()
        
        # Verify all stages were called in order
        mock_read_csv.assert_called_once()
        mock_parse_addresses.assert_called_once_with([sample_address_record])
        mock_consolidate.assert_called_once()
        mock_write_output.assert_called_once()
        mock_display_stats.assert_called_once()
    
    @patch('src.pipeline.tqdm')
    @patch.object(AddressConsolidationPipeline, '_read_csv')
    def test_run_exits_early_with_no_records(
        self,
        mock_read_csv,
        mock_tqdm,
        mock_config
    ):
        """Test that pipeline exits gracefully when no records are loaded."""
        mock_read_csv.return_value = []
        
        pipeline = AddressConsolidationPipeline(mock_config)
        pipeline.run()
        
        # Should only call read_csv, not other stages
        mock_read_csv.assert_called_once()
    
    @patch('src.pipeline.tqdm')
    def test_read_csv_success(self, mock_tqdm, mock_config, sample_address_record):
        """Test successful CSV reading."""
        pipeline = AddressConsolidationPipeline(mock_config)
        
        # Mock the CSV reader
        pipeline.csv_reader.validate_columns = Mock(return_value=(True, []))
        pipeline.csv_reader.read = Mock(return_value=[sample_address_record])
        pipeline.csv_reader.get_malformed_count = Mock(return_value=0)
        
        # Mock tqdm to return the iterator as-is
        mock_tqdm.return_value = [sample_address_record]
        
        records = pipeline._read_csv()
        
        assert len(records) == 1
        assert records[0] == sample_address_record
        assert pipeline._total_records == 1
    
    def test_read_csv_missing_columns(self, mock_config):
        """Test CSV reading with missing required columns."""
        pipeline = AddressConsolidationPipeline(mock_config)
        
        # Mock validation to fail
        pipeline.csv_reader.validate_columns = Mock(
            return_value=(False, ["addr_text", "pincode"])
        )
        
        with pytest.raises(ValueError, match="missing required columns"):
            pipeline._read_csv()
    
    @patch('src.pipeline.tqdm')
    def test_parse_addresses_success(
        self,
        mock_tqdm,
        mock_config,
        sample_address_record,
        sample_parsed_address
    ):
        """Test successful address parsing."""
        pipeline = AddressConsolidationPipeline(mock_config)
        
        # Mock the LLM parser
        pipeline.llm_parser.parse_batch = Mock(return_value=[sample_parsed_address])
        
        # Mock tqdm to return range as-is
        mock_tqdm.return_value = range(0, 1, 10)
        
        records = [sample_address_record]
        results = pipeline._parse_addresses(records)
        
        assert len(results) == 1
        assert results[0] == (sample_address_record, sample_parsed_address)
        assert pipeline._failed_parses == 0
    
    @patch('src.pipeline.tqdm')
    def test_parse_addresses_with_failures(
        self,
        mock_tqdm,
        mock_config,
        sample_address_record
    ):
        """Test address parsing with some failures."""
        pipeline = AddressConsolidationPipeline(mock_config)
        pipeline._total_records = 1
        
        # Mock failed parse
        failed_parsed = ParsedAddress(
            parse_success=False,
            parse_error="API timeout"
        )
        pipeline.llm_parser.parse_batch = Mock(return_value=[failed_parsed])
        
        # Mock tqdm
        mock_tqdm.return_value = range(0, 1, 10)
        
        records = [sample_address_record]
        results = pipeline._parse_addresses(records)
        
        assert len(results) == 1
        assert results[0][0] == sample_address_record
        assert results[0][1].parse_success is False
        assert pipeline._failed_parses == 1
    
    @patch('src.pipeline.tqdm')
    def test_parse_addresses_continues_after_batch_error(
        self,
        mock_tqdm,
        mock_config,
        sample_address_record
    ):
        """Test that parsing continues after a batch error (Requirement 7.4)."""
        pipeline = AddressConsolidationPipeline(mock_config)
        pipeline._total_records = 2
        
        # Create two records
        record1 = sample_address_record
        record2 = AddressRecord(
            addr_hash_key="hash456",
            addr_text="Another address",
            city_id="city2",
            pincode="400002",
            state_id="state1",
            zone_id="zone1",
            address_id="addr2",
            assigned_pickup_dlvd_geo_points="",
            assigned_pickup_dlvd_geo_points_count=0,
            raw_data={}
        )
        
        # Mock parse_batch to raise error on first call
        pipeline.llm_parser.parse_batch = Mock(
            side_effect=Exception("API error")
        )
        
        # Mock tqdm to process in one batch (both records together)
        mock_tqdm.return_value = [0]
        
        records = [record1, record2]
        results = pipeline._parse_addresses(records)
        
        # Should have results for both records despite batch error
        assert len(results) == 2
        # Both records should have error
        assert results[0][1].parse_success is False
        assert "Batch processing error" in results[0][1].parse_error
        assert results[1][1].parse_success is False
        assert "Batch processing error" in results[1][1].parse_error
    
    @patch('src.pipeline.print')
    def test_consolidate_addresses(
        self,
        mock_print,
        mock_config,
        sample_address_record,
        sample_parsed_address
    ):
        """Test address consolidation."""
        pipeline = AddressConsolidationPipeline(mock_config)
        
        # Mock consolidation engine
        mock_group = Mock(spec=ConsolidatedGroup)
        pipeline.consolidation_engine.consolidate = Mock(return_value=[mock_group])
        
        records_with_parsed = [(sample_address_record, sample_parsed_address)]
        groups = pipeline._consolidate_addresses(records_with_parsed)
        
        assert len(groups) == 1
        assert groups[0] == mock_group
        pipeline.consolidation_engine.consolidate.assert_called_once_with(records_with_parsed)
    
    @patch('src.pipeline.print')
    def test_write_output(self, mock_print, mock_config):
        """Test output writing."""
        pipeline = AddressConsolidationPipeline(mock_config)
        
        # Mock output writer
        pipeline.output_writer.write = Mock(return_value=10)
        
        mock_groups = [Mock(spec=ConsolidatedGroup)]
        pipeline._write_output(mock_groups)
        
        pipeline.output_writer.write.assert_called_once_with(mock_groups)
    
    def test_display_statistics(self, mock_config):
        """Test statistics display."""
        pipeline = AddressConsolidationPipeline(mock_config)
        pipeline._total_records = 10
        pipeline._failed_parses = 2
        
        # Mock statistics reporter
        mock_stats = Mock(spec=ConsolidationStats)
        pipeline.statistics_reporter.generate_stats = Mock(return_value=mock_stats)
        pipeline.statistics_reporter.display = Mock()
        
        mock_groups = [Mock(spec=ConsolidatedGroup)]
        pipeline._display_statistics(mock_groups)
        
        pipeline.statistics_reporter.generate_stats.assert_called_once_with(
            consolidated_groups=mock_groups,
            total_records=10,
            failed_parses=2
        )
        pipeline.statistics_reporter.display.assert_called_once_with(mock_stats)
    
    def test_display_statistics_handles_errors(self, mock_config):
        """Test that statistics errors don't crash the pipeline."""
        pipeline = AddressConsolidationPipeline(mock_config)
        
        # Mock statistics reporter to raise error
        pipeline.statistics_reporter.generate_stats = Mock(
            side_effect=Exception("Stats error")
        )
        
        # Should not raise exception
        mock_groups = [Mock(spec=ConsolidatedGroup)]
        pipeline._display_statistics(mock_groups)
    
    @patch('src.pipeline.tqdm')
    @patch.object(AddressConsolidationPipeline, '_display_statistics')
    @patch.object(AddressConsolidationPipeline, '_write_output')
    @patch.object(AddressConsolidationPipeline, '_consolidate_addresses')
    @patch.object(AddressConsolidationPipeline, '_parse_addresses')
    @patch.object(AddressConsolidationPipeline, '_read_csv')
    def test_unparseable_addresses_included_in_output(
        self,
        mock_read_csv,
        mock_parse_addresses,
        mock_consolidate,
        mock_write_output,
        mock_display_stats,
        mock_tqdm,
        mock_config,
        sample_address_record
    ):
        """Test that unparseable addresses are included in output (Requirement 7.3)."""
        # Create a failed parse
        failed_parsed = ParsedAddress(
            parse_success=False,
            parse_error="Invalid JSON"
        )
        
        # Setup mocks
        mock_read_csv.return_value = [sample_address_record]
        mock_parse_addresses.return_value = [(sample_address_record, failed_parsed)]
        mock_consolidate.return_value = [Mock(spec=ConsolidatedGroup)]
        
        # Run pipeline
        pipeline = AddressConsolidationPipeline(mock_config)
        pipeline.run()
        
        # Verify consolidate was called with the failed parse included
        call_args = mock_consolidate.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0][1].parse_success is False
