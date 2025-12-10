"""Tests for the Statistics Reporter component."""

import pytest
from src.statistics_reporter import StatisticsReporter
from src.models import ConsolidatedGroup, ConsolidationStats, AddressRecord, ParsedAddress


class TestStatisticsReporter:
    """Test suite for StatisticsReporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reporter = StatisticsReporter()
    
    def test_generate_stats_basic(self):
        """Test basic statistics generation with simple data."""
        # Create test groups
        group1 = ConsolidatedGroup(
            group_id="g1",
            society_name="Test Society",
            pin_code="123456",
            records=[],
            record_count=5
        )
        group2 = ConsolidatedGroup(
            group_id="g2",
            society_name="Another Society",
            pin_code="654321",
            records=[],
            record_count=3
        )
        
        groups = [group1, group2]
        total_records = 8
        failed_parses = 2
        
        stats = self.reporter.generate_stats(groups, total_records, failed_parses)
        
        assert stats.total_records == 8
        assert stats.total_groups == 2
        assert stats.avg_records_per_group == 4.0
        assert stats.largest_group_size == 5
        assert stats.largest_group_name == "Test Society"
        assert stats.parse_success_count == 6
        assert stats.parse_failure_count == 2
    
    def test_generate_stats_empty_groups(self):
        """Test statistics generation with no groups."""
        groups = []
        total_records = 0
        failed_parses = 0
        
        stats = self.reporter.generate_stats(groups, total_records, failed_parses)
        
        assert stats.total_records == 0
        assert stats.total_groups == 0
        assert stats.avg_records_per_group == 0.0
        assert stats.largest_group_size == 0
        assert stats.largest_group_name == ""
        assert stats.match_percentage == 0.0
        assert stats.parse_success_count == 0
        assert stats.parse_failure_count == 0
    
    def test_generate_stats_single_group(self):
        """Test statistics generation with a single group."""
        group = ConsolidatedGroup(
            group_id="g1",
            society_name="Single Society",
            pin_code="111111",
            records=[],
            record_count=10
        )
        
        groups = [group]
        total_records = 10
        failed_parses = 1
        
        stats = self.reporter.generate_stats(groups, total_records, failed_parses)
        
        assert stats.total_records == 10
        assert stats.total_groups == 1
        assert stats.avg_records_per_group == 10.0
        assert stats.largest_group_size == 10
        assert stats.largest_group_name == "Single Society"
        assert stats.parse_success_count == 9
        assert stats.parse_failure_count == 1
    
    def test_generate_stats_with_unmatched_group(self):
        """Test statistics generation with unmatched records."""
        group1 = ConsolidatedGroup(
            group_id="g1",
            society_name="Matched Society",
            pin_code="123456",
            records=[],
            record_count=7
        )
        unmatched_group = ConsolidatedGroup(
            group_id="g2",
            society_name="",
            pin_code="",
            records=[],
            record_count=3
        )
        
        groups = [group1, unmatched_group]
        total_records = 10
        failed_parses = 0
        
        stats = self.reporter.generate_stats(groups, total_records, failed_parses)
        
        assert stats.total_records == 10
        assert stats.total_groups == 2
        assert stats.unmatched_count == 3
        assert stats.match_percentage == 70.0  # 7 out of 10 matched
    
    def test_generate_stats_all_unmatched(self):
        """Test statistics when all records are unmatched."""
        unmatched_group = ConsolidatedGroup(
            group_id="g1",
            society_name="",
            pin_code="",
            records=[],
            record_count=5
        )
        
        groups = [unmatched_group]
        total_records = 5
        failed_parses = 0
        
        stats = self.reporter.generate_stats(groups, total_records, failed_parses)
        
        assert stats.total_records == 5
        assert stats.unmatched_count == 5
        assert stats.match_percentage == 0.0
    
    def test_generate_stats_all_matched(self):
        """Test statistics when all records are matched."""
        group1 = ConsolidatedGroup(
            group_id="g1",
            society_name="Society A",
            pin_code="111111",
            records=[],
            record_count=3
        )
        group2 = ConsolidatedGroup(
            group_id="g2",
            society_name="Society B",
            pin_code="222222",
            records=[],
            record_count=2
        )
        
        groups = [group1, group2]
        total_records = 5
        failed_parses = 0
        
        stats = self.reporter.generate_stats(groups, total_records, failed_parses)
        
        assert stats.total_records == 5
        assert stats.unmatched_count == 0
        assert stats.match_percentage == 100.0
    
    def test_generate_stats_largest_group_identification(self):
        """Test that the largest group is correctly identified."""
        groups = [
            ConsolidatedGroup("g1", "Small", "111111", [], 2),
            ConsolidatedGroup("g2", "Large", "222222", [], 10),
            ConsolidatedGroup("g3", "Medium", "333333", [], 5),
        ]
        
        stats = self.reporter.generate_stats(groups, 17, 0)
        
        assert stats.largest_group_size == 10
        assert stats.largest_group_name == "Large"
    
    def test_generate_stats_average_calculation(self):
        """Test that average records per group is calculated correctly."""
        groups = [
            ConsolidatedGroup("g1", "A", "111111", [], 4),
            ConsolidatedGroup("g2", "B", "222222", [], 6),
            ConsolidatedGroup("g3", "C", "333333", [], 5),
        ]
        
        stats = self.reporter.generate_stats(groups, 15, 0)
        
        assert stats.avg_records_per_group == 5.0
    
    def test_generate_stats_parse_counts(self):
        """Test that parse success/failure counts are correct."""
        groups = [
            ConsolidatedGroup("g1", "Society", "111111", [], 10),
        ]
        
        total_records = 10
        failed_parses = 3
        
        stats = self.reporter.generate_stats(groups, total_records, failed_parses)
        
        assert stats.parse_success_count == 7
        assert stats.parse_failure_count == 3
        assert stats.parse_success_count + stats.parse_failure_count == total_records
    
    def test_display_basic(self, capsys):
        """Test that display outputs statistics in readable format."""
        stats = ConsolidationStats(
            total_records=100,
            total_groups=10,
            avg_records_per_group=10.0,
            largest_group_size=25,
            largest_group_name="Big Society",
            match_percentage=90.0,
            parse_success_count=95,
            parse_failure_count=5,
            unmatched_count=10
        )
        
        self.reporter.display(stats)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that key information is displayed
        assert "ADDRESS CONSOLIDATION STATISTICS" in output
        assert "Total Records Processed: 100" in output
        assert "Total Consolidated Groups: 10" in output
        assert "Average Records per Group: 10.00" in output
        assert "Big Society" in output
        assert "Size: 25 records" in output
        assert "Match Percentage: 90.00%" in output
        assert "Successfully Parsed: 95" in output
        assert "Failed to Parse: 5" in output
        assert "Parse Success Rate: 95.00%" in output
    
    def test_display_empty_stats(self, capsys):
        """Test display with empty statistics."""
        stats = ConsolidationStats(
            total_records=0,
            total_groups=0,
            avg_records_per_group=0.0,
            largest_group_size=0,
            largest_group_name="",
            match_percentage=0.0,
            parse_success_count=0,
            parse_failure_count=0,
            unmatched_count=0
        )
        
        self.reporter.display(stats)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Total Records Processed: 0" in output
        assert "Total Consolidated Groups: 0" in output
        assert "N/A" in output  # Should show N/A for some fields
    
    def test_display_formatting(self, capsys):
        """Test that large numbers are formatted with commas."""
        stats = ConsolidationStats(
            total_records=1000000,
            total_groups=5000,
            avg_records_per_group=200.0,
            largest_group_size=10000,
            largest_group_name="Huge Society",
            match_percentage=99.5,
            parse_success_count=995000,
            parse_failure_count=5000,
            unmatched_count=5000
        )
        
        self.reporter.display(stats)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that numbers are formatted with commas
        assert "1,000,000" in output
        assert "5,000" in output
        assert "10,000" in output
