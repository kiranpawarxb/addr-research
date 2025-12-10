"""Statistics Reporter component for the Address Consolidation System."""

from typing import List
from src.models import ConsolidatedGroup, ConsolidationStats


class StatisticsReporter:
    """Generate and display consolidation statistics.
    
    Calculates comprehensive metrics about the consolidation process including
    group counts, averages, match percentages, and parsing success rates.
    """
    
    def generate_stats(
        self,
        consolidated_groups: List[ConsolidatedGroup],
        total_records: int,
        failed_parses: int
    ) -> ConsolidationStats:
        """Generate comprehensive statistics about the consolidation.
        
        Args:
            consolidated_groups: List of all consolidated groups
            total_records: Total number of address records processed
            failed_parses: Number of addresses that failed to parse
            
        Returns:
            ConsolidationStats object with all calculated metrics
            
        Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("Generating consolidation statistics...")
        
        # Calculate total groups (Requirement 5.1)
        total_groups = len(consolidated_groups)
        logger.debug(f"Total groups: {total_groups}")
        
        # Calculate average records per group (Requirement 5.2)
        if total_groups > 0:
            avg_records_per_group = total_records / total_groups
        else:
            avg_records_per_group = 0.0
        logger.debug(f"Average records per group: {avg_records_per_group:.2f}")
        
        # Identify largest group (Requirement 5.3)
        largest_group_size = 0
        largest_group_name = ""
        unmatched_count = 0
        
        for group in consolidated_groups:
            if group.record_count > largest_group_size:
                largest_group_size = group.record_count
                largest_group_name = group.society_name
            
            # Track unmatched records
            if group.society_name == "" or group.society_name.lower() == "unmatched":
                unmatched_count = group.record_count
        
        logger.debug(f"Largest group: {largest_group_name} with {largest_group_size} records")
        logger.debug(f"Unmatched records: {unmatched_count}")
        
        # Calculate match percentage (Requirement 5.4)
        matched_records = total_records - unmatched_count
        if total_records > 0:
            match_percentage = (matched_records / total_records) * 100
        else:
            match_percentage = 0.0
        logger.debug(f"Match percentage: {match_percentage:.2f}%")
        
        # Calculate parse success/failure counts (Requirement 5.5)
        parse_success_count = total_records - failed_parses
        parse_failure_count = failed_parses
        logger.debug(f"Parse success: {parse_success_count}, Parse failures: {parse_failure_count}")
        
        logger.info("Statistics generation complete")
        
        return ConsolidationStats(
            total_records=total_records,
            total_groups=total_groups,
            avg_records_per_group=avg_records_per_group,
            largest_group_size=largest_group_size,
            largest_group_name=largest_group_name,
            match_percentage=match_percentage,
            parse_success_count=parse_success_count,
            parse_failure_count=parse_failure_count,
            unmatched_count=unmatched_count
        )
    
    def display(self, stats: ConsolidationStats) -> None:
        """Display statistics in a readable format.
        
        Args:
            stats: ConsolidationStats object to display
            
        Validates: Requirement 5.5
        """
        print("\n" + "=" * 60)
        print("ADDRESS CONSOLIDATION STATISTICS")
        print("=" * 60)
        
        print(f"\nTotal Records Processed: {stats.total_records:,}")
        print(f"Total Consolidated Groups: {stats.total_groups:,}")
        
        if stats.total_groups > 0:
            print(f"Average Records per Group: {stats.avg_records_per_group:.2f}")
        else:
            print("Average Records per Group: N/A (no groups)")
        
        print(f"\nLargest Group:")
        print(f"  Name: {stats.largest_group_name if stats.largest_group_name else 'N/A'}")
        print(f"  Size: {stats.largest_group_size:,} records")
        
        print(f"\nMatching Statistics:")
        print(f"  Matched Records: {stats.total_records - stats.unmatched_count:,}")
        print(f"  Unmatched Records: {stats.unmatched_count:,}")
        print(f"  Match Percentage: {stats.match_percentage:.2f}%")
        
        print(f"\nParsing Statistics:")
        print(f"  Successfully Parsed: {stats.parse_success_count:,}")
        print(f"  Failed to Parse: {stats.parse_failure_count:,}")
        
        if stats.total_records > 0:
            parse_success_rate = (stats.parse_success_count / stats.total_records) * 100
            print(f"  Parse Success Rate: {parse_success_rate:.2f}%")
        else:
            print(f"  Parse Success Rate: N/A")
        
        print("=" * 60 + "\n")
