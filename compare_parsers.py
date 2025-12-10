"""Parser Comparison Tool

Compares the performance of different parsers:
1. Rule-based Local Parser
2. IndicBERT Parser
3. OpenAI Parser (optional)

Generates detailed comparison metrics and visualizations.
"""

import sys
import logging
import time
import pandas as pd
from typing import List, Dict, Any
from dataclasses import asdict

from src.local_llm_parser import LocalLLMParser
from src.indicbert_parser import IndicBERTParser
from src.models import ParsedAddress

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ParserComparison:
    """Compare multiple address parsers."""
    
    def __init__(self):
        """Initialize parsers for comparison."""
        self.parsers = {}
        self.results = {}
        
    def add_parser(self, name: str, parser):
        """Add a parser to compare.
        
        Args:
            name: Display name for the parser
            parser: Parser instance (must have parse_address method)
        """
        self.parsers[name] = parser
        logger.info(f"Added parser: {name}")
    
    def compare(self, addresses: List[str]) -> Dict[str, Any]:
        """Compare all parsers on the given addresses.
        
        Args:
            addresses: List of address strings to parse
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(self.parsers)} parsers on {len(addresses)} addresses")
        
        results = {}
        
        for parser_name, parser in self.parsers.items():
            logger.info(f"\nTesting {parser_name}...")
            
            # Time the parsing
            start_time = time.time()
            parsed_results = []
            
            for i, address in enumerate(addresses):
                try:
                    parsed = parser.parse_address(address)
                    parsed_results.append(parsed)
                except Exception as e:
                    logger.error(f"Error parsing address {i+1} with {parser_name}: {e}")
                    parsed_results.append(ParsedAddress(
                        parse_success=False,
                        parse_error=str(e)
                    ))
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Calculate metrics
            success_count = sum(1 for p in parsed_results if p.parse_success)
            failed_count = len(parsed_results) - success_count
            
            # Field extraction rates
            field_counts = {
                'unit_number': sum(1 for p in parsed_results if p.unit_number),
                'society_name': sum(1 for p in parsed_results if p.society_name),
                'landmark': sum(1 for p in parsed_results if p.landmark),
                'road': sum(1 for p in parsed_results if p.road),
                'sub_locality': sum(1 for p in parsed_results if p.sub_locality),
                'locality': sum(1 for p in parsed_results if p.locality),
                'city': sum(1 for p in parsed_results if p.city),
                'district': sum(1 for p in parsed_results if p.district),
                'state': sum(1 for p in parsed_results if p.state),
                'pin_code': sum(1 for p in parsed_results if p.pin_code),
            }
            
            results[parser_name] = {
                'parsed_results': parsed_results,
                'success_count': success_count,
                'failed_count': failed_count,
                'success_rate': success_count / len(addresses) * 100,
                'elapsed_time': elapsed,
                'avg_time_per_address': elapsed / len(addresses),
                'field_counts': field_counts,
                'field_extraction_rates': {
                    field: count / len(addresses) * 100
                    for field, count in field_counts.items()
                }
            }
            
            logger.info(f"  Success: {success_count}/{len(addresses)} ({results[parser_name]['success_rate']:.1f}%)")
            logger.info(f"  Time: {elapsed:.2f}s ({results[parser_name]['avg_time_per_address']*1000:.1f}ms per address)")
        
        self.results = results
        return results
    
    def print_summary(self):
        """Print comparison summary."""
        if not self.results:
            logger.warning("No results to display. Run compare() first.")
            return
        
        print("\n" + "="*80)
        print("PARSER COMPARISON SUMMARY")
        print("="*80)
        
        # Overall metrics
        print("\n1. OVERALL PERFORMANCE")
        print("-"*80)
        print(f"{'Parser':<25} {'Success Rate':<15} {'Avg Time':<15} {'Total Time':<15}")
        print("-"*80)
        
        for parser_name, result in self.results.items():
            print(f"{parser_name:<25} "
                  f"{result['success_rate']:>6.1f}%        "
                  f"{result['avg_time_per_address']*1000:>6.1f}ms       "
                  f"{result['elapsed_time']:>6.2f}s")
        
        # Field extraction rates
        print("\n2. FIELD EXTRACTION RATES (%)")
        print("-"*80)
        
        # Get all field names
        field_names = list(next(iter(self.results.values()))['field_extraction_rates'].keys())
        
        # Print header
        print(f"{'Field':<20}", end="")
        for parser_name in self.results.keys():
            print(f"{parser_name:<20}", end="")
        print()
        print("-"*80)
        
        # Print each field
        for field in field_names:
            print(f"{field:<20}", end="")
            for parser_name in self.results.keys():
                rate = self.results[parser_name]['field_extraction_rates'][field]
                print(f"{rate:>6.1f}%            ", end="")
            print()
        
        # Winner analysis
        print("\n3. WINNER ANALYSIS")
        print("-"*80)
        
        # Best success rate
        best_success = max(self.results.items(), key=lambda x: x[1]['success_rate'])
        print(f"Best Success Rate:  {best_success[0]} ({best_success[1]['success_rate']:.1f}%)")
        
        # Fastest
        fastest = min(self.results.items(), key=lambda x: x[1]['avg_time_per_address'])
        print(f"Fastest:            {fastest[0]} ({fastest[1]['avg_time_per_address']*1000:.1f}ms per address)")
        
        # Best field extraction
        for field in field_names:
            best_field = max(
                self.results.items(),
                key=lambda x: x[1]['field_extraction_rates'][field]
            )
            rate = best_field[1]['field_extraction_rates'][field]
            if rate > 0:
                print(f"Best {field:<15}: {best_field[0]} ({rate:.1f}%)")
    
    def export_detailed_comparison(self, output_file: str = "parser_comparison.csv"):
        """Export detailed comparison to CSV.
        
        Args:
            output_file: Path to output CSV file
        """
        if not self.results:
            logger.warning("No results to export. Run compare() first.")
            return
        
        # Prepare data for export
        rows = []
        
        # Get addresses from first parser
        first_parser = next(iter(self.results.keys()))
        num_addresses = len(self.results[first_parser]['parsed_results'])
        
        for i in range(num_addresses):
            row = {'address_index': i + 1}
            
            for parser_name, result in self.results.items():
                parsed = result['parsed_results'][i]
                prefix = parser_name.replace(' ', '_').lower()
                
                row[f'{prefix}_success'] = parsed.parse_success
                row[f'{prefix}_unit_number'] = parsed.unit_number
                row[f'{prefix}_society_name'] = parsed.society_name
                row[f'{prefix}_landmark'] = parsed.landmark
                row[f'{prefix}_road'] = parsed.road
                row[f'{prefix}_sub_locality'] = parsed.sub_locality
                row[f'{prefix}_locality'] = parsed.locality
                row[f'{prefix}_city'] = parsed.city
                row[f'{prefix}_district'] = parsed.district
                row[f'{prefix}_state'] = parsed.state
                row[f'{prefix}_pin_code'] = parsed.pin_code
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Detailed comparison exported to: {output_file}")
        print(f"\n✓ Detailed comparison exported to: {output_file}")


def main():
    """Main comparison function."""
    
    # Sample Pune addresses for testing
    test_addresses = [
        "Flat 301, Kumar Paradise, Kalyani Nagar, Pune, Maharashtra 411006",
        "A-204, Amanora Park Town, Hadapsar, Pune 411028",
        "Bungalow 12, Koregaon Park, Near Osho Ashram, Pune 411001",
        "Office 505, Cerebrum IT Park, Kumar Road, Pune, Maharashtra 411001",
        "201, Magarpatta City, Hadapsar, Pune 411013",
        "Flat 102, Seasons Apartment, Baner Road, Baner, Pune 411045",
        "B-Wing 404, Pride Purple Park, Wakad, Pune, Maharashtra 411057",
        "Villa 7, Amanora Neo Towers, Hadapsar, Near Phoenix Mall, Pune 411028",
        "Shop 12, FC Road, Shivajinagar, Pune 411004",
        "3rd Floor, Panchshil Tech Park, Yerwada, Pune, Maharashtra 411006"
    ]
    
    print("="*80)
    print("ADDRESS PARSER COMPARISON TOOL")
    print("="*80)
    print(f"\nComparing parsers on {len(test_addresses)} Pune addresses\n")
    
    # Initialize comparison
    comparison = ParserComparison()
    
    # Add parsers
    print("Initializing parsers...")
    
    # 1. Rule-based Local Parser
    try:
        local_parser = LocalLLMParser(batch_size=1)
        comparison.add_parser("Rule-Based Local", local_parser)
    except Exception as e:
        logger.error(f"Failed to initialize Local Parser: {e}")
    
    # 2. IndicBERT Parser
    try:
        print("\nLoading IndicBERT model (this may take a few minutes on first run)...")
        indicbert_parser = IndicBERTParser(batch_size=1, use_gpu=False)
        comparison.add_parser("IndicBERT", indicbert_parser)
    except Exception as e:
        logger.error(f"Failed to initialize IndicBERT Parser: {e}")
        logger.error("Make sure transformers and torch are installed:")
        logger.error("  pip install transformers torch")
    
    # 3. Libpostal Parser (optional)
    try:
        from src.libpostal_parser import LibpostalParser
        print("\nInitializing Libpostal parser...")
        libpostal_parser = LibpostalParser(batch_size=1)
        comparison.add_parser("Libpostal", libpostal_parser)
    except ImportError:
        logger.warning("Libpostal not available (requires C library installation)")
        logger.warning("See: https://github.com/openvenues/libpostal")
    except Exception as e:
        logger.error(f"Failed to initialize Libpostal Parser: {e}")
    
    # 4. Shiprocket Parser (fine-tuned for Indian addresses)
    try:
        from src.shiprocket_parser import ShiprocketParser
        print("\nLoading Shiprocket model (fine-tuned for Indian addresses)...")
        shiprocket_parser = ShiprocketParser(batch_size=1, use_gpu=False)
        comparison.add_parser("Shiprocket", shiprocket_parser)
    except ImportError:
        logger.error("Failed to initialize Shiprocket Parser")
        logger.error("Make sure transformers and torch are installed:")
        logger.error("  pip install transformers torch")
    except Exception as e:
        logger.error(f"Failed to initialize Shiprocket Parser: {e}")
    
    # Run comparison
    print("\n" + "-"*80)
    print("Starting comparison...")
    print("-"*80)
    
    try:
        results = comparison.compare(test_addresses)
        
        # Print summary
        comparison.print_summary()
        
        # Export detailed results
        comparison.export_detailed_comparison("parser_comparison.csv")
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print("\nReview the detailed results in: parser_comparison.csv")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
        sys.exit(0)
