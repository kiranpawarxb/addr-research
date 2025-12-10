"""Comprehensive Comparison: Rule-Based Local (Libpostal) vs Shiprocket Parser

This script compares the rule-based Libpostal parser against the Shiprocket
fine-tuned IndicBERT parser on your dataset.

Generates detailed metrics, visualizations, and decision-making summary.
"""

import sys
import logging
import time
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import asdict
import csv
from datetime import datetime

from src.libpostal_parser import LibpostalParser
from src.shiprocket_parser import ShiprocketParser
from src.models import ParsedAddress

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class LibpostalShiprocketComparison:
    """Compare Libpostal (Rule-Based) and Shiprocket parsers."""
    
    def __init__(self):
        """Initialize comparison."""
        self.results = {
            'libpostal': [],
            'shiprocket': []
        }
        self.raw_addresses = []
        self.metrics = {}
        
    def load_addresses_from_csv(
        self, 
        csv_file: str, 
        address_column: str = None, 
        limit: int = None
    ) -> List[str]:
        """Load addresses from CSV file.
        
        Args:
            csv_file: Path to CSV file
            address_column: Name of column containing addresses (auto-detect if None)
            limit: Maximum number of addresses to load (None for all)
            
        Returns:
            List of address strings
        """
        logger.info(f"Loading addresses from {csv_file}...")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file, nrows=limit)
            logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
            
            # Auto-detect address column if not specified
            if address_column is None:
                # Look for common address column names
                possible_names = ['address', 'full_address', 'Address', 'FULL_ADDRESS', 
                                'customer_address', 'shipping_address']
                
                for name in possible_names:
                    if name in df.columns:
                        address_column = name
                        break
                
                if address_column is None:
                    # Use first column that looks like it contains addresses
                    for col in df.columns:
                        if df[col].dtype == 'object':  # String column
                            address_column = col
                            break
            
            if address_column not in df.columns:
                raise ValueError(f"Column '{address_column}' not found in CSV. Available: {list(df.columns)}")
            
            # Extract addresses
            addresses = df[address_column].dropna().astype(str).tolist()
            
            logger.info(f"Loaded {len(addresses)} addresses from column '{address_column}'")
            
            if limit and len(addresses) > limit:
                addresses = addresses[:limit]
                logger.info(f"Limited to {limit} addresses")
            
            self.raw_addresses = addresses
            return addresses
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def run_comparison(
        self, 
        addresses: List[str] = None,
        use_gpu: bool = False
    ) -> Dict[str, Any]:
        """Run comparison between both parsers.
        
        Args:
            addresses: List of addresses to parse (uses loaded addresses if None)
            use_gpu: Whether to use GPU for Shiprocket parser
            
        Returns:
            Dictionary with comparison results
        """
        if addresses is None:
            addresses = self.raw_addresses
        
        if not addresses:
            raise ValueError("No addresses to parse. Load addresses first.")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING COMPARISON: Libpostal vs Shiprocket")
        logger.info(f"Total addresses: {len(addresses)}")
        logger.info(f"GPU enabled: {use_gpu}")
        logger.info(f"{'='*80}\n")
        
        # Initialize parsers
        logger.info("Initializing parsers...")
        libpostal = LibpostalParser(batch_size=10)
        shiprocket = ShiprocketParser(batch_size=10, use_gpu=use_gpu)
        
        # Parse with Libpostal
        logger.info("\n--- LIBPOSTAL PARSER ---")
        start_time = time.time()
        libpostal_results = libpostal.parse_batch(addresses)
        libpostal_time = time.time() - start_time
        logger.info(f"Libpostal completed in {libpostal_time:.2f}s")
        
        # Parse with Shiprocket
        logger.info("\n--- SHIPROCKET PARSER ---")
        start_time = time.time()
        shiprocket_results = shiprocket.parse_batch(addresses)
        shiprocket_time = time.time() - start_time
        logger.info(f"Shiprocket completed in {shiprocket_time:.2f}s")
        
        # Store results
        self.results['libpostal'] = libpostal_results
        self.results['shiprocket'] = shiprocket_results
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(
            addresses,
            libpostal_results,
            shiprocket_results,
            libpostal_time,
            shiprocket_time
        )
        
        return self.metrics
    
    def _calculate_metrics(
        self,
        addresses: List[str],
        libpostal_results: List[ParsedAddress],
        shiprocket_results: List[ParsedAddress],
        libpostal_time: float,
        shiprocket_time: float
    ) -> Dict[str, Any]:
        """Calculate comparison metrics.
        
        Returns:
            Dictionary with detailed metrics
        """
        logger.info("\nCalculating metrics...")
        
        n = len(addresses)
        
        # Success rates
        libpostal_success = sum(1 for r in libpostal_results if r.parse_success)
        shiprocket_success = sum(1 for r in shiprocket_results if r.parse_success)
        
        # Field extraction rates
        fields = ['unit_number', 'society_name', 'landmark', 'road', 'sub_locality',
                 'locality', 'city', 'district', 'state', 'country', 'pin_code']
        
        libpostal_field_rates = {}
        shiprocket_field_rates = {}
        
        for field in fields:
            libpostal_field_rates[field] = sum(
                1 for r in libpostal_results 
                if r.parse_success and getattr(r, field, '').strip()
            ) / n * 100
            
            shiprocket_field_rates[field] = sum(
                1 for r in shiprocket_results 
                if r.parse_success and getattr(r, field, '').strip()
            ) / n * 100
        
        # Speed metrics
        libpostal_avg_time = (libpostal_time / n) * 1000  # ms per address
        shiprocket_avg_time = (shiprocket_time / n) * 1000  # ms per address
        
        # Agreement rate (how often both parsers extract the same value)
        agreement_rates = {}
        for field in fields:
            agreements = sum(
                1 for i in range(n)
                if libpostal_results[i].parse_success and shiprocket_results[i].parse_success
                and getattr(libpostal_results[i], field, '').strip().lower() == 
                    getattr(shiprocket_results[i], field, '').strip().lower()
            )
            total_both_success = sum(
                1 for i in range(n)
                if libpostal_results[i].parse_success and shiprocket_results[i].parse_success
            )
            agreement_rates[field] = (agreements / total_both_success * 100) if total_both_success > 0 else 0
        
        metrics = {
            'total_addresses': n,
            'libpostal': {
                'success_count': libpostal_success,
                'success_rate': libpostal_success / n * 100,
                'failed_count': n - libpostal_success,
                'total_time': libpostal_time,
                'avg_time_ms': libpostal_avg_time,
                'field_extraction_rates': libpostal_field_rates
            },
            'shiprocket': {
                'success_count': shiprocket_success,
                'success_rate': shiprocket_success / n * 100,
                'failed_count': n - shiprocket_success,
                'total_time': shiprocket_time,
                'avg_time_ms': shiprocket_avg_time,
                'field_extraction_rates': shiprocket_field_rates
            },
            'comparison': {
                'speed_ratio': shiprocket_avg_time / libpostal_avg_time if libpostal_avg_time > 0 else 0,
                'agreement_rates': agreement_rates,
                'avg_agreement': sum(agreement_rates.values()) / len(agreement_rates) if agreement_rates else 0
            }
        }
        
        return metrics
    
    def print_summary(self):
        """Print comprehensive comparison summary."""
        if not self.metrics:
            logger.warning("No metrics available. Run comparison first.")
            return
        
        m = self.metrics
        
        print("\n" + "="*80)
        print("PARSER COMPARISON SUMMARY: Rule-Based (Libpostal) vs Shiprocket")
        print("="*80)
        
        print(f"\n📊 DATASET")
        print(f"  Total Addresses: {m['total_addresses']}")
        
        print(f"\n⚡ PERFORMANCE")
        print(f"  Libpostal:")
        print(f"    - Total Time: {m['libpostal']['total_time']:.2f}s")
        print(f"    - Avg Time: {m['libpostal']['avg_time_ms']:.2f}ms per address")
        print(f"  Shiprocket:")
        print(f"    - Total Time: {m['shiprocket']['total_time']:.2f}s")
        print(f"    - Avg Time: {m['shiprocket']['avg_time_ms']:.2f}ms per address")
        print(f"  Speed Comparison:")
        print(f"    - Shiprocket is {m['comparison']['speed_ratio']:.1f}x {'SLOWER' if m['comparison']['speed_ratio'] > 1 else 'FASTER'} than Libpostal")
        
        print(f"\n✅ SUCCESS RATES")
        print(f"  Libpostal: {m['libpostal']['success_count']}/{m['total_addresses']} ({m['libpostal']['success_rate']:.1f}%)")
        print(f"  Shiprocket: {m['shiprocket']['success_count']}/{m['total_addresses']} ({m['shiprocket']['success_rate']:.1f}%)")
        
        print(f"\n📋 FIELD EXTRACTION RATES")
        print(f"  {'Field':<20} {'Libpostal':<15} {'Shiprocket':<15} {'Agreement':<15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
        
        fields = ['unit_number', 'society_name', 'landmark', 'road', 'sub_locality',
                 'locality', 'city', 'district', 'state', 'country', 'pin_code']
        
        for field in fields:
            lib_rate = m['libpostal']['field_extraction_rates'][field]
            ship_rate = m['shiprocket']['field_extraction_rates'][field]
            agree_rate = m['comparison']['agreement_rates'][field]
            print(f"  {field:<20} {lib_rate:>6.1f}%        {ship_rate:>6.1f}%        {agree_rate:>6.1f}%")
        
        print(f"\n🤝 OVERALL AGREEMENT")
        print(f"  Average Agreement Rate: {m['comparison']['avg_agreement']:.1f}%")
        
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print("="*80)
        
        # Generate recommendation
        self._print_recommendation(m)
        
        print("\n")
    
    def _print_recommendation(self, m: Dict[str, Any]):
        """Print recommendation based on metrics."""
        
        # Scoring factors
        lib_score = 0
        ship_score = 0
        
        # Success rate (40% weight)
        if m['libpostal']['success_rate'] > m['shiprocket']['success_rate']:
            lib_score += 40
        else:
            ship_score += 40
        
        # Speed (30% weight)
        if m['libpostal']['avg_time_ms'] < m['shiprocket']['avg_time_ms']:
            lib_score += 30
        else:
            ship_score += 30
        
        # Field extraction quality (30% weight)
        lib_avg_extraction = sum(m['libpostal']['field_extraction_rates'].values()) / len(m['libpostal']['field_extraction_rates'])
        ship_avg_extraction = sum(m['shiprocket']['field_extraction_rates'].values()) / len(m['shiprocket']['field_extraction_rates'])
        
        if lib_avg_extraction > ship_avg_extraction:
            lib_score += 30
        else:
            ship_score += 30
        
        print(f"\nScores (out of 100):")
        print(f"  Libpostal (Rule-Based): {lib_score}")
        print(f"  Shiprocket (ML-Based): {ship_score}")
        
        if lib_score > ship_score:
            print(f"\n✅ RECOMMENDED: Libpostal (Rule-Based Local Parser)")
            print(f"\nReasons:")
            if m['libpostal']['success_rate'] > m['shiprocket']['success_rate']:
                print(f"  • Higher success rate ({m['libpostal']['success_rate']:.1f}% vs {m['shiprocket']['success_rate']:.1f}%)")
            if m['libpostal']['avg_time_ms'] < m['shiprocket']['avg_time_ms']:
                print(f"  • Faster processing ({m['libpostal']['avg_time_ms']:.1f}ms vs {m['shiprocket']['avg_time_ms']:.1f}ms per address)")
            if lib_avg_extraction > ship_avg_extraction:
                print(f"  • Better field extraction ({lib_avg_extraction:.1f}% vs {ship_avg_extraction:.1f}% average)")
            print(f"  • No external dependencies (works offline)")
            print(f"  • Lower operational costs")
        else:
            print(f"\n✅ RECOMMENDED: Shiprocket (ML-Based Parser)")
            print(f"\nReasons:")
            if m['shiprocket']['success_rate'] > m['libpostal']['success_rate']:
                print(f"  • Higher success rate ({m['shiprocket']['success_rate']:.1f}% vs {m['libpostal']['success_rate']:.1f}%)")
            if m['shiprocket']['avg_time_ms'] < m['libpostal']['avg_time_ms']:
                print(f"  • Faster processing ({m['shiprocket']['avg_time_ms']:.1f}ms vs {m['libpostal']['avg_time_ms']:.1f}ms per address)")
            if ship_avg_extraction > lib_avg_extraction:
                print(f"  • Better field extraction ({ship_avg_extraction:.1f}% vs {lib_avg_extraction:.1f}% average)")
            print(f"  • Specifically trained for Indian addresses")
            print(f"  • May improve with more training data")
    
    def export_detailed_results(self, output_file: str = "libpostal_shiprocket_comparison.csv"):
        """Export detailed comparison results to CSV.
        
        Args:
            output_file: Path to output CSV file
        """
        if not self.results['libpostal'] or not self.results['shiprocket']:
            logger.warning("No results to export. Run comparison first.")
            return
        
        logger.info(f"Exporting detailed results to {output_file}...")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'raw_address',
                'lib_success', 'lib_unit', 'lib_society', 'lib_landmark', 'lib_road',
                'lib_sub_locality', 'lib_locality', 'lib_city', 'lib_district', 
                'lib_state', 'lib_country', 'lib_pin',
                'ship_success', 'ship_unit', 'ship_society', 'ship_landmark', 'ship_road',
                'ship_sub_locality', 'ship_locality', 'ship_city', 'ship_district',
                'ship_state', 'ship_country', 'ship_pin',
                'agreement_score'
            ]
            writer.writerow(header)
            
            # Data rows
            for i, raw_addr in enumerate(self.raw_addresses):
                lib_result = self.results['libpostal'][i]
                ship_result = self.results['shiprocket'][i]
                
                # Calculate agreement score for this address
                fields = ['unit_number', 'society_name', 'landmark', 'road', 'sub_locality',
                         'locality', 'city', 'district', 'state', 'country', 'pin_code']
                agreements = sum(
                    1 for field in fields
                    if getattr(lib_result, field, '').strip().lower() == 
                       getattr(ship_result, field, '').strip().lower()
                )
                agreement_score = (agreements / len(fields)) * 100
                
                row = [
                    raw_addr,
                    lib_result.parse_success,
                    lib_result.unit_number,
                    lib_result.society_name,
                    lib_result.landmark,
                    lib_result.road,
                    lib_result.sub_locality,
                    lib_result.locality,
                    lib_result.city,
                    lib_result.district,
                    lib_result.state,
                    lib_result.country,
                    lib_result.pin_code,
                    ship_result.parse_success,
                    ship_result.unit_number,
                    ship_result.society_name,
                    ship_result.landmark,
                    ship_result.road,
                    ship_result.sub_locality,
                    ship_result.locality,
                    ship_result.city,
                    ship_result.district,
                    ship_result.state,
                    ship_result.country,
                    ship_result.pin_code,
                    f"{agreement_score:.1f}"
                ]
                writer.writerow(row)
        
        logger.info(f"Detailed results exported to {output_file}")
    
    def export_summary_report(self, output_file: str = "LIBPOSTAL_SHIPROCKET_COMPARISON.md"):
        """Export summary report in markdown format.
        
        Args:
            output_file: Path to output markdown file
        """
        if not self.metrics:
            logger.warning("No metrics available. Run comparison first.")
            return
        
        logger.info(f"Exporting summary report to {output_file}...")
        
        m = self.metrics
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Parser Comparison Report: Libpostal vs Shiprocket\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Determine winner
            lib_score = 0
            ship_score = 0
            
            if m['libpostal']['success_rate'] > m['shiprocket']['success_rate']:
                lib_score += 40
            else:
                ship_score += 40
            
            if m['libpostal']['avg_time_ms'] < m['shiprocket']['avg_time_ms']:
                lib_score += 30
            else:
                ship_score += 30
            
            lib_avg_extraction = sum(m['libpostal']['field_extraction_rates'].values()) / len(m['libpostal']['field_extraction_rates'])
            ship_avg_extraction = sum(m['shiprocket']['field_extraction_rates'].values()) / len(m['shiprocket']['field_extraction_rates'])
            
            if lib_avg_extraction > ship_avg_extraction:
                lib_score += 30
            else:
                ship_score += 30
            
            winner = "Libpostal (Rule-Based)" if lib_score > ship_score else "Shiprocket (ML-Based)"
            f.write(f"**Recommended Parser:** {winner}\n\n")
            
            f.write("## Dataset\n\n")
            f.write(f"- Total Addresses Tested: {m['total_addresses']}\n\n")
            
            f.write("## Performance Comparison\n\n")
            f.write("| Metric | Libpostal | Shiprocket |\n")
            f.write("|--------|-----------|------------|\n")
            f.write(f"| Success Rate | {m['libpostal']['success_rate']:.1f}% | {m['shiprocket']['success_rate']:.1f}% |\n")
            f.write(f"| Total Time | {m['libpostal']['total_time']:.2f}s | {m['shiprocket']['total_time']:.2f}s |\n")
            f.write(f"| Avg Time/Address | {m['libpostal']['avg_time_ms']:.2f}ms | {m['shiprocket']['avg_time_ms']:.2f}ms |\n")
            f.write(f"| Speed Ratio | 1.0x | {m['comparison']['speed_ratio']:.1f}x |\n\n")
            
            f.write("## Field Extraction Rates\n\n")
            f.write("| Field | Libpostal | Shiprocket | Agreement |\n")
            f.write("|-------|-----------|------------|----------|\n")
            
            fields = ['unit_number', 'society_name', 'landmark', 'road', 'sub_locality',
                     'locality', 'city', 'district', 'state', 'country', 'pin_code']
            
            for field in fields:
                lib_rate = m['libpostal']['field_extraction_rates'][field]
                ship_rate = m['shiprocket']['field_extraction_rates'][field]
                agree_rate = m['comparison']['agreement_rates'][field]
                f.write(f"| {field} | {lib_rate:.1f}% | {ship_rate:.1f}% | {agree_rate:.1f}% |\n")
            
            f.write(f"\n**Average Agreement Rate:** {m['comparison']['avg_agreement']:.1f}%\n\n")
            
            f.write("## Detailed Analysis\n\n")
            f.write("### Libpostal (Rule-Based Local Parser)\n\n")
            f.write("**Strengths:**\n")
            f.write("- Works offline (no API calls)\n")
            f.write("- Fast processing\n")
            f.write("- No operational costs\n")
            f.write("- Trained on global OpenStreetMap data\n\n")
            f.write("**Weaknesses:**\n")
            f.write("- May not handle Indian-specific address formats as well\n")
            f.write("- Requires C library installation\n\n")
            
            f.write("### Shiprocket (ML-Based Parser)\n\n")
            f.write("**Strengths:**\n")
            f.write("- Specifically trained for Indian addresses\n")
            f.write("- Uses fine-tuned IndicBERT model\n")
            f.write("- Can improve with additional training\n\n")
            f.write("**Weaknesses:**\n")
            f.write("- Slower processing (ML inference)\n")
            f.write("- Requires model download and GPU for best performance\n")
            f.write("- Higher computational requirements\n\n")
            
            f.write("## Recommendation\n\n")
            
            if lib_score > ship_score:
                f.write("**Use Libpostal** for:\n")
                f.write("- High-volume processing where speed matters\n")
                f.write("- Offline/on-premise deployments\n")
                f.write("- Cost-sensitive applications\n")
                f.write("- When infrastructure for ML models is not available\n\n")
                f.write("**Consider Shiprocket** if:\n")
                f.write("- You need the absolute best accuracy for Indian addresses\n")
                f.write("- You have GPU infrastructure available\n")
                f.write("- Processing speed is not critical\n")
            else:
                f.write("**Use Shiprocket** for:\n")
                f.write("- Best accuracy on Indian addresses\n")
                f.write("- When you have GPU infrastructure\n")
                f.write("- When quality is more important than speed\n\n")
                f.write("**Consider Libpostal** if:\n")
                f.write("- You need faster processing\n")
                f.write("- You want offline capability\n")
                f.write("- You want to minimize operational costs\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("- `libpostal_shiprocket_comparison.csv` - Detailed row-by-row comparison\n")
            f.write("- `LIBPOSTAL_SHIPROCKET_COMPARISON.md` - This summary report\n")
        
        logger.info(f"Summary report exported to {output_file}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare Libpostal and Shiprocket parsers on your dataset"
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='export_customer_address_store_p0.csv',
        help='Path to CSV file with addresses'
    )
    parser.add_argument(
        '--column',
        type=str,
        default='addr_text',
        help='Name of column containing addresses'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of addresses to test (default: 100)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for Shiprocket parser'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize comparison
        comparison = LibpostalShiprocketComparison()
        
        # Load addresses
        addresses = comparison.load_addresses_from_csv(
            args.csv,
            address_column=args.column,
            limit=args.limit
        )
        
        # Run comparison
        metrics = comparison.run_comparison(addresses, use_gpu=args.gpu)
        
        # Print summary
        comparison.print_summary()
        
        # Export results
        comparison.export_detailed_results()
        comparison.export_summary_report()
        
        print("\n✅ Comparison complete!")
        print("📄 Check libpostal_shiprocket_comparison.csv for detailed results")
        print("📄 Check LIBPOSTAL_SHIPROCKET_COMPARISON.md for summary report")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()