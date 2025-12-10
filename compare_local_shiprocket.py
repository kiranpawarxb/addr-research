"""Comprehensive Comparison: Rule-Based Local Parser vs Shiprocket Parser

This script compares the rule-based local parser against the Shiprocket
fine-tuned IndicBERT parser on your dataset.

Generates detailed metrics, visualizations, and decision-making summary.
"""

import sys
import logging
import time
import pandas as pd
from typing import List, Dict, Any
import csv
from datetime import datetime

from src.local_llm_parser import LocalLLMParser
from src.shiprocket_parser import ShiprocketParser
from src.models import ParsedAddress

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class LocalShiprocketComparison:
    """Compare Local Rule-Based and Shiprocket parsers."""
    
    def __init__(self):
        """Initialize comparison."""
        self.results = {
            'local': [],
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
        """Load addresses from CSV file."""
        logger.info(f"Loading addresses from {csv_file}...")
        
        try:
            df = pd.read_csv(csv_file, nrows=limit)
            logger.info(f"Loaded CSV with {len(df)} rows")
            
            if address_column is None:
                possible_names = ['address', 'addr_text', 'full_address', 'Address']
                for name in possible_names:
                    if name in df.columns:
                        address_column = name
                        break
            
            if address_column not in df.columns:
                raise ValueError(f"Column '{address_column}' not found")
            
            addresses = df[address_column].dropna().astype(str).tolist()
            
            if limit and len(addresses) > limit:
                addresses = addresses[:limit]
            
            self.raw_addresses = addresses
            logger.info(f"Loaded {len(addresses)} addresses from '{address_column}'")
            return addresses
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def run_comparison(
        self, 
        addresses: List[str] = None,
        use_gpu: bool = False
    ) -> Dict[str, Any]:
        """Run comparison between both parsers."""
        if addresses is None:
            addresses = self.raw_addresses
        
        if not addresses:
            raise ValueError("No addresses to parse")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING COMPARISON: Local Rule-Based vs Shiprocket")
        logger.info(f"Total addresses: {len(addresses)}")
        logger.info(f"GPU enabled: {use_gpu}")
        logger.info(f"{'='*80}\n")
        
        # Initialize parsers
        logger.info("Initializing parsers...")
        local_parser = LocalLLMParser(batch_size=10)
        shiprocket_parser = ShiprocketParser(batch_size=10, use_gpu=use_gpu)
        
        # Parse with Local
        logger.info("\n--- LOCAL RULE-BASED PARSER ---")
        start_time = time.time()
        local_results = local_parser.parse_batch(addresses)
        local_time = time.time() - start_time
        logger.info(f"Local parser completed in {local_time:.2f}s")
        
        # Parse with Shiprocket
        logger.info("\n--- SHIPROCKET PARSER ---")
        start_time = time.time()
        shiprocket_results = shiprocket_parser.parse_batch(addresses)
        shiprocket_time = time.time() - start_time
        logger.info(f"Shiprocket completed in {shiprocket_time:.2f}s")
        
        # Store results
        self.results['local'] = local_results
        self.results['shiprocket'] = shiprocket_results
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(
            addresses,
            local_results,
            shiprocket_results,
            local_time,
            shiprocket_time
        )
        
        return self.metrics
    
    def _calculate_metrics(
        self,
        addresses: List[str],
        local_results: List[ParsedAddress],
        shiprocket_results: List[ParsedAddress],
        local_time: float,
        shiprocket_time: float
    ) -> Dict[str, Any]:
        """Calculate comparison metrics."""
        logger.info("\nCalculating metrics...")
        
        n = len(addresses)
        
        # Success rates
        local_success = sum(1 for r in local_results if r.parse_success)
        shiprocket_success = sum(1 for r in shiprocket_results if r.parse_success)
        
        # Field extraction rates
        fields = ['unit_number', 'society_name', 'landmark', 'road', 'sub_locality',
                 'locality', 'city', 'district', 'state', 'country', 'pin_code']
        
        local_field_rates = {}
        shiprocket_field_rates = {}
        
        for field in fields:
            local_field_rates[field] = sum(
                1 for r in local_results 
                if r.parse_success and getattr(r, field, '').strip()
            ) / n * 100
            
            shiprocket_field_rates[field] = sum(
                1 for r in shiprocket_results 
                if r.parse_success and getattr(r, field, '').strip()
            ) / n * 100
        
        # Speed metrics
        local_avg_time = (local_time / n) * 1000  # ms per address
        shiprocket_avg_time = (shiprocket_time / n) * 1000  # ms per address
        
        # Agreement rate
        agreement_rates = {}
        for field in fields:
            agreements = sum(
                1 for i in range(n)
                if local_results[i].parse_success and shiprocket_results[i].parse_success
                and getattr(local_results[i], field, '').strip().lower() == 
                    getattr(shiprocket_results[i], field, '').strip().lower()
            )
            total_both_success = sum(
                1 for i in range(n)
                if local_results[i].parse_success and shiprocket_results[i].parse_success
            )
            agreement_rates[field] = (agreements / total_both_success * 100) if total_both_success > 0 else 0
        
        metrics = {
            'total_addresses': n,
            'local': {
                'success_count': local_success,
                'success_rate': local_success / n * 100,
                'failed_count': n - local_success,
                'total_time': local_time,
                'avg_time_ms': local_avg_time,
                'field_extraction_rates': local_field_rates
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
                'speed_ratio': shiprocket_avg_time / local_avg_time if local_avg_time > 0 else 0,
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
        print("PARSER COMPARISON SUMMARY: Rule-Based Local vs Shiprocket")
        print("="*80)
        
        print(f"\n📊 DATASET")
        print(f"  Total Addresses: {m['total_addresses']}")
        
        print(f"\n⚡ PERFORMANCE")
        print(f"  Local Rule-Based:")
        print(f"    - Total Time: {m['local']['total_time']:.2f}s")
        print(f"    - Avg Time: {m['local']['avg_time_ms']:.2f}ms per address")
        print(f"  Shiprocket:")
        print(f"    - Total Time: {m['shiprocket']['total_time']:.2f}s")
        print(f"    - Avg Time: {m['shiprocket']['avg_time_ms']:.2f}ms per address")
        print(f"  Speed Comparison:")
        print(f"    - Shiprocket is {m['comparison']['speed_ratio']:.1f}x {'SLOWER' if m['comparison']['speed_ratio'] > 1 else 'FASTER'} than Local")
        
        print(f"\n✅ SUCCESS RATES")
        print(f"  Local: {m['local']['success_count']}/{m['total_addresses']} ({m['local']['success_rate']:.1f}%)")
        print(f"  Shiprocket: {m['shiprocket']['success_count']}/{m['total_addresses']} ({m['shiprocket']['success_rate']:.1f}%)")
        
        print(f"\n📋 FIELD EXTRACTION RATES")
        print(f"  {'Field':<20} {'Local':<15} {'Shiprocket':<15} {'Agreement':<15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
        
        fields = ['unit_number', 'society_name', 'landmark', 'road', 'sub_locality',
                 'locality', 'city', 'district', 'state', 'country', 'pin_code']
        
        for field in fields:
            local_rate = m['local']['field_extraction_rates'][field]
            ship_rate = m['shiprocket']['field_extraction_rates'][field]
            agree_rate = m['comparison']['agreement_rates'][field]
            print(f"  {field:<20} {local_rate:>6.1f}%        {ship_rate:>6.1f}%        {agree_rate:>6.1f}%")
        
        print(f"\n🤝 OVERALL AGREEMENT")
        print(f"  Average Agreement Rate: {m['comparison']['avg_agreement']:.1f}%")
        
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print("="*80)
        
        self._print_recommendation(m)
        print("\n")
    
    def _print_recommendation(self, m: Dict[str, Any]):
        """Print recommendation based on metrics."""
        
        # Scoring factors
        local_score = 0
        ship_score = 0
        
        # Success rate (40% weight)
        if m['local']['success_rate'] > m['shiprocket']['success_rate']:
            local_score += 40
        else:
            ship_score += 40
        
        # Speed (30% weight)
        if m['local']['avg_time_ms'] < m['shiprocket']['avg_time_ms']:
            local_score += 30
        else:
            ship_score += 30
        
        # Field extraction quality (30% weight)
        local_avg_extraction = sum(m['local']['field_extraction_rates'].values()) / len(m['local']['field_extraction_rates'])
        ship_avg_extraction = sum(m['shiprocket']['field_extraction_rates'].values()) / len(m['shiprocket']['field_extraction_rates'])
        
        if local_avg_extraction > ship_avg_extraction:
            local_score += 30
        else:
            ship_score += 30
        
        print(f"\nScores (out of 100):")
        print(f"  Local Rule-Based: {local_score}")
        print(f"  Shiprocket ML-Based: {ship_score}")
        
        if local_score > ship_score:
            print(f"\n✅ RECOMMENDED: Local Rule-Based Parser")
            print(f"\nReasons:")
            if m['local']['success_rate'] > m['shiprocket']['success_rate']:
                print(f"  • Higher success rate ({m['local']['success_rate']:.1f}% vs {m['shiprocket']['success_rate']:.1f}%)")
            if m['local']['avg_time_ms'] < m['shiprocket']['avg_time_ms']:
                print(f"  • Faster processing ({m['local']['avg_time_ms']:.1f}ms vs {m['shiprocket']['avg_time_ms']:.1f}ms per address)")
            if local_avg_extraction > ship_avg_extraction:
                print(f"  • Better field extraction ({local_avg_extraction:.1f}% vs {ship_avg_extraction:.1f}% average)")
            print(f"  • No model downloads required")
            print(f"  • Lower memory footprint")
            print(f"  • Instant startup time")
        else:
            print(f"\n✅ RECOMMENDED: Shiprocket ML-Based Parser")
            print(f"\nReasons:")
            if m['shiprocket']['success_rate'] > m['local']['success_rate']:
                print(f"  • Higher success rate ({m['shiprocket']['success_rate']:.1f}% vs {m['local']['success_rate']:.1f}%)")
            if m['shiprocket']['avg_time_ms'] < m['local']['avg_time_ms']:
                print(f"  • Faster processing ({m['shiprocket']['avg_time_ms']:.1f}ms vs {m['local']['avg_time_ms']:.1f}ms per address)")
            if ship_avg_extraction > local_avg_extraction:
                print(f"  • Better field extraction ({ship_avg_extraction:.1f}% vs {local_avg_extraction:.1f}% average)")
            print(f"  • Specifically trained for Indian addresses")
            print(f"  • May improve with more training data")
    
    def export_detailed_results(self, output_file: str = "local_shiprocket_comparison.csv"):
        """Export detailed comparison results to CSV."""
        if not self.results['local'] or not self.results['shiprocket']:
            logger.warning("No results to export")
            return
        
        logger.info(f"Exporting detailed results to {output_file}...")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            header = [
                'raw_address',
                'local_success', 'local_unit', 'local_society', 'local_landmark', 'local_road',
                'local_sub_locality', 'local_locality', 'local_city', 'local_district', 
                'local_state', 'local_country', 'local_pin',
                'ship_success', 'ship_unit', 'ship_society', 'ship_landmark', 'ship_road',
                'ship_sub_locality', 'ship_locality', 'ship_city', 'ship_district',
                'ship_state', 'ship_country', 'ship_pin',
                'agreement_score'
            ]
            writer.writerow(header)
            
            for i, raw_addr in enumerate(self.raw_addresses):
                local_result = self.results['local'][i]
                ship_result = self.results['shiprocket'][i]
                
                fields = ['unit_number', 'society_name', 'landmark', 'road', 'sub_locality',
                         'locality', 'city', 'district', 'state', 'country', 'pin_code']
                agreements = sum(
                    1 for field in fields
                    if getattr(local_result, field, '').strip().lower() == 
                       getattr(ship_result, field, '').strip().lower()
                )
                agreement_score = (agreements / len(fields)) * 100
                
                row = [
                    raw_addr,
                    local_result.parse_success,
                    local_result.unit_number,
                    local_result.society_name,
                    local_result.landmark,
                    local_result.road,
                    local_result.sub_locality,
                    local_result.locality,
                    local_result.city,
                    local_result.district,
                    local_result.state,
                    local_result.country,
                    local_result.pin_code,
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
    
    def export_summary_report(self, output_file: str = "LOCAL_SHIPROCKET_COMPARISON.md"):
        """Export summary report in markdown format."""
        if not self.metrics:
            logger.warning("No metrics available")
            return
        
        logger.info(f"Exporting summary report to {output_file}...")
        
        m = self.metrics
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Parser Comparison Report: Local Rule-Based vs Shiprocket\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Determine winner
            local_score = 0
            ship_score = 0
            
            if m['local']['success_rate'] > m['shiprocket']['success_rate']:
                local_score += 40
            else:
                ship_score += 40
            
            if m['local']['avg_time_ms'] < m['shiprocket']['avg_time_ms']:
                local_score += 30
            else:
                ship_score += 30
            
            local_avg_extraction = sum(m['local']['field_extraction_rates'].values()) / len(m['local']['field_extraction_rates'])
            ship_avg_extraction = sum(m['shiprocket']['field_extraction_rates'].values()) / len(m['shiprocket']['field_extraction_rates'])
            
            if local_avg_extraction > ship_avg_extraction:
                local_score += 30
            else:
                ship_score += 30
            
            winner = "Local Rule-Based" if local_score > ship_score else "Shiprocket ML-Based"
            f.write(f"**Recommended Parser:** {winner}\n\n")
            
            f.write("## Dataset\n\n")
            f.write(f"- Total Addresses Tested: {m['total_addresses']}\n\n")
            
            f.write("## Performance Comparison\n\n")
            f.write("| Metric | Local Rule-Based | Shiprocket ML-Based |\n")
            f.write("|--------|------------------|---------------------|\n")
            f.write(f"| Success Rate | {m['local']['success_rate']:.1f}% | {m['shiprocket']['success_rate']:.1f}% |\n")
            f.write(f"| Total Time | {m['local']['total_time']:.2f}s | {m['shiprocket']['total_time']:.2f}s |\n")
            f.write(f"| Avg Time/Address | {m['local']['avg_time_ms']:.2f}ms | {m['shiprocket']['avg_time_ms']:.2f}ms |\n")
            f.write(f"| Speed Ratio | 1.0x | {m['comparison']['speed_ratio']:.1f}x |\n\n")
            
            f.write("## Field Extraction Rates\n\n")
            f.write("| Field | Local | Shiprocket | Agreement |\n")
            f.write("|-------|-------|------------|----------|\n")
            
            fields = ['unit_number', 'society_name', 'landmark', 'road', 'sub_locality',
                     'locality', 'city', 'district', 'state', 'country', 'pin_code']
            
            for field in fields:
                local_rate = m['local']['field_extraction_rates'][field]
                ship_rate = m['shiprocket']['field_extraction_rates'][field]
                agree_rate = m['comparison']['agreement_rates'][field]
                f.write(f"| {field} | {local_rate:.1f}% | {ship_rate:.1f}% | {agree_rate:.1f}% |\n")
            
            f.write(f"\n**Average Agreement Rate:** {m['comparison']['avg_agreement']:.1f}%\n\n")
            
            f.write("## Detailed Analysis\n\n")
            f.write("### Local Rule-Based Parser\n\n")
            f.write("**Strengths:**\n")
            f.write("- Extremely fast (<1ms per address)\n")
            f.write("- No model downloads or dependencies\n")
            f.write("- Instant startup time\n")
            f.write("- Low memory footprint\n")
            f.write("- Works offline\n\n")
            f.write("**Weaknesses:**\n")
            f.write("- May struggle with highly unstructured addresses\n")
            f.write("- Limited adaptability to new patterns\n\n")
            
            f.write("### Shiprocket ML-Based Parser\n\n")
            f.write("**Strengths:**\n")
            f.write("- Specifically trained for Indian addresses\n")
            f.write("- Uses fine-tuned IndicBERT model\n")
            f.write("- Better handling of variations\n")
            f.write("- Can improve with additional training\n\n")
            f.write("**Weaknesses:**\n")
            f.write("- Slower processing (ML inference)\n")
            f.write("- Requires ~500MB model download\n")
            f.write("- Higher memory requirements\n")
            f.write("- Longer startup time\n\n")
            
            f.write("## Recommendation\n\n")
            
            if local_score > ship_score:
                f.write("**Use Local Rule-Based Parser** for:\n")
                f.write("- High-volume processing where speed matters\n")
                f.write("- Production deployments with well-formatted addresses\n")
                f.write("- Resource-constrained environments\n")
                f.write("- When instant startup is required\n\n")
                f.write("**Consider Shiprocket** if:\n")
                f.write("- You need better handling of unstructured addresses\n")
                f.write("- You have GPU infrastructure available\n")
                f.write("- Processing speed is not critical\n")
            else:
                f.write("**Use Shiprocket ML-Based Parser** for:\n")
                f.write("- Best accuracy on Indian addresses\n")
                f.write("- Handling highly unstructured input\n")
                f.write("- When you have GPU infrastructure\n\n")
                f.write("**Consider Local Rule-Based** if:\n")
                f.write("- You need faster processing\n")
                f.write("- You want minimal dependencies\n")
                f.write("- You want to minimize resource usage\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("- `local_shiprocket_comparison.csv` - Detailed row-by-row comparison\n")
            f.write("- `LOCAL_SHIPROCKET_COMPARISON.md` - This summary report\n")
        
        logger.info(f"Summary report exported to {output_file}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare Local Rule-Based and Shiprocket parsers"
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
        comparison = LocalShiprocketComparison()
        
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
        print("📄 Check local_shiprocket_comparison.csv for detailed results")
        print("📄 Check LOCAL_SHIPROCKET_COMPARISON.md for summary report")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
