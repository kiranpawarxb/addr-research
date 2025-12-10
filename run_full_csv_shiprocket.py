#!/usr/bin/env python3
"""
Full CSV Processing with Shiprocket Parser

This script processes all addresses in the CSV file using the Shiprocket parser
and generates comprehensive output with PIN code summaries.

Usage:
    python run_full_csv_shiprocket.py

Output:
    - shiprocket_full_results_YYYYMMDD_HHMMSS.csv: All parsed addresses
    - pincode_summary_YYYYMMDD_HHMMSS.csv: Summary by PIN code
    - processing_report_YYYYMMDD_HHMMSS.txt: Detailed processing report
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Add src to path
sys.path.insert(0, 'src')

from src.pipeline import AddressConsolidationPipeline
from src.config_loader import ConfigLoader
from src.models import ConsolidatedGroup


def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'full_csv_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def create_shiprocket_config():
    """Create configuration for Shiprocket processing."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config_dict = {
        'input': {
            'file_path': 'export_customer_address_store_p0.csv',
            'required_columns': ['addr_text', 'pincode', 'city_id']
        },
        'llm': {
            'parser_type': 'shiprocket',
            'batch_size': 50,  # Process in larger batches for efficiency
            'use_gpu': False,  # Set to True if you have GPU available
            'max_retries': 3,
            'timeout_seconds': 30
        },
        'consolidation': {
            'fuzzy_matching': True,
            'similarity_threshold': 0.85,
            'normalize_society_names': True
        },
        'output': {
            'file_path': f'shiprocket_full_results_{timestamp}.csv',
            'include_statistics': True
        },
        'logging': {
            'level': 'INFO',
            'file_path': f'shiprocket_processing_{timestamp}.log'
        }
    }
    
    return config_dict, timestamp


def analyze_pincode_distribution(consolidated_groups: List[ConsolidatedGroup]) -> Dict[str, Dict]:
    """Analyze distribution of localities and societies by PIN code."""
    
    print("\n🔍 Analyzing PIN code distribution...")
    
    pincode_stats = defaultdict(lambda: {
        'total_addresses': 0,
        'successful_parses': 0,
        'unique_localities': set(),
        'unique_societies': set(),
        'addresses': []
    })
    
    # Process all groups
    for group in consolidated_groups:
        for address_record, parsed_address in group.records:
            pin_code = parsed_address.pin_code or address_record.pincode or 'UNKNOWN'
            
            stats = pincode_stats[pin_code]
            stats['total_addresses'] += 1
            
            if parsed_address.parse_success:
                stats['successful_parses'] += 1
                
                # Add unique localities and societies
                if parsed_address.locality:
                    stats['unique_localities'].add(parsed_address.locality.lower().strip())
                if parsed_address.society_name:
                    stats['unique_societies'].add(parsed_address.society_name.lower().strip())
            
            # Store address info for detailed analysis
            stats['addresses'].append({
                'hash_key': address_record.addr_hash_key,
                'raw_address': address_record.addr_text,
                'society': parsed_address.society_name,
                'locality': parsed_address.locality,
                'parse_success': parsed_address.parse_success
            })
    
    # Convert sets to counts and create final summary
    final_stats = {}
    for pin_code, stats in pincode_stats.items():
        final_stats[pin_code] = {
            'total_addresses': stats['total_addresses'],
            'successful_parses': stats['successful_parses'],
            'success_rate': (stats['successful_parses'] / stats['total_addresses'] * 100) if stats['total_addresses'] > 0 else 0,
            'unique_localities_count': len(stats['unique_localities']),
            'unique_societies_count': len(stats['unique_societies']),
            'unique_localities': sorted(list(stats['unique_localities'])),
            'unique_societies': sorted(list(stats['unique_societies'])),
            'addresses': stats['addresses']
        }
    
    return final_stats


def create_pincode_summary_csv(pincode_stats: Dict[str, Dict], timestamp: str):
    """Create CSV summary of PIN code statistics."""
    
    print(f"\n📊 Creating PIN code summary CSV...")
    
    summary_data = []
    
    for pin_code, stats in pincode_stats.items():
        summary_data.append({
            'pin_code': pin_code,
            'total_addresses': stats['total_addresses'],
            'successful_parses': stats['successful_parses'],
            'success_rate_percent': round(stats['success_rate'], 1),
            'unique_localities_count': stats['unique_localities_count'],
            'unique_societies_count': stats['unique_societies_count'],
            'localities_list': '; '.join(stats['unique_localities'][:10]) + ('...' if len(stats['unique_localities']) > 10 else ''),
            'societies_list': '; '.join(stats['unique_societies'][:10]) + ('...' if len(stats['unique_societies']) > 10 else ''),
            'top_localities': '; '.join(stats['unique_localities'][:5]),
            'top_societies': '; '.join(stats['unique_societies'][:5])
        })
    
    # Sort by total addresses (descending)
    summary_data.sort(key=lambda x: x['total_addresses'], reverse=True)
    
    # Create DataFrame and save
    df_summary = pd.DataFrame(summary_data)
    summary_file = f'pincode_summary_{timestamp}.csv'
    df_summary.to_csv(summary_file, index=False)
    
    print(f"✅ PIN code summary saved to: {summary_file}")
    return summary_file


def create_detailed_report(pincode_stats: Dict[str, Dict], timestamp: str, total_processed: int):
    """Create detailed processing report."""
    
    print(f"\n📋 Creating detailed processing report...")
    
    report_file = f'processing_report_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SHIPROCKET FULL CSV PROCESSING REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Addresses Processed: {total_processed:,}\n")
        f.write(f"Total PIN Codes Found: {len(pincode_stats):,}\n")
        f.write("\n")
        
        # Overall statistics
        total_successful = sum(stats['successful_parses'] for stats in pincode_stats.values())
        total_localities = sum(stats['unique_localities_count'] for stats in pincode_stats.values())
        total_societies = sum(stats['unique_societies_count'] for stats in pincode_stats.values())
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Successful Parses: {total_successful:,} ({total_successful/total_processed*100:.1f}%)\n")
        f.write(f"Total Unique Localities: {total_localities:,}\n")
        f.write(f"Total Unique Societies: {total_societies:,}\n")
        f.write(f"Average Localities per PIN: {total_localities/len(pincode_stats):.1f}\n")
        f.write(f"Average Societies per PIN: {total_societies/len(pincode_stats):.1f}\n")
        f.write("\n")
        
        # Top PIN codes by address count
        sorted_pins = sorted(pincode_stats.items(), key=lambda x: x[1]['total_addresses'], reverse=True)
        
        f.write("TOP 20 PIN CODES BY ADDRESS COUNT\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'PIN Code':<10} {'Addresses':<10} {'Success%':<10} {'Localities':<12} {'Societies':<12}\n")
        f.write("-" * 50 + "\n")
        
        for pin_code, stats in sorted_pins[:20]:
            f.write(f"{pin_code:<10} {stats['total_addresses']:<10} {stats['success_rate']:<10.1f} {stats['unique_localities_count']:<12} {stats['unique_societies_count']:<12}\n")
        
        f.write("\n")
        
        # PIN codes with most localities
        sorted_by_localities = sorted(pincode_stats.items(), key=lambda x: x[1]['unique_localities_count'], reverse=True)
        
        f.write("TOP 10 PIN CODES BY LOCALITY DIVERSITY\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'PIN Code':<10} {'Localities':<12} {'Addresses':<10} {'Success%':<10}\n")
        f.write("-" * 50 + "\n")
        
        for pin_code, stats in sorted_by_localities[:10]:
            f.write(f"{pin_code:<10} {stats['unique_localities_count']:<12} {stats['total_addresses']:<10} {stats['success_rate']:<10.1f}\n")
        
        f.write("\n")
        
        # PIN codes with most societies
        sorted_by_societies = sorted(pincode_stats.items(), key=lambda x: x[1]['unique_societies_count'], reverse=True)
        
        f.write("TOP 10 PIN CODES BY SOCIETY DIVERSITY\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'PIN Code':<10} {'Societies':<12} {'Addresses':<10} {'Success%':<10}\n")
        f.write("-" * 50 + "\n")
        
        for pin_code, stats in sorted_by_societies[:10]:
            f.write(f"{pin_code:<10} {stats['unique_societies_count']:<12} {stats['total_addresses']:<10} {stats['success_rate']:<10.1f}\n")
        
        f.write("\n")
        
        # Quality analysis
        high_success_pins = [pin for pin, stats in pincode_stats.items() if stats['success_rate'] > 90 and stats['total_addresses'] >= 10]
        low_success_pins = [pin for pin, stats in pincode_stats.items() if stats['success_rate'] < 50 and stats['total_addresses'] >= 10]
        
        f.write("QUALITY ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(f"PIN codes with >90% success rate (≥10 addresses): {len(high_success_pins)}\n")
        f.write(f"PIN codes with <50% success rate (≥10 addresses): {len(low_success_pins)}\n")
        f.write("\n")
        
        if low_success_pins:
            f.write("PIN CODES WITH LOW SUCCESS RATES:\n")
            for pin in low_success_pins[:10]:
                stats = pincode_stats[pin]
                f.write(f"  {pin}: {stats['success_rate']:.1f}% success ({stats['successful_parses']}/{stats['total_addresses']} addresses)\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Detailed report saved to: {report_file}")
    return report_file


def main():
    """Main execution function."""
    
    print("🚀 Starting Full CSV Processing with Shiprocket Parser")
    print("=" * 80)
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config_dict, timestamp = create_shiprocket_config()
        
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_from_dict(config_dict)
        
        print(f"📁 Input file: {config.input.file_path}")
        print(f"📁 Output file: {config.output.file_path}")
        print(f"🔧 Parser: Shiprocket IndicBERT")
        print(f"📦 Batch size: {config.llm.batch_size}")
        print()
        
        # Check if input file exists
        if not os.path.exists(config.input.file_path):
            print(f"❌ Error: Input file not found: {config.input.file_path}")
            return
        
        # Initialize and run pipeline
        pipeline = AddressConsolidationPipeline(config)
        
        print("🏃 Running address consolidation pipeline...")
        pipeline.run()
        
        # Load the output CSV to analyze PIN code distribution
        print(f"\n📖 Loading output CSV for PIN code analysis...")
        df_output = pd.read_csv(config.output.file_path)
        
        print(f"✅ Loaded {len(df_output):,} processed addresses")
        
        # Reconstruct consolidated groups from output for analysis
        # Group by group_id to recreate the structure
        groups_by_id = df_output.groupby('group_id')
        
        # Create mock consolidated groups for analysis
        mock_groups = []
        for group_id, group_df in groups_by_id:
            # Create mock records for analysis
            mock_records = []
            for _, row in group_df.iterrows():
                # Create mock address record
                mock_address = type('AddressRecord', (), {
                    'addr_hash_key': row.get('addr_hash_key', ''),
                    'addr_text': row.get('addr_text', ''),
                    'pincode': row.get('pincode', '')
                })()
                
                # Create mock parsed address
                mock_parsed = type('ParsedAddress', (), {
                    'society_name': row.get('SN', ''),
                    'locality': row.get('LOC', ''),
                    'pin_code': row.get('PIN', ''),
                    'parse_success': True  # Assume success if in output
                })()
                
                mock_records.append((mock_address, mock_parsed))
            
            # Create mock consolidated group
            mock_group = type('ConsolidatedGroup', (), {
                'group_id': group_id,
                'records': mock_records
            })()
            
            mock_groups.append(mock_group)
        
        # Analyze PIN code distribution
        pincode_stats = analyze_pincode_distribution(mock_groups)
        
        # Create PIN code summary CSV
        summary_file = create_pincode_summary_csv(pincode_stats, timestamp)
        
        # Create detailed report
        report_file = create_detailed_report(pincode_stats, timestamp, len(df_output))
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"📊 Total addresses processed: {len(df_output):,}")
        print(f"📍 Total PIN codes found: {len(pincode_stats):,}")
        print(f"🏢 Total unique societies: {sum(stats['unique_societies_count'] for stats in pincode_stats.values()):,}")
        print(f"🏘️  Total unique localities: {sum(stats['unique_localities_count'] for stats in pincode_stats.values()):,}")
        print()
        print("📁 Generated Files:")
        print(f"   • Main results: {config.output.file_path}")
        print(f"   • PIN summary: {summary_file}")
        print(f"   • Detailed report: {report_file}")
        print()
        
        # Show top PIN codes
        sorted_pins = sorted(pincode_stats.items(), key=lambda x: x[1]['total_addresses'], reverse=True)
        print("🔝 Top 5 PIN codes by address count:")
        for i, (pin_code, stats) in enumerate(sorted_pins[:5], 1):
            print(f"   {i}. {pin_code}: {stats['total_addresses']:,} addresses, {stats['unique_societies_count']} societies, {stats['unique_localities_count']} localities")
        
        print("\n✅ All files ready for analysis!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Check the log file for detailed error information.")


if __name__ == "__main__":
    main()