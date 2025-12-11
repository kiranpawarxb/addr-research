#!/usr/bin/env python3
"""
Direct Shiprocket Processing for Full CSV

This script processes all addresses in the CSV file using Shiprocket parser directly
and generates the exact output requested: parsed addresses with hash_key, pincode, 
raw address, and PIN code summaries.

Usage:
    python process_full_csv_shiprocket.py

Output:
    - shiprocket_parsed_addresses_YYYYMMDD_HHMMSS.csv: All parsed addresses
    - pincode_locality_society_summary_YYYYMMDD_HHMMSS.csv: Summary by PIN code
"""

import sys
import os
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, 'src')

from src.local_llm_parser import LocalLLMParser


def setup_logging():
    """Set up logging for the script."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'shiprocket_processing_{timestamp}.log')
        ]
    )


def load_csv_data(csv_file: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    
    print(f"📂 Loading CSV file: {csv_file}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    print(f"✅ Loaded {len(df):,} rows")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['addr_hash_key', 'addr_text']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out empty addresses
    df_clean = df.dropna(subset=['addr_text'])
    df_clean = df_clean[df_clean['addr_text'].str.strip() != '']
    
    print(f"✅ Found {len(df_clean):,} valid addresses (removed {len(df) - len(df_clean):,} empty)")
    
    return df_clean


def process_addresses_with_shiprocket(df: pd.DataFrame) -> List[Dict]:
    """Process all addresses using Shiprocket IndicBERT parser with GPU acceleration."""
    
    print(f"\n🔧 Initializing Shiprocket IndicBERT parser with GPU support...")
    
    # Initialize Shiprocket parser with GPU support
    from src.shiprocket_parser import ShiprocketParser
    parser = ShiprocketParser(
        batch_size=20,  # Smaller batches for GPU memory efficiency
        use_gpu=True    # Enable GPU acceleration
    )
    
    print(f"🚀 Processing {len(df):,} addresses with Shiprocket IndicBERT parser (GPU-accelerated)...")
    
    results = []
    batch_size = 20  # Smaller batches for GPU memory efficiency
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches", unit="batch"):
        batch_df = df.iloc[i:i + batch_size]
        
        # Extract addresses for this batch
        addresses = batch_df['addr_text'].tolist()
        
        try:
            # Parse batch
            parsed_results = parser.parse_batch(addresses)
            
            # Combine with original data
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                parsed = parsed_results[idx]
                
                result = {
                    # Original data
                    'hash_key': row.get('addr_hash_key', ''),
                    'raw_address': row.get('addr_text', ''),
                    'original_pincode': row.get('pincode', ''),
                    
                    # Parsed fields
                    'unit_number': parsed.unit_number or '',
                    'society_name': parsed.society_name or '',
                    'landmark': parsed.landmark or '',
                    'road': parsed.road or '',
                    'sub_locality': parsed.sub_locality or '',
                    'locality': parsed.locality or '',
                    'city': parsed.city or '',
                    'district': parsed.district or '',
                    'state': parsed.state or '',
                    'country': parsed.country or 'India',
                    'parsed_pincode': parsed.pin_code or '',
                    
                    # Metadata
                    'parse_success': parsed.parse_success,
                    'parse_error': parsed.parse_error or '',
                    
                    # Additional original columns (preserve all data)
                    **{col: row.get(col, '') for col in df.columns if col not in ['addr_hash_key', 'addr_text', 'pincode']}
                }
                
                results.append(result)
                
        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size + 1}: {e}")
            
            # Add failed results for this batch
            for _, row in batch_df.iterrows():
                result = {
                    'hash_key': row.get('addr_hash_key', ''),
                    'raw_address': row.get('addr_text', ''),
                    'original_pincode': row.get('pincode', ''),
                    'unit_number': '', 'society_name': '', 'landmark': '', 'road': '',
                    'sub_locality': '', 'locality': '', 'city': '', 'district': '',
                    'state': '', 'country': 'India', 'parsed_pincode': '',
                    'parse_success': False,
                    'parse_error': f'Batch processing error: {str(e)}',
                    **{col: row.get(col, '') for col in df.columns if col not in ['addr_hash_key', 'addr_text', 'pincode']}
                }
                results.append(result)
    
    # Get final statistics
    stats = parser.get_statistics()
    
    print(f"\n📊 Processing Statistics:")
    print(f"   Total processed: {stats['total_parsed'] + stats['total_failed']:,}")
    print(f"   Successful: {stats['total_parsed']:,} ({stats['success_rate_percent']}%)")
    print(f"   Failed: {stats['total_failed']:,}")
    print(f"   Retries: {stats['total_retries']:,}")
    
    return results


def create_output_csv(results: List[Dict], timestamp: str) -> str:
    """Create main output CSV with all parsed addresses."""
    
    print(f"\n📝 Creating output CSV...")
    
    # Create DataFrame
    df_output = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'hash_key', 'raw_address', 'original_pincode',
        'unit_number', 'society_name', 'landmark', 'road', 
        'sub_locality', 'locality', 'city', 'district', 'state', 'country',
        'parsed_pincode', 'parse_success', 'parse_error'
    ]
    
    # Add any additional columns that weren't in the standard set
    additional_cols = [col for col in df_output.columns if col not in column_order]
    final_columns = column_order + additional_cols
    
    # Reorder DataFrame
    df_output = df_output[final_columns]
    
    # Save to CSV
    output_file = f'shiprocket_parsed_addresses_{timestamp}.csv'
    df_output.to_csv(output_file, index=False)
    
    print(f"✅ Output CSV saved: {output_file}")
    print(f"   Rows: {len(df_output):,}")
    print(f"   Columns: {len(df_output.columns)}")
    
    return output_file


def analyze_pincode_summaries(results: List[Dict]) -> Dict[str, Dict]:
    """Analyze localities and societies by PIN code."""
    
    print(f"\n🔍 Analyzing PIN code summaries...")
    
    pincode_stats = defaultdict(lambda: {
        'total_addresses': 0,
        'successful_parses': 0,
        'localities': set(),
        'societies': set(),
        'sample_addresses': []
    })
    
    # Process all results
    for result in results:
        # Use parsed PIN code if available, otherwise original
        pin_code = result['parsed_pincode'] or result['original_pincode'] or 'UNKNOWN'
        
        stats = pincode_stats[pin_code]
        stats['total_addresses'] += 1
        
        if result['parse_success']:
            stats['successful_parses'] += 1
            
            # Add unique localities and societies (case-insensitive)
            if result['locality']:
                stats['localities'].add(result['locality'].strip().title())
            if result['society_name']:
                stats['societies'].add(result['society_name'].strip().title())
        
        # Keep sample addresses (first 3 per PIN code)
        if len(stats['sample_addresses']) < 3:
            stats['sample_addresses'].append({
                'hash_key': result['hash_key'],
                'address': result['raw_address'][:100] + '...' if len(result['raw_address']) > 100 else result['raw_address'],
                'society': result['society_name'],
                'locality': result['locality']
            })
    
    # Convert to final format
    final_stats = {}
    for pin_code, stats in pincode_stats.items():
        final_stats[pin_code] = {
            'pin_code': pin_code,
            'total_addresses': stats['total_addresses'],
            'successful_parses': stats['successful_parses'],
            'success_rate': (stats['successful_parses'] / stats['total_addresses'] * 100) if stats['total_addresses'] > 0 else 0,
            'distinct_localities': len(stats['localities']),
            'distinct_societies': len(stats['societies']),
            'localities_list': sorted(list(stats['localities'])),
            'societies_list': sorted(list(stats['societies'])),
            'sample_addresses': stats['sample_addresses']
        }
    
    return final_stats


def create_pincode_summary_csv(pincode_stats: Dict[str, Dict], timestamp: str) -> str:
    """Create PIN code summary CSV."""
    
    print(f"\n📊 Creating PIN code summary CSV...")
    
    summary_data = []
    
    for pin_code, stats in pincode_stats.items():
        # Create main summary row
        row = {
            'pin_code': pin_code,
            'total_addresses': stats['total_addresses'],
            'successful_parses': stats['successful_parses'],
            'success_rate_percent': round(stats['success_rate'], 1),
            'distinct_localities_count': stats['distinct_localities'],
            'distinct_societies_count': stats['distinct_societies'],
            'localities_sample': '; '.join(stats['localities_list'][:5]) + ('...' if len(stats['localities_list']) > 5 else ''),
            'societies_sample': '; '.join(stats['societies_list'][:5]) + ('...' if len(stats['societies_list']) > 5 else ''),
            'all_localities': '; '.join(stats['localities_list']),
            'all_societies': '; '.join(stats['societies_list'])
        }
        
        summary_data.append(row)
    
    # Sort by total addresses (descending)
    summary_data.sort(key=lambda x: x['total_addresses'], reverse=True)
    
    # Create DataFrame and save
    df_summary = pd.DataFrame(summary_data)
    summary_file = f'pincode_locality_society_summary_{timestamp}.csv'
    df_summary.to_csv(summary_file, index=False)
    
    print(f"✅ PIN code summary saved: {summary_file}")
    print(f"   PIN codes analyzed: {len(summary_data):,}")
    
    return summary_file


def print_summary_statistics(pincode_stats: Dict[str, Dict], total_addresses: int):
    """Print summary statistics to console."""
    
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 60)
    
    # Overall stats
    total_pins = len(pincode_stats)
    total_successful = sum(stats['successful_parses'] for stats in pincode_stats.values())
    total_localities = sum(stats['distinct_localities'] for stats in pincode_stats.values())
    total_societies = sum(stats['distinct_societies'] for stats in pincode_stats.values())
    
    print(f"Total Addresses Processed: {total_addresses:,}")
    print(f"Total PIN Codes Found: {total_pins:,}")
    print(f"Overall Success Rate: {total_successful/total_addresses*100:.1f}%")
    print(f"Total Distinct Localities: {total_localities:,}")
    print(f"Total Distinct Societies: {total_societies:,}")
    print(f"Average Localities per PIN: {total_localities/total_pins:.1f}")
    print(f"Average Societies per PIN: {total_societies/total_pins:.1f}")
    
    print(f"\n🔝 TOP 10 PIN CODES BY ADDRESS COUNT:")
    print("-" * 60)
    
    # Sort by address count
    sorted_pins = sorted(pincode_stats.items(), key=lambda x: x[1]['total_addresses'], reverse=True)
    
    print(f"{'PIN Code':<10} {'Addresses':<10} {'Success%':<10} {'Localities':<12} {'Societies':<10}")
    print("-" * 60)
    
    for pin_code, stats in sorted_pins[:10]:
        print(f"{pin_code:<10} {stats['total_addresses']:<10} {stats['success_rate']:<10.1f} {stats['distinct_localities']:<12} {stats['distinct_societies']:<10}")
    
    print(f"\n🏘️  TOP 5 PIN CODES BY LOCALITY DIVERSITY:")
    print("-" * 60)
    
    # Sort by locality count
    sorted_by_localities = sorted(pincode_stats.items(), key=lambda x: x[1]['distinct_localities'], reverse=True)
    
    for i, (pin_code, stats) in enumerate(sorted_by_localities[:5], 1):
        print(f"{i}. PIN {pin_code}: {stats['distinct_localities']} localities, {stats['total_addresses']} addresses")
        if stats['localities_list']:
            print(f"   Sample localities: {', '.join(stats['localities_list'][:3])}")
    
    print(f"\n🏢 TOP 5 PIN CODES BY SOCIETY DIVERSITY:")
    print("-" * 60)
    
    # Sort by society count
    sorted_by_societies = sorted(pincode_stats.items(), key=lambda x: x[1]['distinct_societies'], reverse=True)
    
    for i, (pin_code, stats) in enumerate(sorted_by_societies[:5], 1):
        print(f"{i}. PIN {pin_code}: {stats['distinct_societies']} societies, {stats['total_addresses']} addresses")
        if stats['societies_list']:
            print(f"   Sample societies: {', '.join(stats['societies_list'][:3])}")


def main():
    """Main execution function."""
    
    print("🚀 Shiprocket Full CSV Processing")
    print("=" * 60)
    
    # Set up logging
    setup_logging()
    
    # Configuration
    csv_file = 'export_customer_address_store_p0.csv'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Step 1: Load CSV data
        df = load_csv_data(csv_file)
        
        # Step 2: Process addresses with Shiprocket
        results = process_addresses_with_shiprocket(df)
        
        # Step 3: Create output CSV
        output_file = create_output_csv(results, timestamp)
        
        # Step 4: Analyze PIN code summaries
        pincode_stats = analyze_pincode_summaries(results)
        
        # Step 5: Create PIN code summary CSV
        summary_file = create_pincode_summary_csv(pincode_stats, timestamp)
        
        # Step 6: Print summary statistics
        print_summary_statistics(pincode_stats, len(results))
        
        # Final summary
        print(f"\n🎉 PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"📁 Generated Files:")
        print(f"   • Main results: {output_file}")
        print(f"   • PIN summary: {summary_file}")
        print(f"   • Processing log: shiprocket_processing_{timestamp}.log")
        print()
        print("✅ Ready for analysis!")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Check the log file for detailed error information.")


if __name__ == "__main__":
    main()