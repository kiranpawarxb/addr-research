#!/usr/bin/env python3
"""
Test Shiprocket Processing for 1,000 Addresses with GPU

This script processes 1,000 addresses using Shiprocket parser with GPU acceleration
to get accurate timing estimates before running the full dataset.

Usage:
    python test_1k_addresses_gpu.py
"""

import sys
import os
import pandas as pd
import logging
import time
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, 'src')

def setup_logging():
    """Set up logging for the script."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'test_1k_processing_{timestamp}.log')
        ]
    )

def load_1k_sample_data(csv_file: str) -> pd.DataFrame:
    """Load 1,000 sample addresses from CSV."""
    
    print(f"📂 Loading 1,000 sample addresses from: {csv_file}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Load 1,200 rows to ensure we get 1,000 valid addresses
    df = pd.read_csv(csv_file, nrows=1200)
    
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
    
    # Take exactly 1,000 addresses
    df_sample = df_clean.head(1000)
    
    print(f"✅ Using {len(df_sample):,} sample addresses for testing")
    
    return df_sample

def process_1k_addresses_with_gpu(df: pd.DataFrame) -> List[Dict]:
    """Process 1,000 addresses using Shiprocket with GPU acceleration."""
    
    print(f"\n🔧 Initializing Shiprocket IndicBERT parser with GPU acceleration...")
    
    # Initialize Shiprocket parser with GPU support
    from src.shiprocket_parser import ShiprocketParser
    parser = ShiprocketParser(
        batch_size=20,  # Optimal batch size for GPU
        use_gpu=True    # Enable GPU acceleration
    )
    
    print(f"🚀 Processing {len(df):,} addresses with Shiprocket IndicBERT parser (GPU-accelerated)...")
    
    results = []
    batch_size = 20
    total_start_time = time.time()
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches", unit="batch"):
        batch_start_time = time.time()
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
        
        # Print progress every 10 batches
        if (i // batch_size + 1) % 10 == 0:
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - total_start_time
            processed_so_far = len(results)
            avg_time_per_address = elapsed_time / processed_so_far if processed_so_far > 0 else 0
            
            print(f"  Batch {i//batch_size + 1}: {batch_time:.2f}s | "
                  f"Processed: {processed_so_far:,} | "
                  f"Avg: {avg_time_per_address:.3f}s/addr")
    
    total_time = time.time() - total_start_time
    
    # Get final statistics
    stats = parser.get_statistics()
    
    print(f"\n📊 Processing Statistics:")
    print(f"   Total processed: {stats['total_parsed'] + stats['total_failed']:,}")
    print(f"   Successful: {stats['total_parsed']:,} ({stats['success_rate_percent']}%)")
    print(f"   Failed: {stats['total_failed']:,}")
    print(f"   Retries: {stats['total_retries']:,}")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Average time per address: {total_time/len(results):.3f} seconds")
    
    return results, total_time

def analyze_consolidation_results(results: List[Dict]):
    """Analyze consolidation potential from the results."""
    
    print(f"\n🔍 Analyzing consolidation potential...")
    
    pincode_stats = defaultdict(lambda: {
        'total_addresses': 0,
        'successful_parses': 0,
        'localities': set(),
        'societies': set(),
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
    
    # Calculate consolidation metrics
    total_addresses = len(results)
    total_societies = sum(len(stats['societies']) for stats in pincode_stats.values())
    total_localities = sum(len(stats['localities']) for stats in pincode_stats.values())
    total_pincodes = len(pincode_stats)
    
    print(f"\n📊 CONSOLIDATION ANALYSIS (1,000 addresses sample):")
    print("=" * 60)
    print(f"📍 Total addresses: {total_addresses:,}")
    print(f"📮 Unique PIN codes: {total_pincodes:,}")
    print(f"🏢 Unique societies: {total_societies:,}")
    print(f"🏘️ Unique localities: {total_localities:,}")
    print()
    print(f"📉 Consolidation potential:")
    print(f"   Societies: {total_addresses:,} → {total_societies:,} ({(1-total_societies/total_addresses)*100:.1f}% reduction)")
    print(f"   Localities: {total_addresses:,} → {total_localities:,} ({(1-total_localities/total_addresses)*100:.1f}% reduction)")
    print(f"   PIN codes: {total_addresses:,} → {total_pincodes:,} ({(1-total_pincodes/total_addresses)*100:.1f}% reduction)")
    
    # Show top PIN codes
    sorted_pins = sorted(pincode_stats.items(), key=lambda x: x[1]['total_addresses'], reverse=True)
    
    print(f"\n🔝 TOP 10 PIN CODES BY ADDRESS COUNT:")
    print("-" * 60)
    print(f"{'PIN Code':<10} {'Addresses':<10} {'Societies':<10} {'Localities':<10}")
    print("-" * 60)
    
    for pin_code, stats in sorted_pins[:10]:
        print(f"{pin_code:<10} {stats['total_addresses']:<10} {len(stats['societies']):<10} {len(stats['localities']):<10}")

def estimate_full_dataset_time(sample_time: float, sample_size: int, total_size: int):
    """Estimate time for full dataset based on sample performance."""
    
    print(f"\n⏱️ FULL DATASET TIME ESTIMATION:")
    print("=" * 50)
    
    time_per_address = sample_time / sample_size
    estimated_total_time = total_size * time_per_address
    
    # Add model loading overhead (one-time)
    model_loading_overhead = 10  # seconds
    estimated_total_time += model_loading_overhead
    
    hours = int(estimated_total_time // 3600)
    minutes = int((estimated_total_time % 3600) // 60)
    
    print(f"📊 Sample performance:")
    print(f"   Sample size: {sample_size:,} addresses")
    print(f"   Sample time: {sample_time:.2f} seconds")
    print(f"   Time per address: {time_per_address:.3f} seconds")
    print()
    print(f"📊 Full dataset estimate:")
    print(f"   Total addresses: {total_size:,}")
    print(f"   Estimated time: {hours}h {minutes}m ({estimated_total_time/60:.1f} minutes)")
    print(f"   Model loading overhead: {model_loading_overhead}s (one-time)")
    
    return estimated_total_time

def main():
    """Main execution function."""
    
    print("🧪 Testing Shiprocket GPU Processing on 1,000 Addresses")
    print("=" * 70)
    
    # Set up logging
    setup_logging()
    
    # Configuration
    csv_file = 'export_customer_address_store_p0.csv'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_dataset_size = 264150  # From previous analysis
    
    try:
        # Step 1: Load 1K sample data
        df = load_1k_sample_data(csv_file)
        
        # Step 2: Process addresses with GPU
        results, processing_time = process_1k_addresses_with_gpu(df)
        
        # Step 3: Analyze consolidation results
        analyze_consolidation_results(results)
        
        # Step 4: Estimate full dataset time
        estimated_full_time = estimate_full_dataset_time(processing_time, len(results), total_dataset_size)
        
        # Step 5: Save sample results
        output_file = f'sample_1k_results_{timestamp}.csv'
        df_output = pd.DataFrame(results)
        df_output.to_csv(output_file, index=False)
        
        # Final summary
        print(f"\n🎉 1K SAMPLE TEST COMPLETE!")
        print("=" * 70)
        print(f"✅ Successfully processed {len(results):,} addresses")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📁 Sample results saved: {output_file}")
        print(f"🚀 Estimated full dataset time: {estimated_full_time/3600:.1f} hours")
        print()
        print("🚀 Ready to run full dataset processing!")
        
    except Exception as e:
        logging.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Please fix the issues before running full dataset.")

if __name__ == "__main__":
    main()