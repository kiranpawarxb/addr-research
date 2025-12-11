#!/usr/bin/env python3
"""
Test Shiprocket Processing for 10 Addresses

This script tests the Shiprocket parser on just 10 addresses to verify everything works
before running on the full dataset.

Usage:
    python test_shiprocket_10_addresses.py
"""

import sys
import os
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

# Add src to path
sys.path.insert(0, 'src')

def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_sample_csv_data(csv_file: str, sample_size: int = 10) -> pd.DataFrame:
    """Load and validate CSV data, taking only a sample."""
    
    print(f"📂 Loading CSV file: {csv_file}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Load CSV - just read first few rows for testing
    df = pd.read_csv(csv_file, nrows=sample_size * 2)  # Read extra in case some are empty
    
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
    
    # Take only the sample size we want
    df_sample = df_clean.head(sample_size)
    
    print(f"✅ Using {len(df_sample):,} sample addresses")
    
    return df_sample

def test_shiprocket_parser(df: pd.DataFrame) -> List[Dict]:
    """Test Shiprocket parser on sample addresses."""
    
    print(f"\n🔧 Testing Shiprocket IndicBERT parser with GPU support...")
    
    try:
        # Initialize Shiprocket parser with GPU support
        from src.shiprocket_parser import ShiprocketParser
        parser = ShiprocketParser(
            batch_size=5,   # Small batch for testing
            use_gpu=True    # Enable GPU acceleration
        )
        
        print(f"🚀 Processing {len(df):,} test addresses with Shiprocket IndicBERT parser (GPU-accelerated)...")
        
        results = []
        
        # Process addresses one by one for testing
        for idx, (_, row) in enumerate(df.iterrows()):
            print(f"Processing address {idx + 1}/{len(df)}: {row['addr_text'][:50]}...")
            
            try:
                # Parse single address
                parsed = parser.parse_address(row['addr_text'])
                
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
                }
                
                results.append(result)
                
                # Print result for this address
                print(f"  ✅ Success: {parsed.parse_success}")
                if parsed.parse_success:
                    print(f"     Society: {parsed.society_name}")
                    print(f"     Locality: {parsed.locality}")
                    print(f"     City: {parsed.city}")
                    print(f"     PIN: {parsed.pin_code}")
                else:
                    print(f"     Error: {parsed.parse_error}")
                
            except Exception as e:
                print(f"  ❌ Error processing address {idx + 1}: {e}")
                
                result = {
                    'hash_key': row.get('addr_hash_key', ''),
                    'raw_address': row.get('addr_text', ''),
                    'original_pincode': row.get('pincode', ''),
                    'unit_number': '', 'society_name': '', 'landmark': '', 'road': '',
                    'sub_locality': '', 'locality': '', 'city': '', 'district': '',
                    'state': '', 'country': 'India', 'parsed_pincode': '',
                    'parse_success': False,
                    'parse_error': f'Processing error: {str(e)}',
                }
                results.append(result)
        
        # Get final statistics
        stats = parser.get_statistics()
        
        print(f"\n📊 Test Statistics:")
        print(f"   Total processed: {stats['total_parsed'] + stats['total_failed']:,}")
        print(f"   Successful: {stats['total_parsed']:,} ({stats['success_rate_percent']}%)")
        print(f"   Failed: {stats['total_failed']:,}")
        print(f"   Retries: {stats['total_retries']:,}")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure transformers and torch are installed:")
        print("pip install transformers torch")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

def analyze_test_results(results: List[Dict]):
    """Analyze the test results."""
    
    print(f"\n🔍 Analyzing test results...")
    
    successful = [r for r in results if r['parse_success']]
    failed = [r for r in results if not r['parse_success']]
    
    print(f"✅ Successful parses: {len(successful)}/{len(results)}")
    print(f"❌ Failed parses: {len(failed)}/{len(results)}")
    
    if successful:
        print(f"\n🏢 Societies found:")
        societies = set()
        localities = set()
        
        for result in successful:
            if result['society_name']:
                societies.add(result['society_name'])
            if result['locality']:
                localities.add(result['locality'])
        
        print(f"   Unique societies: {len(societies)}")
        for society in sorted(societies):
            print(f"     - {society}")
        
        print(f"\n🏘️ Localities found:")
        print(f"   Unique localities: {len(localities)}")
        for locality in sorted(localities):
            print(f"     - {locality}")
    
    if failed:
        print(f"\n❌ Failed addresses:")
        for result in failed:
            print(f"   - {result['raw_address'][:50]}... | Error: {result['parse_error']}")

def main():
    """Main execution function."""
    
    print("🧪 Testing Shiprocket Parser on 10 Addresses")
    print("=" * 60)
    
    # Set up logging
    setup_logging()
    
    # Configuration
    csv_file = 'export_customer_address_store_p0.csv'
    
    try:
        # Step 1: Load sample CSV data
        df = load_sample_csv_data(csv_file, sample_size=10)
        
        # Step 2: Test Shiprocket parser
        results = test_shiprocket_parser(df)
        
        # Step 3: Analyze results
        analyze_test_results(results)
        
        # Final summary
        print(f"\n🎉 TEST COMPLETE!")
        print("=" * 60)
        print("✅ Shiprocket parser is working!")
        print("🚀 Ready to run on full dataset with process_full_csv_shiprocket.py")
        
    except Exception as e:
        logging.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Please fix the issues before running on full dataset.")

if __name__ == "__main__":
    main()