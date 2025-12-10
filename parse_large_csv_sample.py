#!/usr/bin/env python3
"""Parse 20 random addresses from the large CSV file."""

import sys
import random
import time
import pandas as pd
from datetime import datetime
sys.path.insert(0, 'src')

def sample_large_csv(csv_file, sample_size=20):
    """Sample random addresses from large CSV file."""
    
    print(f"📂 Sampling {sample_size} addresses from large CSV: {csv_file}")
    
    try:
        # Read CSV in chunks to handle large file
        chunk_size = 1000
        sampled_addresses = []
        total_rows = 0
        
        print("🔄 Reading CSV in chunks...")
        
        for chunk_num, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size), 1):
            total_rows += len(chunk)
            
            # Print column names for first chunk
            if chunk_num == 1:
                print(f"📋 Available columns: {list(chunk.columns)}")
            
            # Find address column
            address_col = None
            for col in ['addr_text', 'raw_address', 'address', 'customer_address', 'full_address', 'shipping_address', 'delivery_address']:
                if col in chunk.columns:
                    address_col = col
                    break
            
            # If no standard address column found, look for columns with 'address' in name
            if not address_col:
                for col in chunk.columns:
                    if 'address' in col.lower():
                        address_col = col
                        print(f"🔍 Found address column: {address_col}")
                        break
            
            if not address_col and len(chunk.columns) > 1:
                # Skip first column (likely ID), use second column
                address_col = chunk.columns[1]
                print(f"⚠️  Using second column as address: {address_col}")
            
            if address_col:
                # Sample from this chunk
                chunk_addresses = chunk[address_col].dropna().tolist()
                sampled_addresses.extend(chunk_addresses)
                
                print(f"   Chunk {chunk_num}: {len(chunk_addresses)} addresses")
                
                # Stop if we have enough samples
                if len(sampled_addresses) >= sample_size * 5:  # Get 5x more for better randomness
                    break
        
        print(f"✅ Collected {len(sampled_addresses)} addresses from {total_rows} total rows")
        
        # Random sample
        if len(sampled_addresses) > sample_size:
            final_sample = random.sample(sampled_addresses, sample_size)
        else:
            final_sample = sampled_addresses
        
        print(f"🎲 Selected {len(final_sample)} random addresses")
        return final_sample
        
    except Exception as e:
        print(f"❌ Error sampling CSV: {e}")
        return []


def main():
    """Sample and parse addresses from large CSV."""
    
    print("📊 Large CSV Shiprocket Parser")
    print("=" * 50)
    
    # Sample addresses from large CSV
    addresses = sample_large_csv('export_customer_address_store_p0.csv', 20)
    
    if not addresses:
        print("❌ No addresses sampled. Exiting.")
        return
    
    # Parse with Shiprocket
    from shiprocket_parser import ShiprocketParser
    
    parser = ShiprocketParser(use_gpu=False)
    
    print(f"\n🚀 Parsing {len(addresses)} addresses...")
    print("-" * 50)
    
    results = []
    for i, addr in enumerate(addresses, 1):
        print(f"{i:2d}. {addr[:60]}...")
        
        start_time = time.time()
        result = parser.parse_address(addr)
        end_time = time.time()
        
        results.append({
            'address': addr,
            'success': result.parse_success,
            'time': end_time - start_time,
            'society': result.society_name or '',
            'unit': result.unit_number or '',
            'locality': result.locality or '',
            'road': result.road or '',
            'city': result.city or '',
            'error': result.parse_error or ''
        })
        
        if result.parse_success:
            print(f"    ✅ ({end_time - start_time:.3f}s)")
        else:
            print(f"    ❌ ({end_time - start_time:.3f}s) - {result.parse_error}")
    
    # Summary
    successful = [r for r in results if r['success']]
    
    print(f"\n📊 Results Summary:")
    print(f"   Success: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"   Society extraction: {sum(1 for r in successful if r['society'])}/{len(successful)} ({sum(1 for r in successful if r['society'])/len(successful)*100:.1f}%)")
    print(f"   Locality extraction: {sum(1 for r in successful if r['locality'])}/{len(successful)} ({sum(1 for r in successful if r['locality'])/len(successful)*100:.1f}%)")
    
    # Create simple CSV report
    output_file = f'large_csv_sample_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\n📁 Report saved: {output_file}")


if __name__ == "__main__":
    main()