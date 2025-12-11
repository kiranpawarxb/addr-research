#!/usr/bin/env python3
"""
Estimate processing time for full CSV dataset
"""

import pandas as pd
import time
import os

def estimate_dataset_size():
    """Estimate the total number of addresses in the dataset."""
    
    print("📊 Estimating dataset size...")
    
    # Get file size
    file_size = os.path.getsize('export_customer_address_store_p0.csv')
    print(f"📁 File size: {file_size/1024/1024:.1f} MB")
    
    # Sample first 1000 rows to estimate total
    sample_df = pd.read_csv('export_customer_address_store_p0.csv', nrows=1000)
    print(f"📋 Sample loaded: {len(sample_df)} rows")
    print(f"📋 Columns: {list(sample_df.columns)}")
    
    # Check for valid addresses in sample
    valid_addresses = sample_df.dropna(subset=['addr_text'])
    valid_addresses = valid_addresses[valid_addresses['addr_text'].str.strip() != '']
    
    print(f"✅ Valid addresses in sample: {len(valid_addresses)}/{len(sample_df)} ({len(valid_addresses)/len(sample_df)*100:.1f}%)")
    
    # Estimate total rows based on file size
    # Average bytes per row in sample
    sample_size_bytes = len(sample_df) * (file_size / 1000)  # Rough estimate
    estimated_total_rows = int(file_size / (sample_size_bytes / len(sample_df)))
    
    # More accurate: count actual lines
    print("🔢 Counting actual lines in file...")
    with open('export_customer_address_store_p0.csv', 'r', encoding='utf-8') as f:
        line_count = sum(1 for line in f) - 1  # Subtract header
    
    print(f"📊 Actual total rows: {line_count:,}")
    
    # Estimate valid addresses
    valid_ratio = len(valid_addresses) / len(sample_df)
    estimated_valid_addresses = int(line_count * valid_ratio)
    
    print(f"📊 Estimated valid addresses: {estimated_valid_addresses:,}")
    
    return estimated_valid_addresses, line_count

def estimate_processing_time(total_addresses):
    """Estimate processing time based on test performance."""
    
    print(f"\n⏱️ Estimating processing time...")
    
    # From our test: 10 addresses took about 10-15 seconds (including model loading)
    # Model loading is one-time cost, so subsequent addresses are faster
    
    # Conservative estimates:
    model_loading_time = 10  # seconds (one-time)
    time_per_address = 0.5   # seconds per address (CPU processing)
    batch_size = 50          # addresses per batch
    
    # Calculate time
    total_processing_time = model_loading_time + (total_addresses * time_per_address)
    
    # Convert to readable format
    hours = int(total_processing_time // 3600)
    minutes = int((total_processing_time % 3600) // 60)
    seconds = int(total_processing_time % 60)
    
    print(f"📊 Processing estimates:")
    print(f"   Model loading: {model_loading_time} seconds (one-time)")
    print(f"   Time per address: {time_per_address} seconds")
    print(f"   Batch size: {batch_size} addresses")
    print(f"   Total addresses: {total_addresses:,}")
    print(f"   Estimated total time: {hours}h {minutes}m {seconds}s")
    print(f"   Estimated total time: {total_processing_time/60:.1f} minutes")
    
    # Conservative estimate (add 50% buffer)
    conservative_time = total_processing_time * 1.5
    c_hours = int(conservative_time // 3600)
    c_minutes = int((conservative_time % 3600) // 60)
    
    print(f"\n🛡️ Conservative estimate (with 50% buffer):")
    print(f"   Total time: {c_hours}h {c_minutes}m")
    print(f"   Total time: {conservative_time/60:.1f} minutes")
    
    return total_processing_time, conservative_time

def main():
    print("⏱️ Processing Time Estimation for Shiprocket Full CSV")
    print("=" * 60)
    
    try:
        # Estimate dataset size
        total_addresses, total_rows = estimate_dataset_size()
        
        # Estimate processing time
        normal_time, conservative_time = estimate_processing_time(total_addresses)
        
        print(f"\n📋 SUMMARY:")
        print("=" * 40)
        print(f"📊 Total rows in CSV: {total_rows:,}")
        print(f"📊 Valid addresses: {total_addresses:,}")
        print(f"⏱️ Estimated time: {normal_time/60:.1f} minutes")
        print(f"🛡️ Conservative time: {conservative_time/60:.1f} minutes")
        
        # Consolidation benefits
        print(f"\n🏢 CONSOLIDATION BENEFITS:")
        print("=" * 40)
        print(f"📊 Current: {total_addresses:,} individual flat addresses")
        print(f"🏢 Expected societies: ~{total_addresses//20:,} (assuming ~20 flats per society)")
        print(f"🏘️ Expected localities: ~{total_addresses//100:,} (assuming ~100 flats per locality)")
        print(f"📉 Potential reduction: ~{(1-1/20)*100:.0f}% for societies, ~{(1-1/100)*100:.0f}% for localities")
        
        print(f"\n✅ Ready to proceed with full processing!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()