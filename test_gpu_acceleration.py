#!/usr/bin/env python3
"""
Test GPU acceleration for Shiprocket parser
"""

import sys
import os
import pandas as pd
import time
import logging
from datetime import datetime

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

def test_gpu_vs_cpu_performance():
    """Test GPU vs CPU performance."""
    
    print("🚀 Testing GPU vs CPU Performance for Shiprocket Parser")
    print("=" * 70)
    
    # Load sample data
    df = pd.read_csv('export_customer_address_store_p0.csv', nrows=20)
    df_clean = df.dropna(subset=['addr_text'])
    df_clean = df_clean[df_clean['addr_text'].str.strip() != '']
    
    addresses = df_clean['addr_text'].tolist()[:10]  # Test with 10 addresses
    
    print(f"📊 Testing with {len(addresses)} addresses")
    
    # Test CPU performance
    print(f"\n🖥️ Testing CPU Performance...")
    try:
        from src.shiprocket_parser import ShiprocketParser
        
        cpu_parser = ShiprocketParser(
            batch_size=5,
            use_gpu=False  # CPU only
        )
        
        start_time = time.time()
        cpu_results = []
        
        for i, addr in enumerate(addresses):
            print(f"  CPU Processing {i+1}/{len(addresses)}: {addr[:50]}...")
            result = cpu_parser.parse_address(addr)
            cpu_results.append(result)
        
        cpu_time = time.time() - start_time
        cpu_stats = cpu_parser.get_statistics()
        
        print(f"  ✅ CPU Results: {cpu_stats['total_parsed']}/{len(addresses)} successful")
        print(f"  ⏱️ CPU Time: {cpu_time:.2f} seconds ({cpu_time/len(addresses):.2f}s per address)")
        
    except Exception as e:
        print(f"  ❌ CPU Test failed: {e}")
        cpu_time = float('inf')
        cpu_results = []
    
    # Test GPU performance
    print(f"\n🎮 Testing GPU Performance...")
    try:
        gpu_parser = ShiprocketParser(
            batch_size=5,
            use_gpu=True  # GPU enabled
        )
        
        start_time = time.time()
        gpu_results = []
        
        for i, addr in enumerate(addresses):
            print(f"  GPU Processing {i+1}/{len(addresses)}: {addr[:50]}...")
            result = gpu_parser.parse_address(addr)
            gpu_results.append(result)
        
        gpu_time = time.time() - start_time
        gpu_stats = gpu_parser.get_statistics()
        
        print(f"  ✅ GPU Results: {gpu_stats['total_parsed']}/{len(addresses)} successful")
        print(f"  ⏱️ GPU Time: {gpu_time:.2f} seconds ({gpu_time/len(addresses):.2f}s per address)")
        
    except Exception as e:
        print(f"  ❌ GPU Test failed: {e}")
        gpu_time = float('inf')
        gpu_results = []
    
    # Compare results
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    if cpu_time != float('inf') and gpu_time != float('inf'):
        speedup = cpu_time / gpu_time
        print(f"🖥️ CPU Time: {cpu_time:.2f}s ({cpu_time/len(addresses):.2f}s per address)")
        print(f"🎮 GPU Time: {gpu_time:.2f}s ({gpu_time/len(addresses):.2f}s per address)")
        print(f"🚀 Speedup: {speedup:.2f}x faster with GPU")
        
        # Estimate full dataset time
        total_addresses = 264150
        cpu_full_time = total_addresses * (cpu_time / len(addresses))
        gpu_full_time = total_addresses * (gpu_time / len(addresses))
        
        print(f"\n📊 FULL DATASET ESTIMATES:")
        print(f"🖥️ CPU Full Dataset: {cpu_full_time/3600:.1f} hours")
        print(f"🎮 GPU Full Dataset: {gpu_full_time/3600:.1f} hours")
        print(f"⏰ Time Saved: {(cpu_full_time - gpu_full_time)/3600:.1f} hours")
        
        return gpu_time / len(addresses)  # Return GPU time per address
    else:
        print("❌ Could not complete performance comparison")
        return None

def main():
    setup_logging()
    
    try:
        # Test GPU acceleration
        gpu_time_per_address = test_gpu_vs_cpu_performance()
        
        if gpu_time_per_address:
            print(f"\n✅ GPU acceleration is working!")
            print(f"🚀 Ready to process full dataset with GPU acceleration")
            print(f"⏱️ Estimated time per address: {gpu_time_per_address:.3f} seconds")
        else:
            print(f"\n❌ GPU acceleration test failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()