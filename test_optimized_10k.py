#!/usr/bin/env python3
"""
Test the optimized Shiprocket parser on 10,000 addresses with full parallel processing.
"""

import sys
import os
import pandas as pd
import logging
import time
from datetime import datetime
from typing import List, Dict

# Add src to path
sys.path.insert(0, 'src')

from src.optimized_shiprocket_parser import OptimizedShiprocketParser


def setup_logging():
    """Set up detailed logging for the test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'optimized_test_10k_{timestamp}.log')
        ]
    )


def load_sample_addresses(count: int = 10000) -> List[str]:
    """Load sample addresses from the dataset."""
    
    print(f"📂 Loading {count:,} sample addresses...")
    
    try:
        # Load from the original CSV
        df = pd.read_csv('export_customer_address_store_p0.csv')
        
        # Filter out empty addresses
        df_clean = df.dropna(subset=['addr_text'])
        df_clean = df_clean[df_clean['addr_text'].str.strip() != '']
        
        # Take first N addresses
        sample_addresses = df_clean['addr_text'].head(count).tolist()
        
        print(f"✅ Loaded {len(sample_addresses):,} addresses for testing")
        return sample_addresses
        
    except FileNotFoundError:
        print("❌ Error: export_customer_address_store_p0.csv not found")
        print("   Using dummy addresses for testing...")
        
        # Generate dummy addresses for testing
        dummy_addresses = []
        for i in range(count):
            dummy_addresses.append(
                f"Flat {i+1}, Building {(i//10)+1}, Sector {(i//100)+1}, "
                f"Phase {(i//1000)+1}, Hinjewadi, Pune, Maharashtra, {411057 + (i%10)}"
            )
        
        return dummy_addresses


def test_optimized_parser():
    """Test the optimized parser with various configurations."""
    
    print("🚀 Testing Optimized Shiprocket Parser")
    print("=" * 60)
    
    # Load test addresses
    addresses = load_sample_addresses(10000)
    
    # Test configurations
    test_configs = [
        {
            "name": "Conservative (Original-like)",
            "batch_size": 20,
            "max_workers": 1,
            "use_intel_gpu": False
        },
        {
            "name": "Optimized Single GPU",
            "batch_size": 100,
            "max_workers": 2,
            "use_intel_gpu": False
        },
        {
            "name": "Multi-GPU Optimized",
            "batch_size": 100,
            "max_workers": 4,
            "use_intel_gpu": True
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🔧 Testing Configuration: {config['name']}")
        print("-" * 60)
        
        try:
            # Initialize parser with config
            parser = OptimizedShiprocketParser(
                batch_size=config['batch_size'],
                max_workers=config['max_workers'],
                use_gpu=True,
                use_intel_gpu=config['use_intel_gpu']
            )
            
            # Run test
            start_time = time.time()
            parsed_results = parser.parse_batch_optimized(addresses)
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            success_count = sum(1 for r in parsed_results if r.parse_success)
            failed_count = len(parsed_results) - success_count
            addresses_per_second = len(addresses) / total_time
            
            # Get parser statistics
            stats = parser.get_statistics()
            
            result = {
                "config": config['name'],
                "total_addresses": len(addresses),
                "success_count": success_count,
                "failed_count": failed_count,
                "success_rate": (success_count / len(addresses)) * 100,
                "total_time": total_time,
                "addresses_per_second": addresses_per_second,
                "addresses_per_minute": addresses_per_second * 60,
                "batch_size": config['batch_size'],
                "max_workers": config['max_workers'],
                "devices_used": stats.get('devices_used', []),
                "avg_processing_time": stats.get('avg_processing_time', 0)
            }
            
            results.append(result)
            
            # Print results
            print(f"✅ Results for {config['name']}:")
            print(f"   Total Time: {total_time:.2f} seconds")
            print(f"   Success Rate: {result['success_rate']:.1f}%")
            print(f"   Speed: {addresses_per_second:.1f} addresses/second")
            print(f"   Speed: {result['addresses_per_minute']:.0f} addresses/minute")
            print(f"   Devices Used: {stats.get('devices_used', [])}")
            
            # Save sample results for this config
            if success_count > 0:
                sample_results = []
                for i, parsed in enumerate(parsed_results[:100]):  # First 100 results
                    if parsed.parse_success:
                        sample_results.append({
                            'original_address': addresses[i],
                            'society_name': parsed.society_name,
                            'locality': parsed.locality,
                            'city': parsed.city,
                            'pin_code': parsed.pin_code
                        })
                
                if sample_results:
                    sample_df = pd.DataFrame(sample_results)
                    filename = f"optimized_sample_{config['name'].lower().replace(' ', '_')}.csv"
                    sample_df.to_csv(filename, index=False)
                    print(f"   Sample results saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error testing {config['name']}: {e}")
            logging.error(f"Configuration {config['name']} failed: {e}", exc_info=True)
    
    # Compare results
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    if results:
        print(f"{'Configuration':<25} {'Time (s)':<10} {'Speed (addr/min)':<15} {'Success %':<10} {'Devices':<15}")
        print("-" * 80)
        
        for result in results:
            devices_str = ', '.join(result['devices_used']) if result['devices_used'] else 'Unknown'
            print(f"{result['config']:<25} {result['total_time']:<10.1f} {result['addresses_per_minute']:<15.0f} "
                  f"{result['success_rate']:<10.1f} {devices_str:<15}")
        
        # Calculate speedup
        if len(results) > 1:
            baseline = results[0]
            print(f"\n🚀 SPEEDUP ANALYSIS:")
            print("-" * 60)
            
            for result in results[1:]:
                speedup = result['addresses_per_second'] / baseline['addresses_per_second']
                time_reduction = ((baseline['total_time'] - result['total_time']) / baseline['total_time']) * 100
                
                print(f"{result['config']} vs {baseline['config']}:")
                print(f"   Speedup: {speedup:.1f}x faster")
                print(f"   Time Reduction: {time_reduction:.1f}%")
                print(f"   Time: {baseline['total_time']:.1f}s → {result['total_time']:.1f}s")
    
    return results


def main():
    """Main execution function."""
    
    setup_logging()
    
    print("🔧 Optimized Shiprocket Parser - 10K Address Test")
    print("=" * 60)
    
    try:
        results = test_optimized_parser()
        
        print(f"\n🎉 Testing Complete!")
        print("Check the log files and CSV outputs for detailed results.")
        
        return results
        
    except Exception as e:
        logging.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Test failed: {e}")
        return []


if __name__ == "__main__":
    results = main()