#!/usr/bin/env python3
"""
Test the enhanced multi-GPU parser with OpenVINO Intel GPU support on 10,000 addresses.
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

from src.enhanced_multi_gpu_parser import EnhancedMultiGPUParser


def setup_logging():
    """Set up detailed logging for the test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'enhanced_multi_gpu_test_{timestamp}.log')
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


def test_enhanced_multi_gpu():
    """Test the enhanced multi-GPU parser."""
    
    print("🚀 Testing Enhanced Multi-GPU Shiprocket Parser with OpenVINO")
    print("=" * 70)
    
    # Load test addresses
    addresses = load_sample_addresses(10000)
    
    # Test configurations
    test_configs = [
        {
            "name": "NVIDIA GPU Only",
            "use_nvidia_gpu": True,
            "use_intel_gpu": False,
            "use_openvino": False,
            "max_workers": 2
        },
        {
            "name": "Enhanced Multi-GPU (NVIDIA + Intel OpenVINO)",
            "use_nvidia_gpu": True,
            "use_intel_gpu": True,
            "use_openvino": True,
            "max_workers": 6
        },
        {
            "name": "CPU Multi-Stream (Fallback)",
            "use_nvidia_gpu": False,
            "use_intel_gpu": False,
            "use_openvino": False,
            "max_workers": 4
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🔧 Testing Configuration: {config['name']}")
        print("-" * 70)
        
        try:
            # Initialize parser with config
            parser = EnhancedMultiGPUParser(
                batch_size=100,
                max_workers=config['max_workers'],
                use_nvidia_gpu=config['use_nvidia_gpu'],
                use_intel_gpu=config['use_intel_gpu'],
                use_openvino=config['use_openvino']
            )
            
            # Run test
            start_time = time.time()
            parsed_results = parser.parse_batch_enhanced(addresses)
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
                "max_workers": config['max_workers'],
                "device_stats": stats.get('device_stats', {}),
                "config_details": config
            }
            
            results.append(result)
            
            # Print results
            print(f"✅ Results for {config['name']}:")
            print(f"   Total Time: {total_time:.2f} seconds")
            print(f"   Success Rate: {result['success_rate']:.1f}%")
            print(f"   Speed: {addresses_per_second:.1f} addresses/second")
            print(f"   Speed: {result['addresses_per_minute']:.0f} addresses/minute")
            
            # Print device-specific stats
            device_stats = stats.get('device_stats', {})
            if device_stats:
                print(f"   Device Performance:")
                for device, device_stat in device_stats.items():
                    if device_stat['processed'] > 0:
                        device_speed = device_stat['processed'] / device_stat['time'] if device_stat['time'] > 0 else 0
                        print(f"     {device}: {device_stat['processed']} addresses, {device_speed:.1f} addr/sec")
            
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
                    filename = f"enhanced_sample_{config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"
                    sample_df.to_csv(filename, index=False)
                    print(f"   Sample results saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error testing {config['name']}: {e}")
            logging.error(f"Configuration {config['name']} failed: {e}", exc_info=True)
    
    # Compare results
    print(f"\n📊 ENHANCED MULTI-GPU PERFORMANCE COMPARISON")
    print("=" * 70)
    
    if results:
        print(f"{'Configuration':<35} {'Time (s)':<10} {'Speed (addr/min)':<15} {'Success %':<10}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['config']:<35} {result['total_time']:<10.1f} {result['addresses_per_minute']:<15.0f} "
                  f"{result['success_rate']:<10.1f}")
        
        # Calculate speedup vs baseline
        if len(results) > 1:
            baseline = results[0]  # First config as baseline
            print(f"\n🚀 SPEEDUP ANALYSIS (vs {baseline['config']}):")
            print("-" * 70)
            
            for result in results[1:]:
                speedup = result['addresses_per_second'] / baseline['addresses_per_second']
                time_reduction = ((baseline['total_time'] - result['total_time']) / baseline['total_time']) * 100
                
                print(f"{result['config']}:")
                print(f"   Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
                print(f"   Time Change: {time_reduction:+.1f}%")
                print(f"   Time: {baseline['total_time']:.1f}s → {result['total_time']:.1f}s")
                
                # Device comparison
                if result['device_stats']:
                    print(f"   Devices Used: {', '.join(result['device_stats'].keys())}")
                print()
        
        # Find best performing configuration
        best_config = max(results, key=lambda x: x['addresses_per_second'])
        print(f"🏆 BEST PERFORMANCE: {best_config['config']}")
        print(f"   Speed: {best_config['addresses_per_second']:.1f} addresses/second")
        print(f"   Time for 10K: {best_config['total_time']:.1f} seconds")
        
        # Estimate full dataset processing time
        full_dataset_size = 263944
        estimated_time = full_dataset_size / best_config['addresses_per_second']
        print(f"   Estimated time for full dataset ({full_dataset_size:,} addresses): {estimated_time/60:.1f} minutes")
    
    return results


def main():
    """Main execution function."""
    
    setup_logging()
    
    print("🔧 Enhanced Multi-GPU Shiprocket Parser Test")
    print("=" * 70)
    
    try:
        results = test_enhanced_multi_gpu()
        
        print(f"\n🎉 Enhanced Multi-GPU Testing Complete!")
        print("Check the log files and CSV outputs for detailed results.")
        
        return results
        
    except Exception as e:
        logging.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Test failed: {e}")
        return []


if __name__ == "__main__":
    results = main()