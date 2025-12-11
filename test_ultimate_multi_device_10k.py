#!/usr/bin/env python3
"""
Test the ULTIMATE multi-device parser using ALL CPUs + ALL GPUs on 10,000 addresses.
This is the maximum performance test using every available processing unit.
"""

import sys
import os
import pandas as pd
import logging
import time
import multiprocessing
from datetime import datetime
from typing import List, Dict

# Add src to path
sys.path.insert(0, 'src')

from src.ultimate_multi_device_parser import UltimateMultiDeviceParser


def setup_logging():
    """Set up detailed logging for the ultimate test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'ultimate_multi_device_test_{timestamp}.log')
        ]
    )


def get_system_info():
    """Get detailed system information."""
    cpu_count = multiprocessing.cpu_count()
    
    # Try to get GPU info
    gpu_info = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_info.append(f"NVIDIA {gpu_name} ({gpu_memory:.1f}GB)")
    except:
        pass
    
    try:
        import openvino as ov
        core = ov.Core()
        for device in core.available_devices:
            if 'GPU' in device:
                gpu_info.append(f"Intel {device}")
    except:
        pass
    
    return {
        "cpu_cores": cpu_count,
        "gpus": gpu_info
    }


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


def test_ultimate_multi_device():
    """Test the ultimate multi-device parser with ALL hardware."""
    
    print("🚀 ULTIMATE MULTI-DEVICE PARSER TEST")
    print("=" * 80)
    
    # Get system information
    system_info = get_system_info()
    print(f"🖥️  System Information:")
    print(f"   CPU Cores: {system_info['cpu_cores']}")
    print(f"   GPUs: {system_info['gpus'] if system_info['gpus'] else 'None detected'}")
    print()
    
    # Load test addresses
    addresses = load_sample_addresses(10000)
    
    # Test configurations - from conservative to maximum
    test_configs = [
        {
            "name": "Conservative Multi-Device",
            "use_nvidia_gpu": True,
            "use_intel_gpu": True,
            "use_all_cpu_cores": True,
            "cpu_core_multiplier": 1.0,  # 1x CPU cores
            "batch_size": 50
        },
        {
            "name": "Aggressive Multi-Device", 
            "use_nvidia_gpu": True,
            "use_intel_gpu": True,
            "use_all_cpu_cores": True,
            "cpu_core_multiplier": 1.5,  # 1.5x CPU cores
            "batch_size": 30
        },
        {
            "name": "MAXIMUM Multi-Device",
            "use_nvidia_gpu": True,
            "use_intel_gpu": True,
            "use_all_cpu_cores": True,
            "cpu_core_multiplier": 2.0,  # 2x CPU cores
            "batch_size": 20
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🔧 Testing Configuration: {config['name']}")
        print("-" * 80)
        
        try:
            # Initialize ultimate parser
            parser = UltimateMultiDeviceParser(
                batch_size=config['batch_size'],
                use_nvidia_gpu=config['use_nvidia_gpu'],
                use_intel_gpu=config['use_intel_gpu'],
                use_all_cpu_cores=config['use_all_cpu_cores'],
                cpu_core_multiplier=config['cpu_core_multiplier']
            )
            
            # Run the ultimate test
            print(f"🚀 Starting {config['name']} test...")
            start_time = time.time()
            parsed_results = parser.parse_ultimate_multi_device(addresses)
            end_time = time.time()
            
            # Calculate comprehensive metrics
            total_time = end_time - start_time
            success_count = sum(1 for r in parsed_results if r.parse_success)
            failed_count = len(parsed_results) - success_count
            addresses_per_second = len(addresses) / total_time
            
            # Get detailed parser statistics
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
                "cpu_cores": stats['cpu_cores'],
                "cpu_workers": stats['cpu_workers'],
                "max_workers": stats['max_workers'],
                "device_stats": stats.get('device_stats', {}),
                "config_details": config
            }
            
            results.append(result)
            
            # Print detailed results
            print(f"✅ Results for {config['name']}:")
            print(f"   Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            print(f"   Success Rate: {result['success_rate']:.1f}%")
            print(f"   Speed: {addresses_per_second:.1f} addresses/second")
            print(f"   Speed: {result['addresses_per_minute']:.0f} addresses/minute")
            print(f"   Workers Used: {stats['max_workers']} ({stats['cpu_workers']} CPU + GPUs)")
            
            # Print device-specific performance
            device_stats = stats.get('device_stats', {})
            if device_stats:
                print(f"   Device Performance:")
                total_device_addresses = 0
                total_device_time = 0
                
                for device, device_stat in device_stats.items():
                    if device_stat['processed'] > 0:
                        device_speed = device_stat['processed'] / device_stat['time'] if device_stat['time'] > 0 else 0
                        total_device_addresses += device_stat['processed']
                        total_device_time += device_stat['time']
                        print(f"     {device}: {device_stat['processed']} addresses, {device_speed:.1f} addr/sec")
                
                # Calculate parallelization efficiency
                if total_device_time > 0:
                    efficiency = (total_device_time / total_time) * 100
                    print(f"   Parallelization Efficiency: {efficiency:.1f}%")
            
            # Save sample results
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
                    filename = f"ultimate_sample_{config['name'].lower().replace(' ', '_').replace('-', '_')}.csv"
                    sample_df.to_csv(filename, index=False)
                    print(f"   Sample results saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error testing {config['name']}: {e}")
            logging.error(f"Configuration {config['name']} failed: {e}", exc_info=True)
    
    # Performance comparison and analysis
    print(f"\n📊 ULTIMATE MULTI-DEVICE PERFORMANCE COMPARISON")
    print("=" * 80)
    
    if results:
        print(f"{'Configuration':<25} {'Time (min)':<12} {'Speed (addr/min)':<15} {'Workers':<10} {'Success %':<10}")
        print("-" * 80)
        
        for result in results:
            time_minutes = result['total_time'] / 60
            print(f"{result['config']:<25} {time_minutes:<12.1f} {result['addresses_per_minute']:<15.0f} "
                  f"{result['max_workers']:<10} {result['success_rate']:<10.1f}")
        
        # Find the absolute best performance
        best_config = max(results, key=lambda x: x['addresses_per_second'])
        
        print(f"\n🏆 ULTIMATE PERFORMANCE CHAMPION: {best_config['config']}")
        print("-" * 80)
        print(f"   Speed: {best_config['addresses_per_second']:.1f} addresses/second")
        print(f"   Time for 10K: {best_config['total_time']:.1f} seconds ({best_config['total_time']/60:.1f} minutes)")
        print(f"   Workers: {best_config['max_workers']} total")
        print(f"   CPU Workers: {best_config['cpu_workers']}")
        
        # Calculate speedup vs original single GPU
        original_speed = 5.5  # From previous tests
        ultimate_speedup = best_config['addresses_per_second'] / original_speed
        print(f"   Ultimate Speedup: {ultimate_speedup:.1f}x vs original single GPU")
        
        # Estimate full dataset processing time
        full_dataset_size = 263944
        estimated_time = full_dataset_size / best_config['addresses_per_second']
        print(f"   Full Dataset ({full_dataset_size:,} addresses): {estimated_time/60:.1f} minutes")
        
        # Hardware utilization summary
        print(f"\n🔧 HARDWARE UTILIZATION SUMMARY:")
        print("-" * 80)
        device_stats = best_config.get('device_stats', {})
        for device, stats in device_stats.items():
            if stats['processed'] > 0:
                utilization = (stats['processed'] / best_config['total_addresses']) * 100
                device_speed = stats['processed'] / stats['time'] if stats['time'] > 0 else 0
                print(f"   {device}: {utilization:.1f}% of work, {device_speed:.1f} addr/sec")
    
    return results


def main():
    """Main execution function."""
    
    setup_logging()
    
    print("🔧 ULTIMATE Multi-Device Shiprocket Parser Test")
    print("Using ALL CPUs + ALL GPUs simultaneously")
    print("=" * 80)
    
    try:
        results = test_ultimate_multi_device()
        
        print(f"\n🎉 ULTIMATE Multi-Device Testing Complete!")
        print("This represents the maximum possible performance on your hardware.")
        print("Check the log files and CSV outputs for detailed results.")
        
        return results
        
    except Exception as e:
        logging.error(f"Ultimate test failed: {e}", exc_info=True)
        print(f"\n❌ Ultimate test failed: {e}")
        return []


if __name__ == "__main__":
    results = main()