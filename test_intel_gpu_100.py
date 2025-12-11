#!/usr/bin/env python3
"""
Test Intel GPU processing with 100 addresses to understand performance characteristics.
"""

import sys
import os
import pandas as pd
import logging
import time
from datetime import datetime
from typing import List

# Add src to path
sys.path.insert(0, 'src')

from src.ultimate_multi_device_parser import UltimateMultiDeviceParser
from src.models import ParsedAddress


def setup_logging():
    """Set up detailed logging for Intel GPU test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'intel_gpu_test_{timestamp}.log', encoding='utf-8')
        ]
    )


def load_sample_addresses(count: int = 100) -> List[str]:
    """Load first 100 addresses from P1 file for testing."""
    
    print(f"📂 Loading {count} sample addresses from P1 file...")
    
    try:
        # Load the P1 CSV file
        df = pd.read_csv('export_customer_address_store_p1.csv')
        
        # Find address column
        address_column = None
        for col in ['addr_text', 'address', 'full_address', 'Address', 'addr']:
            if col in df.columns:
                address_column = col
                break
        
        if not address_column:
            print("❌ Could not find address column")
            return []
        
        # Filter out empty addresses and take first 100
        df_clean = df.dropna(subset=[address_column])
        df_clean = df_clean[df_clean[address_column].str.strip() != '']
        
        # Get first 100 addresses
        addresses = df_clean[address_column].head(count).tolist()
        
        print(f"✅ Loaded {len(addresses)} addresses for Intel GPU test")
        return addresses
        
    except Exception as e:
        print(f"❌ Error loading addresses: {e}")
        return []


class IntelGPUTester(UltimateMultiDeviceParser):
    """Specialized tester for Intel GPU only."""
    
    def __init__(self):
        super().__init__(
            batch_size=10,  # Small batch size for testing
            use_nvidia_gpu=False,  # Disable NVIDIA
            use_intel_gpu=True,   # Enable Intel only
            use_all_cpu_cores=False,  # Disable CPU
            cpu_core_multiplier=1.0
        )
    
    def test_intel_gpu_only(self, addresses: List[str]) -> List[ParsedAddress]:
        """Test Intel GPU processing only."""
        if not addresses:
            return []
        
        total_addresses = len(addresses)
        logging.info(f"🔵 Starting Intel GPU ONLY test with {total_addresses} addresses...")
        
        # Initialize Intel GPU only
        if not self._setup_intel_openvino():
            logging.error("❌ Failed to initialize Intel GPU")
            return [ParsedAddress(parse_success=False, parse_error="Intel GPU init failed") for _ in addresses]
        
        # Initialize device stats for Intel GPU
        self._device_stats["intel_openvino"] = {"processed": 0, "time": 0}
        
        logging.info("✅ Intel GPU initialized successfully")
        
        # Process addresses in small batches with detailed timing
        results = []
        batch_size = 10
        
        overall_start = time.time()
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(addresses) + batch_size - 1) // batch_size
            
            logging.info(f"🔵 Processing batch {batch_num}/{total_batches}: {len(batch)} addresses")
            
            batch_start = time.time()
            
            try:
                # Process batch on Intel GPU
                batch_results = self._process_on_intel(batch)
                results.extend(batch_results)
                
                batch_time = time.time() - batch_start
                batch_rate = len(batch) / batch_time if batch_time > 0 else 0
                
                logging.info(f"✅ Batch {batch_num} completed in {batch_time:.2f}s ({batch_rate:.2f} addr/sec)")
                
                # Progress update
                completed = len(results)
                elapsed = time.time() - overall_start
                overall_rate = completed / elapsed if elapsed > 0 else 0
                remaining = total_addresses - completed
                eta = remaining / overall_rate if overall_rate > 0 else 0
                
                logging.info(f"📊 Progress: {completed}/{total_addresses} ({completed/total_addresses*100:.1f}%) | "
                           f"Rate: {overall_rate:.2f}/sec | ETA: {eta:.1f}s")
                
            except Exception as e:
                logging.error(f"❌ Batch {batch_num} failed: {e}")
                # Add failed results for this batch
                for addr in batch:
                    results.append(ParsedAddress(
                        parse_success=False,
                        parse_error=f"Intel GPU batch error: {str(e)}"
                    ))
        
        total_time = time.time() - overall_start
        success_count = sum(1 for r in results if r.parse_success)
        
        logging.info(f"🎉 Intel GPU test complete!")
        logging.info(f"📊 Results: {success_count}/{total_addresses} success ({success_count/total_addresses*100:.1f}%)")
        logging.info(f"⏱️ Total time: {total_time:.2f}s")
        logging.info(f"🚀 Average speed: {total_addresses/total_time:.2f} addresses/second")
        
        return results


def main():
    """Main execution function."""
    
    setup_logging()
    
    print("🔵 Intel GPU Performance Test")
    print("Testing 100 addresses on Intel GPU only")
    print("=" * 60)
    
    # Load sample addresses
    addresses = load_sample_addresses(100)
    
    if not addresses:
        print("❌ No addresses to test. Exiting.")
        return None
    
    print(f"📊 Test Configuration:")
    print(f"   Addresses: {len(addresses)}")
    print(f"   Device: Intel GPU only")
    print(f"   Batch Size: 10")
    print()
    
    # Create Intel GPU tester
    tester = IntelGPUTester()
    
    try:
        # Run the test
        start_time = time.time()
        results = tester.test_intel_gpu_only(addresses)
        end_time = time.time()
        
        total_time = end_time - start_time
        success_count = sum(1 for r in results if r.parse_success)
        failed_count = len(results) - success_count
        
        # Print results
        print("\n🎉 INTEL GPU TEST COMPLETE!")
        print("=" * 60)
        print(f"📊 RESULTS:")
        print(f"   Total Addresses: {len(addresses)}")
        print(f"   Successfully Parsed: {success_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Success Rate: {(success_count/len(addresses)*100):.1f}%")
        print()
        
        print(f"⏱️  PERFORMANCE:")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   Speed: {len(addresses)/total_time:.2f} addresses/second")
        print()
        
        # Extrapolate performance for full dataset
        if total_time > 0:
            full_dataset_time = (264170 * total_time) / len(addresses)
            print(f"📈 EXTRAPOLATED PERFORMANCE:")
            print(f"   For 264,170 addresses: {full_dataset_time:.0f} seconds ({full_dataset_time/60:.1f} minutes)")
            print(f"   This explains why Intel GPU was timing out at 10 minutes!")
        
        # Save sample results
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'intel_gpu_test_results_{timestamp}.csv'
            
            results_data = []
            for i, (original_addr, parsed) in enumerate(zip(addresses, results)):
                results_data.append({
                    'id': i + 1,
                    'original_address': original_addr,
                    'parse_success': parsed.parse_success,
                    'parse_error': parsed.parse_error if not parsed.parse_success else '',
                    'society_name': parsed.society_name if parsed.parse_success else '',
                    'locality': parsed.locality if parsed.parse_success else '',
                    'city': parsed.city if parsed.parse_success else '',
                    'pin_code': parsed.pin_code if parsed.parse_success else ''
                })
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(output_file, index=False)
            print(f"✅ Sample results saved to: {output_file}")
        
        return {
            'total_addresses': len(addresses),
            'success_count': success_count,
            'total_time': total_time,
            'addresses_per_second': len(addresses)/total_time if total_time > 0 else 0
        }
        
    except Exception as e:
        print(f"\n❌ Intel GPU test failed: {e}")
        logging.error(f"Intel GPU test failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    results = main()