#!/usr/bin/env python3
"""
Hybrid P2 processing with Intel GPU included:
- NVIDIA GPU: 65% of work (fast processing)
- 28 CPU cores: 30% of work (distributed evenly)
- Intel GPU: 5% of work (small load with long timeout)
- Progress logging every 30 seconds
- Realistic timeouts for all devices
"""

import sys
import os
import pandas as pd
import logging
import time
import multiprocessing
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path
sys.path.insert(0, 'src')

from src.ultimate_multi_device_parser import UltimateMultiDeviceParser
from src.models import ParsedAddress


class HybridP2Parser(UltimateMultiDeviceParser):
    """Hybrid parser with NVIDIA GPU + CPU + Intel GPU."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_lock = threading.Lock()
        self.completed_addresses = 0
        self.start_time = None
        self.progress_thread = None
        self.stop_progress = False
    
    def start_progress_logging(self, total_addresses):
        """Start background progress logging."""
        self.start_time = time.time()
        self.total_addresses = total_addresses
        self.stop_progress = False
        
        def log_progress():
            while not self.stop_progress:
                time.sleep(30)  # Log every 30 seconds
                if not self.stop_progress:
                    with self.progress_lock:
                        elapsed = time.time() - self.start_time
                        rate = self.completed_addresses / elapsed if elapsed > 0 else 0
                        remaining = self.total_addresses - self.completed_addresses
                        eta = remaining / rate if rate > 0 else 0
                        
                        logging.info(f"📊 PROGRESS: {self.completed_addresses:,}/{self.total_addresses:,} "
                                   f"({self.completed_addresses/self.total_addresses*100:.1f}%) | "
                                   f"Rate: {rate:.1f}/sec | ETA: {eta/60:.1f} min")
        
        self.progress_thread = threading.Thread(target=log_progress, daemon=True)
        self.progress_thread.start()
    
    def stop_progress_logging(self):
        """Stop background progress logging."""
        self.stop_progress = True
        if self.progress_thread:
            self.progress_thread.join(timeout=1)
    
    def update_progress(self, count):
        """Update progress counter."""
        with self.progress_lock:
            self.completed_addresses += count
    
    def parse_hybrid_multi_device(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse addresses using hybrid work distribution with Intel GPU."""
        if not addresses:
            return []
        
        total_addresses = len(addresses)
        logging.info(f"🚀 Starting HYBRID multi-device parsing of {total_addresses:,} addresses...")
        
        # Start progress logging
        self.start_progress_logging(total_addresses)
        
        # Initialize all devices
        available_devices = []
        
        # Setup NVIDIA GPU
        if self._setup_nvidia_gpu():
            available_devices.append("NVIDIA_GPU")
            self._device_stats["nvidia_gpu"] = {"processed": 0, "time": 0}
            logging.info("✅ NVIDIA GPU initialized")
        
        # Setup Intel GPU
        if self._setup_intel_openvino():
            available_devices.append("INTEL_GPU")
            self._device_stats["intel_openvino"] = {"processed": 0, "time": 0}
            logging.info("✅ Intel GPU initialized")
        
        # Setup CPU cores
        if self._setup_all_cpu_cores():
            available_devices.append("ALL_CPU_CORES")
            self._device_stats["all_cpu_cores"] = {"processed": 0, "time": 0}
            logging.info(f"✅ {self.cpu_workers} CPU cores initialized")
        
        if not available_devices:
            logging.error("❌ No processing devices available!")
            return [ParsedAddress(parse_success=False, parse_error="No devices available") for _ in addresses]
        
        # Hybrid work distribution
        nvidia_chunk_size = 0
        intel_chunk_size = 0
        cpu_total_chunk_size = 0
        
        if "NVIDIA_GPU" in available_devices:
            # NVIDIA gets 65% of work
            nvidia_chunk_size = int(total_addresses * 0.65)
            logging.info(f"🎯 NVIDIA GPU assigned: {nvidia_chunk_size:,} addresses (65%)")
        
        if "INTEL_GPU" in available_devices:
            # Intel gets 5% of work
            intel_chunk_size = int(total_addresses * 0.05)
            logging.info(f"🎯 Intel GPU assigned: {intel_chunk_size:,} addresses (5%)")
        
        if "ALL_CPU_CORES" in available_devices:
            # CPU cores get remaining 30%
            cpu_total_chunk_size = total_addresses - nvidia_chunk_size - intel_chunk_size
            cpu_chunk_size = cpu_total_chunk_size // self.cpu_workers
            logging.info(f"🎯 CPU cores assigned: {cpu_total_chunk_size:,} addresses (30%) = {cpu_chunk_size:,} per core")
        
        # Prepare results array
        results = [None] * total_addresses
        futures = []
        current_idx = 0
        
        with ThreadPoolExecutor(max_workers=self.cpu_workers + 2) as executor:  # +2 for GPUs
            
            # Submit NVIDIA GPU work
            if nvidia_chunk_size > 0:
                nvidia_chunk = addresses[current_idx:current_idx + nvidia_chunk_size]
                nvidia_future = executor.submit(self._process_on_nvidia_with_progress, nvidia_chunk)
                futures.append((nvidia_future, current_idx, current_idx + nvidia_chunk_size, "NVIDIA_GPU"))
                current_idx += nvidia_chunk_size
                logging.info(f"🚀 NVIDIA GPU processing started: {len(nvidia_chunk):,} addresses")
            
            # Submit Intel GPU work
            if intel_chunk_size > 0:
                intel_chunk = addresses[current_idx:current_idx + intel_chunk_size]
                intel_future = executor.submit(self._process_on_intel_with_progress, intel_chunk)
                futures.append((intel_future, current_idx, current_idx + intel_chunk_size, "INTEL_GPU"))
                current_idx += intel_chunk_size
                logging.info(f"🚀 Intel GPU processing started: {len(intel_chunk):,} addresses")
            
            # Submit CPU work
            if cpu_total_chunk_size > 0:
                cpu_chunk_size = cpu_total_chunk_size // self.cpu_workers
                
                for i in range(self.cpu_workers):
                    start_idx = current_idx + (i * cpu_chunk_size)
                    if i == self.cpu_workers - 1:  # Last worker gets remainder
                        end_idx = total_addresses
                    else:
                        end_idx = start_idx + cpu_chunk_size
                    
                    if start_idx < total_addresses:
                        cpu_chunk = addresses[start_idx:end_idx]
                        cpu_future = executor.submit(self._process_on_cpu_core_with_progress, cpu_chunk, i)
                        futures.append((cpu_future, start_idx, end_idx, f"CPU_CORE_{i}"))
                
                logging.info(f"🚀 {self.cpu_workers} CPU cores processing started")
            
            # Collect results with device-specific timeouts
            for future, start_idx, end_idx, device_name in futures:
                try:
                    # Device-specific timeouts
                    if "NVIDIA" in device_name:
                        timeout = 3600  # 1 hour for NVIDIA
                    elif "INTEL" in device_name:
                        timeout = 7200  # 2 hours for Intel (it's slow)
                    else:
                        timeout = 7200  # 2 hours for CPU
                    
                    chunk_results = future.result(timeout=timeout)
                    results[start_idx:end_idx] = chunk_results
                    
                    chunk_size = end_idx - start_idx
                    logging.info(f"✅ {device_name} completed: {chunk_size:,} addresses")
                    
                except Exception as e:
                    logging.error(f"❌ {device_name} failed: {e}")
                    # Fill with failed results
                    for j in range(start_idx, end_idx):
                        results[j] = ParsedAddress(
                            parse_success=False,
                            parse_error=f"{device_name} error: {str(e)}"
                        )
        
        # Stop progress logging
        self.stop_progress_logging()
        
        # Final statistics
        total_time = time.time() - self.start_time
        success_count = sum(1 for r in results if r and r.parse_success)
        
        logging.info(f"🎉 HYBRID processing complete!")
        logging.info(f"📊 Results: {success_count:,}/{total_addresses:,} success ({success_count/total_addresses*100:.1f}%)")
        logging.info(f"⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logging.info(f"🚀 Speed: {total_addresses/total_time:.1f} addresses/second")
        
        return results
    
    def _process_on_nvidia_with_progress(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process on NVIDIA with progress updates."""
        logging.info(f"🟢 NVIDIA GPU starting: {len(addresses):,} addresses")
        start_time = time.time()
        
        results = []
        batch_size = 100  # Process in smaller batches for progress updates
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            batch_results = self._process_on_nvidia(batch)
            results.extend(batch_results)
            
            # Update progress
            self.update_progress(len(batch))
            
            # Log batch completion
            if (i // batch_size) % 10 == 0:  # Every 10 batches
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0
                logging.info(f"🟢 NVIDIA progress: {len(results):,}/{len(addresses):,} ({rate:.1f}/sec)")
        
        processing_time = time.time() - start_time
        self._device_stats["nvidia_gpu"]["processed"] += len(addresses)
        self._device_stats["nvidia_gpu"]["time"] += processing_time
        
        logging.info(f"✅ NVIDIA GPU completed: {len(addresses):,} addresses in {processing_time:.1f}s")
        return results
    
    def _process_on_intel_with_progress(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process on Intel GPU with progress updates."""
        logging.info(f"🔵 Intel GPU starting: {len(addresses):,} addresses")
        start_time = time.time()
        
        results = []
        batch_size = 25  # Smaller batches for Intel GPU
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            batch_results = self._process_on_intel(batch)
            results.extend(batch_results)
            
            # Update progress
            self.update_progress(len(batch))
            
            # Log progress every 5 batches (Intel is slower)
            if (i // batch_size) % 5 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0
                logging.info(f"🔵 Intel progress: {len(results):,}/{len(addresses):,} ({rate:.1f}/sec)")
        
        processing_time = time.time() - start_time
        self._device_stats["intel_openvino"]["processed"] += len(addresses)
        self._device_stats["intel_openvino"]["time"] += processing_time
        
        logging.info(f"✅ Intel GPU completed: {len(addresses):,} addresses in {processing_time:.1f}s")
        return results
    
    def _process_on_cpu_core_with_progress(self, addresses: List[str], core_id: int) -> List[ParsedAddress]:
        """Process on CPU core with progress updates."""
        logging.info(f"🔵 CPU_CORE_{core_id} starting: {len(addresses):,} addresses")
        start_time = time.time()
        
        results = []
        batch_size = 50  # Smaller batches for CPU
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            batch_results = self._process_on_cpu_core(batch, core_id)
            results.extend(batch_results)
            
            # Update progress
            self.update_progress(len(batch))
            
            # Log progress every 20 batches
            if (i // batch_size) % 20 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0
                logging.info(f"🔵 CPU_CORE_{core_id} progress: {len(results):,}/{len(addresses):,} ({rate:.1f}/sec)")
        
        processing_time = time.time() - start_time
        
        logging.info(f"✅ CPU_CORE_{core_id} completed: {len(addresses):,} addresses in {processing_time:.1f}s")
        return results


def setup_logging():
    """Set up detailed logging for hybrid P2 processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'p2_hybrid_processing_{timestamp}.log', encoding='utf-8')
        ]
    )


def load_p2_addresses() -> List[str]:
    """Load addresses from export_customer_address_store_p2.csv."""
    
    print(f"📂 Loading addresses from export_customer_address_store_p2.csv...")
    
    try:
        # Load the P2 CSV file
        df = pd.read_csv('export_customer_address_store_p2.csv')
        
        print(f"✅ Loaded CSV with {len(df)} rows")
        
        # Check column names
        print(f"📋 Columns: {list(df.columns)}")
        
        # Find address column (common names)
        address_column = None
        for col in ['addr_text', 'address', 'full_address', 'Address', 'addr']:
            if col in df.columns:
                address_column = col
                break
        
        if not address_column:
            print("❌ Could not find address column. Available columns:")
            for col in df.columns:
                print(f"   - {col}")
            return []
        
        print(f"🎯 Using address column: '{address_column}'")
        
        # Filter out empty addresses
        df_clean = df.dropna(subset=[address_column])
        df_clean = df_clean[df_clean[address_column].str.strip() != '']
        
        # Get all addresses
        addresses = df_clean[address_column].tolist()
        
        print(f"✅ Loaded {len(addresses):,} valid addresses for processing")
        return addresses
        
    except FileNotFoundError:
        print("❌ Error: export_customer_address_store_p2.csv not found")
        print("   Please ensure the file exists in the current directory")
        return []
    except Exception as e:
        print(f"❌ Error loading P2 file: {e}")
        return []


def main():
    """Main execution function."""
    
    setup_logging()
    
    print("🚀 P2 HYBRID Multi-Device Address Parser")
    print("Configuration: NVIDIA GPU (65%) + Intel GPU (5%) + 28 CPU cores (30%)")
    print("=" * 80)
    
    # Load P2 addresses
    addresses = load_p2_addresses()
    
    if not addresses:
        print("❌ No addresses to process. Exiting.")
        return None
    
    print(f"📊 Processing Details:")
    print(f"   Total Addresses: {len(addresses):,}")
    print(f"   NVIDIA GPU: ~{int(len(addresses) * 0.65):,} addresses (65%)")
    print(f"   Intel GPU: ~{int(len(addresses) * 0.05):,} addresses (5%)")
    print(f"   CPU Cores: ~{int(len(addresses) * 0.30):,} addresses (30%)")
    print(f"   Progress logging: Every 30 seconds")
    print()
    
    # Configure hybrid parser
    parser = HybridP2Parser(
        batch_size=40,
        use_nvidia_gpu=True,
        use_intel_gpu=True,  # Enable Intel GPU for 5% load
        use_all_cpu_cores=True,
        cpu_core_multiplier=1.0
    )
    
    # Override CPU workers to use 28 cores
    parser.cpu_workers = 28
    parser.max_workers = 30  # 28 CPU + 1 NVIDIA + 1 Intel
    
    logging.info(f"🔧 Hybrid parser configured:")
    logging.info(f"   CPU Workers: {parser.cpu_workers}")
    logging.info(f"   Total Workers: {parser.max_workers}")
    logging.info(f"   Intel GPU: Enabled (5% load with 2-hour timeout)")
    
    try:
        # Process all addresses with hybrid distribution
        start_time = time.time()
        parsed_results = parser.parse_hybrid_multi_device(addresses)
        end_time = time.time()
        
        total_time = end_time - start_time
        success_count = sum(1 for r in parsed_results if r.parse_success)
        failed_count = len(parsed_results) - success_count
        addresses_per_second = len(addresses) / total_time
        
        # Print comprehensive results
        print("\n🎉 P2 HYBRID PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"📊 PROCESSING RESULTS:")
        print(f"   Total Addresses: {len(addresses):,}")
        print(f"   Successfully Parsed: {success_count:,}")
        print(f"   Failed: {failed_count:,}")
        print(f"   Success Rate: {(success_count/len(addresses)*100):.1f}%")
        print()
        
        print(f"⏱️  TIMING RESULTS:")
        print(f"   Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Speed: {addresses_per_second:.1f} addresses/second")
        print(f"   Speed: {addresses_per_second*60:.0f} addresses/minute")
        print()
        
        # Device performance breakdown
        device_stats = parser.get_statistics().get('device_stats', {})
        if device_stats:
            print(f"🔧 DEVICE PERFORMANCE:")
            for device, stats in device_stats.items():
                if stats['processed'] > 0:
                    device_speed = stats['processed'] / stats['time'] if stats['time'] > 0 else 0
                    utilization = (stats['processed'] / len(addresses)) * 100
                    
                    print(f"   {device.upper()}:")
                    print(f"     Processed: {stats['processed']:,} addresses ({utilization:.1f}%)")
                    print(f"     Speed: {device_speed:.1f} addresses/second")
                    print(f"     Time: {stats['time']:.1f} seconds")
        
        # Save results to CSV
        print("\n💾 SAVING RESULTS...")
        
        # Create results DataFrame
        results_data = []
        for i, (original_addr, parsed) in enumerate(zip(addresses, parsed_results)):
            results_data.append({
                'id': i + 1,
                'original_address': original_addr,
                'unit_number': parsed.unit_number if parsed.parse_success else '',
                'society_name': parsed.society_name if parsed.parse_success else '',
                'landmark': parsed.landmark if parsed.parse_success else '',
                'road': parsed.road if parsed.parse_success else '',
                'sub_locality': parsed.sub_locality if parsed.parse_success else '',
                'locality': parsed.locality if parsed.parse_success else '',
                'city': parsed.city if parsed.parse_success else '',
                'district': parsed.district if parsed.parse_success else '',
                'state': parsed.state if parsed.parse_success else '',
                'country': parsed.country if parsed.parse_success else '',
                'pin_code': parsed.pin_code if parsed.parse_success else '',
                'parse_success': parsed.parse_success,
                'parse_error': parsed.parse_error if not parsed.parse_success else '',
                'note': parsed.note if parsed.parse_success else ''
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'p2_hybrid_results_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        
        print(f"✅ Results saved to: {output_file}")
        
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   Configuration: NVIDIA GPU (65%) + Intel GPU (5%) + 28 CPU cores (30%)")
        print(f"   Processing Speed: {addresses_per_second:.1f} addresses/second")
        print(f"   Total Time: {total_time/60:.1f} minutes")
        print(f"   Success Rate: {(success_count/len(addresses)*100):.1f}%")
        
        return {
            'total_addresses': len(addresses),
            'success_count': success_count,
            'total_time': total_time,
            'addresses_per_second': addresses_per_second,
            'output_file': output_file
        }
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        logging.error(f"Hybrid P2 processing failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    results = main()