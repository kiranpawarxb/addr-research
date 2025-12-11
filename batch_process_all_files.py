#!/usr/bin/env python3
"""
Batch processor for all remaining files (P5-P31) with optimal configuration:
- NVIDIA GPU: 75% of work (primary processor)
- Intel GPU: 1% of work (minimal test load)
- 28 CPU cores: 24% of work (backup processing)
- Progress logging and comprehensive reporting
- Automatic file detection and processing
"""

import sys
import os
import pandas as pd
import logging
import time
import multiprocessing
import glob
from datetime import datetime
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path
sys.path.insert(0, 'src')

from src.ultimate_multi_device_parser import UltimateMultiDeviceParser
from src.models import ParsedAddress


class BatchProcessor(UltimateMultiDeviceParser):
    """Batch processor for multiple P files with optimal configuration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_lock = threading.Lock()
        self.completed_addresses = 0
        self.start_time = None
        self.progress_thread = None
        self.stop_progress = False
        self.current_file = ""
        self.total_files = 0
        self.completed_files = 0
    
    def start_progress_logging(self, total_addresses, file_name):
        """Start background progress logging."""
        self.start_time = time.time()
        self.total_addresses = total_addresses
        self.current_file = file_name
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
                        
                        logging.info(f"📊 PROGRESS [{self.current_file}]: {self.completed_addresses:,}/{self.total_addresses:,} "
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
    
    def reset_progress(self):
        """Reset progress for new file."""
        with self.progress_lock:
            self.completed_addresses = 0
    
    def process_single_file(self, file_path: str) -> Dict:
        """Process a single P file with optimal configuration."""
        file_name = os.path.basename(file_path)
        logging.info(f"🚀 Starting processing: {file_name}")
        
        # Load addresses from file
        addresses = self.load_addresses_from_file(file_path)
        if not addresses:
            logging.error(f"❌ No addresses loaded from {file_name}")
            return None
        
        total_addresses = len(addresses)
        logging.info(f"📊 {file_name}: {total_addresses:,} addresses to process")
        
        # Reset and start progress logging
        self.reset_progress()
        self.start_progress_logging(total_addresses, file_name)
        
        # Initialize devices (reuse existing connections)
        available_devices = []
        
        # Check NVIDIA GPU
        if hasattr(self, '_nvidia_pipeline') and self._nvidia_pipeline:
            available_devices.append("NVIDIA_GPU")
            if "nvidia_gpu" not in self._device_stats:
                self._device_stats["nvidia_gpu"] = {"processed": 0, "time": 0}
        
        # Check Intel GPU
        if hasattr(self, '_intel_pipeline') and self._intel_pipeline:
            available_devices.append("INTEL_GPU")
            if "intel_openvino" not in self._device_stats:
                self._device_stats["intel_openvino"] = {"processed": 0, "time": 0}
        
        # Check CPU cores
        if hasattr(self, '_cpu_pipelines') and self._cpu_pipelines:
            available_devices.append("ALL_CPU_CORES")
            if "all_cpu_cores" not in self._device_stats:
                self._device_stats["all_cpu_cores"] = {"processed": 0, "time": 0}
        
        if not available_devices:
            logging.error(f"❌ No processing devices available for {file_name}")
            return None
        
        # Optimal work distribution (75% NVIDIA + 1% Intel + 24% CPU)
        nvidia_chunk_size = int(total_addresses * 0.75)
        intel_chunk_size = int(total_addresses * 0.01)
        cpu_total_chunk_size = total_addresses - nvidia_chunk_size - intel_chunk_size
        
        logging.info(f"🎯 Work distribution for {file_name}:")
        logging.info(f"   NVIDIA RTX 4070: {nvidia_chunk_size:,} addresses (75%)")
        logging.info(f"   Intel GPU: {intel_chunk_size:,} addresses (1%)")
        logging.info(f"   CPU cores: {cpu_total_chunk_size:,} addresses (24%)")
        
        # Process with optimal distribution
        results = [None] * total_addresses
        futures = []
        current_idx = 0
        
        file_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.cpu_workers + 2) as executor:
            
            # Submit NVIDIA GPU work
            if nvidia_chunk_size > 0 and "NVIDIA_GPU" in available_devices:
                nvidia_chunk = addresses[current_idx:current_idx + nvidia_chunk_size]
                nvidia_future = executor.submit(self._process_chunk_nvidia, nvidia_chunk)
                futures.append((nvidia_future, current_idx, current_idx + nvidia_chunk_size, "NVIDIA_GPU"))
                current_idx += nvidia_chunk_size
            
            # Submit Intel GPU work
            if intel_chunk_size > 0 and "INTEL_GPU" in available_devices:
                intel_chunk = addresses[current_idx:current_idx + intel_chunk_size]
                intel_future = executor.submit(self._process_chunk_intel, intel_chunk)
                futures.append((intel_future, current_idx, current_idx + intel_chunk_size, "INTEL_GPU"))
                current_idx += intel_chunk_size
            
            # Submit CPU work
            if cpu_total_chunk_size > 0 and "ALL_CPU_CORES" in available_devices:
                cpu_chunk_size = cpu_total_chunk_size // self.cpu_workers
                
                for i in range(self.cpu_workers):
                    start_idx = current_idx + (i * cpu_chunk_size)
                    if i == self.cpu_workers - 1:  # Last worker gets remainder
                        end_idx = total_addresses
                    else:
                        end_idx = start_idx + cpu_chunk_size
                    
                    if start_idx < total_addresses:
                        cpu_chunk = addresses[start_idx:end_idx]
                        cpu_future = executor.submit(self._process_chunk_cpu, cpu_chunk, i)
                        futures.append((cpu_future, start_idx, end_idx, f"CPU_CORE_{i}"))
            
            # Collect results
            for future, start_idx, end_idx, device_name in futures:
                try:
                    # Device-specific timeouts
                    if "NVIDIA" in device_name:
                        timeout = 7200  # 2 hours
                    elif "INTEL" in device_name:
                        timeout = 1800  # 30 minutes
                    else:
                        timeout = 5400  # 1.5 hours
                    
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
        
        # Calculate results
        total_time = time.time() - file_start_time
        success_count = sum(1 for r in results if r and r.parse_success)
        addresses_per_second = total_addresses / total_time
        
        # Save results
        output_file = self.save_results(file_name, addresses, results)
        
        # Device performance breakdown
        logging.info(f"🎉 {file_name} completed!")
        logging.info(f"📊 Results: {success_count:,}/{total_addresses:,} success ({success_count/total_addresses*100:.1f}%)")
        logging.info(f"⏱️ Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logging.info(f"🚀 Speed: {addresses_per_second:.1f} addresses/second")
        
        # Log device performance for this file
        logging.info(f"🔧 DEVICE PERFORMANCE [{file_name}]:")
        for device, stats in self._device_stats.items():
            if stats['processed'] > 0:
                device_speed = stats['processed'] / stats['time'] if stats['time'] > 0 else 0
                utilization = (stats['processed'] / total_addresses) * 100
                
                logging.info(f"   {device.upper()}:")
                logging.info(f"     Processed: {stats['processed']:,} addresses ({utilization:.1f}%)")
                logging.info(f"     Speed: {device_speed:.1f} addresses/second")
                logging.info(f"     Time: {stats['time']:.1f} seconds")
        
        return {
            'file_name': file_name,
            'total_addresses': total_addresses,
            'success_count': success_count,
            'total_time': total_time,
            'addresses_per_second': addresses_per_second,
            'output_file': output_file
        }
    
    def _process_chunk_nvidia(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process chunk on NVIDIA GPU with detailed progress."""
        logging.info(f"🟢 NVIDIA RTX 4070 starting: {len(addresses):,} addresses (75% workload)")
        start_time = time.time()
        
        results = []
        batch_size = 140
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            batch_results = self._process_on_nvidia(batch)
            results.extend(batch_results)
            self.update_progress(len(batch))
            
            # Log progress every 15 batches
            if (i // batch_size) % 15 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0
                progress_pct = (len(results) / len(addresses)) * 100
                logging.info(f"🟢 NVIDIA progress: {len(results):,}/{len(addresses):,} ({progress_pct:.1f}%) | Rate: {rate:.1f}/sec")
        
        processing_time = time.time() - start_time
        self._device_stats["nvidia_gpu"]["processed"] += len(addresses)
        self._device_stats["nvidia_gpu"]["time"] += processing_time
        
        final_rate = len(addresses) / processing_time if processing_time > 0 else 0
        logging.info(f"✅ NVIDIA RTX 4070 completed: {len(addresses):,} addresses in {processing_time:.1f}s ({final_rate:.1f}/sec)")
        return results
    
    def _process_chunk_intel(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process chunk on Intel GPU with detailed progress."""
        logging.info(f"🔵 Intel GPU starting: {len(addresses):,} addresses (1% workload)")
        start_time = time.time()
        
        results = []
        batch_size = 10
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            batch_results = self._process_on_intel(batch)
            results.extend(batch_results)
            self.update_progress(len(batch))
            
            # Log progress every 5 batches
            if (i // batch_size) % 5 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0
                logging.info(f"🔵 Intel progress: {len(results):,}/{len(addresses):,} ({rate:.1f}/sec)")
        
        processing_time = time.time() - start_time
        self._device_stats["intel_openvino"]["processed"] += len(addresses)
        self._device_stats["intel_openvino"]["time"] += processing_time
        
        final_rate = len(addresses) / processing_time if processing_time > 0 else 0
        logging.info(f"✅ Intel GPU completed: {len(addresses):,} addresses in {processing_time:.1f}s ({final_rate:.1f}/sec)")
        return results
    
    def _process_chunk_cpu(self, addresses: List[str], core_id: int) -> List[ParsedAddress]:
        """Process chunk on CPU core with detailed progress."""
        logging.info(f"🔵 CPU_CORE_{core_id} starting: {len(addresses):,} addresses (balanced)")
        start_time = time.time()
        
        results = []
        batch_size = 30
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            batch_results = self._process_on_cpu_core(batch, core_id)
            results.extend(batch_results)
            self.update_progress(len(batch))
            
            # Log progress every 15 batches
            if (i // batch_size) % 15 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0
                logging.info(f"🔵 CPU_CORE_{core_id} progress: {len(results):,}/{len(addresses):,} ({rate:.1f}/sec)")
        
        processing_time = time.time() - start_time
        
        final_rate = len(addresses) / processing_time if processing_time > 0 else 0
        logging.info(f"✅ CPU_CORE_{core_id} completed: {len(addresses):,} addresses in {processing_time:.1f}s ({final_rate:.1f}/sec)")
        return results
    
    def load_addresses_from_file(self, file_path: str) -> List[str]:
        """Load addresses from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Find address column
            address_column = None
            for col in ['addr_text', 'address', 'full_address', 'Address', 'addr']:
                if col in df.columns:
                    address_column = col
                    break
            
            if not address_column:
                logging.error(f"No address column found in {file_path}")
                return []
            
            # Filter out empty addresses
            df_clean = df.dropna(subset=[address_column])
            df_clean = df_clean[df_clean[address_column].str.strip() != '']
            
            return df_clean[address_column].tolist()
            
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return []
    
    def save_results(self, file_name: str, addresses: List[str], results: List[ParsedAddress]) -> str:
        """Save processing results to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = file_name.replace('.csv', '')
        output_file = f'{base_name}_processed_{timestamp}.csv'
        
        results_data = []
        for i, (original_addr, parsed) in enumerate(zip(addresses, results)):
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
        results_df.to_csv(output_file, index=False)
        
        return output_file


def setup_logging():
    """Set up comprehensive logging for batch processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'batch_processing_{timestamp}.log', encoding='utf-8')
        ]
    )


def get_files_to_process(start_num: int = 5, end_num: int = 31) -> List[str]:
    """Get list of P files to process."""
    files = []
    for i in range(start_num, end_num + 1):
        file_pattern = f'export_customer_address_store_p{i}.csv'
        matching_files = glob.glob(file_pattern)
        if matching_files:
            files.extend(matching_files)
        else:
            logging.warning(f"File not found: {file_pattern}")
    
    return sorted(files)


def main():
    """Main batch processing function."""
    
    setup_logging()
    
    print("🚀 BATCH PROCESSOR FOR P5-P31 FILES")
    print("Configuration: NVIDIA RTX 4070 (75%) + Intel GPU (1%) + 28 CPU cores (24%)")
    print("=" * 80)
    
    # Get files to process
    files_to_process = get_files_to_process(5, 31)
    
    if not files_to_process:
        print("❌ No files found to process!")
        return
    
    print(f"📂 Found {len(files_to_process)} files to process:")
    for file_path in files_to_process:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"   - {os.path.basename(file_path)} ({file_size:.1f} MB)")
    
    print()
    
    # Initialize batch processor
    processor = BatchProcessor(
        batch_size=45,
        use_nvidia_gpu=True,
        use_intel_gpu=True,
        use_all_cpu_cores=True,
        cpu_core_multiplier=1.0
    )
    
    processor.cpu_workers = 28
    processor.max_workers = 30
    
    # Initialize devices once
    print("🔧 Initializing processing devices...")
    
    devices_initialized = []
    
    # Setup NVIDIA GPU
    if processor._setup_nvidia_gpu():
        devices_initialized.append("NVIDIA_GPU")
        processor._device_stats["nvidia_gpu"] = {"processed": 0, "time": 0}
        print("✅ NVIDIA RTX 4070 initialized")
    
    # Setup Intel GPU
    if processor._setup_intel_openvino():
        devices_initialized.append("INTEL_GPU")
        processor._device_stats["intel_openvino"] = {"processed": 0, "time": 0}
        print("✅ Intel GPU initialized")
    
    # Setup CPU cores
    if processor._setup_all_cpu_cores():
        devices_initialized.append("ALL_CPU_CORES")
        processor._device_stats["all_cpu_cores"] = {"processed": 0, "time": 0}
        print(f"✅ {processor.cpu_workers} CPU cores initialized")
    
    if not devices_initialized:
        print("❌ No processing devices available!")
        return
    
    print(f"🎯 Devices ready: {', '.join(devices_initialized)}")
    print()
    
    # Process all files
    batch_start_time = time.time()
    all_results = []
    
    for i, file_path in enumerate(files_to_process, 1):
        file_name = os.path.basename(file_path)
        
        print(f"📊 Processing file {i}/{len(files_to_process)}: {file_name}")
        print("-" * 60)
        
        # Log file start with estimated size
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logging.info(f"🚀 BATCH PROGRESS: Starting file {i}/{len(files_to_process)} - {file_name} ({file_size_mb:.1f} MB)")
        except:
            logging.info(f"🚀 BATCH PROGRESS: Starting file {i}/{len(files_to_process)} - {file_name}")
        
        try:
            result = processor.process_single_file(file_path)
            if result:
                all_results.append(result)
                print(f"✅ {file_name} completed successfully!")
                print(f"   Speed: {result['addresses_per_second']:.1f} addresses/second")
                print(f"   Success rate: {(result['success_count']/result['total_addresses']*100):.1f}%")
            else:
                print(f"❌ {file_name} failed!")
            
        except Exception as e:
            logging.error(f"Failed to process {file_name}: {e}")
            print(f"❌ {file_name} failed: {e}")
        
        print()
    
    # Final summary
    batch_total_time = time.time() - batch_start_time
    
    print("🎉 BATCH PROCESSING COMPLETE!")
    print("=" * 80)
    
    if all_results:
        total_addresses = sum(r['total_addresses'] for r in all_results)
        total_success = sum(r['success_count'] for r in all_results)
        avg_speed = sum(r['addresses_per_second'] for r in all_results) / len(all_results)
        
        print(f"📊 BATCH SUMMARY:")
        print(f"   Files processed: {len(all_results)}/{len(files_to_process)}")
        print(f"   Total addresses: {total_addresses:,}")
        print(f"   Total successful: {total_success:,}")
        print(f"   Overall success rate: {(total_success/total_addresses*100):.1f}%")
        print(f"   Average speed: {avg_speed:.1f} addresses/second")
        print(f"   Total batch time: {batch_total_time/60:.1f} minutes")
        print()
        
        print(f"📁 OUTPUT FILES:")
        for result in all_results:
            print(f"   - {result['output_file']}")
    
    return all_results


if __name__ == "__main__":
    results = main()