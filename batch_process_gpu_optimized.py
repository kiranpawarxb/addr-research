#!/usr/bin/env python3
"""
GPU-Optimized Batch processor for P5-P31 files with maximum NVIDIA GPU utilization:
- NVIDIA GPU: 90% of work (maximum utilization)
- Intel GPU: 1% of work (minimal test load)
- 28 CPU cores: 9% of work (minimal backup)
- Larger batch sizes and concurrent GPU streams
- GPU memory optimization and utilization monitoring
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


class GPUOptimizedBatchProcessor(UltimateMultiDeviceParser):
    """GPU-optimized batch processor with maximum NVIDIA utilization."""
    
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
        
        # GPU optimization settings
        self.nvidia_batch_size = 200  # Larger batches for GPU
        self.nvidia_concurrent_streams = 3  # Multiple concurrent GPU streams
    
    def _setup_nvidia_gpu_optimized(self):
        """Set up NVIDIA GPU with maximum optimization."""
        if not self.use_nvidia_gpu:
            return False
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            if not torch.cuda.is_available():
                logging.warning("NVIDIA CUDA not available")
                return False
            
            logging.info("🔧 Setting up OPTIMIZED NVIDIA GPU processing...")
            
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Load model with maximum GPU optimization
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Half precision for speed
                device_map="auto",  # Automatic device mapping
                low_cpu_mem_usage=True
            ).cuda()
            
            # Optimize model for inference
            model.eval()
            model = torch.compile(model, mode="max-autotune")  # PyTorch 2.0 optimization
            
            # Create multiple pipelines for concurrent processing
            self._nvidia_pipelines = []
            for i in range(self.nvidia_concurrent_streams):
                pipeline_obj = pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    device=0,  # CUDA device 0
                    aggregation_strategy="simple",
                    batch_size=self.nvidia_batch_size,
                    return_tensors="pt"
                )
                self._nvidia_pipelines.append(pipeline_obj)
            
            logging.info(f"✅ NVIDIA GPU setup complete with {self.nvidia_concurrent_streams} concurrent streams")
            logging.info(f"   Batch size: {self.nvidia_batch_size}")
            logging.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup optimized NVIDIA GPU: {e}")
            return False
    
    def start_progress_logging(self, total_addresses, file_name):
        """Start background progress logging with GPU monitoring."""
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
                        
                        # Get GPU utilization
                        gpu_util = self._get_gpu_utilization()
                        
                        logging.info(f"📊 PROGRESS [{self.current_file}]: {self.completed_addresses:,}/{self.total_addresses:,} "
                                   f"({self.completed_addresses/self.total_addresses*100:.1f}%) | "
                                   f"Rate: {rate:.1f}/sec | ETA: {eta/60:.1f} min | GPU: {gpu_util:.1f}%")
        
        self.progress_thread = threading.Thread(target=log_progress, daemon=True)
        self.progress_thread.start()
    
    def _get_gpu_utilization(self):
        """Get current GPU utilization percentage."""
        try:
            import torch
            if torch.cuda.is_available():
                # Get GPU memory usage as proxy for utilization
                memory_used = torch.cuda.memory_allocated(0)
                memory_total = torch.cuda.get_device_properties(0).total_memory
                return (memory_used / memory_total) * 100
        except:
            pass
        return 0.0
    
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
    
    def process_single_file_gpu_optimized(self, file_path: str) -> Dict:
        """Process a single P file with GPU optimization."""
        file_name = os.path.basename(file_path)
        logging.info(f"🚀 Starting GPU-optimized processing: {file_name}")
        
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
        
        # Initialize devices with GPU optimization
        available_devices = []
        
        # Setup optimized NVIDIA GPU
        if self._setup_nvidia_gpu_optimized():
            available_devices.append("NVIDIA_GPU_OPTIMIZED")
            if "nvidia_gpu" not in self._device_stats:
                self._device_stats["nvidia_gpu"] = {"processed": 0, "time": 0}
        
        # Setup Intel GPU (minimal)
        if self._setup_intel_openvino():
            available_devices.append("INTEL_GPU")
            if "intel_openvino" not in self._device_stats:
                self._device_stats["intel_openvino"] = {"processed": 0, "time": 0}
        
        # Setup CPU cores (minimal)
        if self._setup_all_cpu_cores():
            available_devices.append("ALL_CPU_CORES")
            if "all_cpu_cores" not in self._device_stats:
                self._device_stats["all_cpu_cores"] = {"processed": 0, "time": 0}
        
        if not available_devices:
            logging.error(f"❌ No processing devices available for {file_name}")
            return None
        
        # GPU-optimized work distribution (90% NVIDIA + 1% Intel + 9% CPU)
        nvidia_chunk_size = int(total_addresses * 0.90)
        intel_chunk_size = int(total_addresses * 0.01)
        cpu_total_chunk_size = total_addresses - nvidia_chunk_size - intel_chunk_size
        
        logging.info(f"🎯 GPU-OPTIMIZED work distribution for {file_name}:")
        logging.info(f"   NVIDIA RTX 4070: {nvidia_chunk_size:,} addresses (90% - MAXIMUM)")
        logging.info(f"   Intel GPU: {intel_chunk_size:,} addresses (1%)")
        logging.info(f"   CPU cores: {cpu_total_chunk_size:,} addresses (9%)")
        
        # Process with GPU optimization
        results = [None] * total_addresses
        futures = []
        current_idx = 0
        
        file_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.cpu_workers + 4) as executor:  # +4 for GPU streams
            
            # Submit NVIDIA GPU work with multiple concurrent streams
            if nvidia_chunk_size > 0 and "NVIDIA_GPU_OPTIMIZED" in available_devices:
                # Split NVIDIA work across multiple concurrent streams
                nvidia_chunk = addresses[current_idx:current_idx + nvidia_chunk_size]
                stream_chunk_size = len(nvidia_chunk) // self.nvidia_concurrent_streams
                
                for stream_id in range(self.nvidia_concurrent_streams):
                    start_idx = stream_id * stream_chunk_size
                    if stream_id == self.nvidia_concurrent_streams - 1:
                        end_idx = len(nvidia_chunk)
                    else:
                        end_idx = start_idx + stream_chunk_size
                    
                    if start_idx < len(nvidia_chunk):
                        stream_chunk = nvidia_chunk[start_idx:end_idx]
                        nvidia_future = executor.submit(self._process_chunk_nvidia_optimized, stream_chunk, stream_id)
                        futures.append((nvidia_future, current_idx + start_idx, current_idx + end_idx, f"NVIDIA_STREAM_{stream_id}"))
                
                current_idx += nvidia_chunk_size
            
            # Submit Intel GPU work (minimal)
            if intel_chunk_size > 0 and "INTEL_GPU" in available_devices:
                intel_chunk = addresses[current_idx:current_idx + intel_chunk_size]
                intel_future = executor.submit(self._process_chunk_intel, intel_chunk)
                futures.append((intel_future, current_idx, current_idx + intel_chunk_size, "INTEL_GPU"))
                current_idx += intel_chunk_size
            
            # Submit CPU work (minimal)
            if cpu_total_chunk_size > 0 and "ALL_CPU_CORES" in available_devices:
                cpu_chunk_size = cpu_total_chunk_size // min(self.cpu_workers, 4)  # Use fewer CPU workers
                
                for i in range(min(self.cpu_workers, 4)):
                    start_idx = current_idx + (i * cpu_chunk_size)
                    if i == min(self.cpu_workers, 4) - 1:
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
                    # Optimized timeouts
                    if "NVIDIA" in device_name:
                        timeout = 10800  # 3 hours for heavy GPU load
                    elif "INTEL" in device_name:
                        timeout = 1800  # 30 minutes
                    else:
                        timeout = 3600  # 1 hour for CPU
                    
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
        
        # Final GPU utilization
        final_gpu_util = self._get_gpu_utilization()
        
        logging.info(f"🎉 {file_name} GPU-OPTIMIZED processing completed!")
        logging.info(f"📊 Results: {success_count:,}/{total_addresses:,} success ({success_count/total_addresses*100:.1f}%)")
        logging.info(f"⏱️ Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logging.info(f"🚀 Speed: {addresses_per_second:.1f} addresses/second")
        logging.info(f"🎯 Final GPU Utilization: {final_gpu_util:.1f}%")
        
        return {
            'file_name': file_name,
            'total_addresses': total_addresses,
            'success_count': success_count,
            'total_time': total_time,
            'addresses_per_second': addresses_per_second,
            'output_file': output_file,
            'gpu_utilization': final_gpu_util
        }
    
    def _process_chunk_nvidia_optimized(self, addresses: List[str], stream_id: int) -> List[ParsedAddress]:
        """Process chunk on NVIDIA GPU with maximum optimization."""
        logging.info(f"🟢 NVIDIA_STREAM_{stream_id} starting: {len(addresses):,} addresses (90% GPU load)")
        start_time = time.time()
        
        results = []
        
        try:
            # Use the specific pipeline for this stream
            pipeline_obj = self._nvidia_pipelines[stream_id % len(self._nvidia_pipelines)]
            
            # Process in larger batches for maximum GPU utilization
            for i in range(0, len(addresses), self.nvidia_batch_size):
                batch = addresses[i:i + self.nvidia_batch_size]
                
                # Clean addresses
                batch_texts = [self._clean_address_text(addr) for addr in batch]
                
                # Process batch on GPU
                entities_batch = pipeline_obj(batch_texts)
                
                # Extract results
                for j, (addr, entities) in enumerate(zip(batch, entities_batch)):
                    parsed = self._extract_fields_from_ner(batch_texts[j], entities)
                    parsed.parse_success = True
                    results.append(parsed)
                
                self.update_progress(len(batch))
                
                # Log progress every 10 batches
                if (i // self.nvidia_batch_size) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0
                    progress_pct = (len(results) / len(addresses)) * 100
                    gpu_util = self._get_gpu_utilization()
                    logging.info(f"🟢 NVIDIA_STREAM_{stream_id} progress: {len(results):,}/{len(addresses):,} "
                               f"({progress_pct:.1f}%) | Rate: {rate:.1f}/sec | GPU: {gpu_util:.1f}%")
                
        except Exception as e:
            logging.error(f"NVIDIA stream {stream_id} error: {e}")
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"NVIDIA stream error: {str(e)}"
                ))
        
        processing_time = time.time() - start_time
        self._device_stats["nvidia_gpu"]["processed"] += len(addresses)
        self._device_stats["nvidia_gpu"]["time"] += processing_time
        
        final_rate = len(addresses) / processing_time if processing_time > 0 else 0
        logging.info(f"✅ NVIDIA_STREAM_{stream_id} completed: {len(addresses):,} addresses in {processing_time:.1f}s ({final_rate:.1f}/sec)")
        return results
    
    def _process_chunk_intel(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process chunk on Intel GPU (minimal load)."""
        logging.info(f"🔵 Intel GPU starting: {len(addresses):,} addresses (1% workload)")
        start_time = time.time()
        
        results = []
        batch_size = 10
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            batch_results = self._process_on_intel(batch)
            results.extend(batch_results)
            self.update_progress(len(batch))
        
        processing_time = time.time() - start_time
        self._device_stats["intel_openvino"]["processed"] += len(addresses)
        self._device_stats["intel_openvino"]["time"] += processing_time
        
        final_rate = len(addresses) / processing_time if processing_time > 0 else 0
        logging.info(f"✅ Intel GPU completed: {len(addresses):,} addresses in {processing_time:.1f}s ({final_rate:.1f}/sec)")
        return results
    
    def _process_chunk_cpu(self, addresses: List[str], core_id: int) -> List[ParsedAddress]:
        """Process chunk on CPU core (minimal load)."""
        logging.info(f"🔵 CPU_CORE_{core_id} starting: {len(addresses):,} addresses (minimal)")
        start_time = time.time()
        
        results = []
        batch_size = 20
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            batch_results = self._process_on_cpu_core(batch, core_id)
            results.extend(batch_results)
            self.update_progress(len(batch))
        
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
        output_file = f'{base_name}_gpu_optimized_{timestamp}.csv'
        
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
    """Set up comprehensive logging for GPU-optimized processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'gpu_optimized_batch_{timestamp}.log', encoding='utf-8')
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
    """Main GPU-optimized batch processing function."""
    
    setup_logging()
    
    print("🚀 GPU-OPTIMIZED BATCH PROCESSOR FOR P5-P31 FILES")
    print("Configuration: NVIDIA RTX 4070 (90% MAX) + Intel GPU (1%) + CPU cores (9%)")
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
    
    # Initialize GPU-optimized processor
    processor = GPUOptimizedBatchProcessor(
        batch_size=200,  # Larger batches for GPU
        use_nvidia_gpu=True,
        use_intel_gpu=True,
        use_all_cpu_cores=True,
        cpu_core_multiplier=0.5  # Fewer CPU workers to focus on GPU
    )
    
    processor.cpu_workers = 4  # Minimal CPU usage
    processor.max_workers = 8  # Focus on GPU streams
    
    print("🔧 Initializing GPU-optimized processing devices...")
    print("🎯 Target: 90%+ NVIDIA GPU utilization")
    print("🎯 Multiple concurrent GPU streams")
    print("🎯 Larger batch sizes for maximum throughput")
    print()
    
    # Process all files
    batch_start_time = time.time()
    all_results = []
    
    for i, file_path in enumerate(files_to_process, 1):
        file_name = os.path.basename(file_path)
        
        print(f"📊 Processing file {i}/{len(files_to_process)}: {file_name}")
        print("-" * 60)
        
        try:
            result = processor.process_single_file_gpu_optimized(file_path)
            if result:
                all_results.append(result)
                print(f"✅ {file_name} completed successfully!")
                print(f"   Speed: {result['addresses_per_second']:.1f} addresses/second")
                print(f"   Success rate: {(result['success_count']/result['total_addresses']*100):.1f}%")
                print(f"   GPU Utilization: {result['gpu_utilization']:.1f}%")
            else:
                print(f"❌ {file_name} failed!")
            
        except Exception as e:
            logging.error(f"Failed to process {file_name}: {e}")
            print(f"❌ {file_name} failed: {e}")
        
        print()
    
    # Final summary
    batch_total_time = time.time() - batch_start_time
    
    print("🎉 GPU-OPTIMIZED BATCH PROCESSING COMPLETE!")
    print("=" * 80)
    
    if all_results:
        total_addresses = sum(r['total_addresses'] for r in all_results)
        total_success = sum(r['success_count'] for r in all_results)
        avg_speed = sum(r['addresses_per_second'] for r in all_results) / len(all_results)
        avg_gpu_util = sum(r['gpu_utilization'] for r in all_results) / len(all_results)
        
        print(f"📊 BATCH SUMMARY:")
        print(f"   Files processed: {len(all_results)}/{len(files_to_process)}")
        print(f"   Total addresses: {total_addresses:,}")
        print(f"   Total successful: {total_success:,}")
        print(f"   Overall success rate: {(total_success/total_addresses*100):.1f}%")
        print(f"   Average speed: {avg_speed:.1f} addresses/second")
        print(f"   Average GPU utilization: {avg_gpu_util:.1f}%")
        print(f"   Total batch time: {batch_total_time/60:.1f} minutes")
        print()
        
        print(f"📁 OUTPUT FILES:")
        for result in all_results:
            print(f"   - {result['output_file']}")
    
    return all_results


if __name__ == "__main__":
    results = main()