#!/usr/bin/env python3
"""
Dataset-Optimized Processor for Maximum GPU Efficiency:
- Uses HuggingFace datasets for proper GPU batching (eliminates "use dataset" warning)
- NVIDIA GPU: 95% allocation with dataset batching
- CPU cores: 5% allocation for remaining work
- Batch size 250+ for maximum GPU throughput
- Smart file detection and real-time monitoring
"""

import sys
import os
import pandas as pd
import logging
import time
import multiprocessing
import glob
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, 'src')

from src.ultimate_multi_device_parser import UltimateMultiDeviceParser
from src.models import ParsedAddress


class DatasetOptimizedProcessor(UltimateMultiDeviceParser):
    """Dataset-optimized processor for maximum GPU efficiency."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_lock = threading.Lock()
        self.completed_addresses = 0
        self.start_time = None
        self.progress_thread = None
        self.stop_progress = False
        self.current_file = ""
        
        # Dataset optimization settings
        self.nvidia_batch_size = 300  # Large batches for dataset processing
        self.cpu_batch_size = 25      # CPU batch size
        self.cpu_workers = 8          # Fewer CPU cores to focus on GPU
        
        self.progress_lock = threading.Lock()
    
    def get_processed_files(self) -> Set[str]:
        """Get set of already processed P files."""
        processed = set()
        
        # Look for output files in current directory
        for file in os.listdir('.'):
            if file.startswith('export_customer_address_store_p') and ('_results_' in file or '_gpu_' in file or '_sustained_' in file or '_balanced_' in file or '_optimized_' in file or '_hybrid_' in file or '_dataset_' in file):
                # Extract P number from output filename
                if '_p' in file:
                    try:
                        p_part = file.split('_p')[1].split('_')[0]
                        if p_part.isdigit():
                            processed.add(f'p{p_part}')
                    except:
                        pass
            
            # Also check for specific result files
            if file.startswith('p') and '_results_' in file:
                try:
                    p_num = file.split('_')[0][1:]  # Remove 'p' prefix
                    if p_num.isdigit():
                        processed.add(f'p{p_num}')
                except:
                    pass
        
        return processed
    
    def get_files_to_process(self, start_num: int = 5, end_num: int = 31) -> List[str]:
        """Get list of P files to process, excluding already processed ones."""
        processed_files = self.get_processed_files()
        logging.info(f"Already processed files: {sorted(processed_files)}")
        
        files_to_process = []
        skipped_files = []
        
        for i in range(start_num, end_num + 1):
            p_identifier = f'p{i}'
            file_pattern = f'export_customer_address_store_p{i}.csv'
            
            if p_identifier in processed_files:
                skipped_files.append(file_pattern)
                continue
            
            matching_files = glob.glob(file_pattern)
            if matching_files:
                files_to_process.extend(matching_files)
            else:
                logging.warning(f"File not found: {file_pattern}")
        
        logging.info(f"Skipped {len(skipped_files)} already processed files: {skipped_files}")
        logging.info(f"Found {len(files_to_process)} files to process")
        
        return sorted(files_to_process)
    
    def _setup_nvidia_gpu_dataset(self):
        """Set up NVIDIA GPU with dataset optimization."""
        if not self.use_nvidia_gpu:
            return False
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            from datasets import Dataset
            
            if not torch.cuda.is_available():
                logging.warning("NVIDIA CUDA not available")
                return False
            
            logging.info("🔧 Setting up DATASET-OPTIMIZED NVIDIA GPU processing...")
            
            # Advanced GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            
            # Load model with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).cuda()
            
            self.model.eval()
            
            # Create dataset-optimized pipeline
            self._nvidia_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0,
                aggregation_strategy="simple",
                batch_size=self.nvidia_batch_size,
                framework="pt"
            )
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logging.info(f"✅ DATASET-OPTIMIZED NVIDIA GPU setup complete:")
            logging.info(f"   GPU: {gpu_name}")
            logging.info(f"   Memory: {gpu_memory:.1f} GB")
            logging.info(f"   Batch size: {self.nvidia_batch_size}")
            logging.info(f"   Dataset optimization: ENABLED")
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup dataset-optimized NVIDIA GPU: {e}")
            return False
    
    def _setup_cpu_cores_dataset(self):
        """Set up CPU cores for dataset processing."""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            logging.info(f"🔧 Setting up {self.cpu_workers} CPU cores for dataset processing...")
            
            # Create CPU pipelines (limited number to avoid memory issues)
            actual_cpu_pipelines = min(self.cpu_workers, 4)
            self._cpu_pipelines = []
            
            for i in range(actual_cpu_pipelines):
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                
                cpu_pipeline = pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU
                    aggregation_strategy="simple",
                    batch_size=self.cpu_batch_size
                )
                
                self._cpu_pipelines.append(cpu_pipeline)
            
            logging.info(f"✅ Created {len(self._cpu_pipelines)} CPU processing pipelines for dataset mode")
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup CPU cores: {e}")
            return False
    
    def _process_on_nvidia_dataset(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process addresses on NVIDIA GPU using dataset optimization."""
        logging.info(f"🟢 NVIDIA GPU starting: {len(addresses):,} addresses (95% DATASET-OPTIMIZED)")
        start_time = time.time()
        
        results = []
        
        try:
            from datasets import Dataset
            
            # Clean addresses first
            cleaned_addresses = [self._clean_address_text(addr) for addr in addresses]
            
            # Create HuggingFace dataset for optimal GPU processing
            dataset = Dataset.from_dict({"text": cleaned_addresses})
            
            logging.info(f"🟢 Created dataset with {len(dataset):,} entries for GPU processing")
            
            # Process using dataset (this eliminates the "use dataset" warning)
            batch_results = []
            batch_size = self.nvidia_batch_size
            
            for i in range(0, len(dataset), batch_size):
                batch_data = dataset[i:i + batch_size]
                batch_texts = batch_data["text"]
                
                # Process batch on GPU
                entities_batch = self._nvidia_pipeline(batch_texts)
                
                # Extract results for this batch
                for j, (addr, entities) in enumerate(zip(addresses[i:i + len(batch_texts)], entities_batch)):
                    parsed = self._extract_fields_from_ner(batch_texts[j], entities)
                    parsed.parse_success = True
                    batch_results.append(parsed)
                
                # Update progress
                self.update_progress(len(batch_texts))
                
                # Log progress every 5 batches
                if (i // batch_size) % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = len(batch_results) / elapsed if elapsed > 0 else 0
                    progress_pct = (len(batch_results) / len(addresses)) * 100
                    gpu_util = self._get_gpu_utilization()
                    logging.info(f"🟢 NVIDIA dataset progress: {len(batch_results):,}/{len(addresses):,} "
                               f"({progress_pct:.1f}%) | Rate: {rate:.1f}/sec | GPU: {gpu_util:.1f}%")
            
            results = batch_results
                
        except Exception as e:
            logging.error(f"NVIDIA dataset processing error: {e}")
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"NVIDIA dataset error: {str(e)}"
                ))
        
        processing_time = time.time() - start_time
        final_rate = len(addresses) / processing_time if processing_time > 0 else 0
        final_gpu_util = self._get_gpu_utilization()
        
        logging.info(f"✅ NVIDIA GPU completed: {len(addresses):,} addresses in {processing_time:.1f}s "
                    f"({final_rate:.1f}/sec) | Final GPU: {final_gpu_util:.1f}%")
        return results
    
    def _process_chunk_cpu_dataset(self, addresses: List[str], core_id: int) -> List[ParsedAddress]:
        """Process chunk on CPU core in dataset mode."""
        logging.info(f"🔵 CPU_CORE_{core_id} starting: {len(addresses):,} addresses (dataset mode)")
        start_time = time.time()
        
        results = []
        
        try:
            pipeline_obj = self._cpu_pipelines[core_id % len(self._cpu_pipelines)]
            
            # Process in CPU batches
            for i in range(0, len(addresses), self.cpu_batch_size):
                batch = addresses[i:i + self.cpu_batch_size]
                
                # Clean addresses
                batch_texts = [self._clean_address_text(addr) for addr in batch]
                
                # Process batch on CPU
                entities_batch = pipeline_obj(batch_texts)
                
                # Extract results
                for j, (addr, entities) in enumerate(zip(batch, entities_batch)):
                    parsed = self._extract_fields_from_ner(batch_texts[j], entities)
                    parsed.parse_success = True
                    results.append(parsed)
                
                self.update_progress(len(batch))
                
                # Log progress every 30 batches
                if (i // self.cpu_batch_size) % 30 == 0:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0
                    logging.info(f"🔵 CPU_CORE_{core_id} progress: {len(results):,}/{len(addresses):,} ({rate:.1f}/sec)")
                
        except Exception as e:
            logging.error(f"CPU core {core_id} error: {e}")
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"CPU core error: {str(e)}"
                ))
        
        processing_time = time.time() - start_time
        final_rate = len(addresses) / processing_time if processing_time > 0 else 0
        logging.info(f"✅ CPU_CORE_{core_id} completed: {len(addresses):,} addresses in {processing_time:.1f}s ({final_rate:.1f}/sec)")
        return results
    
    def _get_gpu_utilization(self):
        """Get current GPU utilization percentage."""
        try:
            import subprocess
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                util = float(result.stdout.strip())
                return util
        except:
            pass
        return 0.0
    
    def start_progress_logging(self, total_addresses, file_name):
        """Start background progress logging with dataset monitoring."""
        
        self.start_time = time.time()
        self.total_addresses = total_addresses
        self.current_file = file_name
        self.stop_progress = False
        
        def log_progress():
            while not self.stop_progress:
                time.sleep(15)  # Log every 15 seconds for dataset monitoring
                if not self.stop_progress:
                    with self.progress_lock:
                        elapsed = time.time() - self.start_time
                        rate = self.completed_addresses / elapsed if elapsed > 0 else 0
                        remaining = self.total_addresses - self.completed_addresses
                        eta = remaining / rate if rate > 0 else 0
                        
                        # Get GPU utilization
                        gpu_util = self._get_gpu_utilization()
                        
                        logging.info(f"📊 DATASET [{self.current_file}]: {self.completed_addresses:,}/{self.total_addresses:,} "
                                   f"({self.completed_addresses/self.total_addresses*100:.1f}%) | "
                                   f"Rate: {rate:.1f}/sec | ETA: {eta/60:.1f} min | GPU: {gpu_util:.1f}%")
        
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
    
    def process_single_file_dataset(self, file_path: str) -> Dict:
        """Process a single P file with dataset optimization."""
        file_name = os.path.basename(file_path)
        logging.info(f"🚀 Starting DATASET-OPTIMIZED processing: {file_name}")
        
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
        
        # Initialize devices
        available_devices = []
        
        # Setup dataset-optimized NVIDIA GPU
        if self._setup_nvidia_gpu_dataset():
            available_devices.append("NVIDIA_GPU_DATASET")
            if "nvidia_gpu" not in self._device_stats:
                self._device_stats["nvidia_gpu"] = {"processed": 0, "time": 0}
        
        # Setup CPU cores
        if self._setup_cpu_cores_dataset():
            available_devices.append("CPU_CORES")
            if "cpu_cores" not in self._device_stats:
                self._device_stats["cpu_cores"] = {"processed": 0, "time": 0}
        
        if not available_devices:
            logging.error(f"❌ No processing devices available for {file_name}")
            return None
        
        # Dataset-optimized work distribution (95% GPU + 5% CPU)
        gpu_chunk_size = int(total_addresses * 0.95)
        cpu_chunk_size = total_addresses - gpu_chunk_size
        
        logging.info(f"🎯 DATASET-OPTIMIZED work distribution for {file_name}:")
        logging.info(f"   NVIDIA RTX 4070: {gpu_chunk_size:,} addresses (95% - DATASET)")
        logging.info(f"   CPU cores ({self.cpu_workers}): {cpu_chunk_size:,} addresses (5%)")
        
        # Process with dataset optimization
        results = [None] * total_addresses
        futures = []
        current_idx = 0
        
        file_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.cpu_workers + 2) as executor:
            
            # Submit NVIDIA GPU work (dataset-optimized)
            if gpu_chunk_size > 0 and "NVIDIA_GPU_DATASET" in available_devices:
                gpu_chunk = addresses[current_idx:current_idx + gpu_chunk_size]
                gpu_future = executor.submit(self._process_on_nvidia_dataset, gpu_chunk)
                futures.append((gpu_future, current_idx, current_idx + gpu_chunk_size, "NVIDIA_GPU_DATASET"))
                current_idx += gpu_chunk_size
            
            # Submit CPU work (minimal)
            if cpu_chunk_size > 0 and "CPU_CORES" in available_devices:
                cpu_per_worker = cpu_chunk_size // self.cpu_workers
                
                for i in range(self.cpu_workers):
                    start_idx = current_idx + (i * cpu_per_worker)
                    if i == self.cpu_workers - 1:  # Last worker gets remainder
                        end_idx = total_addresses
                    else:
                        end_idx = start_idx + cpu_per_worker
                    
                    if start_idx < total_addresses:
                        cpu_chunk = addresses[start_idx:end_idx]
                        cpu_future = executor.submit(self._process_chunk_cpu_dataset, cpu_chunk, i)
                        futures.append((cpu_future, start_idx, end_idx, f"CPU_CORE_{i}"))
            
            # Collect results
            for future, start_idx, end_idx, device_name in futures:
                try:
                    # Optimized timeouts
                    if "NVIDIA" in device_name:
                        timeout = 7200  # 2 hours for GPU
                    else:
                        timeout = 1800  # 30 minutes for CPU
                    
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
        
        logging.info(f"🎉 {file_name} DATASET-OPTIMIZED processing completed!")
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
    
    def load_addresses_from_file(self, file_path: str) -> List[str]:
        """Load addresses from a CSV file."""
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
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
        output_file = f'{base_name}_dataset_optimized_{timestamp}.csv'
        
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
    """Set up comprehensive logging for dataset-optimized processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'dataset_optimized_{timestamp}.log', encoding='utf-8')
        ]
    )


def main():
    """Main dataset-optimized processing function."""
    
    setup_logging()
    
    print("🚀 DATASET-OPTIMIZED PROCESSOR")
    print("Configuration: NVIDIA RTX 4070 (95% DATASET) + CPU (5%)")
    print("Smart file detection - skips already processed files")
    print("=" * 80)
    
    # Initialize processor
    processor = DatasetOptimizedProcessor(
        batch_size=300,  # Large dataset batches
        use_nvidia_gpu=True,
        use_intel_gpu=False,  # Focus on NVIDIA + CPU
        use_all_cpu_cores=True,
        cpu_core_multiplier=0.25
    )
    
    processor.cpu_workers = 8  # Fewer CPU cores to focus on GPU
    processor.max_workers = 10  # 8 CPU + 1 GPU + coordinator
    
    # Get files to process (automatically skips processed ones)
    files_to_process = processor.get_files_to_process(5, 31)
    
    if not files_to_process:
        print("✅ All files have been processed!")
        return
    
    print(f"📂 Found {len(files_to_process)} files to process:")
    for file_path in files_to_process:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"   - {os.path.basename(file_path)} ({file_size:.1f} MB)")
    
    print()
    print("🔧 Initializing DATASET-OPTIMIZED processing...")
    print("🎯 HuggingFace datasets for optimal GPU batching")
    print("🎯 Batch size 300 for maximum GPU throughput")
    print("🎯 95% GPU allocation with dataset optimization")
    print("🎯 Target: 1500+ addresses/second")
    print()
    
    # Process all files
    batch_start_time = time.time()
    all_results = []
    
    for i, file_path in enumerate(files_to_process, 1):
        file_name = os.path.basename(file_path)
        
        print(f"📊 Processing file {i}/{len(files_to_process)}: {file_name}")
        print("-" * 60)
        
        try:
            result = processor.process_single_file_dataset(file_path)
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
    
    print("🎉 DATASET-OPTIMIZED PROCESSING COMPLETE!")
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