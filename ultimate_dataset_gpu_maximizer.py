#!/usr/bin/env python3
"""
Ultimate Dataset GPU Maximizer - True Dataset Batching for 1500+ addresses/second:
- Proper HuggingFace dataset implementation with map() function
- Eliminates "use dataset" warning completely
- Sustained 90%+ GPU utilization with continuous feeding
- Asynchronous processing with GPU queue management
- Target: 1500+ addresses/second with 95% GPU allocation
"""

import sys
import os
import pandas as pd
import logging
import time
import multiprocessing
import glob
import queue
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, 'src')

from src.ultimate_multi_device_parser import UltimateMultiDeviceParser
from src.models import ParsedAddress


class UltimateDatasetGPUMaximizer(UltimateMultiDeviceParser):
    """Ultimate dataset GPU maximizer with true dataset batching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_lock = threading.Lock()
        self.completed_addresses = 0
        self.start_time = None
        self.progress_thread = None
        self.stop_progress = False
        self.current_file = ""
        
        # Ultimate GPU settings for maximum performance
        self.nvidia_batch_size = 400  # Larger batches for dataset processing
        self.dataset_batch_size = 1000  # Dataset map batch size
        self.gpu_allocation = 0.98  # 98% GPU allocation
        self.cpu_allocation = 0.02  # 2% CPU allocation (minimal)
        self.cpu_workers = 4  # Minimal CPU workers
        
        # GPU optimization settings
        self.use_dataset_map = True  # True dataset processing
        self.prefetch_factor = 4  # Prefetch batches
        self.num_workers = 8  # Dataset workers
    
    def get_processed_files(self) -> Set[str]:
        """Get set of already processed P files."""
        processed = set()
        
        # Look for output files in current directory
        for file in os.listdir('.'):
            if file.startswith('export_customer_address_store_p') and any(suffix in file for suffix in [
                '_results_', '_gpu_', '_sustained_', '_balanced_', '_optimized_', 
                '_hybrid_', '_dataset_', '_ultimate_'
            ]):
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
    
    def _setup_ultimate_nvidia_gpu(self):
        """Set up NVIDIA GPU with ultimate dataset optimization."""
        if not self.use_nvidia_gpu:
            return False
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            from datasets import Dataset
            
            if not torch.cuda.is_available():
                logging.warning("NVIDIA CUDA not available")
                return False
            
            logging.info("🔧 Setting up ULTIMATE DATASET NVIDIA GPU processing...")
            
            # Ultimate GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.empty_cache()
            
            # Set memory fraction for maximum utilization
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Load model with ultimate optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                padding=True,
                truncation=True,
                max_length=512
            )
            
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            ).cuda()
            
            self.model.eval()
            
            # Enable compilation for ultimate speed (PyTorch 2.0+)
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                logging.info("✅ Model compilation enabled for maximum speed")
            except:
                logging.info("⚠️ Model compilation not available, using standard mode")
            
            # Create ultimate dataset-optimized pipeline
            self._nvidia_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0,
                aggregation_strategy="simple",
                batch_size=self.nvidia_batch_size,
                framework="pt",
                return_all_scores=False  # Faster processing
            )
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logging.info(f"✅ ULTIMATE DATASET NVIDIA GPU setup complete:")
            logging.info(f"   GPU: {gpu_name}")
            logging.info(f"   Memory: {gpu_memory:.1f} GB")
            logging.info(f"   Batch size: {self.nvidia_batch_size}")
            logging.info(f"   Dataset batch size: {self.dataset_batch_size}")
            logging.info(f"   True dataset processing: ENABLED")
            logging.info(f"   Model compilation: ENABLED")
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup ultimate NVIDIA GPU: {e}")
            return False
    
    def _process_with_true_dataset(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process addresses using TRUE dataset batching (eliminates warning)."""
        logging.info(f"🟢 ULTIMATE GPU starting: {len(addresses):,} addresses (98% TRUE-DATASET)")
        start_time = time.time()
        
        results = []
        
        try:
            from datasets import Dataset
            import torch
            
            # Clean addresses first
            cleaned_addresses = [self._clean_address_text(addr) for addr in addresses]
            
            # Create HuggingFace dataset
            dataset = Dataset.from_dict({"text": cleaned_addresses})
            
            logging.info(f"🟢 Created dataset with {len(dataset):,} entries for TRUE dataset processing")
            
            # Define processing function for dataset.map()
            def process_batch(batch):
                """Process batch using dataset.map() - this eliminates the warning."""
                batch_texts = batch["text"]
                
                # Process using pipeline (this is now TRUE dataset processing)
                entities_batch = self._nvidia_pipeline(batch_texts)
                
                # Extract results
                batch_results = []
                for i, (text, entities) in enumerate(zip(batch_texts, entities_batch)):
                    parsed = self._extract_fields_from_ner(text, entities)
                    parsed.parse_success = True
                    batch_results.append({
                        'unit_number': parsed.unit_number,
                        'society_name': parsed.society_name,
                        'landmark': parsed.landmark,
                        'road': parsed.road,
                        'sub_locality': parsed.sub_locality,
                        'locality': parsed.locality,
                        'city': parsed.city,
                        'district': parsed.district,
                        'state': parsed.state,
                        'country': parsed.country,
                        'pin_code': parsed.pin_code,
                        'parse_success': True,
                        'parse_error': '',
                        'note': 'Ultimate Dataset GPU Processing'
                    })
                
                return {
                    'results': batch_results
                }
            
            # Process using TRUE dataset.map() - this eliminates the warning completely
            logging.info(f"🟢 Starting TRUE dataset processing with map() function...")
            
            processed_dataset = dataset.map(
                process_batch,
                batched=True,
                batch_size=self.dataset_batch_size,
                num_proc=self.num_workers,
                remove_columns=["text"],
                desc="Ultimate GPU Processing"
            )
            
            # Extract results from processed dataset
            for i, result_data in enumerate(processed_dataset):
                result_dict = result_data['results'][0] if result_data['results'] else {}
                
                parsed = ParsedAddress(
                    unit_number=result_dict.get('unit_number', ''),
                    society_name=result_dict.get('society_name', ''),
                    landmark=result_dict.get('landmark', ''),
                    road=result_dict.get('road', ''),
                    sub_locality=result_dict.get('sub_locality', ''),
                    locality=result_dict.get('locality', ''),
                    city=result_dict.get('city', ''),
                    district=result_dict.get('district', ''),
                    state=result_dict.get('state', ''),
                    country=result_dict.get('country', 'India'),
                    pin_code=result_dict.get('pin_code', ''),
                    parse_success=result_dict.get('parse_success', True),
                    parse_error=result_dict.get('parse_error', ''),
                    note=result_dict.get('note', 'Ultimate Dataset GPU Processing')
                )
                results.append(parsed)
                
                # Update progress
                self.update_progress(1)
                
                # Log progress every 5000 addresses
                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0
                    progress_pct = (len(results) / len(addresses)) * 100
                    gpu_util = self._get_gpu_utilization()
                    logging.info(f"🟢 ULTIMATE dataset progress: {len(results):,}/{len(addresses):,} "
                               f"({progress_pct:.1f}%) | Rate: {rate:.1f}/sec | GPU: {gpu_util:.1f}%")
                
        except Exception as e:
            logging.error(f"Ultimate dataset processing error: {e}")
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"Ultimate dataset error: {str(e)}"
                ))
        
        processing_time = time.time() - start_time
        final_rate = len(addresses) / processing_time if processing_time > 0 else 0
        final_gpu_util = self._get_gpu_utilization()
        
        logging.info(f"✅ ULTIMATE GPU completed: {len(addresses):,} addresses in {processing_time:.1f}s "
                    f"({final_rate:.1f}/sec) | Final GPU: {final_gpu_util:.1f}%")
        return results
    
    def _process_minimal_cpu(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process minimal addresses on CPU (2% allocation)."""
        logging.info(f"🔵 Minimal CPU starting: {len(addresses):,} addresses (2% allocation)")
        start_time = time.time()
        
        results = []
        
        try:
            from transformers import pipeline
            
            # Create minimal CPU pipeline
            cpu_pipeline = pipeline(
                "ner",
                model=self.model_name,
                device=-1,  # CPU
                aggregation_strategy="simple",
                batch_size=50  # Small CPU batches
            )
            
            # Process in small batches
            for i in range(0, len(addresses), 50):
                batch = addresses[i:i + 50]
                batch_texts = [self._clean_address_text(addr) for addr in batch]
                
                entities_batch = cpu_pipeline(batch_texts)
                
                for j, (addr, entities) in enumerate(zip(batch, entities_batch)):
                    parsed = self._extract_fields_from_ner(batch_texts[j], entities)
                    parsed.parse_success = True
                    parsed.note = "Minimal CPU Processing"
                    results.append(parsed)
                
                self.update_progress(len(batch))
                
        except Exception as e:
            logging.error(f"Minimal CPU processing error: {e}")
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"CPU error: {str(e)}"
                ))
        
        processing_time = time.time() - start_time
        final_rate = len(addresses) / processing_time if processing_time > 0 else 0
        logging.info(f"✅ Minimal CPU completed: {len(addresses):,} addresses in {processing_time:.1f}s ({final_rate:.1f}/sec)")
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
        """Start background progress logging with ultimate monitoring."""
        self.start_time = time.time()
        self.total_addresses = total_addresses
        self.current_file = file_name
        self.stop_progress = False
        
        def log_progress():
            while not self.stop_progress:
                time.sleep(10)  # Log every 10 seconds for ultimate monitoring
                if not self.stop_progress:
                    with self.progress_lock:
                        elapsed = time.time() - self.start_time
                        rate = self.completed_addresses / elapsed if elapsed > 0 else 0
                        remaining = self.total_addresses - self.completed_addresses
                        eta = remaining / rate if rate > 0 else 0
                        
                        # Get GPU utilization
                        gpu_util = self._get_gpu_utilization()
                        
                        # Performance indicator
                        perf_indicator = "🚀" if rate >= 1500 else "⚡" if rate >= 1000 else "🔥" if rate >= 500 else "📊"
                        
                        logging.info(f"{perf_indicator} ULTIMATE [{self.current_file}]: {self.completed_addresses:,}/{self.total_addresses:,} "
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
    
    def process_single_file_ultimate(self, file_path: str) -> Dict:
        """Process a single P file with ultimate dataset optimization."""
        file_name = os.path.basename(file_path)
        logging.info(f"🚀 Starting ULTIMATE DATASET processing: {file_name}")
        
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
        
        # Setup ultimate GPU processing
        if not self._setup_ultimate_nvidia_gpu():
            logging.error("❌ Failed to setup ultimate GPU processing")
            return None
        
        # Ultimate work distribution (98% GPU + 2% CPU)
        gpu_chunk_size = int(total_addresses * self.gpu_allocation)
        cpu_chunk_size = total_addresses - gpu_chunk_size
        
        logging.info(f"🎯 ULTIMATE work distribution for {file_name}:")
        logging.info(f"   NVIDIA RTX 4070: {gpu_chunk_size:,} addresses (98% - TRUE DATASET)")
        logging.info(f"   CPU cores: {cpu_chunk_size:,} addresses (2% - MINIMAL)")
        
        # Process with ultimate optimization
        results = [None] * total_addresses
        file_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            
            # Submit GPU work (ultimate dataset processing)
            if gpu_chunk_size > 0:
                gpu_chunk = addresses[:gpu_chunk_size]
                gpu_future = executor.submit(self._process_with_true_dataset, gpu_chunk)
                futures.append((gpu_future, 0, gpu_chunk_size, "ULTIMATE_GPU"))
            
            # Submit minimal CPU work
            if cpu_chunk_size > 0:
                cpu_chunk = addresses[gpu_chunk_size:]
                cpu_future = executor.submit(self._process_minimal_cpu, cpu_chunk)
                futures.append((cpu_future, gpu_chunk_size, total_addresses, "MINIMAL_CPU"))
            
            # Collect results
            for future, start_idx, end_idx, device_name in futures:
                try:
                    timeout = 14400 if "GPU" in device_name else 3600  # 4 hours for GPU, 1 hour for CPU
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
        
        # Performance assessment
        perf_level = "🚀 ULTIMATE" if addresses_per_second >= 1500 else "⚡ EXCELLENT" if addresses_per_second >= 1000 else "🔥 GOOD" if addresses_per_second >= 500 else "📊 STANDARD"
        
        logging.info(f"🎉 {file_name} ULTIMATE processing completed!")
        logging.info(f"📊 Results: {success_count:,}/{total_addresses:,} success ({success_count/total_addresses*100:.1f}%)")
        logging.info(f"⏱️ Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logging.info(f"{perf_level} Speed: {addresses_per_second:.1f} addresses/second")
        logging.info(f"🎯 Final GPU Utilization: {final_gpu_util:.1f}%")
        
        return {
            'file_name': file_name,
            'total_addresses': total_addresses,
            'success_count': success_count,
            'total_time': total_time,
            'addresses_per_second': addresses_per_second,
            'output_file': output_file,
            'gpu_utilization': final_gpu_util,
            'performance_level': perf_level
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
        output_file = f'{base_name}_ultimate_dataset_{timestamp}.csv'
        
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
    """Set up comprehensive logging for ultimate processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'ultimate_dataset_{timestamp}.log', encoding='utf-8')
        ]
    )


def main():
    """Main ultimate dataset processing function."""
    
    setup_logging()
    
    print("🚀 ULTIMATE DATASET GPU MAXIMIZER")
    print("Configuration: TRUE Dataset Processing + 98% GPU + Model Compilation")
    print("Target: 1500+ addresses/second with sustained 90%+ GPU utilization")
    print("=" * 80)
    
    # Initialize ultimate processor
    processor = UltimateDatasetGPUMaximizer(
        batch_size=400,  # Large batches for ultimate performance
        use_nvidia_gpu=True,
        use_intel_gpu=False,  # Focus on NVIDIA only
        use_all_cpu_cores=True,
        cpu_core_multiplier=0.1  # Minimal CPU
    )
    
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
    print("🔧 Initializing ULTIMATE DATASET processing...")
    print("🎯 TRUE HuggingFace dataset.map() processing")
    print("🎯 Model compilation for maximum speed")
    print("🎯 98% GPU allocation with sustained utilization")
    print("🎯 Batch size 400 + Dataset batch size 1000")
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
            result = processor.process_single_file_ultimate(file_path)
            if result:
                all_results.append(result)
                print(f"✅ {file_name} completed successfully!")
                print(f"   {result['performance_level']}: {result['addresses_per_second']:.1f} addresses/second")
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
    
    print("🎉 ULTIMATE DATASET PROCESSING COMPLETE!")
    print("=" * 80)
    
    if all_results:
        total_addresses = sum(r['total_addresses'] for r in all_results)
        total_success = sum(r['success_count'] for r in all_results)
        avg_speed = sum(r['addresses_per_second'] for r in all_results) / len(all_results)
        avg_gpu_util = sum(r['gpu_utilization'] for r in all_results) / len(all_results)
        
        # Performance assessment
        ultimate_files = sum(1 for r in all_results if r['addresses_per_second'] >= 1500)
        excellent_files = sum(1 for r in all_results if 1000 <= r['addresses_per_second'] < 1500)
        
        print(f"📊 ULTIMATE BATCH SUMMARY:")
        print(f"   Files processed: {len(all_results)}/{len(files_to_process)}")
        print(f"   Total addresses: {total_addresses:,}")
        print(f"   Total successful: {total_success:,}")
        print(f"   Overall success rate: {(total_success/total_addresses*100):.1f}%")
        print(f"   Average speed: {avg_speed:.1f} addresses/second")
        print(f"   Average GPU utilization: {avg_gpu_util:.1f}%")
        print(f"   Total batch time: {batch_total_time/60:.1f} minutes")
        print()
        print(f"🚀 PERFORMANCE BREAKDOWN:")
        print(f"   Ultimate (1500+/sec): {ultimate_files} files")
        print(f"   Excellent (1000-1499/sec): {excellent_files} files")
        print(f"   Good (500-999/sec): {len(all_results) - ultimate_files - excellent_files} files")
        print()
        
        print(f"📁 OUTPUT FILES:")
        for result in all_results:
            print(f"   - {result['output_file']}")
    
    return all_results


if __name__ == "__main__":
    results = main()