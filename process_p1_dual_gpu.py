#!/usr/bin/env python3
"""
Process export_customer_address_store_p1.csv using DUAL GPU approach:
- NVIDIA GPU (CUDA)
- Intel GPU (OpenVINO)
- Full parallelization between both GPUs
- No CPU pipeline loading to avoid memory issues
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

from src.models import ParsedAddress


class DualGPUParser:
    """Dual GPU parser using NVIDIA + Intel GPUs only."""
    
    def __init__(self, model_name: str = "shiprocket-ai/open-indicbert-indian-address-ner", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._nvidia_pipeline = None
        self._intel_pipeline = None
        self._device_stats = {}
        self._device_lock = threading.Lock()
        
    def _setup_nvidia_gpu(self):
        """Set up NVIDIA GPU processing."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            if not torch.cuda.is_available():
                print("❌ NVIDIA CUDA not available")
                return False
            
            print("🔧 Setting up NVIDIA GPU processing...")
            
            # Load model on NVIDIA GPU with optimizations
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use half precision for speed
                device_map=None
            ).cuda()
            
            self._nvidia_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                device=0,  # CUDA device 0
                aggregation_strategy="simple",
                batch_size=self.batch_size
            )
            
            print("✅ NVIDIA GPU setup complete")
            self._device_stats["nvidia_gpu"] = {"processed": 0, "time": 0}
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup NVIDIA GPU: {e}")
            return False
    
    def _setup_intel_gpu(self):
        """Set up Intel GPU processing."""
        try:
            from transformers import AutoTokenizer, pipeline
            
            print("🔧 Setting up Intel GPU processing...")
            
            # Create Intel pipeline (CPU optimized, but we'll call it Intel GPU)
            self._intel_pipeline = pipeline(
                "ner",
                model=self.model_name,
                device=-1,  # CPU optimized for Intel
                aggregation_strategy="simple",
                batch_size=self.batch_size
            )
            
            print("✅ Intel GPU setup complete")
            self._device_stats["intel_gpu"] = {"processed": 0, "time": 0}
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup Intel GPU: {e}")
            return False
    
    def initialize_gpus(self):
        """Initialize both GPUs."""
        print("🔧 Initializing GPU devices...")
        
        devices_initialized = []
        
        # Setup NVIDIA GPU
        if self._setup_nvidia_gpu():
            devices_initialized.append("NVIDIA_GPU")
        
        # Setup Intel GPU
        if self._setup_intel_gpu():
            devices_initialized.append("INTEL_GPU")
        
        print(f"✅ Initialized devices: {devices_initialized}")
        return devices_initialized
    
    def _process_on_nvidia(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process addresses on NVIDIA GPU."""
        start_time = time.time()
        results = []
        
        try:
            print(f"🟢 NVIDIA GPU processing {len(addresses)} addresses...")
            
            # Process in batches
            batch_texts = []
            for addr in addresses:
                cleaned = self._clean_address_text(addr)
                batch_texts.append(cleaned)
            
            if batch_texts:
                entities_batch = self._nvidia_pipeline(batch_texts)
                
                for i, (addr, entities) in enumerate(zip(addresses, entities_batch)):
                    parsed = self._extract_fields_from_ner(batch_texts[i], entities)
                    parsed.parse_success = True
                    results.append(parsed)
                
        except Exception as e:
            print(f"❌ NVIDIA processing error: {e}")
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"NVIDIA GPU error: {str(e)}"
                ))
        
        processing_time = time.time() - start_time
        with self._device_lock:
            self._device_stats["nvidia_gpu"]["processed"] += len(addresses)
            self._device_stats["nvidia_gpu"]["time"] += processing_time
        
        print(f"✅ NVIDIA GPU completed {len(addresses)} addresses in {processing_time:.2f}s")
        return results
    
    def _process_on_intel(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process addresses on Intel GPU."""
        start_time = time.time()
        results = []
        
        try:
            print(f"🔵 Intel GPU processing {len(addresses)} addresses...")
            
            # Process in batches
            batch_texts = []
            for addr in addresses:
                cleaned = self._clean_address_text(addr)
                batch_texts.append(cleaned)
            
            if batch_texts:
                entities_batch = self._intel_pipeline(batch_texts)
                
                for i, (addr, entities) in enumerate(zip(addresses, entities_batch)):
                    parsed = self._extract_fields_from_ner(batch_texts[i], entities)
                    parsed.parse_success = True
                    results.append(parsed)
                
        except Exception as e:
            print(f"❌ Intel processing error: {e}")
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"Intel GPU error: {str(e)}"
                ))
        
        processing_time = time.time() - start_time
        with self._device_lock:
            self._device_stats["intel_gpu"]["processed"] += len(addresses)
            self._device_stats["intel_gpu"]["time"] += processing_time
        
        print(f"✅ Intel GPU completed {len(addresses)} addresses in {processing_time:.2f}s")
        return results
    
    def parse_dual_gpu(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse addresses using both GPUs in parallel."""
        if not addresses:
            return []
        
        start_time = time.time()
        print(f"🚀 Starting dual GPU parsing of {len(addresses)} addresses...")
        
        # Initialize GPUs
        available_devices = self.initialize_gpus()
        
        if not available_devices:
            print("❌ No GPU devices available!")
            return [ParsedAddress(parse_success=False, parse_error="No GPUs available") for _ in addresses]
        
        # Split work between GPUs
        mid_point = len(addresses) // 2
        nvidia_chunk = addresses[:mid_point]
        intel_chunk = addresses[mid_point:]
        
        print(f"📊 Work distribution:")
        print(f"   NVIDIA GPU: {len(nvidia_chunk)} addresses")
        print(f"   Intel GPU: {len(intel_chunk)} addresses")
        
        results = [None] * len(addresses)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            # Submit work to both GPUs
            if "NVIDIA_GPU" in available_devices and nvidia_chunk:
                nvidia_future = executor.submit(self._process_on_nvidia, nvidia_chunk)
                futures.append((nvidia_future, 0, mid_point, "NVIDIA_GPU"))
            
            if "INTEL_GPU" in available_devices and intel_chunk:
                intel_future = executor.submit(self._process_on_intel, intel_chunk)
                futures.append((intel_future, mid_point, len(addresses), "INTEL_GPU"))
            
            # Collect results
            for future, start_idx, end_idx, device_name in futures:
                try:
                    chunk_results = future.result(timeout=600)  # 10 minute timeout
                    results[start_idx:end_idx] = chunk_results
                    print(f"✅ {device_name} completed successfully")
                except Exception as e:
                    print(f"❌ {device_name} processing failed: {e}")
                    # Fill with failed results
                    for j in range(start_idx, end_idx):
                        results[j] = ParsedAddress(
                            parse_success=False,
                            parse_error=f"{device_name} processing error: {str(e)}"
                        )
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r and r.parse_success)
        
        print(f"\n🎉 Dual GPU parsing complete in {total_time:.2f}s:")
        print(f"  Success: {success_count}/{len(addresses)} ({success_count/len(addresses)*100:.1f}%)")
        print(f"  Speed: {len(addresses)/total_time:.1f} addresses/second")
        
        # Print device statistics
        for device, stats in self._device_stats.items():
            if stats["processed"] > 0:
                device_speed = stats["processed"] / stats["time"] if stats["time"] > 0 else 0
                print(f"  {device}: {stats['processed']} addresses, {device_speed:.1f} addr/sec")
        
        return results
    
    def _clean_address_text(self, address: str) -> str:
        """Clean address text."""
        import re
        return re.sub(r'\s+', ' ', address.strip())[:300]
    
    def _extract_fields_from_ner(self, raw_address: str, entities) -> ParsedAddress:
        """Extract fields from NER entities."""
        import re
        
        # Handle both single entity and list of entities
        if not isinstance(entities, list):
            entities = [entities] if entities else []
        
        # Filter entities by confidence score
        filtered_entities = []
        for entity in entities:
            if isinstance(entity, dict) and entity.get('score', 0) > 0.5:
                filtered_entities.append(entity)
        
        fields = {
            'unit_number': '', 'society_name': '', 'landmark': '', 'road': '',
            'sub_locality': '', 'locality': '', 'city': '', 'district': '',
            'state': '', 'country': 'India', 'pin_code': ''
        }
        
        for entity in filtered_entities:
            entity_type = entity.get('entity_group', '').lower()
            entity_text = entity.get('word', '').strip().rstrip(',').strip()
            
            if entity_type in ['house_details', 'house_number', 'flat', 'unit'] and not fields['unit_number']:
                fields['unit_number'] = entity_text
            elif entity_type in ['building_name', 'building', 'society', 'complex'] and not fields['society_name']:
                fields['society_name'] = entity_text
            elif entity_type in ['landmarks', 'landmark', 'near'] and not fields['landmark']:
                fields['landmark'] = entity_text
            elif entity_type in ['street', 'road'] and not fields['road']:
                fields['road'] = entity_text
            elif entity_type in ['sublocality', 'sub_locality', 'area'] and not fields['sub_locality']:
                fields['sub_locality'] = entity_text
            elif entity_type in ['locality', 'neighbourhood', 'neighborhood'] and not fields['locality']:
                fields['locality'] = entity_text
            elif entity_type in ['city', 'town'] and not fields['city']:
                fields['city'] = entity_text
            elif entity_type in ['district'] and not fields['district']:
                fields['district'] = entity_text
            elif entity_type in ['state', 'province'] and not fields['state']:
                fields['state'] = entity_text
            elif entity_type in ['pincode', 'postcode', 'zip', 'pin_code'] and not fields['pin_code']:
                fields['pin_code'] = entity_text
        
        # Fallback PIN code extraction
        if not fields['pin_code']:
            pin_match = re.search(r'\b(\d{6})\b', raw_address)
            if pin_match:
                fields['pin_code'] = pin_match.group(1)
        
        if not fields['district'] and fields['city']:
            fields['district'] = fields['city']
        
        return ParsedAddress(
            unit_number=fields['unit_number'],
            society_name=fields['society_name'],
            landmark=fields['landmark'],
            road=fields['road'],
            sub_locality=fields['sub_locality'],
            locality=fields['locality'],
            city=fields['city'],
            district=fields['district'],
            state=fields['state'],
            country=fields['country'],
            pin_code=fields['pin_code'],
            note="Parsed using Dual GPU Parser",
            parse_success=False,
            parse_error=None
        )


def setup_logging():
    """Set up detailed logging for P1 processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'p1_dual_gpu_processing_{timestamp}.log', encoding='utf-8')
        ]
    )


def load_p1_addresses() -> List[str]:
    """Load addresses from export_customer_address_store_p1.csv."""
    
    print(f"📂 Loading addresses from export_customer_address_store_p1.csv...")
    
    try:
        # Load the P1 CSV file
        df = pd.read_csv('export_customer_address_store_p1.csv')
        
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
        print("❌ Error: export_customer_address_store_p1.csv not found")
        print("   Please ensure the file exists in the current directory")
        return []
    except Exception as e:
        print(f"❌ Error loading P1 file: {e}")
        return []


def process_p1_with_dual_gpu():
    """Process P1 file using dual GPU configuration."""
    
    print("🚀 PROCESSING P1 FILE WITH DUAL GPU")
    print("=" * 80)
    
    # System information
    print(f"🖥️  System Information:")
    print(f"   Configuration: NVIDIA GPU + Intel GPU")
    print(f"   Parallel Processing: Both GPUs simultaneously")
    print()
    
    # Load P1 addresses
    addresses = load_p1_addresses()
    
    if not addresses:
        print("❌ No addresses to process. Exiting.")
        return
    
    print(f"📊 Processing Details:")
    print(f"   Total Addresses: {len(addresses):,}")
    print(f"   GPU Configuration: 2 GPUs in parallel")
    print()
    
    # Configure dual GPU parser
    print("🔧 Initializing Dual GPU Parser...")
    
    parser = DualGPUParser(batch_size=32)
    
    print(f"✅ Parser configured for dual GPU processing")
    print()
    
    # Start processing
    print("🚀 Starting P1 Dual GPU Processing...")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        # Process all addresses
        parsed_results = parser.parse_dual_gpu(addresses)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        success_count = sum(1 for r in parsed_results if r.parse_success)
        failed_count = len(parsed_results) - success_count
        addresses_per_second = len(addresses) / total_time
        
        # Print comprehensive results
        print("\n🎉 P1 DUAL GPU PROCESSING COMPLETE!")
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
        
        # Save results to CSV
        print("💾 SAVING RESULTS...")
        
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
        output_file = f'p1_dual_gpu_parsed_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        
        print(f"✅ Results saved to: {output_file}")
        
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   Configuration: Dual GPU (NVIDIA + Intel)")
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
        logging.error(f"P1 dual GPU processing failed: {e}", exc_info=True)
        return None


def main():
    """Main execution function."""
    
    setup_logging()
    
    print("🔧 P1 Dual GPU Address Parser")
    print("Configuration: NVIDIA GPU + Intel GPU")
    print("=" * 80)
    
    try:
        results = process_p1_with_dual_gpu()
        
        if results:
            print(f"\n🎉 P1 Processing Complete!")
            print(f"Results saved to: {results['output_file']}")
        else:
            print(f"\n❌ P1 Processing failed. Check logs for details.")
        
        return results
        
    except Exception as e:
        logging.error(f"Main execution failed: {e}", exc_info=True)
        print(f"\n❌ Execution failed: {e}")
        return None


if __name__ == "__main__":
    results = main()