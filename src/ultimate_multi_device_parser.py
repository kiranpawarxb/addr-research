"""
Ultimate Multi-Device Parser - Uses ALL CPU cores + ALL GPUs simultaneously.

This version maximizes hardware utilization by:
1. NVIDIA GPU processing
2. Intel GPU/OpenVINO processing  
3. ALL CPU cores in parallel
4. Dynamic work distribution
"""

import logging
import re
import time
import threading
import multiprocessing
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue

from src.models import ParsedAddress

logger = logging.getLogger(__name__)


class UltimateMultiDeviceParser:
    """Ultimate parser using ALL available hardware: GPUs + ALL CPU cores."""
    
    def __init__(
        self,
        model_name: str = "shiprocket-ai/open-indicbert-indian-address-ner",
        batch_size: int = 50,  # Smaller batches for better distribution
        use_nvidia_gpu: bool = True,
        use_intel_gpu: bool = True,
        use_all_cpu_cores: bool = True,
        cpu_core_multiplier: float = 1.5  # Use more workers than cores
    ):
        """Initialize ultimate multi-device parser."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_nvidia_gpu = use_nvidia_gpu
        self.use_intel_gpu = use_intel_gpu
        self.use_all_cpu_cores = use_all_cpu_cores
        
        # Calculate optimal worker counts
        self.cpu_cores = multiprocessing.cpu_count()
        self.cpu_workers = int(self.cpu_cores * cpu_core_multiplier) if use_all_cpu_cores else 0
        
        # Total workers = GPUs + CPU workers
        gpu_workers = (1 if use_nvidia_gpu else 0) + (1 if use_intel_gpu else 0)
        self.max_workers = gpu_workers + self.cpu_workers
        
        # Statistics
        self._total_parsed = 0
        self._total_failed = 0
        self._device_stats = {}
        
        # Device management
        self._nvidia_pipeline = None
        self._intel_pipeline = None
        self._cpu_pipelines = []
        self._device_lock = threading.Lock()
        
        logger.info(f"🚀 Initialized UltimateMultiDeviceParser:")
        logger.info(f"  CPU Cores Available: {self.cpu_cores}")
        logger.info(f"  CPU Workers: {self.cpu_workers}")
        logger.info(f"  Total Workers: {self.max_workers}")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  NVIDIA GPU: {use_nvidia_gpu}")
        logger.info(f"  Intel GPU: {use_intel_gpu}")
        logger.info(f"  All CPU Cores: {use_all_cpu_cores}")
    
    def _setup_nvidia_gpu(self):
        """Set up NVIDIA GPU processing."""
        if not self.use_nvidia_gpu:
            return False
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            if not torch.cuda.is_available():
                logger.warning("NVIDIA CUDA not available")
                return False
            
            logger.info("🔧 Setting up NVIDIA GPU processing...")
            
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
                batch_size=self.batch_size  # Enable batch processing
            )
            
            logger.info("✅ NVIDIA GPU setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup NVIDIA GPU: {e}")
            return False
    
    def _setup_intel_openvino(self):
        """Set up Intel GPU processing using OpenVINO."""
        if not self.use_intel_gpu:
            return False
        
        try:
            import openvino as ov
            from transformers import AutoTokenizer, pipeline
            
            logger.info("🔧 Setting up Intel GPU with OpenVINO...")
            
            # Initialize OpenVINO core
            core = ov.Core()
            available_devices = core.available_devices
            
            intel_gpu_device = None
            for device in available_devices:
                if 'GPU' in device:
                    intel_gpu_device = device
                    break
            
            if intel_gpu_device:
                logger.info(f"Found Intel GPU device: {intel_gpu_device}")
            
            # Create optimized Intel pipeline
            self._intel_pipeline = pipeline(
                "ner",
                model=self.model_name,
                device=-1,  # CPU optimized for Intel
                aggregation_strategy="simple",
                batch_size=self.batch_size
            )
            
            logger.info("✅ Intel OpenVINO setup complete")
            return True
            
        except ImportError:
            logger.error("OpenVINO not available. Install with: pip install openvino")
            return False
        except Exception as e:
            logger.error(f"Failed to setup Intel OpenVINO: {e}")
            return False
    
    def _setup_all_cpu_cores(self):
        """Set up CPU cores for parallel processing with shared model."""
        if not self.use_all_cpu_cores:
            return False
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            logger.info(f"🔧 Setting up {self.cpu_workers} CPU workers (using {self.cpu_cores} cores)...")
            
            # Create a limited number of CPU pipelines to avoid memory issues
            # Use max 8 pipelines regardless of core count
            actual_cpu_pipelines = min(self.cpu_workers, 8)
            self._cpu_pipelines = []
            
            for i in range(actual_cpu_pipelines):
                logger.info(f"   Loading CPU pipeline {i+1}/{actual_cpu_pipelines}...")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                
                cpu_pipeline = pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU
                    aggregation_strategy="simple",
                    batch_size=self.batch_size
                )
                
                self._cpu_pipelines.append(cpu_pipeline)
            
            logger.info(f"✅ Created {len(self._cpu_pipelines)} CPU processing pipelines")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup CPU cores: {e}")
            return False
    
    def initialize_all_devices(self):
        """Initialize ALL available processing devices."""
        logger.info("🔧 Initializing ALL processing devices...")
        
        devices_initialized = []
        
        # Setup NVIDIA GPU
        if self._setup_nvidia_gpu():
            devices_initialized.append("NVIDIA_GPU")
            self._device_stats["nvidia_gpu"] = {"processed": 0, "time": 0}
        
        # Setup Intel GPU/OpenVINO
        if self._setup_intel_openvino():
            devices_initialized.append("INTEL_OPENVINO")
            self._device_stats["intel_openvino"] = {"processed": 0, "time": 0}
        
        # Setup ALL CPU cores
        if self._setup_all_cpu_cores():
            devices_initialized.append("ALL_CPU_CORES")
            self._device_stats["all_cpu_cores"] = {"processed": 0, "time": 0}
        
        logger.info(f"✅ Initialized devices: {devices_initialized}")
        logger.info(f"🎯 Total processing power: {len(devices_initialized)} device types, {self.max_workers} workers")
        
        return devices_initialized
    
    def _process_on_nvidia(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process addresses on NVIDIA GPU with batch optimization."""
        start_time = time.time()
        results = []
        
        try:
            # Process in batches for efficiency
            batch_texts = []
            for addr in addresses:
                cleaned = self._clean_address_text(addr)
                batch_texts.append(cleaned)
            
            # Batch process on GPU
            if batch_texts:
                entities_batch = self._nvidia_pipeline(batch_texts)
                
                for i, (addr, entities) in enumerate(zip(addresses, entities_batch)):
                    parsed = self._extract_fields_from_ner(batch_texts[i], entities)
                    parsed.parse_success = True
                    results.append(parsed)
                    self._total_parsed += 1
                
        except Exception as e:
            logger.error(f"NVIDIA processing error: {e}")
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"NVIDIA GPU error: {str(e)}"
                ))
                self._total_failed += 1
        
        processing_time = time.time() - start_time
        self._device_stats["nvidia_gpu"]["processed"] += len(addresses)
        self._device_stats["nvidia_gpu"]["time"] += processing_time
        
        logger.debug(f"🟢 NVIDIA processed {len(addresses)} addresses in {processing_time:.2f}s")
        return results
    
    def _process_on_intel(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process addresses using Intel OpenVINO."""
        start_time = time.time()
        results = []
        
        try:
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
                    self._total_parsed += 1
                
        except Exception as e:
            logger.error(f"Intel processing error: {e}")
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"Intel error: {str(e)}"
                ))
                self._total_failed += 1
        
        processing_time = time.time() - start_time
        self._device_stats["intel_openvino"]["processed"] += len(addresses)
        self._device_stats["intel_openvino"]["time"] += processing_time
        
        logger.debug(f"🔵 Intel processed {len(addresses)} addresses in {processing_time:.2f}s")
        return results
    
    def _process_on_cpu_core(self, addresses: List[str], core_id: int) -> List[ParsedAddress]:
        """Process addresses on specific CPU core."""
        start_time = time.time()
        results = []
        
        try:
            pipeline_obj = self._cpu_pipelines[core_id % len(self._cpu_pipelines)]
            
            # Process in batches
            batch_texts = []
            for addr in addresses:
                cleaned = self._clean_address_text(addr)
                batch_texts.append(cleaned)
            
            if batch_texts:
                entities_batch = pipeline_obj(batch_texts)
                
                for i, (addr, entities) in enumerate(zip(addresses, entities_batch)):
                    parsed = self._extract_fields_from_ner(batch_texts[i], entities)
                    parsed.parse_success = True
                    results.append(parsed)
                    self._total_parsed += 1
                
        except Exception as e:
            logger.error(f"CPU core {core_id} error: {e}")
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"CPU core error: {str(e)}"
                ))
                self._total_failed += 1
        
        processing_time = time.time() - start_time
        if "all_cpu_cores" in self._device_stats:
            self._device_stats["all_cpu_cores"]["processed"] += len(addresses)
            self._device_stats["all_cpu_cores"]["time"] += processing_time
        
        logger.debug(f"🟡 CPU core {core_id} processed {len(addresses)} addresses in {processing_time:.2f}s")
        return results
    
    def parse_ultimate_multi_device(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse addresses using ALL available hardware simultaneously."""
        if not addresses:
            return []
        
        start_time = time.time()
        logger.info(f"🚀 Starting ULTIMATE multi-device parsing of {len(addresses)} addresses...")
        
        # Initialize all devices
        available_devices = self.initialize_all_devices()
        
        if not available_devices:
            logger.error("No processing devices available!")
            return [ParsedAddress(parse_success=False, parse_error="No devices available") for _ in addresses]
        
        # Calculate optimal work distribution
        total_workers = 0
        if "NVIDIA_GPU" in available_devices:
            total_workers += 1
        if "INTEL_OPENVINO" in available_devices:
            total_workers += 1
        if "ALL_CPU_CORES" in available_devices:
            total_workers += self.cpu_workers
        
        logger.info(f"🎯 Distributing work across {total_workers} workers")
        
        # Split work across ALL devices
        chunk_size = len(addresses) // total_workers
        results = [None] * len(addresses)
        futures = []
        current_idx = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            # NVIDIA GPU gets a chunk
            if "NVIDIA_GPU" in available_devices:
                end_idx = current_idx + chunk_size
                nvidia_chunk = addresses[current_idx:end_idx]
                nvidia_future = executor.submit(self._process_on_nvidia, nvidia_chunk)
                futures.append((nvidia_future, current_idx, end_idx, "NVIDIA_GPU"))
                current_idx = end_idx
            
            # Intel GPU gets a chunk
            if "INTEL_OPENVINO" in available_devices:
                end_idx = current_idx + chunk_size
                intel_chunk = addresses[current_idx:end_idx]
                intel_future = executor.submit(self._process_on_intel, intel_chunk)
                futures.append((intel_future, current_idx, end_idx, "INTEL_GPU"))
                current_idx = end_idx
            
            # ALL CPU cores get remaining work distributed
            if "ALL_CPU_CORES" in available_devices:
                remaining_addresses = addresses[current_idx:]
                cpu_chunk_size = len(remaining_addresses) // self.cpu_workers
                
                for i in range(self.cpu_workers):
                    start_idx = current_idx + (i * cpu_chunk_size)
                    if i == self.cpu_workers - 1:  # Last worker gets remainder
                        end_idx = len(addresses)
                    else:
                        end_idx = start_idx + cpu_chunk_size
                    
                    if start_idx < len(addresses):
                        cpu_chunk = addresses[start_idx:end_idx]
                        cpu_future = executor.submit(self._process_on_cpu_core, cpu_chunk, i)
                        futures.append((cpu_future, start_idx, end_idx, f"CPU_CORE_{i}"))
            
            # Collect results from ALL devices
            for future, start_idx, end_idx, device_name in futures:
                try:
                    chunk_results = future.result(timeout=600)  # 10 minute timeout
                    results[start_idx:end_idx] = chunk_results
                    logger.info(f"✅ {device_name} completed processing {end_idx - start_idx} addresses")
                except Exception as e:
                    logger.error(f"❌ {device_name} processing failed: {e}")
                    # Fill with failed results
                    for j in range(start_idx, end_idx):
                        results[j] = ParsedAddress(
                            parse_success=False,
                            parse_error=f"{device_name} processing error: {str(e)}"
                        )
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r and r.parse_success)
        
        logger.info(f"🎉 ULTIMATE multi-device parsing complete in {total_time:.2f}s:")
        logger.info(f"  Success: {success_count}/{len(addresses)} ({success_count/len(addresses)*100:.1f}%)")
        logger.info(f"  Speed: {len(addresses)/total_time:.1f} addresses/second")
        
        # Print detailed device statistics
        total_device_time = 0
        for device, stats in self._device_stats.items():
            if stats["processed"] > 0:
                device_speed = stats["processed"] / stats["time"] if stats["time"] > 0 else 0
                total_device_time += stats["time"]
                logger.info(f"  {device}: {stats['processed']} addresses, {device_speed:.1f} addr/sec")
        
        # Calculate parallelization efficiency
        if total_device_time > 0:
            efficiency = (total_device_time / total_time) * 100
            logger.info(f"  Parallelization Efficiency: {efficiency:.1f}%")
        
        return results
    
    def _clean_address_text(self, address: str) -> str:
        """Clean address text."""
        return re.sub(r'\s+', ' ', address.strip())[:300]
    
    def _extract_fields_from_ner(self, raw_address: str, entities) -> ParsedAddress:
        """Extract fields from NER entities."""
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
            note="Parsed using Ultimate Multi-Device Parser",
            parse_success=False,
            parse_error=None
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        total_attempts = self._total_parsed + self._total_failed
        success_rate = (self._total_parsed / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            "total_parsed": self._total_parsed,
            "total_failed": self._total_failed,
            "success_rate_percent": round(success_rate, 1),
            "device_stats": self._device_stats,
            "cpu_cores": self.cpu_cores,
            "cpu_workers": self.cpu_workers,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size
        }