"""
Optimized Shiprocket Address Parser with Multi-GPU and Performance Enhancements.

This module implements an optimized version of the Shiprocket parser that:
1. Uses larger batch sizes for better GPU utilization
2. Implements parallel processing where possible
3. Optimizes memory usage
4. Supports multi-GPU processing (NVIDIA + Intel)
"""

import logging
import re
import time
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

from src.models import ParsedAddress

logger = logging.getLogger(__name__)


class OptimizedShiprocketParser:
    """Optimized Shiprocket address parser with multi-GPU support and performance enhancements."""
    
    def __init__(
        self,
        model_name: str = "shiprocket-ai/open-indicbert-indian-address-ner",
        batch_size: int = 100,  # Increased from 20
        use_gpu: bool = True,
        max_workers: int = 4,   # Parallel workers
        use_intel_gpu: bool = False  # Enable Intel GPU support
    ):
        """Initialize optimized Shiprocket parser.
        
        Args:
            model_name: Hugging Face model name
            batch_size: Number of addresses to process in parallel (increased)
            use_gpu: Whether to use GPU acceleration
            max_workers: Number of parallel workers
            use_intel_gpu: Whether to try using Intel GPU as well
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.use_intel_gpu = use_intel_gpu
        
        # Statistics
        self._total_parsed = 0
        self._total_failed = 0
        self._total_retries = 0
        self._processing_times = []
        
        # Model instances (for multi-GPU)
        self._models = {}
        self._tokenizers = {}
        self._pipelines = {}
        self._device_lock = threading.Lock()
        
        logger.info(f"Initialized OptimizedShiprocketParser with:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  Max Workers: {max_workers}")
        logger.info(f"  GPU Enabled: {use_gpu}")
        logger.info(f"  Intel GPU: {use_intel_gpu}")
    
    def _get_available_devices(self):
        """Get list of available devices for processing."""
        devices = []
        
        if self.use_gpu:
            import torch
            
            # Add NVIDIA GPUs
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    devices.append(f"cuda:{i}")
                    logger.info(f"Added NVIDIA GPU {i} to device pool")
            
            # Try to add Intel GPU
            if self.use_intel_gpu:
                try:
                    import intel_extension_for_pytorch as ipex
                    if hasattr(torch, 'xpu') and torch.xpu.is_available():
                        for i in range(torch.xpu.device_count()):
                            devices.append(f"xpu:{i}")
                            logger.info(f"Added Intel XPU {i} to device pool")
                except ImportError:
                    logger.warning("Intel Extension for PyTorch not available")
        
        # Fallback to CPU if no GPUs
        if not devices:
            devices.append("cpu")
            logger.info("Using CPU as fallback device")
        
        return devices
    
    def _load_model_on_device(self, device: str):
        """Load model on specific device."""
        if device in self._models:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            import torch
            
            logger.info(f"Loading Shiprocket model on device: {device}")
            
            # Load tokenizer (device-agnostic)
            if 'tokenizer' not in self._tokenizers:
                self._tokenizers['tokenizer'] = AutoTokenizer.from_pretrained(
                    self.model_name,
                    device_map=None
                )
            
            # Determine torch dtype based on device
            if device.startswith('cuda'):
                torch_dtype = torch.float16  # FP16 for NVIDIA GPUs
            elif device.startswith('xpu'):
                torch_dtype = torch.float32  # FP32 for Intel GPUs (safer)
            else:
                torch_dtype = torch.float32  # FP32 for CPU
            
            # Load model
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                device_map=None,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            
            # Move to device
            if device != "cpu":
                model = model.to(device)
            
            self._models[device] = model
            
            # Create pipeline
            device_id = -1 if device == "cpu" else int(device.split(':')[1])
            if device.startswith('xpu'):
                device_id = -1  # Intel XPU handling
            
            pipeline_obj = pipeline(
                "ner",
                model=model,
                tokenizer=self._tokenizers['tokenizer'],
                device=device_id if not device.startswith('xpu') else device,
                aggregation_strategy="simple"
            )
            
            self._pipelines[device] = pipeline_obj
            
            logger.info(f"Successfully loaded model on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model on {device}: {e}")
            raise
    
    def _process_batch_on_device(self, addresses: List[str], device: str) -> List[ParsedAddress]:
        """Process a batch of addresses on a specific device."""
        start_time = time.time()
        
        # Ensure model is loaded on device
        with self._device_lock:
            if device not in self._pipelines:
                self._load_model_on_device(device)
        
        pipeline_obj = self._pipelines[device]
        results = []
        
        # Process addresses in smaller sub-batches for memory efficiency
        sub_batch_size = min(25, len(addresses))  # Process 25 at a time
        
        for i in range(0, len(addresses), sub_batch_size):
            sub_batch = addresses[i:i + sub_batch_size]
            
            for addr in sub_batch:
                try:
                    # Clean address
                    cleaned_address = self._clean_address_text(addr)
                    
                    # Extract entities
                    entities = self._safe_ner_extraction(cleaned_address, pipeline_obj)
                    
                    # Parse fields
                    parsed_address = self._extract_fields_from_ner(cleaned_address, entities)
                    parsed_address.parse_success = True
                    
                    results.append(parsed_address)
                    self._total_parsed += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to parse address on {device}: {e}")
                    results.append(ParsedAddress(
                        parse_success=False,
                        parse_error=f"Device {device} error: {str(e)}"
                    ))
                    self._total_failed += 1
        
        processing_time = time.time() - start_time
        self._processing_times.append(processing_time)
        
        logger.debug(f"Processed {len(addresses)} addresses on {device} in {processing_time:.2f}s")
        return results
    
    def parse_batch_optimized(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse multiple addresses using optimized multi-device processing."""
        if not addresses:
            return []
        
        start_time = time.time()
        logger.info(f"Starting optimized batch parsing of {len(addresses)} addresses...")
        
        # Get available devices
        devices = self._get_available_devices()
        logger.info(f"Using devices: {devices}")
        
        # Split addresses across devices
        results = [None] * len(addresses)  # Maintain order
        
        if len(devices) == 1:
            # Single device processing (optimized)
            device = devices[0]
            batch_results = self._process_batch_on_device(addresses, device)
            results = batch_results
        else:
            # Multi-device processing
            chunk_size = max(1, len(addresses) // len(devices))
            futures = []
            
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(devices))) as executor:
                for i, device in enumerate(devices):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < len(devices) - 1 else len(addresses)
                    
                    if start_idx < len(addresses):
                        chunk = addresses[start_idx:end_idx]
                        future = executor.submit(self._process_batch_on_device, chunk, device)
                        futures.append((future, start_idx, end_idx))
                
                # Collect results maintaining order
                for future, start_idx, end_idx in futures:
                    try:
                        chunk_results = future.result(timeout=300)  # 5 minute timeout
                        results[start_idx:end_idx] = chunk_results
                    except Exception as e:
                        logger.error(f"Device processing failed: {e}")
                        # Fill with failed results
                        for j in range(start_idx, end_idx):
                            results[j] = ParsedAddress(
                                parse_success=False,
                                parse_error=f"Multi-device processing error: {str(e)}"
                            )
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r and r.parse_success)
        failed_count = len(results) - success_count
        
        logger.info(f"Optimized batch parsing complete in {total_time:.2f}s:")
        logger.info(f"  Success: {success_count}/{len(addresses)} ({success_count/len(addresses)*100:.1f}%)")
        logger.info(f"  Failed: {failed_count}/{len(addresses)}")
        logger.info(f"  Speed: {len(addresses)/total_time:.1f} addresses/second")
        
        return results
    
    def _clean_address_text(self, address: str) -> str:
        """Clean and prepare address text for NER processing."""
        cleaned = re.sub(r'\s+', ' ', address.strip())
        if len(cleaned) > 300:
            cleaned = cleaned[:300]
        return cleaned
    
    def _safe_ner_extraction(self, address: str, pipeline_obj) -> List[Dict]:
        """Safely extract NER entities with error handling."""
        try:
            entities = pipeline_obj(address)
            
            if not isinstance(entities, list):
                entities = [entities] if entities else []
            
            # Filter high-confidence entities
            filtered_entities = []
            for entity in entities:
                if isinstance(entity, dict) and entity.get('score', 0) > 0.5:
                    filtered_entities.append(entity)
            
            return filtered_entities
            
        except Exception as e:
            logger.debug(f"NER extraction failed: {e}")
            return []
    
    def _extract_fields_from_ner(self, raw_address: str, entities: List[Dict]) -> ParsedAddress:
        """Extract address fields from NER entities."""
        fields = {
            'unit_number': '',
            'society_name': '',
            'landmark': '',
            'road': '',
            'sub_locality': '',
            'locality': '',
            'city': '',
            'district': '',
            'state': '',
            'country': 'India',
            'pin_code': '',
        }
        
        # Extract from NER entities
        for entity in entities:
            entity_type = entity.get('entity_group', '').lower()
            entity_text = entity.get('word', '').strip().rstrip(',').strip()
            
            # Map entity types to fields
            if entity_type in ['house_details', 'house_number', 'flat', 'unit']:
                if not fields['unit_number']:
                    fields['unit_number'] = entity_text
            elif entity_type in ['building_name', 'building', 'society', 'complex']:
                if not fields['society_name']:
                    fields['society_name'] = entity_text
            elif entity_type in ['landmarks', 'landmark', 'near']:
                if not fields['landmark']:
                    fields['landmark'] = entity_text
            elif entity_type in ['street', 'road']:
                if not fields['road']:
                    fields['road'] = entity_text
            elif entity_type in ['sublocality', 'sub_locality', 'area']:
                if not fields['sub_locality']:
                    fields['sub_locality'] = entity_text
            elif entity_type in ['locality', 'neighbourhood', 'neighborhood']:
                if not fields['locality']:
                    fields['locality'] = entity_text
            elif entity_type in ['city', 'town']:
                if not fields['city']:
                    fields['city'] = entity_text
            elif entity_type in ['district']:
                if not fields['district']:
                    fields['district'] = entity_text
            elif entity_type in ['state', 'province']:
                if not fields['state']:
                    fields['state'] = entity_text
            elif entity_type in ['pincode', 'postcode', 'zip', 'pin_code']:
                if not fields['pin_code']:
                    fields['pin_code'] = entity_text
        
        # Fallback PIN code extraction
        if not fields['pin_code']:
            pin_match = re.search(r'\b(\d{6})\b', raw_address)
            if pin_match:
                fields['pin_code'] = pin_match.group(1)
        
        # Set district to city if not found
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
            note="Parsed using Optimized Shiprocket IndicBERT NER model",
            parse_success=False,  # Will be set by caller
            parse_error=None
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive parsing statistics."""
        total_attempts = self._total_parsed + self._total_failed
        success_rate = (self._total_parsed / total_attempts * 100) if total_attempts > 0 else 0
        
        avg_processing_time = sum(self._processing_times) / len(self._processing_times) if self._processing_times else 0
        
        return {
            "total_parsed": self._total_parsed,
            "total_failed": self._total_failed,
            "total_retries": self._total_retries,
            "success_rate_percent": round(success_rate, 1),
            "total_attempts": total_attempts,
            "avg_processing_time": round(avg_processing_time, 3),
            "total_processing_time": sum(self._processing_times),
            "devices_used": list(self._pipelines.keys()),
            "batch_size": self.batch_size,
            "max_workers": self.max_workers
        }