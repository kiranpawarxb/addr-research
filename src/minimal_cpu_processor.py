"""Minimal CPU Processor for overflow and fallback processing.

This module implements lightweight CPU processing for handling overflow addresses
and GPU failure recovery scenarios. It uses minimal resources to avoid interfering
with GPU processing while providing reliable fallback capabilities.

Requirements: 4.2, 4.4, 7.1
"""

import logging
import time
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

try:
    from .models import ParsedAddress
    from .hybrid_config import ProcessingConfiguration
    from .error_handling import ErrorRecoveryManager
except ImportError:
    # Fallback for direct execution
    from models import ParsedAddress
    from hybrid_config import ProcessingConfiguration
    from error_handling import ErrorRecoveryManager


@dataclass
class CPUProcessingStats:
    """Statistics for CPU processing operations."""
    total_processed: int = 0
    overflow_processed: int = 0
    fallback_processed: int = 0
    success_count: int = 0
    error_count: int = 0
    processing_time: float = 0.0
    average_rate: float = 0.0


class MinimalCPUProcessor:
    """Lightweight CPU processor for overflow and fallback address processing.
    
    Handles CPU processing for overflow addresses and GPU failure recovery
    with minimal resource usage to avoid interfering with GPU operations.
    
    Requirements: 4.2, 4.4, 7.1
    """
    
    def __init__(self, config: ProcessingConfiguration):
        """Initialize the minimal CPU processor.
        
        Args:
            config: Processing configuration with CPU settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # CPU Processing Configuration
        self.cpu_batch_size = config.cpu_batch_size  # Small batches (50-100)
        self.cpu_worker_count = config.cpu_worker_count  # Minimal workers (2-4)
        
        # Processing Components
        self._cpu_pipeline = None
        self._tokenizer = None
        self._model = None
        
        # Processing State
        self.is_initialized = False
        self.processing_lock = threading.Lock()
        
        # Statistics
        self.stats = CPUProcessingStats()
        
        # Error handling
        self.error_recovery_manager = None
        
        # Thread Pool for CPU processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.cpu_worker_count,
            thread_name_prefix="MinimalCPU"
        )
        
        self.logger.info(f"Initialized MinimalCPUProcessor with "
                        f"batch_size={self.cpu_batch_size}, "
                        f"workers={self.cpu_worker_count}")
    
    def setup_minimal_cpu_pipeline(self) -> bool:
        """Initialize lightweight CPU processing pipeline with optimized configuration.
        
        Sets up a minimal CPU pipeline using lightweight models and configurations
        optimized for quick processing without GPU interference.
        
        Returns:
            True if setup successful, False otherwise
            
        Requirements: 4.2, 4.4
        """
        if self.is_initialized:
            self.logger.warning("CPU pipeline already initialized")
            return True
        
        try:
            self.logger.info("Setting up minimal CPU processing pipeline...")
            
            # Import required libraries
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            import torch
            
            # Use lightweight model for CPU processing
            model_name = "shiprocket-ai/open-indicbert-indian-address-ner"
            
            self.logger.info(f"Loading lightweight CPU model: {model_name}")
            
            # Load tokenizer with minimal configuration
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                device_map=None,  # Prevent auto device mapping
                use_fast=True  # Use fast tokenizer for better CPU performance
            )
            
            # Load model with CPU-optimized settings
            self._model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                device_map=None,
                torch_dtype=torch.float32,  # Use float32 for CPU stability
                trust_remote_code=True
            )
            
            # Ensure model is on CPU
            self._model = self._model.cpu()
            
            # Create minimal CPU pipeline with conservative settings
            self._cpu_pipeline = pipeline(
                "ner",
                model=self._model,
                tokenizer=self._tokenizer,
                device=-1,  # Force CPU usage
                aggregation_strategy="simple",
                batch_size=1  # Process one at a time for stability
            )
            
            self.is_initialized = True
            self.logger.info("✅ Minimal CPU pipeline initialized successfully")
            
            return True
            
        except ImportError as e:
            self.logger.error(f"Failed to import required libraries: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to setup CPU pipeline: {e}")
            return False
    
    def process_cpu_overflow(self, addresses: List[str]) -> List[ParsedAddress]:
        """Process overflow addresses using minimal CPU resources.
        
        Handles addresses that exceed GPU capacity using lightweight CPU processing
        with small batch sizes and minimal worker counts.
        
        Args:
            addresses: List of overflow addresses to process
            
        Returns:
            List of parsed addresses
            
        Requirements: 4.2, 4.4
        """
        if not addresses:
            return []
        
        if not self.is_initialized:
            self.logger.warning("CPU pipeline not initialized, attempting setup...")
            if not self.setup_minimal_cpu_pipeline():
                self.logger.error("Failed to initialize CPU pipeline for overflow processing")
                return [ParsedAddress(parse_success=False, 
                                    parse_error="CPU pipeline initialization failed") 
                       for _ in addresses]
        
        self.logger.info(f"Processing {len(addresses)} overflow addresses on CPU")
        
        start_time = time.time()
        
        with self.processing_lock:
            try:
                # Process in small batches to minimize resource usage
                results = []
                batch_size = min(self.cpu_batch_size, len(addresses))
                
                for i in range(0, len(addresses), batch_size):
                    batch = addresses[i:i + batch_size]
                    batch_results = self._process_cpu_batch(batch, "overflow")
                    results.extend(batch_results)
                    
                    # Brief pause between batches to avoid overwhelming CPU
                    if i + batch_size < len(addresses):
                        time.sleep(0.01)  # 10ms pause
                
                # Update statistics
                processing_time = time.time() - start_time
                self.stats.total_processed += len(results)
                self.stats.overflow_processed += len(results)
                self.stats.success_count += sum(1 for r in results if r.parse_success)
                self.stats.error_count += sum(1 for r in results if not r.parse_success)
                self.stats.processing_time += processing_time
                
                rate = len(results) / processing_time if processing_time > 0 else 0
                self.logger.info(f"✅ CPU overflow processing completed: "
                               f"{len(results)} addresses in {processing_time:.2f}s "
                               f"({rate:.1f} addr/sec)")
                
                return results
                
            except Exception as e:
                self.logger.error(f"CPU overflow processing failed: {e}")
                return [ParsedAddress(parse_success=False, 
                                    parse_error=f"CPU overflow processing error: {str(e)}") 
                       for _ in addresses]
    
    def handle_gpu_fallback(self, failed_addresses: List[str]) -> List[ParsedAddress]:
        """Handle GPU failure recovery using CPU processing.
        
        Processes addresses that failed on GPU using CPU fallback with
        error recovery mechanisms and alternative processing strategies.
        
        Args:
            failed_addresses: List of addresses that failed GPU processing
            
        Returns:
            List of parsed addresses from CPU fallback
            
        Requirements: 7.1
        """
        if not failed_addresses:
            return []
        
        if not self.is_initialized:
            self.logger.warning("CPU pipeline not initialized for fallback, attempting setup...")
            if not self.setup_minimal_cpu_pipeline():
                self.logger.error("Failed to initialize CPU pipeline for fallback processing")
                return [ParsedAddress(parse_success=False, 
                                    parse_error="CPU fallback initialization failed") 
                       for _ in failed_addresses]
        
        self.logger.info(f"Processing {len(failed_addresses)} failed addresses using CPU fallback")
        
        start_time = time.time()
        
        with self.processing_lock:
            try:
                # Use even smaller batches for fallback processing to ensure reliability
                fallback_batch_size = min(self.cpu_batch_size // 2, 25)  # Smaller batches for stability
                results = []
                
                for i in range(0, len(failed_addresses), fallback_batch_size):
                    batch = failed_addresses[i:i + fallback_batch_size]
                    
                    # Process with retry logic for fallback
                    batch_results = self._process_cpu_batch_with_retry(batch, "fallback")
                    results.extend(batch_results)
                    
                    # Longer pause between fallback batches for stability
                    if i + fallback_batch_size < len(failed_addresses):
                        time.sleep(0.05)  # 50ms pause for stability
                
                # Update statistics
                processing_time = time.time() - start_time
                self.stats.total_processed += len(results)
                self.stats.fallback_processed += len(results)
                self.stats.success_count += sum(1 for r in results if r.parse_success)
                self.stats.error_count += sum(1 for r in results if not r.parse_success)
                self.stats.processing_time += processing_time
                
                rate = len(results) / processing_time if processing_time > 0 else 0
                success_count = sum(1 for r in results if r.parse_success)
                
                self.logger.info(f"✅ CPU fallback processing completed: "
                               f"{success_count}/{len(results)} successful in {processing_time:.2f}s "
                               f"({rate:.1f} addr/sec)")
                
                return results
                
            except Exception as e:
                self.logger.error(f"CPU fallback processing failed: {e}")
                return [ParsedAddress(parse_success=False, 
                                    parse_error=f"CPU fallback processing error: {str(e)}") 
                       for _ in failed_addresses]
    
    def _process_cpu_batch(self, addresses: List[str], processing_type: str) -> List[ParsedAddress]:
        """Process a batch of addresses using CPU pipeline.
        
        Args:
            addresses: Batch of addresses to process
            processing_type: Type of processing ("overflow" or "fallback")
            
        Returns:
            List of parsed addresses
        """
        if not addresses:
            return []
        
        results = []
        
        try:
            for addr in addresses:
                if not addr or not addr.strip():
                    results.append(ParsedAddress(
                        parse_success=False,
                        parse_error="Empty address text"
                    ))
                    continue
                
                # Clean address text
                cleaned_addr = self._clean_address_text(addr)
                
                # Process with CPU pipeline
                try:
                    entities = self._cpu_pipeline(cleaned_addr)
                    parsed = self._extract_fields_from_ner(cleaned_addr, entities)
                    parsed.parse_success = True
                    parsed.note = f"Processed on CPU ({processing_type})"
                    results.append(parsed)
                    
                except Exception as e:
                    self.logger.debug(f"CPU processing failed for address: {e}")
                    results.append(ParsedAddress(
                        parse_success=False,
                        parse_error=f"CPU {processing_type} processing error: {str(e)}"
                    ))
            
        except Exception as e:
            self.logger.error(f"CPU batch processing failed: {e}")
            # Return failed results for all addresses in batch
            for addr in addresses:
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"CPU batch processing error: {str(e)}"
                ))
        
        return results
    
    def _process_cpu_batch_with_retry(self, addresses: List[str], processing_type: str, max_retries: int = 2) -> List[ParsedAddress]:
        """Process a batch with retry logic for fallback scenarios.
        
        Args:
            addresses: Batch of addresses to process
            processing_type: Type of processing
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of parsed addresses
        """
        for attempt in range(max_retries + 1):
            try:
                results = self._process_cpu_batch(addresses, processing_type)
                
                # Check if we got reasonable results
                success_count = sum(1 for r in results if r.parse_success)
                if success_count > 0 or attempt == max_retries:
                    return results
                
                # If no successes and not final attempt, retry
                if attempt < max_retries:
                    self.logger.debug(f"CPU batch retry {attempt + 1}/{max_retries}")
                    time.sleep(0.1)  # Brief pause before retry
                    
            except Exception as e:
                if attempt == max_retries:
                    self.logger.error(f"CPU batch processing failed after {max_retries + 1} attempts: {e}")
                    return [ParsedAddress(
                        parse_success=False,
                        parse_error=f"CPU processing failed after retries: {str(e)}"
                    ) for _ in addresses]
                else:
                    self.logger.debug(f"CPU batch attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(0.1)
        
        return []
    
    def _clean_address_text(self, address: str) -> str:
        """Clean and prepare address text for CPU processing.
        
        Args:
            address: Raw address text
            
        Returns:
            Cleaned address text
        """
        import re
        
        # Remove excessive whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', address.strip())
        
        # Limit length for CPU processing stability
        if len(cleaned) > 250:  # Shorter limit for CPU processing
            cleaned = cleaned[:250]
        
        return cleaned
    
    def _extract_fields_from_ner(self, raw_address: str, entities: List[Dict[str, Any]]) -> ParsedAddress:
        """Extract address fields from NER entities for CPU processing.
        
        Args:
            raw_address: Raw address text
            entities: NER entities from CPU pipeline
            
        Returns:
            ParsedAddress with extracted fields
        """
        import re
        
        # Initialize fields
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
        
        # Extract from NER entities (similar to ShiprocketParser)
        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            entity_type = entity.get('entity_group', '').lower()
            entity_text = entity.get('word', '').strip()
            
            # Skip low confidence entities
            if entity.get('score', 0) < 0.5:
                continue
            
            # Clean up entity text
            entity_text = entity_text.rstrip(',').strip()
            
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
        
        # Fallback: Use regex for PIN code if not found by NER
        if not fields['pin_code']:
            pin_match = re.search(r'\b(\d{6})\b', raw_address)
            if pin_match:
                fields['pin_code'] = pin_match.group(1)
        
        # Set district to city if not found
        if not fields['district'] and fields['city']:
            fields['district'] = fields['city']
        
        # Create ParsedAddress
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
            note="",  # Will be set by caller
            parse_success=False,  # Will be set by caller
            parse_error=None
        )
    
    def get_statistics(self) -> CPUProcessingStats:
        """Get CPU processing statistics.
        
        Returns:
            CPUProcessingStats with current statistics
        """
        # Calculate average rate
        if self.stats.processing_time > 0:
            self.stats.average_rate = self.stats.total_processed / self.stats.processing_time
        
        return self.stats
    
    def reset_statistics(self) -> None:
        """Reset CPU processing statistics."""
        self.stats = CPUProcessingStats()
    
    def shutdown(self) -> None:
        """Gracefully shutdown the CPU processor and cleanup resources."""
        self.logger.info("Shutting down minimal CPU processor...")
        
        # Shutdown thread executor
        self.executor.shutdown(wait=True)
        
        # Clear model references to free memory
        self._cpu_pipeline = None
        self._model = None
        self._tokenizer = None
        
        self.is_initialized = False
        self.logger.info("Minimal CPU processor shutdown completed")
    
    def set_error_recovery_manager(self, error_manager: 'ErrorRecoveryManager') -> None:
        """Set error recovery manager for comprehensive error handling."""
        self.error_recovery_manager = error_manager


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    
    # Add src to path for testing
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from hybrid_config import ProcessingConfiguration
    
    # Test configuration
    config = ProcessingConfiguration(
        cpu_batch_size=25,
        cpu_worker_count=2
    )
    
    # Create processor
    processor = MinimalCPUProcessor(config)
    
    # Test addresses
    test_addresses = [
        "Flat 101, Sunrise Apartments, Near City Mall, MG Road, Pune, Maharashtra 411001",
        "House No 45, Sector 12, Dwarka, New Delhi 110075",
        "Shop 23, Commercial Complex, Bandra West, Mumbai 400050"
    ]
    
    print("Testing MinimalCPUProcessor...")
    
    # Test setup
    if processor.setup_minimal_cpu_pipeline():
        print("✅ CPU pipeline setup successful")
        
        # Test overflow processing
        results = processor.process_cpu_overflow(test_addresses)
        print(f"✅ Processed {len(results)} overflow addresses")
        
        # Test fallback processing
        fallback_results = processor.handle_gpu_fallback(test_addresses[:2])
        print(f"✅ Processed {len(fallback_results)} fallback addresses")
        
        # Print statistics
        stats = processor.get_statistics()
        print(f"📊 Statistics: {stats.total_processed} total, "
              f"{stats.success_count} successful, "
              f"{stats.average_rate:.1f} addr/sec")
        
        processor.shutdown()
    else:
        print("❌ CPU pipeline setup failed")