"""Shiprocket Address Parser Integration.

This module uses Shiprocket's fine-tuned IndicBERT model for Indian address NER.
Model: shiprocket-ai/open-indicbert-indian-address-ner

The model is specifically trained for extracting address components from
Indian addresses using Named Entity Recognition.
"""

import logging
import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models import ParsedAddress

logger = logging.getLogger(__name__)


class ShiprocketParser:
    """Shiprocket address parser using fine-tuned IndicBERT for Indian addresses.
    
    Uses the shiprocket-ai/open-indicbert-indian-address-ner model from Hugging Face,
    which is specifically trained for Indian address component extraction.
    """
    
    def __init__(
        self,
        model_name: str = "shiprocket-ai/open-indicbert-indian-address-ner",
        batch_size: int = 10,
        use_gpu: bool = False
    ):
        """Initialize Shiprocket parser.
        
        Args:
            model_name: Hugging Face model name (default: shiprocket-ai/open-indicbert-indian-address-ner)
            batch_size: Number of addresses to process in parallel
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        
        # Statistics
        self._total_parsed = 0
        self._total_failed = 0
        self._total_retries = 0
        
        # Lazy load model
        self._model = None
        self._tokenizer = None
        self._ner_pipeline = None
        
        logger.info(f"Initialized ShiprocketParser with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the Shiprocket IndicBERT model with reliability fixes."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            import torch
            
            logger.info(f"Loading Shiprocket model: {self.model_name}")
            logger.info("This may take a few minutes on first run (downloading model)...")
            
            # Determine device first
            device_name = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            device_id = 0 if device_name == "cuda" else -1
            
            logger.info(f"Using device: {device_name}")
            
            # Load tokenizer without device mapping issues
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                device_map=None  # Prevent auto device mapping issues
            )
            
            # Load model with explicit device and dtype settings
            torch_dtype = torch.float16 if device_name == "cuda" else torch.float32
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                device_map=None,  # Prevent auto device mapping
                torch_dtype=torch_dtype,
                trust_remote_code=True  # Allow custom model code if needed
            )
            
            # Explicit device placement after loading
            if device_name == "cuda":
                self._model = self._model.cuda()
                logger.info("Shiprocket model moved to GPU")
            else:
                logger.info("Shiprocket model loaded on CPU")
            
            # Create NER pipeline with conservative settings for stability
            self._ner_pipeline = pipeline(
                "ner",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device_id,
                aggregation_strategy="simple"
            )
            
            logger.info("Shiprocket model loaded successfully with reliability fixes")
                
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Please install: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load Shiprocket model: {e}")
            logger.error("Consider installing with: pip install transformers[torch] accelerate")
            raise
    
    def parse_address(self, raw_address: str, max_retries: int = 2) -> ParsedAddress:
        """Parse a single address using Shiprocket NER model with retry logic.
        
        Args:
            raw_address: Unstructured address text to parse
            max_retries: Maximum number of retry attempts for failed parses
            
        Returns:
            ParsedAddress object with extracted fields
        """
        if not raw_address or not raw_address.strip():
            logger.warning("Empty address text provided")
            return ParsedAddress(
                parse_success=False,
                parse_error="Empty address text"
            )
        
        logger.debug(f"Parsing address with Shiprocket: {raw_address[:100]}...")
        
        # Retry logic for reliability
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Ensure model is loaded
                self._load_model()
                
                # Clean and prepare address text
                cleaned_address = self._clean_address_text(raw_address)
                
                # Use NER pipeline to extract entities with error handling
                entities = self._safe_ner_extraction(cleaned_address)
                
                # Extract fields from NER entities
                parsed_address = self._extract_fields_from_ner(cleaned_address, entities)
                parsed_address.parse_success = True
                
                self._total_parsed += 1
                logger.debug(f"Successfully parsed address with Shiprocket (attempt {attempt + 1})")
                
                return parsed_address
                
            except Exception as e:
                last_error = e
                logger.warning(f"Shiprocket parsing attempt {attempt + 1} failed: {e}")
                
                # If this is not the last attempt, wait briefly and retry
                if attempt < max_retries:
                    self._total_retries += 1
                    import time
                    time.sleep(0.1)  # Brief pause before retry
                    continue
        
        # All attempts failed
        logger.error(f"All {max_retries + 1} parsing attempts failed. Last error: {last_error}")
        self._total_failed += 1
        return ParsedAddress(
            parse_success=False,
            parse_error=f"Shiprocket parsing failed after {max_retries + 1} attempts: {str(last_error)}"
        )
    
    def _clean_address_text(self, address: str) -> str:
        """Clean and prepare address text for NER processing.
        
        Args:
            address: Raw address text
            
        Returns:
            Cleaned address text
        """
        # Remove excessive whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', address.strip())
        
        # Limit length to prevent model issues (most addresses are <200 chars)
        if len(cleaned) > 300:
            cleaned = cleaned[:300]
            logger.debug("Address truncated to 300 characters for processing")
        
        return cleaned
    
    def _safe_ner_extraction(self, address: str) -> List[Dict]:
        """Safely extract NER entities with error handling.
        
        Args:
            address: Cleaned address text
            
        Returns:
            List of NER entities
        """
        try:
            # Use pipeline with conservative settings
            entities = self._ner_pipeline(address)
            
            # Validate entities structure
            if not isinstance(entities, list):
                logger.warning("NER pipeline returned non-list result, converting")
                entities = [entities] if entities else []
            
            # Filter out low-confidence entities
            filtered_entities = []
            for entity in entities:
                if isinstance(entity, dict) and entity.get('score', 0) > 0.5:
                    filtered_entities.append(entity)
            
            logger.debug(f"Extracted {len(filtered_entities)} high-confidence entities")
            return filtered_entities
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            # Return empty list to allow graceful degradation
            return []
    
    def parse_batch(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse multiple addresses in parallel.
        
        Args:
            addresses: List of unstructured address texts to parse
            
        Returns:
            List of ParsedAddress objects in the same order as input
        """
        if not addresses:
            return []
        
        logger.info(f"Starting batch parsing of {len(addresses)} addresses with Shiprocket...")
        
        results = [None] * len(addresses)
        
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_index = {
                executor.submit(self.parse_address, addr): i
                for i, addr in enumerate(addresses)
            }
            
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    completed += 1
                    
                    if completed % 10 == 0:
                        logger.debug(f"Batch progress: {completed}/{len(addresses)}")
                        
                except Exception as e:
                    logger.error(f"Error processing address at index {index}: {e}")
                    results[index] = ParsedAddress(
                        parse_success=False,
                        parse_error=f"Batch processing error: {str(e)}"
                    )
        
        success_count = sum(1 for r in results if r.parse_success)
        failed_count = sum(1 for r in results if not r.parse_success)
        
        logger.info(
            f"Batch parsing complete. "
            f"Success: {success_count}/{len(addresses)}, "
            f"Failed: {failed_count}/{len(addresses)}"
        )
        
        return results
    
    def _extract_fields_from_ner(self, raw_address: str, entities: List[Dict]) -> ParsedAddress:
        """Extract address fields from NER entities.
        
        The Shiprocket model outputs entities with labels specific to Indian addresses.
        
        Args:
            raw_address: Raw address text
            entities: NER entities from Shiprocket model
            
        Returns:
            ParsedAddress with extracted fields
        """
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
        
        # Extract from NER entities
        # The Shiprocket model uses these actual labels (discovered through testing):
        # - building_name: society/building names
        # - house_details: unit/flat numbers
        # - landmarks: landmarks/nearby places
        # - locality: locality/area names
        # - city: city names
        # - street: street/road names (if present)
        # - state: state names (if present)
        # - pincode: PIN codes (if present)
        
        for entity in entities:
            entity_type = entity.get('entity_group', '').lower()  # Use lowercase for consistency
            entity_text = entity.get('word', '').strip()
            
            # Clean up trailing commas and whitespace
            entity_text = entity_text.rstrip(',').strip()
            
            # Map Shiprocket entity types to our fields
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
        parsed_address = ParsedAddress(
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
            note="Parsed using Shiprocket IndicBERT NER model",
            parse_success=False,  # Will be set by caller
            parse_error=None
        )
        
        return parsed_address
    
    def get_statistics(self) -> Dict[str, int]:
        """Get parsing statistics.
        
        Returns:
            Dictionary with parsing statistics
        """
        total_attempts = self._total_parsed + self._total_failed
        success_rate = (self._total_parsed / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            "total_parsed": self._total_parsed,
            "total_failed": self._total_failed,
            "total_retries": self._total_retries,
            "success_rate_percent": round(success_rate, 1),
            "total_attempts": total_attempts
        }


# INTEGRATION TEMPLATE
# ====================
# Once you provide the Shiprocket parser details, use one of these templates:

# Template 1: Python Package
# --------------------------
"""
from shiprocket_address_parser import ShiprocketAddressParser

class ShiprocketParser:
    def __init__(self, api_key: str = None, batch_size: int = 10):
        self.parser = ShiprocketAddressParser(api_key=api_key)
        self.batch_size = batch_size
        self._total_parsed = 0
        self._total_failed = 0
    
    def parse_address(self, raw_address: str) -> ParsedAddress:
        try:
            result = self.parser.parse(raw_address)
            
            parsed = ParsedAddress(
                unit_number=result.get('unit_number', ''),
                society_name=result.get('society_name', ''),
                landmark=result.get('landmark', ''),
                road=result.get('road', ''),
                sub_locality=result.get('sub_locality', ''),
                locality=result.get('locality', ''),
                city=result.get('city', ''),
                district=result.get('district', ''),
                state=result.get('state', ''),
                country=result.get('country', 'India'),
                pin_code=result.get('pin_code', ''),
                note="Parsed using Shiprocket parser",
                parse_success=True,
                parse_error=None
            )
            
            self._total_parsed += 1
            return parsed
            
        except Exception as e:
            self._total_failed += 1
            return ParsedAddress(
                parse_success=False,
                parse_error=f"Shiprocket error: {str(e)}"
            )
"""

# Template 2: API-Based
# ---------------------
"""
import requests

class ShiprocketParser:
    def __init__(self, api_key: str, api_endpoint: str, batch_size: int = 10):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.batch_size = batch_size
        self._total_parsed = 0
        self._total_failed = 0
    
    def parse_address(self, raw_address: str) -> ParsedAddress:
        try:
            response = requests.post(
                self.api_endpoint,
                headers={'Authorization': f'Bearer {self.api_key}'},
                json={'address': raw_address},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            parsed = ParsedAddress(
                unit_number=result.get('unit_number', ''),
                society_name=result.get('society_name', ''),
                # ... map other fields
                parse_success=True,
                parse_error=None
            )
            
            self._total_parsed += 1
            return parsed
            
        except Exception as e:
            self._total_failed += 1
            return ParsedAddress(
                parse_success=False,
                parse_error=f"Shiprocket API error: {str(e)}"
            )
"""
