"""IndicBERT-based Parser for Indian Address Parsing.

This module uses the IndicBERT transformer model for Named Entity Recognition
to extract address components from Indian addresses.
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models import ParsedAddress


logger = logging.getLogger(__name__)


class IndicBERTParser:
    """Extracts structured address components using IndicBERT model.
    
    This parser uses the ai4bharat/indic-bert transformer model for
    Named Entity Recognition on Indian addresses.
    """
    
    def __init__(
        self,
        model_name: str = "ai4bharat/indic-bert",
        batch_size: int = 10,
        use_gpu: bool = False
    ):
        """Initialize parser with IndicBERT model.
        
        Args:
            model_name: Hugging Face model name (default: ai4bharat/indic-bert)
            batch_size: Number of addresses to process in parallel (default: 10)
            use_gpu: Whether to use GPU acceleration (default: False)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        
        # Statistics
        self._total_parsed = 0
        self._total_failed = 0
        
        # Lazy load model
        self._model = None
        self._tokenizer = None
        self._ner_pipeline = None
        
        logger.info(f"Initialized IndicBERTParser with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the IndicBERT model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            import torch
            
            logger.info(f"Loading IndicBERT model: {self.model_name}")
            logger.info("This may take a few minutes on first run (downloading ~500MB)...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            
            # Determine device
            device = -1  # CPU
            if self.use_gpu and torch.cuda.is_available():
                device = 0  # GPU
                self._model = self._model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
            
            # Create NER pipeline
            self._ner_pipeline = pipeline(
                "ner",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device,
                aggregation_strategy="simple"
            )
            
            logger.info("IndicBERT model loaded successfully")
                
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Please install: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def parse_address(self, raw_address: str) -> ParsedAddress:
        """Parse a single address using IndicBERT NER.
        
        Args:
            raw_address: Unstructured address text to parse
            
        Returns:
            ParsedAddress object with extracted fields
        """
        if not raw_address or not raw_address.strip():
            logger.warning("Empty address text provided")
            return ParsedAddress(
                parse_success=False,
                parse_error="Empty address text"
            )
        
        logger.debug(f"Parsing address with IndicBERT: {raw_address[:100]}...")
        
        try:
            # Ensure model is loaded
            self._load_model()
            
            # Use NER pipeline to extract entities
            entities = self._ner_pipeline(raw_address)
            
            # Extract fields using both NER and rule-based fallback
            parsed_address = self._extract_fields_hybrid(raw_address, entities)
            parsed_address.parse_success = True
            
            self._total_parsed += 1
            logger.debug("Successfully parsed address with IndicBERT")
            
            return parsed_address
            
        except Exception as e:
            logger.error(f"Error parsing address with IndicBERT: {e}", exc_info=True)
            self._total_failed += 1
            return ParsedAddress(
                parse_success=False,
                parse_error=f"IndicBERT parsing error: {str(e)}"
            )
    
    def parse_batch(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse multiple addresses in parallel.
        
        Args:
            addresses: List of unstructured address texts to parse
            
        Returns:
            List of ParsedAddress objects in the same order as input
        """
        if not addresses:
            return []
        
        logger.info(f"Starting batch parsing of {len(addresses)} addresses with IndicBERT...")
        
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
    
    def _extract_fields_hybrid(self, raw_address: str, entities: List[Dict]) -> ParsedAddress:
        """Extract address fields using hybrid approach (NER + rules).
        
        Combines IndicBERT NER results with rule-based extraction for robustness.
        
        Args:
            raw_address: Raw address text
            entities: NER entities from IndicBERT
            
        Returns:
            ParsedAddress with extracted fields
        """
        address = raw_address.strip()
        
        # Initialize fields
        unit_number = ""
        society_name = ""
        landmark = ""
        road = ""
        sub_locality = ""
        locality = ""
        city = ""
        district = ""
        state = ""
        country = "India"
        pin_code = ""
        
        # Extract from NER entities (if model is trained for address components)
        # Note: Standard IndicBERT may not have address-specific labels
        # This is a placeholder for when using a fine-tuned model
        for entity in entities:
            entity_type = entity.get('entity_group', '').upper()
            entity_text = entity.get('word', '').strip()
            
            # Map entity types to address fields
            # These mappings would need to be adjusted based on actual model output
            if entity_type in ['LOC', 'LOCATION']:
                if not city:
                    city = entity_text
            elif entity_type in ['ORG', 'ORGANIZATION']:
                if not society_name:
                    society_name = entity_text
        
        # Rule-based extraction as fallback (same as LocalLLMParser)
        # Extract PIN code
        if not pin_code:
            pin_match = re.search(r'\b(\d{6})\b', address)
            if pin_match:
                pin_code = pin_match.group(1)
        
        # Extract unit number
        if not unit_number:
            unit_patterns = [
                r'(?:Flat|flat|FLAT|Apartment|apartment|APT|apt|Unit|unit|Office|office|Shop|shop|Villa|villa|Bungalow|bungalow)\s*[#:-]?\s*([A-Z0-9/-]+)',
                r'^([A-Z0-9]+-?[0-9]+)\s*,',
                r'^([0-9]+(?:st|nd|rd|th)?\s+Floor)\s*,',
                r'\b([A-Z]-Wing\s+[0-9]+)\b',
                r'\b([0-9]+/[A-Z0-9]+)\b',
            ]
            for pattern in unit_patterns:
                match = re.search(pattern, address, re.IGNORECASE)
                if match:
                    unit_number = match.group(1).strip()
                    break
        
        # Extract state
        if not state:
            states = [
                "Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Gujarat",
                "Rajasthan", "Uttar Pradesh", "Madhya Pradesh", "West Bengal",
                "Andhra Pradesh", "Telangana", "Kerala", "Punjab", "Haryana",
                "Bihar", "Odisha", "Assam", "Jharkhand", "Chhattisgarh"
            ]
            for s in states:
                if s.lower() in address.lower():
                    state = s
                    break
        
        # Extract city
        if not city:
            cities = [
                "Pune", "Mumbai", "Bangalore", "Delhi", "Hyderabad", "Chennai",
                "Kolkata", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur",
                "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad"
            ]
            for c in cities:
                if c.lower() in address.lower():
                    city = c
                    break
        
        # Extract society name
        if not society_name:
            society_patterns = [
                r'(?:Flat|flat|FLAT|Apartment|apartment|APT|apt|Unit|unit|Office|office|Shop|shop|Villa|villa|Bungalow|bungalow)\s*[#:-]?\s*[A-Z0-9/-]+\s*,\s*([A-Z][A-Za-z\s]+?)(?:,|Near|near)',
                r'^[A-Z0-9]+-?[0-9]+\s*,\s*([A-Z][A-Za-z\s]+?)(?:,|Near|near)',
                r'^[A-Z]-Wing\s+[0-9]+\s*,\s*([A-Z][A-Za-z\s]+?)(?:,|Near|near)',
                r'^[0-9]+(?:st|nd|rd|th)?\s+Floor\s*,\s*([A-Z][A-Za-z\s]+?)(?:,|Near|near)',
            ]
            for pattern in society_patterns:
                match = re.search(pattern, address, re.IGNORECASE)
                if match:
                    society_name = match.group(1).strip()
                    society_name = re.sub(r'\s+', ' ', society_name)
                    break
        
        # Extract landmark
        if not landmark:
            landmark_match = re.search(r'(?:Near|near|Opposite|opposite|Opp|opp\.?)\s+([^,]+)', address)
            if landmark_match:
                landmark = landmark_match.group(1).strip()
        
        # Extract road
        if not road:
            road_patterns = [
                r'([A-Z][A-Za-z\s]+(?:Road|road|ROAD|Street|street|STREET|Marg|marg|MARG))',
                r'([A-Z][A-Za-z\s]+\s+Rd\.?)',
            ]
            for pattern in road_patterns:
                match = re.search(pattern, address)
                if match:
                    road = match.group(1).strip()
                    break
        
        # Extract locality/sub-locality
        if city:
            city_index = address.lower().find(city.lower())
            if city_index > 0:
                before_city = address[:city_index].strip()
                parts = [p.strip() for p in before_city.split(',')]
                if len(parts) >= 2:
                    sub_locality = parts[-1]
                    if len(parts) >= 3:
                        locality = parts[-2]
        
        # Set district
        if not district:
            if city == "Pune":
                district = "Pune"
            elif city:
                district = city
        
        # Create ParsedAddress object
        parsed_address = ParsedAddress(
            unit_number=unit_number,
            society_name=society_name,
            landmark=landmark,
            road=road,
            sub_locality=sub_locality,
            locality=locality,
            city=city,
            district=district,
            state=state,
            country=country,
            pin_code=pin_code,
            note="Parsed using IndicBERT + rule-based hybrid",
            parse_success=False,  # Will be set by caller
            parse_error=None
        )
        
        return parsed_address
    
    def get_statistics(self) -> Dict[str, int]:
        """Get parsing statistics.
        
        Returns:
            Dictionary with parsing statistics
        """
        return {
            "total_parsed": self._total_parsed,
            "total_failed": self._total_failed,
            "total_retries": 0
        }
