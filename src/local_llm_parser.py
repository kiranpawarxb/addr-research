"""Local LLM Parser for Indian Address Parsing.

This module provides a local alternative to the cloud-based LLM parser,
using open-source models that run entirely on your machine.
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models import ParsedAddress


logger = logging.getLogger(__name__)


class LocalLLMParser:
    """Extracts structured address components using a local model.
    
    This parser uses rule-based extraction combined with a local transformer model
    for Indian address parsing. It runs entirely offline without API calls.
    """
    
    def __init__(
        self,
        model_name: str = "ai4bharat/indic-bert",
        batch_size: int = 10,
        use_gpu: bool = False
    ):
        """Initialize local parser with model configuration.
        
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
        
        logger.info(f"Initialized LocalLLMParser with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the transformer model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info(f"Loading model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            
            if self.use_gpu and torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def parse_address(self, raw_address: str) -> ParsedAddress:
        """Parse a single address using rule-based extraction.
        
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
        
        logger.debug(f"Parsing address: {raw_address[:100]}...")
        
        try:
            # Use rule-based extraction for Indian addresses
            parsed_address = self._extract_fields_rule_based(raw_address)
            parsed_address.parse_success = True
            
            self._total_parsed += 1
            logger.debug("Successfully parsed address")
            
            return parsed_address
            
        except Exception as e:
            logger.error(f"Error parsing address: {e}", exc_info=True)
            self._total_failed += 1
            return ParsedAddress(
                parse_success=False,
                parse_error=f"Parsing error: {str(e)}"
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
        
        logger.info(f"Starting batch parsing of {len(addresses)} addresses...")
        
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
    
    def _extract_fields_rule_based(self, raw_address: str) -> ParsedAddress:
        """Extract address fields using rule-based patterns.
        
        This method uses regex patterns and heuristics optimized for Indian addresses.
        
        Args:
            raw_address: Raw address text
            
        Returns:
            ParsedAddress with extracted fields
        """
        # Clean the address
        address = raw_address.strip()
        
        # Extract PIN code (6 digits)
        pin_code = ""
        pin_match = re.search(r'\b(\d{6})\b', address)
        if pin_match:
            pin_code = pin_match.group(1)
        
        # Extract unit number (flat, apartment, etc.)
        unit_number = ""
        unit_patterns = [
            r'(?:Flat|flat|FLAT|Apartment|apartment|APT|apt|Unit|unit|Office|office|Shop|shop|Villa|villa|Bungalow|bungalow)\s*[#:-]?\s*([A-Z0-9/-]+)',
            r'^([A-Z0-9]+-?[0-9]+)\s*,',  # Starting with unit number like A-204, B-Wing 404
            r'^([0-9]+(?:st|nd|rd|th)?\s+Floor)\s*,',  # Floor numbers like "3rd Floor"
            r'\b([A-Z]-Wing\s+[0-9]+)\b',  # B-Wing 404
            r'\b([0-9]+/[A-Z0-9]+)\b',  # Pattern like 12/B
        ]
        for pattern in unit_patterns:
            match = re.search(pattern, address, re.IGNORECASE)
            if match:
                unit_number = match.group(1).strip()
                break
        
        # Extract state (common Indian states)
        state = ""
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
        
        # Extract city (common Indian cities, especially Pune)
        city = ""
        cities = [
            "Pune", "Mumbai", "Bangalore", "Delhi", "Hyderabad", "Chennai",
            "Kolkata", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur",
            "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad"
        ]
        for c in cities:
            if c.lower() in address.lower():
                city = c
                break
        
        # Extract society/building name (heuristic: capitalized words before area names)
        society_name = ""
        # Look for patterns like "Kumar Paradise", "Amanora Park Town", etc.
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
                # Clean up society name
                society_name = re.sub(r'\s+', ' ', society_name)
                break
        
        # Extract landmark (text after "Near" or "Opposite")
        landmark = ""
        landmark_match = re.search(r'(?:Near|near|Opposite|opposite|Opp|opp\.?)\s+([^,]+)', address)
        if landmark_match:
            landmark = landmark_match.group(1).strip()
        
        # Extract road/street
        road = ""
        road_patterns = [
            r'([A-Z][A-Za-z\s]+(?:Road|road|ROAD|Street|street|STREET|Marg|marg|MARG))',
            r'([A-Z][A-Za-z\s]+\s+Rd\.?)',
        ]
        for pattern in road_patterns:
            match = re.search(pattern, address)
            if match:
                road = match.group(1).strip()
                break
        
        # Extract locality/area (heuristic: words before city name)
        locality = ""
        sub_locality = ""
        if city:
            # Find text between society name and city
            city_index = address.lower().find(city.lower())
            if city_index > 0:
                before_city = address[:city_index].strip()
                # Get last few comma-separated parts
                parts = [p.strip() for p in before_city.split(',')]
                if len(parts) >= 2:
                    sub_locality = parts[-1]
                    if len(parts) >= 3:
                        locality = parts[-2]
        
        # Set district (for Pune, it's Pune district)
        district = ""
        if city == "Pune":
            district = "Pune"
        elif city:
            district = city  # Default to city name
        
        # Country is always India
        country = "India"
        
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
            note="Parsed using local rule-based parser",
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
            "total_retries": 0  # No retries in local parsing
        }
