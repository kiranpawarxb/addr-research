"""Libpostal Address Parser Integration.

Libpostal is a C library for parsing/normalizing street addresses using
statistical NLP and open data. It works well for Indian addresses.

Installation:
    pip install postal

Note: Requires libpostal C library to be installed first.
See: https://github.com/openvenues/libpostal
"""

import logging
import re
from typing import List, Dict, Any
from src.models import ParsedAddress

logger = logging.getLogger(__name__)


class LibpostalParser:
    """Address parser using libpostal library.
    
    Libpostal is a statistical address parser trained on OpenStreetMap data.
    Works well for addresses worldwide, including India.
    """
    
    def __init__(self, batch_size: int = 10):
        """Initialize libpostal parser.
        
        Args:
            batch_size: Number of addresses to process in parallel
        """
        self.batch_size = batch_size
        
        # Statistics
        self._total_parsed = 0
        self._total_failed = 0
        
        # Lazy load postal
        self._postal = None
        
        logger.info("Initialized LibpostalParser")
    
    def _load_postal(self):
        """Lazy load postal library."""
        if self._postal is not None:
            return
        
        try:
            import postal.parser
            self._postal = postal.parser
            logger.info("Libpostal library loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import postal: {e}")
            logger.error("Install with: pip install postal")
            logger.error("Note: Requires libpostal C library. See: https://github.com/openvenues/libpostal")
            raise
    
    def parse_address(self, raw_address: str) -> ParsedAddress:
        """Parse a single address using libpostal.
        
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
        
        logger.debug(f"Parsing address with libpostal: {raw_address[:100]}...")
        
        try:
            # Ensure postal is loaded
            self._load_postal()
            
            # Parse address
            parsed_components = self._postal.parse_address(raw_address)
            
            # Convert libpostal components to our format
            parsed_address = self._map_libpostal_to_parsed_address(
                raw_address,
                parsed_components
            )
            parsed_address.parse_success = True
            
            self._total_parsed += 1
            logger.debug("Successfully parsed address with libpostal")
            
            return parsed_address
            
        except Exception as e:
            logger.error(f"Error parsing address with libpostal: {e}", exc_info=True)
            self._total_failed += 1
            return ParsedAddress(
                parse_success=False,
                parse_error=f"Libpostal parsing error: {str(e)}"
            )
    
    def parse_batch(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse multiple addresses.
        
        Args:
            addresses: List of unstructured address texts to parse
            
        Returns:
            List of ParsedAddress objects in the same order as input
        """
        if not addresses:
            return []
        
        logger.info(f"Starting batch parsing of {len(addresses)} addresses with libpostal...")
        
        results = []
        for i, address in enumerate(addresses):
            try:
                parsed = self.parse_address(address)
                results.append(parsed)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Batch progress: {i+1}/{len(addresses)}")
                    
            except Exception as e:
                logger.error(f"Error processing address at index {i}: {e}")
                results.append(ParsedAddress(
                    parse_success=False,
                    parse_error=f"Batch processing error: {str(e)}"
                ))
        
        success_count = sum(1 for r in results if r.parse_success)
        failed_count = sum(1 for r in results if not r.parse_success)
        
        logger.info(
            f"Batch parsing complete. "
            f"Success: {success_count}/{len(addresses)}, "
            f"Failed: {failed_count}/{len(addresses)}"
        )
        
        return results
    
    def _map_libpostal_to_parsed_address(
        self,
        raw_address: str,
        components: List[tuple]
    ) -> ParsedAddress:
        """Map libpostal components to ParsedAddress format.
        
        Libpostal returns components like:
        [('301', 'house_number'), ('kumar paradise', 'house'), ('pune', 'city'), ...]
        
        Args:
            raw_address: Original address text
            components: List of (value, label) tuples from libpostal
            
        Returns:
            ParsedAddress with mapped fields
        """
        # Create a dictionary of components
        comp_dict = {}
        for value, label in components:
            if label not in comp_dict:
                comp_dict[label] = []
            comp_dict[label].append(value)
        
        # Map libpostal labels to our fields
        unit_number = ' '.join(comp_dict.get('house_number', []))
        
        # Society name might be in 'house' or 'building'
        society_name = ' '.join(
            comp_dict.get('house', []) + 
            comp_dict.get('building', [])
        )
        
        # Landmark might be in 'near' or 'po_box'
        landmark = ' '.join(comp_dict.get('near', []))
        
        # Road
        road = ' '.join(
            comp_dict.get('road', []) + 
            comp_dict.get('street', [])
        )
        
        # Localities
        sub_locality = ' '.join(comp_dict.get('suburb', []))
        locality = ' '.join(
            comp_dict.get('city_district', []) +
            comp_dict.get('neighbourhood', [])
        )
        
        # City
        city = ' '.join(comp_dict.get('city', []))
        
        # District
        district = ' '.join(comp_dict.get('state_district', []))
        if not district and city:
            district = city  # Fallback
        
        # State
        state = ' '.join(comp_dict.get('state', []))
        
        # Country
        country = ' '.join(comp_dict.get('country', []))
        if not country:
            country = "India"  # Default for Indian addresses
        
        # PIN code (postcode)
        pin_code = ' '.join(comp_dict.get('postcode', []))
        
        # If PIN not found by libpostal, try regex
        if not pin_code:
            pin_match = re.search(r'\b(\d{6})\b', raw_address)
            if pin_match:
                pin_code = pin_match.group(1)
        
        # Create ParsedAddress
        parsed_address = ParsedAddress(
            unit_number=unit_number.strip(),
            society_name=society_name.strip(),
            landmark=landmark.strip(),
            road=road.strip(),
            sub_locality=sub_locality.strip(),
            locality=locality.strip(),
            city=city.strip(),
            district=district.strip(),
            state=state.strip(),
            country=country.strip(),
            pin_code=pin_code.strip(),
            note="Parsed using libpostal",
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
