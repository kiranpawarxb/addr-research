"""Consolidation Engine component for the Address Consolidation System."""

import logging
import re
import uuid
from typing import List, Dict, Tuple, Optional
from rapidfuzz import fuzz

from src.models import ParsedAddress, AddressRecord, ConsolidatedGroup


logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Groups addresses based on Society Name and geographic identifiers.
    
    Implements normalization, exact matching, and optional fuzzy matching
    to consolidate addresses belonging to the same physical location.
    """
    
    # Special group ID for unmatched addresses
    UNMATCHED_GROUP_ID = "UNMATCHED"
    
    def __init__(
        self,
        fuzzy_matching: bool = False,
        similarity_threshold: float = 0.85,
        normalize_society_names: bool = True
    ):
        """Initialize engine with matching configuration.
        
        Args:
            fuzzy_matching: Enable fuzzy string matching for society names
            similarity_threshold: Minimum similarity score (0-1) for fuzzy matching
            normalize_society_names: Apply normalization to society names
        """
        self.fuzzy_matching = fuzzy_matching
        self.similarity_threshold = similarity_threshold
        self.normalize_society_names = normalize_society_names
        
        # Statistics
        self._total_groups = 0
        self._unmatched_count = 0
    
    def consolidate(
        self,
        records_with_parsed: List[Tuple[AddressRecord, ParsedAddress]]
    ) -> List[ConsolidatedGroup]:
        """Group addresses by Society Name and PIN code.
        
        Creates consolidated groups where addresses share the same
        (normalized) Society Name and PIN code. Handles empty society
        names by placing them in an "unmatched" group.
        
        Args:
            records_with_parsed: List of tuples (AddressRecord, ParsedAddress)
            
        Returns:
            List of ConsolidatedGroup objects
        """
        if not records_with_parsed:
            logger.info("No records to consolidate")
            return []
        
        logger.info(f"Starting consolidation of {len(records_with_parsed)} records...")
        logger.info(
            f"Consolidation settings: fuzzy_matching={self.fuzzy_matching}, "
            f"similarity_threshold={self.similarity_threshold}, "
            f"normalize_society_names={self.normalize_society_names}"
        )
        
        # Reset statistics
        self._total_groups = 0
        self._unmatched_count = 0
        fuzzy_matches_count = 0
        
        # Dictionary to store groups: key = (normalized_society_name, pin_code)
        groups_dict: Dict[Tuple[str, str], ConsolidatedGroup] = {}
        
        # Unmatched group for addresses without society names
        unmatched_group = ConsolidatedGroup(
            group_id=self.UNMATCHED_GROUP_ID,
            society_name="",
            pin_code="",
            records=[],
            record_count=0
        )
        
        # Process each record
        for idx, (address_record, parsed_address) in enumerate(records_with_parsed):
            society_name = parsed_address.society_name
            pin_code = parsed_address.pin_code
            
            # Log progress every 1000 records
            if (idx + 1) % 1000 == 0:
                logger.debug(f"Consolidation progress: {idx + 1}/{len(records_with_parsed)} records processed")
            
            # Check if society name is empty or missing
            if not society_name or society_name.strip() == "":
                # Add to unmatched group
                unmatched_group.add_record(address_record, parsed_address)
                self._unmatched_count += 1
                logger.debug(f"Record {idx + 1} added to unmatched group (empty society name)")
                continue
            
            # Normalize society name if enabled
            if self.normalize_society_names:
                normalized_name = self._normalize_society_name(society_name)
            else:
                normalized_name = society_name
            
            # Create group key
            group_key = (normalized_name, pin_code)
            
            # Check if we should use fuzzy matching
            if self.fuzzy_matching:
                # Try to find a similar existing group
                matched_key = self._find_fuzzy_match(
                    normalized_name,
                    pin_code,
                    groups_dict
                )
                if matched_key:
                    group_key = matched_key
                    fuzzy_matches_count += 1
            
            # Add to existing group or create new one
            if group_key in groups_dict:
                groups_dict[group_key].add_record(address_record, parsed_address)
            else:
                # Create new group with unique ID
                group_id = str(uuid.uuid4())
                new_group = ConsolidatedGroup(
                    group_id=group_id,
                    society_name=society_name,  # Use original name for display
                    pin_code=pin_code,
                    records=[(address_record, parsed_address)],
                    record_count=1
                )
                groups_dict[group_key] = new_group
                logger.debug(f"Created new group: {society_name} (PIN: {pin_code})")
        
        # Convert dictionary to list
        consolidated_groups = list(groups_dict.values())
        
        # Add unmatched group if it has records
        if unmatched_group.record_count > 0:
            consolidated_groups.append(unmatched_group)
            logger.info(f"Unmatched group contains {unmatched_group.record_count} records")
        
        self._total_groups = len(consolidated_groups)
        
        logger.info(
            f"Consolidation complete. Created {self._total_groups} groups "
            f"({self._unmatched_count} unmatched records)"
        )
        
        if self.fuzzy_matching:
            logger.info(f"Fuzzy matching merged {fuzzy_matches_count} records into existing groups")
        
        return consolidated_groups
    
    def _normalize_society_name(self, society_name: str) -> str:
        """Normalize society names for matching.
        
        Applies the following transformations:
        - Convert to lowercase
        - Trim leading/trailing whitespace
        - Replace special characters with spaces (to handle hyphens, etc.)
        - Remove remaining special characters (keep only alphanumeric and spaces)
        - Collapse multiple spaces to single space
        
        Args:
            society_name: Original society name
            
        Returns:
            Normalized society name
        """
        if not society_name:
            return ""
        
        # Convert to lowercase
        normalized = society_name.lower()
        
        # Trim whitespace
        normalized = normalized.strip()
        
        # Replace common separators (hyphens, underscores) with spaces
        normalized = re.sub(r'[-_]', ' ', normalized)
        
        # Remove remaining special characters (keep alphanumeric and spaces)
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        
        # Collapse multiple spaces to single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Final trim
        normalized = normalized.strip()
        
        return normalized
    
    def _find_fuzzy_match(
        self,
        normalized_name: str,
        pin_code: str,
        groups_dict: Dict[Tuple[str, str], ConsolidatedGroup]
    ) -> Optional[Tuple[str, str]]:
        """Find a fuzzy match for the given society name and PIN code.
        
        Searches existing groups for a society name that is similar enough
        (above the similarity threshold) and has the same PIN code.
        
        Args:
            normalized_name: Normalized society name to match
            pin_code: PIN code to match
            groups_dict: Dictionary of existing groups
            
        Returns:
            Matching group key (normalized_name, pin_code) or None if no match
        """
        if not normalized_name:
            return None
        
        # Look for groups with the same PIN code
        for (existing_name, existing_pin), group in groups_dict.items():
            # PIN code must match exactly
            if existing_pin != pin_code:
                continue
            
            # Calculate similarity between society names
            similarity = self._calculate_similarity(normalized_name, existing_name)
            
            # Check if similarity exceeds threshold
            if similarity >= self.similarity_threshold:
                logger.debug(
                    f"Fuzzy match found: '{normalized_name}' matches '{existing_name}' "
                    f"(similarity: {similarity:.2f})"
                )
                return (existing_name, existing_pin)
        
        # No match found
        return None
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate string similarity score between two society names.
        
        Uses rapidfuzz's token_sort_ratio which handles word order differences
        and is good for matching similar names with slight variations.
        
        Args:
            name1: First society name
            name2: Second society name
            
        Returns:
            Similarity score between 0 and 1
        """
        if not name1 or not name2:
            return 0.0
        
        # Use token_sort_ratio which handles word order
        # Returns score 0-100, normalize to 0-1
        score = fuzz.token_sort_ratio(name1, name2)
        return score / 100.0
    
    def get_statistics(self) -> Dict[str, int]:
        """Get consolidation statistics.
        
        Returns:
            Dictionary with consolidation statistics
        """
        return {
            "total_groups": self._total_groups,
            "unmatched_count": self._unmatched_count
        }
