"""Hybrid Address Parser - Intelligent routing between Local and Shiprocket parsers.

This parser automatically routes addresses to the most appropriate parser:
- Complex addresses → Shiprocket (high quality)
- Simple addresses → Local (high speed)
- Fallback logic for reliability
"""

import logging
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models import ParsedAddress
from src.local_llm_parser import LocalLLMParser
from src.shiprocket_parser import ShiprocketParser

logger = logging.getLogger(__name__)


class HybridParser:
    """Hybrid parser that intelligently routes addresses between Local and Shiprocket parsers.
    
    Uses complexity analysis to determine the best parser for each address:
    - Complex addresses (multiple components) → Shiprocket for quality
    - Simple addresses (basic structure) → Local for speed
    - Automatic fallback for reliability
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        batch_size: int = 10,
        complexity_threshold: float = 0.6,
        enable_fallback: bool = True
    ):
        """Initialize hybrid parser.
        
        Args:
            use_gpu: Whether to use GPU for Shiprocket parser
            batch_size: Number of addresses to process in parallel
            complexity_threshold: Threshold for routing (0.0-1.0, higher = more to Shiprocket)
            enable_fallback: Whether to fallback to other parser on failure
        """
        self.complexity_threshold = complexity_threshold
        self.enable_fallback = enable_fallback
        self.batch_size = batch_size
        
        # Initialize parsers
        self.local_parser = LocalLLMParser()
        self.shiprocket_parser = ShiprocketParser(use_gpu=use_gpu, batch_size=batch_size)
        
        # Statistics
        self._total_parsed = 0
        self._total_failed = 0
        self._local_used = 0
        self._shiprocket_used = 0
        self._fallback_used = 0
        
        logger.info(f"Initialized HybridParser (complexity_threshold={complexity_threshold})")
    
    def parse_address(self, raw_address: str) -> ParsedAddress:
        """Parse a single address using intelligent routing.
        
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
        
        logger.debug(f"Parsing address with hybrid routing: {raw_address[:100]}...")
        
        # Analyze address complexity
        complexity_score = self._analyze_complexity(raw_address)
        use_shiprocket = complexity_score >= self.complexity_threshold
        
        logger.debug(f"Complexity score: {complexity_score:.2f}, using {'Shiprocket' if use_shiprocket else 'Local'}")
        
        # Try primary parser
        primary_parser = self.shiprocket_parser if use_shiprocket else self.local_parser
        primary_name = "Shiprocket" if use_shiprocket else "Local"
        
        try:
            result = primary_parser.parse_address(raw_address)
            
            if result.parse_success:
                # Success with primary parser
                if use_shiprocket:
                    self._shiprocket_used += 1
                else:
                    self._local_used += 1
                
                result.note = f"Parsed using {primary_name} parser (complexity: {complexity_score:.2f})"
                self._total_parsed += 1
                
                logger.debug(f"Successfully parsed with {primary_name} parser")
                return result
            
            else:
                logger.warning(f"{primary_name} parser failed: {result.parse_error}")
                
                # Try fallback if enabled
                if self.enable_fallback:
                    return self._try_fallback(raw_address, use_shiprocket, complexity_score)
                else:
                    self._total_failed += 1
                    return result
                    
        except Exception as e:
            logger.error(f"Error with {primary_name} parser: {e}")
            
            # Try fallback if enabled
            if self.enable_fallback:
                return self._try_fallback(raw_address, use_shiprocket, complexity_score)
            else:
                self._total_failed += 1
                return ParsedAddress(
                    parse_success=False,
                    parse_error=f"{primary_name} parser error: {str(e)}"
                )
    
    def _try_fallback(self, raw_address: str, primary_was_shiprocket: bool, complexity_score: float) -> ParsedAddress:
        """Try fallback parser when primary fails.
        
        Args:
            raw_address: Address text to parse
            primary_was_shiprocket: Whether primary parser was Shiprocket
            complexity_score: Complexity score of the address
            
        Returns:
            ParsedAddress from fallback parser
        """
        fallback_parser = self.local_parser if primary_was_shiprocket else self.shiprocket_parser
        fallback_name = "Local" if primary_was_shiprocket else "Shiprocket"
        
        logger.debug(f"Trying fallback parser: {fallback_name}")
        
        try:
            result = fallback_parser.parse_address(raw_address)
            
            if result.parse_success:
                # Success with fallback
                self._fallback_used += 1
                if primary_was_shiprocket:
                    self._local_used += 1
                else:
                    self._shiprocket_used += 1
                
                result.note = f"Parsed using {fallback_name} parser (fallback, complexity: {complexity_score:.2f})"
                self._total_parsed += 1
                
                logger.debug(f"Successfully parsed with fallback {fallback_name} parser")
                return result
            
            else:
                # Both parsers failed
                logger.error(f"Both parsers failed for address: {raw_address[:100]}")
                self._total_failed += 1
                
                return ParsedAddress(
                    parse_success=False,
                    parse_error=f"Both parsers failed. Primary: {primary_was_shiprocket}, Fallback: {result.parse_error}"
                )
                
        except Exception as e:
            logger.error(f"Fallback parser {fallback_name} error: {e}")
            self._total_failed += 1
            
            return ParsedAddress(
                parse_success=False,
                parse_error=f"Both parsers failed. Primary: {primary_was_shiprocket}, Fallback error: {str(e)}"
            )
    
    def _analyze_complexity(self, address: str) -> float:
        """Analyze address complexity to determine routing.
        
        Args:
            address: Raw address text
            
        Returns:
            Complexity score (0.0-1.0, higher = more complex)
        """
        address_lower = address.lower()
        
        # Complexity indicators (weighted) - Updated for better detection
        indicators = {
            # Road/street indicators (high value for Shiprocket)
            'road_indicators': {
                'patterns': ['road', 'street', 'lane', 'marg', 'path', 'avenue', 'cross', 'galli'],
                'weight': 0.30
            },
            
            # Landmark indicators (high value for Shiprocket)
            'landmark_indicators': {
                'patterns': ['near', 'opposite', 'behind', 'beside', 'next to', 'close to', 'besides'],
                'weight': 0.25
            },
            
            # Complex society/building indicators
            'society_indicators': {
                'patterns': ['society', 'complex', 'residency', 'apartments', 'towers', 'enclave', 'chembers', 'nisarg'],
                'weight': 0.20
            },
            
            # Area/locality indicators
            'area_indicators': {
                'patterns': ['phase', 'sector', 'block', 'wing', 'plot', 'survey', 'nagar', 'vihar'],
                'weight': 0.15
            },
            
            # Multiple components (commas indicate complexity)
            'structure_indicators': {
                'patterns': [','],
                'weight': 0.05  # Per comma
            },
            
            # Specific location details
            'detail_indicators': {
                'patterns': ['floor', 'flat', 'unit', 'apartment', 'shop', 'office', 'daisy', 'center'],
                'weight': 0.05
            }
        }
        
        total_score = 0.0
        
        for category, config in indicators.items():
            patterns = config['patterns']
            weight = config['weight']
            
            # Count matches for this category
            matches = sum(1 for pattern in patterns if pattern in address_lower)
            
            if category == 'structure_indicators':
                # For commas, each comma adds to complexity
                category_score = matches * weight
            else:
                # For other categories, normalize by pattern count
                category_score = min(matches / len(patterns), 1.0) * weight
            
            total_score += category_score
            
            logger.debug(f"Category {category}: {matches} matches, score: {category_score:.3f}")
        
        # Additional factors
        
        # Length factor (longer addresses tend to be more complex)
        length_factor = min(len(address) / 100.0, 1.0) * 0.10
        total_score += length_factor
        
        # Word count factor (more words = more complex)
        word_count = len(address.split())
        word_factor = min(word_count / 10.0, 1.0) * 0.10
        total_score += word_factor
        
        # Numeric complexity (multiple numbers suggest detailed addressing)
        numbers = re.findall(r'\d+', address)
        if len(numbers) >= 3:  # Multiple numbers suggest complexity
            total_score += 0.15
        
        # Ensure score is between 0 and 1
        final_score = min(max(total_score, 0.0), 1.0)
        
        logger.debug(f"Final complexity score: {final_score:.3f}")
        return final_score
    
    def parse_batch(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse multiple addresses in parallel with intelligent routing.
        
        Args:
            addresses: List of unstructured address texts to parse
            
        Returns:
            List of ParsedAddress objects in the same order as input
        """
        if not addresses:
            return []
        
        logger.info(f"Starting hybrid batch parsing of {len(addresses)} addresses...")
        
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
                        logger.debug(f"Hybrid batch progress: {completed}/{len(addresses)}")
                        
                except Exception as e:
                    logger.error(f"Error processing address at index {index}: {e}")
                    results[index] = ParsedAddress(
                        parse_success=False,
                        parse_error=f"Hybrid batch processing error: {str(e)}"
                    )
        
        success_count = sum(1 for r in results if r.parse_success)
        failed_count = sum(1 for r in results if not r.parse_success)
        
        logger.info(
            f"Hybrid batch parsing complete. "
            f"Success: {success_count}/{len(addresses)}, "
            f"Failed: {failed_count}/{len(addresses)}"
        )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics.
        
        Returns:
            Dictionary with parsing statistics
        """
        total_attempts = self._total_parsed + self._total_failed
        success_rate = (self._total_parsed / total_attempts * 100) if total_attempts > 0 else 0
        
        # Calculate parser usage percentages
        total_used = self._local_used + self._shiprocket_used
        local_pct = (self._local_used / total_used * 100) if total_used > 0 else 0
        shiprocket_pct = (self._shiprocket_used / total_used * 100) if total_used > 0 else 0
        fallback_pct = (self._fallback_used / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            "total_parsed": self._total_parsed,
            "total_failed": self._total_failed,
            "success_rate_percent": round(success_rate, 1),
            "total_attempts": total_attempts,
            "local_used": self._local_used,
            "shiprocket_used": self._shiprocket_used,
            "fallback_used": self._fallback_used,
            "local_usage_percent": round(local_pct, 1),
            "shiprocket_usage_percent": round(shiprocket_pct, 1),
            "fallback_usage_percent": round(fallback_pct, 1),
            "complexity_threshold": self.complexity_threshold
        }
    
    def get_parser_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed statistics from individual parsers.
        
        Returns:
            Dictionary with statistics from both parsers
        """
        return {
            "local_parser": self.local_parser.get_statistics(),
            "shiprocket_parser": self.shiprocket_parser.get_statistics()
        }
    
    def tune_complexity_threshold(self, addresses: List[str], target_shiprocket_usage: float = 0.3) -> float:
        """Automatically tune complexity threshold based on sample addresses.
        
        Args:
            addresses: Sample addresses to analyze
            target_shiprocket_usage: Target percentage of addresses to route to Shiprocket (0.0-1.0)
            
        Returns:
            Recommended complexity threshold
        """
        if not addresses:
            return self.complexity_threshold
        
        logger.info(f"Tuning complexity threshold with {len(addresses)} addresses...")
        
        # Calculate complexity scores for all addresses
        complexity_scores = [self._analyze_complexity(addr) for addr in addresses]
        complexity_scores.sort()
        
        # Find threshold that achieves target usage
        target_index = int(len(complexity_scores) * (1.0 - target_shiprocket_usage))
        recommended_threshold = complexity_scores[target_index] if target_index < len(complexity_scores) else 0.5
        
        logger.info(f"Recommended complexity threshold: {recommended_threshold:.3f} (target usage: {target_shiprocket_usage*100:.1f}%)")
        
        return recommended_threshold