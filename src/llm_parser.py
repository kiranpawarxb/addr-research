"""LLM Parser component for the Address Consolidation System."""

import json
import logging
import time
from typing import List, Optional, Dict, Any
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models import ParsedAddress


logger = logging.getLogger(__name__)


class LLMParser:
    """Extracts structured address components from raw address text using an LLM API.
    
    Uses OpenAI-compatible API to parse unstructured Indian addresses into
    standardized fields. Implements retry logic and batch processing for efficiency.
    """
    
    # Prompt template with Indian address examples
    PROMPT_TEMPLATE = """You are an expert at parsing Indian addresses. Extract the following fields from the given address text and return them as a JSON object:

Fields to extract:
- UN (Unit Number): Flat, apartment, or unit identifier
- SN (Society Name): Housing society, building, or residential complex name
- LN (Landmark): Nearby notable location or reference point
- RD (Road): Street or road name
- SL (Sub-locality): Sub-area or neighborhood within a locality
- LOC (Locality): Broader area or locality name
- CY (City): City name
- DIS (District): District name
- ST (State): State name
- CN (Country): Country name (typically "India")
- PIN (PIN code): Postal index number
- Note: Any additional information or parsing notes

Examples:

Input: "Flat 301, Prestige Shantiniketan, Whitefield Main Road, Whitefield, Bangalore, Karnataka 560066"
Output: {{
  "UN": "Flat 301",
  "SN": "Prestige Shantiniketan",
  "LN": "",
  "RD": "Whitefield Main Road",
  "SL": "Whitefield",
  "LOC": "Whitefield",
  "CY": "Bangalore",
  "DIS": "Bangalore Urban",
  "ST": "Karnataka",
  "CN": "India",
  "PIN": "560066",
  "Note": ""
}}

Input: "A-204, Lodha Splendora, Ghodbunder Road, Near Hypercity Mall, Thane West, Thane, Maharashtra 400607"
Output: {{
  "UN": "A-204",
  "SN": "Lodha Splendora",
  "LN": "Near Hypercity Mall",
  "RD": "Ghodbunder Road",
  "SL": "Thane West",
  "LOC": "Thane West",
  "CY": "Thane",
  "DIS": "Thane",
  "ST": "Maharashtra",
  "CN": "India",
  "PIN": "400607",
  "Note": ""
}}

Input: "12/B, Sector 15, Rohini, Delhi 110085"
Output: {{
  "UN": "12/B",
  "SN": "",
  "LN": "",
  "RD": "",
  "SL": "Sector 15",
  "LOC": "Rohini",
  "CY": "Delhi",
  "DIS": "North West Delhi",
  "ST": "Delhi",
  "CN": "India",
  "PIN": "110085",
  "Note": ""
}}

Now parse this address:
{address_text}

Return ONLY the JSON object, no additional text."""
    
    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        model: str = "gpt-4",
        max_retries: int = 3,
        timeout_seconds: int = 30,
        batch_size: int = 10
    ):
        """Initialize parser with LLM API configuration.
        
        Args:
            api_endpoint: URL of the LLM API endpoint
            api_key: API key for authentication
            model: Model name to use (default: gpt-4)
            max_retries: Maximum number of retry attempts (default: 3)
            timeout_seconds: Request timeout in seconds (default: 30)
            batch_size: Number of addresses to process in parallel (default: 10)
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        
        # Statistics
        self._total_parsed = 0
        self._total_failed = 0
        self._total_retries = 0
    
    def parse_address(self, raw_address: str) -> ParsedAddress:
        """Parse a single address and return structured components.
        
        Implements retry logic with exponential backoff for failed requests.
        
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
        
        # Build the prompt
        prompt = self._build_prompt(raw_address)
        
        # Attempt parsing with retries
        for attempt in range(self.max_retries):
            try:
                # Make API request
                response_data = self._make_api_request(prompt)
                
                # Extract and parse JSON from response
                parsed_address = self._extract_fields(response_data)
                parsed_address.parse_success = True
                
                self._total_parsed += 1
                if attempt > 0:
                    self._total_retries += attempt
                    logger.info(f"Successfully parsed address after {attempt + 1} attempts")
                else:
                    logger.debug(f"Successfully parsed address on first attempt")
                
                return parsed_address
                
            except requests.exceptions.Timeout:
                # Log timeout error (Requirement 7.2)
                logger.warning(
                    f"API request timeout (attempt {attempt + 1}/{self.max_retries}) "
                    f"for address: {raw_address[:50]}..."
                )
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 2^attempt seconds
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    # Final attempt failed
                    self._total_failed += 1
                    logger.error(
                        f"API timeout after {self.max_retries} retries for address: {raw_address[:50]}..."
                    )
                    return ParsedAddress(
                        parse_success=False,
                        parse_error=f"API timeout after {self.max_retries} retries"
                    )
                    
            except requests.exceptions.RequestException as e:
                # Log API error (Requirement 7.2)
                logger.error(
                    f"API request error (attempt {attempt + 1}/{self.max_retries}): {e} "
                    f"for address: {raw_address[:50]}..."
                )
                if attempt < self.max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self._total_failed += 1
                    logger.error(
                        f"API error after {self.max_retries} retries: {str(e)}"
                    )
                    return ParsedAddress(
                        parse_success=False,
                        parse_error=f"API error after {self.max_retries} retries: {str(e)}"
                    )
                    
            except json.JSONDecodeError as e:
                # Log invalid JSON error (Requirement 2.5)
                logger.error(
                    f"Invalid JSON response (attempt {attempt + 1}/{self.max_retries}): {e} "
                    f"for address: {raw_address[:50]}..."
                )
                if attempt < self.max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self._total_failed += 1
                    logger.error(
                        f"Invalid JSON after {self.max_retries} retries for address: {raw_address[:50]}..."
                    )
                    return ParsedAddress(
                        parse_success=False,
                        parse_error=f"Invalid JSON after {self.max_retries} retries"
                    )
                    
            except Exception as e:
                logger.error(f"Unexpected error parsing address: {e}", exc_info=True)
                self._total_failed += 1
                return ParsedAddress(
                    parse_success=False,
                    parse_error=f"Unexpected error: {str(e)}"
                )
        
        # Should not reach here, but just in case
        self._total_failed += 1
        return ParsedAddress(
            parse_success=False,
            parse_error="Unknown error during parsing"
        )
    
    def parse_batch(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse multiple addresses in parallel for efficiency.
        
        Uses ThreadPoolExecutor to process addresses concurrently.
        
        Args:
            addresses: List of unstructured address texts to parse
            
        Returns:
            List of ParsedAddress objects in the same order as input
        """
        if not addresses:
            return []
        
        logger.info(f"Starting batch parsing of {len(addresses)} addresses...")
        
        # Use ThreadPoolExecutor for parallel processing
        results = [None] * len(addresses)
        
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.parse_address, addr): i
                for i, addr in enumerate(addresses)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    completed += 1
                    
                    # Log progress every 10 completions
                    if completed % 10 == 0:
                        logger.debug(f"Batch progress: {completed}/{len(addresses)} addresses processed")
                        
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
    
    def _build_prompt(self, raw_address: str) -> str:
        """Build the prompt with examples for the LLM.
        
        Args:
            raw_address: The address text to parse
            
        Returns:
            Complete prompt string
        """
        return self.PROMPT_TEMPLATE.format(address_text=raw_address)
    
    def _make_api_request(self, prompt: str) -> Dict[str, Any]:
        """Make an API request to the LLM endpoint.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Response data from the API
            
        Raises:
            requests.exceptions.RequestException: For API errors
            requests.exceptions.Timeout: For timeout errors
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at parsing Indian addresses. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent parsing
            "max_tokens": 500
        }
        
        response = requests.post(
            self.api_endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds
        )
        
        # Check for HTTP errors
        response.raise_for_status()
        
        return response.json()
    
    def _extract_fields(self, response_data: Dict[str, Any]) -> ParsedAddress:
        """Extract and parse JSON fields from the LLM response.
        
        Args:
            response_data: Raw response data from the API
            
        Returns:
            ParsedAddress object with extracted fields
            
        Raises:
            json.JSONDecodeError: If the response contains invalid JSON
            KeyError: If the response structure is unexpected
        """
        # Extract the content from the response
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise KeyError(f"Unexpected API response structure: {e}")
        
        # Parse the JSON content
        # The LLM might wrap the JSON in markdown code blocks, so clean it
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.startswith("```"):
            content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        content = content.strip()
        
        # Parse JSON
        parsed_json = json.loads(content)
        
        # Extract fields with defaults for missing values
        parsed_address = ParsedAddress(
            unit_number=parsed_json.get("UN", ""),
            society_name=parsed_json.get("SN", ""),
            landmark=parsed_json.get("LN", ""),
            road=parsed_json.get("RD", ""),
            sub_locality=parsed_json.get("SL", ""),
            locality=parsed_json.get("LOC", ""),
            city=parsed_json.get("CY", ""),
            district=parsed_json.get("DIS", ""),
            state=parsed_json.get("ST", ""),
            country=parsed_json.get("CN", ""),
            pin_code=parsed_json.get("PIN", ""),
            note=parsed_json.get("Note", ""),
            parse_success=False,  # Will be set to True by caller
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
            "total_retries": self._total_retries
        }
