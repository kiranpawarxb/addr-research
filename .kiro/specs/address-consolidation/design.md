# Design Document: Address Consolidation System

## Overview

The Address Consolidation System is a Python-based application that processes large CSV files containing Indian address data. It uses LLM-based parsing to extract structured address components from unstructured text, then consolidates addresses based on Society Name and geographic identifiers. The system is designed to handle large datasets efficiently through batch processing and parallel execution.

## Architecture

The system follows a pipeline architecture with four main stages:

1. **Input Stage**: CSV file reading and validation
2. **Parsing Stage**: LLM-based address component extraction
3. **Consolidation Stage**: Grouping addresses by Society Name and PIN code
4. **Output Stage**: Writing results to CSV with parsed components and group identifiers

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   CSV       │────▶│   LLM        │────▶│ Consolidation│────▶│   Output    │
│   Reader    │     │   Parser     │     │   Engine     │     │   Writer    │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
      │                    │                     │                    │
      ▼                    ▼                     ▼                    ▼
  Validation         Retry Logic          Fuzzy Matching        Statistics
```

## Components and Interfaces

### 1. CSV Reader Component

**Responsibility**: Load and validate CSV files containing address data.

**Interface**:
```python
class CSVReader:
    def __init__(self, file_path: str, required_columns: List[str]):
        """Initialize reader with file path and required columns."""
        
    def read(self) -> Iterator[AddressRecord]:
        """Read CSV file and yield AddressRecord objects."""
        
    def validate_columns(self) -> Tuple[bool, List[str]]:
        """Validate that required columns exist. Returns (is_valid, missing_columns)."""
```

**Key behaviors**:
- Streams large CSV files using iterators to avoid memory issues
- Validates required columns before processing
- Skips malformed rows and logs warnings with row numbers
- Reports total count of successfully loaded records

### 2. LLM Parser Component

**Responsibility**: Extract structured address components from raw address text using an LLM API.

**Interface**:
```python
class LLMParser:
    def __init__(self, api_endpoint: str, api_key: str, max_retries: int = 3):
        """Initialize parser with LLM API configuration."""
        
    def parse_address(self, raw_address: str) -> ParsedAddress:
        """Parse a single address and return structured components."""
        
    def parse_batch(self, addresses: List[str]) -> List[ParsedAddress]:
        """Parse multiple addresses in parallel for efficiency."""
        
    def _build_prompt(self, raw_address: str) -> str:
        """Build the prompt with examples for the LLM."""
```

**Key behaviors**:
- Uses the standardized prompt format with Indian address examples
- Implements retry logic (up to 3 attempts) for failed API calls
- Parses JSON response and extracts all 12 fields
- Handles invalid JSON responses gracefully
- Supports batch processing for improved performance

### 3. Consolidation Engine Component

**Responsibility**: Group addresses based on Society Name and geographic identifiers.

**Interface**:
```python
class ConsolidationEngine:
    def __init__(self, fuzzy_matching: bool = False, similarity_threshold: float = 0.85):
        """Initialize engine with matching configuration."""
        
    def consolidate(self, parsed_addresses: List[ParsedAddress]) -> List[ConsolidatedGroup]:
        """Group addresses by Society Name and PIN code."""
        
    def _normalize_society_name(self, society_name: str) -> str:
        """Normalize society names for matching."""
        
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate string similarity score between two society names."""
```

**Key behaviors**:
- Groups addresses with matching Society Name (SN) and PIN code
- Normalizes society names (lowercase, trim whitespace, remove special chars)
- Optionally uses fuzzy matching with configurable similarity threshold
- Assigns unique group IDs to each consolidated group
- Places addresses with empty/missing SN in an "unmatched" group

### 4. Output Writer Component

**Responsibility**: Write consolidated results to CSV with all parsed components.

**Interface**:
```python
class OutputWriter:
    def __init__(self, output_path: str):
        """Initialize writer with output file path."""
        
    def write(self, consolidated_groups: List[ConsolidatedGroup]) -> int:
        """Write consolidated groups to CSV. Returns record count."""
        
    def _build_output_row(self, record: AddressRecord, parsed: ParsedAddress, group_id: str) -> Dict:
        """Build a single output row with all fields."""
```

**Key behaviors**:
- Writes CSV with original columns plus parsed fields (UN, SN, LN, RD, SL, LOC, CY, DIS, ST, CN, PIN, Note)
- Adds group_id column to identify consolidated groups
- Preserves all original data from input CSV
- Reports output file path and total record count

### 5. Statistics Reporter Component

**Responsibility**: Calculate and display consolidation statistics.

**Interface**:
```python
class StatisticsReporter:
    def generate_stats(self, consolidated_groups: List[ConsolidatedGroup], 
                      total_records: int, 
                      failed_parses: int) -> ConsolidationStats:
        """Generate comprehensive statistics about the consolidation."""
        
    def display(self, stats: ConsolidationStats) -> None:
        """Display statistics in a readable format."""
```

**Key behaviors**:
- Calculates total number of consolidated groups
- Computes average addresses per group
- Identifies largest group by record count
- Calculates percentage of successfully matched addresses
- Reports parsing success/failure rates

## Data Models

### AddressRecord
```python
@dataclass
class AddressRecord:
    """Raw address record from CSV file."""
    addr_hash_key: str
    addr_text: str
    city_id: str
    pincode: str
    state_id: str
    zone_id: str
    address_id: str
    assigned_pickup_dlvd_geo_points: str
    assigned_pickup_dlvd_geo_points_count: int
    raw_data: Dict[str, Any]  # Preserve all original columns
```

### ParsedAddress
```python
@dataclass
class ParsedAddress:
    """Structured address components extracted by LLM."""
    unit_number: str  # UN
    society_name: str  # SN
    landmark: str  # LN
    road: str  # RD
    sub_locality: str  # SL
    locality: str  # LOC
    city: str  # CY
    district: str  # DIS
    state: str  # ST
    country: str  # CN
    pin_code: str  # PIN
    note: str  # Note
    parse_success: bool
    parse_error: Optional[str]
```

### ConsolidatedGroup
```python
@dataclass
class ConsolidatedGroup:
    """A group of addresses belonging to the same location."""
    group_id: str
    society_name: str
    pin_code: str
    records: List[Tuple[AddressRecord, ParsedAddress]]
    record_count: int
```

### ConsolidationStats
```python
@dataclass
class ConsolidationStats:
    """Statistics about the consolidation process."""
    total_records: int
    total_groups: int
    avg_records_per_group: float
    largest_group_size: int
    largest_group_name: str
    match_percentage: float
    parse_success_count: int
    parse_failure_count: int
    unmatched_count: int
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: CSV loading preserves record count
*For any* valid CSV file with required columns, the number of Address Records loaded should equal the number of non-malformed rows in the file.
**Validates: Requirements 1.1, 1.2, 1.5**

### Property 2: Malformed rows are skipped gracefully
*For any* CSV file containing a mix of valid and malformed rows, the system should load all valid rows and skip malformed rows without failing.
**Validates: Requirements 1.4**

### Property 3: LLM parser extracts all fields
*For any* valid JSON response from the LLM, all 12 address fields (UN, SN, LN, RD, SL, LOC, CY, DIS, ST, CN, PIN, Note) should be extracted into the ParsedAddress object.
**Validates: Requirements 2.2**

### Property 4: Missing fields get default values
*For any* LLM response with missing fields, those fields should be populated with empty string or "NA" in the ParsedAddress.
**Validates: Requirements 2.3**

### Property 5: Invalid JSON is handled gracefully
*For any* invalid JSON response from the LLM, the system should mark the address as unparseable and continue processing without crashing.
**Validates: Requirements 2.5**

### Property 6: Grouping requires both Society Name and PIN match
*For any* set of ParsedAddresses, two addresses should be in the same ConsolidatedGroup if and only if they have matching Society Name (SN) and PIN code.
**Validates: Requirements 3.1, 3.2**

### Property 7: Empty Society Names go to unmatched group
*For any* ParsedAddress with empty or missing Society Name, it should be placed in the "unmatched" group regardless of other fields.
**Validates: Requirements 3.5**

### Property 8: Data preservation through pipeline
*For any* Address Record processed through the system, all original CSV columns and all parsed fields should appear in the output CSV.
**Validates: Requirements 3.3, 4.4**

### Property 9: Output contains all required columns
*For any* export operation, the output CSV should contain all 12 parsed field columns (UN, SN, LN, RD, SL, LOC, CY, DIS, ST, CN, PIN, Note) plus a group_id column.
**Validates: Requirements 4.2, 4.3**

### Property 10: Group count matches unique (SN, PIN) pairs
*For any* consolidation result, the number of ConsolidatedGroups should equal the number of unique (Society Name, PIN code) combinations in the parsed addresses.
**Validates: Requirements 3.4, 5.1**

### Property 11: Average calculation is accurate
*For any* consolidation result, the average records per group should equal total records divided by total groups.
**Validates: Requirements 5.2**

### Property 12: Largest group identification is correct
*For any* consolidation result, the identified largest group should have a record count greater than or equal to all other groups.
**Validates: Requirements 5.3**

### Property 13: Match percentage calculation is accurate
*For any* consolidation result, the match percentage should equal (number of matched records / total records) × 100.
**Validates: Requirements 5.4**

### Property 14: Fuzzy matching respects threshold
*For any* two Society Names and a configured similarity threshold, they should be grouped together if and only if their similarity score exceeds the threshold.
**Validates: Requirements 6.2, 6.3**

### Property 15: Retry logic executes correctly
*For any* LLM timeout, the system should retry the request exactly 3 times before marking the address as unparseable.
**Validates: Requirements 7.1**

### Property 16: Processing continues after errors
*For any* set of addresses where some fail to parse, the system should successfully process all parseable addresses and include unparseable ones in output with empty parsed fields.
**Validates: Requirements 7.2, 7.3, 7.4**

### Property 17: Parse success/failure counts are accurate
*For any* processing run, the sum of successfully parsed and failed addresses should equal the total number of input addresses.
**Validates: Requirements 7.5**

## Error Handling

The system implements comprehensive error handling at each stage:

### CSV Reading Errors
- **Missing required columns**: Report error with specific column names, exit gracefully
- **File not found**: Report clear error message with file path
- **Malformed rows**: Skip row, log warning with row number, continue processing
- **Empty file**: Report error, exit gracefully

### LLM Parsing Errors
- **API timeout**: Retry up to 3 times with exponential backoff
- **Invalid JSON response**: Log error, mark address as unparseable, continue
- **Missing required fields**: Use default values ("NA" or empty string)
- **API authentication failure**: Report error, exit gracefully
- **Rate limiting**: Implement exponential backoff and retry

### Consolidation Errors
- **Empty Society Name**: Place in "unmatched" group
- **Invalid PIN code**: Use as-is for grouping (don't validate format)
- **Duplicate group IDs**: Use UUID to ensure uniqueness

### Output Writing Errors
- **File write permission denied**: Report error with file path
- **Disk space full**: Report error, attempt to clean up partial file
- **Invalid characters in data**: Escape or replace with safe characters

## Testing Strategy

The Address Consolidation System will use a dual testing approach combining unit tests and property-based tests to ensure comprehensive coverage.

### Unit Testing Approach

Unit tests will verify specific examples and integration points:

- **CSV Reader**: Test with sample CSV files containing known data
- **LLM Parser**: Test with mocked LLM responses for known address formats
- **Consolidation Engine**: Test with specific address sets and expected groupings
- **Output Writer**: Test that output files are created with correct structure
- **Statistics Reporter**: Test calculations with known input values

### Property-Based Testing Approach

Property-based tests will verify universal properties across all inputs using **Hypothesis** (Python's property-based testing library).

**Configuration**:
- Each property test will run a minimum of 100 iterations
- Tests will use custom generators for realistic Indian address data
- Each test will be tagged with a comment referencing the design document property

**Test Tagging Format**:
```python
# Feature: address-consolidation, Property 1: CSV loading preserves record count
@given(csv_data=valid_csv_strategy())
def test_csv_loading_preserves_count(csv_data):
    # Test implementation
```

**Key Property Tests**:

1. **CSV Loading Properties** (Properties 1-2)
   - Generate random CSV files with varying structures
   - Verify record counts and malformed row handling

2. **Parsing Properties** (Properties 3-5)
   - Generate random LLM responses (valid and invalid JSON)
   - Verify field extraction and error handling

3. **Consolidation Properties** (Properties 6-7, 10)
   - Generate random sets of parsed addresses
   - Verify grouping logic and group counts

4. **Data Preservation Properties** (Properties 8-9)
   - Generate random address records
   - Verify all data appears in output

5. **Statistics Properties** (Properties 11-13)
   - Generate random consolidation results
   - Verify calculation accuracy

6. **Fuzzy Matching Properties** (Property 14)
   - Generate pairs of similar/dissimilar society names
   - Verify threshold-based grouping

7. **Error Handling Properties** (Properties 15-17)
   - Generate scenarios with timeouts and errors
   - Verify retry logic and error recovery

**Test Generators**:

The test suite will include custom Hypothesis strategies for generating realistic test data:

```python
# Generate realistic Indian address components
@composite
def indian_address_strategy(draw):
    """Generate realistic Indian address text."""
    unit = draw(st.text(min_size=1, max_size=10))
    society = draw(st.sampled_from(COMMON_SOCIETY_NAMES))
    locality = draw(st.sampled_from(COMMON_LOCALITIES))
    city = draw(st.sampled_from(INDIAN_CITIES))
    pincode = draw(st.integers(min_value=100000, max_value=999999))
    return f"{unit}, {society}, {locality}, {city} {pincode}"

# Generate valid CSV data
@composite
def valid_csv_strategy(draw):
    """Generate valid CSV data with required columns."""
    num_rows = draw(st.integers(min_value=1, max_value=100))
    # Generate rows with required columns
    
# Generate LLM JSON responses
@composite
def llm_response_strategy(draw, valid=True):
    """Generate LLM JSON responses (valid or invalid)."""
    if valid:
        return {
            "UN": draw(st.text(min_size=0, max_size=20)),
            "SN": draw(st.text(min_size=0, max_size=50)),
            # ... other fields
        }
    else:
        # Generate invalid JSON
```

### Integration Testing

Integration tests will verify end-to-end workflows:

- Load sample CSV → Parse → Consolidate → Export → Verify output
- Test with real LLM API (using test API key)
- Test with large files (performance testing)
- Test error recovery scenarios

### Test Coverage Goals

- Unit test coverage: 80% minimum
- Property test coverage: All 17 correctness properties
- Integration test coverage: All major workflows
- Error path coverage: All error handling branches

## Performance Considerations

### Scalability

- **Streaming CSV reading**: Process files larger than available memory
- **Batch processing**: Send multiple addresses to LLM in parallel
- **Async I/O**: Use asyncio for concurrent LLM API calls
- **Progress tracking**: Report progress for long-running operations

### Optimization Strategies

- **Caching**: Cache normalized society names to avoid recomputation
- **Connection pooling**: Reuse HTTP connections for LLM API calls
- **Lazy evaluation**: Only parse addresses that need consolidation
- **Parallel processing**: Use multiprocessing for CPU-bound operations

### Expected Performance

- **Small files** (<10K records): < 5 minutes
- **Medium files** (10K-100K records): 10-30 minutes
- **Large files** (>100K records): 30-60 minutes

Performance depends heavily on LLM API response times and rate limits.

## Configuration

The system will support configuration via a YAML file:

```yaml
# config.yaml
input:
  file_path: "export_customer_address_store_p0.csv"
  required_columns:
    - addr_text
    - pincode
    - city_id

llm:
  api_endpoint: "https://api.openai.com/v1/chat/completions"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
  max_retries: 3
  timeout_seconds: 30
  batch_size: 10

consolidation:
  fuzzy_matching: true
  similarity_threshold: 0.85
  normalize_society_names: true

output:
  file_path: "consolidated_addresses.csv"
  include_statistics: true

logging:
  level: "INFO"
  file_path: "consolidation.log"
```

## Dependencies

- **Python 3.9+**: Core language
- **pandas**: CSV reading and writing
- **requests**: HTTP client for LLM API calls
- **hypothesis**: Property-based testing framework
- **pytest**: Unit testing framework
- **pyyaml**: Configuration file parsing
- **rapidfuzz**: Fuzzy string matching
- **tqdm**: Progress bar display

## Future Enhancements

1. **Support for multiple LLM providers** (OpenAI, Anthropic, local models)
2. **Interactive review mode** for manual verification of groupings
3. **Address standardization** using postal service APIs
4. **Geocoding integration** to validate addresses with coordinates
5. **Web UI** for easier configuration and result visualization
6. **Database storage** for large-scale processing and querying
