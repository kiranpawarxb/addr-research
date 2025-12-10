# Full CSV Address Processing Design

## Overview

This system processes the entire CSV dataset using the Shiprocket parser to generate comprehensive parsed address data and statistical summaries. The design focuses on scalability, reliability, and detailed analytics.

## Architecture

### High-Level Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Reader    │───▶│  Batch Processor │───▶│  Output Writer  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Shiprocket Parser│
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Statistics Engine│
                    └──────────────────┘
```

### Processing Flow

1. **CSV Loading**: Read and validate input CSV structure
2. **Batch Processing**: Process addresses in configurable batches (default: 100)
3. **Address Parsing**: Use Shiprocket parser for each address
4. **Data Aggregation**: Collect results and build statistics
5. **Output Generation**: Create output CSV and summary reports

## Components and Interfaces

### CSVProcessor Class

```python
class FullCSVProcessor:
    def __init__(self, 
                 csv_file: str,
                 batch_size: int = 100,
                 output_dir: str = "output"):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.shiprocket = ShiprocketParser(use_gpu=False)
        self.statistics = ProcessingStatistics()
    
    def process_all_addresses(self) -> ProcessingResult
    def generate_output_csv(self, results: List[ParsedResult]) -> str
    def generate_pincode_summary(self, results: List[ParsedResult]) -> Dict
    def create_comprehensive_report(self) -> str
```

### ProcessingResult Data Model

```python
@dataclass
class ProcessedAddress:
    hash_key: str
    raw_address: str
    original_pincode: str
    
    # Parsed fields
    unit_number: str
    society_name: str
    landmark: str
    road: str
    sub_locality: str
    locality: str
    city: str
    district: str
    state: str
    country: str
    parsed_pincode: str
    
    # Metadata
    parse_success: bool
    parse_error: str
    processing_time: float
```

### Statistics Engine

```python
class ProcessingStatistics:
    def __init__(self):
        self.total_addresses = 0
        self.successful_parses = 0
        self.failed_parses = 0
        self.field_extraction_counts = defaultdict(int)
        self.pincode_stats = defaultdict(dict)
    
    def update_address_stats(self, result: ProcessedAddress)
    def calculate_extraction_rates(self) -> Dict[str, float]
    def generate_pincode_summary(self) -> Dict[str, Dict]
```

## Data Models

### Input CSV Structure
- `addr_hash_key`: Unique identifier
- `addr_text`: Raw address text
- `pincode`: Original PIN code (if available)
- Additional columns preserved in output

### Output CSV Structure
```
hash_key, raw_address, original_pincode,
unit_number, society_name, landmark, road, sub_locality, locality,
city, district, state, country, parsed_pincode,
parse_success, parse_error, processing_time
```

### PIN Code Summary Structure
```json
{
  "411001": {
    "total_addresses": 150,
    "successful_parses": 142,
    "success_rate": 94.7,
    "unique_localities": ["Koregaon Park", "Camp", "Cantonment"],
    "unique_societies": ["Panchshil Towers", "Kumar Pacific", "..."],
    "locality_count": 3,
    "society_count": 45,
    "extraction_rates": {
      "society": 89.4,
      "locality": 95.1,
      "road": 34.5
    }
  }
}
```

## Error Handling

### Robust Processing Strategy

1. **Individual Address Failures**: Continue processing, log errors
2. **Memory Management**: Process in batches, clear intermediate data
3. **Parser Failures**: Retry with exponential backoff
4. **File I/O Errors**: Graceful degradation with partial results
5. **Validation Errors**: Flag inconsistencies, continue processing

### Error Categories

- **Parse Errors**: Shiprocket model failures
- **Data Errors**: Invalid CSV structure or missing fields
- **System Errors**: Memory, disk space, or network issues
- **Validation Errors**: Inconsistent PIN codes or field formats

## Testing Strategy

### Unit Testing Approach

- **CSV Reader Tests**: Validate file parsing and error handling
- **Batch Processor Tests**: Test batch processing logic and memory management
- **Statistics Engine Tests**: Verify calculation accuracy and edge cases
- **Output Writer Tests**: Ensure correct CSV generation and formatting

### Integration Testing

- **End-to-End Processing**: Test complete workflow with sample data
- **Large Dataset Testing**: Validate performance with 10K+ addresses
- **Error Recovery Testing**: Simulate failures and verify recovery
- **Memory Stress Testing**: Process large batches to test memory limits

### Performance Testing

- **Throughput Testing**: Measure addresses processed per minute
- **Memory Usage Testing**: Monitor peak memory consumption
- **Scalability Testing**: Test with datasets of varying sizes
- **Reliability Testing**: Long-running processing validation

## Implementation Plan

### Phase 1: Core Processing Engine (Week 1)
1. Implement CSVProcessor class with batch processing
2. Integrate Shiprocket parser with error handling
3. Create ProcessedAddress data model
4. Implement basic progress tracking

### Phase 2: Output Generation (Week 1)
1. Implement output CSV generation
2. Create statistics calculation engine
3. Add PIN code grouping and analysis
4. Implement comprehensive reporting

### Phase 3: Quality and Performance (Week 2)
1. Add validation and quality metrics
2. Implement memory optimization
3. Add detailed error logging
4. Performance tuning and optimization

### Phase 4: Production Features (Week 2)
1. Add configuration management
2. Implement resume capability for interrupted processing
3. Add detailed progress reporting
4. Create deployment documentation