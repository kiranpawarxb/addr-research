# Configuration System

The Address Consolidation System uses a YAML-based configuration system with support for environment variable substitution and comprehensive validation.

## Features

- **YAML-based configuration**: Easy to read and edit configuration files
- **Environment variable substitution**: Use `${VAR_NAME}` syntax to inject environment variables
- **Comprehensive validation**: All configuration values are validated with helpful error messages
- **Default values**: Sensible defaults for all optional configuration fields
- **Type coercion**: Automatic conversion of string values to appropriate types

## Configuration Structure

The configuration file is organized into five main sections:

### 1. Input Configuration

```yaml
input:
  file_path: "export_customer_address_store_p0.csv"
  required_columns:
    - addr_text
    - pincode
    - city_id
```

- `file_path` (required): Path to the input CSV file
- `required_columns` (optional): List of required column names (default: `["addr_text", "pincode", "city_id"]`)

### 2. LLM Configuration

```yaml
llm:
  api_endpoint: "https://api.openai.com/v1/chat/completions"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
  max_retries: 3
  timeout_seconds: 30
  batch_size: 10
```

- `api_endpoint` (optional): LLM API endpoint URL (default: OpenAI endpoint)
- `api_key` (optional): API key for authentication (supports environment variables)
- `model` (optional): LLM model name (default: "gpt-4")
- `max_retries` (optional): Number of retry attempts for failed API calls (default: 3, must be ≥ 0)
- `timeout_seconds` (optional): Request timeout in seconds (default: 30, must be > 0)
- `batch_size` (optional): Number of addresses to process in parallel (default: 10, must be > 0)

### 3. Consolidation Configuration

```yaml
consolidation:
  fuzzy_matching: true
  similarity_threshold: 0.85
  normalize_society_names: true
```

- `fuzzy_matching` (optional): Enable fuzzy string matching for society names (default: true)
- `similarity_threshold` (optional): Minimum similarity score for fuzzy matching (default: 0.85, range: 0.0-1.0)
- `normalize_society_names` (optional): Normalize society names before matching (default: true)

### 4. Output Configuration

```yaml
output:
  file_path: "consolidated_addresses.csv"
  include_statistics: true
```

- `file_path` (optional): Path to the output CSV file (default: "consolidated_addresses.csv")
- `include_statistics` (optional): Display statistics after processing (default: true)

### 5. Logging Configuration

```yaml
logging:
  level: "INFO"
  file_path: "consolidation.log"
```

- `level` (optional): Logging level (default: "INFO", valid: DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `file_path` (optional): Path to the log file (default: "consolidation.log")

## Environment Variable Substitution

You can use environment variables in any string value using the `${VAR_NAME}` syntax:

```yaml
llm:
  api_key: "${OPENAI_API_KEY}"

input:
  file_path: "${DATA_DIR}/addresses.csv"
```

If an environment variable is not set, the placeholder will be kept as-is in the configuration.

## Usage

### Loading Configuration

```python
from src.config_loader import load_config, ConfigurationError

try:
    config = load_config('config/config.yaml')
    
    # Access configuration values
    print(f"Input file: {config.input.file_path}")
    print(f"LLM model: {config.llm.model}")
    print(f"Fuzzy matching: {config.consolidation.fuzzy_matching}")
    
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Configuration Objects

The `load_config()` function returns a `Config` object with the following structure:

```python
@dataclass
class Config:
    input: InputConfig
    llm: LLMConfig
    consolidation: ConsolidationConfig
    output: OutputConfig
    logging: LoggingConfig
```

Each section has its own dataclass with typed fields for easy access and IDE autocomplete support.

## Error Handling

The configuration loader provides detailed error messages for common issues:

- **Missing file**: "Configuration file not found: {path}"
- **Invalid YAML**: "Invalid YAML in configuration file: {error}"
- **Missing required fields**: "input.file_path is required"
- **Invalid values**: "llm.max_retries must be non-negative"
- **Invalid types**: "consolidation.fuzzy_matching must be a boolean"

## Validation Rules

The configuration loader validates all values according to these rules:

- `input.file_path`: Must be provided (string)
- `input.required_columns`: Must be a list of strings
- `llm.max_retries`: Must be a non-negative integer
- `llm.timeout_seconds`: Must be a positive integer
- `llm.batch_size`: Must be a positive integer
- `consolidation.similarity_threshold`: Must be a float between 0.0 and 1.0
- `consolidation.fuzzy_matching`: Must be a boolean
- `consolidation.normalize_society_names`: Must be a boolean
- `output.include_statistics`: Must be a boolean
- `logging.level`: Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive)

## Example

See `examples/config_usage.py` for a complete example of loading and using configuration.

Run the example:

```bash
python -m examples.config_usage
```

## Testing

The configuration system includes comprehensive tests covering:

- Valid configuration loading
- Environment variable substitution
- Default value handling
- Validation of all fields
- Error handling for invalid configurations
- Type coercion

Run the tests:

```bash
python -m pytest tests/test_config_loader.py -v
```
