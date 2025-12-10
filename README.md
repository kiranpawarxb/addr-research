# Address Consolidation System

A Python-based system that processes CSV files containing Indian address data, uses LLM-based parsing to extract structured address components, and consolidates addresses based on Society Name and geographic identifiers.

## Features

- **LLM-Powered Parsing**: Extracts 12 structured address fields from unstructured Indian address text
- **Smart Consolidation**: Groups addresses by Society Name and PIN code with fuzzy matching support
- **Batch Processing**: Processes multiple addresses in parallel for improved performance
- **Robust Error Handling**: Continues processing even when some addresses fail to parse
- **Comprehensive Statistics**: Provides detailed metrics about consolidation quality
- **Flexible Configuration**: YAML-based configuration with environment variable support
- **Progress Tracking**: Real-time progress bars for long-running operations

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Input/Output Formats](#inputoutput-formats)
- [Command-Line Options](#command-line-options)
- [Examples](#examples)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- OpenAI API key (or compatible LLM API)

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd address-consolidation

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install the Package (Optional)

For easier command-line access:

```bash
pip install -e .
```

This allows you to run `address-consolidation` from anywhere.

### Step 5: Set Up API Key

Set your OpenAI API key as an environment variable:

**Windows:**
```bash
set OPENAI_API_KEY=sk-your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

For permanent setup, add to your shell profile (`.bashrc`, `.zshrc`, etc.):
```bash
echo 'export OPENAI_API_KEY=sk-your-api-key-here' >> ~/.bashrc
source ~/.bashrc
```

## Quick Start

1. **Prepare your input CSV** with required columns: `addr_text`, `pincode`, `city_id`

2. **Update configuration** (optional):
   ```bash
   # Edit config/config.yaml to set input/output paths
   notepad config/config.yaml  # Windows
   nano config/config.yaml     # Linux/Mac
   ```

3. **Run the consolidation**:
   ```bash
   address-consolidation --input your_addresses.csv --output results.csv
   ```

4. **View results**:
   - Consolidated addresses: `results.csv`
   - Processing log: `consolidation.log`
   - Statistics displayed in console

## Configuration

The system uses a YAML configuration file (`config/config.yaml`) with five main sections:

### Input Configuration

```yaml
input:
  file_path: "export_customer_address_store_p0.csv"
  required_columns:
    - addr_text
    - pincode
    - city_id
```

### LLM Configuration

```yaml
llm:
  api_endpoint: "https://api.openai.com/v1/chat/completions"
  api_key: "${OPENAI_API_KEY}"  # Use environment variable
  model: "gpt-4"
  max_retries: 3
  timeout_seconds: 30
  batch_size: 10
```

### Consolidation Configuration

```yaml
consolidation:
  fuzzy_matching: true
  similarity_threshold: 0.85  # 0.0 to 1.0
  normalize_society_names: true
```

### Output Configuration

```yaml
output:
  file_path: "consolidated_addresses.csv"
  include_statistics: true
```

### Logging Configuration

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file_path: "consolidation.log"
```

**For detailed configuration options, see [`docs/configuration.md`](docs/configuration.md)**

## Usage

### Running the Application

After installation, you can run the application in several ways:

**1. Using the installed command (recommended):**
```bash
address-consolidation
```

**2. Using Python module syntax:**
```bash
python -m src
```

**3. Direct script execution:**
```bash
python src/cli.py
```

## Input/Output Formats

### Input CSV Format

The input CSV file must contain at least these required columns:

| Column | Description | Example |
|--------|-------------|---------|
| `addr_text` | Raw address text to parse | "Flat 301, Prestige Lakeside, Varthur Road, Bangalore 560103" |
| `pincode` | PIN code | "560103" |
| `city_id` | City identifier | "BLR" |

**Optional columns** (preserved in output):
- `addr_hash_key`: Unique address identifier
- `state_id`: State identifier
- `zone_id`: Zone identifier
- `address_id`: Address ID from source system
- `assigned_pickup_dlvd_geo_points`: Geographic points
- `assigned_pickup_dlvd_geo_points_count`: Count of geo points

**Example input CSV:**
```csv
addr_hash_key,addr_text,city_id,pincode,state_id,zone_id,address_id
abc123,"Flat 301, Prestige Lakeside, Varthur Road, Bangalore",BLR,560103,KA,SOUTH,addr_001
def456,"#402, Prestige Lake Side, Whitefield, Bangalore",BLR,560103,KA,SOUTH,addr_002
```

### Output CSV Format

The output CSV contains all original columns plus 13 additional columns:

**Parsed Address Fields (12 columns):**

| Column | Abbreviation | Description | Example |
|--------|--------------|-------------|---------|
| `unit_number` | UN | Flat/apartment number | "301" |
| `society_name` | SN | Building/society name | "Prestige Lakeside" |
| `landmark` | LN | Nearby landmark | "Near Forum Mall" |
| `road` | RD | Street/road name | "Varthur Road" |
| `sub_locality` | SL | Sub-area/neighborhood | "Whitefield" |
| `locality` | LOC | Broader locality | "East Bangalore" |
| `city` | CY | City name | "Bangalore" |
| `district` | DIS | District name | "Bangalore Urban" |
| `state` | ST | State name | "Karnataka" |
| `country` | CN | Country name | "India" |
| `pin_code` | PIN | Postal code | "560103" |
| `note` | Note | Additional parsing notes | "Verified address" |

**Consolidation Field (1 column):**

| Column | Description | Example |
|--------|-------------|---------|
| `group_id` | Unique identifier for consolidated group | "a1b2c3d4-e5f6-7890-abcd-ef1234567890" |

**Example output CSV:**
```csv
addr_hash_key,addr_text,city_id,pincode,...,unit_number,society_name,landmark,road,sub_locality,locality,city,district,state,country,pin_code,note,group_id
abc123,"Flat 301, Prestige...",BLR,560103,...,301,Prestige Lakeside,,Varthur Road,Whitefield,,Bangalore,Bangalore Urban,Karnataka,India,560103,,uuid-1234
def456,"#402, Prestige...",BLR,560103,...,402,Prestige Lakeside,,Varthur Road,Whitefield,,Bangalore,Bangalore Urban,Karnataka,India,560103,,uuid-1234
```

**Note:** Addresses with the same `group_id` belong to the same consolidated group (same Society Name and PIN code).

### Statistics Output

After processing, the system displays consolidation statistics:

```
=== Consolidation Statistics ===
Total Records Processed: 1,000
Total Consolidated Groups: 250
Average Records per Group: 4.0
Largest Group: Prestige Lakeside (25 records)
Match Percentage: 95.0%
Parse Success Rate: 98.0% (980/1000)
Parse Failure Rate: 2.0% (20/1000)
Unmatched Records: 50
```

## Command-Line Options

```
usage: address-consolidation [-h] [-c CONFIG] [-i INPUT] [-o OUTPUT] [-v] [-q]
                             [--log-file LOG_FILE] [--no-stats] [--version]

Address Consolidation System - Parse and consolidate Indian addresses using LLM

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to configuration YAML file (default: config/config.yaml)
  -i INPUT, --input INPUT
                        Path to input CSV file (overrides config file setting)
  -o OUTPUT, --output OUTPUT
                        Path to output CSV file (overrides config file setting)
  -v, --verbose         Enable verbose (DEBUG) logging
  -q, --quiet           Suppress progress output (WARNING level only)
  --log-file LOG_FILE   Path to log file (overrides config file setting)
  --no-stats            Disable statistics display at the end
  --version             show program's version number and exit
```

## Examples

### Basic Usage

**Run with default configuration:**
```bash
address-consolidation
```

**Specify input and output files:**
```bash
address-consolidation --input addresses.csv --output results.csv
```

### Custom Configuration

**Use a custom config file:**
```bash
address-consolidation --config production_config.yaml
```

**Override specific settings:**
```bash
address-consolidation --config config.yaml --input data.csv --output out.csv
```

### Logging Options

**Enable debug logging:**
```bash
address-consolidation --verbose --log-file debug.log
```

**Quiet mode (minimal output):**
```bash
address-consolidation --quiet --no-stats
```

**Custom log file:**
```bash
address-consolidation --log-file logs/processing_$(date +%Y%m%d).log
```

### Advanced Usage

**Process large file with custom settings:**
```bash
address-consolidation \
  --input large_dataset.csv \
  --output consolidated_large.csv \
  --verbose \
  --log-file large_processing.log
```

**Batch processing multiple files:**
```bash
# Linux/Mac
for file in data/*.csv; do
  address-consolidation --input "$file" --output "results/$(basename $file)"
done

# Windows PowerShell
Get-ChildItem data\*.csv | ForEach-Object {
  address-consolidation --input $_.FullName --output "results\$($_.Name)"
}
```

## Testing

The project includes comprehensive unit tests and property-based tests.

### Run All Tests

```bash
pytest
```

### Run with Coverage Report

```bash
pytest --cov=src --cov-report=html
```

View coverage report: `htmlcov/index.html`

### Run Specific Test Types

```bash
# Unit tests only
pytest -m unit

# Property-based tests only
pytest -m property

# Integration tests only
pytest -m integration
```

### Run Specific Test Files

```bash
pytest tests/test_csv_reader.py
pytest tests/test_llm_parser.py
pytest tests/test_consolidation_engine.py
```

### Run with Verbose Output

```bash
pytest -v
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Missing Required Columns Error

**Error:**
```
Error: Missing required columns: ['addr_text']
```

**Solution:**
- Verify your CSV file has columns: `addr_text`, `pincode`, `city_id`
- Check for typos in column names (case-sensitive)
- Update `required_columns` in `config/config.yaml` if your CSV uses different names

#### 2. API Authentication Errors

**Error:**
```
Error: Invalid API key or authentication failed
```

**Solution:**
- Verify `OPENAI_API_KEY` environment variable is set correctly
- Check that the API key is valid and has not expired
- Ensure no extra spaces or quotes in the environment variable
- Test the API key with a simple curl command:
  ```bash
  curl https://api.openai.com/v1/models \
    -H "Authorization: Bearer $OPENAI_API_KEY"
  ```

#### 3. Timeout Errors

**Error:**
```
Warning: Request timeout after 30 seconds, retrying...
```

**Solution:**
- Increase `timeout_seconds` in `config/config.yaml` (e.g., 60)
- Reduce `batch_size` to process fewer addresses in parallel
- Check your internet connection
- Verify the LLM API endpoint is accessible

#### 4. Rate Limit Errors

**Error:**
```
Error: Rate limit exceeded
```

**Solution:**
- Reduce `batch_size` in `config/config.yaml` (e.g., from 10 to 5)
- Add delays between batches (contact support for custom implementation)
- Upgrade your API plan for higher rate limits
- Use a different model (e.g., `gpt-3.5-turbo` instead of `gpt-4`)

#### 5. Too Many/Too Few Consolidated Groups

**Issue:** Addresses that should be grouped together are in separate groups, or vice versa.

**Solution:**
- **Too many groups** (under-consolidation):
  - Enable `fuzzy_matching: true` in config
  - Lower `similarity_threshold` (e.g., from 0.85 to 0.75)
  - Ensure `normalize_society_names: true`
- **Too few groups** (over-consolidation):
  - Increase `similarity_threshold` (e.g., from 0.85 to 0.95)
  - Disable fuzzy matching: `fuzzy_matching: false`

#### 6. File Not Found Error

**Error:**
```
Error: Configuration file not found: config/config.yaml
```

**Solution:**
- Verify the file path is correct
- Use absolute paths if relative paths don't work
- Check current working directory: `pwd` (Linux/Mac) or `cd` (Windows)
- Specify config file explicitly: `--config /full/path/to/config.yaml`

#### 7. Memory Issues with Large Files

**Issue:** System runs out of memory when processing large CSV files.

**Solution:**
- The system uses streaming to handle large files efficiently
- If still experiencing issues:
  - Split large CSV into smaller chunks
  - Reduce `batch_size` to lower memory usage
  - Close other applications to free up memory
  - Process on a machine with more RAM

#### 8. Invalid JSON Response from LLM

**Error:**
```
Warning: Invalid JSON response from LLM, marking address as unparseable
```

**Solution:**
- This is expected for some addresses and handled gracefully
- Check the log file for details about which addresses failed
- If many addresses fail:
  - Try a different model (e.g., `gpt-4` instead of `gpt-3.5-turbo`)
  - Verify the address text is in a reasonable format
  - Check if the LLM API is functioning correctly

#### 9. Permission Denied When Writing Output

**Error:**
```
Error: Permission denied: consolidated_addresses.csv
```

**Solution:**
- Check file permissions on the output directory
- Ensure the output file is not open in another program (Excel, etc.)
- Try writing to a different directory
- Run with appropriate permissions (avoid running as admin unless necessary)

#### 10. Slow Processing Speed

**Issue:** Processing takes longer than expected.

**Solution:**
- Increase `batch_size` for more parallel processing (watch for rate limits)
- Use a faster model: `gpt-3.5-turbo` instead of `gpt-4`
- Check network latency to the LLM API
- Verify the LLM API is not experiencing outages
- Consider using a local LLM for faster processing (requires setup)

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Review `consolidation.log` for detailed error messages
2. **Enable debug logging**: Run with `--verbose` flag for more information
3. **Review configuration**: Verify all settings in `config/config.yaml`
4. **Test with small dataset**: Try processing a small CSV file first
5. **Check API status**: Verify the LLM API service is operational

## Project Structure

```
.
├── config/                      # Configuration files
│   └── config.yaml              # Main configuration file
├── docs/                        # Documentation
│   └── configuration.md         # Detailed configuration guide
├── examples/                    # Example scripts
│   └── config_usage.py          # Configuration usage example
├── src/                         # Source code
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Entry point for python -m src
│   ├── cli.py                   # Command-line interface
│   ├── config_loader.py         # Configuration loading and validation
│   ├── consolidation_engine.py  # Address consolidation logic
│   ├── csv_reader.py            # CSV file reading
│   ├── llm_parser.py            # LLM-based address parsing
│   ├── models.py                # Data models
│   ├── output_writer.py         # CSV output writing
│   ├── pipeline.py              # Main pipeline orchestrator
│   └── statistics_reporter.py   # Statistics calculation and display
├── tests/                       # Test suite
│   ├── __init__.py              # Test package initialization
│   ├── test_cli.py              # CLI tests
│   ├── test_config_loader.py    # Configuration tests
│   ├── test_consolidation_engine.py  # Consolidation tests
│   ├── test_csv_reader.py       # CSV reader tests
│   ├── test_llm_parser.py       # LLM parser tests
│   ├── test_models.py           # Data model tests
│   ├── test_output_writer.py    # Output writer tests
│   ├── test_pipeline.py         # Pipeline tests
│   └── test_statistics_reporter.py  # Statistics tests
├── .gitignore                   # Git ignore rules
├── pytest.ini                   # Pytest configuration
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup configuration
```

## Dependencies

### Core Dependencies

- **pandas** (≥1.5.0): CSV reading and writing, data manipulation
- **requests** (≥2.28.0): HTTP client for LLM API calls
- **pyyaml** (≥6.0): YAML configuration file parsing
- **rapidfuzz** (≥2.13.0): Fast fuzzy string matching for society names
- **tqdm** (≥4.64.0): Progress bar display for long operations

### Testing Dependencies

- **pytest** (≥7.2.0): Testing framework
- **hypothesis** (≥6.68.0): Property-based testing
- **pytest-cov** (≥4.0.0): Code coverage reporting

### Installation

All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd address-consolidation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .

# Run tests
pytest
```

### Running Tests During Development

```bash
# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run tests in watch mode (requires pytest-watch)
pip install pytest-watch
ptw

# Run specific test
pytest tests/test_csv_reader.py::test_read_valid_csv -v
```

### Code Quality

This project follows Python best practices:

- **Type hints**: All functions include type annotations
- **Docstrings**: Comprehensive documentation for all modules and functions
- **Testing**: Both unit tests and property-based tests for comprehensive coverage
- **Error handling**: Graceful error handling with informative messages
- **Logging**: Structured logging for debugging and monitoring

## License

TBD

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For questions, issues, or feature requests, please open an issue on the project repository.
