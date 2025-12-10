# Shiprocket Parser Integration Guide

## Current Status

I've created a **placeholder implementation** for the Shiprocket parser at `src/shiprocket_parser.py`. To complete the integration, I need more information about the Shiprocket address parser.

## What I Need From You

Please provide **ONE** of the following:

### Option 1: Python Package

If Shiprocket has a Python package:

```bash
# Package name
pip install <package-name>

# Example:
pip install shiprocket-address-parser
```

**Please provide:**
- Exact package name
- Import statement (e.g., `from shiprocket import AddressParser`)
- Basic usage example

### Option 2: API Access

If Shiprocket provides an API:

**Please provide:**
- API endpoint URL
- Authentication method (API key, OAuth, etc.)
- Request/response format
- Rate limits
- Documentation link

**Example:**
```python
import requests

response = requests.post(
    'https://api.shiprocket.in/v1/parse-address',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={'address': 'Flat 301, Kumar Paradise, Pune 411006'}
)
```

### Option 3: GitHub Repository

If it's open source:

**Please provide:**
- GitHub repository URL
- Installation instructions
- Usage example

### Option 4: Custom Solution

If you have custom code or scripts:

**Please provide:**
- The code/scripts
- Dependencies
- Usage instructions

## Alternative: Libpostal Parser

While waiting for Shiprocket details, I've also integrated **libpostal**, which is an excellent open-source address parser that works well for Indian addresses.

### Libpostal Setup

```bash
# 1. Install libpostal C library (one-time setup)
# Ubuntu/Debian:
sudo apt-get install curl autoconf automake libtool pkg-config
git clone https://github.com/openvenues/libpostal
cd libpostal
./bootstrap.sh
./configure --datadir=/usr/local/share/libpostal
make -j4
sudo make install
sudo ldconfig

# 2. Install Python bindings
pip install postal
```

### Use Libpostal

```yaml
# config.yaml
llm:
  parser_type: "libpostal"
```

## Current Parser Options

You currently have **4 parser options** (3 working + 1 placeholder):

| Parser | Status | Speed | Accuracy | Setup |
|--------|--------|-------|----------|-------|
| Rule-Based | ✅ Working | ⚡ Very Fast | ✅ Excellent | ✅ Instant |
| IndicBERT | ✅ Working | 🐌 Slower | ✅ Excellent | ⏳ 500MB download |
| Libpostal | ✅ Working | ⚡ Fast | ✅ Excellent | ⏳ C library install |
| Shiprocket | ⏳ Pending | ? | ? | ❓ Need details |
| OpenAI | ✅ Working | 🐌 Slowest | ✅ Excellent | ✅ API key |

## Integration Template

Once you provide the details, I'll use this template to integrate:

```python
# src/shiprocket_parser.py

from shiprocket_package import ShiprocketAPI  # Your package here
from src.models import ParsedAddress

class ShiprocketParser:
    def __init__(self, api_key: str = None, batch_size: int = 10):
        self.client = ShiprocketAPI(api_key=api_key)
        self.batch_size = batch_size
        self._total_parsed = 0
        self._total_failed = 0
    
    def parse_address(self, raw_address: str) -> ParsedAddress:
        try:
            # Call Shiprocket API/library
            result = self.client.parse(raw_address)
            
            # Map to our ParsedAddress format
            parsed = ParsedAddress(
                unit_number=result.get('unit_number', ''),
                society_name=result.get('society_name', ''),
                landmark=result.get('landmark', ''),
                road=result.get('road', ''),
                sub_locality=result.get('sub_locality', ''),
                locality=result.get('locality', ''),
                city=result.get('city', ''),
                district=result.get('district', ''),
                state=result.get('state', ''),
                country=result.get('country', 'India'),
                pin_code=result.get('pin_code', ''),
                note="Parsed using Shiprocket parser",
                parse_success=True,
                parse_error=None
            )
            
            self._total_parsed += 1
            return parsed
            
        except Exception as e:
            self._total_failed += 1
            return ParsedAddress(
                parse_success=False,
                parse_error=f"Shiprocket error: {str(e)}"
            )
    
    def parse_batch(self, addresses: List[str]) -> List[ParsedAddress]:
        return [self.parse_address(addr) for addr in addresses]
    
    def get_statistics(self) -> Dict[str, int]:
        return {
            "total_parsed": self._total_parsed,
            "total_failed": self._total_failed,
            "total_retries": 0
        }
```

## Next Steps

1. **Provide Shiprocket details** (package name, API, or GitHub repo)
2. I'll complete the integration in `src/shiprocket_parser.py`
3. Add to `config.yaml` as `parser_type: "shiprocket"`
4. Update comparison tools to include Shiprocket
5. Run comparison: `python compare_parsers.py`

## Questions to Answer

To help me integrate quickly, please answer:

1. **What is the Shiprocket parser?**
   - Python package?
   - REST API?
   - Open source library?
   - Custom solution?

2. **How do you access it?**
   - Package name for pip install?
   - API endpoint and authentication?
   - GitHub repository?

3. **What's the input/output format?**
   - How do you send an address?
   - What format does it return?

4. **Any costs or limits?**
   - Free or paid?
   - Rate limits?
   - API quotas?

## Contact

Once you provide the details, I can complete the integration in minutes and add Shiprocket to the comparison!

**Current working parsers you can use right now:**
- ✅ Rule-Based Local (fastest, recommended)
- ✅ IndicBERT (ML-based)
- ✅ Libpostal (statistical NLP)
- ✅ OpenAI (cloud-based)

Try them with: `python compare_parsers.py`
