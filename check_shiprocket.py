"""Check for Shiprocket address parser availability."""

import subprocess
import sys

print("Checking for Shiprocket address parser...")
print("="*80)

# Common package names to check
packages_to_check = [
    "shiprocket",
    "shiprocket-address-parser",
    "indian-address-parser",
    "address-parser-india",
    "pyaddress",
    "libpostal",
]

print("\nSearching for Indian address parser packages...\n")

for package in packages_to_check:
    try:
        result = subprocess.run(
            ["pip", "show", package],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ Found: {package}")
            print(result.stdout[:200])
            print()
        else:
            print(f"✗ Not installed: {package}")
    except Exception as e:
        print(f"✗ Error checking {package}: {e}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("""
If you have a specific Shiprocket parser in mind, please provide:
1. Package name (e.g., 'pip install shiprocket-parser')
2. GitHub repository URL
3. Installation instructions

Common Indian address parsers:
- libpostal: Universal address parser (C library with Python bindings)
- pyaddress: Python address parsing library
- Custom Shiprocket solution (may need API access)

I can integrate any of these once you provide the details!
""")
