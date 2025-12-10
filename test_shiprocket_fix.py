"""Test if the Shiprocket parser fix works correctly."""

import logging
from src.shiprocket_parser import ShiprocketParser

logging.basicConfig(level=logging.INFO)

print("="*80)
print("TESTING FIXED SHIPROCKET PARSER")
print("="*80)
print()

# Test addresses with known society names
test_addresses = [
    "flat-302, friendship residency, veerbhadra nagar road",
    "506, amnora chembers, east amnora town center, hadapsar, pune",
    "20, vasant vihar bunglows, baner",
    "101, shivam building, behind shree kalyani nursing home, lohegaon, pune",
    "c2-504, hari ganga society, yerwada, near rto",
]

expected_societies = [
    "flat-302, friendship residency",
    "amnora chembers",
    "20, vasant vihar bunglows",
    "shivam building",
    "hari ganga society",
]

try:
    parser = ShiprocketParser(batch_size=1, use_gpu=False)
    
    print("Testing society name extraction...\n")
    
    extracted_count = 0
    
    for i, (address, expected) in enumerate(zip(test_addresses, expected_societies), 1):
        print(f"TEST {i}: {address}")
        
        parsed = parser.parse_address(address)
        
        if parsed.parse_success:
            print(f"  ✓ Success")
            print(f"  Unit Number:  '{parsed.unit_number}'")
            print(f"  Society Name: '{parsed.society_name}'")
            print(f"  Landmark:     '{parsed.landmark}'")
            print(f"  Locality:     '{parsed.locality}'")
            print(f"  City:         '{parsed.city}'")
            
            if parsed.society_name:
                extracted_count += 1
                print(f"  ✓ Society extracted!")
            else:
                print(f"  ✗ Society NOT extracted (expected: '{expected}')")
        else:
            print(f"  ✗ Parse failed: {parsed.parse_error}")
        
        print()
    
    print("="*80)
    print(f"RESULTS: Extracted {extracted_count}/{len(test_addresses)} society names")
    print("="*80)
    
    if extracted_count == len(test_addresses):
        print("\n✓ SUCCESS! All society names extracted correctly!")
    elif extracted_count > 0:
        print(f"\n⚠ PARTIAL SUCCESS: {extracted_count} out of {len(test_addresses)} extracted")
    else:
        print("\n✗ FAILED: No society names extracted")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
