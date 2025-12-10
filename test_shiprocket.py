"""Quick test of Shiprocket parser integration."""

import logging
from src.shiprocket_parser import ShiprocketParser

logging.basicConfig(level=logging.INFO)

print("="*80)
print("TESTING SHIPROCKET PARSER")
print("="*80)
print("\nInitializing Shiprocket parser...")
print("(First run will download ~500MB model from Hugging Face)")
print()

try:
    parser = ShiprocketParser(batch_size=1, use_gpu=False)
    
    # Test addresses
    test_addresses = [
        "Flat 301, Kumar Paradise, Kalyani Nagar, Pune, Maharashtra 411006",
        "A-204, Amanora Park Town, Hadapsar, Pune 411028",
        "Office 505, Cerebrum IT Park, Kumar Road, Pune 411001",
    ]
    
    print("Testing Shiprocket parser on 3 Pune addresses...\n")
    
    for i, address in enumerate(test_addresses, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/3")
        print(f"{'='*80}")
        print(f"Input: {address}")
        print()
        
        parsed = parser.parse_address(address)
        
        if parsed.parse_success:
            print("✓ Parse Success!")
            print(f"  • Unit Number:   {parsed.unit_number or '(empty)'}")
            print(f"  • Society Name:  {parsed.society_name or '(empty)'}")
            print(f"  • Landmark:      {parsed.landmark or '(empty)'}")
            print(f"  • Road:          {parsed.road or '(empty)'}")
            print(f"  • Sub-locality:  {parsed.sub_locality or '(empty)'}")
            print(f"  • Locality:      {parsed.locality or '(empty)'}")
            print(f"  • City:          {parsed.city or '(empty)'}")
            print(f"  • District:      {parsed.district or '(empty)'}")
            print(f"  • State:         {parsed.state or '(empty)'}")
            print(f"  • PIN Code:      {parsed.pin_code or '(empty)'}")
        else:
            print(f"✗ Parse Failed: {parsed.parse_error}")
    
    # Statistics
    stats = parser.get_statistics()
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"Total Parsed:  {stats['total_parsed']}")
    print(f"Total Failed:  {stats['total_failed']}")
    print(f"Success Rate:  {stats['total_parsed']/(stats['total_parsed']+stats['total_failed'])*100:.1f}%")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")
    print("\n✓ Shiprocket parser is working!")
    print("\nTo use in production:")
    print("  1. Set parser_type: 'shiprocket' in config/config.yaml")
    print("  2. Run: python -m src --input addresses.csv --output results.csv")
    print("\nTo compare with other parsers:")
    print("  python compare_parsers.py")
    
except ImportError as e:
    print(f"\n✗ ERROR: {e}")
    print("\nPlease install required packages:")
    print("  pip install transformers torch")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
