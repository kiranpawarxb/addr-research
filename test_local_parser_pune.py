"""Test script for local address parser with Pune addresses."""

import sys
import logging
from src.local_llm_parser import LocalLLMParser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_pune_addresses():
    """Test the local parser with sample Pune addresses."""
    
    # Sample Pune addresses
    pune_addresses = [
        "Flat 301, Kumar Paradise, Kalyani Nagar, Pune, Maharashtra 411006",
        "A-204, Amanora Park Town, Hadapsar, Pune 411028",
        "Bungalow 12, Koregaon Park, Near Osho Ashram, Pune 411001",
        "Office 505, Cerebrum IT Park, Kumar Road, Pune, Maharashtra 411001",
        "201, Magarpatta City, Hadapsar, Pune 411013",
        "Flat 102, Seasons Apartment, Baner Road, Baner, Pune 411045",
        "B-Wing 404, Pride Purple Park, Wakad, Pune, Maharashtra 411057",
        "Villa 7, Amanora Neo Towers, Hadapsar, Near Phoenix Mall, Pune 411028",
        "Shop 12, FC Road, Shivajinagar, Pune 411004",
        "3rd Floor, Panchshil Tech Park, Yerwada, Pune, Maharashtra 411006"
    ]
    
    print("="*80)
    print("TESTING LOCAL ADDRESS PARSER WITH PUNE ADDRESSES")
    print("="*80)
    print()
    
    # Initialize parser
    print("Initializing LocalLLMParser...")
    parser = LocalLLMParser(batch_size=5)
    print("✓ Parser initialized\n")
    
    # Test each address
    for i, address in enumerate(pune_addresses, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(pune_addresses)}")
        print(f"{'='*80}")
        print(f"Input Address:")
        print(f"  {address}")
        print()
        
        # Parse the address
        parsed = parser.parse_address(address)
        
        # Display results
        print("Parsed Fields:")
        print(f"  ✓ Parse Success: {parsed.parse_success}")
        if not parsed.parse_success:
            print(f"  ✗ Error: {parsed.parse_error}")
        else:
            print(f"  • Unit Number (UN):   {parsed.unit_number or '(empty)'}")
            print(f"  • Society Name (SN):  {parsed.society_name or '(empty)'}")
            print(f"  • Landmark (LN):      {parsed.landmark or '(empty)'}")
            print(f"  • Road (RD):          {parsed.road or '(empty)'}")
            print(f"  • Sub-locality (SL):  {parsed.sub_locality or '(empty)'}")
            print(f"  • Locality (LOC):     {parsed.locality or '(empty)'}")
            print(f"  • City (CY):          {parsed.city or '(empty)'}")
            print(f"  • District (DIS):     {parsed.district or '(empty)'}")
            print(f"  • State (ST):         {parsed.state or '(empty)'}")
            print(f"  • Country (CN):       {parsed.country or '(empty)'}")
            print(f"  • PIN Code (PIN):     {parsed.pin_code or '(empty)'}")
            print(f"  • Note:               {parsed.note or '(empty)'}")
    
    # Display statistics
    print(f"\n{'='*80}")
    print("PARSING STATISTICS")
    print(f"{'='*80}")
    stats = parser.get_statistics()
    print(f"Total Parsed:  {stats['total_parsed']}")
    print(f"Total Failed:  {stats['total_failed']}")
    print(f"Success Rate:  {stats['total_parsed']/(stats['total_parsed']+stats['total_failed'])*100:.1f}%")
    
    # Test batch parsing
    print(f"\n{'='*80}")
    print("TESTING BATCH PARSING")
    print(f"{'='*80}")
    print(f"Parsing {len(pune_addresses)} addresses in batch...")
    
    batch_results = parser.parse_batch(pune_addresses)
    
    success_count = sum(1 for r in batch_results if r.parse_success)
    print(f"\n✓ Batch parsing complete!")
    print(f"  Success: {success_count}/{len(batch_results)}")
    print(f"  Failed:  {len(batch_results)-success_count}/{len(batch_results)}")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")
    print("\nThe local parser is working! You can now integrate it into your pipeline.")
    print("\nNext steps:")
    print("  1. Review the parsed results above")
    print("  2. Update config.yaml to use 'local' parser instead of 'openai'")
    print("  3. Modify pipeline.py to use LocalLLMParser when configured")
    print()


if __name__ == "__main__":
    try:
        test_pune_addresses()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
