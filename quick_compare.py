"""Quick comparison of Rule-Based vs IndicBERT parsers.

This script provides a side-by-side comparison of parsing results.
"""

import sys
import logging
from src.local_llm_parser import LocalLLMParser

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(levelname)s - %(message)s'
)

def compare_single_address(address: str):
    """Compare parsers on a single address."""
    
    print(f"\n{'='*80}")
    print(f"INPUT ADDRESS:")
    print(f"  {address}")
    print(f"{'='*80}")
    
    # Rule-based parser
    print(f"\n1. RULE-BASED LOCAL PARSER")
    print(f"{'-'*80}")
    
    try:
        local_parser = LocalLLMParser()
        parsed_local = local_parser.parse_address(address)
        
        if parsed_local.parse_success:
            print(f"  ✓ Success")
            print(f"  • Unit Number:   {parsed_local.unit_number or '(empty)'}")
            print(f"  • Society Name:  {parsed_local.society_name or '(empty)'}")
            print(f"  • Landmark:      {parsed_local.landmark or '(empty)'}")
            print(f"  • Road:          {parsed_local.road or '(empty)'}")
            print(f"  • Sub-locality:  {parsed_local.sub_locality or '(empty)'}")
            print(f"  • Locality:      {parsed_local.locality or '(empty)'}")
            print(f"  • City:          {parsed_local.city or '(empty)'}")
            print(f"  • District:      {parsed_local.district or '(empty)'}")
            print(f"  • State:         {parsed_local.state or '(empty)'}")
            print(f"  • PIN Code:      {parsed_local.pin_code or '(empty)'}")
        else:
            print(f"  ✗ Failed: {parsed_local.parse_error}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # IndicBERT parser
    print(f"\n2. INDICBERT PARSER")
    print(f"{'-'*80}")
    
    try:
        from src.indicbert_parser import IndicBERTParser
        
        print("  Loading IndicBERT model (first run may take a few minutes)...")
        indicbert_parser = IndicBERTParser()
        parsed_indicbert = indicbert_parser.parse_address(address)
        
        if parsed_indicbert.parse_success:
            print(f"  ✓ Success")
            print(f"  • Unit Number:   {parsed_indicbert.unit_number or '(empty)'}")
            print(f"  • Society Name:  {parsed_indicbert.society_name or '(empty)'}")
            print(f"  • Landmark:      {parsed_indicbert.landmark or '(empty)'}")
            print(f"  • Road:          {parsed_indicbert.road or '(empty)'}")
            print(f"  • Sub-locality:  {parsed_indicbert.sub_locality or '(empty)'}")
            print(f"  • Locality:      {parsed_indicbert.locality or '(empty)'}")
            print(f"  • City:          {parsed_indicbert.city or '(empty)'}")
            print(f"  • District:      {parsed_indicbert.district or '(empty)'}")
            print(f"  • State:         {parsed_indicbert.state or '(empty)'}")
            print(f"  • PIN Code:      {parsed_indicbert.pin_code or '(empty)'}")
        else:
            print(f"  ✗ Failed: {parsed_indicbert.parse_error}")
            
        # Comparison
        print(f"\n3. COMPARISON")
        print(f"{'-'*80}")
        
        fields = [
            ('Unit Number', 'unit_number'),
            ('Society Name', 'society_name'),
            ('Landmark', 'landmark'),
            ('Road', 'road'),
            ('Sub-locality', 'sub_locality'),
            ('Locality', 'locality'),
            ('City', 'city'),
            ('District', 'district'),
            ('State', 'state'),
            ('PIN Code', 'pin_code'),
        ]
        
        for field_name, field_attr in fields:
            local_val = getattr(parsed_local, field_attr) or ''
            indicbert_val = getattr(parsed_indicbert, field_attr) or ''
            
            if local_val == indicbert_val:
                status = "✓ SAME"
            else:
                status = "✗ DIFFERENT"
            
            print(f"  {status:<15} {field_name}")
            if local_val != indicbert_val:
                print(f"    Rule-based: '{local_val}'")
                print(f"    IndicBERT:  '{indicbert_val}'")
        
    except ImportError:
        print("  ⚠ IndicBERT parser not available")
        print("  Install with: pip install transformers torch")
    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """Main function."""
    
    print("="*80)
    print("QUICK PARSER COMPARISON")
    print("="*80)
    print("\nComparing Rule-Based Local Parser vs IndicBERT Parser")
    
    # Test addresses
    test_addresses = [
        "Flat 301, Kumar Paradise, Kalyani Nagar, Pune, Maharashtra 411006",
        "A-204, Amanora Park Town, Hadapsar, Pune 411028",
        "Office 505, Cerebrum IT Park, Kumar Road, Pune, Maharashtra 411001",
    ]
    
    for address in test_addresses:
        compare_single_address(address)
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nFor full comparison with metrics, run: python compare_parsers.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
        sys.exit(0)
