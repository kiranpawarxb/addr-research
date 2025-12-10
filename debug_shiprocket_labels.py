"""Debug script to see what entity labels Shiprocket model actually outputs."""

import logging
from src.shiprocket_parser import ShiprocketParser

logging.basicConfig(level=logging.INFO)

print("="*80)
print("DEBUGGING SHIPROCKET ENTITY LABELS")
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

try:
    parser = ShiprocketParser(batch_size=1, use_gpu=False)
    
    # Load model
    parser._load_model()
    
    print("Model loaded successfully!\n")
    
    for i, address in enumerate(test_addresses, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {address}")
        print(f"{'='*80}")
        
        try:
            # Get raw NER entities
            entities = parser._ner_pipeline(address)
            
            print(f"\nRaw NER Entities ({len(entities)} found):")
            for entity in entities:
                entity_group = entity.get('entity_group', 'UNKNOWN')
                word = entity.get('word', '')
                score = entity.get('score', 0)
                print(f"  - {entity_group:<20} | {word:<30} | confidence: {score:.3f}")
            
            # Now parse normally
            parsed = parser.parse_address(address)
            
            print(f"\nParsed Result:")
            print(f"  Success: {parsed.parse_success}")
            print(f"  Society: '{parsed.society_name}'")
            print(f"  Unit: '{parsed.unit_number}'")
            print(f"  Road: '{parsed.road}'")
            print(f"  Locality: '{parsed.locality}'")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("SUMMARY OF ENTITY LABELS FOUND")
    print(f"{'='*80}")
    
    # Collect all unique entity labels
    all_labels = set()
    for address in test_addresses:
        try:
            entities = parser._ner_pipeline(address)
            for entity in entities:
                all_labels.add(entity.get('entity_group', 'UNKNOWN'))
        except:
            pass
    
    print("\nUnique entity labels used by Shiprocket model:")
    for label in sorted(all_labels):
        print(f"  - {label}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
except ImportError as e:
    print(f"\nERROR: {e}")
    print("Please install: pip install transformers torch")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
