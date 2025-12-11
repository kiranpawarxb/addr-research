#!/usr/bin/env python3
"""
Debug Shiprocket NER pipeline issue
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, 'src')

def debug_ner_pipeline():
    """Debug the NER pipeline issue."""
    
    print("🔍 Debugging Shiprocket NER Pipeline")
    print("=" * 50)
    
    try:
        from src.shiprocket_parser import ShiprocketParser
        
        # Initialize parser
        parser = ShiprocketParser(use_gpu=True)
        
        # Test address
        test_address = "ace almighty, indira college road tathwade, wakad"
        
        print(f"📍 Test address: {test_address}")
        
        # Load model manually
        print("🔧 Loading model...")
        parser._load_model()
        
        print(f"✅ Model loaded successfully")
        print(f"   Model: {parser._model}")
        print(f"   Tokenizer: {parser._tokenizer}")
        print(f"   Pipeline: {parser._ner_pipeline}")
        
        # Test NER pipeline directly
        print(f"\n🧪 Testing NER pipeline directly...")
        
        if parser._ner_pipeline is None:
            print("❌ NER pipeline is None!")
            return
        
        # Test pipeline call
        print(f"📞 Calling pipeline with: '{test_address}'")
        
        try:
            entities = parser._ner_pipeline(test_address)
            print(f"✅ Pipeline call successful!")
            print(f"   Raw entities: {entities}")
            print(f"   Type: {type(entities)}")
            
            if isinstance(entities, list):
                print(f"   Count: {len(entities)}")
                for i, entity in enumerate(entities):
                    print(f"   Entity {i}: {entity}")
            
        except Exception as e:
            print(f"❌ Pipeline call failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test safe extraction method
        print(f"\n🛡️ Testing safe extraction method...")
        try:
            safe_entities = parser._safe_ner_extraction(test_address)
            print(f"✅ Safe extraction successful!")
            print(f"   Filtered entities: {safe_entities}")
            
        except Exception as e:
            print(f"❌ Safe extraction failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test full parsing
        print(f"\n🔄 Testing full address parsing...")
        try:
            result = parser.parse_address(test_address)
            print(f"✅ Full parsing successful!")
            print(f"   Parse success: {result.parse_success}")
            print(f"   Society: {result.society_name}")
            print(f"   Locality: {result.locality}")
            print(f"   City: {result.city}")
            print(f"   Error: {result.parse_error}")
            
        except Exception as e:
            print(f"❌ Full parsing failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ner_pipeline()