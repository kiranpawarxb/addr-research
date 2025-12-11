#!/usr/bin/env python3
"""
Test the fixed batch processing
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_fixed_batch():
    """Test the fixed batch processing."""
    
    print("🧪 Testing Fixed Batch Processing")
    print("=" * 50)
    
    try:
        from src.shiprocket_parser import ShiprocketParser
        
        # Initialize parser
        parser = ShiprocketParser(use_gpu=True, batch_size=5)
        
        # Test addresses
        test_addresses = [
            "ace almighty, indira college road tathwade, wakad",
            "flat-302, friendship residency, veerbhadra nagar road",
            "802 marvel exotica lane 7 koregaon park pune",
            "124/1/8, sadguru housing society, sadgurunagar, pune",
            "506, amnora chembers, east amnora town center, amnora"
        ]
        
        print(f"📍 Testing {len(test_addresses)} addresses")
        
        # Test batch processing
        print(f"\n🔄 Running batch processing...")
        results = parser.parse_batch(test_addresses)
        
        print(f"\n📊 Results:")
        print(f"   Total processed: {len(results)}")
        
        success_count = sum(1 for r in results if r.parse_success)
        print(f"   Successful: {success_count}/{len(results)}")
        
        print(f"\n📋 Detailed Results:")
        for i, (addr, result) in enumerate(zip(test_addresses, results)):
            print(f"\n{i+1}. {addr[:50]}...")
            print(f"   ✅ Success: {result.parse_success}")
            if result.parse_success:
                print(f"   🏢 Society: '{result.society_name}'")
                print(f"   🏘️ Locality: '{result.locality}'")
                print(f"   🏙️ City: '{result.city}'")
                print(f"   📮 PIN: '{result.pin_code}'")
            else:
                print(f"   ❌ Error: {result.parse_error}")
        
        # Test statistics
        stats = parser.get_statistics()
        print(f"\n📊 Parser Statistics:")
        print(f"   Total parsed: {stats['total_parsed']}")
        print(f"   Total failed: {stats['total_failed']}")
        print(f"   Success rate: {stats['success_rate_percent']}%")
        
        if success_count > 0:
            print(f"\n✅ Batch processing is working correctly!")
            return True
        else:
            print(f"\n❌ Batch processing failed - no successful parses")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_batch()
    if success:
        print(f"\n🚀 Ready to run full dataset processing!")
    else:
        print(f"\n🛠️ Need to fix issues before full processing")