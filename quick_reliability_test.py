#!/usr/bin/env python3
"""Quick test of Shiprocket reliability fixes."""

import sys
import time
sys.path.insert(0, 'src')

def test_basic_functionality():
    """Test basic Shiprocket functionality."""
    
    print("🔧 Quick Shiprocket Reliability Test")
    print("=" * 40)
    
    try:
        from shiprocket_parser import ShiprocketParser
        
        # Test simple initialization
        print("1. Testing initialization...")
        parser = ShiprocketParser(use_gpu=False, batch_size=1)
        print("   ✅ Parser initialized successfully")
        
        # Test single address
        print("\n2. Testing single address parsing...")
        test_address = "flat 302, friendship residency, pune"
        
        start_time = time.time()
        result = parser.parse_address(test_address)
        end_time = time.time()
        
        print(f"   Address: {test_address}")
        print(f"   Time: {end_time - start_time:.3f}s")
        
        if result.parse_success:
            print("   ✅ Parsing successful!")
            print(f"   Society: '{result.society_name}'")
            print(f"   Unit: '{result.unit_number}'")
            print(f"   Locality: '{result.locality}'")
        else:
            print("   ❌ Parsing failed")
            print(f"   Error: {result.parse_error}")
        
        # Get statistics
        stats = parser.get_statistics()
        print(f"\n3. Statistics:")
        print(f"   Success Rate: {stats['success_rate_percent']}%")
        print(f"   Retries: {stats['total_retries']}")
        
        return result.parse_success
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install dependencies: pip install transformers torch")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    
    if success:
        print("\n🌟 Basic test passed! Reliability fixes are working.")
    else:
        print("\n⚠️  Basic test failed. Check the error above.")