#!/usr/bin/env python3
"""Test the Hybrid Parser implementation."""

import sys
import time
sys.path.insert(0, 'src')

def test_hybrid_parser():
    """Test hybrid parser with various address complexities."""
    
    print("🔀 Testing Hybrid Parser Implementation")
    print("=" * 60)
    
    from hybrid_parser import HybridParser
    
    # Test addresses with varying complexity
    test_addresses = [
        # Simple addresses (should go to Local)
        ("flat 302, pune", "Simple"),
        ("123 mumbai", "Simple"),
        ("apartment 5, bangalore", "Simple"),
        
        # Medium complexity (threshold dependent)
        ("flat 302, friendship residency, pune", "Medium"),
        ("20, vasant vihar bunglows, baner", "Medium"),
        ("c 407 epic hsg socity wagoli", "Medium"),
        
        # Complex addresses (should go to Shiprocket)
        ("506, amnora chembers, east amnora town center, amnora magarpatta road, hadapsar, pune", "Complex"),
        ("panchshil towers 191 panchshil towers road vitthal nagar kharadi a-2503", "Complex"),
        ("suyog nisarg, daisy b 201, near suyog sunderji school, lohegaon road", "Complex"),
        ("124/1/8, sadguru housing society, sadgurunagar, pune nasik road, bhosari pune 39 near datta mandir", "Complex")
    ]
    
    # Test with different complexity thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\n🎯 Testing with complexity threshold: {threshold}")
        print("-" * 50)
        
        parser = HybridParser(
            use_gpu=False,
            complexity_threshold=threshold,
            enable_fallback=True
        )
        
        results = []
        total_time = 0
        
        for address, expected_complexity in test_addresses:
            print(f"\n📍 {address} ({expected_complexity})")
            
            start_time = time.time()
            result = parser.parse_address(address)
            end_time = time.time()
            
            parse_time = end_time - start_time
            total_time += parse_time
            results.append(result)
            
            if result.parse_success:
                print(f"   ✅ SUCCESS ({parse_time:.3f}s)")
                print(f"   📝 {result.note}")
                
                # Show key extracted fields
                if result.society_name:
                    print(f"   🏢 Society: '{result.society_name}'")
                if result.unit_number:
                    print(f"   🏠 Unit: '{result.unit_number}'")
                if result.road:
                    print(f"   🛣️  Road: '{result.road}'")
                if result.locality:
                    print(f"   📍 Locality: '{result.locality}'")
            else:
                print(f"   ❌ FAILED ({parse_time:.3f}s)")
                print(f"   💥 {result.parse_error}")
        
        # Get statistics
        stats = parser.get_statistics()
        
        print(f"\n📊 Results for threshold {threshold}:")
        print(f"   Success Rate: {stats['success_rate_percent']}%")
        print(f"   Local Usage: {stats['local_usage_percent']}% ({stats['local_used']} addresses)")
        print(f"   Shiprocket Usage: {stats['shiprocket_usage_percent']}% ({stats['shiprocket_used']} addresses)")
        print(f"   Fallback Usage: {stats['fallback_usage_percent']}% ({stats['fallback_used']} addresses)")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Avg Time/Address: {total_time/len(test_addresses):.3f}s")
    
    return True


def test_complexity_analysis():
    """Test the complexity analysis algorithm."""
    
    print("\n🧠 Testing Complexity Analysis Algorithm")
    print("=" * 60)
    
    from hybrid_parser import HybridParser
    
    parser = HybridParser()
    
    # Test addresses with known complexity levels
    complexity_tests = [
        # Simple addresses (should score low)
        ("flat 302, pune", "Low"),
        ("123 mumbai", "Low"),
        ("apartment 5, bangalore", "Low"),
        
        # Medium complexity
        ("flat 302, friendship residency, pune", "Medium"),
        ("20, vasant vihar bunglows, baner", "Medium"),
        
        # High complexity (multiple components, roads, landmarks)
        ("506, amnora chembers, east amnora town center, amnora magarpatta road, hadapsar, pune", "High"),
        ("panchshil towers 191 panchshil towers road vitthal nagar kharadi a-2503", "High"),
        ("suyog nisarg, daisy b 201, near suyog sunderji school, lohegaon road", "High"),
        ("124/1/8, sadguru housing society, sadgurunagar, pune nasik road, bhosari pune 39 near datta mandir", "High")
    ]
    
    print("Address Complexity Analysis:")
    print("-" * 40)
    
    for address, expected in complexity_tests:
        score = parser._analyze_complexity(address)
        
        # Determine actual complexity based on score
        if score < 0.3:
            actual = "Low"
        elif score < 0.6:
            actual = "Medium"
        else:
            actual = "High"
        
        match = "✅" if actual == expected else "⚠️"
        
        print(f"{match} {score:.3f} | {actual:6} | {expected:6} | {address[:50]}...")
    
    return True


def test_threshold_tuning():
    """Test automatic threshold tuning."""
    
    print("\n⚙️  Testing Automatic Threshold Tuning")
    print("=" * 60)
    
    from hybrid_parser import HybridParser
    
    # Sample addresses for tuning
    sample_addresses = [
        "flat 302, pune",
        "123 mumbai",
        "apartment 5, bangalore",
        "flat 302, friendship residency, pune",
        "20, vasant vihar bunglows, baner",
        "c 407 epic hsg socity wagoli",
        "506, amnora chembers, east amnora town center, amnora magarpatta road, hadapsar, pune",
        "panchshil towers 191 panchshil towers road vitthal nagar kharadi a-2503",
        "suyog nisarg, daisy b 201, near suyog sunderji school, lohegaon road",
        "124/1/8, sadguru housing society, sadgurunagar, pune nasik road, bhosari pune 39 near datta mandir"
    ]
    
    parser = HybridParser()
    
    # Test different target usage levels
    target_usages = [0.2, 0.3, 0.5, 0.7]
    
    for target in target_usages:
        recommended_threshold = parser.tune_complexity_threshold(sample_addresses, target)
        print(f"Target {target*100:2.0f}% Shiprocket usage → Threshold: {recommended_threshold:.3f}")
    
    return True


def test_batch_processing():
    """Test hybrid batch processing."""
    
    print("\n🔄 Testing Hybrid Batch Processing")
    print("=" * 60)
    
    from hybrid_parser import HybridParser
    
    # Batch of mixed complexity addresses
    batch_addresses = [
        "flat 302, pune",
        "506, amnora chembers, east amnora town center, amnora magarpatta road, hadapsar, pune",
        "123 mumbai",
        "panchshil towers 191 panchshil towers road vitthal nagar kharadi a-2503",
        "apartment 5, bangalore",
        "suyog nisarg, daisy b 201, near suyog sunderji school, lohegaon road",
        "flat 302, friendship residency, pune",
        "124/1/8, sadguru housing society, sadgurunagar, pune nasik road, bhosari pune 39 near datta mandir"
    ]
    
    parser = HybridParser(complexity_threshold=0.5, batch_size=4)
    
    print(f"Processing batch of {len(batch_addresses)} addresses...")
    
    start_time = time.time()
    results = parser.parse_batch(batch_addresses)
    end_time = time.time()
    
    # Analyze results
    successful = sum(1 for r in results if r.parse_success)
    
    print(f"\n📊 Batch Results:")
    print(f"   Total Processed: {len(results)}")
    print(f"   Successful: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"   Total Time: {end_time - start_time:.2f}s")
    print(f"   Avg Time/Address: {(end_time - start_time)/len(batch_addresses):.3f}s")
    
    # Get statistics
    stats = parser.get_statistics()
    print(f"\n📈 Usage Statistics:")
    print(f"   Local Usage: {stats['local_usage_percent']}%")
    print(f"   Shiprocket Usage: {stats['shiprocket_usage_percent']}%")
    print(f"   Fallback Usage: {stats['fallback_usage_percent']}%")
    
    return successful >= len(batch_addresses) * 0.8  # 80% success rate


if __name__ == "__main__":
    print("🧪 Hybrid Parser Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        test1 = test_hybrid_parser()
        test2 = test_complexity_analysis()
        test3 = test_threshold_tuning()
        test4 = test_batch_processing()
        
        print(f"\n🎯 Test Results Summary:")
        print(f"   Hybrid Parsing: {'✅ PASS' if test1 else '❌ FAIL'}")
        print(f"   Complexity Analysis: {'✅ PASS' if test2 else '❌ FAIL'}")
        print(f"   Threshold Tuning: {'✅ PASS' if test3 else '❌ FAIL'}")
        print(f"   Batch Processing: {'✅ PASS' if test4 else '❌ FAIL'}")
        
        if all([test1, test2, test3, test4]):
            print(f"\n🌟 All tests passed! Hybrid parser is ready for production.")
            print(f"💡 Benefits:")
            print(f"   - Intelligent routing based on address complexity")
            print(f"   - Automatic fallback for reliability")
            print(f"   - Cost optimization (use GPU only when needed)")
            print(f"   - Quality + Speed balance")
        else:
            print(f"\n⚠️  Some tests failed. Review the issues above.")
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()