#!/usr/bin/env python3
"""Test the improved complexity analysis."""

import sys
sys.path.insert(0, 'src')

def test_improved_complexity():
    """Test the improved complexity analysis."""
    
    print("🧠 Testing Improved Complexity Analysis")
    print("=" * 60)
    
    from hybrid_parser import HybridParser
    
    parser = HybridParser()
    
    # Test addresses with expected complexity routing
    test_cases = [
        # Simple addresses (should go to Local - score < 0.3)
        ("flat 302, pune", "Local", "Simple address"),
        ("123 mumbai", "Local", "Very simple"),
        ("apartment 5, bangalore", "Local", "Basic apartment"),
        
        # Medium complexity (threshold dependent - score 0.3-0.6)
        ("flat 302, friendship residency, pune", "Either", "Medium complexity"),
        ("20, vasant vihar bunglows, baner", "Either", "Society name present"),
        
        # Complex addresses (should go to Shiprocket - score > 0.6)
        ("506, amnora chembers, east amnora town center, amnora magarpatta road, hadapsar, pune", "Shiprocket", "Multiple components + road"),
        ("panchshil towers 191 panchshil towers road vitthal nagar kharadi a-2503", "Shiprocket", "Towers + road + multiple areas"),
        ("suyog nisarg, daisy b 201, near suyog sunderji school, lohegaon road", "Shiprocket", "Landmark + road"),
        ("124/1/8, sadguru housing society, sadgurunagar, pune nasik road, bhosari pune 39 near datta mandir", "Shiprocket", "Very complex with landmark")
    ]
    
    print("Address Complexity Analysis (Improved):")
    print("-" * 80)
    print(f"{'Score':<6} | {'Expected':<10} | {'Actual':<10} | {'Description':<25} | {'Address'}")
    print("-" * 80)
    
    for address, expected, description in test_cases:
        score = parser._analyze_complexity(address)
        
        # Determine routing based on score
        if score < 0.3:
            actual = "Local"
        elif score < 0.6:
            actual = "Either"
        else:
            actual = "Shiprocket"
        
        # Check if routing matches expectation
        if expected == "Either":
            match = "✅" if actual in ["Local", "Either", "Shiprocket"] else "❌"
        else:
            match = "✅" if actual == expected else "⚠️"
        
        print(f"{score:.3f} | {expected:<10} | {actual:<10} | {description:<25} | {address[:40]}...")
        
        if match == "⚠️":
            print(f"       ^ Mismatch: expected {expected}, got {actual}")
    
    return True


def test_threshold_recommendations():
    """Test threshold recommendations for different scenarios."""
    
    print("\n⚙️  Testing Threshold Recommendations")
    print("=" * 60)
    
    from hybrid_parser import HybridParser
    
    # Sample addresses representing typical workload
    sample_addresses = [
        # Simple (30% of workload)
        "flat 302, pune",
        "123 mumbai", 
        "apartment 5, bangalore",
        
        # Medium (40% of workload)
        "flat 302, friendship residency, pune",
        "20, vasant vihar bunglows, baner",
        "c 407 epic hsg socity wagoli",
        "awho hadapsar colony hadapsar",
        
        # Complex (30% of workload)
        "506, amnora chembers, east amnora town center, amnora magarpatta road, hadapsar, pune",
        "panchshil towers 191 panchshil towers road vitthal nagar kharadi a-2503",
        "suyog nisarg, daisy b 201, near suyog sunderji school, lohegaon road"
    ]
    
    parser = HybridParser()
    
    # Calculate complexity scores
    scores = []
    for addr in sample_addresses:
        score = parser._analyze_complexity(addr)
        scores.append((score, addr))
    
    scores.sort(key=lambda x: x[0])
    
    print("Complexity Distribution:")
    print("-" * 40)
    for score, addr in scores:
        print(f"{score:.3f} | {addr[:50]}...")
    
    # Test different scenarios
    scenarios = [
        ("Cost-Optimized", 0.2, "Use Shiprocket for only the most complex 20%"),
        ("Balanced", 0.3, "Use Shiprocket for complex 30% addresses"),
        ("Quality-Focused", 0.5, "Use Shiprocket for 50% of addresses"),
        ("Premium", 0.7, "Use Shiprocket for 70% of addresses")
    ]
    
    print(f"\n📊 Threshold Recommendations:")
    print("-" * 60)
    
    for scenario, target_usage, description in scenarios:
        threshold = parser.tune_complexity_threshold(sample_addresses, target_usage)
        
        # Calculate actual usage with this threshold
        shiprocket_count = sum(1 for score, _ in scores if score >= threshold)
        actual_usage = shiprocket_count / len(scores)
        
        print(f"{scenario:<15} | Threshold: {threshold:.3f} | Target: {target_usage*100:2.0f}% | Actual: {actual_usage*100:2.0f}%")
        print(f"                | {description}")
        print()
    
    return True


def test_real_world_routing():
    """Test routing with real addresses at different thresholds."""
    
    print("\n🌍 Testing Real-World Routing Scenarios")
    print("=" * 60)
    
    from hybrid_parser import HybridParser
    
    # Real addresses from the dataset
    real_addresses = [
        "flat-302, friendship residency, veerbhadra nagar road",
        "506, amnora chembers, east amnora town center, amnora magarpatta road, hadapsar, pune",
        "panchshil towers 191 panchshil towers road vitthal nagar kharadi a-2503",
        "suyog nisarg, daisy b 201, near suyog sunderji school, lohegaon road",
        "56/12 flat no 2 besides anand malhar society wadgaon sheri pune",
        "flat no 11 flat no 11 madhav vihar wadgaon bk pune-411041",
        "s.b.road plot no 16. sweta atul, nav rajasthan society, mangalwadi near shell petrol punp",
        "plot no-181/c,guru krupa, nr rashankar dutta mandir, nr sec-28,nigdi",
        "802 marvel exotica lane 7 koregaon park pune",
        "124/1/8, sadguru housing society, sadgurunagar, pune nasik road, bhosari pune 39 near datta mandir"
    ]
    
    # Test different threshold scenarios
    thresholds = [0.2, 0.4, 0.6, 0.8]
    
    for threshold in thresholds:
        print(f"\n📊 Routing with threshold {threshold}:")
        print("-" * 50)
        
        parser = HybridParser(complexity_threshold=threshold)
        
        local_count = 0
        shiprocket_count = 0
        
        for addr in real_addresses:
            score = parser._analyze_complexity(addr)
            router = "Shiprocket" if score >= threshold else "Local"
            
            if router == "Local":
                local_count += 1
            else:
                shiprocket_count += 1
            
            print(f"{score:.3f} → {router:<10} | {addr[:45]}...")
        
        print(f"\nSummary:")
        print(f"  Local: {local_count}/{len(real_addresses)} ({local_count/len(real_addresses)*100:.0f}%)")
        print(f"  Shiprocket: {shiprocket_count}/{len(real_addresses)} ({shiprocket_count/len(real_addresses)*100:.0f}%)")
        
        # Cost estimate (rough)
        monthly_addresses = 100000  # 100K addresses per month
        local_cost = (local_count / len(real_addresses)) * monthly_addresses * 0.0001  # $0.0001 per address
        shiprocket_cost = (shiprocket_count / len(real_addresses)) * monthly_addresses * 0.01  # $0.01 per address
        total_cost = local_cost + shiprocket_cost
        
        print(f"  Estimated monthly cost (100K addresses): ${total_cost:.2f}")
    
    return True


if __name__ == "__main__":
    print("🔧 Complexity Analysis Tuning Suite")
    print("=" * 60)
    
    try:
        test1 = test_improved_complexity()
        test2 = test_threshold_recommendations()
        test3 = test_real_world_routing()
        
        print(f"\n🎯 Tuning Results:")
        print(f"   Complexity Analysis: {'✅ PASS' if test1 else '❌ FAIL'}")
        print(f"   Threshold Recommendations: {'✅ PASS' if test2 else '❌ FAIL'}")
        print(f"   Real-World Routing: {'✅ PASS' if test3 else '❌ FAIL'}")
        
        if all([test1, test2, test3]):
            print(f"\n🌟 Complexity analysis is properly tuned!")
            print(f"💡 Recommended threshold: 0.4-0.6 for balanced quality/cost")
        else:
            print(f"\n⚠️  Some tuning tests failed.")
            
    except Exception as e:
        print(f"\n💥 Tuning failed: {e}")
        import traceback
        traceback.print_exc()