#!/usr/bin/env python3
"""Test Shiprocket quality improvements after reliability fixes."""

import sys
import time
import csv
sys.path.insert(0, 'src')

def test_quality_improvements():
    """Test quality improvements on the original comparison dataset."""
    
    print("🎯 Testing Shiprocket Quality After Reliability Fixes")
    print("=" * 60)
    
    from shiprocket_parser import ShiprocketParser
    
    # Test addresses from the original comparison
    test_addresses = [
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
    
    parser = ShiprocketParser(use_gpu=False, batch_size=1)
    
    print(f"Testing {len(test_addresses)} addresses...")
    print("-" * 60)
    
    results = []
    total_time = 0
    
    for i, address in enumerate(test_addresses, 1):
        print(f"\n{i:2d}. {address}")
        
        start_time = time.time()
        result = parser.parse_address(address)
        end_time = time.time()
        
        parse_time = end_time - start_time
        total_time += parse_time
        
        results.append(result)
        
        if result.parse_success:
            print(f"    ✅ SUCCESS ({parse_time:.3f}s)")
            
            # Show extracted fields
            fields_found = []
            if result.unit_number:
                fields_found.append(f"Unit: '{result.unit_number}'")
            if result.society_name:
                fields_found.append(f"Society: '{result.society_name}'")
            if result.landmark:
                fields_found.append(f"Landmark: '{result.landmark}'")
            if result.road:
                fields_found.append(f"Road: '{result.road}'")
            if result.locality:
                fields_found.append(f"Locality: '{result.locality}'")
            if result.city:
                fields_found.append(f"City: '{result.city}'")
            if result.pin_code:
                fields_found.append(f"PIN: '{result.pin_code}'")
            
            if fields_found:
                for field in fields_found:
                    print(f"    📍 {field}")
            else:
                print("    ⚠️  No fields extracted")
        else:
            print(f"    ❌ FAILED ({parse_time:.3f}s)")
            print(f"    💥 {result.parse_error}")
    
    # Calculate statistics
    successful = [r for r in results if r.parse_success]
    success_count = len(successful)
    total_count = len(results)
    
    print(f"\n📊 Overall Results:")
    print(f"   Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Avg Time/Address: {total_time/total_count:.3f}s")
    
    # Quality analysis
    if successful:
        unit_count = sum(1 for r in successful if r.unit_number)
        society_count = sum(1 for r in successful if r.society_name)
        landmark_count = sum(1 for r in successful if r.landmark)
        road_count = sum(1 for r in successful if r.road)
        locality_count = sum(1 for r in successful if r.locality)
        city_count = sum(1 for r in successful if r.city)
        pin_count = sum(1 for r in successful if r.pin_code)
        
        print(f"\n🎯 Quality Analysis (of {len(successful)} successful parses):")
        print(f"   Unit Numbers: {unit_count}/{len(successful)} ({unit_count/len(successful)*100:.1f}%)")
        print(f"   Society Names: {society_count}/{len(successful)} ({society_count/len(successful)*100:.1f}%)")
        print(f"   Landmarks: {landmark_count}/{len(successful)} ({landmark_count/len(successful)*100:.1f}%)")
        print(f"   Roads: {road_count}/{len(successful)} ({road_count/len(successful)*100:.1f}%)")
        print(f"   Localities: {locality_count}/{len(successful)} ({locality_count/len(successful)*100:.1f}%)")
        print(f"   Cities: {city_count}/{len(successful)} ({city_count/len(successful)*100:.1f}%)")
        print(f"   PIN Codes: {pin_count}/{len(successful)} ({pin_count/len(successful)*100:.1f}%)")
    
    # Get parser statistics
    stats = parser.get_statistics()
    print(f"\n📈 Parser Statistics:")
    print(f"   Total Parsed: {stats['total_parsed']}")
    print(f"   Total Failed: {stats['total_failed']}")
    print(f"   Total Retries: {stats['total_retries']}")
    print(f"   Success Rate: {stats['success_rate_percent']}%")
    
    # Compare with previous results
    print(f"\n📋 Comparison with Previous Results:")
    print(f"   Previous Success Rate: 40% (8/20)")
    print(f"   Current Success Rate: {stats['success_rate_percent']}%")
    
    if stats['success_rate_percent'] > 40:
        improvement = stats['success_rate_percent'] - 40
        print(f"   🎉 Improvement: +{improvement:.1f}% success rate!")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if stats['success_rate_percent'] >= 90:
        print("   ✅ Excellent reliability - ready for production!")
        print("   🚀 Consider GPU scaling for higher throughput")
        
        if society_count >= len(successful) * 0.3:  # 30%+ society extraction
            print("   🌟 Good society name extraction quality")
        
        if road_count >= len(successful) * 0.2:  # 20%+ road extraction
            print("   🛣️  Good road name extraction quality")
            
    elif stats['success_rate_percent'] >= 70:
        print("   ⚠️  Good reliability but could be better")
        print("   🔧 Consider further optimization")
        
    else:
        print("   ❌ Poor reliability - needs more work")
        print("   💡 Stick with Local parser for now")
    
    return stats['success_rate_percent'], society_count, road_count, locality_count


if __name__ == "__main__":
    try:
        success_rate, societies, roads, localities = test_quality_improvements()
        
        print(f"\n🏁 Final Assessment:")
        
        if success_rate >= 90:
            print("   🌟 READY FOR PRODUCTION")
            print("   💰 Quality justifies GPU costs")
            print("   📈 Proceed with scaling plan")
        elif success_rate >= 70:
            print("   ⚠️  NEEDS MINOR IMPROVEMENTS")
            print("   🔧 Fix remaining issues first")
        else:
            print("   ❌ NOT READY FOR PRODUCTION")
            print("   💡 Use Local parser instead")
            
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()