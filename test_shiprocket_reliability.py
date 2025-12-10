#!/usr/bin/env python3
"""Test Shiprocket parser reliability fixes.

This script tests the improved Shiprocket parser with:
1. Better error handling
2. Retry logic
3. Device management fixes
4. Graceful degradation
"""

import sys
import time
import logging
from typing import List

# Add src to path
sys.path.insert(0, 'src')

from shiprocket_parser import ShiprocketParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_reliability_fixes():
    """Test the reliability improvements in Shiprocket parser."""
    
    print("🔧 Testing Shiprocket Parser Reliability Fixes")
    print("=" * 60)
    
    # Test addresses (mix of simple and complex)
    test_addresses = [
        "flat-302, friendship residency, veerbhadra nagar road",
        "506, amnora chembers, east amnora town center, amnora magarpatta road, hadapsar, pune",
        "panchshil towers 191 panchshil towers road vitthal nagar kharadi a-2503",
        "suyog nisarg, daisy b 201, near suyog sunderji school, lohegaon road",
        "101, shivam building, behind shree kalyani nursing home, lohegaon, pune",
        "flat no 11 madhav vihar wadgaon bk pune-411041",
        "plot no-181/c,guru krupa, nr rashankar dutta mandir, nr sec-28,nigdi",
        "802 marvel exotica lane 7 koregaon park pune",
        "20, vasant vihar bunglows, baner",
        "c 407 epic hsg socity wagoli"
    ]
    
    # Test CPU version first (more reliable)
    print("\n📊 Testing CPU Version (Recommended for Production)")
    print("-" * 50)
    
    parser_cpu = ShiprocketParser(use_gpu=False, batch_size=1)
    
    cpu_results = []
    cpu_start_time = time.time()
    
    for i, address in enumerate(test_addresses, 1):
        print(f"\n{i:2d}. Testing: {address[:50]}...")
        
        start_time = time.time()
        result = parser_cpu.parse_address(address)
        end_time = time.time()
        
        cpu_results.append(result)
        
        if result.parse_success:
            print(f"    ✅ SUCCESS ({end_time - start_time:.3f}s)")
            if result.society_name:
                print(f"    🏢 Society: {result.society_name}")
            if result.unit_number:
                print(f"    🏠 Unit: {result.unit_number}")
            if result.locality:
                print(f"    📍 Locality: {result.locality}")
            if result.road:
                print(f"    🛣️  Road: {result.road}")
        else:
            print(f"    ❌ FAILED ({end_time - start_time:.3f}s)")
            print(f"    💥 Error: {result.parse_error}")
    
    cpu_total_time = time.time() - cpu_start_time
    
    # Get CPU statistics
    cpu_stats = parser_cpu.get_statistics()
    
    print(f"\n📈 CPU Results Summary:")
    print(f"   Success Rate: {cpu_stats['success_rate_percent']}% ({cpu_stats['total_parsed']}/{cpu_stats['total_attempts']})")
    print(f"   Total Retries: {cpu_stats['total_retries']}")
    print(f"   Total Time: {cpu_total_time:.2f}s")
    print(f"   Avg Time/Address: {cpu_total_time/len(test_addresses):.3f}s")
    
    # Test GPU version if available
    try:
        import torch
        if torch.cuda.is_available():
            print("\n🚀 Testing GPU Version (Higher Performance)")
            print("-" * 50)
            
            parser_gpu = ShiprocketParser(use_gpu=True, batch_size=1)
            
            gpu_results = []
            gpu_start_time = time.time()
            
            for i, address in enumerate(test_addresses, 1):
                print(f"\n{i:2d}. Testing: {address[:50]}...")
                
                start_time = time.time()
                result = parser_gpu.parse_address(address)
                end_time = time.time()
                
                gpu_results.append(result)
                
                if result.parse_success:
                    print(f"    ✅ SUCCESS ({end_time - start_time:.3f}s)")
                    if result.society_name:
                        print(f"    🏢 Society: {result.society_name}")
                    if result.unit_number:
                        print(f"    🏠 Unit: {result.unit_number}")
                    if result.locality:
                        print(f"    📍 Locality: {result.locality}")
                    if result.road:
                        print(f"    🛣️  Road: {result.road}")
                else:
                    print(f"    ❌ FAILED ({end_time - start_time:.3f}s)")
                    print(f"    💥 Error: {result.parse_error}")
            
            gpu_total_time = time.time() - gpu_start_time
            
            # Get GPU statistics
            gpu_stats = parser_gpu.get_statistics()
            
            print(f"\n📈 GPU Results Summary:")
            print(f"   Success Rate: {gpu_stats['success_rate_percent']}% ({gpu_stats['total_parsed']}/{gpu_stats['total_attempts']})")
            print(f"   Total Retries: {gpu_stats['total_retries']}")
            print(f"   Total Time: {gpu_total_time:.2f}s")
            print(f"   Avg Time/Address: {gpu_total_time/len(test_addresses):.3f}s")
            
            # Compare CPU vs GPU
            print(f"\n⚡ Performance Comparison:")
            print(f"   CPU Success Rate: {cpu_stats['success_rate_percent']}%")
            print(f"   GPU Success Rate: {gpu_stats['success_rate_percent']}%")
            print(f"   CPU Avg Time: {cpu_total_time/len(test_addresses):.3f}s")
            print(f"   GPU Avg Time: {gpu_total_time/len(test_addresses):.3f}s")
            
            if gpu_total_time > 0:
                speedup = cpu_total_time / gpu_total_time
                print(f"   GPU Speedup: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        
        else:
            print("\n⚠️  GPU not available - skipping GPU tests")
            
    except ImportError:
        print("\n⚠️  PyTorch not available - skipping GPU tests")
    
    # Quality analysis
    print(f"\n🎯 Quality Analysis:")
    successful_results = [r for r in cpu_results if r.parse_success]
    
    if successful_results:
        society_count = sum(1 for r in successful_results if r.society_name)
        unit_count = sum(1 for r in successful_results if r.unit_number)
        locality_count = sum(1 for r in successful_results if r.locality)
        road_count = sum(1 for r in successful_results if r.road)
        
        print(f"   Society Names: {society_count}/{len(successful_results)} ({society_count/len(successful_results)*100:.1f}%)")
        print(f"   Unit Numbers: {unit_count}/{len(successful_results)} ({unit_count/len(successful_results)*100:.1f}%)")
        print(f"   Localities: {locality_count}/{len(successful_results)} ({locality_count/len(successful_results)*100:.1f}%)")
        print(f"   Roads: {road_count}/{len(successful_results)} ({road_count/len(successful_results)*100:.1f}%)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if cpu_stats['success_rate_percent'] >= 90:
        print("   ✅ Reliability fixes are working well!")
        print("   ✅ Ready for production use")
        
        if cpu_stats['success_rate_percent'] >= 95:
            print("   🌟 Excellent reliability - consider scaling up")
        
    elif cpu_stats['success_rate_percent'] >= 70:
        print("   ⚠️  Good reliability but room for improvement")
        print("   💡 Consider increasing retry attempts or model optimization")
        
    else:
        print("   ❌ Poor reliability - needs more work")
        print("   🔧 Check model installation and dependencies")
        print("   💡 Consider using Local parser for production")
    
    print(f"\n🏁 Test Complete!")
    return cpu_stats['success_rate_percent'] >= 90


def test_batch_processing():
    """Test batch processing reliability."""
    
    print("\n🔄 Testing Batch Processing Reliability")
    print("=" * 50)
    
    # Larger test set
    addresses = [
        "flat-302, friendship residency, veerbhadra nagar road",
        "506, amnora chembers, east amnora town center, hadapsar, pune",
        "panchshil towers 191 panchshil towers road vitthal nagar kharadi",
        "suyog nisarg, daisy b 201, near suyog sunderji school, lohegaon road",
        "101, shivam building, behind nursing home, lohegaon, pune",
        "flat no 11 madhav vihar wadgaon bk pune-411041",
        "plot no-181/c,guru krupa, nr dutta mandir, nigdi",
        "802 marvel exotica lane 7 koregaon park pune",
        "20, vasant vihar bunglows, baner",
        "c 407 epic hsg socity wagoli",
        "ace almighty, indira college road tathwade, wakad",
        "m 305-310 2nd floor mega center, hadapsar pune",
        "flat no. 304 gulab vishwa behind modi ganpati pune",
        "profile enclave, wamanrao g more road, aundh road, khadki",
        "awho hadapsar colony hadapsar"
    ]
    
    parser = ShiprocketParser(use_gpu=False, batch_size=5)
    
    print(f"Processing {len(addresses)} addresses in batch...")
    
    start_time = time.time()
    results = parser.parse_batch(addresses)
    end_time = time.time()
    
    # Analyze results
    successful = sum(1 for r in results if r.parse_success)
    failed = len(results) - successful
    
    print(f"\n📊 Batch Results:")
    print(f"   Total Processed: {len(results)}")
    print(f"   Successful: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"   Failed: {failed} ({failed/len(results)*100:.1f}%)")
    print(f"   Total Time: {end_time - start_time:.2f}s")
    print(f"   Avg Time/Address: {(end_time - start_time)/len(addresses):.3f}s")
    
    # Get final statistics
    stats = parser.get_statistics()
    print(f"\n📈 Final Statistics:")
    print(f"   Success Rate: {stats['success_rate_percent']}%")
    print(f"   Total Retries: {stats['total_retries']}")
    
    return stats['success_rate_percent'] >= 85


if __name__ == "__main__":
    print("🧪 Shiprocket Parser Reliability Test Suite")
    print("=" * 60)
    
    try:
        # Test individual parsing
        individual_success = test_reliability_fixes()
        
        # Test batch processing
        batch_success = test_batch_processing()
        
        print(f"\n🎯 Overall Test Results:")
        print(f"   Individual Parsing: {'✅ PASS' if individual_success else '❌ FAIL'}")
        print(f"   Batch Processing: {'✅ PASS' if batch_success else '❌ FAIL'}")
        
        if individual_success and batch_success:
            print(f"\n🌟 All tests passed! Shiprocket parser is ready for production.")
            print(f"💡 Next steps: Scale up with GPU instances for better performance")
        else:
            print(f"\n⚠️  Some tests failed. Review the issues above.")
            print(f"💡 Consider using Local parser until reliability improves")
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        print(f"🔧 Check dependencies: pip install transformers torch")
        import traceback
        traceback.print_exc()