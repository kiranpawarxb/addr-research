#!/usr/bin/env python3
"""
Process export_customer_address_store_p1.csv using optimized multi-device approach:
- 28 CPU cores (out of 32 available)
- NVIDIA GPU + Intel GPU
- Parallel processing with optimal work distribution
"""

import sys
import os
import pandas as pd
import logging
import time
import multiprocessing
from datetime import datetime
from typing import List, Dict

# Add src to path
sys.path.insert(0, 'src')

from src.ultimate_multi_device_parser import UltimateMultiDeviceParser


def setup_logging():
    """Set up detailed logging for P1 processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'p1_multi_device_processing_{timestamp}.log', encoding='utf-8')
        ]
    )


def load_p1_addresses() -> List[str]:
    """Load addresses from export_customer_address_store_p1.csv."""
    
    print(f"📂 Loading addresses from export_customer_address_store_p1.csv...")
    
    try:
        # Load the P1 CSV file
        df = pd.read_csv('export_customer_address_store_p1.csv')
        
        print(f"✅ Loaded CSV with {len(df)} rows")
        
        # Check column names
        print(f"📋 Columns: {list(df.columns)}")
        
        # Find address column (common names)
        address_column = None
        for col in ['addr_text', 'address', 'full_address', 'Address', 'addr']:
            if col in df.columns:
                address_column = col
                break
        
        if not address_column:
            print("❌ Could not find address column. Available columns:")
            for col in df.columns:
                print(f"   - {col}")
            return []
        
        print(f"🎯 Using address column: '{address_column}'")
        
        # Filter out empty addresses
        df_clean = df.dropna(subset=[address_column])
        df_clean = df_clean[df_clean[address_column].str.strip() != '']
        
        # Get all addresses
        addresses = df_clean[address_column].tolist()
        
        print(f"✅ Loaded {len(addresses):,} valid addresses for processing")
        return addresses
        
    except FileNotFoundError:
        print("❌ Error: export_customer_address_store_p1.csv not found")
        print("   Please ensure the file exists in the current directory")
        return []
    except Exception as e:
        print(f"❌ Error loading P1 file: {e}")
        return []


def process_p1_with_multi_device():
    """Process P1 file using optimized multi-device configuration."""
    
    print("🚀 PROCESSING P1 FILE WITH OPTIMIZED MULTI-DEVICE")
    print("=" * 80)
    
    # System information
    total_cpu_cores = multiprocessing.cpu_count()
    print(f"🖥️  System Information:")
    print(f"   Total CPU Cores: {total_cpu_cores}")
    print(f"   Using CPU Workers: 28 (as requested)")
    print(f"   GPUs: NVIDIA + Intel (parallel processing)")
    print()
    
    # Load P1 addresses
    addresses = load_p1_addresses()
    
    if not addresses:
        print("❌ No addresses to process. Exiting.")
        return
    
    print(f"📊 Processing Details:")
    print(f"   Total Addresses: {len(addresses):,}")
    print(f"   Target Configuration: 28 CPU + 2 GPUs = 30 workers")
    print()
    
    # Configure optimized multi-device parser
    print("🔧 Initializing Optimized Multi-Device Parser...")
    
    # Calculate CPU workers: Use 28 cores as requested
    cpu_cores_to_use = 28  # Use 28 CPU workers as requested
    cpu_multiplier = 1.0  # Conservative for stability
    
    parser = UltimateMultiDeviceParser(
        batch_size=40,  # Optimized batch size
        use_nvidia_gpu=True,
        use_intel_gpu=True,
        use_all_cpu_cores=True,
        cpu_core_multiplier=cpu_multiplier
    )
    
    # Override CPU workers to use 28 cores
    parser.cpu_workers = cpu_cores_to_use
    parser.max_workers = cpu_cores_to_use + 2  # 28 CPU + 2 GPUs
    
    print(f"✅ Parser configured:")
    print(f"   CPU Workers: {parser.cpu_workers}")
    print(f"   Total Workers: {parser.max_workers}")
    print(f"   Batch Size: {parser.batch_size}")
    print()
    
    # Start processing
    print("🚀 Starting P1 Multi-Device Processing...")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        # Process all addresses
        parsed_results = parser.parse_ultimate_multi_device(addresses)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        success_count = sum(1 for r in parsed_results if r.parse_success)
        failed_count = len(parsed_results) - success_count
        addresses_per_second = len(addresses) / total_time
        
        # Get detailed statistics
        stats = parser.get_statistics()
        
        # Print comprehensive results
        print("\n🎉 P1 MULTI-DEVICE PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"📊 PROCESSING RESULTS:")
        print(f"   Total Addresses: {len(addresses):,}")
        print(f"   Successfully Parsed: {success_count:,}")
        print(f"   Failed: {failed_count:,}")
        print(f"   Success Rate: {(success_count/len(addresses)*100):.1f}%")
        print()
        
        print(f"⏱️  TIMING RESULTS:")
        print(f"   Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Speed: {addresses_per_second:.1f} addresses/second")
        print(f"   Speed: {addresses_per_second*60:.0f} addresses/minute")
        print()
        
        # Device-specific performance
        device_stats = stats.get('device_stats', {})
        if device_stats:
            print(f"🔧 DEVICE PERFORMANCE:")
            total_device_time = 0
            
            for device, device_stat in device_stats.items():
                if device_stat['processed'] > 0:
                    device_speed = device_stat['processed'] / device_stat['time'] if device_stat['time'] > 0 else 0
                    total_device_time += device_stat['time']
                    utilization = (device_stat['processed'] / len(addresses)) * 100
                    
                    print(f"   {device}:")
                    print(f"     Processed: {device_stat['processed']:,} addresses ({utilization:.1f}%)")
                    print(f"     Speed: {device_speed:.1f} addresses/second")
                    print(f"     Time: {device_stat['time']:.1f} seconds")
            
            # Calculate parallelization efficiency
            if total_device_time > 0:
                efficiency = (total_device_time / total_time) * 100
                print(f"   Parallelization Efficiency: {efficiency:.1f}%")
        
        print()
        
        # Save results to CSV
        print("💾 SAVING RESULTS...")
        
        # Create results DataFrame
        results_data = []
        for i, (original_addr, parsed) in enumerate(zip(addresses, parsed_results)):
            results_data.append({
                'id': i + 1,
                'original_address': original_addr,
                'unit_number': parsed.unit_number if parsed.parse_success else '',
                'society_name': parsed.society_name if parsed.parse_success else '',
                'landmark': parsed.landmark if parsed.parse_success else '',
                'road': parsed.road if parsed.parse_success else '',
                'sub_locality': parsed.sub_locality if parsed.parse_success else '',
                'locality': parsed.locality if parsed.parse_success else '',
                'city': parsed.city if parsed.parse_success else '',
                'district': parsed.district if parsed.parse_success else '',
                'state': parsed.state if parsed.parse_success else '',
                'country': parsed.country if parsed.parse_success else '',
                'pin_code': parsed.pin_code if parsed.parse_success else '',
                'parse_success': parsed.parse_success,
                'parse_error': parsed.parse_error if not parsed.parse_success else '',
                'note': parsed.note if parsed.parse_success else ''
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'p1_parsed_addresses_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        
        print(f"✅ Results saved to: {output_file}")
        
        # Create consolidation summary
        if success_count > 0:
            print("\n📈 CREATING CONSOLIDATION SUMMARY...")
            
            # Group by PIN code for consolidation analysis
            successful_results = [r for r in parsed_results if r.parse_success]
            
            consolidation_data = {}
            for parsed in successful_results:
                pin_code = parsed.pin_code or 'Unknown'
                
                if pin_code not in consolidation_data:
                    consolidation_data[pin_code] = {
                        'total_addresses': 0,
                        'societies': set(),
                        'localities': set()
                    }
                
                consolidation_data[pin_code]['total_addresses'] += 1
                
                if parsed.society_name:
                    consolidation_data[pin_code]['societies'].add(parsed.society_name)
                
                if parsed.locality:
                    consolidation_data[pin_code]['localities'].add(parsed.locality)
            
            # Create consolidation summary
            consolidation_summary = []
            for pin_code, data in consolidation_data.items():
                consolidation_summary.append({
                    'pin_code': pin_code,
                    'total_addresses': data['total_addresses'],
                    'distinct_societies': len(data['societies']),
                    'distinct_localities': len(data['localities']),
                    'society_consolidation_pct': ((data['total_addresses'] - len(data['societies'])) / data['total_addresses'] * 100) if data['total_addresses'] > 0 else 0,
                    'locality_consolidation_pct': ((data['total_addresses'] - len(data['localities'])) / data['total_addresses'] * 100) if data['total_addresses'] > 0 else 0
                })
            
            consolidation_df = pd.DataFrame(consolidation_summary)
            consolidation_df = consolidation_df.sort_values('total_addresses', ascending=False)
            
            consolidation_file = f'p1_consolidation_summary_{timestamp}.csv'
            consolidation_df.to_csv(consolidation_file, index=False)
            
            print(f"✅ Consolidation summary saved to: {consolidation_file}")
            
            # Print top consolidation stats
            if len(consolidation_df) > 0:
                total_addresses_processed = consolidation_df['total_addresses'].sum()
                total_societies = consolidation_df['distinct_societies'].sum()
                total_localities = consolidation_df['distinct_localities'].sum()
                
                overall_society_consolidation = ((total_addresses_processed - total_societies) / total_addresses_processed * 100) if total_addresses_processed > 0 else 0
                overall_locality_consolidation = ((total_addresses_processed - total_localities) / total_addresses_processed * 100) if total_addresses_processed > 0 else 0
                
                print(f"\n📊 CONSOLIDATION RESULTS:")
                print(f"   Total Addresses: {total_addresses_processed:,}")
                print(f"   Distinct Societies: {total_societies:,}")
                print(f"   Distinct Localities: {total_localities:,}")
                print(f"   Society Consolidation: {overall_society_consolidation:.1f}%")
                print(f"   Locality Consolidation: {overall_locality_consolidation:.1f}%")
        
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   Configuration: 8 CPU workers + 2 GPUs")
        print(f"   Processing Speed: {addresses_per_second:.1f} addresses/second")
        print(f"   Total Time: {total_time/60:.1f} minutes")
        print(f"   Success Rate: {(success_count/len(addresses)*100):.1f}%")
        
        return {
            'total_addresses': len(addresses),
            'success_count': success_count,
            'total_time': total_time,
            'addresses_per_second': addresses_per_second,
            'output_file': output_file,
            'consolidation_file': consolidation_file if success_count > 0 else None
        }
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        logging.error(f"P1 processing failed: {e}", exc_info=True)
        return None


def main():
    """Main execution function."""
    
    setup_logging()
    
    print("🔧 P1 Multi-Device Address Parser")
    print("Configuration: 28 CPU workers + NVIDIA GPU + Intel GPU")
    print("=" * 80)
    
    try:
        results = process_p1_with_multi_device()
        
        if results:
            print(f"\n🎉 P1 Processing Complete!")
            print(f"Results saved to: {results['output_file']}")
            if results.get('consolidation_file'):
                print(f"Consolidation summary: {results['consolidation_file']}")
        else:
            print(f"\n❌ P1 Processing failed. Check logs for details.")
        
        return results
        
    except Exception as e:
        logging.error(f"Main execution failed: {e}", exc_info=True)
        print(f"\n❌ Execution failed: {e}")
        return None


if __name__ == "__main__":
    results = main()