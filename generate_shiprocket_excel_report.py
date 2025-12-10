#!/usr/bin/env python3
"""Generate Excel report for Shiprocket parser on 20 random addresses."""

import sys
import csv
import random
import time
import pandas as pd
from datetime import datetime
sys.path.insert(0, 'src')

def load_random_addresses(csv_file, count=20):
    """Load random addresses from CSV file."""
    
    print(f"📂 Loading addresses from {csv_file}...")
    
    # Increase CSV field size limit
    csv.field_size_limit(1000000)
    
    addresses = []
    try:
        # Use pandas to handle large CSV efficiently
        import pandas as pd
        
        print("🔄 Reading large CSV with pandas...")
        
        # Read first chunk to get column info
        chunk_iter = pd.read_csv(csv_file, chunksize=1000)
        first_chunk = next(chunk_iter)
        
        fieldnames = list(first_chunk.columns)
        print(f"📋 Available columns: {fieldnames}")
        
        # Find address column
        address_col = None
        for col in ['addr_text', 'raw_address', 'address', 'customer_address', 'full_address']:
            if col in fieldnames:
                address_col = col
                break
        
        if not address_col:
            address_col = fieldnames[0] if fieldnames else None
            print(f"⚠️  Using first column as address: {address_col}")
        else:
            print(f"✅ Using address column: {address_col}")
        
        # Collect addresses from multiple chunks
        all_addresses = []
        chunk_count = 0
        
        # Process first chunk
        chunk_addresses = first_chunk[address_col].dropna().tolist()
        all_addresses.extend(chunk_addresses)
        chunk_count += 1
        
        # Process additional chunks until we have enough samples
        for chunk in chunk_iter:
            chunk_addresses = chunk[address_col].dropna().tolist()
            all_addresses.extend(chunk_addresses)
            chunk_count += 1
            
            if len(all_addresses) >= count * 5:  # Get 5x more for better randomness
                break
        
        print(f"📊 Collected {len(all_addresses)} addresses from {chunk_count} chunks")
        addresses = all_addresses
        
        print(f"✅ Loaded {len(addresses)} addresses from CSV")
        
        # Select random sample
        if len(addresses) > count:
            selected = random.sample(addresses, count)
            print(f"🎲 Selected {count} random addresses")
        else:
            selected = addresses
            print(f"⚠️  Only {len(addresses)} addresses available, using all")
        
        return selected
        
    except FileNotFoundError:
        print(f"❌ CSV file not found: {csv_file}")
        return []
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []


def parse_addresses_with_shiprocket(addresses):
    """Parse addresses using the improved Shiprocket parser."""
    
    print(f"\n🚀 Parsing {len(addresses)} addresses with Shiprocket...")
    print("=" * 60)
    
    from shiprocket_parser import ShiprocketParser
    
    # Initialize parser
    parser = ShiprocketParser(use_gpu=False, batch_size=1)
    
    results = []
    total_time = 0
    
    for i, address in enumerate(addresses, 1):
        print(f"\n{i:2d}. Parsing: {address[:60]}...")
        
        start_time = time.time()
        result = parser.parse_address(address)
        end_time = time.time()
        
        parse_time = end_time - start_time
        total_time += parse_time
        
        # Create result record
        record = {
            'Index': i,
            'Raw_Address': address,
            'Success': result.parse_success,
            'Parse_Time_Seconds': round(parse_time, 3),
            'Unit_Number': result.unit_number or '',
            'Society_Name': result.society_name or '',
            'Landmark': result.landmark or '',
            'Road': result.road or '',
            'Sub_Locality': result.sub_locality or '',
            'Locality': result.locality or '',
            'City': result.city or '',
            'District': result.district or '',
            'State': result.state or '',
            'Country': result.country or '',
            'PIN_Code': result.pin_code or '',
            'Note': result.note or '',
            'Error': result.parse_error or ''
        }
        
        results.append(record)
        
        # Show progress
        if result.parse_success:
            print(f"    ✅ SUCCESS ({parse_time:.3f}s)")
            
            # Show extracted fields
            extracted = []
            if result.unit_number:
                extracted.append(f"Unit: '{result.unit_number}'")
            if result.society_name:
                extracted.append(f"Society: '{result.society_name}'")
            if result.landmark:
                extracted.append(f"Landmark: '{result.landmark}'")
            if result.road:
                extracted.append(f"Road: '{result.road}'")
            if result.locality:
                extracted.append(f"Locality: '{result.locality}'")
            if result.city:
                extracted.append(f"City: '{result.city}'")
            
            if extracted:
                for field in extracted[:3]:  # Show first 3 fields
                    print(f"       📍 {field}")
                if len(extracted) > 3:
                    print(f"       📍 ... and {len(extracted)-3} more fields")
            else:
                print("       ⚠️  No fields extracted")
        else:
            print(f"    ❌ FAILED ({parse_time:.3f}s)")
            print(f"       💥 {result.parse_error}")
    
    # Calculate summary statistics
    successful = [r for r in results if r['Success']]
    success_count = len(successful)
    
    print(f"\n📊 Parsing Summary:")
    print(f"   Total Addresses: {len(results)}")
    print(f"   Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"   Failed: {len(results)-success_count}")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Avg Time/Address: {total_time/len(results):.3f}s")
    
    # Get parser statistics
    stats = parser.get_statistics()
    print(f"\n📈 Parser Statistics:")
    print(f"   Success Rate: {stats['success_rate_percent']}%")
    print(f"   Total Retries: {stats['total_retries']}")
    
    return results, stats


def calculate_quality_metrics(results):
    """Calculate quality metrics from parsing results."""
    
    successful = [r for r in results if r['Success']]
    
    if not successful:
        return {}
    
    metrics = {
        'total_addresses': len(results),
        'successful_parses': len(successful),
        'success_rate': len(successful) / len(results) * 100,
        'unit_extraction_rate': sum(1 for r in successful if r['Unit_Number']) / len(successful) * 100,
        'society_extraction_rate': sum(1 for r in successful if r['Society_Name']) / len(successful) * 100,
        'landmark_extraction_rate': sum(1 for r in successful if r['Landmark']) / len(successful) * 100,
        'road_extraction_rate': sum(1 for r in successful if r['Road']) / len(successful) * 100,
        'locality_extraction_rate': sum(1 for r in successful if r['Locality']) / len(successful) * 100,
        'city_extraction_rate': sum(1 for r in successful if r['City']) / len(successful) * 100,
        'pin_extraction_rate': sum(1 for r in successful if r['PIN_Code']) / len(successful) * 100,
        'avg_parse_time': sum(r['Parse_Time_Seconds'] for r in results) / len(results),
        'total_parse_time': sum(r['Parse_Time_Seconds'] for r in results)
    }
    
    return metrics


def create_excel_report(results, stats, metrics, output_file):
    """Create comprehensive Excel report."""
    
    print(f"\n📊 Creating Excel report: {output_file}")
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Sheet 1: Detailed Results
        df_results = pd.DataFrame(results)
        df_results.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # Sheet 2: Summary Statistics
        summary_data = {
            'Metric': [
                'Total Addresses',
                'Successful Parses', 
                'Failed Parses',
                'Success Rate (%)',
                'Average Parse Time (s)',
                'Total Parse Time (s)',
                '',
                'Unit Number Extraction (%)',
                'Society Name Extraction (%)',
                'Landmark Extraction (%)',
                'Road Extraction (%)',
                'Locality Extraction (%)',
                'City Extraction (%)',
                'PIN Code Extraction (%)',
                '',
                'Parser Total Parsed',
                'Parser Total Failed',
                'Parser Total Retries',
                'Parser Success Rate (%)'
            ],
            'Value': [
                metrics['total_addresses'],
                metrics['successful_parses'],
                metrics['total_addresses'] - metrics['successful_parses'],
                f"{metrics['success_rate']:.1f}%",
                f"{metrics['avg_parse_time']:.3f}",
                f"{metrics['total_parse_time']:.2f}",
                '',
                f"{metrics['unit_extraction_rate']:.1f}%",
                f"{metrics['society_extraction_rate']:.1f}%",
                f"{metrics['landmark_extraction_rate']:.1f}%",
                f"{metrics['road_extraction_rate']:.1f}%",
                f"{metrics['locality_extraction_rate']:.1f}%",
                f"{metrics['city_extraction_rate']:.1f}%",
                f"{metrics['pin_extraction_rate']:.1f}%",
                '',
                stats['total_parsed'],
                stats['total_failed'],
                stats['total_retries'],
                f"{stats['success_rate_percent']}%"
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Sheet 3: Quality Analysis
        successful_results = [r for r in results if r['Success']]
        
        quality_data = []
        for result in successful_results:
            quality_data.append({
                'Address': result['Raw_Address'][:50] + '...' if len(result['Raw_Address']) > 50 else result['Raw_Address'],
                'Fields_Extracted': sum([
                    1 if result['Unit_Number'] else 0,
                    1 if result['Society_Name'] else 0,
                    1 if result['Landmark'] else 0,
                    1 if result['Road'] else 0,
                    1 if result['Locality'] else 0,
                    1 if result['City'] else 0,
                    1 if result['PIN_Code'] else 0
                ]),
                'Has_Unit': 'Yes' if result['Unit_Number'] else 'No',
                'Has_Society': 'Yes' if result['Society_Name'] else 'No',
                'Has_Landmark': 'Yes' if result['Landmark'] else 'No',
                'Has_Road': 'Yes' if result['Road'] else 'No',
                'Has_Locality': 'Yes' if result['Locality'] else 'No',
                'Has_City': 'Yes' if result['City'] else 'No',
                'Has_PIN': 'Yes' if result['PIN_Code'] else 'No',
                'Parse_Time': result['Parse_Time_Seconds']
            })
        
        if quality_data:
            df_quality = pd.DataFrame(quality_data)
            df_quality.to_excel(writer, sheet_name='Quality_Analysis', index=False)
        
        # Sheet 4: Failed Addresses (if any)
        failed_results = [r for r in results if not r['Success']]
        if failed_results:
            failed_data = []
            for result in failed_results:
                failed_data.append({
                    'Address': result['Raw_Address'],
                    'Error': result['Error'],
                    'Parse_Time': result['Parse_Time_Seconds']
                })
            
            df_failed = pd.DataFrame(failed_data)
            df_failed.to_excel(writer, sheet_name='Failed_Addresses', index=False)
        
        # Sheet 5: Comparison with Targets
        comparison_data = {
            'Metric': [
                'Success Rate',
                'Society Extraction',
                'Locality Extraction', 
                'Road Extraction',
                'Unit Extraction',
                'Processing Time'
            ],
            'Target': [
                '>95%',
                '>30%',
                '>30%',
                '>15%',
                '>50%',
                '<2s'
            ],
            'Achieved': [
                f"{metrics['success_rate']:.1f}%",
                f"{metrics['society_extraction_rate']:.1f}%",
                f"{metrics['locality_extraction_rate']:.1f}%",
                f"{metrics['road_extraction_rate']:.1f}%",
                f"{metrics['unit_extraction_rate']:.1f}%",
                f"{metrics['avg_parse_time']:.3f}s"
            ],
            'Status': [
                '✅ PASS' if metrics['success_rate'] >= 95 else '⚠️ REVIEW',
                '✅ PASS' if metrics['society_extraction_rate'] >= 30 else '⚠️ REVIEW',
                '✅ PASS' if metrics['locality_extraction_rate'] >= 30 else '⚠️ REVIEW',
                '✅ PASS' if metrics['road_extraction_rate'] >= 15 else '⚠️ REVIEW',
                '✅ PASS' if metrics['unit_extraction_rate'] >= 50 else '⚠️ REVIEW',
                '✅ PASS' if metrics['avg_parse_time'] <= 2.0 else '⚠️ REVIEW'
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_excel(writer, sheet_name='Target_Comparison', index=False)
    
    print(f"✅ Excel report created successfully!")
    return output_file


def main():
    """Main function to generate Shiprocket Excel report."""
    
    print("📊 Shiprocket Parser - Excel Report Generator")
    print("=" * 60)
    print(f"🕒 Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration - use the large CSV with proper column
    csv_file = 'export_customer_address_store_p0.csv'
    output_file = f'shiprocket_large_csv_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    address_count = 20
    
    try:
        # Step 1: Load random addresses
        addresses = load_random_addresses(csv_file, address_count)
        
        if not addresses:
            print("❌ No addresses loaded. Exiting.")
            return
        
        # Step 2: Parse addresses with Shiprocket
        results, stats = parse_addresses_with_shiprocket(addresses)
        
        # Step 3: Calculate quality metrics
        metrics = calculate_quality_metrics(results)
        
        # Step 4: Create Excel report
        report_file = create_excel_report(results, stats, metrics, output_file)
        
        # Step 5: Display final summary
        print(f"\n🎯 Final Report Summary:")
        print(f"   📁 Report File: {report_file}")
        print(f"   📊 Addresses Processed: {len(results)}")
        print(f"   ✅ Success Rate: {metrics['success_rate']:.1f}%")
        print(f"   🏢 Society Extraction: {metrics['society_extraction_rate']:.1f}%")
        print(f"   📍 Locality Extraction: {metrics['locality_extraction_rate']:.1f}%")
        print(f"   🛣️  Road Extraction: {metrics['road_extraction_rate']:.1f}%")
        print(f"   ⏱️  Avg Parse Time: {metrics['avg_parse_time']:.3f}s")
        
        # Quality assessment
        print(f"\n💡 Quality Assessment:")
        if metrics['success_rate'] >= 95:
            print("   ✅ Excellent reliability")
        elif metrics['success_rate'] >= 80:
            print("   ⚠️  Good reliability")
        else:
            print("   ❌ Poor reliability")
            
        if metrics['society_extraction_rate'] >= 80:
            print("   ✅ Excellent society extraction")
        elif metrics['society_extraction_rate'] >= 50:
            print("   ⚠️  Good society extraction")
        else:
            print("   ❌ Poor society extraction")
            
        if metrics['locality_extraction_rate'] >= 80:
            print("   ✅ Excellent locality extraction")
        elif metrics['locality_extraction_rate'] >= 50:
            print("   ⚠️  Good locality extraction")
        else:
            print("   ❌ Poor locality extraction")
        
        print(f"\n🏁 Report generation complete!")
        print(f"📂 Open {report_file} in Excel to view detailed results.")
        
    except Exception as e:
        print(f"\n💥 Error generating report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()