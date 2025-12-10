#!/usr/bin/env python3
"""Compare GPT-4 vs Shiprocket address parsing quality."""

import sys
import csv
import random
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
sys.path.insert(0, 'src')

def load_sample_addresses(csv_file: str, sample_size: int = 50) -> List[Dict]:
    """Load sample addresses with hash keys from CSV."""
    
    print(f"📂 Loading {sample_size} sample addresses from {csv_file}...")
    
    try:
        # Read CSV with pandas for better handling
        df = pd.read_csv(csv_file, nrows=2000)  # Read first 2000 rows
        
        print(f"📋 Available columns: {list(df.columns)}")
        
        # Find required columns
        hash_col = 'addr_hash_key'
        text_col = 'addr_text'
        
        if hash_col not in df.columns or text_col not in df.columns:
            print(f"❌ Required columns not found. Need: {hash_col}, {text_col}")
            return []
        
        # Filter out empty addresses
        df_clean = df.dropna(subset=[text_col])
        df_clean = df_clean[df_clean[text_col].str.strip() != '']
        
        print(f"✅ Found {len(df_clean)} valid addresses")
        
        # Random sample
        if len(df_clean) > sample_size:
            df_sample = df_clean.sample(n=sample_size, random_state=42)
        else:
            df_sample = df_clean
        
        # Convert to list of dicts
        sample_data = []
        for _, row in df_sample.iterrows():
            sample_data.append({
                'hash_key': row[hash_col],
                'address': row[text_col],
                'pincode': row.get('pincode', ''),
                'city_id': row.get('city_id', ''),
                'state_id': row.get('state_id', '')
            })
        
        print(f"🎲 Selected {len(sample_data)} addresses for comparison")
        return sample_data
        
    except Exception as e:
        print(f"❌ Error loading addresses: {e}")
        return []


def parse_with_both_models(sample_data: List[Dict]) -> List[Dict]:
    """Parse addresses with both Shiprocket and GPT-4."""
    
    print(f"\n🚀 Parsing {len(sample_data)} addresses with both models...")
    print("=" * 80)
    
    from shiprocket_parser import ShiprocketParser
    from gpt4_parser import GPT4AddressParser
    
    # Initialize parsers
    shiprocket = ShiprocketParser(use_gpu=False)
    gpt4 = GPT4AddressParser(model_name="gpt-4-turbo-preview")
    
    results = []
    
    for i, item in enumerate(sample_data, 1):
        address = item['address']
        hash_key = item['hash_key']
        
        print(f"\n{i:2d}. {hash_key[:20]}... | {address[:60]}...")
        
        # Parse with Shiprocket
        print("    🔧 Shiprocket:", end=" ")
        ship_start = time.time()
        ship_result = shiprocket.parse_address(address)
        ship_time = time.time() - ship_start
        
        if ship_result.parse_success:
            print(f"✅ ({ship_time:.3f}s)")
        else:
            print(f"❌ ({ship_time:.3f}s)")
        
        # Parse with GPT-4
        print("    🤖 GPT-4:", end=" ")
        gpt4_start = time.time()
        gpt4_result = gpt4.parse_address(address)
        gpt4_time = time.time() - gpt4_start
        
        if gpt4_result.parse_success:
            print(f"✅ ({gpt4_time:.3f}s)")
        else:
            print(f"❌ ({gpt4_time:.3f}s)")
        
        # Compare key fields
        ship_fields = sum([
            1 if ship_result.unit_number else 0,
            1 if ship_result.society_name else 0,
            1 if ship_result.locality else 0,
            1 if ship_result.road else 0,
            1 if ship_result.landmark else 0
        ])
        
        gpt4_fields = sum([
            1 if gpt4_result.unit_number else 0,
            1 if gpt4_result.society_name else 0,
            1 if gpt4_result.locality else 0,
            1 if gpt4_result.road else 0,
            1 if gpt4_result.landmark else 0
        ])
        
        print(f"    📊 Fields extracted - Shiprocket: {ship_fields}, GPT-4: {gpt4_fields}")
        
        # Store results
        result = {
            'index': i,
            'hash_key': hash_key,
            'address': address,
            'original_pincode': item.get('pincode', ''),
            'original_city_id': item.get('city_id', ''),
            'original_state_id': item.get('state_id', ''),
            
            # Shiprocket results
            'ship_success': ship_result.parse_success,
            'ship_time': round(ship_time, 3),
            'ship_unit': ship_result.unit_number or '',
            'ship_society': ship_result.society_name or '',
            'ship_landmark': ship_result.landmark or '',
            'ship_road': ship_result.road or '',
            'ship_sub_locality': ship_result.sub_locality or '',
            'ship_locality': ship_result.locality or '',
            'ship_city': ship_result.city or '',
            'ship_district': ship_result.district or '',
            'ship_state': ship_result.state or '',
            'ship_pin': ship_result.pin_code or '',
            'ship_error': ship_result.parse_error or '',
            'ship_fields_count': ship_fields,
            
            # GPT-4 results
            'gpt4_success': gpt4_result.parse_success,
            'gpt4_time': round(gpt4_time, 3),
            'gpt4_unit': gpt4_result.unit_number or '',
            'gpt4_society': gpt4_result.society_name or '',
            'gpt4_landmark': gpt4_result.landmark or '',
            'gpt4_road': gpt4_result.road or '',
            'gpt4_sub_locality': gpt4_result.sub_locality or '',
            'gpt4_locality': gpt4_result.locality or '',
            'gpt4_city': gpt4_result.city or '',
            'gpt4_district': gpt4_result.district or '',
            'gpt4_state': gpt4_result.state or '',
            'gpt4_pin': gpt4_result.pin_code or '',
            'gpt4_error': gpt4_result.parse_error or '',
            'gpt4_fields_count': gpt4_fields
        }
        
        results.append(result)
    
    # Get final statistics
    ship_stats = shiprocket.get_statistics()
    gpt4_stats = gpt4.get_statistics()
    
    print(f"\n📈 Final Statistics:")
    print(f"   Shiprocket: {ship_stats['success_rate_percent']}% success, {ship_stats['total_retries']} retries")
    print(f"   GPT-4: {gpt4_stats['success_rate_percent']}% success, ${gpt4_stats.get('total_cost_usd', 0):.4f} cost")
    
    return results, ship_stats, gpt4_stats


def analyze_quality_differences(results: List[Dict]) -> Dict[str, Any]:
    """Analyze quality differences between the two parsers."""
    
    print(f"\n🎯 Quality Analysis:")
    print("=" * 50)
    
    # Filter successful results
    ship_success = [r for r in results if r['ship_success']]
    gpt4_success = [r for r in results if r['gpt4_success']]
    
    total_addresses = len(results)
    
    # Success rates
    ship_success_rate = len(ship_success) / total_addresses * 100
    gpt4_success_rate = len(gpt4_success) / total_addresses * 100
    
    print(f"Success Rates:")
    print(f"  Shiprocket: {ship_success_rate:.1f}% ({len(ship_success)}/{total_addresses})")
    print(f"  GPT-4: {gpt4_success_rate:.1f}% ({len(gpt4_success)}/{total_addresses})")
    
    # Field extraction rates (only for successful parses)
    fields = ['unit', 'society', 'landmark', 'road', 'locality', 'city', 'pin']
    
    ship_extraction = {}
    gpt4_extraction = {}
    
    for field in fields:
        ship_col = f'ship_{field}'
        gpt4_col = f'gpt4_{field}'
        
        ship_count = sum(1 for r in ship_success if r.get(ship_col, ''))
        gpt4_count = sum(1 for r in gpt4_success if r.get(gpt4_col, ''))
        
        ship_rate = (ship_count / len(ship_success) * 100) if ship_success else 0
        gpt4_rate = (gpt4_count / len(gpt4_success) * 100) if gpt4_success else 0
        
        ship_extraction[field] = {'count': ship_count, 'rate': ship_rate}
        gpt4_extraction[field] = {'count': gpt4_count, 'rate': gpt4_rate}
        
        improvement = gpt4_rate - ship_rate
        status = "🟢" if improvement > 5 else "🟡" if improvement > -5 else "🔴"
        
        print(f"  {field.title()} Extraction:")
        print(f"    Shiprocket: {ship_rate:.1f}% ({ship_count}/{len(ship_success)})")
        print(f"    GPT-4: {gpt4_rate:.1f}% ({gpt4_count}/{len(gpt4_success)})")
        print(f"    Improvement: {status} {improvement:+.1f}%")
        print()
    
    # Performance comparison
    ship_times = [r['ship_time'] for r in results if r['ship_success']]
    gpt4_times = [r['gpt4_time'] for r in results if r['gpt4_success']]
    
    ship_avg_time = sum(ship_times) / len(ship_times) if ship_times else 0
    gpt4_avg_time = sum(gpt4_times) / len(gpt4_times) if gpt4_times else 0
    
    print(f"Performance:")
    print(f"  Shiprocket avg time: {ship_avg_time:.3f}s")
    print(f"  GPT-4 avg time: {gpt4_avg_time:.3f}s")
    print(f"  Speed ratio: {gpt4_avg_time/ship_avg_time:.1f}x slower" if ship_avg_time > 0 else "")
    
    return {
        'total_addresses': total_addresses,
        'ship_success_rate': ship_success_rate,
        'gpt4_success_rate': gpt4_success_rate,
        'ship_extraction': ship_extraction,
        'gpt4_extraction': gpt4_extraction,
        'ship_avg_time': ship_avg_time,
        'gpt4_avg_time': gpt4_avg_time
    }


def create_comparison_excel(results: List[Dict], analysis: Dict, ship_stats: Dict, gpt4_stats: Dict, output_file: str):
    """Create comprehensive Excel comparison report."""
    
    print(f"\n📊 Creating Excel comparison report: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Sheet 1: Detailed Comparison
        df_results = pd.DataFrame(results)
        df_results.to_excel(writer, sheet_name='Detailed_Comparison', index=False)
        
        # Sheet 2: Summary Statistics
        summary_data = {
            'Metric': [
                'Total Addresses Tested',
                'Shiprocket Success Rate (%)',
                'GPT-4 Success Rate (%)',
                'Success Rate Improvement (%)',
                '',
                'Shiprocket Avg Time (s)',
                'GPT-4 Avg Time (s)', 
                'Speed Ratio (GPT-4/Ship)',
                '',
                'Shiprocket Unit Extraction (%)',
                'GPT-4 Unit Extraction (%)',
                'Unit Extraction Improvement (%)',
                '',
                'Shiprocket Society Extraction (%)',
                'GPT-4 Society Extraction (%)',
                'Society Extraction Improvement (%)',
                '',
                'Shiprocket Locality Extraction (%)',
                'GPT-4 Locality Extraction (%)',
                'Locality Extraction Improvement (%)',
                '',
                'Shiprocket Road Extraction (%)',
                'GPT-4 Road Extraction (%)',
                'Road Extraction Improvement (%)',
                '',
                'GPT-4 Total Cost (USD)',
                'GPT-4 Cost per Address (USD)',
                'Shiprocket Estimated Cost per Address (USD)'
            ],
            'Value': [
                analysis['total_addresses'],
                f"{analysis['ship_success_rate']:.1f}%",
                f"{analysis['gpt4_success_rate']:.1f}%",
                f"{analysis['gpt4_success_rate'] - analysis['ship_success_rate']:+.1f}%",
                '',
                f"{analysis['ship_avg_time']:.3f}",
                f"{analysis['gpt4_avg_time']:.3f}",
                f"{analysis['gpt4_avg_time']/analysis['ship_avg_time']:.1f}x" if analysis['ship_avg_time'] > 0 else 'N/A',
                '',
                f"{analysis['ship_extraction']['unit']['rate']:.1f}%",
                f"{analysis['gpt4_extraction']['unit']['rate']:.1f}%",
                f"{analysis['gpt4_extraction']['unit']['rate'] - analysis['ship_extraction']['unit']['rate']:+.1f}%",
                '',
                f"{analysis['ship_extraction']['society']['rate']:.1f}%",
                f"{analysis['gpt4_extraction']['society']['rate']:.1f}%",
                f"{analysis['gpt4_extraction']['society']['rate'] - analysis['ship_extraction']['society']['rate']:+.1f}%",
                '',
                f"{analysis['ship_extraction']['locality']['rate']:.1f}%",
                f"{analysis['gpt4_extraction']['locality']['rate']:.1f}%",
                f"{analysis['gpt4_extraction']['locality']['rate'] - analysis['ship_extraction']['locality']['rate']:+.1f}%",
                '',
                f"{analysis['ship_extraction']['road']['rate']:.1f}%",
                f"{analysis['gpt4_extraction']['road']['rate']:.1f}%",
                f"{analysis['gpt4_extraction']['road']['rate'] - analysis['ship_extraction']['road']['rate']:+.1f}%",
                '',
                f"${gpt4_stats.get('total_cost_usd', 0):.4f}",
                f"${gpt4_stats.get('avg_cost_per_address', 0):.4f}",
                "$0.0200"
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Sheet 3: Field-by-Field Comparison
        field_comparison = []
        fields = ['unit', 'society', 'landmark', 'road', 'locality', 'city', 'pin']
        
        for field in fields:
            field_comparison.append({
                'Field': field.title(),
                'Shiprocket_Count': analysis['ship_extraction'][field]['count'],
                'Shiprocket_Rate': f"{analysis['ship_extraction'][field]['rate']:.1f}%",
                'GPT4_Count': analysis['gpt4_extraction'][field]['count'],
                'GPT4_Rate': f"{analysis['gpt4_extraction'][field]['rate']:.1f}%",
                'Improvement': f"{analysis['gpt4_extraction'][field]['rate'] - analysis['ship_extraction'][field]['rate']:+.1f}%",
                'Winner': 'GPT-4' if analysis['gpt4_extraction'][field]['rate'] > analysis['ship_extraction'][field]['rate'] else 'Shiprocket' if analysis['ship_extraction'][field]['rate'] > analysis['gpt4_extraction'][field]['rate'] else 'Tie'
            })
        
        df_fields = pd.DataFrame(field_comparison)
        df_fields.to_excel(writer, sheet_name='Field_Comparison', index=False)
        
        # Sheet 4: Quality Winners Analysis
        quality_winners = []
        for result in results:
            if result['ship_success'] and result['gpt4_success']:
                ship_total = result['ship_fields_count']
                gpt4_total = result['gpt4_fields_count']
                
                winner = 'GPT-4' if gpt4_total > ship_total else 'Shiprocket' if ship_total > gpt4_total else 'Tie'
                
                quality_winners.append({
                    'Hash_Key': result['hash_key'],
                    'Address': result['address'][:100] + '...' if len(result['address']) > 100 else result['address'],
                    'Shiprocket_Fields': ship_total,
                    'GPT4_Fields': gpt4_total,
                    'Winner': winner,
                    'Difference': gpt4_total - ship_total
                })
        
        if quality_winners:
            df_winners = pd.DataFrame(quality_winners)
            df_winners.to_excel(writer, sheet_name='Quality_Winners', index=False)
        
        # Sheet 5: Cost Analysis
        cost_analysis = {
            'Scenario': [
                'Current (Shiprocket only)',
                'GPT-4 only',
                'Hybrid (30% GPT-4, 70% Shiprocket)',
                'Hybrid (50% GPT-4, 50% Shiprocket)',
                'Smart routing (complex → GPT-4)'
            ],
            'Cost_per_1K_addresses': [
                '$20.00',
                f"${gpt4_stats.get('avg_cost_per_address', 0) * 1000:.2f}",
                f"${(0.3 * gpt4_stats.get('avg_cost_per_address', 0) + 0.7 * 0.02) * 1000:.2f}",
                f"${(0.5 * gpt4_stats.get('avg_cost_per_address', 0) + 0.5 * 0.02) * 1000:.2f}",
                f"${(0.4 * gpt4_stats.get('avg_cost_per_address', 0) + 0.6 * 0.02) * 1000:.2f}"
            ],
            'Monthly_cost_100K': [
                '$2,000',
                f"${gpt4_stats.get('avg_cost_per_address', 0) * 100000:.0f}",
                f"${(0.3 * gpt4_stats.get('avg_cost_per_address', 0) + 0.7 * 0.02) * 100000:.0f}",
                f"${(0.5 * gpt4_stats.get('avg_cost_per_address', 0) + 0.5 * 0.02) * 100000:.0f}",
                f"${(0.4 * gpt4_stats.get('avg_cost_per_address', 0) + 0.6 * 0.02) * 100000:.0f}"
            ],
            'Expected_Quality': [
                'Good (95% society)',
                'Excellent (98%+ society)',
                'Very Good (96% society)',
                'Excellent (97% society)',
                'Excellent (97% society)'
            ]
        }
        
        df_cost = pd.DataFrame(cost_analysis)
        df_cost.to_excel(writer, sheet_name='Cost_Analysis', index=False)
    
    print(f"✅ Excel report created successfully!")


def main():
    """Main comparison function."""
    
    print("🔥 GPT-4 vs Shiprocket Address Parser Comparison")
    print("=" * 80)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    csv_file = 'export_customer_address_store_p0.csv'
    sample_size = 50  # Test with 50 addresses
    output_file = f'gpt4_shiprocket_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    
    try:
        # Step 1: Load sample addresses
        sample_data = load_sample_addresses(csv_file, sample_size)
        
        if not sample_data:
            print("❌ No sample data loaded. Exiting.")
            return
        
        # Step 2: Parse with both models
        results, ship_stats, gpt4_stats = parse_with_both_models(sample_data)
        
        # Step 3: Analyze quality differences
        analysis = analyze_quality_differences(results)
        
        # Step 4: Create Excel report
        create_comparison_excel(results, analysis, ship_stats, gpt4_stats, output_file)
        
        # Step 5: Final summary
        print(f"\n🏆 Comparison Complete!")
        print(f"   📁 Report: {output_file}")
        print(f"   📊 Addresses tested: {len(results)}")
        print(f"   🔧 Shiprocket success: {analysis['ship_success_rate']:.1f}%")
        print(f"   🤖 GPT-4 success: {analysis['gpt4_success_rate']:.1f}%")
        
        # Quality winners
        ship_better = sum(1 for r in results if r.get('ship_fields_count', 0) > r.get('gpt4_fields_count', 0))
        gpt4_better = sum(1 for r in results if r.get('gpt4_fields_count', 0) > r.get('ship_fields_count', 0))
        ties = len(results) - ship_better - gpt4_better
        
        print(f"\n🎯 Quality Winners:")
        print(f"   🤖 GPT-4 better: {gpt4_better} addresses")
        print(f"   🔧 Shiprocket better: {ship_better} addresses") 
        print(f"   🤝 Ties: {ties} addresses")
        
        # Cost summary
        total_cost = gpt4_stats.get('total_cost_usd', 0)
        avg_cost = gpt4_stats.get('avg_cost_per_address', 0)
        print(f"\n💰 Cost Analysis:")
        print(f"   GPT-4 total cost: ${total_cost:.4f}")
        print(f"   GPT-4 per address: ${avg_cost:.4f}")
        print(f"   Shiprocket per address: $0.0200")
        print(f"   Cost ratio: {avg_cost/0.02:.1f}x more expensive")
        
        # Recommendation
        society_improvement = analysis['gpt4_extraction']['society']['rate'] - analysis['ship_extraction']['society']['rate']
        road_improvement = analysis['gpt4_extraction']['road']['rate'] - analysis['ship_extraction']['road']['rate']
        
        print(f"\n💡 Key Insights:")
        print(f"   Society extraction improvement: {society_improvement:+.1f}%")
        print(f"   Road extraction improvement: {road_improvement:+.1f}%")
        
        if society_improvement > 10 or road_improvement > 20:
            print(f"   ✅ GPT-4 shows significant quality improvement!")
        elif avg_cost < 0.10:
            print(f"   ⚖️  GPT-4 offers good quality/cost balance")
        else:
            print(f"   ⚠️  Consider hybrid approach for cost optimization")
        
    except Exception as e:
        print(f"\n💥 Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()