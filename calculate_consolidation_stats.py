#!/usr/bin/env python3
"""
Calculate exact consolidation statistics from the Shiprocket parsing results.
"""

import pandas as pd

def calculate_consolidation_stats():
    """Calculate consolidation percentages from the summary data."""
    
    print("📊 Calculating Address Consolidation Statistics")
    print("=" * 60)
    
    # Read the summary CSV
    summary_file = 'pincode_locality_society_summary_20251210_123813.csv'
    df = pd.read_csv(summary_file)
    
    # Calculate totals
    total_addresses = df['total_addresses'].sum()
    total_localities = df['distinct_localities_count'].sum()
    total_societies = df['distinct_societies_count'].sum()
    total_pincodes = len(df)
    
    print(f"📈 CONSOLIDATION ANALYSIS:")
    print(f"   Original Individual Addresses: {total_addresses:,}")
    print(f"   Consolidated to Societies: {total_societies:,}")
    print(f"   Consolidated to Localities: {total_localities:,}")
    print(f"   Consolidated to PIN Codes: {total_pincodes:,}")
    print()
    
    # Calculate consolidation percentages
    society_consolidation = ((total_addresses - total_societies) / total_addresses) * 100
    locality_consolidation = ((total_addresses - total_localities) / total_addresses) * 100
    pincode_consolidation = ((total_addresses - total_pincodes) / total_addresses) * 100
    
    print(f"🏢 SOCIETY/BUILDING LEVEL CONSOLIDATION:")
    print(f"   From {total_addresses:,} individual addresses")
    print(f"   To {total_societies:,} distinct societies/buildings")
    print(f"   Consolidation: {society_consolidation:.1f}%")
    print(f"   Reduction Factor: {total_addresses/total_societies:.1f}x")
    print()
    
    print(f"🏘️  LOCALITY LEVEL CONSOLIDATION:")
    print(f"   From {total_addresses:,} individual addresses")
    print(f"   To {total_localities:,} distinct localities")
    print(f"   Consolidation: {locality_consolidation:.1f}%")
    print(f"   Reduction Factor: {total_addresses/total_localities:.1f}x")
    print()
    
    print(f"📍 PIN CODE LEVEL CONSOLIDATION:")
    print(f"   From {total_addresses:,} individual addresses")
    print(f"   To {total_pincodes:,} distinct PIN codes")
    print(f"   Consolidation: {pincode_consolidation:.1f}%")
    print(f"   Reduction Factor: {total_addresses/total_pincodes:.1f}x")
    print()
    
    # Top PIN codes analysis
    print(f"🔝 TOP 5 PIN CODES BY VOLUME:")
    print("-" * 60)
    top_5 = df.nlargest(5, 'total_addresses')
    
    for idx, row in top_5.iterrows():
        pin_code = row['pin_code']
        addresses = row['total_addresses']
        societies = row['distinct_societies_count']
        localities = row['distinct_localities_count']
        
        society_reduction = ((addresses - societies) / addresses) * 100
        locality_reduction = ((addresses - localities) / addresses) * 100
        
        print(f"PIN {pin_code}:")
        print(f"  • {addresses:,} addresses → {societies:,} societies ({society_reduction:.1f}% reduction)")
        print(f"  • {addresses:,} addresses → {localities:,} localities ({locality_reduction:.1f}% reduction)")
        print()
    
    # Summary for delivery optimization
    print(f"📦 DELIVERY OPTIMIZATION IMPACT:")
    print("-" * 60)
    print(f"Instead of delivering to {total_addresses:,} individual addresses:")
    print(f"• Route by societies: {total_societies:,} stops ({society_consolidation:.1f}% fewer stops)")
    print(f"• Route by localities: {total_localities:,} stops ({locality_consolidation:.1f}% fewer stops)")
    print(f"• Route by PIN codes: {total_pincodes:,} stops ({pincode_consolidation:.1f}% fewer stops)")
    print()
    
    # Average consolidation per PIN code
    avg_addresses_per_pin = total_addresses / total_pincodes
    avg_societies_per_pin = total_societies / total_pincodes
    avg_localities_per_pin = total_localities / total_pincodes
    
    print(f"📊 AVERAGE CONSOLIDATION PER PIN CODE:")
    print(f"   • {avg_addresses_per_pin:.0f} addresses per PIN code")
    print(f"   • {avg_societies_per_pin:.0f} societies per PIN code")
    print(f"   • {avg_localities_per_pin:.0f} localities per PIN code")

if __name__ == "__main__":
    calculate_consolidation_stats()