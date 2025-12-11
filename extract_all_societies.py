#!/usr/bin/env python3
"""
Extract all unique society names from the Shiprocket parsing results.
"""

import pandas as pd
import csv

def extract_all_societies():
    """Extract all unique society names from the parsed addresses."""
    
    print("🏢 Extracting All Society Names from Dataset")
    print("=" * 60)
    
    try:
        # Read the main parsed addresses file
        print("📂 Loading parsed addresses file...")
        
        # Since the file is large, we'll read it in chunks
        chunk_size = 10000
        all_societies = set()
        
        # Read the CSV in chunks to handle large file
        for chunk_num, chunk in enumerate(pd.read_csv('shiprocket_parsed_addresses_20251210_123813.csv', chunksize=chunk_size)):
            print(f"   Processing chunk {chunk_num + 1}...")
            
            # Extract society names from this chunk
            societies_in_chunk = chunk['society_name'].dropna()
            societies_in_chunk = societies_in_chunk[societies_in_chunk.str.strip() != '']
            
            # Add to our set (automatically handles duplicates)
            for society in societies_in_chunk:
                if society and society.strip():
                    all_societies.add(society.strip())
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Found {len(all_societies):,} unique society names")
        
        # Convert to sorted list
        sorted_societies = sorted(list(all_societies))
        
        # Save to file
        output_file = 'all_unique_societies.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for society in sorted_societies:
                f.write(f"{society}\n")
        
        print(f"💾 Saved all societies to: {output_file}")
        
        # Also create a CSV version
        csv_output = 'all_unique_societies.csv'
        with open(csv_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['society_name'])
            for society in sorted_societies:
                writer.writerow([society])
        
        print(f"💾 Saved CSV version to: {csv_output}")
        
        # Show some statistics
        print(f"\n📈 SOCIETY STATISTICS:")
        print(f"   Total unique societies: {len(sorted_societies):,}")
        
        # Show first 20 societies as sample
        print(f"\n🔝 FIRST 20 SOCIETIES (Alphabetically):")
        print("-" * 60)
        for i, society in enumerate(sorted_societies[:20], 1):
            print(f"{i:2d}. {society}")
        
        if len(sorted_societies) > 20:
            print(f"    ... and {len(sorted_societies) - 20:,} more societies")
        
        # Show some interesting statistics
        print(f"\n📊 INTERESTING STATISTICS:")
        
        # Count societies with certain keywords
        apartment_count = sum(1 for s in sorted_societies if 'apartment' in s.lower())
        society_count = sum(1 for s in sorted_societies if 'society' in s.lower())
        complex_count = sum(1 for s in sorted_societies if 'complex' in s.lower())
        tower_count = sum(1 for s in sorted_societies if 'tower' in s.lower())
        residency_count = sum(1 for s in sorted_societies if 'residency' in s.lower())
        
        print(f"   • Societies with 'Apartment': {apartment_count:,}")
        print(f"   • Societies with 'Society': {society_count:,}")
        print(f"   • Societies with 'Complex': {complex_count:,}")
        print(f"   • Societies with 'Tower': {tower_count:,}")
        print(f"   • Societies with 'Residency': {residency_count:,}")
        
        return sorted_societies
        
    except FileNotFoundError:
        print("❌ Error: shiprocket_parsed_addresses_20251210_123813.csv not found")
        print("   Make sure you're in the correct directory with the parsed results")
        return []
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return []

if __name__ == "__main__":
    societies = extract_all_societies()