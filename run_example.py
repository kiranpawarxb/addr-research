#!/usr/bin/env python3
"""
Example script showing how to use the Sustained GPU Maximizer
with sample data or your own CSV files.
"""

import os
import pandas as pd
from sustained_gpu_maximizer import SustainedGPUMaximizer, setup_logging

def create_sample_data():
    """Create a sample CSV file for testing."""
    sample_addresses = [
        "Flat 301, Sunrise Apartments, Near City Mall, MG Road, Koramangala, Bangalore, Karnataka 560034",
        "House No 45, Green Valley Society, Opposite Metro Station, Sector 18, Noida, Uttar Pradesh 201301",
        "B-204, Royal Heights, Behind Big Bazaar, Andheri West, Mumbai, Maharashtra 400058",
        "Villa 12, Palm Grove, Near International Airport, Whitefield, Bangalore, Karnataka 560066",
        "Apartment 5A, Silver Oak Complex, Next to Hospital, Banjara Hills, Hyderabad, Telangana 500034",
        "Shop 23, Commercial Complex, Main Market, Lajpat Nagar, New Delhi, Delhi 110024",
        "Flat 102, Ocean View Towers, Marine Drive, Kochi, Kerala 682031",
        "House 67, Shanti Nagar, Near Railway Station, Jaipur, Rajasthan 302006",
        "Office 401, Tech Park, IT Corridor, Gachibowli, Hyderabad, Telangana 500032",
        "Bungalow 8, Elite Residency, Golf Course Road, Gurgaon, Haryana 122002"
    ]
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'id': range(1, len(sample_addresses) + 1),
        'addr_text': sample_addresses,
        'city_id': ['BLR', 'NOI', 'MUM', 'BLR', 'HYD', 'DEL', 'COK', 'JAI', 'HYD', 'GUR'],
        'pincode': ['560034', '201301', '400058', '560066', '500034', '110024', '682031', '302006', '500032', '122002']
    })
    
    # Save to CSV
    sample_file = 'sample_addresses.csv'
    df.to_csv(sample_file, index=False)
    print(f"✅ Created sample file: {sample_file}")
    return sample_file

def run_example():
    """Run the sustained GPU maximizer example."""
    
    # Setup logging
    setup_logging()
    
    print("🚀 SUSTAINED GPU MAXIMIZER - EXAMPLE RUN")
    print("=" * 50)
    
    # Check if we have any CSV files to process
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'addr' in f.lower()]
    
    if not csv_files:
        print("📝 No address CSV files found. Creating sample data...")
        sample_file = create_sample_data()
        csv_files = [sample_file]
    
    print(f"📂 Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"   - {file}")
    
    # Initialize the processor
    processor = SustainedGPUMaximizer(
        batch_size=50,  # Smaller batch for example
        use_nvidia_gpu=True,
        use_intel_gpu=False,
        use_all_cpu_cores=False,
        cpu_core_multiplier=0.1
    )
    
    print("\n🔧 Processor Configuration:")
    print(f"   - Batch Size: {processor.nvidia_batch_size}")
    print(f"   - GPU Streams: {processor.num_gpu_streams}")
    print(f"   - Queue Size: {processor.gpu_queue_size}")
    print(f"   - NVIDIA GPU: {processor.use_nvidia_gpu}")
    
    # Process each file
    results = []
    for csv_file in csv_files:
        print(f"\n📊 Processing: {csv_file}")
        print("-" * 40)
        
        try:
            result = processor.process_single_file_sustained(csv_file)
            if result:
                results.append(result)
                print(f"✅ Success! Processed {result['total_addresses']} addresses")
                print(f"   Speed: {result['addresses_per_second']:.1f} addr/sec")
                print(f"   Success Rate: {(result['success_count']/result['total_addresses']*100):.1f}%")
                print(f"   Output: {result['output_file']}")
                print(f"   GPU Utilization: {result['gpu_utilization']:.1f}%")
            else:
                print(f"❌ Failed to process {csv_file}")
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
    
    # Summary
    print("\n🎉 EXAMPLE COMPLETE!")
    print("=" * 50)
    
    if results:
        total_addresses = sum(r['total_addresses'] for r in results)
        total_success = sum(r['success_count'] for r in results)
        avg_speed = sum(r['addresses_per_second'] for r in results) / len(results)
        
        print(f"📊 Summary:")
        print(f"   Files Processed: {len(results)}")
        print(f"   Total Addresses: {total_addresses:,}")
        print(f"   Success Rate: {(total_success/total_addresses*100):.1f}%")
        print(f"   Average Speed: {avg_speed:.1f} addresses/second")
        
        print(f"\n📁 Output Files:")
        for result in results:
            print(f"   - {result['output_file']}")
    
    print(f"\n💡 Tips:")
    print(f"   - Place your CSV files in this directory")
    print(f"   - Ensure address column is named 'addr_text', 'address', or similar")
    print(f"   - Monitor GPU usage with: nvidia-smi")
    print(f"   - Check log files for detailed processing information")

if __name__ == "__main__":
    run_example()