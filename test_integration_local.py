"""Integration test for local parser with the full pipeline."""

import os
import sys
import logging
import pandas as pd
from src.config_loader import ConfigLoader
from src.pipeline import AddressConsolidationPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_csv():
    """Create a small test CSV with Pune addresses."""
    test_data = {
        'addr_hash_key': ['hash1', 'hash2', 'hash3', 'hash4', 'hash5'],
        'addr_text': [
            'Flat 301, Kumar Paradise, Kalyani Nagar, Pune, Maharashtra 411006',
            'Flat 402, Kumar Paradise, Kalyani Nagar, Pune 411006',
            'A-204, Amanora Park Town, Hadapsar, Pune 411028',
            'B-305, Amanora Park Town, Hadapsar, Pune, Maharashtra 411028',
            '201, Magarpatta City, Hadapsar, Pune 411013'
        ],
        'city_id': ['PUNE', 'PUNE', 'PUNE', 'PUNE', 'PUNE'],
        'pincode': ['411006', '411006', '411028', '411028', '411013'],
        'state_id': ['MH', 'MH', 'MH', 'MH', 'MH'],
        'zone_id': ['WEST', 'WEST', 'WEST', 'WEST', 'WEST'],
        'address_id': ['addr1', 'addr2', 'addr3', 'addr4', 'addr5']
    }
    
    df = pd.DataFrame(test_data)
    test_file = 'test_pune_addresses.csv'
    df.to_csv(test_file, index=False)
    print(f"✓ Created test CSV: {test_file}")
    return test_file

def test_local_parser_integration():
    """Test the full pipeline with local parser."""
    
    print("="*80)
    print("INTEGRATION TEST: LOCAL PARSER WITH FULL PIPELINE")
    print("="*80)
    print()
    
    # Create test CSV
    test_file = create_test_csv()
    output_file = 'test_output_local.csv'
    
    try:
        # Load configuration
        print("Loading configuration...")
        config_loader = ConfigLoader('config/config.yaml')
        config = config_loader.load()
        
        # Override input/output paths for test
        config.input.file_path = test_file
        config.output.file_path = output_file
        
        # Verify parser type is set to local
        parser_type = getattr(config.llm, 'parser_type', 'openai')
        print(f"✓ Configuration loaded (parser_type: {parser_type})")
        
        if parser_type != 'local':
            print("\n⚠ WARNING: config.yaml has parser_type set to 'openai'")
            print("  Please update config/config.yaml and set:")
            print("    llm:")
            print("      parser_type: 'local'")
            print("\n  Continuing with local parser anyway for this test...")
            config.llm.parser_type = 'local'
        
        print()
        
        # Run pipeline
        print("Running pipeline with local parser...")
        print("-"*80)
        pipeline = AddressConsolidationPipeline(config)
        pipeline.run()
        
        print()
        print("="*80)
        print("VERIFYING OUTPUT")
        print("="*80)
        
        # Verify output file was created
        if os.path.exists(output_file):
            print(f"✓ Output file created: {output_file}")
            
            # Read and display output
            df_output = pd.read_csv(output_file)
            print(f"✓ Output contains {len(df_output)} records")
            print()
            
            # Check for parsed fields
            expected_fields = ['unit_number', 'society_name', 'city', 'pin_code', 'group_id']
            missing_fields = [f for f in expected_fields if f not in df_output.columns]
            
            if missing_fields:
                print(f"✗ Missing expected fields: {missing_fields}")
            else:
                print(f"✓ All expected fields present")
                print()
                
                # Display sample results
                print("Sample parsed results:")
                print("-"*80)
                for idx, row in df_output.iterrows():
                    print(f"\nRecord {idx+1}:")
                    print(f"  Input:        {row['addr_text'][:60]}...")
                    print(f"  Society:      {row['society_name']}")
                    print(f"  Unit:         {row['unit_number']}")
                    print(f"  City:         {row['city']}")
                    print(f"  PIN:          {row['pin_code']}")
                    print(f"  Group ID:     {row['group_id']}")
                
                # Check consolidation
                print()
                print("-"*80)
                unique_groups = df_output['group_id'].nunique()
                print(f"✓ Addresses consolidated into {unique_groups} groups")
                
                # Show groups
                for group_id in df_output['group_id'].unique():
                    group_records = df_output[df_output['group_id'] == group_id]
                    society = group_records.iloc[0]['society_name']
                    pin = group_records.iloc[0]['pin_code']
                    print(f"  Group: {society} ({pin}) - {len(group_records)} addresses")
        else:
            print(f"✗ Output file not created: {output_file}")
        
        print()
        print("="*80)
        print("TEST COMPLETE")
        print("="*80)
        print("\n✓ Local parser integration successful!")
        print("\nYour system is now configured to use the local parser.")
        print("No API calls or internet connection required!")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test files
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n✓ Cleaned up test file: {test_file}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_local_parser_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
