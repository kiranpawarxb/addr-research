"""Manual test of CSVReader with the actual CSV file."""

import logging
from src.csv_reader import CSVReader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Test with the actual CSV file
reader = CSVReader(
    'export_customer_address_store_p0.csv',
    ['addr_text', 'pincode', 'city_id']
)

# Validate columns
is_valid, missing = reader.validate_columns()
print(f"Column validation: valid={is_valid}, missing={missing}")

# Read first 5 records
print("\nReading first 5 records:")
count = 0
for record in reader.read():
    count += 1
    print(f"\nRecord {count}:")
    print(f"  Hash: {record.addr_hash_key}")
    print(f"  Address: {record.addr_text}")
    print(f"  City ID: {record.city_id}")
    print(f"  Pincode: {record.pincode}")
    print(f"  Geo Points Count: {record.assigned_pickup_dlvd_geo_points_count}")
    
    if count >= 5:
        break

print(f"\nTotal loaded so far: {reader.get_total_loaded()}")
print(f"Malformed rows: {reader.get_malformed_count()}")
