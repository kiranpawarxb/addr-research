print("Starting BallTree clustering test...")

# Test imports first
try:
    print("Importing pandas...")
    import pandas as pd
    print("✓ pandas imported")
    
    print("Importing numpy...")
    import numpy as np
    print("✓ numpy imported")
    
    print("Importing sklearn...")
    from sklearn.neighbors import BallTree
    print("✓ sklearn BallTree imported")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install pandas numpy scikit-learn")
    exit(1)

print("All dependencies available!")

# Test with small sample first
print("Testing BallTree with sample data...")

# Create sample coordinates (Pune area)
sample_coords = np.array([
    [18.5204, 73.8567],  # Pune center
    [18.5214, 73.8577],  # 100m away
    [18.5224, 73.8587],  # 200m away
    [18.5304, 73.8667],  # 1km away
])

print(f"Sample coordinates: {len(sample_coords)} points")

# Convert to radians for BallTree
coords_rad = np.radians(sample_coords)
print("Coordinates converted to radians")

# Build BallTree
tree = BallTree(coords_rad, metric='haversine')
print("✓ BallTree built successfully")

# Test radius query (300m = 300/6371000 radians)
radius_rad = 300 / 6371000
indices = tree.query_radius(coords_rad[0:1], r=radius_rad)[0]
print(f"✓ Radius query successful: found {len(indices)} points within 300m")

print("BallTree test completed successfully!")
print("Now testing CSV loading...")

# Test CSV loading
try:
    print("Loading CSV file...")
    df = pd.read_csv('export_customer_address_store_p0.csv', nrows=100)  # Just first 100 rows
    print(f"✓ CSV loaded: {len(df)} rows (sample)")
    print(f"Columns: {list(df.columns)}")
    
    # Find geo column
    geo_col = None
    for col in df.columns:
        if 'geo' in col.lower():
            geo_col = col
            break
    
    if geo_col:
        print(f"✓ Found geo column: {geo_col}")
        sample_geo = df[geo_col].iloc[0]
        print(f"Sample geo data: {str(sample_geo)[:100]}...")
    else:
        print("❌ No geo column found")
        
except Exception as e:
    print(f"❌ CSV loading error: {e}")

print("Test completed!")