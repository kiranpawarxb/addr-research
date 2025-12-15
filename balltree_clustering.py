import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import math
import time
import re
from datetime import datetime, timedelta

print("=" * 80)
print("BALLTREE CIRCULAR CLUSTERING - OPTIMIZED FOR GEOGRAPHIC DATA")
print("=" * 80)

def log_progress(stage, message, progress_pct=None):
    """Log progress with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {stage}: {message}"
    
    if progress_pct is not None:
        log_entry += f" ({progress_pct:.1f}%)"
        
    print(log_entry)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters"""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def extract_coordinates_robust(multipoint_str):
    """Extract latitude and longitude from MULTIPOINT string"""
    if pd.isna(multipoint_str) or not isinstance(multipoint_str, str) or multipoint_str.strip() == '':
        return None, None
    
    # Clean the string
    clean_str = ' '.join(multipoint_str.split())
    
    # Multiple patterns to handle different formats
    patterns = [
        r'MULTIPOINT\s*\(\s*\(\s*([0-9.-]+)\s+([0-9.-]+)\s*\)\s*\)',
        r'MULTIPOINT\s*\(\s*([0-9.-]+)\s+([0-9.-]+)\s*\)',
        r'\(\s*([0-9.-]+)\s+([0-9.-]+)\s*\)',
        r'([0-9.-]+)\s+([0-9.-]+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, clean_str)
        if matches:
            try:
                lon, lat = float(matches[0][0]), float(matches[0][1])
                # Validate coordinates for India
                if 6 <= lat <= 38 and 68 <= lon <= 98:
                    return lat, lon
            except (ValueError, IndexError):
                continue
    
    return None, None

def max_distance_in_group_fast(coordinates):
    """Fast maximum distance calculation using vectorized operations"""
    if len(coordinates) <= 1:
        return 0
    
    coords_array = np.array(coordinates)
    
    # Convert to radians for haversine
    coords_rad = np.radians(coords_array)
    
    # Vectorized haversine calculation
    lat1 = coords_rad[:, 0][:, np.newaxis]
    lon1 = coords_rad[:, 1][:, np.newaxis]
    lat2 = coords_rad[:, 0]
    lon2 = coords_rad[:, 1]
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    distances = 6371000 * c  # Earth radius in meters
    
    # Return maximum distance
    return np.max(distances)

def load_data_with_pandas(filename='addr-research/export_customer_address_store_p0.csv'):
    """Load data using pandas for proper CSV handling"""
    log_progress("DATA_LOADING", f"Loading from {filename} using pandas")
    
    try:
        # Read CSV with pandas
        log_progress("DATA_LOADING", "Reading CSV file...")
        df = pd.read_csv(filename, encoding='utf-8', on_bad_lines='skip')
        
        log_progress("DATA_LOADING", f"Loaded {len(df):,} rows from CSV")
        
        # Extract coordinates
        log_progress("DATA_LOADING", "Extracting coordinates from geocode column")
        
        addresses = []
        valid_count = 0
        invalid_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Find geocode column
                geo_col = None
                for col in df.columns:
                    if 'geo_points' in col.lower() or 'multipoint' in str(row.get(col, '')).upper():
                        geo_col = col
                        break
                
                if geo_col is None and len(df.columns) > 7:
                    geo_col = df.columns[7]
                
                geo_points = row.get(geo_col, '') if geo_col else ''
                
                # Extract coordinates
                lat, lon = extract_coordinates_robust(geo_points)
                
                if lat is not None and lon is not None:
                    addresses.append({
                        'id': row.get('address_id', f'addr_{idx}'),
                        'latitude': lat,
                        'longitude': lon,
                        'address': str(row.get('addr_text', 'Unknown'))[:100],
                        'pincode': str(row.get('pincode', '000000')),
                        'addr_hash_key': str(row.get('addr_hash_key', '')),
                        'original_row': idx + 1
                    })
                    valid_count += 1
                else:
                    invalid_count += 1
                
            except Exception as e:
                invalid_count += 1
            
            # Progress update
            if (idx + 1) % 20000 == 0:
                progress = ((idx + 1) / len(df)) * 100
                log_progress("DATA_LOADING", 
                    f"Processed {idx+1:,}/{len(df):,} rows, valid: {valid_count:,}, invalid: {invalid_count:,}", progress)
        
        log_progress("DATA_LOADING", f"Extraction complete: {valid_count:,} valid addresses from {len(df):,} rows")
        log_progress("DATA_LOADING", f"Success rate: {(valid_count/len(df)*100):.1f}%")
        
        return addresses
        
    except Exception as e:
        log_progress("ERROR", f"Failed to load data: {str(e)}")
        return []

def balltree_circular_clustering(addresses, max_diameter_m=300, min_points=3):
    """
    BallTree-based circular clustering - O(n log n) complexity
    """
    log_progress("CLUSTERING", f"Starting BallTree clustering (diameter: {max_diameter_m}m, min points: {min_points})")
    start_time = time.time()
    
    # Convert addresses to coordinate array
    log_progress("CLUSTERING", "Converting coordinates for BallTree")
    coordinates = np.array([[addr['latitude'], addr['longitude']] for addr in addresses])
    
    # Convert coordinates to radians for haversine metric
    coordinates_rad = np.radians(coordinates)
    
    # Build BallTree with haversine metric (great for geographic data)
    log_progress("CLUSTERING", f"Building BallTree index for {len(addresses):,} points")
    tree_start = time.time()
    
    tree = BallTree(coordinates_rad, metric='haversine')
    
    tree_time = time.time() - tree_start
    log_progress("CLUSTERING", f"BallTree built in {tree_time:.2f} seconds")
    
    # Convert diameter to radians for BallTree queries
    max_radius_rad = (max_diameter_m / 2) / 6371000  # Earth radius in meters
    
    clusters = []
    assigned = [False] * len(addresses)
    cluster_id = 0
    
    log_progress("CLUSTERING", f"Finding circular clusters using BallTree radius queries")
    
    for i in range(len(addresses)):
        if assigned[i]:
            continue
        
        # Query BallTree for all points within radius
        # This is O(log n) per query!
        indices = tree.query_radius(coordinates_rad[i:i+1], r=max_radius_rad)[0]
        
        # Filter out already assigned points
        candidate_indices = [idx for idx in indices if not assigned[idx]]
        
        if len(candidate_indices) < min_points:
            continue
        
        # Get coordinates for distance check
        candidate_coords = [coordinates[idx] for idx in candidate_indices]
        
        # Fast vectorized max distance calculation
        max_dist = max_distance_in_group_fast(candidate_coords)
        
        if max_dist <= max_diameter_m:
            # Create cluster
            for idx in candidate_indices:
                assigned[idx] = True
                addresses[idx]['cluster_id'] = cluster_id
            
            # Calculate cluster center
            center_lat = np.mean([coordinates[idx][0] for idx in candidate_indices])
            center_lon = np.mean([coordinates[idx][1] for idx in candidate_indices])
            
            clusters.append({
                'cluster_id': cluster_id,
                'size': len(candidate_indices),
                'points': candidate_indices,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'max_distance_m': max_dist
            })
            
            if len(clusters) % 1000 == 0:
                log_progress("CLUSTERING", f"Created {len(clusters):,} clusters")
            
            cluster_id += 1
        
        # Progress update
        if (i + 1) % 10000 == 0:
            progress = ((i + 1) / len(addresses)) * 100
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(addresses) - i - 1) / rate
            eta_str = str(timedelta(seconds=int(eta)))
            log_progress("CLUSTERING", 
                f"Processed {i+1:,}/{len(addresses):,} addresses ({rate:.0f}/sec), ETA: {eta_str}", progress)
    
    # Mark unassigned points as noise
    for i in range(len(addresses)):
        if not assigned[i]:
            addresses[i]['cluster_id'] = -1
    
    elapsed = time.time() - start_time
    log_progress("CLUSTERING", f"BallTree clustering complete: {len(clusters):,} clusters in {elapsed:.2f} seconds")
    
    # Verify clusters
    log_progress("VERIFICATION", "Verifying diameter constraints")
    violations = 0
    for cluster in clusters:
        if cluster['max_distance_m'] > max_diameter_m:
            violations += 1
    
    if violations == 0:
        log_progress("VERIFICATION", f"✓ All {len(clusters):,} clusters verified within {max_diameter_m}m diameter")
    else:
        log_progress("WARNING", f"⚠ {violations} clusters exceed diameter constraint")
    
    return clusters

def save_results(addresses, clusters, filename='balltree_clustered.csv'):
    """Save results to CSV"""
    log_progress("OUTPUT", f"Saving {len(addresses):,} addresses to {filename}")
    
    # Create cluster info lookup
    cluster_info = {}
    for cluster in clusters:
        cluster_info[cluster['cluster_id']] = {
            'size': cluster['size'],
            'center_lat': cluster['center_lat'],
            'center_lon': cluster['center_lon'],
            'max_distance_m': cluster['max_distance_m']
        }
    
    # Prepare data for DataFrame
    data_for_df = []
    for addr in addresses:
        cluster_id = addr['cluster_id']
        
        if cluster_id != -1 and cluster_id in cluster_info:
            info = cluster_info[cluster_id]
            cluster_size = info['size']
            center_lat = info['center_lat']
            center_lon = info['center_lon']
            max_distance_m = info['max_distance_m']
        else:
            cluster_size = 0
            center_lat = None
            center_lon = None
            max_distance_m = None
        
        data_for_df.append({
            'id': addr['id'],
            'addr_hash_key': addr['addr_hash_key'],
            'address': addr['address'],
            'latitude': addr['latitude'],
            'longitude': addr['longitude'],
            'pincode': addr['pincode'],
            'cluster_id': cluster_id,
            'cluster_size': cluster_size,
            'cluster_center_lat': center_lat,
            'cluster_center_lon': center_lon,
            'cluster_max_distance_m': max_distance_m,
            'original_row': addr['original_row']
        })
    
    # Save using pandas for efficiency
    df = pd.DataFrame(data_for_df)
    df.to_csv(filename, index=False)
    
    log_progress("OUTPUT", f"Results saved to {filename}")

# Main execution
if __name__ == "__main__":
    overall_start = time.time()
    
    log_progress("INITIALIZATION", "Starting BallTree-optimized circular clustering")
    
    # Load data
    addresses = load_data_with_pandas('addr-research/export_customer_address_store_p0.csv')
    
    if not addresses:
        log_progress("ERROR", "No valid addresses loaded. Exiting.")
        exit(1)
    
    # Perform BallTree clustering
    clusters = balltree_circular_clustering(addresses, max_diameter_m=50, min_points=3)
    
    # Save results
    save_results(addresses, clusters, 'addr-research/balltree_clustered_50m.csv')
    
    # Final statistics
    total_time = time.time() - overall_start
    noise_count = sum(1 for addr in addresses if addr['cluster_id'] == -1)
    clustered_count = len(addresses) - noise_count
    
    log_progress("COMPLETION", f"TOTAL PROCESS COMPLETED in {total_time:.2f} seconds")
    log_progress("RESULTS", f"Total addresses processed: {len(addresses):,}")
    log_progress("RESULTS", f"BallTree clusters formed: {len(clusters):,}")
    log_progress("RESULTS", f"Clustered addresses: {clustered_count:,}")
    log_progress("RESULTS", f"Noise (unclustered): {noise_count:,}")
    log_progress("PERFORMANCE", f"Processing rate: {len(addresses)/total_time:.0f} addresses/second")
    
    if len(addresses) > 0:
        log_progress("RESULTS", f"Clustering efficiency: {(clustered_count/len(addresses)*100):.1f}%")
    
    print("\n" + "=" * 80)
    print("BALLTREE CLUSTER ANALYSIS")
    print("=" * 80)
    
    if clusters:
        sizes = [cluster['size'] for cluster in clusters]
        distances = [cluster['max_distance_m'] for cluster in clusters]
        
        print(f"Algorithm Complexity: O(n log n) with BallTree")
        print(f"Cluster Statistics:")
        print(f"  Total clusters: {len(clusters):,}")
        print(f"  Average cluster size: {sum(sizes)/len(sizes):.1f} addresses")
        print(f"  Largest cluster: {max(sizes):,} addresses")
        print(f"  Smallest cluster: {min(sizes):,} addresses")
        print(f"  Average max distance: {sum(distances)/len(distances):.1f}m")
        print(f"  Maximum distance found: {max(distances):.1f}m")
        print(f"  All clusters ≤ 50m diameter: {all(d <= 50 for d in distances)}")
        
        # Show top clusters
        sorted_clusters = sorted(clusters, key=lambda x: x['size'], reverse=True)
        print(f"\nTop 10 largest clusters:")
        for i, cluster in enumerate(sorted_clusters[:10]):
            print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} addresses, "
                  f"max distance: {cluster['max_distance_m']:.1f}m")
        
        # Size distribution
        size_ranges = [
            ('3-5', 3, 5),
            ('6-10', 6, 10),
            ('11-20', 11, 20),
            ('21-50', 21, 50),
            ('51-100', 51, 100),
            ('100+', 101, float('inf'))
        ]
        
        print(f"\nCluster Size Distribution:")
        for range_name, min_size, max_size in size_ranges:
            count = len([s for s in sizes if min_size <= s <= max_size])
            if count > 0:
                print(f"  {range_name} addresses: {count:,} clusters")
    else:
        print("No clusters found!")
    
    print(f"\nOutput saved to: balltree_clustered.csv")
    print("=" * 80)