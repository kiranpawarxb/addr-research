#!/usr/bin/env python3
"""
Complete Cluster Visualization Script

Creates an interactive map showing ALL geocoded clusters with optimizations for large datasets.
Uses marker clustering and efficient rendering for better performance.
"""

import pandas as pd
import numpy as np
import folium
from folium import plugins
import re
import random
import colorsys
from typing import List, Tuple, Dict
import webbrowser
import os

def extract_coordinates(multipoint_str: str) -> Tuple[float, float]:
    """Extract first coordinate from MULTIPOINT string"""
    if pd.isna(multipoint_str) or not multipoint_str:
        return None, None
    
    pattern = r'\(([0-9.-]+)\s+([0-9.-]+)\)'
    match = re.search(pattern, multipoint_str)
    
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except ValueError:
            return None, None
    return None, None

def generate_distinct_colors(n: int) -> List[str]:
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1
        lightness = 0.5 + (i % 2) * 0.2
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

def load_cluster_data(csv_file: str) -> pd.DataFrame:
    """Load cluster analysis results"""
    print(f"📖 Loading cluster data from {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} clusters")
        return df
    except Exception as e:
        print(f"❌ Error loading cluster data: {e}")
        return pd.DataFrame()

def load_original_data_sample(csv_file: str = 'export_customer_address_store_p0.csv', 
                             sample_size: int = 100000) -> pd.DataFrame:
    """Load sample of original address data for performance"""
    print(f"📖 Loading original address data (sample: {sample_size:,})...")
    
    try:
        df = pd.read_csv(csv_file, nrows=sample_size)
        print(f"✅ Loaded {len(df):,} original addresses")
        return df
    except Exception as e:
        print(f"❌ Error loading original data: {e}")
        return pd.DataFrame()

def create_complete_cluster_map(cluster_df: pd.DataFrame, original_df: pd.DataFrame) -> folium.Map:
    """Create interactive map with ALL clusters using performance optimizations"""
    print(f"🗺️ Creating complete cluster map with {len(cluster_df):,} clusters...")
    
    if len(cluster_df) == 0:
        print("❌ No cluster data to visualize")
        return None
    
    # Calculate map center from cluster centers
    center_lat = cluster_df['center_latitude'].mean()
    center_lon = cluster_df['center_longitude'].mean()
    
    # Create base map with better tiles for large datasets
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap',
        prefer_canvas=True  # Better performance for many markers
    )
    
    # Create address lookup for faster matching
    print("🔍 Creating address lookup index...")
    addr_lookup = {}
    for idx, row in original_df.iterrows():
        addr_lookup[row['addr_hash_key']] = row
    print(f"✅ Address lookup created with {len(addr_lookup):,} entries")
    
    # Generate colors (cycle through if too many clusters)
    max_colors = 50
    base_colors = generate_distinct_colors(max_colors)
    
    # Create marker cluster for better performance
    marker_cluster = plugins.MarkerCluster(
        name="Address Clusters",
        overlay=True,
        control=True,
        options={
            'maxClusterRadius': 30,
            'spiderfyOnMaxZoom': True,
            'showCoverageOnHover': False
        }
    )
    
    # Add cluster circles layer
    circle_layer = folium.FeatureGroup(name="300m Circles", overlay=True, control=True)
    
    # Process clusters in batches for progress tracking
    batch_size = 100
    total_clusters = len(cluster_df)
    processed_addresses = 0
    
    print(f"🗺️ Processing {total_clusters:,} clusters...")
    
    for batch_start in range(0, total_clusters, batch_size):
        batch_end = min(batch_start + batch_size, total_clusters)
        batch_clusters = cluster_df.iloc[batch_start:batch_end]
        
        for idx, (_, cluster_row) in enumerate(batch_clusters.iterrows()):
            cluster_id = cluster_row['group_id']
            color_idx = cluster_id % max_colors
            cluster_color = base_colors[color_idx]
            
            # Get addresses for this cluster
            addr_hash_keys = cluster_row['addr_hash_keys'].split('; ')
            
            # Find matching addresses using lookup
            cluster_addresses = []
            for hash_key in addr_hash_keys:
                if hash_key in addr_lookup:
                    addr_row = addr_lookup[hash_key]
                    lon, lat = extract_coordinates(addr_row['assigned_pickup_dlvd_geo_points'])
                    if lon is not None and lat is not None:
                        cluster_addresses.append({
                            'lat': lat,
                            'lon': lon,
                            'address': addr_row['addr_text'],
                            'pincode': addr_row['pincode'],
                            'hash_key': hash_key
                        })
            
            if not cluster_addresses:
                continue
            
            processed_addresses += len(cluster_addresses)
            
            # Add 300m radius circle to circle layer
            folium.Circle(
                location=[cluster_row['center_latitude'], cluster_row['center_longitude']],
                radius=150,
                popup=f"Cluster {cluster_id} ({cluster_row['address_count']} addresses)",
                tooltip=f"Cluster {cluster_id}",
                color=cluster_color,
                weight=1,
                fill=True,
                fillColor=cluster_color,
                fillOpacity=0.1
            ).add_to(circle_layer)
            
            # Add cluster center marker
            center_popup = f"""
            <div style="width: 250px;">
                <h4 style="color: {cluster_color};">🎯 Cluster {cluster_id}</h4>
                <b>Addresses:</b> {cluster_row['address_count']}<br>
                <b>Same Society:</b> {'✅ Yes' if cluster_row['same_society'] else '❌ No'}<br>
                <b>Max Distance:</b> {cluster_row.get('max_distance_from_center', 'N/A'):.0f}m<br>
                <b>Societies:</b> {cluster_row['unique_societies_count']}<br>
            </div>
            """
            
            folium.Marker(
                location=[cluster_row['center_latitude'], cluster_row['center_longitude']],
                popup=folium.Popup(center_popup, max_width=300),
                tooltip=f"Cluster {cluster_id} ({cluster_row['address_count']} addresses)",
                icon=folium.Icon(color='red', icon='bullseye', prefix='fa')
            ).add_to(marker_cluster)
            
            # Add individual address markers to cluster
            for i, addr in enumerate(cluster_addresses):
                addr_popup = f"""
                <div style="width: 300px;">
                    <h4 style="color: {cluster_color};">📍 Cluster {cluster_id} - Address {i+1}</h4>
                    <b>Address:</b> {addr['address'][:100]}{'...' if len(addr['address']) > 100 else ''}<br>
                    <b>Pincode:</b> {addr['pincode']}<br>
                    <b>Coordinates:</b> {addr['lat']:.6f}, {addr['lon']:.6f}<br>
                </div>
                """
                
                folium.CircleMarker(
                    location=[addr['lat'], addr['lon']],
                    radius=4,
                    popup=folium.Popup(addr_popup, max_width=350),
                    tooltip=f"Cluster {cluster_id}: {addr['address'][:30]}...",
                    color='white',
                    weight=1,
                    fill=True,
                    fillColor=cluster_color,
                    fillOpacity=0.7
                ).add_to(marker_cluster)
        
        # Progress update
        progress_pct = batch_end / total_clusters * 100
        print(f"   🔄 Processed {batch_end:,}/{total_clusters:,} clusters ({progress_pct:.1f}%) | "
              f"Addresses: {processed_addresses:,}")
    
    # Add layers to map
    marker_cluster.add_to(m)
    circle_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add custom legend
    legend_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <h4>🗺️ Complete Cluster Map</h4>
    <p><i class="fa fa-bullseye" style="color:red"></i> Cluster Centers</p>
    <p><i class="fa fa-circle" style="color:blue"></i> Individual Addresses</p>
    <p><i class="fa fa-circle-o" style="color:green"></i> 300m Circles</p>
    <br>
    <p><b>📊 Total:</b> {len(cluster_df):,} clusters</p>
    <p><b>🏠 Addresses:</b> {processed_addresses:,}</p>
    <p><b>💡 Tip:</b> Use layer control to toggle circles</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    print(f"✅ Map created with {len(cluster_df):,} clusters and {processed_addresses:,} addresses")
    return m

def main():
    print("🗺️ COMPLETE CLUSTER VISUALIZATION")
    print("=" * 60)
    
    # Load cluster analysis results
    cluster_file = 'geocode_groups_FIXED.csv'
    if not os.path.exists(cluster_file):
        print(f"❌ Cluster file '{cluster_file}' not found!")
        print("Please run the fixed clustering analysis first:")
        print("py geocode_grouping_fixed.py")
        return
    
    cluster_df = load_cluster_data(cluster_file)
    if len(cluster_df) == 0:
        return
    
    # Load original address data (sample for performance)
    original_df = load_original_data_sample(sample_size=100000)
    if len(original_df) == 0:
        return
    
    # Create complete interactive map
    print(f"\n🗺️ Creating complete interactive map...")
    cluster_map = create_complete_cluster_map(cluster_df, original_df)
    
    if cluster_map is None:
        return
    
    # Save map
    map_file = 'complete_cluster_map.html'
    print(f"💾 Saving complete map...")
    cluster_map.save(map_file)
    print(f"✅ Complete map saved as '{map_file}'")
    
    # Create summary
    print(f"\n📊 COMPLETE MAP SUMMARY:")
    print(f"   🎯 Total clusters: {len(cluster_df):,}")
    print(f"   📏 Size range: {cluster_df['address_count'].min()}-{cluster_df['address_count'].max()} addresses")
    print(f"   📊 Average size: {cluster_df['address_count'].mean():.1f} addresses")
    if 'max_distance_from_center' in cluster_df.columns:
        print(f"   📏 Average radius: {cluster_df['max_distance_from_center'].mean():.0f}m")
        print(f"   📏 Max radius: {cluster_df['max_distance_from_center'].max():.0f}m")
    
    # Open in browser
    try:
        webbrowser.open(f'file://{os.path.abspath(map_file)}')
        print(f"🌐 Opening complete map in browser...")
    except:
        print(f"💡 Manually open '{map_file}' in your browser")
    
    print(f"\n✅ Complete visualization ready!")
    print(f"📁 File: {map_file}")
    print(f"💡 Use layer controls to toggle 300m circles and address clusters")

if __name__ == "__main__":
    main()