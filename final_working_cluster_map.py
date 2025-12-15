import pandas as pd
import folium
from folium import plugins
import numpy as np
import random
import json

def generate_cluster_colors(num_clusters):
    """Generate distinct colors for clusters"""
    base_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2',
        '#A3E4D7', '#F9E79F', '#D5A6BD', '#AED6F1', '#A9DFBF'
    ]
    
    colors = []
    for i in range(num_clusters):
        if i < len(base_colors):
            colors.append(base_colors[i])
        else:
            colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    
    return colors

def create_final_working_map(csv_path):
    """Create final working map with guaranteed circle removal"""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['latitude', 'longitude', 'cluster_center_lat', 'cluster_center_lon'])
    
    print(f"Loaded {len(df)} addresses across {df['cluster_id'].nunique()} clusters")
    
    # Calculate map center
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Generate colors for clusters
    unique_clusters = sorted(df['cluster_id'].unique())
    cluster_colors = generate_cluster_colors(len(unique_clusters))
    color_map = dict(zip(unique_clusters, cluster_colors))
    
    # Get cluster information
    print("Processing cluster information...")
    cluster_info = df.groupby('cluster_id').agg({
        'cluster_center_lat': 'first',
        'cluster_center_lon': 'first',
        'cluster_size': 'first',
        'cluster_max_distance_m': 'first'
    }).reset_index()
    
    # Prepare complete cluster data
    print("Preparing complete cluster data...")
    complete_cluster_data = {}
    
    for cluster_id in unique_clusters:
        if cluster_id % 500 == 0:
            print(f"Processing cluster {cluster_id}/{len(unique_clusters)}")
            
        cluster_row = cluster_info[cluster_info['cluster_id'] == cluster_id].iloc[0]
        cluster_ad