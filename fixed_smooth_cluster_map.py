import pandas as pd
import folium
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

def create_fixed_smooth_map(csv_path):
    """Create smooth map with guaranteed address point loading"""
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
    
    # Prepare complete cluster data with ALL addresses
    print("Preparing complete cluster data with addresses...")
    complete_cluster_data = {}
    
    for cluster_id in unique_clusters:
        if cluster_id % 500 == 0:
            print(f"Processing cluster {cluster_id}/{len(unique_clusters)}")
            
        # Get cluster info
        cluster_row = cluster_info[cluster_info['cluster_id'] == cluster_id].iloc[0]
        
        # Get ALL addresses for this cluster
        cluster_addresses = df[df['cluster_id'] == cluster_id]
        
        addresses_list = []
        for _, addr in cluster_addresses.iterrows():
            addresses_list.append({
                'lat': float(addr['latitude']),
                'lon': float(addr['longitude']),
                'address': str(addr['address'])[:80],
                'pincode': str(addr['pincode'])
            })
        
        complete_cluster_data[str(cluster_id)] = {
            'center_lat': float(cluster_row['cluster_center_lat']),
            'center_lon': float(cluster_row['cluster_center_lon']),
            'size': int(cluster_row['cluster_size']),
            'max_distance': float(cluster_row['cluster_max_distance_m']),
            'color': color_map[cluster_id],
            'addresses': addresses_list
        }
    
    print(f"Prepared data for {len(complete_cluster_data)} clusters")
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # Add JavaScript for smooth interaction with guaranteed address loading
    print("Adding optimized JavaScript with address loading...")
    
    # Split cluster data into manageable chunks for JavaScript
    chunk_size = 50
    cluster_ids = list(complete_cluster_data.keys())
    
    optimized_js = """
    <script>
    // Complete cluster data storage
    var allClusterData = {};
    var visibleCentroids = {};
    var activeCircles = {};
    var activeAddresses = {};
    var mapInstance = null;
    var currentZoom = 11;
    
    // Performance settings
    var MAX_VISIBLE_CENTROIDS = 300;
    var BATCH_SIZE = 20;
    
    // Initialize map reference
    setTimeout(function() {
        mapInstance = window[Object.keys(window).find(key => key.startsWith('map_'))];
        if (mapInstance) {
            setupEventListeners();
            loadInitialCentroids();
        }
    }, 1000);
    
    function setupEventListeners() {
        mapInstance.on('zoomend', function() {
            currentZoom = mapInstance.getZoom();
            updateVisibleCentroids();
        });
        
        mapInstance.on('moveend', function() {
            updateVisibleCentroids();
        });
    }
    """
    
    # Add cluster data in chunks
    for i in range(0, len(cluster_ids), chunk_size):
        chunk_ids = cluster_ids[i:i + chunk_size]
        chunk_data = {cid: complete_cluster_data[cid] for cid in chunk_ids}
        chunk_js = json.dumps(chunk_data, separators=(',', ':'))
        optimized_js += f"Object.assign(allClusterData, {chunk_js});\n"
    
    optimized_js += """
    
    function loadInitialCentroids() {
        console.log('Loading initial centroids...');
        var clusterIds = Object.keys(allClusterData);
        
        // Sort by size and load largest first
        clusterIds.sort(function(a, b) {
            return allClusterData[b].size - allClusterData[a].size;
        });
        
        // Load in viewport
        updateVisibleCentroids();
    }
    
    function updateVisibleCentroids() {
        if (!mapInstance) return;
        
        var bounds = mapInstance.getBounds();
        var visibleIds = [];
        
        // Find clusters in viewport
        Object.keys(allClusterData).forEach(function(clusterId) {
            var cluster = allClusterData[clusterId];
            if (bounds.contains([cluster.center_lat, cluster.center_lon])) {
                visibleIds.push(clusterId);
            }
        });
        
        // Limit for performance
        if (visibleIds.length > MAX_VISIBLE_CENTROIDS) {
            visibleIds.sort(function(a, b) {
                return allClusterData[b].size - allClusterData[a].size;
            });
            visibleIds = visibleIds.slice(0, MAX_VISIBLE_CENTROIDS);
        }
        
        // Add new centroids
        visibleIds.forEach(function(clusterId) {
            if (!visibleCentroids[clusterId]) {
                addCentroid(clusterId);
            }
        });
        
        // Remove out-of-view centroids
        Object.keys(visibleCentroids).forEach(function(clusterId) {
            if (!visibleIds.includes(clusterId)) {
                removeCentroid(clusterId);
            }
        });
    }
    
    function addCentroid(clusterId) {
        var cluster = allClusterData[clusterId];
        if (!cluster) return;
        
        var markerSize = Math.min(15, Math.max(6, Math.log(cluster.size) * 2));
        
        var marker = L.circleMarker([cluster.center_lat, cluster.center_lon], {
            radius: markerSize,
            color: 'black',
            weight: 2,
            fillColor: cluster.color,
            fillOpacity: 0.8
        }).addTo(mapInstance);
        
        marker.bindPopup(`
            <div style="width: 220px;">
                <h4 style="color: ${cluster.color}; margin: 0 0 8px 0;">Cluster ${clusterId}</h4>
                <b>Size:</b> ${cluster.size} addresses<br>
                <b>Max Distance:</b> ${cluster.max_distance.toFixed(1)}m<br>
                <button onclick="toggleClusterDetails(${clusterId})" 
                        style="margin-top: 10px; padding: 6px 12px; background-color: ${cluster.color}; 
                               color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">
                    Show Circle & Addresses
                </button>
            </div>
        `);
        
        marker.bindTooltip(`Cluster ${clusterId} (${cluster.size} addresses) - Click to expand`);
        
        visibleCentroids[clusterId] = marker;
    }
    
    function removeCentroid(clusterId) {
        if (visibleCentroids[clusterId]) {
            mapInstance.removeLayer(visibleCentroids[clusterId]);
            delete visibleCentroids[clusterId];
        }
        
        // Also remove details
        hideClusterDetails(clusterId);
    }
    
    function toggleClusterDetails(clusterId) {
        if (activeCircles[clusterId] || activeAddresses[clusterId]) {
            hideClusterDetails(clusterId);
        } else {
            showClusterDetails(clusterId);
        }
    }
    
    function showClusterDetails(clusterId) {
        var cluster = allClusterData[clusterId];
        if (!cluster) {
            console.log('Cluster not found:', clusterId);
            return;
        }
        
        console.log(`Showing details for cluster ${clusterId} with ${cluster.addresses.length} addresses`);
        
        // Add 300m circle
        var circle = L.circle([cluster.center_lat, cluster.center_lon], {
            radius: 300,
            color: cluster.color,
            weight: 3,
            fillColor: cluster.color,
            fillOpacity: 0.15
        }).addTo(mapInstance);
        
        circle.bindPopup(`
            <div style="width: 200px;">
                <h4 style="color: ${cluster.color};">Cluster ${clusterId} - 300m Circle</h4>
                <b>Total Addresses:</b> ${cluster.size}<br>
                <b>Showing:</b> ${cluster.addresses.length} address points
            </div>
        `);
        
        activeCircles[clusterId] = circle;
        
        // Add ALL address points in batches
        loadAddressesInBatches(clusterId, cluster);
    }
    
    function loadAddressesInBatches(clusterId, cluster) {
        var addresses = cluster.addresses;
        var addressMarkers = [];
        var currentBatch = 0;
        
        console.log(`Loading ${addresses.length} addresses for cluster ${clusterId} in batches`);
        
        function loadBatch() {
            var start = currentBatch * BATCH_SIZE;
            var end = Math.min(start + BATCH_SIZE, addresses.length);
            
            for (var i = start; i < end; i++) {
                var addr = addresses[i];
                var marker = L.circleMarker([addr.lat, addr.lon], {
                    radius: 4,
                    color: 'white',
                    weight: 1,
                    fillColor: cluster.color,
                    fillOpacity: 0.8
                }).addTo(mapInstance);
                
                marker.bindPopup(`
                    <div style="width: 220px;">
                        <h4 style="color: ${cluster.color}; margin: 0 0 5px 0;">Cluster ${clusterId}</h4>
                        <hr style="margin: 5px 0;">
                        <b>Address:</b> ${addr.address}<br>
                        <b>Pincode:</b> ${addr.pincode}<br>
                        <b>Coordinates:</b> (${addr.lat.toFixed(6)}, ${addr.lon.toFixed(6)})
                    </div>
                `);
                
                marker.bindTooltip(`Address in Cluster ${clusterId}`);
                addressMarkers.push(marker);
            }
            
            currentBatch++;
            if (end < addresses.length) {
                setTimeout(loadBatch, 50); // Smooth loading
            } else {
                console.log(`Finished loading ${addressMarkers.length} addresses for cluster ${clusterId}`);
            }
        }
        
        loadBatch();
        activeAddresses[clusterId] = addressMarkers;
    }
    
    function hideClusterDetails(clusterId) {
        // Remove circle
        if (activeCircles[clusterId]) {
            mapInstance.removeLayer(activeCircles[clusterId]);
            delete activeCircles[clusterId];
        }
        
        // Remove addresses
        if (activeAddresses[clusterId]) {
            activeAddresses[clusterId].forEach(function(marker) {
                mapInstance.removeLayer(marker);
            });
            delete activeAddresses[clusterId];
        }
        
        console.log(`Hidden details for cluster ${clusterId}`);
    }
    
    // Global control functions
    function showTopClusters() {
        var sortedClusters = Object.keys(allClusterData).sort(function(a, b) {
            return allClusterData[b].size - allClusterData[a].size;
        });
        
        for (var i = 0; i < Math.min(10, sortedClusters.length); i++) {
            var clusterId = sortedClusters[i];
            if (visibleCentroids[clusterId] && !activeCircles[clusterId]) {
                showClusterDetails(clusterId);
            }
        }
    }
    
    function hideAllDetails() {
        Object.keys(activeCircles).forEach(function(clusterId) {
            hideClusterDetails(clusterId);
        });
    }
    
    function zoomToCluster(clusterId) {
        var cluster = allClusterData[clusterId];
        if (cluster && mapInstance) {
            mapInstance.setView([cluster.center_lat, cluster.center_lon], 16);
            setTimeout(function() {
                if (!activeCircles[clusterId]) {
                    showClusterDetails(clusterId);
                }
            }, 500);
        }
    }
    </script>
    """
    
    # Add the JavaScript to the map
    m.get_root().html.add_child(folium.Element(optimized_js))
    
    return m, color_map, cluster_info, len(df)

def create_enhanced_legend(color_map, cluster_info, total_addresses):
    """Create enhanced legend with clear instructions"""
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 320px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 15px; border-radius: 5px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);">
    
    <h3 style="margin: 0 0 10px 0; color: #333;">📍 Interactive Cluster Map</h3>
    
    <div style="background-color: #e8f5e8; padding: 8px; border-radius: 3px; margin-bottom: 10px;">
        <b>✅ Click any centroid to see:</b><br>
        <span style="color: #666;">• 300m radius circle</span><br>
        <span style="color: #666;">• ALL address points in that cluster</span><br>
        <span style="color: #666;">• Click again to hide details</span>
    </div>
    
    <div style="margin-bottom: 10px;">
        <button onclick="showTopClusters()" 
                style="padding: 5px 10px; margin: 2px; background-color: #2196F3; 
                       color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">
            Show Top 10
        </button>
        <button onclick="hideAllDetails()" 
                style="padding: 5px 10px; margin: 2px; background-color: #f44336; 
                       color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">
            Hide All
        </button>
    </div>
    
    <b>Dataset Statistics:</b><br>
    <span style="color: #666;">📊 Total Addresses: {total_addresses:,}</span><br>
    <span style="color: #666;">🎯 Total Clusters: {len(color_map):,}</span><br>
    <span style="color: #666;">📈 Avg Size: {cluster_info["cluster_size"].mean():.1f} addresses</span><br>
    <span style="color: #666;">🏆 Largest: {cluster_info["cluster_size"].max()} addresses</span><br>
    
    <hr style="margin: 10px 0;">
    
    <b>🔍 Quick Access (Click to Zoom & Show):</b><br>
    '''
    
    top_clusters = cluster_info.nlargest(6, 'cluster_size')
    for _, cluster in top_clusters.iterrows():
        cluster_id = cluster['cluster_id']
        color = color_map[cluster_id]
        legend_html += f'''
        <div style="margin: 3px 0; font-size: 11px; cursor: pointer; padding: 3px; 
                    border-radius: 3px; border: 1px solid #ddd;" 
             onclick="zoomToCluster({cluster_id})"
             onmouseover="this.style.backgroundColor='#f0f0f0'"
             onmouseout="this.style.backgroundColor='white'">
            <span style="background-color: {color}; width: 14px; height: 14px; 
                         display: inline-block; margin-right: 6px; border: 1px solid black; border-radius: 2px;"></span>
            <b>Cluster {cluster_id}</b> ({cluster['cluster_size']} addresses) 🔍
        </div>
        '''
    
    legend_html += f'''
    <div style="margin-top: 8px; font-style: italic; color: #666; font-size: 10px;">
        💡 <b>Tip:</b> Pan around to see more clusters dynamically loaded<br>
        🚀 Smooth performance with viewport-based loading
    </div>
    </div>
    '''
    
    return legend_html

def main():
    random.seed(42)
    
    csv_path = r"D:\dev\repos\addr-ana-by-geocodegroup\balltree_clustered.csv"
    
    print("Creating fixed smooth cluster map with guaranteed address point loading...")
    print("="*80)
    
    # Create the fixed map
    fixed_map, color_map, cluster_info, total_addresses = create_fixed_smooth_map(csv_path)
    
    # Add enhanced legend
    legend_html = create_enhanced_legend(color_map, cluster_info, total_addresses)
    fixed_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <div style="position: absolute; top: 10px; left: 50%; transform: translateX(-50%); 
               z-index: 9999; background-color: white; padding: 12px; 
               border: 2px solid grey; border-radius: 5px; font-size: 14px;
               box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <h2 style="margin: 0; color: #333;">📍 Fixed Smooth Cluster Map</h2>
        <p style="margin: 5px 0 0 0; color: #666; font-size: 12px;">
            <b>✅ Guaranteed:</b> Click any centroid → See 300m circle + ALL address points
        </p>
    </div>
    '''
    fixed_map.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    output_file = "fixed_smooth_cluster_map.html"
    fixed_map.save(output_file)
    
    print(f"\nFixed smooth cluster map completed!")
    print(f"Map saved as: {output_file}")
    print(f"\n✅ GUARANTEED FEATURES:")
    print(f"📍 Click any centroid → Shows 300m circle")
    print(f"📍 Click any centroid → Shows ALL address points for that cluster")
    print(f"📍 Smooth batch loading of addresses (20 per batch)")
    print(f"📍 Viewport-based centroid loading for performance")
    print(f"📍 Console logging for debugging address loading")
    print(f"📍 Enhanced popups with full address information")
    print(f"\n🎯 Usage:")
    print(f"1. Pan/zoom to explore clusters")
    print(f"2. Click any colored centroid")
    print(f"3. Watch as circle + address points appear")
    print(f"4. Click 'Show Top 10' for largest clusters")
    print(f"5. Use legend quick access for specific clusters")

if __name__ == "__main__":
    main()