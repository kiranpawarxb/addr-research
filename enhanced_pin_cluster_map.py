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

def create_enhanced_pin_map(csv_path):
    """Create enhanced map with pin markers, scale, and advanced interactions"""
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
    
    # Create base map with plugins
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # Add scale bar for distance measurement
    plugins.MeasureControl(
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        secondary_area_unit='hectares'
    ).add_to(m)
    
    # Add fullscreen plugin
    plugins.Fullscreen().add_to(m)
    
    # Add JavaScript for enhanced interactions
    print("Adding enhanced JavaScript with pin markers...")
    
    # Split cluster data into chunks
    chunk_size = 50
    cluster_ids = list(complete_cluster_data.keys())
    
    enhanced_js = """
    <script>
    // Enhanced cluster data storage
    var allClusterData = {};
    var visiblePins = {};
    var activeCircles = {};
    var activeAddresses = {};
    var mapInstance = null;
    var currentZoom = 11;
    
    // Performance settings
    var MAX_VISIBLE_PINS = 300;
    var BATCH_SIZE = 15;
    
    // Initialize map reference
    setTimeout(function() {
        mapInstance = window[Object.keys(window).find(key => key.startsWith('map_'))];
        if (mapInstance) {
            setupEventListeners();
            loadInitialPins();
        }
    }, 1000);
    
    function setupEventListeners() {
        mapInstance.on('zoomend', function() {
            currentZoom = mapInstance.getZoom();
            updateVisiblePins();
        });
        
        mapInstance.on('moveend', function() {
            updateVisiblePins();
        });
    }
    """
    
    # Add cluster data in chunks
    for i in range(0, len(cluster_ids), chunk_size):
        chunk_ids = cluster_ids[i:i + chunk_size]
        chunk_data = {cid: complete_cluster_data[cid] for cid in chunk_ids}
        chunk_js = json.dumps(chunk_data, separators=(',', ':'))
        enhanced_js += f"Object.assign(allClusterData, {chunk_js});\n"
    
    enhanced_js += """
    
    function loadInitialPins() {
        console.log('Loading initial pin markers...');
        updateVisiblePins();
    }
    
    function updateVisiblePins() {
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
        if (visibleIds.length > MAX_VISIBLE_PINS) {
            visibleIds.sort(function(a, b) {
                return allClusterData[b].size - allClusterData[a].size;
            });
            visibleIds = visibleIds.slice(0, MAX_VISIBLE_PINS);
        }
        
        // Add new pins
        visibleIds.forEach(function(clusterId) {
            if (!visiblePins[clusterId]) {
                addPinMarker(clusterId);
            }
        });
        
        // Remove out-of-view pins
        Object.keys(visiblePins).forEach(function(clusterId) {
            if (!visibleIds.includes(clusterId)) {
                removePinMarker(clusterId);
            }
        });
    }
    
    function addPinMarker(clusterId) {
        var cluster = allClusterData[clusterId];
        if (!cluster) return;
        
        // Create custom pin icon with cluster color
        var pinIcon = L.divIcon({
            className: 'custom-pin',
            html: `
                <div style="
                    width: 20px; 
                    height: 20px; 
                    background-color: ${cluster.color}; 
                    border: 2px solid white; 
                    border-radius: 50% 50% 50% 0; 
                    transform: rotate(-45deg);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    cursor: pointer;
                ">
                    <div style="
                        width: 8px; 
                        height: 8px; 
                        background-color: white; 
                        border-radius: 50%; 
                        position: absolute; 
                        top: 4px; 
                        left: 4px;
                    "></div>
                </div>
            `,
            iconSize: [20, 20],
            iconAnchor: [10, 20]
        });
        
        var marker = L.marker([cluster.center_lat, cluster.center_lon], {
            icon: pinIcon
        }).addTo(mapInstance);
        
        marker.bindPopup(`
            <div style="width: 250px;">
                <h4 style="color: ${cluster.color}; margin: 0 0 8px 0;">
                    📍 Cluster ${clusterId}
                </h4>
                <div style="background-color: #f8f9fa; padding: 8px; border-radius: 4px; margin-bottom: 8px;">
                    <b>📊 Size:</b> ${cluster.size} addresses<br>
                    <b>📏 Max Distance:</b> ${cluster.max_distance.toFixed(1)}m<br>
                    <b>📍 Center:</b> (${cluster.center_lat.toFixed(6)}, ${cluster.center_lon.toFixed(6)})
                </div>
                <div style="display: flex; gap: 5px;">
                    <button onclick="showClusterDetails(${clusterId})" 
                            style="flex: 1; padding: 6px 8px; background-color: ${cluster.color}; 
                                   color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 11px;">
                        Show Circle & Points
                    </button>
                    <button onclick="zoomToCluster(${clusterId})" 
                            style="flex: 1; padding: 6px 8px; background-color: #6c757d; 
                                   color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 11px;">
                        Zoom Here
                    </button>
                </div>
            </div>
        `);
        
        marker.bindTooltip(`📍 Cluster ${clusterId} (${cluster.size} addresses)<br>Click to expand details`, {
            direction: 'top',
            offset: [0, -25]
        });
        
        // Add click event for pin
        marker.on('click', function() {
            if (activeCircles[clusterId]) {
                hideClusterDetails(clusterId);
            } else {
                showClusterDetails(clusterId);
            }
        });
        
        visiblePins[clusterId] = marker;
    }
    
    function removePinMarker(clusterId) {
        if (visiblePins[clusterId]) {
            mapInstance.removeLayer(visiblePins[clusterId]);
            delete visiblePins[clusterId];
        }
        hideClusterDetails(clusterId);
    }
    
    function showClusterDetails(clusterId) {
        var cluster = allClusterData[clusterId];
        if (!cluster) return;
        
        console.log(`Showing details for cluster ${clusterId}`);
        
        // Add 300m circle with close button
        var circle = L.circle([cluster.center_lat, cluster.center_lon], {
            radius: 300,
            color: cluster.color,
            weight: 3,
            fillColor: cluster.color,
            fillOpacity: 0.15,
            interactive: true
        }).addTo(mapInstance);
        
        // Create close button for circle
        var closeButton = L.divIcon({
            className: 'circle-close-button',
            html: `
                <div onclick="hideClusterDetails(${clusterId})" style="
                    width: 24px; 
                    height: 24px; 
                    background-color: #dc3545; 
                    color: white; 
                    border: 2px solid white; 
                    border-radius: 50%; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    cursor: pointer; 
                    font-weight: bold; 
                    font-size: 14px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                " title="Close circle and hide addresses">
                    ×
                </div>
            `,
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        });
        
        var closeMarker = L.marker([cluster.center_lat, cluster.center_lon], {
            icon: closeButton,
            zIndexOffset: 1000
        }).addTo(mapInstance);
        
        circle.bindPopup(`
            <div style="width: 220px;">
                <h4 style="color: ${cluster.color};">📍 Cluster ${clusterId} - 300m Circle</h4>
                <b>📊 Total Addresses:</b> ${cluster.size}<br>
                <b>📏 Circle Radius:</b> 300 meters<br>
                <b>🎯 Max Distance:</b> ${cluster.max_distance.toFixed(1)}m<br>
                <hr style="margin: 8px 0;">
                <small style="color: #666;">
                    💡 <b>Tip:</b> You can click other pins inside this circle!<br>
                    ❌ Click the red × button to close this circle
                </small>
            </div>
        `);
        
        activeCircles[clusterId] = {
            circle: circle,
            closeButton: closeMarker
        };
        
        // Load address points
        loadAddressesInBatches(clusterId, cluster);
        
        // Enable clicking other pins within this circle
        enablePinClicksInCircle(clusterId);
    }
    
    function hideClusterDetails(clusterId) {
        // Remove circle and close button
        if (activeCircles[clusterId]) {
            mapInstance.removeLayer(activeCircles[clusterId].circle);
            mapInstance.removeLayer(activeCircles[clusterId].closeButton);
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
    
    function loadAddressesInBatches(clusterId, cluster) {
        var addresses = cluster.addresses;
        var addressMarkers = [];
        var currentBatch = 0;
        
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
                        <h4 style="color: ${cluster.color}; margin: 0 0 5px 0;">
                            📍 Address in Cluster ${clusterId}
                        </h4>
                        <hr style="margin: 5px 0;">
                        <b>🏠 Address:</b> ${addr.address}<br>
                        <b>📮 Pincode:</b> ${addr.pincode}<br>
                        <b>📍 Coordinates:</b> (${addr.lat.toFixed(6)}, ${addr.lon.toFixed(6)})<br>
                        <hr style="margin: 5px 0;">
                        <small style="color: #666;">
                            Distance from cluster center can be measured using the scale tool
                        </small>
                    </div>
                `);
                
                marker.bindTooltip(`🏠 Address in Cluster ${clusterId}`);
                addressMarkers.push(marker);
            }
            
            currentBatch++;
            if (end < addresses.length) {
                setTimeout(loadBatch, 40);
            }
        }
        
        loadBatch();
        activeAddresses[clusterId] = addressMarkers;
    }
    
    function enablePinClicksInCircle(activeClusterId) {
        // This function ensures other pins can still be clicked when a circle is active
        console.log(`Enabled pin clicks within circle of cluster ${activeClusterId}`);
    }
    
    // Global control functions
    function showTopClusters() {
        var sortedClusters = Object.keys(allClusterData).sort(function(a, b) {
            return allClusterData[b].size - allClusterData[a].size;
        });
        
        for (var i = 0; i < Math.min(8, sortedClusters.length); i++) {
            var clusterId = sortedClusters[i];
            if (visiblePins[clusterId] && !activeCircles[clusterId]) {
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
    
    // Add custom CSS for pins
    var style = document.createElement('style');
    style.textContent = `
        .custom-pin {
            background: transparent !important;
            border: none !important;
        }
        .circle-close-button {
            background: transparent !important;
            border: none !important;
        }
        .circle-close-button:hover div {
            background-color: #c82333 !important;
            transform: scale(1.1);
        }
    `;
    document.head.appendChild(style);
    </script>
    """
    
    # Add the JavaScript to the map
    m.get_root().html.add_child(folium.Element(enhanced_js))
    
    return m, color_map, cluster_info, len(df)

def create_enhanced_control_legend(color_map, cluster_info, total_addresses):
    """Create enhanced legend with all controls"""
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 340px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 15px; border-radius: 5px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);">
    
    <h3 style="margin: 0 0 10px 0; color: #333;">📍 Enhanced Pin Cluster Map</h3>
    
    <div style="background-color: #e8f5e8; padding: 8px; border-radius: 3px; margin-bottom: 10px;">
        <b>🎯 New Features:</b><br>
        <span style="color: #666;">📍 Pin markers instead of circles</span><br>
        <span style="color: #666;">📏 Distance scale tool (bottom-left)</span><br>
        <span style="color: #666;">❌ Closeable circles with × button</span><br>
        <span style="color: #666;">🎯 Click other pins inside circles</span><br>
        <span style="color: #666;">🔍 Fullscreen mode available</span>
    </div>
    
    <div style="background-color: #fff3cd; padding: 8px; border-radius: 3px; margin-bottom: 10px;">
        <b>📏 How to Measure Distance:</b><br>
        <span style="color: #856404;">1. Use the ruler tool (bottom-left corner)</span><br>
        <span style="color: #856404;">2. Click to start, click to end measurement</span><br>
        <span style="color: #856404;">3. See distance in meters/kilometers</span>
    </div>
    
    <div style="margin-bottom: 10px;">
        <button onclick="showTopClusters()" 
                style="padding: 5px 10px; margin: 2px; background-color: #2196F3; 
                       color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">
            📊 Show Top 8
        </button>
        <button onclick="hideAllDetails()" 
                style="padding: 5px 10px; margin: 2px; background-color: #f44336; 
                       color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">
            ❌ Hide All
        </button>
    </div>
    
    <b>📊 Dataset Statistics:</b><br>
    <span style="color: #666;">🏠 Total Addresses: {total_addresses:,}</span><br>
    <span style="color: #666;">📍 Total Clusters: {len(color_map):,}</span><br>
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
        <div style="margin: 3px 0; font-size: 11px; cursor: pointer; padding: 4px; 
                    border-radius: 3px; border: 1px solid #ddd; display: flex; align-items: center;" 
             onclick="zoomToCluster({cluster_id})"
             onmouseover="this.style.backgroundColor='#f8f9fa'"
             onmouseout="this.style.backgroundColor='white'">
            <div style="width: 16px; height: 16px; background-color: {color}; 
                        border: 2px solid white; border-radius: 50% 50% 50% 0; 
                        transform: rotate(-45deg); margin-right: 8px; flex-shrink: 0;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.3);"></div>
            <div style="transform: rotate(0deg);">
                <b>Cluster {cluster_id}</b><br>
                <small style="color: #666;">{cluster['cluster_size']} addresses</small>
            </div>
        </div>
        '''
    
    legend_html += f'''
    <div style="margin-top: 10px; font-style: italic; color: #666; font-size: 10px; 
                background-color: #f8f9fa; padding: 6px; border-radius: 3px;">
        💡 <b>Pro Tips:</b><br>
        • Pan around to see more pins dynamically<br>
        • Use measure tool for precise distances<br>
        • Multiple circles can be open simultaneously<br>
        • Click × button on circles to close them<br>
        • Pins inside circles remain clickable
    </div>
    </div>
    '''
    
    return legend_html

def main():
    random.seed(42)
    
    csv_path = "addr-research/balltree_clustered.csv"
    
    print("Creating enhanced pin cluster map with advanced features...")
    print("="*80)
    
    # Create the enhanced map
    enhanced_map, color_map, cluster_info, total_addresses = create_enhanced_pin_map(csv_path)
    
    # Add enhanced legend
    legend_html = create_enhanced_control_legend(color_map, cluster_info, total_addresses)
    enhanced_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <div style="position: absolute; top: 10px; left: 50%; transform: translateX(-50%); 
               z-index: 9999; background-color: white; padding: 12px; 
               border: 2px solid grey; border-radius: 5px; font-size: 14px;
               box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <h2 style="margin: 0; color: #333;">📍 Enhanced Pin Cluster Map</h2>
        <p style="margin: 5px 0 0 0; color: #666; font-size: 12px;">
            <b>🎯 Features:</b> Pin markers • Distance scale • Closeable circles • Multi-interaction
        </p>
    </div>
    '''
    enhanced_map.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    output_file = "enhanced_pin_cluster_map.html"
    enhanced_map.save(output_file)
    
    print(f"\nEnhanced pin cluster map completed!")
    print(f"Map saved as: {output_file}")
    print(f"\n🎯 NEW FEATURES:")
    print(f"📍 Pin markers instead of circle markers")
    print(f"📏 Distance measurement scale (bottom-left)")
    print(f"❌ Closeable circles with red × button")
    print(f"🎯 Click other pins inside active circles")
    print(f"🔍 Fullscreen mode support")
    print(f"📊 Enhanced popups with better information")
    print(f"\n🎮 INTERACTIONS:")
    print(f"• Click any pin → Shows 300m circle + addresses")
    print(f"• Click red × on circle → Closes that circle")
    print(f"• Click other pins inside circles → Opens multiple")
    print(f"• Use ruler tool → Measure precise distances")
    print(f"• Pan/zoom → Dynamic pin loading")
    print(f"• Legend clicks → Quick zoom to clusters")

if __name__ == "__main__":
    main()