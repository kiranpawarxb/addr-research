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

def create_enhanced_pin_map_100m(csv_path):
    """Create enhanced map with pin markers for 100m clustering data"""
    print("Loading 100m clustering data...")
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
        if cluster_id % 2000 == 0:
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
            'id': int(cluster_id),
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
        zoom_start=12,  # Start with higher zoom for 100m clusters
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # Add scale bar for distance measurement
    plugins.MeasureControl(
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        secondary_area_unit='hectares',
        position='bottomleft'
    ).add_to(m)
    
    # Add fullscreen plugin
    plugins.Fullscreen(position='topleft').add_to(m)
    
    # Add minimap for better navigation with many clusters
    plugins.MiniMap(toggle_display=True, position='bottomright').add_to(m)
    
    # Add JavaScript for enhanced interactions with 100m data
    print("Adding enhanced JavaScript for 100m clusters...")
    
    # Split cluster data into chunks (smaller chunks for better performance)
    chunk_size = 30
    cluster_ids = list(complete_cluster_data.keys())
    
    enhanced_js = """
    <script>
    // Enhanced cluster data storage for 100m clusters
    var allClusterData = {};
    var visiblePins = {};
    var activeCircles = {};
    var activeAddresses = {};
    var mapInstance = null;
    var currentZoom = 12;
    var controlPanel = null;
    
    // Performance settings optimized for 100m clusters (more clusters, smaller size)
    var MAX_VISIBLE_PINS = 200;  // Reduced for better performance
    var BATCH_SIZE = 10;         // Smaller batches
    var MIN_ZOOM_FOR_PINS = 11;  // Only show pins at higher zoom levels
    
    // Initialize map reference
    setTimeout(function() {
        mapInstance = window[Object.keys(window).find(key => key.startsWith('map_'))];
        if (mapInstance) {
            setupEventListeners();
            createControlPanel();
            loadInitialPins();
            showWelcomeMessage();
        }
    }, 1000);
    
    function setupEventListeners() {
        mapInstance.on('zoomend', function() {
            currentZoom = mapInstance.getZoom();
            updateVisiblePins();
            updateControlPanel();
        });
        
        mapInstance.on('moveend', function() {
            updateVisiblePins();
            updateControlPanel();
        });
    }
    
    function createControlPanel() {
        controlPanel = document.createElement('div');
        controlPanel.id = 'clusterControlPanel';
        controlPanel.style.cssText = `
            position: fixed;
            top: 80px;
            left: 10px;
            background: white;
            border: 2px solid #007bff;
            border-radius: 8px;
            padding: 12px;
            z-index: 10000;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            min-width: 200px;
            max-height: 400px;
            overflow-y: auto;
            font-family: Arial, sans-serif;
            font-size: 12px;
        `;
        controlPanel.innerHTML = `
            <h4 style="margin: 0 0 10px 0; color: #007bff; border-bottom: 2px solid #007bff; padding-bottom: 5px;">
                🎯 100m Cluster Control
            </h4>
            <div id="clusterStats">Loading...</div>
            <div id="activeClustersList" style="margin-top: 10px;">
                <b>Active Circles:</b><br>
                <div id="activeCirclesContent">None</div>
            </div>
            <div style="margin-top: 10px;">
                <button onclick="closeAllCircles()" 
                        style="width: 100%; padding: 8px; background-color: #dc3545; color: white; 
                               border: none; border-radius: 4px; cursor: pointer; font-size: 11px;">
                    ❌ Close All Circles
                </button>
            </div>
        `;
        document.body.appendChild(controlPanel);
    }
    
    function showWelcomeMessage() {
        var welcomeDiv = document.createElement('div');
        welcomeDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border: 3px solid #28a745;
            border-radius: 10px;
            padding: 20px;
            z-index: 20000;
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
            max-width: 400px;
            text-align: center;
            font-family: Arial, sans-serif;
        `;
        welcomeDiv.innerHTML = `
            <h3 style="color: #28a745; margin: 0 0 15px 0;">🎯 100m Cluster Map!</h3>
            <div style="text-align: left; margin-bottom: 15px;">
                <b>🔬 100m Clustering Features:</b><br>
                • Much smaller, tighter clusters (avg 7.4 addresses)<br>
                • 23,120 total clusters vs 7,448 with 300m<br>
                • Better for hyper-local services<br>
                • 100m service circles instead of 300m<br>
                • Control panel shows live stats
            </div>
            <div style="background-color: #d4edda; padding: 10px; border-radius: 6px; margin-bottom: 15px; text-align: left;">
                <b>📊 Quick Stats:</b><br>
                • Total Clusters: 23,120<br>
                • Average Size: 7.4 addresses<br>
                • Largest Cluster: 458 addresses<br>
                • All clusters ≤ 100m diameter
            </div>
            <button onclick="this.parentElement.remove()" 
                    style="padding: 10px 20px; background-color: #28a745; color: white; 
                           border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
                🚀 Explore 100m Clusters!
            </button>
        `;
        document.body.appendChild(welcomeDiv);
        
        setTimeout(function() {
            if (welcomeDiv.parentElement) {
                welcomeDiv.remove();
            }
        }, 12000);
    }
    
    function updateControlPanel() {
        var statsDiv = document.getElementById('clusterStats');
        var activeDiv = document.getElementById('activeCirclesContent');
        
        var visibleCount = Object.keys(visiblePins).length;
        var activeCount = Object.keys(activeCircles).length;
        var totalAddresses = 0;
        
        Object.keys(activeAddresses).forEach(function(clusterId) {
            if (activeAddresses[clusterId]) {
                totalAddresses += activeAddresses[clusterId].length;
            }
        });
        
        statsDiv.innerHTML = `
            <div style="background-color: #f8f9fa; padding: 8px; border-radius: 4px;">
                <b>📊 Current View:</b><br>
                Zoom Level: ${currentZoom}<br>
                Visible Pins: ${visibleCount}<br>
                Active Circles: ${activeCount}<br>
                Shown Addresses: ${totalAddresses}
            </div>
        `;
        
        if (activeCount === 0) {
            activeDiv.innerHTML = '<div style="color: #666; font-style: italic;">None</div>';
        } else {
            var listHTML = '';
            Object.keys(activeCircles).forEach(function(clusterId) {
                var cluster = allClusterData[clusterId];
                if (cluster) {
                    listHTML += `
                        <div style="margin: 4px 0; padding: 6px; border: 1px solid #ddd; border-radius: 4px; background-color: #f8f9fa;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-size: 11px;">
                                    <span style="color: ${cluster.color}; font-weight: bold;">Cluster ${cluster.id}</span><br>
                                    <small style="color: #666;">${cluster.size} addresses</small>
                                </div>
                                <button onclick="closeSpecificCircle('${clusterId}')" 
                                        style="padding: 2px 6px; background-color: #dc3545; color: white; 
                                               border: none; border-radius: 3px; cursor: pointer; font-size: 10px;"
                                        title="Close this circle">
                                    ✕
                                </button>
                            </div>
                        </div>
                    `;
                }
            });
            activeDiv.innerHTML = listHTML;
        }
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
        console.log('Loading initial pin markers for 100m clusters...');
        updateVisiblePins();
        updateControlPanel();
    }
    
    function updateVisiblePins() {
        if (!mapInstance) return;
        
        // Only show pins at higher zoom levels for performance
        if (currentZoom < MIN_ZOOM_FOR_PINS) {
            Object.keys(visiblePins).forEach(function(clusterId) {
                removePinMarker(clusterId);
            });
            return;
        }
        
        var bounds = mapInstance.getBounds();
        var visibleIds = [];
        
        // Find clusters in viewport
        Object.keys(allClusterData).forEach(function(clusterId) {
            var cluster = allClusterData[clusterId];
            if (bounds.contains([cluster.center_lat, cluster.center_lon])) {
                visibleIds.push(clusterId);
            }
        });
        
        // Limit for performance (important with 23k clusters)
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
        
        // Create smaller pin icon for 100m clusters
        var pinIcon = L.divIcon({
            className: 'custom-pin-100m',
            html: `
                <div style="
                    width: 18px; 
                    height: 18px; 
                    background-color: ${cluster.color}; 
                    border: 2px solid white; 
                    border-radius: 50% 50% 50% 0; 
                    transform: rotate(-45deg);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
                    cursor: pointer;
                ">
                    <div style="
                        width: 6px; 
                        height: 6px; 
                        background-color: white; 
                        border-radius: 50%; 
                        position: absolute; 
                        top: 4px; 
                        left: 4px;
                    "></div>
                </div>
            `,
            iconSize: [18, 18],
            iconAnchor: [9, 18]
        });
        
        var marker = L.marker([cluster.center_lat, cluster.center_lon], {
            icon: pinIcon
        }).addTo(mapInstance);
        
        marker.bindPopup(`
            <div style="width: 280px; font-family: Arial, sans-serif;">
                <h4 style="color: ${cluster.color}; margin: 0 0 8px 0; border-bottom: 2px solid ${cluster.color}; padding-bottom: 4px;">
                    🎯 100m Cluster ${cluster.id}
                </h4>
                <div style="background-color: #f8f9fa; padding: 8px; border-radius: 4px; margin-bottom: 8px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 11px;">
                        <div><b>📊 Size:</b> ${cluster.size} addresses</div>
                        <div><b>📏 Max Dist:</b> ${cluster.max_distance.toFixed(1)}m</div>
                        <div style="grid-column: 1 / -1;"><b>📍 Center:</b> (${cluster.center_lat.toFixed(6)}, ${cluster.center_lon.toFixed(6)})</div>
                    </div>
                </div>
                <div style="background-color: #e8f5e8; padding: 6px; border-radius: 4px; margin-bottom: 8px; font-size: 11px;">
                    <b>🔬 100m Cluster Benefits:</b><br>
                    • Hyper-local service area (100m circle)<br>
                    • Tighter address grouping<br>
                    • Better for walking distance services
                </div>
                <div style="display: flex; gap: 4px;">
                    <button onclick="showClusterDetails('${clusterId}')" 
                            style="flex: 1; padding: 8px; background-color: ${cluster.color}; 
                                   color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: bold;">
                        🎯 Show 100m Circle
                    </button>
                    <button onclick="zoomToCluster('${clusterId}')" 
                            style="flex: 1; padding: 8px; background-color: #6c757d; 
                                   color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 11px;">
                        🔍 Zoom In
                    </button>
                </div>
            </div>
        `);
        
        marker.bindTooltip(`🎯 100m Cluster ${cluster.id} (${cluster.size} addresses)<br>Click to show 100m service area`, {
            direction: 'top',
            offset: [0, -22],
            className: 'custom-tooltip-100m'
        });
        
        // Add click event for pin
        marker.on('click', function() {
            if (activeCircles[clusterId]) {
                closeSpecificCircle(clusterId);
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
    }
    
    function showClusterDetails(clusterId) {
        var cluster = allClusterData[clusterId];
        if (!cluster) return;
        
        console.log(`Showing 100m circle for cluster ${cluster.id}`);
        
        // Add 100m circle (smaller than 300m)
        var circle = L.circle([cluster.center_lat, cluster.center_lon], {
            radius: 100,  // 100m radius instead of 300m
            color: cluster.color,
            weight: 3,
            fillColor: cluster.color,
            fillOpacity: 0.2,
            dashArray: '8, 4'
        }).addTo(mapInstance);
        
        circle.bindPopup(`
            <div style="width: 260px; font-family: Arial, sans-serif;">
                <h4 style="color: ${cluster.color}; margin: 0 0 8px 0;">🎯 100m Service Area - Cluster ${cluster.id}</h4>
                <div style="background-color: #f8f9fa; padding: 8px; border-radius: 4px; margin-bottom: 8px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 11px;">
                        <div><b>📊 Addresses:</b> ${cluster.size}</div>
                        <div><b>📏 Circle:</b> 100m radius</div>
                        <div><b>🎯 Max Dist:</b> ${cluster.max_distance.toFixed(1)}m</div>
                        <div><b>🚶 Walk Time:</b> ~1-2 minutes</div>
                    </div>
                </div>
                <div style="background-color: #d4edda; padding: 6px; border-radius: 4px; font-size: 10px;">
                    <b>💡 100m Benefits:</b> Perfect for local delivery, walking services, neighborhood coverage
                </div>
            </div>
        `);
        
        activeCircles[clusterId] = circle;
        updateControlPanel();
        
        // Load address points
        loadAddressesInBatches(clusterId, cluster);
    }
    
    function closeSpecificCircle(clusterId) {
        if (activeCircles[clusterId]) {
            mapInstance.removeLayer(activeCircles[clusterId]);
            delete activeCircles[clusterId];
        }
        
        if (activeAddresses[clusterId]) {
            activeAddresses[clusterId].forEach(function(marker) {
                mapInstance.removeLayer(marker);
            });
            delete activeAddresses[clusterId];
        }
        
        updateControlPanel();
        console.log(`Closed 100m circle for cluster ${clusterId}`);
    }
    
    function closeAllCircles() {
        Object.keys(activeCircles).forEach(function(clusterId) {
            closeSpecificCircle(clusterId);
        });
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
                    <div style="width: 240px; font-family: Arial, sans-serif;">
                        <h4 style="color: ${cluster.color}; margin: 0 0 6px 0;">
                            🏠 Address in 100m Cluster ${cluster.id}
                        </h4>
                        <div style="background-color: #f8f9fa; padding: 6px; border-radius: 4px; margin-bottom: 6px; font-size: 11px;">
                            <b>🏠 Address:</b> ${addr.address}<br>
                            <b>📮 Pincode:</b> ${addr.pincode}<br>
                            <b>📍 Coordinates:</b> (${addr.lat.toFixed(6)}, ${addr.lon.toFixed(6)})
                        </div>
                        <div style="background-color: #e8f5e8; padding: 4px; border-radius: 3px; font-size: 10px;">
                            💡 Within 100m of cluster center - ideal for hyper-local services
                        </div>
                    </div>
                `);
                
                marker.bindTooltip(`🏠 Address in 100m Cluster ${cluster.id}`, {
                    className: 'custom-tooltip-100m'
                });
                
                addressMarkers.push(marker);
            }
            
            currentBatch++;
            if (end < addresses.length) {
                setTimeout(loadBatch, 50);
            } else {
                updateControlPanel();
            }
        }
        
        loadBatch();
        activeAddresses[clusterId] = addressMarkers;
    }
    
    function zoomToCluster(clusterId) {
        var cluster = allClusterData[clusterId];
        if (cluster && mapInstance) {
            mapInstance.setView([cluster.center_lat, cluster.center_lon], 17);  // Higher zoom for 100m
            setTimeout(function() {
                if (!activeCircles[clusterId]) {
                    showClusterDetails(clusterId);
                }
            }, 500);
        }
    }
    
    // Add custom CSS for 100m clusters
    var style = document.createElement('style');
    style.textContent = `
        .custom-pin-100m {
            background: transparent !important;
            border: none !important;
        }
        .custom-tooltip-100m {
            background-color: rgba(40, 167, 69, 0.9) !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            font-size: 10px !important;
        }
        #clusterControlPanel {
            font-family: Arial, sans-serif;
        }
    `;
    document.head.appendChild(style);
    </script>
    """
    
    # Add the JavaScript to the map
    m.get_root().html.add_child(folium.Element(enhanced_js))
    
    return m, color_map, cluster_info, len(df)

def create_100m_legend(color_map, cluster_info, total_addresses):
    """Create enhanced legend for 100m clustering"""
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 360px; height: auto; 
                background-color: white; border:3px solid #28a745; z-index:9999; 
                font-size:12px; padding: 15px; border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3); font-family: Arial, sans-serif;">
    
    <h3 style="margin: 0 0 12px 0; color: #28a745; border-bottom: 2px solid #28a745; padding-bottom: 6px;">
        🎯 100m Cluster Map
    </h3>
    
    <div style="background-color: #d4edda; padding: 10px; border-radius: 6px; margin-bottom: 12px;">
        <b>🔬 100m Clustering Benefits:</b><br>
        <span style="color: #155724;">• Hyper-local service areas (100m circles)</span><br>
        <span style="color: #155724;">• Perfect for walking distance services</span><br>
        <span style="color: #155724;">• Tighter address grouping</span><br>
        <span style="color: #155724;">• 3x more clusters than 300m version</span><br>
        <span style="color: #155724;">• Average cluster size: 7.4 addresses</span>
    </div>
    
    <div style="background-color: #fff3cd; padding: 10px; border-radius: 6px; margin-bottom: 12px;">
        <b>📊 Comparison with 300m:</b><br>
        <span style="color: #856404;">• Clusters: 23,120 vs 7,448 (3x more)</span><br>
        <span style="color: #856404;">• Avg Size: 7.4 vs 25.4 addresses</span><br>
        <span style="color: #856404;">• Service Area: 100m vs 300m radius</span><br>
        <span style="color: #856404;">• Use Case: Local vs Regional services</span>
    </div>
    
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 6px; margin-bottom: 12px;">
        <b>📊 Dataset Statistics:</b><br>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 11px; margin-top: 5px;">
            <span>🏠 Total Addresses: {total_addresses:,}</span>
            <span>📍 Total Clusters: {len(color_map):,}</span>
            <span>📈 Avg Size: {cluster_info["cluster_size"].mean():.1f}</span>
            <span>🏆 Largest: {cluster_info["cluster_size"].max()}</span>
            <span>📏 Avg Distance: {cluster_info["cluster_max_distance_m"].mean():.1f}m</span>
            <span>🎯 Max Distance: {cluster_info["cluster_max_distance_m"].max():.1f}m</span>
        </div>
    </div>
    
    <b>🔍 Top Clusters (Click to Explore):</b><br>
    '''
    
    top_clusters = cluster_info.nlargest(6, 'cluster_size')
    for _, cluster in top_clusters.iterrows():
        cluster_id = cluster['cluster_id']
        color = color_map[cluster_id]
        legend_html += f'''
        <div style="margin: 4px 0; font-size: 11px; cursor: pointer; padding: 6px; 
                    border-radius: 6px; border: 1px solid #ddd; display: flex; align-items: center; 
                    transition: all 0.2s ease;" 
             onclick="zoomToCluster('{cluster_id}')"
             onmouseover="this.style.backgroundColor='#f8f9fa'; this.style.borderColor='{color}'; this.style.transform='scale(1.02)'"
             onmouseout="this.style.backgroundColor='white'; this.style.borderColor='#ddd'; this.style.transform='scale(1)'">
            <div style="width: 16px; height: 16px; background-color: {color}; 
                        border: 2px solid white; border-radius: 50% 50% 50% 0; 
                        transform: rotate(-45deg); margin-right: 8px; flex-shrink: 0;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>
            <div style="transform: rotate(0deg);">
                <b>Cluster {cluster_id}</b><br>
                <small style="color: #666;">{cluster['cluster_size']} addresses • {cluster['cluster_max_distance_m']:.1f}m max</small>
            </div>
        </div>
        '''
    
    legend_html += f'''
    <div style="margin-top: 12px; font-style: italic; color: #666; font-size: 10px; 
                background-color: #e9ecef; padding: 8px; border-radius: 6px;">
        🎯 <b>100m Use Cases:</b><br>
        • Local food delivery (walking distance)<br>
        • Neighborhood services (plumber, electrician)<br>
        • Hyperlocal marketing campaigns<br>
        • Community-based service planning<br>
        • Walking route optimization
    </div>
    </div>
    '''
    
    return legend_html

def main():
    random.seed(42)
    
    csv_path = "addr-research/balltree_clustered_100m.csv"
    
    print("Creating enhanced pin cluster map for 100m clustering data...")
    print("="*80)
    
    # Create the enhanced map for 100m data
    enhanced_map, color_map, cluster_info, total_addresses = create_enhanced_pin_map_100m(csv_path)
    
    # Add enhanced legend for 100m
    legend_html = create_100m_legend(color_map, cluster_info, total_addresses)
    enhanced_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <div style="position: absolute; top: 10px; left: 50%; transform: translateX(-50%); 
               z-index: 9999; background: linear-gradient(135deg, #28a745, #1e7e34); color: white; padding: 15px; 
               border: none; border-radius: 8px; font-size: 16px; text-align: center;
               box-shadow: 0 4px 12px rgba(0,0,0,0.3); font-family: Arial, sans-serif;">
        <h2 style="margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🎯 100m Cluster Map</h2>
        <p style="margin: 8px 0 0 0; font-size: 13px; opacity: 0.9;">
            <b>🔬 Hyper-Local Clustering:</b> 23,120 clusters • 100m service areas • 7.4 avg size
        </p>
    </div>
    '''
    enhanced_map.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    output_file = "enhanced_pin_cluster_map_100m.html"
    enhanced_map.save(output_file)
    
    print(f"\n🎯 100m Enhanced pin cluster map completed!")
    print(f"Map saved as: {output_file}")
    print(f"\n🔬 100m CLUSTERING FEATURES:")
    print(f"📍 Smaller pin markers optimized for many clusters")
    print(f"🎯 100m service circles instead of 300m")
    print(f"📊 Control panel with live statistics")
    print(f"🔍 Higher zoom levels for better cluster visibility")
    print(f"⚡ Performance optimized for 23,120 clusters")
    print(f"📏 Distance measurement scale")
    print(f"🗺️ Minimap for navigation")
    print(f"\n📊 KEY DIFFERENCES FROM 300m:")
    print(f"• 3x more clusters (23,120 vs 7,448)")
    print(f"• Smaller average size (7.4 vs 25.4 addresses)")
    print(f"• Tighter service areas (100m vs 300m)")
    print(f"• Better for hyper-local services")
    print(f"• Walking distance optimization")
    print(f"\n🎮 OPTIMIZED INTERACTIONS:")
    print(f"• Zoom-based pin loading (performance)")
    print(f"• Control panel for active circle management")
    print(f"• Smaller batches for smooth address loading")
    print(f"• Enhanced tooltips with 100m context")

if __name__ == "__main__":
    main()