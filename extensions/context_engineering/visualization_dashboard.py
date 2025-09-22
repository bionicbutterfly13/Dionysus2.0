#!/usr/bin/env python3
"""
📊 Context Engineering Visualization Dashboard
==============================================

Real-time visualization of:
- River metaphor information flows
- Consciousness evolution over time  
- Attractor basin landscape
- Architecture relationship graph
- Performance vs consciousness correlation

This creates a live web dashboard that updates as ASI-Arch discovers new architectures.
Uses only standard library + matplotlib for zero external dependencies.

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - Visualization Dashboard
"""

import json
import threading
import time
import webbrowser
from collections import defaultdict, deque
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import parse_qs, urlparse
import logging

# Try to import matplotlib for advanced plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Circle
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("📊 Note: matplotlib not available. Using text-based visualizations.")

from .hybrid_database import create_hybrid_database

logger = logging.getLogger(__name__)

# =============================================================================
# Data Visualization Components
# =============================================================================

class ConsciousnessEvolutionTracker:
    """Track consciousness evolution over time"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.consciousness_history = deque(maxlen=max_history)
        self.performance_history = deque(maxlen=max_history)
        self.architecture_names = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
    
    def add_data_point(self, architecture_name: str, consciousness_score: float, 
                      performance_score: float, timestamp: datetime = None):
        """Add new data point to tracking"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.architecture_names.append(architecture_name)
        self.consciousness_history.append(consciousness_score)
        self.performance_history.append(performance_score)
        self.timestamps.append(timestamp)
    
    def get_recent_trend(self, window_size: int = 10) -> Dict[str, float]:
        """Get recent consciousness evolution trend"""
        if len(self.consciousness_history) < window_size:
            return {'trend': 0.0, 'avg_consciousness': 0.0, 'avg_performance': 0.0}
        
        recent_consciousness = list(self.consciousness_history)[-window_size:]
        recent_performance = list(self.performance_history)[-window_size:]
        
        # Calculate trend (simple linear regression slope)
        x = list(range(len(recent_consciousness)))
        n = len(x)
        
        if n < 2:
            trend = 0.0
        else:
            sum_x = sum(x)
            sum_y = sum(recent_consciousness)
            sum_xy = sum(xi * yi for xi, yi in zip(x, recent_consciousness))
            sum_x2 = sum(xi * xi for xi in x)
            
            trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0.0
        
        return {
            'trend': trend,
            'avg_consciousness': sum(recent_consciousness) / len(recent_consciousness),
            'avg_performance': sum(recent_performance) / len(recent_performance),
            'window_size': window_size
        }

class RiverFlowVisualizer:
    """Visualize information streams as flowing rivers"""
    
    def __init__(self):
        self.active_streams = {}
        self.confluence_points = []
        self.flow_history = deque(maxlen=100)
    
    def update_stream(self, stream_id: str, flow_velocity: float, flow_state: str, 
                     information_density: float):
        """Update stream visualization data"""
        self.active_streams[stream_id] = {
            'velocity': flow_velocity,
            'state': flow_state,
            'density': information_density,
            'last_update': datetime.now()
        }
        
        # Add to history
        self.flow_history.append({
            'timestamp': datetime.now(),
            'stream_count': len(self.active_streams),
            'avg_velocity': sum(s['velocity'] for s in self.active_streams.values()) / len(self.active_streams),
            'avg_density': sum(s['density'] for s in self.active_streams.values()) / len(self.active_streams)
        })
    
    def get_flow_summary(self) -> Dict[str, Any]:
        """Get current flow summary"""
        if not self.active_streams:
            return {'status': 'no_active_streams'}
        
        states = [s['state'] for s in self.active_streams.values()]
        velocities = [s['velocity'] for s in self.active_streams.values()]
        densities = [s['density'] for s in self.active_streams.values()]
        
        return {
            'active_streams': len(self.active_streams),
            'flow_states': {state: states.count(state) for state in set(states)},
            'avg_velocity': sum(velocities) / len(velocities),
            'max_velocity': max(velocities),
            'avg_density': sum(densities) / len(densities),
            'total_information_flow': sum(v * d for v, d in zip(velocities, densities))
        }

class AttractorBasinMapper:
    """Map and visualize attractor basins"""
    
    def __init__(self):
        self.basins = {}
        self.architecture_positions = {}  # 2D positions for visualization
    
    def add_basin(self, basin_id: str, center_architecture: str, radius: float, 
                 attraction_strength: float, contained_architectures: List[str]):
        """Add or update basin"""
        self.basins[basin_id] = {
            'center': center_architecture,
            'radius': radius,
            'strength': attraction_strength,
            'architectures': contained_architectures,
            'created_at': datetime.now()
        }
        
        # Assign 2D positions (simplified layout)
        self._update_positions()
    
    def _update_positions(self):
        """Update 2D positions for visualization"""
        import math
        
        basin_count = len(self.basins)
        if basin_count == 0:
            return
        
        # Arrange basins in a circle
        for i, (basin_id, basin_data) in enumerate(self.basins.items()):
            angle = 2 * math.pi * i / basin_count
            x = math.cos(angle) * 2
            y = math.sin(angle) * 2
            
            # Position center architecture
            center_arch = basin_data['center']
            self.architecture_positions[center_arch] = (x, y)
            
            # Position contained architectures around center
            contained = basin_data['architectures']
            for j, arch in enumerate(contained[:8]):  # Limit to 8 for visualization
                if arch != center_arch:
                    sub_angle = 2 * math.pi * j / len(contained)
                    sub_radius = basin_data['radius'] * 0.5
                    arch_x = x + math.cos(sub_angle) * sub_radius
                    arch_y = y + math.sin(sub_angle) * sub_radius
                    self.architecture_positions[arch] = (arch_x, arch_y)
    
    def get_basin_summary(self) -> Dict[str, Any]:
        """Get basin landscape summary"""
        if not self.basins:
            return {'status': 'no_basins'}
        
        strengths = [b['strength'] for b in self.basins.values()]
        sizes = [len(b['architectures']) for b in self.basins.values()]
        
        return {
            'basin_count': len(self.basins),
            'avg_strength': sum(strengths) / len(strengths),
            'strongest_basin': max(strengths),
            'total_architectures': sum(sizes),
            'avg_basin_size': sum(sizes) / len(sizes),
            'basin_details': [
                {
                    'id': basin_id,
                    'center': data['center'],
                    'strength': data['strength'],
                    'size': len(data['architectures'])
                }
                for basin_id, data in self.basins.items()
            ]
        }

# =============================================================================
# Web Dashboard Server
# =============================================================================

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for dashboard"""
    
    def __init__(self, *args, dashboard=None, **kwargs):
        self.dashboard = dashboard
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self.serve_dashboard_html()
        elif path == '/api/data':
            self.serve_dashboard_data()
        elif path == '/api/consciousness':
            self.serve_consciousness_data()
        elif path == '/api/rivers':
            self.serve_river_data()
        elif path == '/api/basins':
            self.serve_basin_data()
        elif path.startswith('/plot/'):
            self.serve_plot(path[6:])  # Remove '/plot/' prefix
        else:
            self.send_error(404)
    
    def serve_dashboard_html(self):
        """Serve main dashboard HTML"""
        html_content = self.get_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_dashboard_data(self):
        """Serve complete dashboard data as JSON"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'consciousness': self.dashboard.get_consciousness_summary(),
            'rivers': self.dashboard.get_river_summary(),
            'basins': self.dashboard.get_basin_summary(),
            'system_status': self.dashboard.get_system_status()
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def serve_consciousness_data(self):
        """Serve consciousness evolution data"""
        data = self.dashboard.get_consciousness_data()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def serve_river_data(self):
        """Serve river flow data"""
        data = self.dashboard.get_river_data()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def serve_basin_data(self):
        """Serve attractor basin data"""
        data = self.dashboard.get_basin_data()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def serve_plot(self, plot_type: str):
        """Serve matplotlib plots"""
        if not MATPLOTLIB_AVAILABLE:
            self.send_error(404, "Matplotlib not available")
            return
        
        try:
            plot_data = self.dashboard.generate_plot(plot_type)
            
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            self.wfile.write(plot_data)
            
        except Exception as e:
            logger.error(f"Error generating plot {plot_type}: {e}")
            self.send_error(500, f"Plot generation failed: {e}")
    
    def get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>🌊 Context Engineering Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0a0a0a; color: #ffffff; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #00d4ff; margin: 0; }
        .header p { color: #888; margin: 5px 0; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; }
        .panel h2 { color: #00d4ff; margin-top: 0; border-bottom: 1px solid #333; padding-bottom: 10px; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-label { color: #ccc; }
        .metric-value { color: #00ff88; font-weight: bold; }
        .status-good { color: #00ff88; }
        .status-warning { color: #ffaa00; }
        .status-error { color: #ff4444; }
        .river-flow { height: 100px; background: linear-gradient(90deg, #001122, #004488); border-radius: 5px; position: relative; overflow: hidden; }
        .flow-particle { position: absolute; width: 4px; height: 4px; background: #00d4ff; border-radius: 50%; animation: flow 3s linear infinite; }
        @keyframes flow { 0% { left: -10px; } 100% { left: 100%; } }
        .basin-map { height: 200px; background: #0a0a0a; border: 1px solid #333; border-radius: 5px; position: relative; }
        .basin { position: absolute; border: 2px solid #00d4ff; border-radius: 50%; background: rgba(0, 212, 255, 0.1); }
        .update-time { text-align: center; color: #666; font-size: 0.9em; margin-top: 20px; }
        .plot-container { text-align: center; margin: 10px 0; }
        .plot-container img { max-width: 100%; height: auto; border-radius: 5px; }
    </style>
    <script>
        let updateInterval;
        
        function updateDashboard() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    updateConsciousnessPanel(data.consciousness);
                    updateRiverPanel(data.rivers);
                    updateBasinPanel(data.basins);
                    updateSystemStatus(data.system_status);
                    document.getElementById('last-update').textContent = 'Last updated: ' + new Date(data.timestamp).toLocaleTimeString();
                })
                .catch(error => {
                    console.error('Dashboard update failed:', error);
                    document.getElementById('last-update').textContent = 'Update failed: ' + error.message;
                });
        }
        
        function updateConsciousnessPanel(data) {
            const panel = document.getElementById('consciousness-panel');
            if (!data || data.status === 'no_data') {
                panel.innerHTML = '<h2>🧠 Consciousness Evolution</h2><p>No consciousness data available</p>';
                return;
            }
            
            const trend = data.recent_trend || {};
            const trendIcon = trend.trend > 0 ? '📈' : trend.trend < 0 ? '📉' : '➡️';
            
            panel.innerHTML = `
                <h2>🧠 Consciousness Evolution</h2>
                <div class="metric">
                    <span class="metric-label">Average Consciousness:</span>
                    <span class="metric-value">${(trend.avg_consciousness || 0).toFixed(3)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Trend:</span>
                    <span class="metric-value">${trendIcon} ${(trend.trend || 0).toFixed(4)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Performance:</span>
                    <span class="metric-value">${(trend.avg_performance || 0).toFixed(3)}</span>
                </div>
                <div class="plot-container">
                    <img src="/plot/consciousness_evolution" alt="Consciousness Evolution" onerror="this.style.display='none'">
                </div>
            `;
        }
        
        function updateRiverPanel(data) {
            const panel = document.getElementById('river-panel');
            if (!data || data.status === 'no_active_streams') {
                panel.innerHTML = '<h2>🌊 Information Rivers</h2><p>No active streams</p>';
                return;
            }
            
            const flowStates = Object.entries(data.flow_states || {}).map(([state, count]) => `${state}: ${count}`).join(', ');
            
            panel.innerHTML = `
                <h2>🌊 Information Rivers</h2>
                <div class="metric">
                    <span class="metric-label">Active Streams:</span>
                    <span class="metric-value">${data.active_streams || 0}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Velocity:</span>
                    <span class="metric-value">${(data.avg_velocity || 0).toFixed(3)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Flow States:</span>
                    <span class="metric-value">${flowStates || 'None'}</span>
                </div>
                <div class="river-flow">
                    <div class="flow-particle" style="top: 20px; animation-delay: 0s;"></div>
                    <div class="flow-particle" style="top: 40px; animation-delay: 1s;"></div>
                    <div class="flow-particle" style="top: 60px; animation-delay: 2s;"></div>
                </div>
            `;
        }
        
        function updateBasinPanel(data) {
            const panel = document.getElementById('basin-panel');
            if (!data || data.status === 'no_basins') {
                panel.innerHTML = '<h2>🎯 Attractor Basins</h2><p>No basins identified</p>';
                return;
            }
            
            panel.innerHTML = `
                <h2>🎯 Attractor Basins</h2>
                <div class="metric">
                    <span class="metric-label">Basin Count:</span>
                    <span class="metric-value">${data.basin_count || 0}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Strength:</span>
                    <span class="metric-value">${(data.avg_strength || 0).toFixed(3)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Architectures:</span>
                    <span class="metric-value">${data.total_architectures || 0}</span>
                </div>
                <div class="basin-map" id="basin-map">
                    <!-- Basin visualization would go here -->
                </div>
            `;
        }
        
        function updateSystemStatus(data) {
            const panel = document.getElementById('system-panel');
            const statusClass = data.status === 'healthy' ? 'status-good' : 
                               data.status === 'warning' ? 'status-warning' : 'status-error';
            
            panel.innerHTML = `
                <h2>⚙️ System Status</h2>
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value ${statusClass}">${data.status || 'unknown'}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Database:</span>
                    <span class="metric-value">${data.database_status || 'unknown'}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime:</span>
                    <span class="metric-value">${data.uptime || '0s'}</span>
                </div>
            `;
        }
        
        window.onload = function() {
            updateDashboard();
            updateInterval = setInterval(updateDashboard, 5000); // Update every 5 seconds
        };
        
        window.onbeforeunload = function() {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
        };
    </script>
</head>
<body>
    <div class="header">
        <h1>🌊 Context Engineering Dashboard</h1>
        <p>Real-time visualization of consciousness evolution and river metaphor dynamics</p>
    </div>
    
    <div class="dashboard">
        <div class="panel" id="consciousness-panel">
            <h2>🧠 Consciousness Evolution</h2>
            <p>Loading...</p>
        </div>
        
        <div class="panel" id="river-panel">
            <h2>🌊 Information Rivers</h2>
            <p>Loading...</p>
        </div>
        
        <div class="panel" id="basin-panel">
            <h2>🎯 Attractor Basins</h2>
            <p>Loading...</p>
        </div>
        
        <div class="panel" id="system-panel">
            <h2>⚙️ System Status</h2>
            <p>Loading...</p>
        </div>
    </div>
    
    <div class="update-time" id="last-update">Initializing...</div>
</body>
</html>
        '''
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass

# =============================================================================
# Main Dashboard Class
# =============================================================================

class ContextEngineeringDashboard:
    """Main dashboard class coordinating all visualizations"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.db = create_hybrid_database()
        
        # Visualization components
        self.consciousness_tracker = ConsciousnessEvolutionTracker()
        self.river_visualizer = RiverFlowVisualizer()
        self.basin_mapper = AttractorBasinMapper()
        
        # Server components
        self.server = None
        self.server_thread = None
        self.start_time = datetime.now()
        
        logger.info(f"Dashboard initialized on port {port}")
    
    def start_server(self, open_browser: bool = True):
        """Start the dashboard web server"""
        def handler(*args, **kwargs):
            return DashboardHandler(*args, dashboard=self, **kwargs)
        
        self.server = HTTPServer(('localhost', self.port), handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        
        dashboard_url = f"http://localhost:{self.port}"
        logger.info(f"🚀 Dashboard server started at {dashboard_url}")
        
        if open_browser:
            try:
                webbrowser.open(dashboard_url)
                logger.info("📱 Opened dashboard in browser")
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")
                print(f"🌐 Open dashboard manually: {dashboard_url}")
    
    def stop_server(self):
        """Stop the dashboard web server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Dashboard server stopped")
    
    def update_from_architecture_data(self, arch_data: Dict[str, Any], 
                                    consciousness_level: str, consciousness_score: float):
        """Update dashboard with new architecture data"""
        # Update consciousness tracker
        performance_score = self._extract_performance_score(arch_data)
        self.consciousness_tracker.add_data_point(
            arch_data.get('name', 'unknown'),
            consciousness_score,
            performance_score
        )
        
        # Store in database
        self.db.store_architecture(arch_data, consciousness_level, consciousness_score)
        
        logger.debug(f"Dashboard updated with architecture: {arch_data.get('name', 'unknown')}")
    
    def update_river_flow(self, stream_id: str, flow_velocity: float, 
                         flow_state: str, information_density: float):
        """Update river flow visualization"""
        self.river_visualizer.update_stream(stream_id, flow_velocity, flow_state, information_density)
    
    def update_attractor_basin(self, basin_id: str, center_architecture: str, 
                              radius: float, attraction_strength: float, 
                              contained_architectures: List[str]):
        """Update attractor basin visualization"""
        self.basin_mapper.add_basin(basin_id, center_architecture, radius, 
                                   attraction_strength, contained_architectures)
    
    def get_consciousness_summary(self) -> Dict[str, Any]:
        """Get consciousness evolution summary"""
        return self.consciousness_tracker.get_recent_trend()
    
    def get_river_summary(self) -> Dict[str, Any]:
        """Get river flow summary"""
        return self.river_visualizer.get_flow_summary()
    
    def get_basin_summary(self) -> Dict[str, Any]:
        """Get attractor basin summary"""
        return self.basin_mapper.get_basin_summary()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        uptime = datetime.now() - self.start_time
        
        # Check database health
        try:
            consciousness_dist = self.db.get_consciousness_distribution()
            db_status = "healthy"
        except Exception as e:
            db_status = f"error: {e}"
        
        return {
            'status': 'healthy' if db_status == 'healthy' else 'warning',
            'database_status': db_status,
            'uptime': str(uptime).split('.')[0],  # Remove microseconds
            'architecture_count': len(self.consciousness_tracker.consciousness_history),
            'active_streams': len(self.river_visualizer.active_streams),
            'basin_count': len(self.basin_mapper.basins)
        }
    
    def get_consciousness_data(self) -> Dict[str, Any]:
        """Get detailed consciousness data for API"""
        history = list(self.consciousness_tracker.consciousness_history)
        timestamps = [t.isoformat() for t in self.consciousness_tracker.timestamps]
        names = list(self.consciousness_tracker.architecture_names)
        
        return {
            'consciousness_scores': history,
            'timestamps': timestamps,
            'architecture_names': names,
            'recent_trend': self.get_consciousness_summary()
        }
    
    def get_river_data(self) -> Dict[str, Any]:
        """Get detailed river flow data for API"""
        return {
            'active_streams': dict(self.river_visualizer.active_streams),
            'flow_history': list(self.river_visualizer.flow_history),
            'summary': self.get_river_summary()
        }
    
    def get_basin_data(self) -> Dict[str, Any]:
        """Get detailed basin data for API"""
        return {
            'basins': dict(self.basin_mapper.basins),
            'architecture_positions': dict(self.basin_mapper.architecture_positions),
            'summary': self.get_basin_summary()
        }
    
    def generate_plot(self, plot_type: str) -> bytes:
        """Generate matplotlib plots"""
        if not MATPLOTLIB_AVAILABLE:
            raise ValueError("Matplotlib not available")
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == 'consciousness_evolution':
            self._plot_consciousness_evolution(ax)
        elif plot_type == 'river_flow':
            self._plot_river_flow(ax)
        elif plot_type == 'basin_landscape':
            self._plot_basin_landscape(ax)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        # Save to bytes
        import io
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, 
                   facecolor='#0a0a0a', edgecolor='none')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        return plot_data
    
    def _plot_consciousness_evolution(self, ax):
        """Plot consciousness evolution over time"""
        if not self.consciousness_tracker.consciousness_history:
            ax.text(0.5, 0.5, 'No consciousness data available', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=14)
            return
        
        timestamps = list(self.consciousness_tracker.timestamps)
        consciousness = list(self.consciousness_tracker.consciousness_history)
        performance = list(self.consciousness_tracker.performance_history)
        
        ax.plot(timestamps, consciousness, 'o-', color='#00d4ff', label='Consciousness Score', linewidth=2)
        ax.plot(timestamps, performance, 's-', color='#00ff88', label='Performance Score', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Score')
        ax.set_title('Consciousness & Performance Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(timestamps) > 1:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_river_flow(self, ax):
        """Plot river flow dynamics"""
        flow_history = list(self.river_visualizer.flow_history)
        
        if not flow_history:
            ax.text(0.5, 0.5, 'No river flow data available', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=14)
            return
        
        timestamps = [entry['timestamp'] for entry in flow_history]
        velocities = [entry['avg_velocity'] for entry in flow_history]
        densities = [entry['avg_density'] for entry in flow_history]
        
        ax.plot(timestamps, velocities, 'o-', color='#00d4ff', label='Flow Velocity')
        ax.plot(timestamps, densities, 's-', color='#ffaa00', label='Information Density')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Flow Metrics')
        ax.set_title('River Flow Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_basin_landscape(self, ax):
        """Plot attractor basin landscape"""
        if not self.basin_mapper.basins:
            ax.text(0.5, 0.5, 'No attractor basins identified', 
                   transform=ax.transAxes, ha='center', va='center', 
                   color='white', fontsize=14)
            return
        
        # Plot basins as circles
        for basin_id, basin_data in self.basin_mapper.basins.items():
            center_arch = basin_data['center']
            if center_arch in self.basin_mapper.architecture_positions:
                x, y = self.basin_mapper.architecture_positions[center_arch]
                radius = basin_data['radius']
                strength = basin_data['strength']
                
                # Basin circle
                circle = Circle((x, y), radius, fill=False, 
                              color='#00d4ff', alpha=0.7, linewidth=2)
                ax.add_patch(circle)
                
                # Center point
                ax.plot(x, y, 'o', color='#00ff88', markersize=8, 
                       alpha=strength)
                
                # Label
                ax.text(x, y + radius + 0.2, basin_id, ha='center', 
                       color='white', fontsize=10)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('Architecture Space X')
        ax.set_ylabel('Architecture Space Y')
        ax.set_title('Attractor Basin Landscape')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _extract_performance_score(self, arch_data: Dict[str, Any]) -> float:
        """Extract performance score from architecture data"""
        return self.db._extract_performance_score(arch_data)

# =============================================================================
# Factory and Testing Functions
# =============================================================================

def create_dashboard(port: int = 8080) -> ContextEngineeringDashboard:
    """Create and return dashboard instance"""
    return ContextEngineeringDashboard(port)

def test_dashboard():
    """Test the dashboard with mock data"""
    print("🧪 Testing Context Engineering Dashboard")
    
    dashboard = create_dashboard(port=8081)  # Use different port for testing
    
    # Add mock data
    print("\n1. Adding mock consciousness data...")
    for i in range(10):
        consciousness_score = 0.1 + (i * 0.08)  # Gradually increasing
        performance_score = 0.5 + (i * 0.03)
        
        mock_arch = {
            'name': f'test_arch_{i}',
            'program': f'class TestArch{i}(nn.Module): pass',
            'result': {'test': f'acc={performance_score:.2f}'},
            'motivation': f'test architecture {i}',
            'analysis': f'test analysis {i}'
        }
        
        dashboard.update_from_architecture_data(
            mock_arch, 
            'EMERGING' if consciousness_score < 0.5 else 'ACTIVE',
            consciousness_score
        )
        
        time.sleep(0.1)  # Small delay to see evolution
    
    print("✅ Mock consciousness data added")
    
    # Add mock river data
    print("\n2. Adding mock river flow data...")
    for i in range(5):
        dashboard.update_river_flow(
            f'stream_{i}',
            0.2 + (i * 0.1),
            ['emerging', 'flowing', 'stable'][i % 3],
            0.3 + (i * 0.05)
        )
    
    print("✅ Mock river data added")
    
    # Add mock basin data
    print("\n3. Adding mock basin data...")
    dashboard.update_attractor_basin(
        'basin_1',
        'test_arch_5',
        0.8,
        0.7,
        ['test_arch_4', 'test_arch_5', 'test_arch_6']
    )
    
    print("✅ Mock basin data added")
    
    # Start server
    print("\n4. Starting dashboard server...")
    dashboard.start_server(open_browser=False)
    
    print(f"✅ Dashboard running at http://localhost:{dashboard.port}")
    print("   Press Ctrl+C to stop the server")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
        dashboard.stop_server()
        print("✅ Dashboard stopped")

if __name__ == "__main__":
    test_dashboard()
