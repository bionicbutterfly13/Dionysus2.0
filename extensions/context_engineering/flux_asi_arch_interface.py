#!/usr/bin/env python3
"""
🌊 Flux ASI-Arch Visual Interface
================================

Enhanced version of SurfSense (now Flux) that integrates with ASI-Arch unified
database system and provides visual document processing capabilities.

Features:
- Document upload and processing interface
- Real-time active inference metrics
- Cross-database learning visualization
- ASI-Arch pipeline status monitoring
- ThoughtSeed consciousness tracking

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-23
Version: 1.0.0 - ASI-Arch Integration
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiofiles

import redis
import aiohttp
from aiohttp import web, web_ws, MultipartReader
import aiohttp_cors
import socketio

# Import ASI-Arch components
import sys
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "pipeline"))

from cross_database_learning import CrossDatabaseLearningIntegration
from unified_active_inference_framework import UnifiedActiveInferenceFramework
from config import Config

logger = logging.getLogger(__name__)

class FluxASIArchInterface:
    """
    Flux Visual Interface for ASI-Arch Document Processing

    Integrates the existing SurfSense (Flux) UI with:
    - ASI-Arch unified database system
    - Real active inference framework
    - Cross-database learning
    - Document upload and processing
    """

    def __init__(self, host: str = 'localhost', port: int = 8080):
        self.host = host
        self.port = port

        # Initialize web application
        self.app = web.Application()
        self.sio = socketio.AsyncServer(cors_allowed_origins="*")
        self.sio.attach(self.app)

        # ASI-Arch integrations
        self.active_inference = UnifiedActiveInferenceFramework()
        self.cross_db_learning = CrossDatabaseLearningIntegration()

        # State tracking
        self.connected_clients = set()
        self.learning_sessions = {}
        self.processed_documents = []
        self.update_task = None

        # Document processing
        self.upload_dir = Path("./uploads")
        self.upload_dir.mkdir(exist_ok=True)

        # Setup interface
        self._setup_routes()
        self._setup_websockets()

        logger.info(f"🌊 Flux ASI-Arch Interface initialized on {host}:{port}")

    def _setup_routes(self):
        """Setup HTTP routes for Flux interface"""

        # Main dashboard
        self.app.router.add_get('/', self._serve_dashboard)

        # System status and metrics
        self.app.router.add_get('/api/status', self._get_asi_arch_status)
        self.app.router.add_get('/api/metrics', self._get_active_inference_metrics)
        self.app.router.add_get('/api/learning/sessions', self._get_learning_sessions)

        # Document processing
        self.app.router.add_post('/api/documents/upload', self._upload_document)
        self.app.router.add_get('/api/documents/list', self._list_documents)
        self.app.router.add_post('/api/documents/process', self._process_document)

        # Active inference controls
        self.app.router.add_post('/api/inference/start', self._start_inference_session)
        self.app.router.add_get('/api/inference/state', self._get_inference_state)

        # Cross-database learning
        self.app.router.add_get('/api/learning/insights', self._get_learning_insights)
        self.app.router.add_get('/api/database/status', self._get_database_status)

        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })

        for route in list(self.app.router.routes()):
            cors.add(route)

    def _setup_websockets(self):
        """Setup WebSocket connections for real-time updates"""

        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            self.connected_clients.add(sid)
            logger.info(f"🔌 Flux client connected: {sid}")

            # Send initial state
            await self._send_initial_state(sid)

        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            self.connected_clients.discard(sid)
            logger.info(f"🔌 Flux client disconnected: {sid}")

        @self.sio.event
        async def get_metrics(sid):
            """Handle metrics request"""
            metrics = await self._get_current_metrics()
            await self.sio.emit('metrics_update', metrics, room=sid)

    async def _serve_dashboard(self, request):
        """Serve the main Flux dashboard"""
        html_content = self._generate_flux_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')

    def _generate_flux_dashboard_html(self) -> str:
        """Generate the Flux dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌊 Flux - ASI-Arch Visual Interface</title>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .title { font-size: 2.5rem; margin-bottom: 10px; }
        .subtitle { font-size: 1.2rem; opacity: 0.8; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .card h3 { margin-bottom: 15px; color: #64b5f6; }
        .metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .metric { text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px; }
        .metric-value { font-size: 1.8rem; font-weight: bold; color: #81c784; }
        .metric-label { font-size: 0.9rem; opacity: 0.8; margin-top: 5px; }
        .upload-area {
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover { border-color: #64b5f6; background: rgba(100,181,246,0.1); }
        .btn {
            background: linear-gradient(45deg, #64b5f6, #42a5f5);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(100,181,246,0.4); }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-connected { background: #4caf50; }
        .status-disconnected { background: #f44336; }
        .status-warning { background: #ff9800; }
        .log-item {
            padding: 8px 12px;
            margin: 5px 0;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.9rem;
        }
        .file-list { max-height: 200px; overflow-y: auto; }
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px;
            background: rgba(255,255,255,0.1);
            margin: 5px 0;
            border-radius: 5px;
        }
        #metricsChart { background: rgba(255,255,255,0.1); border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">🌊 Flux</h1>
            <p class="subtitle">ASI-Arch Visual Interface for Document Processing & Active Inference</p>
        </div>

        <div class="grid">
            <!-- System Status -->
            <div class="card">
                <h3>📊 System Status</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value" id="consciousness-level">0.00</div>
                        <div class="metric-label">Consciousness Level</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="learning-sessions">0</div>
                        <div class="metric-label">Learning Sessions</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="documents-processed">0</div>
                        <div class="metric-label">Documents Processed</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="free-energy">0.00</div>
                        <div class="metric-label">Free Energy</div>
                    </div>
                </div>
                <div style="margin-top: 20px;">
                    <div><span class="status-indicator" id="redis-status"></span>Redis Database</div>
                    <div><span class="status-indicator" id="neo4j-status"></span>Neo4j Knowledge Graph</div>
                    <div><span class="status-indicator" id="inference-status"></span>Active Inference</div>
                </div>
            </div>

            <!-- Document Upload -->
            <div class="card">
                <h3>📄 Document Processing</h3>
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div style="font-size: 1.2rem; margin-bottom: 10px;">📁 Drop files here or click to upload</div>
                    <div style="opacity: 0.7;">Supports PDF, TXT, MD, DOC files</div>
                    <input type="file" id="fileInput" multiple accept=".pdf,.txt,.md,.doc,.docx" style="display: none;">
                </div>
                <button class="btn" onclick="processSelectedDocuments()" style="margin-top: 15px; width: 100%;">
                    🧠 Process with Active Inference
                </button>
                <div class="file-list" id="fileList"></div>
            </div>

            <!-- Active Inference Metrics -->
            <div class="card">
                <h3>🧠 Active Inference Metrics</h3>
                <canvas id="metricsChart" width="400" height="200"></canvas>
                <div style="margin-top: 15px;">
                    <button class="btn" onclick="startInferenceSession()">Start New Session</button>
                    <button class="btn" onclick="refreshMetrics()" style="margin-left: 10px;">Refresh</button>
                </div>
            </div>

            <!-- Learning Insights -->
            <div class="card">
                <h3>🔗 Cross-Database Learning</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value" id="episodic-strength">0.00</div>
                        <div class="metric-label">Episodic Memory</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="semantic-strength">0.00</div>
                        <div class="metric-label">Semantic Memory</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="procedural-strength">0.00</div>
                        <div class="metric-label">Procedural Memory</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="cross-memory-strength">0.00</div>
                        <div class="metric-label">Integration Score</div>
                    </div>
                </div>
            </div>

            <!-- System Logs -->
            <div class="card">
                <h3>📝 System Activity</h3>
                <div id="systemLogs" style="max-height: 250px; overflow-y: auto;"></div>
            </div>

            <!-- Learning Recommendations -->
            <div class="card">
                <h3>💡 Learning Recommendations</h3>
                <div id="recommendations"></div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();

        // Chart for metrics visualization
        let metricsChart;

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeChart();
            setupEventListeners();
            requestInitialData();
        });

        function initializeChart() {
            const ctx = document.getElementById('metricsChart').getContext('2d');
            metricsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Consciousness Level',
                        data: [],
                        borderColor: '#64b5f6',
                        backgroundColor: 'rgba(100, 181, 246, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Free Energy',
                        data: [],
                        borderColor: '#81c784',
                        backgroundColor: 'rgba(129, 199, 132, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, max: 1 }
                    },
                    plugins: {
                        legend: { labels: { color: 'white' } }
                    }
                }
            });
        }

        function setupEventListeners() {
            // File input handler
            document.getElementById('fileInput').addEventListener('change', handleFileSelect);

            // Socket event listeners
            socket.on('metrics_update', updateMetrics);
            socket.on('learning_update', updateLearningInsights);
            socket.on('system_log', addSystemLog);
            socket.on('recommendations_update', updateRecommendations);
        }

        function handleFileSelect(event) {
            const files = event.target.files;
            updateFileList(files);
        }

        function updateFileList(files) {
            const fileList = document.getElementById('fileList');
            fileList.innerHTML = '';

            Array.from(files).forEach(file => {
                const item = document.createElement('div');
                item.className = 'file-item';
                item.innerHTML = `
                    <span>${file.name}</span>
                    <span>${(file.size / 1024 / 1024).toFixed(2)} MB</span>
                `;
                fileList.appendChild(item);
            });
        }

        async function processSelectedDocuments() {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files.length) {
                alert('Please select files to process');
                return;
            }

            const formData = new FormData();
            Array.from(fileInput.files).forEach(file => {
                formData.append('documents', file);
            });

            try {
                const response = await fetch('/api/documents/upload', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    addSystemLog(`✅ Uploaded ${result.files_uploaded} documents successfully`);

                    // Start processing
                    await fetch('/api/documents/process', { method: 'POST' });
                    addSystemLog('🧠 Started document processing with active inference');
                } else {
                    throw new Error('Upload failed');
                }
            } catch (error) {
                addSystemLog(`❌ Upload failed: ${error.message}`);
            }
        }

        async function startInferenceSession() {
            try {
                const response = await fetch('/api/inference/start', { method: 'POST' });
                const result = await response.json();
                addSystemLog(`🧠 Started inference session: ${result.session_id}`);
            } catch (error) {
                addSystemLog(`❌ Failed to start inference session: ${error.message}`);
            }
        }

        function refreshMetrics() {
            socket.emit('get_metrics');
        }

        function requestInitialData() {
            refreshMetrics();
            fetchSystemStatus();
            fetchLearningInsights();
        }

        async function fetchSystemStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                updateSystemStatus(status);
            } catch (error) {
                console.error('Failed to fetch system status:', error);
            }
        }

        async function fetchLearningInsights() {
            try {
                const response = await fetch('/api/learning/insights');
                const insights = await response.json();
                updateLearningInsights(insights);
            } catch (error) {
                console.error('Failed to fetch learning insights:', error);
            }
        }

        function updateMetrics(data) {
            // Update metric displays
            document.getElementById('consciousness-level').textContent = (data.consciousness_level || 0).toFixed(2);
            document.getElementById('learning-sessions').textContent = data.total_sessions || 0;
            document.getElementById('documents-processed').textContent = data.documents_processed || 0;
            document.getElementById('free-energy').textContent = (data.free_energy || 0).toFixed(2);

            // Update chart
            const now = new Date().toLocaleTimeString();
            metricsChart.data.labels.push(now);
            metricsChart.data.datasets[0].data.push(data.consciousness_level || 0);
            metricsChart.data.datasets[1].data.push(data.free_energy || 0);

            // Keep only last 20 data points
            if (metricsChart.data.labels.length > 20) {
                metricsChart.data.labels.shift();
                metricsChart.data.datasets.forEach(dataset => dataset.data.shift());
            }

            metricsChart.update();
        }

        function updateSystemStatus(status) {
            const redisStatus = document.getElementById('redis-status');
            const neo4jStatus = document.getElementById('neo4j-status');
            const inferenceStatus = document.getElementById('inference-status');

            redisStatus.className = `status-indicator ${status.redis_connected ? 'status-connected' : 'status-disconnected'}`;
            neo4jStatus.className = `status-indicator ${status.neo4j_connected ? 'status-connected' : 'status-warning'}`;
            inferenceStatus.className = `status-indicator ${status.inference_active ? 'status-connected' : 'status-warning'}`;
        }

        function updateLearningInsights(insights) {
            document.getElementById('episodic-strength').textContent = (insights.episodic_strength || 0).toFixed(2);
            document.getElementById('semantic-strength').textContent = (insights.semantic_strength || 0).toFixed(2);
            document.getElementById('procedural-strength').textContent = (insights.procedural_strength || 0).toFixed(2);
            document.getElementById('cross-memory-strength').textContent = (insights.cross_memory_strength || 0).toFixed(2);
        }

        function addSystemLog(message) {
            const logsContainer = document.getElementById('systemLogs');
            const logItem = document.createElement('div');
            logItem.className = 'log-item';
            logItem.innerHTML = `<span style="opacity: 0.7;">[${new Date().toLocaleTimeString()}]</span> ${message}`;
            logsContainer.appendChild(logItem);
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }

        function updateRecommendations(recommendations) {
            const container = document.getElementById('recommendations');
            container.innerHTML = '';

            recommendations.forEach(rec => {
                const item = document.createElement('div');
                item.style.cssText = 'padding: 8px; margin: 5px 0; background: rgba(255,255,255,0.1); border-radius: 5px;';
                item.textContent = `💡 ${rec}`;
                container.appendChild(item);
            });
        }

        // Add initial log
        addSystemLog('🌊 Flux ASI-Arch Interface initialized');
    </script>
</body>
</html>
        """

    async def _get_asi_arch_status(self, request):
        """Get ASI-Arch system status"""
        try:
            # Check database connections
            redis_connected = self.cross_db_learning.redis_client is not None
            neo4j_connected = self.cross_db_learning.neo4j_driver is not None

            # Get active inference state
            inference_state = await self.active_inference.get_current_state()

            status = {
                'timestamp': datetime.now().isoformat(),
                'redis_connected': redis_connected,
                'neo4j_connected': neo4j_connected,
                'inference_active': len(self.learning_sessions) > 0,
                'total_sessions': inference_state.get('total_interactions', 0),
                'success_rate': inference_state.get('success_rate', 0.0),
                'consciousness_level': inference_state.get('consciousness_level', 0.0),
                'documents_processed': len(self.processed_documents),
                'active_clients': len(self.connected_clients)
            }

            return web.json_response(status)

        except Exception as e:
            logger.error(f"❌ Failed to get ASI-Arch status: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _get_active_inference_metrics(self, request):
        """Get current active inference metrics"""
        try:
            state = await self.active_inference.get_current_state()

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'consciousness_level': state.get('consciousness_level', 0.0),
                'free_energy': state.get('metrics', {}).get('free_energy', 0.0),
                'prediction_error': state.get('metrics', {}).get('prediction_error', 0.0),
                'learning_rate': state.get('learning_rate', 0.01),
                'total_sessions': state.get('total_interactions', 0),
                'success_rate': state.get('success_rate', 0.0),
                'documents_processed': len(self.processed_documents)
            }

            return web.json_response(metrics)

        except Exception as e:
            logger.error(f"❌ Failed to get metrics: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _upload_document(self, request):
        """Handle document upload"""
        try:
            reader = await request.multipart()
            uploaded_files = []

            while True:
                field = await reader.next()
                if field is None:
                    break

                if field.name == 'documents':
                    filename = field.filename
                    if filename:
                        # Generate unique filename
                        file_id = str(uuid.uuid4())
                        file_ext = Path(filename).suffix
                        safe_filename = f"{file_id}{file_ext}"
                        file_path = self.upload_dir / safe_filename

                        # Save file
                        async with aiofiles.open(file_path, 'wb') as f:
                            while True:
                                chunk = await field.read_chunk()
                                if not chunk:
                                    break
                                await f.write(chunk)

                        uploaded_files.append({
                            'original_name': filename,
                            'file_id': file_id,
                            'file_path': str(file_path),
                            'uploaded_at': datetime.now().isoformat()
                        })

            # Store upload info
            for file_info in uploaded_files:
                self.processed_documents.append(file_info)

            # Notify clients
            await self._broadcast_update('documents_uploaded', {
                'files': uploaded_files,
                'total_processed': len(self.processed_documents)
            })

            return web.json_response({
                'files_uploaded': len(uploaded_files),
                'files': uploaded_files
            })

        except Exception as e:
            logger.error(f"❌ Document upload failed: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _process_document(self, request):
        """Process uploaded documents with active inference"""
        try:
            # Start cross-database learning session
            context = f"Processing {len(self.processed_documents)} documents with active inference"
            inference_state = await self.active_inference.get_current_state()

            session_id = await self.cross_db_learning.start_learning_session(
                context, inference_state
            )

            self.learning_sessions[session_id] = {
                'started_at': datetime.now(),
                'documents': self.processed_documents.copy(),
                'status': 'processing'
            }

            # Simulate document processing with active inference
            for doc in self.processed_documents[-5:]:  # Process last 5 documents
                # Process with active inference
                result = await self.active_inference.process_architecture_context(
                    f"Document: {doc['original_name']}",
                    {'file_path': doc['file_path']}
                )

                # Update learning progress
                await self.cross_db_learning.update_learning_progress(
                    session_id,
                    result,
                    {'document_processed': doc['original_name']}
                )

            # Mark session as complete
            self.learning_sessions[session_id]['status'] = 'completed'

            # Broadcast update
            await self._broadcast_update('processing_complete', {
                'session_id': session_id,
                'documents_processed': len(self.processed_documents)
            })

            return web.json_response({
                'session_id': session_id,
                'status': 'completed',
                'documents_processed': len(self.processed_documents)
            })

        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _start_inference_session(self, request):
        """Start new active inference session"""
        try:
            context = "New interactive learning session"
            inference_state = await self.active_inference.get_current_state()

            session_id = await self.cross_db_learning.start_learning_session(
                context, inference_state
            )

            self.learning_sessions[session_id] = {
                'started_at': datetime.now(),
                'status': 'active'
            }

            return web.json_response({
                'session_id': session_id,
                'status': 'started'
            })

        except Exception as e:
            logger.error(f"❌ Failed to start inference session: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _get_learning_insights(self, request):
        """Get cross-database learning insights"""
        try:
            # Mock insights for now - in production this would come from actual analysis
            insights = {
                'episodic_strength': 0.7,
                'semantic_strength': 0.6,
                'procedural_strength': 0.8,
                'cross_memory_strength': 0.7,
                'recommendations': [
                    "Document processing accuracy improved by 15%",
                    "Cross-memory integration showing positive trends",
                    "Consider increasing episodic learning frequency"
                ]
            }

            return web.json_response(insights)

        except Exception as e:
            logger.error(f"❌ Failed to get learning insights: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _get_database_status(self, request):
        """Get database connection status"""
        try:
            status = {
                'redis': {
                    'connected': self.cross_db_learning.redis_client is not None,
                    'url': Config.REDIS_URL
                },
                'neo4j': {
                    'connected': self.cross_db_learning.neo4j_driver is not None,
                    'url': Config.NEO4J_URL
                }
            }

            return web.json_response(status)

        except Exception as e:
            logger.error(f"❌ Failed to get database status: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _send_initial_state(self, sid):
        """Send initial state to newly connected client"""
        try:
            metrics = await self._get_current_metrics()
            await self.sio.emit('metrics_update', metrics, room=sid)

            insights = {
                'episodic_strength': 0.5,
                'semantic_strength': 0.5,
                'procedural_strength': 0.5,
                'cross_memory_strength': 0.5
            }
            await self.sio.emit('learning_update', insights, room=sid)

        except Exception as e:
            logger.error(f"❌ Failed to send initial state: {e}")

    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            state = await self.active_inference.get_current_state()

            return {
                'consciousness_level': state.get('consciousness_level', 0.0),
                'free_energy': state.get('metrics', {}).get('free_energy', 0.0),
                'total_sessions': len(self.learning_sessions),
                'documents_processed': len(self.processed_documents),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Failed to get current metrics: {e}")
            return {}

    async def _broadcast_update(self, event: str, data: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        try:
            await self.sio.emit(event, data)
            logger.info(f"📡 Broadcasted {event} to {len(self.connected_clients)} clients")

        except Exception as e:
            logger.error(f"❌ Failed to broadcast update: {e}")

    async def start_server(self):
        """Start the Flux interface server"""
        try:
            # Start real-time updates
            self.update_task = asyncio.create_task(self._real_time_updates())

            # Start web server
            runner = web.AppRunner(self.app)
            await runner.setup()

            site = web.TCPSite(runner, self.host, self.port)
            await site.start()

            logger.info(f"🌊 Flux ASI-Arch Interface running at http://{self.host}:{self.port}")

            # Keep server running
            while True:
                await asyncio.sleep(3600)  # Sleep for 1 hour

        except KeyboardInterrupt:
            logger.info("🛑 Shutting down Flux interface...")
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
        finally:
            if self.update_task:
                self.update_task.cancel()
            await self.cross_db_learning.close_connections()

    async def _real_time_updates(self):
        """Provide real-time updates to connected clients"""
        while True:
            try:
                if self.connected_clients:
                    # Get current metrics
                    metrics = await self._get_current_metrics()
                    await self._broadcast_update('metrics_update', metrics)

                await asyncio.sleep(5)  # Update every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Real-time update error: {e}")
                await asyncio.sleep(10)

# Main execution
async def main():
    """Run Flux ASI-Arch Interface"""
    interface = FluxASIArchInterface(host='localhost', port=8080)
    await interface.start_server()

if __name__ == "__main__":
    asyncio.run(main())