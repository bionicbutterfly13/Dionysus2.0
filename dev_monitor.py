#!/usr/bin/env python
"""
Real-Time Development Monitor for Flux
Provides live visibility into system state, data flow, and test results.
"""
import asyncio
import websockets
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import threading
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SystemMetric:
    """A single system metric with timestamp."""
    name: str
    value: Any
    timestamp: float
    category: str
    unit: str = ""
    details: Dict[str, Any] = None


class FluxDevMonitor:
    """Real-time monitoring system for Flux development."""

    def __init__(self):
        self.metrics: List[SystemMetric] = []
        self.active_tests: Dict[str, Any] = {}
        self.port_status: Dict[str, Any] = {}
        self.database_status: Dict[str, Any] = {}
        self.running = False
        self.websocket_clients = set()

    def add_metric(self, name: str, value: Any, category: str, unit: str = "", details: Dict = None):
        """Add a new metric to the monitoring system."""
        metric = SystemMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            category=category,
            unit=unit,
            details=details or {}
        )

        self.metrics.append(metric)

        # Keep only last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]

        # Broadcast to websocket clients
        asyncio.create_task(self.broadcast_metric(metric))

    async def broadcast_metric(self, metric: SystemMetric):
        """Broadcast metric to all connected websocket clients."""
        if self.websocket_clients:
            message = {
                "type": "metric",
                "data": asdict(metric)
            }

            # Send to all connected clients
            disconnected = set()
            for client in self.websocket_clients:
                try:
                    await client.send(json.dumps(message))
                except:
                    disconnected.add(client)

            # Remove disconnected clients
            self.websocket_clients -= disconnected

    async def monitor_tests(self):
        """Monitor test execution in real-time."""
        while self.running:
            try:
                # Run quick test check
                result = subprocess.run([
                    sys.executable, "-m", "pytest",
                    "backend/tests/test_port_management.py",
                    "backend/tests/test_database_health.py",
                    "-v", "--tb=no", "-q"
                ], capture_output=True, text=True, timeout=10)

                passed = result.returncode == 0
                test_count = len([line for line in result.stdout.split('\n') if line.strip().endswith('PASSED')])

                self.add_metric("tests_passing", passed, "tests", details={
                    "test_count": test_count,
                    "output": result.stdout[-200:]  # Last 200 chars
                })

            except subprocess.TimeoutExpired:
                self.add_metric("tests_timeout", True, "tests")
            except Exception as e:
                self.add_metric("tests_error", str(e), "tests")

            await asyncio.sleep(5)  # Check every 5 seconds

    async def monitor_ports(self):
        """Monitor port allocation and conflicts."""
        while self.running:
            try:
                # Import our port manager
                sys.path.append('backend/src')
                from utils.port_manager import check_port_conflicts

                status = check_port_conflicts()
                self.port_status = status

                self.add_metric("ports_available", status['all_ports_available'], "ports", details=status)
                self.add_metric("port_conflicts", status['conflicts_detected'], "ports", "count")

            except Exception as e:
                self.add_metric("port_check_error", str(e), "ports")

            await asyncio.sleep(3)  # Check every 3 seconds

    async def monitor_databases(self):
        """Monitor database health."""
        while self.running:
            try:
                sys.path.append('backend/src')
                from services.database_health import get_database_health

                health = get_database_health()
                self.database_status = health

                self.add_metric("db_overall_status", health['overall_status'], "database")
                self.add_metric("db_healthy_count", health['healthy_count'], "database", "count")

                # Individual database status
                for db_name in ['neo4j', 'redis', 'qdrant']:
                    db_health = health[db_name]
                    self.add_metric(f"db_{db_name}_status", db_health['status'], "database")
                    if 'response_time_ms' in db_health and db_health['response_time_ms']:
                        self.add_metric(f"db_{db_name}_response_time",
                                      db_health['response_time_ms'], "performance", "ms")

            except Exception as e:
                self.add_metric("database_check_error", str(e), "database")

            await asyncio.sleep(10)  # Check every 10 seconds

    async def websocket_handler(self, websocket, path):
        """Handle websocket connections from the dashboard."""
        self.websocket_clients.add(websocket)
        print(f"📱 Dashboard connected from {websocket.remote_address}")

        try:
            # Send current status
            await websocket.send(json.dumps({
                "type": "status",
                "data": {
                    "ports": self.port_status,
                    "databases": self.database_status,
                    "recent_metrics": [asdict(m) for m in self.metrics[-20:]]
                }
            }))

            # Keep connection alive
            await websocket.wait_closed()

        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.websocket_clients.discard(websocket)
            print("📱 Dashboard disconnected")

    def print_terminal_status(self):
        """Print status to terminal for non-web monitoring."""
        os.system('clear' if os.name == 'posix' else 'cls')

        print("🖥️  Flux Development Monitor")
        print("=" * 60)
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        print()

        # Recent metrics by category
        categories = {}
        for metric in self.metrics[-20:]:
            if metric.category not in categories:
                categories[metric.category] = []
            categories[metric.category].append(metric)

        for category, metrics in categories.items():
            print(f"📊 {category.upper()}:")
            for metric in metrics[-5:]:  # Last 5 per category
                timestamp = datetime.fromtimestamp(metric.timestamp).strftime('%H:%M:%S')
                icon = "✅" if metric.name.endswith("_passing") and metric.value else "❌" if metric.name.endswith("_error") else "📈"
                print(f"  {icon} [{timestamp}] {metric.name}: {metric.value} {metric.unit}")
            print()

        # Current status summary
        print("🎯 CURRENT STATUS:")
        if self.port_status:
            icon = "✅" if self.port_status['all_ports_available'] else "⚠️"
            print(f"  {icon} Ports: {self.port_status.get('conflicts_detected', 0)} conflicts")

        if self.database_status:
            healthy = self.database_status.get('healthy_count', 0)
            total = self.database_status.get('total_count', 3)
            icon = "✅" if healthy == total else "⚠️" if healthy > 0 else "❌"
            print(f"  {icon} Databases: {healthy}/{total} healthy")

        print(f"\n📡 WebSocket clients: {len(self.websocket_clients)}")
        print("\n🔗 Connect dashboard: http://localhost:8765")
        print("Press Ctrl+C to stop monitoring")

    async def run_monitoring(self):
        """Run all monitoring tasks."""
        self.running = True

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self.monitor_tests()),
            asyncio.create_task(self.monitor_ports()),
            asyncio.create_task(self.monitor_databases())
        ]

        # Start terminal display
        def terminal_loop():
            while self.running:
                self.print_terminal_status()
                time.sleep(2)

        terminal_thread = threading.Thread(target=terminal_loop, daemon=True)
        terminal_thread.start()

        # Start WebSocket server
        websocket_server = websockets.serve(self.websocket_handler, "localhost", 8765)

        print("🚀 Flux Development Monitor starting...")
        print("📡 WebSocket server on ws://localhost:8765")

        try:
            await asyncio.gather(*tasks, websocket_server)
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitor...")
            self.running = False


def main():
    """Main entry point."""
    monitor = FluxDevMonitor()

    try:
        asyncio.run(monitor.run_monitoring())
    except KeyboardInterrupt:
        print("\n✅ Monitor stopped")


if __name__ == "__main__":
    main()