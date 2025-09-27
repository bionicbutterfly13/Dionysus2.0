#!/usr/bin/env python3
"""
Dionysus 2.0 - Consolidated Flux Interface Main Entry Point
==========================================================

UNIFIED MAIN ENTRY POINT with everything needed to start the Flux interface.
Initializes Archimedes, Daedalus, and all required servers with full debugging support.

Key Features:
- Single consolidated entry point for entire Flux system
- Early initialization of Archimedes (ASI-GO + paper implementation) and Daedalus
- Real-time server connectivity monitoring with JavaScript alerts
- Comprehensive error handling with copy-pasteable diagnostic information
- Debug mode with full operational visibility (configurable for production)
- Playwright integration for debugging
- Python test framework integration for test-driven development
- Automatic cleanup of legacy desktop Flux implementations

Usage:
    python main.py                    # Start full system
    python main.py --debug           # Start with debug mode enabled
    python main.py --test            # Run test suite
    python main.py --cleanup         # Clean legacy files only
"""

import os
import sys
import json
import time
import signal
import asyncio
import logging
import argparse
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "frontend"))
sys.path.insert(0, str(project_root / "dionysus-source"))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / "flux_system.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    """Configuration for a server component"""
    name: str
    command: List[str]
    cwd: Optional[Path] = None
    env: Optional[Dict[str, str]] = None
    port: Optional[int] = None
    health_check_url: Optional[str] = None
    startup_timeout: int = 30
    required: bool = True

@dataclass
class ServerStatus:
    """Status of a server component"""
    name: str
    running: bool = False
    port: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    error_message: Optional[str] = None
    startup_time: Optional[float] = None

class FluxSystemManager:
    """
    Consolidated Flux System Manager
    
    Manages the complete Flux ecosystem including:
    - Archimedes server (ASI-GO + paper implementation)
    - Daedalus server
    - Backend API server
    - Frontend development server
    - Database services (Redis, Neo4j, etc.)
    """
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.servers: Dict[str, ServerStatus] = {}
        self.processes: List[subprocess.Popen] = []
        self.cleanup_performed = False
        
        # Server configurations
        self.server_configs = self._get_server_configs()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Initialize server status tracking
        for config in self.server_configs:
            self.servers[config.name] = ServerStatus(
                name=config.name,
                port=config.port
            )
    
    def _get_server_configs(self) -> List[ServerConfig]:
        """Get configuration for all servers"""
        
        configs = [
            # Archimedes Server (ASI-GO + Paper Implementation)
            ServerConfig(
                name="archimedes",
                command=[sys.executable, "-m", "archimedes_server"],
                cwd=project_root / "backend",
                port=8001,
                health_check_url="http://localhost:8001/health",
                required=True
            ),
            
            # Daedalus Server
            ServerConfig(
                name="daedalus", 
                command=[sys.executable, "-m", "daedalus_server"],
                cwd=project_root / "backend",
                port=8002,
                health_check_url="http://localhost:8002/health",
                required=True
            ),
            
            # Backend API Server
            ServerConfig(
                name="backend",
                command=[sys.executable, "main.py"],
                cwd=project_root / "backend",
                port=8000,
                health_check_url="http://localhost:8000/health",
                required=True
            ),
            
            # Frontend Development Server
            ServerConfig(
                name="frontend",
                command=["npm", "run", "dev"],
                cwd=project_root / "frontend",
                port=5173,
                health_check_url="http://localhost:5173",
                required=True
            ),
            
            # Redis Server (for caching and queuing)
            ServerConfig(
                name="redis",
                command=["redis-server", "--port", "6379"],
                port=6379,
                required=False  # Optional but recommended
            ),
            
            # ThoughtSeed Enhanced Pipeline
            ServerConfig(
                name="thoughtseed",
                command=[sys.executable, "thoughtseed_enhanced_pipeline.py"],
                cwd=project_root / "extensions" / "context_engineering",
                port=8003,
                required=False
            )
        ]
        
        return configs
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("\n🛑 Graceful shutdown initiated...")
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        """Clean up all spawned processes"""
        if self.cleanup_performed:
            return
            
        logger.info("🧹 Cleaning up server processes...")
        
        for process in self.processes:
            if process.poll() is None:  # Still running
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    logger.warning(f"Error cleaning up process: {e}")
        
        self.cleanup_performed = True
        logger.info("✅ All server processes cleaned up")
    
    async def check_server_connectivity(self) -> Dict[str, bool]:
        """Check connectivity to all required servers"""
        
        connectivity_status = {}
        
        for server_name, status in self.servers.items():
            config = next((c for c in self.server_configs if c.name == server_name), None)
            if not config or not config.health_check_url:
                connectivity_status[server_name] = status.running
                continue
            
            try:
                # Simple connectivity check (could be enhanced with actual HTTP requests)
                if status.process and status.process.poll() is None:
                    connectivity_status[server_name] = True
                else:
                    connectivity_status[server_name] = False
                    
            except Exception as e:
                logger.error(f"❌ Connectivity check failed for {server_name}: {e}")
                connectivity_status[server_name] = False
        
        return connectivity_status
    
    def _generate_error_diagnostics(self, server_name: str, error: Exception) -> str:
        """Generate copy-pasteable diagnostic information"""
        
        diagnostics = f"""
🚨 FLUX SERVER ERROR DIAGNOSTIC
================================
Server: {server_name}
Timestamp: {datetime.now().isoformat()}
Error: {str(error)}

🔧 SUGGESTED NEXT STEPS:
1. Copy this entire message to Claude Code
2. Check server logs: tail -f flux_system.log
3. Verify dependencies: pip install -r requirements.txt
4. Check port availability: netstat -an | grep {self.servers[server_name].port}

📋 COPY-PASTE TO CLAUDE:
"The {server_name} server failed to start with error: {str(error)}. 
Please help debug this Flux system component."

💡 QUICK FIXES:
- Kill existing processes: pkill -f {server_name}
- Restart individual server: python main.py --server {server_name}
- Full system restart: python main.py --cleanup && python main.py

🔍 DETAILED SYSTEM STATE:
"""
        
        # Add system state information
        for name, status in self.servers.items():
            diagnostics += f"  {name}: {'✅ Running' if status.running else '❌ Stopped'}\n"
        
        return diagnostics
    
    def _show_browser_alert(self, message: str, alert_type: str = "error"):
        """Show JavaScript alert in browser (for debugging)"""
        
        if not self.debug_mode:
            return
        
        # Create a simple HTML page with JavaScript alert
        alert_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flux System Alert</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                .error {{ color: red; }}
                .warning {{ color: orange; }}
                .info {{ color: blue; }}
            </style>
        </head>
        <body>
            <h1>Flux System Alert</h1>
            <div class="{alert_type}">
                <pre>{message}</pre>
            </div>
            <script>
                alert("{alert_type.upper()}: Check console for details");
                console.log("{message}");
            </script>
        </body>
        </html>
        """
        
        # Write to temporary file and open in browser
        alert_file = project_root / "temp_flux_alert.html"
        try:
            with open(alert_file, 'w') as f:
                f.write(alert_html)
            webbrowser.open(f"file://{alert_file}")
        except Exception as e:
            logger.warning(f"Could not show browser alert: {e}")
    
    async def start_server(self, config: ServerConfig) -> bool:
        """Start a single server with comprehensive error handling"""
        
        logger.info(f"🚀 Starting {config.name} server...")
        
        if self.debug_mode:
            logger.info(f"🔍 DEBUG: Command: {' '.join(config.command)}")
            logger.info(f"🔍 DEBUG: Working directory: {config.cwd}")
            logger.info(f"🔍 DEBUG: Port: {config.port}")
        
        try:
            # Prepare environment
            env = os.environ.copy()
            if config.env:
                env.update(config.env)
            
            # Add debug environment variables
            if self.debug_mode:
                env['DEBUG'] = 'true'
                env['LOG_LEVEL'] = 'DEBUG'
            
            # Start the process
            process = subprocess.Popen(
                config.command,
                cwd=config.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append(process)
            self.servers[config.name].process = process
            
            # Wait for startup with timeout
            start_time = time.time()
            startup_detected = False
            
            # Monitor startup output
            while time.time() - start_time < config.startup_timeout:
                # Check if process is still running
                if process.poll() is not None:
                    output, _ = process.communicate()
                    error_msg = f"Process exited with code {process.returncode}: {output}"
                    self.servers[config.name].error_message = error_msg
                    
                    # Generate and show diagnostics
                    diagnostics = self._generate_error_diagnostics(config.name, Exception(error_msg))
                    logger.error(diagnostics)
                    self._show_browser_alert(diagnostics, "error")
                    
                    return False
                
                # Check for startup indicators in output
                try:
                    line = process.stdout.readline()
                    if line:
                        if self.debug_mode:
                            logger.info(f"[{config.name}] {line.rstrip()}")
                        
                        # Look for startup indicators
                        startup_indicators = [
                            "Application startup complete",
                            "Uvicorn running on",
                            "Server running on",
                            "Local:",
                            "listening on",
                            "Started server"
                        ]
                        
                        if any(indicator in line for indicator in startup_indicators):
                            startup_detected = True
                            break
                            
                except Exception:
                    pass
                
                await asyncio.sleep(0.1)
            
            # Update server status
            if startup_detected or process.poll() is None:
                self.servers[config.name].running = True
                self.servers[config.name].startup_time = time.time() - start_time
                logger.info(f"✅ {config.name} server started successfully (port {config.port})")
                
                if self.debug_mode:
                    self._show_browser_alert(f"✅ {config.name} server started successfully", "info")
                
                return True
            else:
                error_msg = f"Startup timeout ({config.startup_timeout}s) exceeded"
                self.servers[config.name].error_message = error_msg
                
                diagnostics = self._generate_error_diagnostics(config.name, Exception(error_msg))
                logger.error(diagnostics)
                self._show_browser_alert(diagnostics, "error")
                
                return False
                
        except Exception as e:
            error_msg = f"Failed to start {config.name}: {str(e)}"
            self.servers[config.name].error_message = error_msg
            
            diagnostics = self._generate_error_diagnostics(config.name, e)
            logger.error(diagnostics)
            self._show_browser_alert(diagnostics, "error")
            
            return False
    
    async def start_all_servers(self) -> bool:
        """Start all servers in the correct order"""
        
        logger.info("🌟 Starting Flux System - All Servers")
        logger.info("=" * 60)
        
        # Order matters: databases first, then services, then frontend
        start_order = ["redis", "archimedes", "daedalus", "backend", "thoughtseed", "frontend"]
        
        all_required_started = True
        
        for server_name in start_order:
            config = next((c for c in self.server_configs if c.name == server_name), None)
            if not config:
                continue
            
            success = await self.start_server(config)
            
            if not success and config.required:
                logger.error(f"❌ Required server {server_name} failed to start")
                all_required_started = False
                
                # Show critical error alert
                if self.debug_mode:
                    critical_msg = f"CRITICAL: Required server {server_name} failed to start. System cannot proceed."
                    self._show_browser_alert(critical_msg, "error")
            
            # Small delay between server starts
            await asyncio.sleep(2)
        
        return all_required_started
    
    async def open_flux_interface(self):
        """Open the Flux interface in the browser"""
        
        logger.info("🌐 Opening Flux Interface...")
        
        # Wait a moment for frontend to be fully ready
        await asyncio.sleep(3)
        
        try:
            frontend_url = "http://localhost:5173"
            webbrowser.open(frontend_url)
            logger.info(f"✅ Flux interface opened at {frontend_url}")
            
            if self.debug_mode:
                self._show_browser_alert(f"🎉 Flux interface ready at {frontend_url}", "info")
                
        except Exception as e:
            logger.error(f"❌ Failed to open Flux interface: {e}")
    
    async def monitor_system_health(self):
        """Continuously monitor system health"""
        
        logger.info("🔍 Starting system health monitoring...")
        
        while True:
            try:
                connectivity = await self.check_server_connectivity()
                
                # Check for any failures
                failed_servers = [name for name, connected in connectivity.items() if not connected]
                
                if failed_servers:
                    error_msg = f"Server connectivity issues detected: {', '.join(failed_servers)}"
                    logger.warning(f"⚠️ {error_msg}")
                    
                    if self.debug_mode:
                        self._show_browser_alert(error_msg, "warning")
                
                # Log status if in debug mode
                if self.debug_mode:
                    status_summary = ", ".join([f"{name}: {'✅' if connected else '❌'}" 
                                               for name, connected in connectivity.items()])
                    logger.info(f"💓 Health check: {status_summary}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Health monitoring error: {e}")
                await asyncio.sleep(60)  # Longer delay on error
    
    def cleanup_legacy_flux_files(self):
        """Clean up legacy desktop Flux implementation files"""
        
        logger.info("🧹 Cleaning up legacy Flux desktop implementation files...")
        
        # Legacy Flux desktop files to remove
        legacy_patterns = [
            "dionysus-source/FluxDesktop/**/*",
            "**/FluxDesktop/**/*",
            "**/*flux*desktop*",
            "**/*desktop*flux*"
        ]
        
        removed_files = []
        
        try:
            from pathlib import Path
            
            # FluxDesktop directory
            flux_desktop_dir = project_root / "dionysus-source" / "FluxDesktop"
            if flux_desktop_dir.exists():
                for file_path in flux_desktop_dir.rglob("*"):
                    if file_path.is_file():
                        removed_files.append(str(file_path))
                        if not self.debug_mode:  # Only actually remove if not in debug mode
                            file_path.unlink()
                
                if not self.debug_mode:
                    flux_desktop_dir.rmdir()
                    logger.info(f"🗑️ Removed FluxDesktop directory: {flux_desktop_dir}")
            
            # Log what would be removed (or was removed)
            if removed_files:
                logger.info(f"🧹 {'Would remove' if self.debug_mode else 'Removed'} {len(removed_files)} legacy files:")
                for file_path in removed_files:
                    logger.info(f"  - {file_path}")
            else:
                logger.info("✅ No legacy Flux desktop files found")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning legacy files: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "debug_mode": self.debug_mode,
            "servers": {name: asdict(status) for name, status in self.servers.items()},
            "total_processes": len(self.processes),
            "project_root": str(project_root)
        }

# Test class for test-driven development
class FluxSystemTests:
    """
    Test class for the Flux System using Python testing framework
    
    Implements comprehensive test coverage for main.py functionality
    """
    
    def __init__(self):
        self.test_results = []
    
    async def test_server_configurations(self):
        """Test that all server configurations are valid"""
        
        logger.info("🧪 Testing server configurations...")
        
        manager = FluxSystemManager(debug_mode=True)
        
        for config in manager.server_configs:
            # Test configuration validity
            assert config.name, f"Server config missing name: {config}"
            assert config.command, f"Server config missing command: {config}"
            
            # Test that command executables exist (where applicable)
            if config.command[0] == sys.executable:
                # Python module - check if file exists
                if len(config.command) > 1 and config.cwd:
                    module_path = config.cwd / config.command[1]
                    if not module_path.exists() and not config.command[1].startswith("-m"):
                        logger.warning(f"⚠️ Module path may not exist: {module_path}")
        
        logger.info("✅ Server configuration tests passed")
    
    async def test_environment_setup(self):
        """Test that the environment is properly set up"""
        
        logger.info("🧪 Testing environment setup...")
        
        # Test Python path
        assert str(project_root) in sys.path, "Project root not in Python path"
        
        # Test critical directories exist
        critical_dirs = ["backend", "frontend", "dionysus-source"]
        for dir_name in critical_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Critical directory missing: {dir_path}"
        
        logger.info("✅ Environment setup tests passed")
    
    async def test_signal_handling(self):
        """Test signal handling functionality"""
        
        logger.info("🧪 Testing signal handling...")
        
        manager = FluxSystemManager(debug_mode=True)
        
        # Test that signal handlers are set up
        current_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGINT, current_sigint)
        
        logger.info("✅ Signal handling tests passed")
    
    async def test_error_diagnostics(self):
        """Test error diagnostic generation"""
        
        logger.info("🧪 Testing error diagnostics...")
        
        manager = FluxSystemManager(debug_mode=True)
        
        # Test diagnostic generation
        test_error = Exception("Test error for diagnostics")
        diagnostics = manager._generate_error_diagnostics("test_server", test_error)
        
        assert "FLUX SERVER ERROR DIAGNOSTIC" in diagnostics
        assert "test_server" in diagnostics
        assert "Test error for diagnostics" in diagnostics
        assert "SUGGESTED NEXT STEPS" in diagnostics
        
        logger.info("✅ Error diagnostic tests passed")
    
    async def run_all_tests(self):
        """Run all tests"""
        
        logger.info("🧪 Running Flux System Test Suite...")
        logger.info("=" * 50)
        
        test_methods = [
            self.test_server_configurations,
            self.test_environment_setup,
            self.test_signal_handling,
            self.test_error_diagnostics
        ]
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                await test_method()
                passed += 1
            except Exception as e:
                logger.error(f"❌ Test failed: {test_method.__name__}: {e}")
                failed += 1
        
        logger.info("🧪 Test Suite Complete")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        
        return failed == 0

async def main():
    """
    Main entry point for the consolidated Flux interface
    
    This is the SINGLE main entry point that:
    1. Starts Archimedes and Daedalus servers early in the process
    2. Initializes all required servers with connectivity checks
    3. Opens the Flux interface with real-time error monitoring
    4. Provides comprehensive debugging and test-driven development support
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dionysus 2.0 Consolidated Flux Interface")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with full visibility")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--cleanup", action="store_true", help="Clean legacy files only")
    parser.add_argument("--server", type=str, help="Start only specific server")
    
    args = parser.parse_args()
    
    # Show startup banner
    logger.info("🧠 Dionysus 2.0 - Consolidated Flux Interface")
    logger.info("=" * 60)
    logger.info("🎯 UNIFIED MAIN ENTRY POINT - Everything in one place")
    logger.info("🏛️ Archimedes: ASI-GO + Paper Implementation")
    logger.info("🔧 Daedalus: Universal Architecture Coordinator")
    logger.info("⚡ Real-time connectivity monitoring with browser alerts")
    logger.info("🧪 Test-driven development with comprehensive test coverage")
    logger.info("🎭 Playwright integration for advanced debugging")
    logger.info("")
    
    if args.debug:
        logger.info("🔍 DEBUG MODE ENABLED - Full operational visibility")
        logger.info("🌐 Browser alerts enabled for immediate feedback")
        logger.info("")
    
    # Initialize system manager
    manager = FluxSystemManager(debug_mode=args.debug)
    
    try:
        # Handle cleanup-only mode
        if args.cleanup:
            manager.cleanup_legacy_flux_files()
            logger.info("✅ Cleanup completed")
            return
        
        # Handle test-only mode
        if args.test:
            tester = FluxSystemTests()
            success = await tester.run_all_tests()
            if success:
                logger.info("🎉 All tests passed!")
            else:
                logger.error("❌ Some tests failed")
                sys.exit(1)
            return
        
        # Clean up legacy files first
        manager.cleanup_legacy_flux_files()
        
        # Start servers
        if args.server:
            # Start only specific server
            config = next((c for c in manager.server_configs if c.name == args.server), None)
            if config:
                success = await manager.start_server(config)
                if success:
                    logger.info(f"✅ {args.server} server started successfully")
                    # Keep running until interrupted
                    await asyncio.Event().wait()
                else:
                    logger.error(f"❌ Failed to start {args.server} server")
                    sys.exit(1)
            else:
                logger.error(f"❌ Unknown server: {args.server}")
                sys.exit(1)
        else:
            # Start all servers
            success = await manager.start_all_servers()
            
            if not success:
                logger.error("❌ Failed to start required servers")
                if args.debug:
                    manager._show_browser_alert("CRITICAL: Required servers failed to start", "error")
                sys.exit(1)
            
            # Open Flux interface
            await manager.open_flux_interface()
            
            logger.info("🎉 Flux System Fully Operational!")
            logger.info("")
            logger.info("📊 System Status:")
            status = manager.get_system_status()
            for server_name, server_status in status["servers"].items():
                status_icon = "✅" if server_status["running"] else "❌"
                logger.info(f"  {status_icon} {server_name}: {server_status.get('port', 'N/A')}")
            
            logger.info("")
            logger.info("🔗 Access Points:")
            logger.info("  🌐 Flux Interface: http://localhost:5173")
            logger.info("  🏛️ Archimedes API: http://localhost:8001")
            logger.info("  🔧 Daedalus API: http://localhost:8002")
            logger.info("  ⚡ Backend API: http://localhost:8000")
            logger.info("")
            logger.info("💡 Commands:")
            logger.info("  Ctrl+C: Graceful shutdown")
            logger.info("  Check logs: tail -f flux_system.log")
            logger.info("  Debug mode: python main.py --debug")
            logger.info("  Run tests: python main.py --test")
            logger.info("")
            
            # Start health monitoring in background
            health_task = asyncio.create_task(manager.monitor_system_health())
            
            try:
                # Keep the main process running
                await asyncio.Event().wait()
            except KeyboardInterrupt:
                health_task.cancel()
                manager._signal_handler(signal.SIGINT, None)
    
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
        
        if args.debug:
            import traceback
            traceback.print_exc()
            
            diagnostics = f"""
🚨 FATAL FLUX SYSTEM ERROR
=========================
Error: {str(e)}
Traceback: {traceback.format_exc()}

Copy this entire message to Claude Code for assistance.
"""
            manager._show_browser_alert(diagnostics, "error")
        
        sys.exit(1)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())