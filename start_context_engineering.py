#!/usr/bin/env python3
"""
🚀 Context Engineering Startup Script
=====================================

Simple script to start ASI-Arch with Context Engineering enhancements.

Usage:
    python start_context_engineering.py                    # Full system with dashboard
    python start_context_engineering.py --no-dashboard     # No web dashboard
    python start_context_engineering.py --test             # Test mode with mock data
    python start_context_engineering.py --port 9090        # Custom dashboard port

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - Startup Script
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from extensions.context_engineering.live_integration import (
    start_enhanced_pipeline, 
    stop_enhanced_pipeline,
    test_live_integration
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def print_banner():
    """Print startup banner"""
    print("""
🌊 ═══════════════════════════════════════════════════════════════
   ASI-Arch Enhanced with Context Engineering
   
   🧠 Consciousness Detection    🌊 River Metaphor Analysis
   🎯 Attractor Basin Mapping   📊 Real-time Dashboard
   🔗 Seamless ASI-Arch Integration
   
   Ready to discover conscious architectures!
═══════════════════════════════════════════════════════════════ 🚀
    """)

def print_status(service):
    """Print system status"""
    print("\n📊 System Status:")
    print("   ✅ Context Engineering: ACTIVE")
    print("   ✅ Consciousness Detection: ENABLED")
    print("   ✅ River Metaphor: FLOWING")
    print("   ✅ Attractor Basins: MAPPING")
    
    if hasattr(service, 'dashboard') and service.dashboard:
        print(f"   🌐 Dashboard: http://localhost:{service.dashboard_port}")
    
    if hasattr(service, 'pipeline_enhancer'):
        print("   🔗 ASI-Arch Integration: ENHANCED")
    else:
        print("   🔧 ASI-Arch Integration: STANDALONE MODE")

def main():
    parser = argparse.ArgumentParser(description="Start ASI-Arch with Context Engineering")
    parser.add_argument('--port', type=int, default=8080, 
                       help='Dashboard port (default: 8080)')
    parser.add_argument('--no-dashboard', action='store_true', 
                       help='Disable web dashboard')
    parser.add_argument('--no-integration', action='store_true', 
                       help='Disable direct ASI-Arch pipeline integration')
    parser.add_argument('--test', action='store_true', 
                       help='Run test mode with mock data')
    parser.add_argument('--quiet', action='store_true', 
                       help='Reduce log output')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    print_banner()
    
    if args.test:
        print("🧪 Starting in test mode...")
        asyncio.run(test_live_integration())
        return
    
    print("🚀 Initializing Context Engineering...")
    
    try:
        # Start the enhanced pipeline
        service = start_enhanced_pipeline(
            dashboard_port=args.port,
            enable_dashboard=not args.no_dashboard,
            integrate_with_asi_arch=not args.no_integration
        )
        
        print_status(service)
        
        print("\n🎯 Context Engineering is now active!")
        print("   Your ASI-Arch experiments will now include:")
        print("     • Enhanced evolution context with river metaphor insights")
        print("     • Real-time consciousness detection and tracking")
        print("     • Attractor basin identification and mapping")
        print("     • Knowledge graph construction of architecture relationships")
        
        if not args.no_dashboard:
            print(f"\n📱 Open your browser to http://localhost:{args.port} to view the dashboard")
        
        print("\n⏸️  Press Ctrl+C to stop the system...")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down Context Engineering...")
            stop_enhanced_pipeline(service)
            print("✅ Shutdown complete. Thank you for exploring consciousness!")
            
    except Exception as e:
        print(f"\n❌ Error starting Context Engineering: {e}")
        print("   Check the logs above for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
