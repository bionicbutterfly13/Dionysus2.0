"""
Flux Backend Main Entry Point
Starts the FastAPI server with uvicorn.
"""

import uvicorn
import os
from dotenv import load_dotenv
from src.utils.port_manager import get_flux_backend_port, check_port_conflicts

load_dotenv()

if __name__ == "__main__":
    # Use port manager for conflict detection and auto-resolution
    port = get_flux_backend_port()
    host = os.getenv("HOST", "127.0.0.1")

    # Check for port conflicts and notify
    port_status = check_port_conflicts()
    if not port_status['all_ports_available']:
        print(f"⚠️ Port conflicts detected:")
        for notification in port_status['notifications']:
            print(f"  - {notification}")
        print(f"✅ Auto-resolved to port {port}")
    else:
        print(f"✅ All Flux ports available - starting on preferred port {port}")

    uvicorn.run(
        "app_factory:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )