"""
Flux Complete Launcher
Starts both backend (FastAPI) and frontend (React), then opens browser.
Press Play/Debug and Flux opens automatically!
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path
import threading
from dotenv import load_dotenv

# Ensure src package is importable when running as a script
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.port_manager import check_port_conflicts  # type: ignore  # noqa: E402

load_dotenv()

FRONTEND_DIR = CURRENT_DIR.parent / "frontend"
FRONTEND_URL = "http://localhost:9243"  # Flux frontend port (configured in vite.config.ts)


def start_frontend():
    """Start the frontend dev server in background."""
    print("🎨 Starting Flux frontend...")
    try:
        subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait a moment for frontend to start
        time.sleep(3)
        print(f"✅ Frontend running at {FRONTEND_URL}")
    except Exception as e:
        print(f"⚠️  Could not start frontend: {e}")
        print("   You can start it manually: cd frontend && npm run dev")


def open_browser():
    """Open Flux in the default browser."""
    time.sleep(2)  # Give servers time to fully start
    print(f"🌐 Opening Flux in browser at {FRONTEND_URL}")
    webbrowser.open(FRONTEND_URL)


def main() -> None:
    """Start the complete Flux stack: backend + frontend + browser."""
    port_status = check_port_conflicts()
    allocated_ports = port_status.get("allocated_ports", {})
    default_port = allocated_ports.get("backend_api", 9127)

    port = int(os.getenv("FLUX_BACKEND_PORT", default_port))
    host = os.getenv("HOST", "127.0.0.1")

    if not port_status.get("all_ports_available", True):
        print("⚠️ Port conflicts detected:")
        for notification in port_status.get("notifications", []):
            print(f"  - {notification}")
        print(f"✅ Auto-resolved to port {port}")
    else:
        print(f"✅ Flux backend starting on port {port}")

    # Start frontend in separate thread
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    frontend_thread.start()

    # Open browser in separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    print("🚀 Flux is launching...")
    print(f"   Backend API: http://{host}:{port}")
    print(f"   Frontend UI: {FRONTEND_URL}")
    print(f"   API Docs: http://{host}:{port}/docs")
    print("\n✨ Press CTRL+C to stop Flux\n")

    # Start backend (this blocks)
    import uvicorn
    uvicorn.run(
        "src.app_factory:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
        factory=False,
    )


if __name__ == "__main__":
    main()
