"""
Debug Pipeline Launcher
=======================

Helper script for local development: starts the Flux backend + frontend, opens
the debug pipeline dashboard, and queues a sample document through the new
debug processing API so you immediately see events streaming.

Usage: press the Debug ▶️ button in VS Code (see .vscode/launch.json entry) or
run `python debug_pipeline_launcher.py` from the project root.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from pathlib import Path

try:
    from utils.port_manager import check_port_conflicts, PortManager  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for packaged path
    from backend.src.utils.port_manager import check_port_conflicts, PortManager  # type: ignore

# ---------------------------------------------------------------------------
# Workspace paths / imports
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
SRC_DIR = BACKEND_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

FRONTEND_URL_TEMPLATE = "http://localhost:{port}"
_port_checker = PortManager()


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

def start_frontend(port: int) -> None:
    """Launch the Vite dev server for the Flux frontend."""
    if not _port_checker.is_port_available(port):
        print(f"ℹ️  Frontend dev server already running on port {port}; skipping auto-start.")
        return

    print(f"🎨 Starting Flux frontend on port {port}...")
    try:
        subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)
        print(f"✅ Frontend running at http://localhost:{port}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️  Could not start frontend automatically: {exc}")
        print("   Start manually with: cd frontend && npm run dev")


def open_browser_routes(frontend_port: int) -> None:
    """Open both the main dashboard and the debug pipeline panel."""
    time.sleep(2)
    base_url = FRONTEND_URL_TEMPLATE.format(port=frontend_port)
    print(f"🌐 Opening Flux UI at {base_url}")
    webbrowser.open(base_url)

    time.sleep(1)
    debug_url = f"{base_url}/debug/pipeline"
    print(f"🌐 Opening Debug Pipeline at {debug_url}")
    webbrowser.open(debug_url)


def wait_for_backend(host: str, port: int, attempts: int = 30) -> bool:
    """Poll the health endpoint until the backend is ready."""
    health_url = f"http://{host}:{port}/health"
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(health_url, timeout=2):
                return True
        except Exception:
            time.sleep(1)
    return False


def queue_sample_document(host: str, port: int) -> None:
    """
    Send a small synthetic document through the debug processor so the panel
    lights up automatically once the stack is live.
    """
    ready = wait_for_backend(host, port)
    if not ready:
        print("⚠️  Backend health check timed out; skipping demo enqueue.")
        return

    endpoint = f"http://{host}:{port}/api/debug/process-document"
    boundary = f"----FluxDebugBoundary{uuid.uuid4().hex}"
    sample_text = (
        "Flux Debug Pipeline Demo\n"
        "=========================\n\n"
        "This document is generated automatically by debug_pipeline_launcher.py\n"
        "to demonstrate the real-time LangGraph telemetry. You should see queue,\n"
        "node transitions, concept activations, and quality metrics flowing in\n"
        "the debug dashboard.\n"
    )

    parts = [
        f"--{boundary}",
        'Content-Disposition: form-data; name="file"; filename="debug-demo.txt"',
        "Content-Type: text/plain",
        "",
        sample_text,
        f"--{boundary}--",
        "",
    ]
    body = "\r\n".join(parts).encode("utf-8")

    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if 200 <= response.status < 300:
                print("📡 Sample debug document queued successfully.")
            else:  # pragma: no cover - defensive
                print(f"⚠️  Unexpected response code: {response.status}")
    except urllib.error.HTTPError as exc:
        print(f"⚠️  Failed to queue debug document (HTTP {exc.code}): {exc.reason}")
    except urllib.error.URLError as exc:
        print(f"⚠️  Failed to queue debug document: {exc.reason}")


# ---------------------------------------------------------------------------
# Main launch routine
# ---------------------------------------------------------------------------

def main() -> None:
    port_status = check_port_conflicts()
    allocated_ports = port_status.get("allocated_ports", {})
    port = int(os.getenv("FLUX_BACKEND_PORT", allocated_ports.get("backend_api", 9127)))
    host = os.getenv("HOST", "127.0.0.1")
    frontend_port = allocated_ports.get("frontend_dev", 9243)
    frontend_url = FRONTEND_URL_TEMPLATE.format(port=frontend_port)

    if not port_status.get("all_ports_available", True):
        print("⚠️  Port conflicts detected:")
        for notification in port_status.get("notifications", []):
            print(f"  - {notification}")
        print(f"✅ Auto-resolved to port {port}")
    else:
        print(f"✅ Flux backend starting on port {port}")

    is_primary_launch = os.environ.get("FLUX_DEBUG_LAUNCHER_PRIMARY") != "1"

    if is_primary_launch:
        os.environ["FLUX_DEBUG_LAUNCHER_PRIMARY"] = "1"

        threading.Thread(target=start_frontend, args=(frontend_port,), daemon=True).start()
        threading.Thread(target=open_browser_routes, args=(frontend_port,), daemon=True).start()
        threading.Thread(
            target=queue_sample_document,
            args=(host, port),
            daemon=True,
        ).start()
    else:
        print("🔁 Detected uvicorn reload worker; skipping frontend/browser launch.")

    print("🚀 Flux Debug Pipeline launcher starting...")
    print(f"   Backend API: http://{host}:{port}")
    print(f"   Debug Panel: {frontend_url}/debug/pipeline")
    print("\n✨ Press CTRL+C to stop Flux\n")

    import uvicorn

    uvicorn.run(
        "backend.src.app_factory:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
        factory=False,
    )


if __name__ == "__main__":
    main()
