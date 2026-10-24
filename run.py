"""
Start both the FastAPI backend and Streamlit frontend together.
Usage:  python run.py   (run from the project root)
"""
import subprocess
import sys
import os
import time
import socket

ROOT = os.path.dirname(os.path.abspath(__file__))

# Always use the venv Python
if os.name == "nt":
    VENV_PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
else:
    VENV_PYTHON = os.path.join(ROOT, "venv", "bin", "python")

PYTHON = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable


# ── Helpers ──────────────────────────────────────────────────────────────────

def kill_port(port: int):
    """Kill any process listening on the given port (Windows + Unix)."""
    try:
        if os.name == "nt":
            result = subprocess.run(
                f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{port} ^| findstr LISTENING\') do taskkill /PID %a /F',
                shell=True, capture_output=True
            )
        else:
            subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
    except Exception:
        pass


def wait_for_port(port: int, timeout: int = 30) -> bool:
    """Poll until a port is open (service is ready), or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


# ── Main ─────────────────────────────────────────────────────────────────────

print("=" * 55)
print("  DocIntel Platform - Starting All Services")
print("=" * 55)
print(f"\n  Python: {PYTHON}")

# 1. Free up ports from any previous run
print("\n[0/2] Clearing old processes on ports 8000 and 8501...")
kill_port(8000)
kill_port(8501)
time.sleep(1)

# 2. Start FastAPI (lazy init: Qdrant opens on first request, not here)
print("[1/2] Starting FastAPI backend on http://localhost:8000 ...")
backend = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "app.main:app",
     "--host", "0.0.0.0",
     "--port", "8000"],
    cwd=ROOT,
)

# Wait until FastAPI port is actually accepting connections
print("      Waiting for FastAPI to be ready...", end="", flush=True)
ready = wait_for_port(8000, timeout=30)
if ready:
    print(" ready!")
else:
    print(" timed out! Starting Streamlit anyway.")

# 3. Start Streamlit (now FastAPI is up, so USE_API=True, Qdrant stays with FastAPI)
print("[2/2] Starting Streamlit frontend on http://localhost:8501 ...")
frontend = subprocess.Popen(
    [PYTHON, "-m", "streamlit", "run", "streamlit_app.py",
     "--server.port", "8501",
     "--server.headless", "true"],
    cwd=ROOT,
)

print("\nBoth services are running.")
print("  API docs  ->  http://localhost:8000/docs")
print("  App UI    ->  http://localhost:8501")
print("\nPress Ctrl+C to stop everything.\n")

try:
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    backend.terminate()
    frontend.terminate()
    backend.wait()
    frontend.wait()
    print("Stopped.")
