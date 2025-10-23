#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

echo "🔍 Validating constitutional dependencies with ${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PYCODE'
import importlib

def check_numpy():
    numpy = importlib.import_module("numpy")
    version = numpy.__version__
    if not version.startswith("2."):
        raise SystemExit(f"❌ NumPy {version} violates constitution (must be 2.x)")
    print(f"✅ NumPy {version} compliant")

def check_optional(package_name):
    try:
        module = importlib.import_module(package_name)
    except ModuleNotFoundError:
        print(f"ℹ️ {package_name} not installed (skipping)")
        return
    version = getattr(module, "__version__", "unknown")
    print(f"ℹ️ {package_name}: {version}")

check_numpy()
for optional in ("torch", "transformers"):
    check_optional(optional)
PYCODE

echo "✅ Environment validation complete"
