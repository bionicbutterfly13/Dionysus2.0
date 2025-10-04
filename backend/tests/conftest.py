"""
Pytest configuration for backend tests.
Centralizes Python path setup - NO sys.path in individual test files!

Per Constitution Article I, Section 1.4:
- MANDATORY for all test directories
- Eliminates sys.path manipulation in individual test files
- Prevents fragile import chains through __init__.py
"""
import sys
from pathlib import Path

# Add backend/src to path ONCE for all tests
backend_src = Path(__file__).parent.parent / "src"
if str(backend_src) not in sys.path:
    sys.path.insert(0, str(backend_src))

# Verify path was added correctly
assert backend_src.exists(), f"Backend src path doesn't exist: {backend_src}"
