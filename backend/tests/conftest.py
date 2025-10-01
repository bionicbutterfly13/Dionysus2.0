"""
Pytest configuration for backend tests.
Ensures proper Python path setup for imports.
"""
import sys
from pathlib import Path

# Add backend directory to Python path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
