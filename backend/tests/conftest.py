"""Pytest configuration for backend tests."""

import sys
from pathlib import Path

import pytest_asyncio
from httpx import AsyncClient

# Add backend root and backend/src to path ONCE for all tests
backend_root = Path(__file__).parent.parent
backend_src = backend_root / "src"

for path in (backend_root, backend_src):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

assert backend_root.exists(), f"Backend root path doesn't exist: {backend_root}"
assert backend_src.exists(), f"Backend src path doesn't exist: {backend_src}"

from src.app_factory import create_app  # noqa: E402 - imported after sys.path mutation


@pytest_asyncio.fixture
async def test_client():
    """Provide an AsyncClient backed by the FastAPI application."""
    from httpx import ASGITransport

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
