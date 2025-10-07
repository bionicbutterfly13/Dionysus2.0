#!/usr/bin/env python3
"""
Contract Tests Configuration - Cleanup fixtures for Neo4j test data
"""

import pytest
import pytest_asyncio
import sys
from pathlib import Path

# Add backend root and src to path
backend_root = Path(__file__).parent.parent.parent
backend_src = backend_root / "src"

for path in (backend_root, backend_src):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Import after path setup (uses src prefix like main conftest)
from daedalus_gateway import get_graph_channel  # noqa: E402


@pytest_asyncio.fixture(autouse=True)
async def cleanup_test_documents():
    """
    Automatically clean up test documents before and after each test.

    This prevents duplicate document errors in contract tests.
    """
    channel = get_graph_channel()

    # Test document IDs to clean up
    test_ids = [
        'doc_test_001',
        'doc_test_duplicate',
        'doc_test_duplicate_2',
        'doc_test_invalid',
        'doc_test_performance',
        'doc_filename_test_1',
        'doc_filename_test_2',
        'doc_metadata_test',
        'doc_metadata_test_2'
    ]

    # Cleanup BEFORE test - delete all test documents in one query
    cleanup_query = '''
    MATCH (d:Document)
    WHERE d.document_id IN $test_ids
    DETACH DELETE d
    '''
    try:
        await channel.execute_write(cleanup_query, {'test_ids': test_ids})
    except Exception:
        pass  # Ignore if documents don't exist

    yield  # Run the test

    # Cleanup AFTER test - delete all test documents in one query
    try:
        await channel.execute_write(cleanup_query, {'test_ids': test_ids})
    except Exception:
        pass  # Ignore if documents don't exist
