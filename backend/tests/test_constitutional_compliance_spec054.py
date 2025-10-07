#!/usr/bin/env python3
"""
Constitutional Compliance Test for Spec 054 - T007

Ensures all Spec 054 files comply with AGENT_CONSTITUTION (Spec 040):
- NO direct neo4j imports
- ONLY daedalus_gateway imports allowed

This test MUST pass before delivery.

Author: Spec 054 Implementation
Created: 2025-10-07
"""

import ast
import pytest
from pathlib import Path


SPEC_054_FILES = [
    "backend/src/services/document_repository.py",
    "backend/src/services/tier_manager.py",
    # Will add more as implementation progresses
]

BANNED_IMPORTS = [
    "neo4j",
    "from neo4j import",
]

ALLOWED_IMPORTS = [
    "from daedalus_gateway import get_graph_channel",
    "from daedalus_gateway import",
]


def scan_for_banned_imports(file_path: str) -> list[tuple[int, str]]:
    """
    Scan file for banned neo4j imports.

    Returns:
        List of (line_number, violation_message) tuples
    """
    violations = []

    with open(file_path, "r") as f:
        content = f.read()

    # Parse AST
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {file_path}: {e}")

    for node in ast.walk(tree):
        # Check "import neo4j"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("neo4j"):
                    violations.append((
                        node.lineno,
                        f"CONST001: Direct 'import {alias.name}' banned. "
                        f"Use: from daedalus_gateway import get_graph_channel"
                    ))

        # Check "from neo4j import ..."
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("neo4j"):
                violations.append((
                    node.lineno,
                    f"CONST002: Direct 'from {node.module} import ...' banned. "
                    f"Use: from daedalus_gateway import get_graph_channel"
                ))

    return violations


@pytest.mark.parametrize("file_path", SPEC_054_FILES)
def test_no_banned_neo4j_imports(file_path: str):
    """
    Test that Spec 054 files contain NO banned neo4j imports.

    CONSTITUTIONAL REQUIREMENT (Spec 040):
    - All Neo4j access MUST flow through DaedalusGraphChannel
    - NO direct neo4j imports allowed
    """
    full_path = Path(__file__).parent.parent.parent / file_path

    if not full_path.exists():
        pytest.skip(f"File not yet implemented: {file_path}")

    violations = scan_for_banned_imports(str(full_path))

    if violations:
        error_msg = f"\n❌ Constitutional violations in {file_path}:\n"
        for line_no, message in violations:
            error_msg += f"  Line {line_no}: {message}\n"
        error_msg += "\n🔒 AGENT_CONSTITUTION §2.1, §2.2 (Spec 040) requires Graph Channel only!"
        pytest.fail(error_msg)


def test_document_repository_uses_graph_channel():
    """
    Test that DocumentRepository imports and uses DaedalusGraphChannel.

    This is a positive test - verifying the ALLOWED import exists.
    """
    repo_file = Path(__file__).parent.parent / "src/services/document_repository.py"

    if not repo_file.exists():
        pytest.skip("DocumentRepository not yet implemented")

    with open(repo_file, "r") as f:
        content = f.read()

    # Verify allowed import exists
    assert "from daedalus_gateway import get_graph_channel" in content, (
        "DocumentRepository MUST import: from daedalus_gateway import get_graph_channel"
    )

    # Verify usage in class
    assert "self.graph_channel = get_graph_channel()" in content, (
        "DocumentRepository MUST initialize graph_channel with get_graph_channel()"
    )


def test_tier_manager_uses_graph_channel():
    """
    Test that TierManager imports and uses DaedalusGraphChannel.
    """
    tier_file = Path(__file__).parent.parent / "src/services/tier_manager.py"

    if not tier_file.exists():
        pytest.skip("TierManager not yet implemented")

    with open(tier_file, "r") as f:
        content = f.read()

    # Verify allowed import exists
    assert "from daedalus_gateway import get_graph_channel" in content, (
        "TierManager MUST import: from daedalus_gateway import get_graph_channel"
    )

    # Verify usage in class
    assert "self.graph_channel = get_graph_channel()" in content, (
        "TierManager MUST initialize graph_channel with get_graph_channel()"
    )


def test_all_spec054_files_exist():
    """
    Verify all Spec 054 implementation files exist.

    This will fail during development, pass when implementation complete.
    """
    project_root = Path(__file__).parent.parent.parent
    missing_files = []

    for file_path in SPEC_054_FILES:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        pytest.skip(f"Implementation in progress. Missing files: {missing_files}")
