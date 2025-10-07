"""
Ruff plugin for Constitutional Compliance enforcement (Spec 040 M3)

This custom linter detects direct neo4j imports that violate
AGENT_CONSTITUTION Sections 2.1 and 2.2.

Usage:
    ruff check --select=CONST backend/src/

Constitutional Requirements:
    - ALL Neo4j access MUST flow through DaedalusGraphChannel
    - ONLY daedalus-gateway may import neo4j directly
    - Backend services MUST use: from daedalus_gateway import get_graph_channel
"""

import ast
import sys
from pathlib import Path
from typing import Iterator, Tuple


class ConstitutionalComplianceChecker(ast.NodeVisitor):
    """AST visitor that detects banned neo4j imports."""

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: list[Tuple[int, int, str]] = []
        self.in_try_except = False

    def visit_Try(self, node: ast.Try) -> None:
        """Track when we're inside a try/except block."""
        old_in_try = self.in_try_except
        self.in_try_except = True
        self.generic_visit(node)
        self.in_try_except = old_in_try

    def visit_Import(self, node: ast.Import) -> None:
        """Check 'import neo4j' statements."""
        for alias in node.names:
            if alias.name.startswith("neo4j"):
                # Allow deprecated imports in try/except blocks (backwards compatibility)
                if not self.in_try_except:
                    self.violations.append(
                        (
                            node.lineno,
                            node.col_offset,
                            f"CONST001 Direct neo4j import banned (AGENT_CONSTITUTION §2.1, §2.2). "
                            f"Use: from daedalus_gateway import get_graph_channel",
                        )
                    )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check 'from neo4j import ...' statements."""
        if node.module and node.module.startswith("neo4j"):
            # Allow deprecated imports in try/except blocks (backwards compatibility)
            if not self.in_try_except:
                self.violations.append(
                    (
                        node.lineno,
                        node.col_offset,
                        f"CONST002 Direct neo4j import banned (AGENT_CONSTITUTION §2.1, §2.2). "
                        f"Use: from daedalus_gateway import get_graph_channel",
                    )
                )
        self.generic_visit(node)


def check_file(filepath: Path) -> list[Tuple[int, int, str]]:
    """
    Check a Python file for constitutional violations.

    Returns:
        List of (line, col, message) tuples for violations
    """
    # Skip allowed paths
    path_str = str(filepath)
    if any(
        excluded in path_str
        for excluded in ["tests/", "daedalus-gateway", "backup/deprecated"]
    ):
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        checker = ConstitutionalComplianceChecker(str(filepath))
        checker.visit(tree)
        return checker.violations

    except SyntaxError:
        # Skip files with syntax errors (will be caught by other tools)
        return []


def main() -> int:
    """Run constitutional compliance check on all backend/src files."""
    violations_found = False

    # Check all Python files in backend/src
    backend_src = Path("backend/src")
    if not backend_src.exists():
        print("❌ backend/src directory not found")
        return 1

    print("🔍 Checking Constitutional Compliance (AGENT_CONSTITUTION §2.1, §2.2)...")

    for pyfile in backend_src.rglob("*.py"):
        violations = check_file(pyfile)
        if violations:
            violations_found = True
            print(f"\n❌ {pyfile}")
            for line, col, message in violations:
                print(f"   Line {line}, Col {col}: {message}")

    if violations_found:
        print("\n" + "━" * 70)
        print("🚨 CONSTITUTIONAL COMPLIANCE FAILURE")
        print("━" * 70)
        print("\nDirect neo4j imports detected in backend code.")
        print("All graph access MUST flow through DaedalusGraphChannel.")
        print("\nMigration guide: GRAPH_CHANNEL_MIGRATION_QUICK_REFERENCE.md")
        print("Constitution: AGENT_CONSTITUTION.md")
        print("━" * 70)
        return 1

    print("✅ All files compliant - No constitutional violations detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
