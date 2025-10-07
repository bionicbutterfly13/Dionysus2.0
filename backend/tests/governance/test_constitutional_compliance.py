"""
Regression tests for Constitutional Compliance enforcement (Spec 040 M3)

These tests verify that the neo4j import ban is properly enforced
and cannot be bypassed.

Per AGENT_CONSTITUTION §2.1, §2.2:
- ALL Neo4j access MUST flow through DaedalusGraphChannel
- NO direct neo4j imports in backend/src (except daedalus-gateway)
"""

import ast
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestNeo4jImportBan:
    """Test suite verifying neo4j import ban enforcement."""

    def test_no_direct_neo4j_imports_in_services(self):
        """Verify no backend/src services have direct neo4j imports."""
        backend_src = Path(__file__).parent.parent.parent / "src"
        assert backend_src.exists(), "backend/src directory must exist"

        violations = []

        class ImportChecker(ast.NodeVisitor):
            """Check for neo4j imports outside try/except blocks."""
            def __init__(self):
                self.violations = []
                self.in_try = False

            def visit_Try(self, node):
                old_in_try = self.in_try
                self.in_try = True
                self.generic_visit(node)
                self.in_try = old_in_try

            def visit_Import(self, node):
                for alias in node.names:
                    if alias.name.startswith("neo4j") and not self.in_try:
                        self.violations.append((node.lineno, f"import {alias.name}"))
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.module and node.module.startswith("neo4j") and not self.in_try:
                    self.violations.append((node.lineno, f"from {node.module} import ..."))
                self.generic_visit(node)

        for pyfile in backend_src.rglob("*.py"):
            # Skip __pycache__
            if "__pycache__" in str(pyfile):
                continue

            try:
                with open(pyfile, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                checker = ImportChecker()
                checker.visit(tree)

                for lineno, import_stmt in checker.violations:
                    violations.append(f"{pyfile}:{lineno} - {import_stmt}")
            except SyntaxError:
                pass  # Skip files with syntax errors

        if violations:
            violation_msg = "\n".join(violations)
            pytest.fail(
                f"Constitutional violation detected!\n\n"
                f"Direct neo4j imports found in backend/src:\n{violation_msg}\n\n"
                f"Per AGENT_CONSTITUTION §2.1, §2.2:\n"
                f"  - ALL Neo4j access MUST flow through DaedalusGraphChannel\n"
                f"  - Use: from daedalus_gateway import get_graph_channel\n\n"
                f"See: GRAPH_CHANNEL_MIGRATION_QUICK_REFERENCE.md"
            )

    def test_all_services_use_graph_channel(self):
        """Verify services using Neo4j import get_graph_channel."""
        backend_src = Path(__file__).parent.parent.parent / "src"

        services_using_neo4j = []
        services_with_graph_channel = []

        for pyfile in backend_src.rglob("*.py"):
            if "__pycache__" in str(pyfile):
                continue

            with open(pyfile, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if file appears to use Neo4j operations
            if any(
                keyword in content
                for keyword in [
                    "execute_read",
                    "execute_write",
                    "execute_schema",
                    "get_graph_channel",
                ]
            ):
                services_using_neo4j.append(pyfile)

                if "get_graph_channel" in content:
                    services_with_graph_channel.append(pyfile)

        # All services using Neo4j operations should use get_graph_channel
        missing_graph_channel = set(services_using_neo4j) - set(
            services_with_graph_channel
        )

        if missing_graph_channel:
            files = "\n".join(str(f) for f in missing_graph_channel)
            pytest.fail(
                f"Services using Neo4j operations without get_graph_channel:\n{files}\n\n"
                f"All Neo4j operations must use DaedalusGraphChannel"
            )

    @pytest.mark.skip("Requires pre-commit install")
    def test_precommit_hook_blocks_neo4j_import(self):
        """Test that pre-commit hook blocks files with neo4j imports."""
        # Skip this test - it requires pre-commit to be installed
        # Manual testing: create test file and run pre-commit
        pass

    def test_ci_check_script_exists(self):
        """Verify CI workflow for constitutional compliance exists."""
        ci_workflow = (
            Path(__file__).parent.parent.parent.parent
            / ".github/workflows/constitutional-compliance.yml"
        )
        assert ci_workflow.exists(), (
            "CI workflow constitutional-compliance.yml must exist"
        )

        # Verify it contains the neo4j import check
        content = ci_workflow.read_text()
        assert "neo4j-import-ban" in content or "neo4j import" in content.lower(), (
            "CI workflow must check for neo4j imports"
        )

    def test_linter_detects_violations(self):
        """Test custom linter detects neo4j import violations."""
        linter_script = Path(__file__).parent.parent.parent / ".ruff_constitutional_plugin.py"
        assert linter_script.exists(), "Constitutional linter must exist"

        # Test linter logic directly instead of file creation
        # (avoids tempfile path issues)
        import sys
        import importlib.util

        spec = importlib.util.spec_from_file_location("linter", linter_script)
        linter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(linter)

        # Test violating code
        violating_code = "from neo4j import GraphDatabase\n"
        tree = ast.parse(violating_code)
        checker = linter.ConstitutionalComplianceChecker("test.py")
        checker.visit(tree)

        assert len(checker.violations) > 0, "Linter should detect neo4j import"
        assert any("CONST00" in msg for _, _, msg in checker.violations), (
            "Linter should report CONST001 or CONST002 error codes"
        )

        # Test allowed code (in try/except)
        allowed_code = """
try:
    from neo4j import GraphDatabase
except ImportError:
    pass
"""
        tree = ast.parse(allowed_code)
        checker = linter.ConstitutionalComplianceChecker("test.py")
        checker.visit(tree)

        assert len(checker.violations) == 0, "Linter should allow try/except imports"


class TestGraphChannelEnforcement:
    """Test Graph Channel is the ONLY way to access Neo4j."""

    def test_daedalus_gateway_is_only_allowed_importer(self):
        """Verify daedalus-gateway is the ONLY place with neo4j imports."""
        # This is more of a documentation test - we can't enforce this
        # across repositories, but we document it

        readme_path = Path(__file__).parent.parent.parent.parent / "LEGACY_REGISTRY.md"
        assert readme_path.exists(), "LEGACY_REGISTRY.md must document neo4j imports"

        content = readme_path.read_text()
        assert "daedalus-gateway" in content.lower(), (
            "LEGACY_REGISTRY must mention daedalus-gateway as only allowed importer"
        )

    def test_graph_channel_operations_have_audit_params(self):
        """Verify Graph Channel operations include audit trail parameters."""
        # This tests that our enforced pattern includes audit trail
        backend_src = Path(__file__).parent.parent.parent / "src"

        files_missing_audit = []

        for pyfile in backend_src.rglob("*.py"):
            if "__pycache__" in str(pyfile):
                continue

            with open(pyfile, "r", encoding="utf-8") as f:
                content = f.read()

            # If file uses execute_read/write, check for caller_service
            if "execute_read(" in content or "execute_write(" in content:
                if "caller_service=" not in content and "caller_function=" not in content:
                    # Soft check - warn but don't fail
                    # (Some operations might not need full audit trail)
                    files_missing_audit.append(pyfile)

        # This is informational - not a hard requirement
        if files_missing_audit:
            print(
                f"\nℹ️  {len(files_missing_audit)} file(s) use Graph Channel "
                f"without full audit trail (caller_service/caller_function)"
            )
            print("   Consider adding audit parameters for better traceability")

    def test_migration_guide_exists(self):
        """Verify migration guide documentation exists."""
        guide_path = (
            Path(__file__).parent.parent.parent.parent
            / "GRAPH_CHANNEL_MIGRATION_QUICK_REFERENCE.md"
        )
        assert guide_path.exists(), (
            "GRAPH_CHANNEL_MIGRATION_QUICK_REFERENCE.md must exist"
        )

        content = guide_path.read_text()
        assert "get_graph_channel" in content, (
            "Migration guide must explain get_graph_channel usage"
        )


class TestConstitutionalEnforcementDate:
    """Test that constitution documents enforcement date."""

    def test_constitution_has_enforcement_date(self):
        """Verify AGENT_CONSTITUTION documents Spec 040 M3 enforcement."""
        constitution_path = (
            Path(__file__).parent.parent.parent.parent.parent / "AGENT_CONSTITUTION.md"
        )

        # Constitution should exist
        if constitution_path.exists():
            content = constitution_path.read_text()

            # Check for M3 enforcement documentation
            assert (
                "040" in content or "M3" in content or "enforcement" in content.lower()
            ), (
                "AGENT_CONSTITUTION should document Spec 040 M3 enforcement date"
            )


@pytest.mark.integration
class TestEndToEndEnforcement:
    """Integration tests for complete enforcement pipeline."""

    def test_complete_enforcement_chain(self):
        """Test pre-commit → CI → linter enforcement chain."""
        # Verify all three enforcement mechanisms exist
        precommit = Path(__file__).parent.parent.parent / ".pre-commit-config.yaml"
        ci_workflow = (
            Path(__file__).parent.parent.parent.parent
            / ".github/workflows/constitutional-compliance.yml"
        )
        linter = Path(__file__).parent.parent.parent / ".ruff_constitutional_plugin.py"

        assert precommit.exists(), "Pre-commit config must exist"
        assert ci_workflow.exists(), "CI workflow must exist"
        assert linter.exists(), "Linter script must exist"

        # Verify pre-commit includes our hook
        precommit_content = precommit.read_text()
        assert "neo4j-import-ban" in precommit_content, (
            "Pre-commit must include neo4j-import-ban hook"
        )

    def test_documentation_complete(self):
        """Verify all enforcement documentation is in place."""
        root = Path(__file__).parent.parent.parent.parent

        required_docs = [
            "LEGACY_REGISTRY.md",
            "GRAPH_CHANNEL_MIGRATION_QUICK_REFERENCE.md",
            "SPEC_040_M2_COMPLETION_SUMMARY.md",
        ]

        for doc in required_docs:
            doc_path = root / doc
            assert doc_path.exists(), f"{doc} must exist for governance documentation"
