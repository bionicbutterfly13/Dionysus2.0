"""
Test suite for verifying complete BMAD removal from codebase.

These tests should FAIL before BMAD removal (RED phase).
They should PASS after BMAD is completely removed (GREEN phase).

Test Strategy:
1. Search for BMAD references in active code
2. Verify BMAD-specific files are removed/archived
3. Check Neo4j schema has no BMAD nodes
4. Verify documentation cleanup
5. Ensure Gemini agents are removed
"""

import pytest
from pathlib import Path
import subprocess
import re
from typing import List, Set


# Root directory of the project
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Allowed historical/archive directories where BMAD references are OK
ALLOWED_HISTORICAL_PATHS = {
    "backup",
    "archive",
    "deprecated",
    ".git",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
}

# File extensions to search
CODE_EXTENSIONS = [".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".toml", ".yaml", ".yml", ".json"]


def is_in_allowed_path(file_path: Path) -> bool:
    """Check if file is in an allowed historical/archive directory."""
    parts = file_path.parts
    return any(allowed in parts for allowed in ALLOWED_HISTORICAL_PATHS)


def find_files_with_pattern(pattern: str, case_sensitive: bool = False) -> List[Path]:
    """
    Find all files containing the pattern in active codebase.

    Returns list of file paths that match (excluding allowed historical paths).
    """
    matches = []
    flags = 0 if case_sensitive else re.IGNORECASE

    for ext in CODE_EXTENSIONS:
        for file_path in PROJECT_ROOT.rglob(f"*{ext}"):
            # Skip allowed historical paths
            if is_in_allowed_path(file_path):
                continue

            try:
                content = file_path.read_text(encoding='utf-8')
                if re.search(pattern, content, flags):
                    matches.append(file_path.relative_to(PROJECT_ROOT))
            except (UnicodeDecodeError, PermissionError):
                # Skip binary or protected files
                continue

    return matches


def test_no_bmad_in_python_code():
    """Verify no BMAD references in Python code (excluding historical docs)."""
    matches = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        # Skip allowed paths
        if is_in_allowed_path(py_file):
            continue

        # Skip this test file itself
        if py_file.name == "test_bmad_removal.py":
            continue

        try:
            content = py_file.read_text(encoding='utf-8')

            # Search for BMAD references (case-insensitive)
            if re.search(r'\bbmad\b', content, re.IGNORECASE):
                # Get line numbers for better debugging
                lines_with_bmad = [
                    (i + 1, line.strip())
                    for i, line in enumerate(content.split('\n'))
                    if re.search(r'\bbmad\b', line, re.IGNORECASE)
                ]
                matches.append((py_file.relative_to(PROJECT_ROOT), lines_with_bmad))

        except (UnicodeDecodeError, PermissionError):
            continue

    assert len(matches) == 0, (
        f"Found BMAD references in {len(matches)} Python files:\n" +
        "\n".join([
            f"  {path}:\n" + "\n".join([f"    Line {ln}: {line}" for ln, line in lines])
            for path, lines in matches
        ])
    )


def test_no_bmad_in_typescript_code():
    """Verify no BMAD references in TypeScript/JavaScript code."""
    matches = []

    for file_path in PROJECT_ROOT.rglob("*.ts"):
        if is_in_allowed_path(file_path):
            continue

        try:
            content = file_path.read_text(encoding='utf-8')
            if re.search(r'\bbmad\b', content, re.IGNORECASE):
                lines_with_bmad = [
                    (i + 1, line.strip())
                    for i, line in enumerate(content.split('\n'))
                    if re.search(r'\bbmad\b', line, re.IGNORECASE)
                ]
                matches.append((file_path.relative_to(PROJECT_ROOT), lines_with_bmad))
        except (UnicodeDecodeError, PermissionError):
            continue

    assert len(matches) == 0, (
        f"Found BMAD references in {len(matches)} TypeScript files:\n" +
        "\n".join([
            f"  {path}:\n" + "\n".join([f"    Line {ln}: {line}" for ln, line in lines])
            for path, lines in matches
        ])
    )


def test_no_bmad_in_markdown_docs():
    """Verify no BMAD references in active Markdown documentation."""
    matches = []

    for md_file in PROJECT_ROOT.rglob("*.md"):
        if is_in_allowed_path(md_file):
            continue

        # Skip README files that might have historical context
        if md_file.name in ["CHANGELOG.md", "HISTORY.md", "MIGRATION_HISTORY.md", "BMAD_REMOVAL_COMPLETE.md"]:
            continue

        # Allow historical references like "Replaced BMAD with OpenSpec"
        try:
            content = md_file.read_text(encoding='utf-8')
            # Exclude lines that are historical context (past tense replacements)
            content_lines = content.split('\n')
            bmad_lines = [line for line in content_lines if re.search(r'\bbmad\b', line, re.IGNORECASE)]
            # Filter out acceptable historical references
            unacceptable_lines = [
                line for line in bmad_lines
                if not re.search(r'replaced\s+bmad|removed\s+bmad|migrated.*bmad', line, re.IGNORECASE)
            ]
            if not unacceptable_lines:
                continue  # All references are historical/acceptable
        except (UnicodeDecodeError, PermissionError):
            pass  # Will be caught in main check below

        try:
            content = md_file.read_text(encoding='utf-8')
            if re.search(r'\bbmad\b', content, re.IGNORECASE):
                lines_with_bmad = [
                    (i + 1, line.strip())
                    for i, line in enumerate(content.split('\n'))
                    if re.search(r'\bbmad\b', line, re.IGNORECASE)
                ]
                matches.append((md_file.relative_to(PROJECT_ROOT), lines_with_bmad))
        except (UnicodeDecodeError, PermissionError):
            continue

    assert len(matches) == 0, (
        f"Found BMAD references in {len(matches)} Markdown files:\n" +
        "\n".join([
            f"  {path}:\n" + "\n".join([f"    Line {ln}: {line}" for ln, line in lines])
            for path, lines in matches
        ])
    )


def test_no_bmad_migration_script():
    """Verify migrate_bmad_to_consciousness.py is archived or removed."""
    migration_script = PROJECT_ROOT / "backend" / "src" / "services" / "migrate_bmad_to_consciousness.py"

    assert not migration_script.exists(), (
        f"BMAD migration script still exists at {migration_script.relative_to(PROJECT_ROOT)}. "
        "It should be moved to backup/deprecated/ or removed."
    )


def test_no_bmad_check_function():
    """Verify check_bmad_migration() function doesn't exist in check_consciousness_systems.py."""
    check_script = PROJECT_ROOT / "backend" / "src" / "services" / "check_consciousness_systems.py"

    if not check_script.exists():
        pytest.skip("check_consciousness_systems.py doesn't exist")

    content = check_script.read_text(encoding='utf-8')

    # Check for function definition
    assert not re.search(r'def\s+check_bmad_migration\s*\(', content), (
        "check_bmad_migration() function still exists in check_consciousness_systems.py"
    )

    # Check for function calls
    assert not re.search(r'check_bmad_migration\s*\(', content), (
        "check_bmad_migration() function is still being called in check_consciousness_systems.py"
    )


def test_gemini_agents_removed():
    """Verify .gemini/commands/agents/ directory is empty or deleted."""
    gemini_agents_dir = PROJECT_ROOT / ".gemini" / "commands" / "agents"

    if not gemini_agents_dir.exists():
        # Directory doesn't exist - that's fine
        return

    # If directory exists, it should be empty
    agent_files = list(gemini_agents_dir.glob("*.md"))

    assert len(agent_files) == 0, (
        f"Found {len(agent_files)} agent files in .gemini/commands/agents/:\n" +
        "\n".join([f"  - {f.name}" for f in agent_files]) +
        "\nThese should be moved to backup/deprecated/"
    )


def test_neo4j_schema_bmad_free():
    """Verify Neo4j schema has no BMAD-specific nodes (Decision, Project with BMAD phases)."""
    schema_file = PROJECT_ROOT / "backend" / "src" / "services" / "neo4j_schema_init.py"

    if not schema_file.exists():
        pytest.skip("neo4j_schema_init.py doesn't exist")

    content = schema_file.read_text(encoding='utf-8')

    # Check for BMAD-specific node labels
    bmad_node_patterns = [
        r'CREATE\s+\(.*:Decision\b',  # Decision nodes
        r'CREATE\s+\(.*:Project\b.*phase.*:.*brainstorm',  # Project nodes with BMAD phases
        r'CREATE\s+\(.*:Project\b.*phase.*:.*model',
        r'CREATE\s+\(.*:Project\b.*phase.*:.*act',
        r'CREATE\s+\(.*:Project\b.*phase.*:.*deploy',
        r'CREATE\s+\(.*:Pattern\b.*bmad',  # Pattern nodes related to BMAD
    ]

    found_patterns = []
    for pattern in bmad_node_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            found_patterns.append(pattern)

    assert len(found_patterns) == 0, (
        f"Found {len(found_patterns)} BMAD-specific patterns in Neo4j schema:\n" +
        "\n".join([f"  - {p}" for p in found_patterns])
    )


def test_claude_md_no_bmad_migration_docs():
    """Verify CLAUDE.md doesn't have BMAD migration workflow sections."""
    claude_md = PROJECT_ROOT / "CLAUDE.md"

    if not claude_md.exists():
        pytest.skip("CLAUDE.md doesn't exist")

    content = claude_md.read_text(encoding='utf-8')

    # Check for BMAD migration sections
    forbidden_sections = [
        r'##.*BMAD.*Migration',
        r'###.*migrate.*bmad',
        r'migrate_bmad_to_consciousness',
        r'check_bmad_migration',
    ]

    found_sections = []
    for pattern in forbidden_sections:
        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
        if matches:
            found_sections.extend(matches)

    assert len(found_sections) == 0, (
        f"Found {len(found_sections)} BMAD migration references in CLAUDE.md:\n" +
        "\n".join([f"  - {s}" for s in found_sections])
    )


def test_no_bmad_in_config_files():
    """Verify no BMAD references in configuration files."""
    config_extensions = [".toml", ".yaml", ".yml", ".json", ".env"]
    matches = []

    for ext in config_extensions:
        for config_file in PROJECT_ROOT.rglob(f"*{ext}"):
            if is_in_allowed_path(config_file):
                continue

            # Skip package-lock.json and similar large generated files
            if config_file.name in ["package-lock.json", "yarn.lock", "poetry.lock"]:
                continue

            try:
                content = config_file.read_text(encoding='utf-8')
                if re.search(r'\bbmad\b', content, re.IGNORECASE):
                    lines_with_bmad = [
                        (i + 1, line.strip())
                        for i, line in enumerate(content.split('\n'))
                        if re.search(r'\bbmad\b', line, re.IGNORECASE)
                    ]
                    matches.append((config_file.relative_to(PROJECT_ROOT), lines_with_bmad))
            except (UnicodeDecodeError, PermissionError):
                continue

    assert len(matches) == 0, (
        f"Found BMAD references in {len(matches)} config files:\n" +
        "\n".join([
            f"  {path}:\n" + "\n".join([f"    Line {ln}: {line}" for ln, line in lines])
            for path, lines in matches
        ])
    )


def test_no_bmad_imports():
    """Verify no imports of BMAD-related modules."""
    import_patterns = [
        r'from\s+.*bmad.*\s+import',
        r'import\s+.*bmad',
        r'from\s+.*migrate_bmad',
    ]

    matches = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        if is_in_allowed_path(py_file):
            continue

        try:
            content = py_file.read_text(encoding='utf-8')

            for pattern in import_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    lines_with_import = [
                        (i + 1, line.strip())
                        for i, line in enumerate(content.split('\n'))
                        if re.search(pattern, line, re.IGNORECASE)
                    ]
                    if lines_with_import:
                        matches.append((py_file.relative_to(PROJECT_ROOT), lines_with_import))
        except (UnicodeDecodeError, PermissionError):
            continue

    assert len(matches) == 0, (
        f"Found BMAD imports in {len(matches)} Python files:\n" +
        "\n".join([
            f"  {path}:\n" + "\n".join([f"    Line {ln}: {line}" for ln, line in lines])
            for path, lines in matches
        ])
    )


def test_summary_report():
    """Generate summary report of all BMAD references found (for debugging)."""
    print("\n" + "=" * 80)
    print("BMAD REMOVAL TEST SUMMARY")
    print("=" * 80)

    # This test always passes - it's just for reporting
    all_matches = find_files_with_pattern(r'\bbmad\b')

    if len(all_matches) == 0:
        print("\n✅ NO BMAD REFERENCES FOUND - All tests should pass!")
    else:
        print(f"\n❌ Found BMAD references in {len(all_matches)} files:")
        for match in all_matches:
            print(f"  - {match}")
        print("\nThese files need BMAD removal before tests pass.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
