"""
Governance and Constitutional Compliance Tests

This package contains tests that enforce AGENT_CONSTITUTION requirements.
These are REGRESSION tests - they prevent backsliding on constitutional compliance.

Test Categories:
- Constitutional Compliance: Verifies Spec 040 M3 enforcement
- Import Bans: Ensures no direct neo4j imports in backend/src
- Graph Channel Usage: Validates DaedalusGraphChannel is sole access path
- Audit Trail: Checks caller_service/caller_function parameters
- Documentation: Ensures governance docs are complete
"""
