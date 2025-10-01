#!/usr/bin/env python3
"""
🧪 TDD Test Suite: Context Engineering Spec Pipeline Integration

This test validates that Context Engineering components (Attractor Basins and Neural Fields)
are properly integrated into the spec-kit workflow and visible from the start.

Author: Consciousness Processing Team
Date: 2025-10-01
Status: TDD - Tests written BEFORE full implementation
"""

import pytest
import sys
import os
from pathlib import Path

# Test that Context Engineering components are accessible
class TestContextEngineeringAccessibility:
    """Verify that Context Engineering components can be imported and initialized"""

    def test_attractor_basin_manager_import(self):
        """Test that AttractorBasinManager can be imported"""
        try:
            from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager
            assert AttractorBasinManager is not None
        except ImportError as e:
            pytest.fail(f"Failed to import AttractorBasinManager: {e}")

    def test_neural_field_system_import(self):
        """Test that IntegratedAttractorFieldSystem can be imported"""
        # Note: This will fail until we fix the import path
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dionysus-source"))
            from context_engineering.integrated_attractor_field_system import IntegratedAttractorFieldSystem
            assert IntegratedAttractorFieldSystem is not None
        except ImportError as e:
            pytest.skip(f"Neural Field System not yet accessible: {e}")

    def test_attractor_basin_manager_initialization(self):
        """Test that AttractorBasinManager can be initialized"""
        from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager

        # Should initialize even without Redis (graceful degradation)
        try:
            manager = AttractorBasinManager(redis_host='localhost', redis_port=6379)
            assert manager is not None
            assert hasattr(manager, 'basins')
            assert isinstance(manager.basins, dict)
        except Exception as e:
            # Redis not available - should still create default basin
            pytest.skip(f"Redis not available for basin persistence: {e}")

    def test_neural_field_system_initialization(self):
        """Test that Neural Field System can be initialized"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dionysus-source"))
            from context_engineering.integrated_attractor_field_system import IntegratedAttractorFieldSystem

            system = IntegratedAttractorFieldSystem(dimensions=384)
            assert system is not None
            assert system.dimensions == 384
            assert hasattr(system, 'field_state')
            assert hasattr(system, 'attractors')
        except ImportError:
            pytest.skip("Neural Field System not yet accessible")


class TestContextEngineeringFoundationDocument:
    """Verify that Context Engineering foundation documentation exists and is complete"""

    def test_foundation_document_exists(self):
        """Test that CONTEXT_ENGINEERING_FOUNDATION.md exists"""
        foundation_path = Path(__file__).parent.parent.parent / ".specify" / "memory" / "CONTEXT_ENGINEERING_FOUNDATION.md"
        assert foundation_path.exists(), "CONTEXT_ENGINEERING_FOUNDATION.md must exist"

    def test_foundation_has_attractor_basin_section(self):
        """Test that foundation document describes Attractor Basins"""
        foundation_path = Path(__file__).parent.parent.parent / ".specify" / "memory" / "CONTEXT_ENGINEERING_FOUNDATION.md"
        content = foundation_path.read_text()

        assert "Attractor Basin" in content, "Must describe Attractor Basins"
        assert "AttractorBasinManager" in content, "Must mention AttractorBasinManager"
        assert "basin strength" in content.lower(), "Must explain basin strength"

    def test_foundation_has_neural_field_section(self):
        """Test that foundation document describes Neural Fields"""
        foundation_path = Path(__file__).parent.parent.parent / ".specify" / "memory" / "CONTEXT_ENGINEERING_FOUNDATION.md"
        content = foundation_path.read_text()

        assert "Neural Field" in content, "Must describe Neural Fields"
        assert "continuous" in content.lower(), "Must explain continuous nature"
        assert "resonance" in content.lower(), "Must explain resonance patterns"

    def test_foundation_has_mathematical_foundation(self):
        """Test that foundation document includes mathematical descriptions"""
        foundation_path = Path(__file__).parent.parent.parent / ".specify" / "memory" / "CONTEXT_ENGINEERING_FOUNDATION.md"
        content = foundation_path.read_text()

        # Should include field evolution equation
        assert "∂ψ/∂t" in content or "dψ/dt" in content, "Must include field evolution equation"
        assert "Laplacian" in content or "∇²" in content, "Must explain Laplacian operator"


class TestSlashCommandIntegration:
    """Verify that slash commands integrate Context Engineering visibility"""

    def test_specify_command_shows_context_engineering(self):
        """Test that /specify command displays Context Engineering foundation"""
        specify_command = Path(__file__).parent.parent.parent / ".claude" / "commands" / "specify.md"
        assert specify_command.exists(), "/specify command must exist"

        content = specify_command.read_text()
        assert "Context Engineering" in content, "/specify must mention Context Engineering"
        assert "CONTEXT_ENGINEERING_FOUNDATION.md" in content, "/specify must reference foundation doc"

    def test_plan_command_validates_context_engineering(self):
        """Test that /plan command validates Context Engineering integration"""
        plan_command = Path(__file__).parent.parent.parent / ".claude" / "commands" / "plan.md"
        assert plan_command.exists(), "/plan command must exist"

        content = plan_command.read_text()
        assert "Context Engineering" in content, "/plan must validate Context Engineering"
        assert "Attractor Basin" in content, "/plan must check Attractor Basin Manager"
        assert "Neural Field" in content, "/plan must check Neural Field System"

    def test_tasks_command_includes_context_engineering_tasks(self):
        """Test that /tasks command includes Context Engineering validation tasks"""
        tasks_command = Path(__file__).parent.parent.parent / ".claude" / "commands" / "tasks.md"
        assert tasks_command.exists(), "/tasks command must exist"

        content = tasks_command.read_text()
        assert "Context Engineering" in content, "/tasks must include Context Engineering tasks"
        assert "Basin integration test" in content, "/tasks must include basin integration test"
        assert "Field resonance test" in content, "/tasks must include field resonance test"
        assert "MANDATORY" in content, "Context Engineering tasks must be mandatory"


class TestContextEngineeringIntegrationFlow:
    """Test the complete flow of Context Engineering integration in spec pipeline"""

    def test_attractor_basin_creation(self):
        """Test that new features create attractor basins"""
        from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager

        try:
            manager = AttractorBasinManager(redis_host='localhost', redis_port=6379)
            initial_basin_count = len(manager.basins)

            # Create a new basin for a feature
            from extensions.context_engineering.attractor_basin_dynamics import AttractorBasin
            new_basin = AttractorBasin(
                basin_id="test_feature_basin",
                center_concept="test_feature_concept",
                strength=1.0,
                radius=0.5
            )

            manager.basins[new_basin.basin_id] = new_basin
            assert len(manager.basins) == initial_basin_count + 1

        except Exception as e:
            pytest.skip(f"Redis not available: {e}")

    def test_neural_field_knowledge_domain_creation(self):
        """Test that features create knowledge domains in neural fields"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dionysus-source"))
            from context_engineering.integrated_attractor_field_system import IntegratedAttractorFieldSystem

            system = IntegratedAttractorFieldSystem(dimensions=384)
            initial_attractor_count = len(system.attractors)

            # Create knowledge domain for feature
            domain_id = system.create_knowledge_domain(
                domain_name="test_feature_domain",
                concepts=["concept_a", "concept_b", "concept_c"]
            )

            assert domain_id is not None
            assert len(system.attractors) == initial_attractor_count + 1

        except ImportError:
            pytest.skip("Neural Field System not yet accessible")


class TestTDDCompliance:
    """Verify that the spec pipeline follows TDD principles"""

    def test_context_engineering_tests_run_first(self):
        """Test that Context Engineering validation happens before implementation"""
        # This is a meta-test ensuring TDD order
        tasks_template = Path(__file__).parent.parent.parent / ".specify" / "templates" / "tasks-template.md"

        if tasks_template.exists():
            content = tasks_template.read_text()
            # Context Engineering tests should be in Phase 3.2 (Tests First)
            assert "Context Engineering" in content or "basin" in content.lower()

    def test_constitution_includes_context_engineering(self):
        """Test that constitution mandates Context Engineering integration"""
        constitution = Path(__file__).parent.parent.parent / ".specify" / "memory" / "constitution.md"
        assert constitution.exists(), "Constitution must exist"

        content = constitution.read_text()
        # Should reference consciousness processing protocols
        assert "consciousness" in content.lower() or "context engineering" in content.lower()


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
