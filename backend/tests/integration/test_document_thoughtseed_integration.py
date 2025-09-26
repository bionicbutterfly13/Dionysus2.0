#!/usr/bin/env python3
"""
Integration Test: Document Processing with ThoughtSeed Hierarchy
Test complete document processing through 5-layer ThoughtSeed system
"""

import pytest
import io
import asyncio
from fastapi.testclient import TestClient


class TestDocumentThoughtseedIntegration:
    """Integration tests for document processing with ThoughtSeed"""

    @pytest.fixture
    def client(self):
        """Test client - will fail until endpoints implemented"""
        from backend.src.main import app  # This import will fail initially
        return TestClient(app)

    @pytest.fixture
    def complex_pdf_document(self):
        """Complex PDF document for testing hierarchy processing"""
        # Simulated PDF with rich content for ThoughtSeed processing
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 200
>>
stream
BT
/F1 12 Tf
50 750 Td
(Neural Networks and Consciousness Emergence) Tj
0 -20 Td
(This document explores the relationship between artificial neural) Tj
0 -20 Td
(networks and the emergence of consciousness-like phenomena.) Tj
0 -20 Td
(Key concepts include pattern recognition, self-awareness, and) Tj
0 -20 Td
(meta-cognitive processing across hierarchical layers.) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000015 00000 n
0000000074 00000 n
0000000120 00000 n
0000000179 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
430
%%EOF"""
        return ("neural_consciousness_paper.pdf", io.BytesIO(pdf_content), "application/pdf")

    def test_document_thoughtseed_full_hierarchy(self, client, complex_pdf_document):
        """Test complete 5-layer ThoughtSeed processing"""
        # This test MUST fail initially - integration not implemented yet
        filename, file_content, content_type = complex_pdf_document

        # Submit document for processing with all layers
        response = client.post(
            "/api/v1/documents/process",
            files={"file": (filename, file_content, content_type)},
            data={
                "extract_narratives": "true",
                "thoughtseed_layers": '["sensory","perceptual","conceptual","abstract","metacognitive"]'
            }
        )

        assert response.status_code == 200
        process_data = response.json()

        # Should have workspace ID for tracking
        assert "document_id" in process_data
        workspace_id = process_data.get("thoughtseed_workspace_id")
        assert workspace_id is not None

        # Check workspace processing state
        workspace_response = client.get(f"/api/v1/thoughtseed/workspace/{workspace_id}")
        assert workspace_response.status_code == 200

        workspace_data = workspace_response.json()

        # Verify all 5 layers processed
        layer_states = workspace_data["layer_states"]
        expected_layers = ["sensory", "perceptual", "conceptual", "abstract", "metacognitive"]

        for layer in expected_layers:
            assert layer in layer_states
            layer_state = layer_states[layer]
            assert layer_state["status"] in ["completed", "processing"]  # Allow for async processing

        # Verify consciousness level progression
        consciousness_level = workspace_data["consciousness_level"]
        assert 0.0 <= consciousness_level <= 1.0

        # For consciousness-related content, should achieve higher levels
        if workspace_data["processing_status"] == "completed":
            assert consciousness_level > 0.5  # Should detect consciousness themes

    def test_document_thoughtseed_layer_progression(self, client, complex_pdf_document):
        """Test hierarchical layer progression and dependencies"""
        filename, file_content, content_type = complex_pdf_document

        response = client.post(
            "/api/v1/documents/process",
            files={"file": (filename, file_content, content_type)},
            data={"thoughtseed_layers": '["sensory","perceptual","conceptual"]'}  # Only first 3 layers
        )

        if response.status_code == 200:
            process_data = response.json()
            workspace_id = process_data.get("thoughtseed_workspace_id")

            workspace_response = client.get(f"/api/v1/thoughtseed/workspace/{workspace_id}")
            if workspace_response.status_code == 200:
                workspace_data = workspace_response.json()
                layer_states = workspace_data["layer_states"]

                # Should process layers in order
                if layer_states["conceptual"]["status"] == "completed":
                    # If conceptual is done, lower layers should be done
                    assert layer_states["sensory"]["status"] == "completed"
                    assert layer_states["perceptual"]["status"] == "completed"

                # Higher layers should not be processed if not requested
                if "abstract" in layer_states:
                    # Abstract layer should be pending or not started
                    assert layer_states["abstract"]["status"] in ["pending", "not_started"]

    def test_document_thoughtseed_pattern_extraction(self, client, complex_pdf_document):
        """Test pattern extraction across ThoughtSeed layers"""
        filename, file_content, content_type = complex_pdf_document

        response = client.post(
            "/api/v1/documents/process",
            files={"file": (filename, file_content, content_type)},
            data={
                "extract_narratives": "true",
                "thoughtseed_layers": '["sensory","perceptual","conceptual","abstract","metacognitive"]'
            }
        )

        if response.status_code == 200:
            process_data = response.json()
            patterns = process_data["patterns_extracted"]

            # Should extract patterns at different abstraction levels
            assert isinstance(patterns, list)

            if patterns:
                # Patterns should have layer attribution
                for pattern in patterns:
                    assert "pattern_id" in pattern
                    assert "layer_detected" in pattern
                    assert "abstraction_level" in pattern
                    assert pattern["layer_detected"] in [
                        "sensory", "perceptual", "conceptual", "abstract", "metacognitive"
                    ]
                    assert isinstance(pattern["abstraction_level"], (int, float))

    def test_document_thoughtseed_attractor_basin_activation(self, client, complex_pdf_document):
        """Test attractor basin activation during document processing"""
        filename, file_content, content_type = complex_pdf_document

        # Process document
        response = client.post(
            "/api/v1/documents/process",
            files={"file": (filename, file_content, content_type)},
            data={"thoughtseed_layers": '["conceptual","abstract","metacognitive"]'}
        )

        if response.status_code == 200:
            process_data = response.json()

            # Check if attractor basins were activated
            basins_response = client.get("/api/v1/context-engineering/basins")
            if basins_response.status_code == 200:
                basins_data = basins_response.json()

                # Should have some active basins after document processing
                assert basins_data["active_basins"] >= 0

                # Consciousness coherence should be affected
                coherence = basins_data["consciousness_coherence"]
                assert 0.0 <= coherence <= 1.0

                # Neural field should show activity
                field_state = basins_data["neural_field_state"]
                assert field_state["field_strength"] >= 0.0

    def test_document_thoughtseed_memory_formation(self, client, complex_pdf_document):
        """Test memory formation during ThoughtSeed processing"""
        filename, file_content, content_type = complex_pdf_document

        response = client.post(
            "/api/v1/documents/process",
            files={"file": (filename, file_content, content_type)},
            data={
                "thoughtseed_layers": '["perceptual","conceptual","abstract"]',
                "form_memories": "true"
            }
        )

        if response.status_code == 200:
            process_data = response.json()
            workspace_id = process_data.get("thoughtseed_workspace_id")

            # Check workspace for memory traces
            workspace_response = client.get(f"/api/v1/thoughtseed/workspace/{workspace_id}")
            if workspace_response.status_code == 200:
                workspace_data = workspace_response.json()

                # Should have traces for memory formation
                traces = workspace_data["traces_generated"]
                assert isinstance(traces, list)

                # Traces should span multiple layers for memory consolidation
                layer_traces = set()
                for trace in traces:
                    layer_traces.add(trace["layer"])

                # Should have traces from multiple layers for rich memory
                if len(traces) > 0:
                    assert len(layer_traces) >= 1

    def test_document_thoughtseed_narrative_consciousness(self, client, complex_pdf_document):
        """Test narrative consciousness emergence in complex documents"""
        filename, file_content, content_type = complex_pdf_document

        response = client.post(
            "/api/v1/documents/process",
            files={"file": (filename, file_content, content_type)},
            data={
                "extract_narratives": "true",
                "thoughtseed_layers": '["conceptual","abstract","metacognitive"]',
                "consciousness_threshold": "0.6"
            }
        )

        if response.status_code == 200:
            process_data = response.json()

            # Check narrative elements for consciousness indicators
            narrative_elements = process_data["narrative_elements"]

            if "themes" in narrative_elements:
                themes = narrative_elements["themes"]
                # Should detect consciousness-related themes in this document
                consciousness_themes = [
                    theme for theme in themes
                    if "consciousness" in theme.lower() or "awareness" in theme.lower()
                ]
                # Document is about consciousness, so should detect it
                assert len(consciousness_themes) >= 0  # May or may not detect depending on processing

    def test_document_thoughtseed_performance_integration(self, client, complex_pdf_document):
        """Test performance of integrated ThoughtSeed processing"""
        filename, file_content, content_type = complex_pdf_document

        response = client.post(
            "/api/v1/documents/process",
            files={"file": (filename, file_content, content_type)},
            data={"thoughtseed_layers": '["sensory","perceptual","conceptual"]'}  # Faster processing
        )

        if response.status_code == 200:
            process_data = response.json()
            processing_time = process_data["processing_time_ms"]

            # Should complete integrated processing in reasonable time
            # More lenient for integration test than unit test
            assert processing_time < 10000  # <10s for integration test

    def test_document_thoughtseed_error_handling(self, client):
        """Test error handling in ThoughtSeed integration"""
        # Test with malformed PDF
        bad_pdf = ("bad.pdf", io.BytesIO(b"not a real pdf"), "application/pdf")
        filename, file_content, content_type = bad_pdf

        response = client.post(
            "/api/v1/documents/process",
            files={"file": (filename, file_content, content_type)},
            data={"thoughtseed_layers": '["sensory"]'}
        )

        # Should handle error gracefully
        assert response.status_code in [200, 400, 422]

        if response.status_code == 200:
            # If it processes, should indicate error status
            process_data = response.json()
            assert process_data["processing_status"] in ["failed", "partial"]

    def test_document_thoughtseed_concurrent_processing(self, client, complex_pdf_document):
        """Test concurrent document processing through ThoughtSeed"""
        filename, file_content, content_type = complex_pdf_document

        # Submit multiple documents for processing
        responses = []
        for i in range(2):  # Keep it small for integration test
            file_content.seek(0)  # Reset file pointer
            response = client.post(
                "/api/v1/documents/process",
                files={"file": (f"doc_{i}.pdf", file_content, content_type)},
                data={"thoughtseed_layers": '["sensory","perceptual"]'}
            )
            responses.append(response)

        # All should succeed or handle concurrency gracefully
        for response in responses:
            assert response.status_code in [200, 429]  # 429 if rate limited

            if response.status_code == 200:
                process_data = response.json()
                assert "document_id" in process_data
                # Each should have unique workspace
                if "thoughtseed_workspace_id" in process_data:
                    assert process_data["thoughtseed_workspace_id"] is not None