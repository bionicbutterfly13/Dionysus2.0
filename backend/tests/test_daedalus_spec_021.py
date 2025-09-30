#!/usr/bin/env python3
"""
Test Driven Development for Spec 021: Clean Daedalus Class
Per FR-001: Tests MUST be written before any Daedalus class refactoring begins

This test defines the EXACT requirements for the simplified Daedalus class.
All tests MUST fail initially (Red), then implementation makes them pass (Green).
"""

import pytest
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch


class TestDaedalusSpecification:
    """TDD for Daedalus clean up per spec 021-remove-all-that"""

    def test_daedalus_class_exists(self):
        """Test that Daedalus class can be imported and instantiated"""
        # This MUST fail initially - forces implementation
        from src.services.daedalus import Daedalus
        
        daedalus = Daedalus()
        assert daedalus is not None

    def test_daedalus_has_single_responsibility(self):
        """FR-002 & FR-006: Daedalus MUST have only core perceptual information reception"""
        from src.services.daedalus import Daedalus
        
        daedalus = Daedalus()
        
        # Should have exactly ONE public method for receiving information
        public_methods = [method for method in dir(daedalus) 
                         if not method.startswith('_') and callable(getattr(daedalus, method))]
        
        assert len(public_methods) == 1, f"Expected 1 method, found {len(public_methods)}: {public_methods}"
        assert 'receive_perceptual_information' in public_methods

    def test_daedalus_receives_uploaded_data(self):
        """FR-002 & FR-007: Daedalus MUST receive perceptual information from uploaded data"""
        from src.services.daedalus import Daedalus
        
        daedalus = Daedalus()
        
        # Test with file-like object
        test_file = io.BytesIO(b"Test document content")
        test_file.name = "test_document.pdf"
        
        result = daedalus.receive_perceptual_information(test_file)
        
        # Should return acknowledgment of reception
        assert result is not None
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'received_data' in result

    def test_daedalus_handles_all_file_types(self):
        """FR-007: System MUST preserve upload-triggered information flow for all file types"""
        from src.services.daedalus import Daedalus
        
        daedalus = Daedalus()
        
        # Test different file types
        file_types = [
            ('document.pdf', b'%PDF-1.4 fake pdf content'),
            ('text.txt', b'Plain text content'),
            ('data.json', b'{"key": "value"}'),
            ('image.png', b'\x89PNG\r\n\x1a\n fake png'),
            ('unknown.xyz', b'Unknown file type content')
        ]
        
        for filename, content in file_types:
            test_file = io.BytesIO(content)
            test_file.name = filename
            
            result = daedalus.receive_perceptual_information(test_file)
            
            assert result['status'] == 'received', f"Failed to handle {filename}"
            assert 'received_data' in result

    def test_daedalus_serves_as_gateway(self):
        """FR-003: Daedalus MUST serve as the primary gateway for external information"""
        from src.services.daedalus import Daedalus
        
        daedalus = Daedalus()
        
        # Should be marked as gateway
        assert hasattr(daedalus, '_is_gateway')
        assert daedalus._is_gateway is True

    def test_daedalus_interfaces_with_langgraph(self):
        """FR-004: Daedalus MUST interface with LangGraph architecture for agent creation"""
        from src.services.daedalus import Daedalus
        
        daedalus = Daedalus()
        
        test_file = io.BytesIO(b"Complex document for agent processing")
        test_file.name = "complex_doc.pdf"
        
        with patch('src.services.daedalus.create_langgraph_agents') as mock_agents:
            mock_agents.return_value = ['agent_1', 'agent_2']
            
            result = daedalus.receive_perceptual_information(test_file)
            
            # Should trigger LangGraph agent creation
            mock_agents.assert_called_once()
            assert 'agents_created' in result
            assert result['agents_created'] == ['agent_1', 'agent_2']

    def test_daedalus_no_extra_functionality(self):
        """FR-005: System MUST remove all non-essential functionality from Daedalus class"""
        from src.services.daedalus import Daedalus
        
        daedalus = Daedalus()
        
        # Should NOT have these methods (common in bloated classes)
        forbidden_methods = [
            'process_document', 'analyze_content', 'extract_features',
            'save_to_database', 'send_notification', 'log_activity',
            'validate_input', 'transform_data', 'generate_report',
            'update_metrics', 'check_health', 'configure_settings'
        ]
        
        for method in forbidden_methods:
            assert not hasattr(daedalus, method), f"Found forbidden method: {method}"

    def test_daedalus_archived_functionality(self):
        """FR-008: Removed functionality MUST be archived to backup/deprecated folder"""
        backup_folder = Path("../backup/deprecated/daedalus_removed_features")
        
        # Should have created backup folder with old functionality
        assert backup_folder.exists(), "backup/deprecated/daedalus_removed_features folder must exist"
        
        # Should contain some archived files
        archived_files = list(backup_folder.glob("*.py"))
        assert len(archived_files) > 0, "Should have archived some removed functionality"

    def test_daedalus_maintains_information_flow(self):
        """Test that upload → Daedalus flow is maintained"""
        from src.services.daedalus import Daedalus
        
        daedalus = Daedalus()
        
        # Simulate upload trigger
        upload_data = {
            'file': io.BytesIO(b"Uploaded document content"),
            'filename': 'uploaded_doc.pdf',
            'upload_time': '2025-09-30T12:00:00Z'
        }
        upload_data['file'].name = upload_data['filename']
        
        result = daedalus.receive_perceptual_information(upload_data['file'])
        
        # Should maintain the flow
        assert result['status'] == 'received'
        assert 'timestamp' in result
        assert 'source' in result
        assert result['source'] == 'upload_trigger'

    def test_daedalus_error_handling(self):
        """Test error handling for edge cases"""
        from src.services.daedalus import Daedalus
        
        daedalus = Daedalus()
        
        # Test with None
        result = daedalus.receive_perceptual_information(None)
        assert result['status'] == 'error'
        assert 'error_message' in result
        
        # Test with empty file
        empty_file = io.BytesIO(b"")
        empty_file.name = "empty.txt"
        result = daedalus.receive_perceptual_information(empty_file)
        assert result['status'] == 'received'  # Should still handle empty files

    def test_daedalus_performance_requirement(self):
        """Test that perceptual information reception is fast"""
        from src.services.daedalus import Daedalus
        import time
        
        daedalus = Daedalus()
        test_file = io.BytesIO(b"Performance test content" * 1000)
        test_file.name = "large_test.txt"
        
        start_time = time.time()
        result = daedalus.receive_perceptual_information(test_file)
        end_time = time.time()
        
        # Should be very fast (< 100ms) since it's just reception, not processing
        assert (end_time - start_time) < 0.1, "Reception should be under 100ms"
        assert result['status'] == 'received'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])