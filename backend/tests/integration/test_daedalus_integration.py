"""
Integration test for Daedalus Gateway
Ensures clean Daedalus implementation integrates properly with Dionysus-2.0
Per Spec 021: Test simplified Daedalus class with single responsibility
"""

import pytest
import io
from src.services.daedalus import Daedalus


class TestDaedalusIntegration:
    """Test that Dionysus-2.0 can use extracted Daedalus Gateway"""
    
    def test_can_import_daedalus(self):
        """Test that we can import Daedalus from external package"""
        # Should be able to import without error
        assert Daedalus is not None
        
    def test_can_instantiate_daedalus(self):
        """Test that we can create Daedalus instance"""
        gateway = Daedalus()
        assert gateway is not None
        assert hasattr(gateway, '_is_gateway')
        assert gateway._is_gateway is True
        
    def test_can_use_daedalus_functionality(self):
        """Test that extracted Daedalus works as expected"""
        gateway = Daedalus()
        
        # Test with sample data
        test_data = io.BytesIO(b"Integration test document")
        test_data.name = "integration_test.txt"
        
        result = gateway.receive_perceptual_information(test_data)
        
        # Should receive data successfully
        assert result['status'] == 'received'
        assert 'received_data' in result
        assert 'agents_created' in result
        assert result['source'] == 'upload_trigger'
        
    def test_modular_benefits(self):
        """Test that modularization provides expected benefits"""
        gateway = Daedalus()
        
        # Should have exactly one public method (single responsibility)
        public_methods = [method for method in dir(gateway) 
                         if not method.startswith('_') and callable(getattr(gateway, method))]
        assert len(public_methods) == 1
        assert 'receive_perceptual_information' in public_methods
        
        # Should not have forbidden methods (clean implementation)
        forbidden_methods = [
            'process_document', 'analyze_content', 'extract_features',
            'save_to_database', 'send_notification', 'log_activity'
        ]
        for method in forbidden_methods:
            assert not hasattr(gateway, method), f"Found forbidden method: {method}"