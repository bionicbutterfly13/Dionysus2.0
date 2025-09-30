"""
Daedalus: Perceptual Information Gateway
Simplified per Spec 021-remove-all-that

Single Responsibility: Receive perceptual information from external sources.
This class has ONE function: receive information from the outside.
"""

import time
from typing import Any, BinaryIO, Dict, List, Optional
from pathlib import Path


def create_langgraph_agents(data: Any) -> List[str]:
    """Create LangGraph agents for processing received data"""
    # Mock implementation - would interface with actual LangGraph
    return [f"agent_{int(time.time())}_1", f"agent_{int(time.time())}_2"]


class Daedalus:
    """
    Perceptual Information Gateway
    
    Clean implementation per spec 021:
    - Single responsibility: receive perceptual information
    - Interface with LangGraph architecture
    - Handle all file types from uploads
    - No additional functionality
    """
    
    def __init__(self):
        """Initialize the perceptual information gateway"""
        self._is_gateway = True
    
    def receive_perceptual_information(self, data: Optional[BinaryIO]) -> Dict[str, Any]:
        """
        Receive perceptual information from uploaded data.
        
        This is the ONLY public method - single responsibility principle.
        
        Args:
            data: File-like object containing perceptual information
            
        Returns:
            Dict containing reception status and metadata
        """
        if data is None:
            return {
                'status': 'error',
                'error_message': 'No data provided',
                'timestamp': time.time()
            }
        
        try:
            # Read the perceptual information
            content = data.read()
            filename = getattr(data, 'name', 'unknown_file')
            
            # Reset file pointer for potential future use
            if hasattr(data, 'seek'):
                data.seek(0)
            
            # Create LangGraph agents for processing
            agents = create_langgraph_agents(content)
            
            return {
                'status': 'received',
                'received_data': {
                    'filename': filename,
                    'size_bytes': len(content),
                    'content_preview': content[:100] if len(content) > 100 else content
                },
                'agents_created': agents,
                'timestamp': time.time(),
                'source': 'upload_trigger'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'timestamp': time.time()
            }