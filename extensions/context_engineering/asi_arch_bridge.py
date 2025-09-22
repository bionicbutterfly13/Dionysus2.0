#!/usr/bin/env python3
"""
🌉 ASI-Arch Context Engineering Bridge
=====================================

Integration bridge between ASI-Arch's existing pipeline and our self-contained
context engineering system. This module provides clean integration points
without modifying ASI-Arch's core code.

Integration Points:
1. Enhanced context generation for evolution
2. Architecture space analysis for insights
3. Consciousness-guided evolution suggestions
4. River metaphor visualization and monitoring

Usage:
    # In ASI-Arch pipeline
    from extensions.context_engineering.asi_arch_bridge import enhance_evolution_context
    
    enhanced_context = await enhance_evolution_context(original_context, parent_data)

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - ASI-Arch Integration Bridge
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import our self-contained context engineering
from .core_implementation import (
    ContextEngineeringService,
    ConsciousnessLevel,
    FlowState,
    create_context_engineering_service
)

logger = logging.getLogger(__name__)

# =============================================================================
# ASI-Arch Data Conversion
# =============================================================================

class ASIArchDataConverter:
    """Converts between ASI-Arch DataElement format and our context engineering models"""
    
    @staticmethod
    def dataelem_to_dict(data_element) -> Dict[str, Any]:
        """Convert ASI-Arch DataElement to dictionary format"""
        if hasattr(data_element, 'to_dict'):
            return data_element.to_dict()
        elif hasattr(data_element, '__dict__'):
            return data_element.__dict__
        else:
            # Assume it's already a dictionary
            return data_element
    
    @staticmethod
    def extract_architecture_data(data_element) -> Dict[str, Any]:
        """Extract relevant data from ASI-Arch DataElement"""
        data_dict = ASIArchDataConverter.dataelem_to_dict(data_element)
        
        return {
            'name': data_dict.get('name', 'unknown'),
            'program': data_dict.get('program', ''),
            'result': data_dict.get('result', {}),
            'motivation': data_dict.get('motivation', ''),
            'analysis': data_dict.get('analysis', ''),
            'cognition': data_dict.get('cognition', ''),
            'parent': data_dict.get('parent'),
            'index': data_dict.get('index'),
            'time': data_dict.get('time', datetime.now().isoformat())
        }

# =============================================================================
# Main Integration Functions
# =============================================================================

# Global service instance (lazy initialization)
_context_service: Optional[ContextEngineeringService] = None

def get_context_service() -> ContextEngineeringService:
    """Get or create the context engineering service"""
    global _context_service
    if _context_service is None:
        _context_service = create_context_engineering_service()
    return _context_service

async def enhance_evolution_context(original_context: str, 
                                  parent_data_element) -> str:
    """
    Main integration function: Enhance ASI-Arch evolution context with river metaphor insights
    
    Args:
        original_context: The original context string from ASI-Arch
        parent_data_element: ASI-Arch DataElement of the parent architecture
    
    Returns:
        Enhanced context string with context engineering insights
    """
    try:
        service = get_context_service()
        
        # Convert ASI-Arch data to our format
        parent_data = ASIArchDataConverter.extract_architecture_data(parent_data_element)
        
        # Enhance the context
        enhanced_context = await service.enhance_asi_arch_context(original_context, parent_data)
        
        logger.info(f"Enhanced evolution context for architecture: {parent_data.get('name', 'unknown')}")
        return enhanced_context
        
    except Exception as e:
        logger.error(f"Error enhancing evolution context: {e}")
        # Fallback: return original context if enhancement fails
        return original_context

async def analyze_architecture_portfolio(data_elements: List) -> Dict[str, Any]:
    """
    Analyze a collection of architectures using context engineering
    
    Args:
        data_elements: List of ASI-Arch DataElements
    
    Returns:
        Comprehensive analysis of the architecture space
    """
    try:
        service = get_context_service()
        
        # Convert all data elements
        arch_data_list = [
            ASIArchDataConverter.extract_architecture_data(elem) 
            for elem in data_elements
        ]
        
        # Perform analysis
        analysis = await service.analyze_architecture_space(arch_data_list)
        
        logger.info(f"Analyzed portfolio of {len(data_elements)} architectures")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing architecture portfolio: {e}")
        return {
            'error': str(e),
            'analyzed_count': 0,
            'recommendations': {'status': 'analysis_failed'}
        }

async def get_consciousness_level(data_element) -> Tuple[str, float]:
    """
    Get consciousness level for a single architecture
    
    Args:
        data_element: ASI-Arch DataElement
    
    Returns:
        Tuple of (consciousness_level_name, consciousness_score)
    """
    try:
        service = get_context_service()
        arch_data = ASIArchDataConverter.extract_architecture_data(data_element)
        
        consciousness_level = await service.evolution.consciousness_detector.detect_consciousness_level(arch_data)
        
        return consciousness_level.name, consciousness_level.value
        
    except Exception as e:
        logger.error(f"Error detecting consciousness level: {e}")
        return "UNKNOWN", 0.0

async def suggest_evolution_improvements(parent_data_element) -> Dict[str, Any]:
    """
    Get evolution suggestions based on context engineering analysis
    
    Args:
        parent_data_element: ASI-Arch DataElement of parent architecture
    
    Returns:
        Dictionary of evolution suggestions and insights
    """
    try:
        service = get_context_service()
        parent_data = ASIArchDataConverter.extract_architecture_data(parent_data_element)
        
        suggestions = await service.evolution.suggest_evolution_direction(parent_data)
        
        return {
            'suggestions': suggestions,
            'parent_name': parent_data.get('name', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating evolution suggestions: {e}")
        return {
            'error': str(e),
            'suggestions': {'status': 'suggestion_failed'}
        }

# =============================================================================
# Integration with ASI-Arch Pipeline Steps
# =============================================================================

class ContextEngineeringIntegration:
    """Integration class for embedding context engineering into ASI-Arch pipeline"""
    
    def __init__(self):
        self.service = get_context_service()
        self.integration_enabled = True
    
    async def pre_evolution_analysis(self, context: str, parent_data_element) -> str:
        """Called before evolution step to enhance context"""
        if not self.integration_enabled:
            return context
        
        return await enhance_evolution_context(context, parent_data_element)
    
    async def post_evaluation_analysis(self, data_element, evaluation_success: bool) -> Dict[str, Any]:
        """Called after evaluation to analyze results with context engineering"""
        if not self.integration_enabled:
            return {}
        
        consciousness_level, consciousness_score = await get_consciousness_level(data_element)
        
        analysis = {
            'consciousness_analysis': {
                'level': consciousness_level,
                'score': consciousness_score
            },
            'evaluation_success': evaluation_success,
            'architecture_name': getattr(data_element, 'name', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    async def periodic_portfolio_analysis(self, all_data_elements: List) -> Dict[str, Any]:
        """Periodic analysis of entire architecture portfolio"""
        if not self.integration_enabled:
            return {}
        
        return await analyze_architecture_portfolio(all_data_elements)
    
    def enable_integration(self):
        """Enable context engineering integration"""
        self.integration_enabled = True
        logger.info("Context engineering integration enabled")
    
    def disable_integration(self):
        """Disable context engineering integration (fallback to original ASI-Arch)"""
        self.integration_enabled = False
        logger.info("Context engineering integration disabled")

# =============================================================================
# Monitoring and Visualization
# =============================================================================

class ContextEngineeringMonitor:
    """Monitor context engineering metrics and provide insights"""
    
    def __init__(self):
        self.service = get_context_service()
        self.metrics_history = []
    
    async def collect_metrics(self, data_elements: List) -> Dict[str, Any]:
        """Collect current context engineering metrics"""
        if not data_elements:
            return {}
        
        # Analyze current state
        analysis = await analyze_architecture_portfolio(data_elements)
        
        # Add timestamp
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'architecture_count': len(data_elements),
            **analysis
        }
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Keep only last 100 entries
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of context engineering metrics over time"""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        # Calculate trends
        consciousness_scores = []
        basin_counts = []
        
        for metrics in recent_metrics:
            consciousness_analysis = metrics.get('consciousness_analysis', {})
            basin_analysis = metrics.get('attractor_basins', {})
            
            consciousness_scores.append(consciousness_analysis.get('avg_consciousness', 0.0))
            basin_counts.append(basin_analysis.get('count', 0))
        
        summary = {
            'total_measurements': len(self.metrics_history),
            'recent_trend': {
                'consciousness_trend': 'increasing' if len(consciousness_scores) > 1 and consciousness_scores[-1] > consciousness_scores[0] else 'stable',
                'basin_evolution': 'expanding' if len(basin_counts) > 1 and basin_counts[-1] > basin_counts[0] else 'stable',
                'avg_consciousness': sum(consciousness_scores) / len(consciousness_scores) if consciousness_scores else 0.0,
                'avg_basins': sum(basin_counts) / len(basin_counts) if basin_counts else 0.0
            },
            'latest_metrics': self.metrics_history[-1] if self.metrics_history else {}
        }
        
        return summary
    
    def export_metrics(self, filepath: str = "context_engineering_metrics.json"):
        """Export metrics history to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'export_timestamp': datetime.now().isoformat(),
                    'metrics_count': len(self.metrics_history),
                    'metrics_history': self.metrics_history
                }, f, indent=2)
            
            logger.info(f"Exported {len(self.metrics_history)} metric entries to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False

# =============================================================================
# Global Integration Instance
# =============================================================================

# Global integration instance for easy access
_integration: Optional[ContextEngineeringIntegration] = None
_monitor: Optional[ContextEngineeringMonitor] = None

def get_integration() -> ContextEngineeringIntegration:
    """Get global context engineering integration instance"""
    global _integration
    if _integration is None:
        _integration = ContextEngineeringIntegration()
    return _integration

def get_monitor() -> ContextEngineeringMonitor:
    """Get global context engineering monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = ContextEngineeringMonitor()
    return _monitor

# =============================================================================
# Usage Examples and Testing
# =============================================================================

async def test_integration():
    """Test the ASI-Arch integration"""
    print("🧪 Testing ASI-Arch Context Engineering Integration")
    
    # Mock ASI-Arch DataElement
    class MockDataElement:
        def __init__(self, name, program, result, motivation, analysis):
            self.name = name
            self.program = program
            self.result = result
            self.motivation = motivation
            self.analysis = analysis
            self.cognition = ""
            self.parent = None
            self.index = 1
            self.time = datetime.now().isoformat()
    
    # Create test data
    parent_arch = MockDataElement(
        name="linear_attention_base",
        program="class LinearAttention(nn.Module): def forward(self, x): return x @ x.T",
        result={"train": "loss=0.5", "test": "acc=0.80"},
        motivation="baseline linear attention implementation",
        analysis="shows reasonable performance but lacks sophistication"
    )
    
    # Test context enhancement
    print("\n1. Testing context enhancement...")
    original_context = "Create an improved linear attention mechanism."
    enhanced_context = await enhance_evolution_context(original_context, parent_arch)
    print(f"Original length: {len(original_context)}")
    print(f"Enhanced length: {len(enhanced_context)}")
    print(f"Enhancement added: {len(enhanced_context) - len(original_context)} characters")
    
    # Test consciousness detection
    print("\n2. Testing consciousness detection...")
    consciousness_level, consciousness_score = await get_consciousness_level(parent_arch)
    print(f"Consciousness Level: {consciousness_level}")
    print(f"Consciousness Score: {consciousness_score:.3f}")
    
    # Test evolution suggestions
    print("\n3. Testing evolution suggestions...")
    suggestions = await suggest_evolution_improvements(parent_arch)
    print(f"Suggestions received: {len(suggestions.get('suggestions', {}))}")
    
    # Test portfolio analysis
    print("\n4. Testing portfolio analysis...")
    mock_portfolio = [parent_arch]  # In real use, this would be many architectures
    analysis = await analyze_architecture_portfolio(mock_portfolio)
    print(f"Analysis keys: {list(analysis.keys())}")
    
    print("\n✅ Integration test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_integration())
