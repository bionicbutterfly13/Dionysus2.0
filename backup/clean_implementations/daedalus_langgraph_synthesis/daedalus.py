"""
Daedalus: Perceptual Information Gateway
Simplified per Spec 021-remove-all-that

Single Responsibility: Receive perceptual information from external sources.
This class has ONE function: receive information from the outside.

Integration: LangGraph-based workflow for consciousness-enhanced processing:
- DocumentProcessingGraph: Complete ASI-GO-2 + R-Zero + Active Inference workflow
  * Extract & Process (SurfSense patterns)
  * Research Planning (ASI-GO-2 Researcher + R-Zero questions)
  * Consciousness Processing (Attractor basins + ThoughtSeeds)
  * Analysis (ASI-GO-2 Analyst + Meta-cognitive tracking)
  * Iterative Refinement (Quality-driven improvement)
- Knowledge Graph: Store in Neo4j memory
"""

import time
from typing import Any, BinaryIO, Dict, List, Optional
from .document_processing_graph import DocumentProcessingGraph


class Daedalus:
    """
    Perceptual Information Gateway

    Clean implementation per spec 021:
    - Single responsibility: receive perceptual information
    - LangGraph workflow architecture
    - Handle all file types from uploads
    - No additional functionality

    Uses DocumentProcessingGraph (SurfSense + ASI-GO-2 + R-Zero + Active Inference)
    """

    def __init__(self):
        """Initialize the perceptual information gateway"""
        self._is_gateway = True
        self.processing_graph = DocumentProcessingGraph()
    
    def receive_perceptual_information(self,
                                      data: Optional[BinaryIO],
                                      tags: List[str] = None,
                                      max_iterations: int = 3,
                                      quality_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Receive perceptual information from uploaded data.

        This is the ONLY public method - single responsibility principle.

        Uses DocumentProcessingGraph: LangGraph workflow with complete integration:
        - SurfSense extraction patterns
        - ASI-GO-2 iterative refinement
        - R-Zero curiosity-driven exploration
        - Active Inference prediction error minimization
        - Dionysus consciousness processing

        Args:
            data: File-like object containing perceptual information
            tags: Optional categorization tags
            max_iterations: Maximum refinement iterations (default: 3)
            quality_threshold: Quality score threshold for completion (default: 0.7)

        Returns:
            Dict containing complete processing results from LangGraph workflow
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

            # Process through LangGraph workflow
            result = self.processing_graph.process_document(
                content=content,
                filename=filename,
                tags=tags or [],
                max_iterations=max_iterations,
                quality_threshold=quality_threshold
            )

            # Package response
            return {
                'status': 'received',
                'document': result['document'],
                'extraction': result['extraction'],
                'consciousness': result['consciousness'],
                'research': result['research'],
                'quality': result['quality'],
                'meta_cognitive': result['meta_cognitive'],
                'workflow': {
                    'iterations': result['iterations'],
                    'messages': result['messages']
                },
                'timestamp': time.time(),
                'source': 'langgraph_workflow'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'timestamp': time.time()
            }

    def get_cognition_summary(self) -> Dict[str, Any]:
        """Get summary of learned patterns and strategies across all processing sessions"""
        return self.processing_graph.get_cognition_summary()