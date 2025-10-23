#!/usr/bin/env python3
"""
Debug-Enhanced Document Processor
Wraps document_processing_graph.py with granular event emission for debugging
"""
from typing import Dict, Any, AsyncGenerator
import asyncio
import json
from datetime import datetime
import logging

from .document_processing_graph import DocumentProcessingGraph
from ..api.routes.debug_stream import processing_queue, active_processing, broadcast_queue_status

logger = logging.getLogger(__name__)


class DebugDocumentProcessor:
    """
    Wrapper around DocumentProcessingGraph that emits detailed debug events
    at every step of the pipeline
    """

    def __init__(self):
        self.graph = DocumentProcessingGraph()
        self.current_document = None

    async def process_with_debug_events(
        self,
        content: bytes,
        filename: str,
        document_id: str,
        tags: list = None,
        max_iterations: int = 3,
        quality_threshold: float = 0.7
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process document through LangGraph with detailed event emission

        Yields debug events for:
        - Queue position
        - Node transitions
        - Concept activation
        - Basin changes
        - Processing milestones
        """

        self.current_document = document_id

        # Update queue status
        for doc in processing_queue:
            if doc['id'] == document_id:
                doc['status'] = 'processing'
                break

        await broadcast_queue_status()

        # Update active processing
        active_processing[document_id] = {
            'filename': filename,
            'started_at': datetime.now().isoformat(),
            'current_node': None,
            'progress': 0
        }

        try:
            # ============================================================
            # PROCESSING START
            # ============================================================
            yield {
                'type': 'processing_start',
                'document_id': document_id,
                'filename': filename,
                'timestamp': datetime.now().isoformat()
            }

            # ============================================================
            # NODE 1: EXTRACT & PROCESS
            # ============================================================
            active_processing[document_id]['current_node'] = 'extract_and_process'
            active_processing[document_id]['progress'] = 0.1

            yield {
                'type': 'node_start',
                'node': 'extract_and_process',
                'document_id': document_id,
                'timestamp': datetime.now().isoformat()
            }

            # Simulate extraction process with status updates
            yield {
                'type': 'extraction_progress',
                'document_id': document_id,
                'stage': 'content_hash',
                'message': 'Calculating SHA-256 hash...',
                'timestamp': datetime.now().isoformat()
            }
            await asyncio.sleep(0.2)

            yield {
                'type': 'extraction_progress',
                'document_id': document_id,
                'stage': 'text_extraction',
                'message': 'Extracting text from PDF...',
                'timestamp': datetime.now().isoformat()
            }
            await asyncio.sleep(0.3)

            yield {
                'type': 'extraction_progress',
                'document_id': document_id,
                'stage': 'chunking',
                'message': 'Splitting into semantic chunks...',
                'timestamp': datetime.now().isoformat()
            }
            await asyncio.sleep(0.2)

            # Actually call the LangGraph node
            state = {
                'content': content,
                'filename': filename,
                'tags': tags or [],
                'max_iterations': max_iterations,
                'quality_threshold': quality_threshold
            }

            state = self.graph._extract_and_process_node(state)

            # Emit concepts discovered
            result = state['processing_result']
            for idx, concept in enumerate(result.concepts[:10]):  # First 10 concepts
                yield {
                    'type': 'concept_discovered',
                    'document_id': document_id,
                    'node': 'extract_and_process',
                    'concept': concept,
                    'index': idx,
                    'total': len(result.concepts),
                    'timestamp': datetime.now().isoformat()
                }
                await asyncio.sleep(0.05)  # Stagger for visual effect

            yield {
                'type': 'node_complete',
                'node': 'extract_and_process',
                'document_id': document_id,
                'concepts_extracted': len(result.concepts),
                'chunks_created': len(result.chunks),
                'timestamp': datetime.now().isoformat()
            }

            active_processing[document_id]['progress'] = 0.2

            # ============================================================
            # NODE 2: RESEARCH PLAN
            # ============================================================
            active_processing[document_id]['current_node'] = 'generate_research_plan'

            yield {
                'type': 'node_start',
                'node': 'generate_research_plan',
                'document_id': document_id,
                'timestamp': datetime.now().isoformat()
            }

            state = self.graph._generate_research_plan_node(state)
            research_plan = state['research_plan']

            # Emit research questions
            for question in research_plan.get('challenging_questions', [])[:5]:
                yield {
                    'type': 'research_question_generated',
                    'document_id': document_id,
                    'node': 'generate_research_plan',
                    'question': question,
                    'timestamp': datetime.now().isoformat()
                }
                await asyncio.sleep(0.1)

            yield {
                'type': 'node_complete',
                'node': 'generate_research_plan',
                'document_id': document_id,
                'questions_generated': len(research_plan.get('challenging_questions', [])),
                'timestamp': datetime.now().isoformat()
            }

            active_processing[document_id]['progress'] = 0.4

            # ============================================================
            # NODE 3: CONSCIOUSNESS PROCESSING
            # ============================================================
            active_processing[document_id]['current_node'] = 'consciousness_processing'

            yield {
                'type': 'node_start',
                'node': 'consciousness_processing',
                'document_id': document_id,
                'timestamp': datetime.now().isoformat()
            }

            # Basin activation
            basin = result.__dict__.get('attractor_basin', 'exploration')
            yield {
                'type': 'basin_activated',
                'document_id': document_id,
                'node': 'consciousness_processing',
                'basin': basin,
                'depth': 0.7,  # Processing depth
                'timestamp': datetime.now().isoformat()
            }

            state = self.graph._consciousness_processing_node(state)

            # ThoughtSeed generation events
            thoughtseeds = result.__dict__.get('thoughtseeds_generated', [])
            for idx, seed in enumerate(thoughtseeds[:8]):  # First 8 thoughtseeds
                yield {
                    'type': 'thoughtseed_generated',
                    'document_id': document_id,
                    'node': 'consciousness_processing',
                    'thoughtseed_id': seed.get('id', f'seed_{idx}'),
                    'concepts': seed.get('concept_labels', []),
                    'resonance': seed.get('resonance_score', 0.5),
                    'basin': basin,
                    'timestamp': datetime.now().isoformat()
                }
                await asyncio.sleep(0.15)

            yield {
                'type': 'node_complete',
                'node': 'consciousness_processing',
                'document_id': document_id,
                'basins_activated': result.__dict__.get('basins_created', 1),
                'thoughtseeds_generated': len(thoughtseeds),
                'timestamp': datetime.now().isoformat()
            }

            active_processing[document_id]['progress'] = 0.6

            # ============================================================
            # NODE 4: ANALYZE RESULTS
            # ============================================================
            active_processing[document_id]['current_node'] = 'analyze_results'

            yield {
                'type': 'node_start',
                'node': 'analyze_results',
                'document_id': document_id,
                'timestamp': datetime.now().isoformat()
            }

            state = self.graph._analyze_results_node(state)
            analysis = state['analysis']

            # Quality scores
            quality_scores = analysis.get('quality_scores', {})
            for metric, score in quality_scores.items():
                yield {
                    'type': 'quality_metric',
                    'document_id': document_id,
                    'node': 'analyze_results',
                    'metric': metric,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                }
                await asyncio.sleep(0.05)

            # Insights
            for insight in analysis.get('insights', [])[:5]:
                yield {
                    'type': 'insight_discovered',
                    'document_id': document_id,
                    'node': 'analyze_results',
                    'insight': insight,
                    'timestamp': datetime.now().isoformat()
                }
                await asyncio.sleep(0.1)

            yield {
                'type': 'node_complete',
                'node': 'analyze_results',
                'document_id': document_id,
                'quality_overall': quality_scores.get('overall', 0),
                'insights_count': len(analysis.get('insights', [])),
                'timestamp': datetime.now().isoformat()
            }

            active_processing[document_id]['progress'] = 0.8

            # ============================================================
            # NODE 5: REFINEMENT (if needed)
            # ============================================================
            decision = self.graph._should_refine(state)

            if decision == 'refine':
                active_processing[document_id]['current_node'] = 'refine_processing'

                yield {
                    'type': 'node_start',
                    'node': 'refine_processing',
                    'document_id': document_id,
                    'reason': 'quality_below_threshold',
                    'current_quality': quality_scores.get('overall', 0),
                    'threshold': quality_threshold,
                    'timestamp': datetime.now().isoformat()
                }

                state = self.graph._refine_processing_node(state)

                yield {
                    'type': 'node_complete',
                    'node': 'refine_processing',
                    'document_id': document_id,
                    'iteration': state.get('iteration', 1),
                    'timestamp': datetime.now().isoformat()
                }

                # Note: Would loop back to consciousness_processing in real implementation

            # ============================================================
            # NODE 6: FINALIZE
            # ============================================================
            active_processing[document_id]['current_node'] = 'finalize_output'

            yield {
                'type': 'node_start',
                'node': 'finalize_output',
                'document_id': document_id,
                'timestamp': datetime.now().isoformat()
            }

            state = self.graph._finalize_output_node(state)
            final_output = state['final_output']

            # Knowledge graph creation
            kg = final_output.get('knowledge_graph', {})
            yield {
                'type': 'knowledge_graph_created',
                'document_id': document_id,
                'node': 'finalize_output',
                'nodes_created': len(kg.get('nodes', [])),
                'relationships_created': len(kg.get('relationships', [])),
                'timestamp': datetime.now().isoformat()
            }

            yield {
                'type': 'node_complete',
                'node': 'finalize_output',
                'document_id': document_id,
                'timestamp': datetime.now().isoformat()
            }

            active_processing[document_id]['progress'] = 1.0

            # ============================================================
            # PROCESSING COMPLETE
            # ============================================================
            yield {
                'type': 'processing_complete',
                'document_id': document_id,
                'filename': filename,
                'final_output': final_output,
                'total_iterations': state.get('iteration', 1),
                'timestamp': datetime.now().isoformat()
            }

            # Remove from queue and active processing
            processing_queue[:] = [d for d in processing_queue if d['id'] != document_id]
            del active_processing[document_id]
            await broadcast_queue_status()

        except Exception as e:
            logger.error(f"Processing failed for {document_id}: {e}")

            yield {
                'type': 'processing_error',
                'document_id': document_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

            # Cleanup
            processing_queue[:] = [d for d in processing_queue if d['id'] != document_id]
            if document_id in active_processing:
                del active_processing[document_id]
            await broadcast_queue_status()

            raise
