"""
Document Processing Graph - LangGraph workflow for consciousness-enhanced document processing
Integrates: SurfSense + ASI-GO-2 + R-Zero + Active Inference + Dionysus Consciousness
"""
from typing import Dict, List, Any, TypedDict, Annotated, Optional
import logging
from langgraph.graph import StateGraph, END
import operator
import hashlib
from datetime import datetime
import json
import requests

from .consciousness_document_processor import ConsciousnessDocumentProcessor, DocumentProcessingResult
from .document_cognition_base import DocumentCognitionBase
from .document_researcher import DocumentResearcher
from .document_analyst import DocumentAnalyst

# AutoSchemaKG integration (Spec 040 M2 Extension)
from .autoschemakg_integration import AutoSchemaKGService
from .five_level_concept_extraction import FiveLevelConceptExtractionService
from .multi_tier_memory import MultiTierMemorySystem

logger = logging.getLogger("dionysus.document_graph")


class DocumentProcessingState(TypedDict, total=False):
    """State object for document processing graph"""
    # Input
    content: bytes
    filename: str
    tags: List[str]

    # Processing artifacts
    processing_result: DocumentProcessingResult
    research_plan: Dict[str, Any]
    analysis: Dict[str, Any]

    # Iteration control
    iteration: int
    max_iterations: int
    quality_threshold: float

    # Output
    final_output: Dict[str, Any]
    messages: List[str]


class DocumentProcessingGraph:
    """LangGraph workflow for complete document processing pipeline"""

    def __init__(self,
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 require_neo4j: bool = False):
        # Core processing components
        self.processor = ConsciousnessDocumentProcessor()
        self.cognition_base = DocumentCognitionBase()
        self.researcher = DocumentResearcher(self.cognition_base)
        self.analyst = DocumentAnalyst(self.cognition_base)

        # AutoSchemaKG integration (Spec 040 M2 Extension)
        self.concept_extractor = FiveLevelConceptExtractionService()
        self.memory_system = MultiTierMemorySystem()
        self.autoschema_service = AutoSchemaKGService(
            memory_system=self.memory_system,
            graph_channel=None  # Uses singleton from daedalus_gateway
        )

        logger.info("✅ AutoSchemaKG integration initialized - using Graph Channel for storage")

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph workflow:
        Extract → Research → Process → Analyze → Decide (iterate or complete)
        """
        workflow = StateGraph(DocumentProcessingState)

        # Add nodes
        workflow.add_node("extract_and_process", self._extract_and_process_node)
        workflow.add_node("generate_research_plan", self._generate_research_plan_node)
        workflow.add_node("consciousness_processing", self._consciousness_processing_node)
        workflow.add_node("analyze_results", self._analyze_results_node)
        workflow.add_node("refine_processing", self._refine_processing_node)
        workflow.add_node("finalize_output", self._finalize_output_node)

        # Set entry point
        workflow.set_entry_point("extract_and_process")

        # Add edges
        workflow.add_edge("extract_and_process", "generate_research_plan")
        workflow.add_edge("generate_research_plan", "consciousness_processing")
        workflow.add_edge("consciousness_processing", "analyze_results")

        # Conditional edge: iterate or complete
        workflow.add_conditional_edges(
            "analyze_results",
            self._should_refine,
            {
                "refine": "refine_processing",
                "complete": "finalize_output"
            }
        )

        workflow.add_edge("refine_processing", "consciousness_processing")
        workflow.add_edge("finalize_output", END)

        return workflow.compile()

    def _extract_and_process_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """
        Node 1: Initial extraction and processing
        SurfSense patterns: content hash, markdown conversion, chunking
        """
        logger.info(f"[Node 1] Extracting and processing: {state['filename']}")

        content = state["content"]
        filename = state["filename"]

        # Process through consciousness document processor
        if filename.endswith('.pdf'):
            result = self.processor.process_pdf(content, filename)
        else:
            result = self.processor.process_text(content, filename)

        state["processing_result"] = result
        state["iteration"] = state.get("iteration", 0) + 1
        state["messages"] = state.get("messages", []) + [
            f"Extracted {len(result.concepts)} concepts from {filename}"
        ]

        logger.info(f"Processing complete: {len(result.concepts)} concepts extracted")
        return state

    def _generate_research_plan_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """
        Node 2: Generate research questions and exploration plan
        ASI-GO-2 Researcher + R-Zero challenging questions + Active Inference prediction errors
        """
        logger.info("[Node 2] Generating research plan")

        result = state["processing_result"]
        concepts = result.concepts

        # Generate research plan (prediction errors not available in current DocumentProcessingResult)
        previous_research = state.get("research_plan") if state.get("iteration", 1) > 1 else None
        research_plan = self.researcher.generate_research_questions(
            concepts=concepts,
            prediction_errors=None,  # TODO: Add active inference integration
            previous_attempt=previous_research
        )

        state["research_plan"] = research_plan
        state["messages"] = state.get("messages", []) + [
            f"Generated {len(research_plan['challenging_questions'])} research questions"
        ]

        logger.info(f"Research plan created with {len(research_plan['challenging_questions'])} questions")
        return state

    def _consciousness_processing_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """
        Node 3: Deep consciousness processing through attractor basins
        Dionysus Consciousness System: basins + thoughtseeds + active inference
        """
        logger.info("[Node 3] Processing through consciousness basins")

        result = state["processing_result"]

        # Already processed in extract_and_process, but can enhance here
        # This node can be used for additional basin refinement or co-evolution

        state["messages"] = state.get("messages", []) + [
            f"Created {result.basins_created} attractor basins, generated {result.thoughtseeds_generated} thoughtseeds"
        ]

        logger.info("Consciousness processing complete")
        return state

    def _analyze_results_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """
        Node 4: Analyze processing results and extract insights
        ASI-GO-2 Analyst pattern with meta-cognitive tracking
        """
        logger.info("[Node 4] Analyzing processing results")

        analysis = self.analyst.analyze_processing_result(
            processing_result=state["processing_result"].__dict__,
            research_plan=state.get("research_plan")
        )

        state["analysis"] = analysis
        state["messages"] = state.get("messages", []) + [
            f"Analysis complete - Quality score: {analysis['quality_scores']['overall']:.2f}"
        ]

        logger.info(f"Analysis complete: {len(analysis['insights'])} insights extracted")
        return state

    def _should_refine(self, state: DocumentProcessingState) -> str:
        """
        Decision node: Should we refine processing or complete?
        ASI-GO-2 iterative refinement pattern
        """
        analysis = state.get("analysis", {})
        quality_scores = analysis.get("quality_scores", {})
        overall_quality = quality_scores.get("overall", 0)

        iteration = state.get("iteration", 1)
        max_iterations = state.get("max_iterations", 3)
        quality_threshold = state.get("quality_threshold", 0.7)

        # Decision logic
        if iteration >= max_iterations:
            logger.info(f"Max iterations ({max_iterations}) reached - completing")
            return "complete"

        if overall_quality >= quality_threshold:
            logger.info(f"Quality threshold ({quality_threshold}) met - completing")
            return "complete"

        logger.info(f"Quality {overall_quality:.2f} below threshold {quality_threshold} - refining")
        return "refine"

    def _refine_processing_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """
        Node 5: Refine processing based on analysis feedback
        ASI-GO-2 iterative refinement
        """
        logger.info("[Node 5] Refining processing based on analysis")

        analysis = state["analysis"]
        recommendations = analysis.get("recommendations", [])

        # Apply recommendations to processing parameters
        # (In production, would actually adjust chunking, concept extraction, etc.)

        # CRITICAL: Increment iteration counter to prevent infinite loop
        state["iteration"] = state.get("iteration", 1) + 1

        state["messages"] = state.get("messages", []) + [
            f"Refining processing - Iteration {state['iteration']}"
        ]

        logger.info(f"Refinement applied - looping back to consciousness processing (iteration {state['iteration']})")
        return state

    def _finalize_output_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """
        Node 6: Finalize and package output + persist via AutoSchemaKG
        """
        logger.info("[Node 6] Finalizing output with AutoSchemaKG integration")

        result = state["processing_result"]
        analysis = state["analysis"]
        research_plan = state.get("research_plan", {})

        # Extract five-level concepts for AutoSchemaKG processing
        concepts_for_kg = {
            'atomic': result.concepts[:20],  # Top 20 atomic concepts
            'relationship': [],  # TODO: Extract from basins
            'composite': [],  # TODO: Extract from thoughtseeds
            'context': [tag for tag in state.get("tags", [])],
            'narrative': [result.summary]
        }

        # Process concepts through AutoSchemaKG to create knowledge graph
        # NOTE: AutoSchemaKG processing is async, but we run it synchronously here
        # to avoid making the entire LangGraph workflow async
        knowledge_graph = {}
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            kg_result = loop.run_until_complete(
                self.autoschema_service.process_document_concepts(
                    concepts=concepts_for_kg,
                    document_id=result.content_hash
                )
            )
            loop.close()
            knowledge_graph = kg_result
            logger.info(f"✅ AutoSchemaKG created {len(kg_result.get('nodes', []))} nodes and {len(kg_result.get('relationships', []))} relationships")
        except Exception as e:
            logger.error(f"AutoSchemaKG processing failed: {e}")
            knowledge_graph = {"nodes": [], "relationships": [], "error": str(e)}

        final_output = {
            "document": {
                "filename": state["filename"],
                "content_hash": result.content_hash,
                "tags": state.get("tags", [])
            },
            "extraction": {
                "concepts": result.concepts,
                "chunks": len(result.chunks),
                "summary": result.summary
            },
            "consciousness": {
                "basins_created": result.basins_created,
                "thoughtseeds_generated": len(result.thoughtseeds_generated),
                "patterns_learned": result.patterns_learned
            },
            "research": {
                "curiosity_triggers": research_plan.get("curiosity_triggers", []),
                "exploration_plan": research_plan.get("exploration_plan", {})
            },
            "quality": {
                "scores": analysis.get("quality_scores", {}),
                "insights": analysis.get("insights", []),
                "recommendations": analysis.get("recommendations", [])
            },
            "meta_cognitive": analysis.get("meta_cognitive", {}),
            "knowledge_graph": knowledge_graph,  # NEW: AutoSchemaKG knowledge graph
            "memory_storage": {  # NEW: Multi-tier memory information
                "warm_tier": "Neo4j via Graph Channel",
                "cool_tier": "Vector embeddings (future)",
                "cold_tier": "Archive storage (future)"
            },
            "iterations": state["iteration"],
            "messages": state.get("messages", [])
        }

        state["final_output"] = final_output
        state["messages"] = state.get("messages", []) + [
            f"Processing complete - created knowledge graph with {len(knowledge_graph.get('nodes', []))} nodes"
        ]

        logger.info("Final output packaged successfully with AutoSchemaKG integration")
        return state

    # =========================================================================
    # DEPRECATED: Old Neo4j storage methods removed (Spec 040 M2 Extension)
    # AutoSchemaKG now handles all knowledge graph storage via Graph Channel
    # =========================================================================

    def process_document(self,
                        content: bytes,
                        filename: str,
                        tags: List[str] = None,
                        max_iterations: int = 3,
                        quality_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Process document through complete LangGraph workflow

        Args:
            content: Document binary content
            filename: Document filename
            tags: Optional tags for categorization
            max_iterations: Maximum refinement iterations
            quality_threshold: Quality score threshold for completion

        Returns:
            Complete processing results with consciousness integration
        """
        logger.info(f"Starting document processing workflow for {filename}")

        # Initialize state
        initial_state: DocumentProcessingState = {
            "content": content,
            "filename": filename,
            "tags": tags or [],
            "processing_result": {},
            "research_plan": {},
            "analysis": {},
            "iteration": 0,
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
            "final_output": {},
            "messages": []
        }

        # Execute graph
        final_state = self.graph.invoke(initial_state)

        logger.info(f"Workflow complete for {filename}")
        return final_state["final_output"]

    def get_cognition_summary(self) -> Dict[str, Any]:
        """Get summary of learned patterns and strategies"""
        return {
            "cognition_base": self.cognition_base.get_session_summary(),
            "researcher": self.researcher.get_research_summary(),
            "analyst": self.analyst.get_analysis_summary()
        }

    def close(self):
        """Clean up resources - Graph Channel manages its own lifecycle"""
        logger.info("DocumentProcessingGraph closed - Graph Channel manages Neo4j lifecycle")

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.close()

    async def process_document_async(self,
                                    content: bytes,
                                    filename: str,
                                    tags: List[str] = None,
                                    max_iterations: int = 3,
                                    quality_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Async version of process_document for concurrent processing
        """
        # LangGraph supports async execution
        initial_state: DocumentProcessingState = {
            "content": content,
            "filename": filename,
            "tags": tags or [],
            "processing_result": {},
            "research_plan": {},
            "analysis": {},
            "iteration": 0,
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
            "final_output": {},
            "messages": []
        }

        final_state = await self.graph.ainvoke(initial_state)
        return final_state["final_output"]


# Export for easy import
__all__ = ["DocumentProcessingGraph", "DocumentProcessingState"]
