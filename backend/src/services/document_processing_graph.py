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
from ..config.settings import settings

# Neo4j imports for persistence
try:
    import sys
    from pathlib import Path
    # Add extensions directory to path
    extensions_path = Path(__file__).parent.parent.parent.parent / "extensions" / "context_engineering"
    sys.path.insert(0, str(extensions_path))
    from neo4j_unified_schema import Neo4jUnifiedSchema
    NEO4J_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Neo4j schema not available: {e}. Document storage will be skipped.")
    NEO4J_AVAILABLE = False

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
                 require_neo4j: bool = True):
        self.processor = ConsciousnessDocumentProcessor()
        self.cognition_base = DocumentCognitionBase()
        self.researcher = DocumentResearcher(self.cognition_base)
        self.analyst = DocumentAnalyst(self.cognition_base)

        # Read credentials from .env if not provided
        final_uri = neo4j_uri or settings.NEO4J_URI
        final_user = neo4j_user or settings.NEO4J_USER
        final_password = neo4j_password or settings.NEO4J_PASSWORD

        # Initialize Neo4j connection (REQUIRED by default)
        self.neo4j = None
        self.neo4j_connected = False

        if not NEO4J_AVAILABLE:
            error_msg = "Neo4j schema not available - cannot process documents without storage"
            logger.error(f"❌ {error_msg}")
            if require_neo4j:
                raise RuntimeError(error_msg)
        else:
            try:
                self.neo4j = Neo4jUnifiedSchema(uri=final_uri, user=final_user, password=final_password)
                self.neo4j_connected = self.neo4j.connect()

                if self.neo4j_connected:
                    logger.info(f"✅ Neo4j connected to {final_uri} - document persistence enabled")
                else:
                    error_msg = f"Neo4j connection failed to {final_uri}"
                    logger.error(f"❌ {error_msg}")
                    if require_neo4j:
                        raise RuntimeError(error_msg)
                    else:
                        logger.warning("⚠️ Neo4j required but unavailable - processing disabled")
            except Exception as e:
                error_msg = f"Neo4j initialization failed: {e}"
                logger.error(f"❌ {error_msg}")
                if require_neo4j:
                    raise RuntimeError(error_msg)
                else:
                    logger.warning(f"⚠️ {error_msg} - processing disabled")

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

        state["messages"] = state.get("messages", []) + [
            f"Refining processing - Iteration {state['iteration'] + 1}"
        ]

        logger.info("Refinement applied - looping back to consciousness processing")
        return state

    def _finalize_output_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """
        Node 6: Finalize and package output + persist to Neo4j
        """
        logger.info("[Node 6] Finalizing output")

        result = state["processing_result"]
        analysis = state["analysis"]
        research_plan = state.get("research_plan", {})

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
            "iterations": state["iteration"],
            "messages": state.get("messages", [])
        }

        state["final_output"] = final_output

        # Persist to Neo4j (REQUIRED)
        if not self.neo4j_connected:
            error_msg = "Neo4j not connected - cannot complete processing without storage"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        try:
            self._store_to_neo4j(state, final_output)
            state["messages"] = state.get("messages", []) + [
                "Processing complete - output finalized and persisted to Neo4j"
            ]
            logger.info("✅ Data persisted to Neo4j successfully")
        except Exception as e:
            error_msg = f"Neo4j storage failed: {e}"
            logger.error(f"❌ {error_msg}")
            state["messages"] = state.get("messages", []) + [
                f"FAILED: {error_msg}"
            ]
            # Re-raise to fail the workflow
            raise RuntimeError(error_msg) from e

        logger.info("Final output packaged successfully")
        return state

    def _store_to_neo4j(self, state: DocumentProcessingState, final_output: Dict[str, Any]):
        """
        Store document processing results to Neo4j knowledge graph
        Creates: Document node + Concept nodes + AttractorBasin nodes + Relationships
        """
        if not self.neo4j or not self.neo4j_connected:
            return

        result = state["processing_result"]
        filename = state["filename"]
        content_hash = result.content_hash

        # 1. Create Document node
        document_id = self._create_document_node(
            filename=filename,
            content_hash=content_hash,
            tags=state.get("tags", []),
            final_output=final_output
        )

        # 2. Create Concept nodes and link to document
        concept_ids = []
        concepts_list = result.concepts[:20]  # Limit to top 20 concepts
        for idx, concept in enumerate(concepts_list):
            concept_id = self._create_concept_node(
                concept_text=concept,
                document_id=document_id,
                index=idx
            )
            concept_ids.append(concept_id)

        # 2b. Extract and create concept-to-concept relationships
        # Use basin information to guide relationship extraction
        basin_context = self._get_basin_context_for_concepts(concepts_list)

        relationships_created = self._create_concept_relationships(
            concepts=concepts_list,
            document_id=document_id,
            basin_context=basin_context
        )

        # 2c. LEARNING LOOP: Update Cognition Base with relationship quality
        self._update_cognition_from_results(
            concepts=concepts_list,
            relationships=relationships_created,
            quality_score=final_output.get("quality", {}).get("scores", {}).get("overall", 0),
            basin_context=basin_context
        )

        # 3. Create AttractorBasin nodes from patterns learned
        basin_ids = []
        for pattern in result.patterns_learned[:10]:  # Limit to top 10 patterns
            basin_id = self._create_basin_node(
                basin_data={
                    'center_concept': pattern.get('concept', 'unknown'),
                    'strength': 1.0,
                    'stability': 0.8,
                    'related_concepts': []  # Could extract from pattern if available
                },
                document_id=document_id,
                concept_ids=concept_ids
            )
            basin_ids.append(basin_id)

        # 4. Create research curiosity trigger relationships
        research_plan = state.get("research_plan", {})
        curiosity_triggers = research_plan.get("curiosity_triggers", [])
        for trigger in curiosity_triggers[:10]:
            self._link_curiosity_trigger(
                document_id=document_id,
                trigger=trigger,
                concept_ids=concept_ids
            )

        logger.info(f"Stored to Neo4j: doc={document_id}, concepts={len(concept_ids)}, basins={len(basin_ids)}")

    def _create_document_node(self, filename: str, content_hash: str, tags: List[str], final_output: Dict[str, Any]) -> str:
        """Create Document node in Neo4j"""
        with self.neo4j.driver.session() as session:
            result = session.run("""
                CREATE (d:Document {
                    id: $id,
                    filename: $filename,
                    content_hash: $content_hash,
                    tags: $tags,
                    extracted_text: $summary,
                    upload_timestamp: $timestamp,
                    processing_status: 'completed',
                    chunks_count: $chunks_count,
                    concepts_count: $concepts_count,
                    basins_count: $basins_count,
                    quality_score: $quality_score,
                    iterations: $iterations
                })
                RETURN d.id as document_id
            """,
                id=content_hash,
                filename=filename,
                content_hash=content_hash,
                tags=tags,
                summary=final_output.get("extraction", {}).get("summary", ""),
                timestamp=datetime.now().isoformat(),
                chunks_count=final_output.get("extraction", {}).get("chunks", 0),
                concepts_count=len(final_output.get("extraction", {}).get("concepts", [])),
                basins_count=final_output.get("consciousness", {}).get("basins_created", 0),
                quality_score=final_output.get("quality", {}).get("scores", {}).get("overall", 0),
                iterations=final_output.get("iterations", 0)
            )
            return result.single()["document_id"]

    def _create_concept_node(self, concept_text: str, document_id: str, index: int) -> str:
        """Create Concept node and link to Document"""
        concept_id = hashlib.md5(f"{document_id}:{concept_text}".encode()).hexdigest()[:16]

        with self.neo4j.driver.session() as session:
            session.run("""
                MATCH (d:Document {id: $document_id})
                MERGE (c:Concept {id: $concept_id})
                ON CREATE SET
                    c.text = $concept_text,
                    c.created_at = $timestamp
                MERGE (d)-[r:HAS_CONCEPT]->(c)
                ON CREATE SET
                    r.extraction_index = $index,
                    r.created_at = $timestamp
            """,
                document_id=document_id,
                concept_id=concept_id,
                concept_text=concept_text,
                index=index,
                timestamp=datetime.now().isoformat()
            )
        return concept_id

    def _create_basin_node(self, basin_data: Dict[str, Any], document_id: str, concept_ids: List[str]) -> str:
        """Create AttractorBasin node and link to relevant concepts"""
        basin_id = hashlib.md5(f"{document_id}:{basin_data.get('center_concept', '')}".encode()).hexdigest()[:16]

        with self.neo4j.driver.session() as session:
            # Create basin node
            session.run("""
                MATCH (d:Document {id: $document_id})
                CREATE (b:AttractorBasin {
                    id: $basin_id,
                    center_concept: $center_concept,
                    strength: $strength,
                    stability: $stability,
                    created_at: $timestamp
                })
                CREATE (d)-[:CREATED_BASIN]->(b)
            """,
                document_id=document_id,
                basin_id=basin_id,
                center_concept=basin_data.get('center_concept', 'unknown'),
                strength=basin_data.get('strength', 0.0),
                stability=basin_data.get('stability', 0.0),
                timestamp=datetime.now().isoformat()
            )

            # Link basin to related concepts
            related_concepts = basin_data.get('related_concepts', [])
            for concept_text in related_concepts[:5]:  # Top 5 related concepts
                session.run("""
                    MATCH (b:AttractorBasin {id: $basin_id})
                    MATCH (c:Concept)
                    WHERE c.text = $concept_text
                    MERGE (b)-[:ATTRACTS]->(c)
                """,
                    basin_id=basin_id,
                    concept_text=concept_text
                )

        return basin_id

    def _link_curiosity_trigger(self, document_id: str, trigger: Dict[str, Any], concept_ids: List[str]):
        """Create relationship for curiosity trigger pointing to concept"""
        concept = trigger.get('concept', '')
        prediction_error = trigger.get('prediction_error', 0.0)
        priority = trigger.get('priority', 'low')

        with self.neo4j.driver.session() as session:
            session.run("""
                MATCH (d:Document {id: $document_id})
                MATCH (c:Concept)
                WHERE c.text = $concept
                MERGE (d)-[r:CURIOSITY_TRIGGER]->(c)
                SET r.prediction_error = $prediction_error,
                    r.priority = $priority,
                    r.created_at = $timestamp
            """,
                document_id=document_id,
                concept=concept,
                prediction_error=prediction_error,
                priority=priority,
                timestamp=datetime.now().isoformat()
            )

    def _get_basin_context_for_concepts(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Get attractor basin context for concepts to guide relationship extraction.

        Basins tell us which concepts cluster together = likely relationships.
        """
        # TODO: Query Redis for basin information
        # For now, return placeholder
        return {
            "basins_available": False,
            "concept_clusters": [],
            "basin_strengths": {}
        }

    def _update_cognition_from_results(self,
                                      concepts: List[str],
                                      relationships: List[Dict[str, Any]],
                                      quality_score: float,
                                      basin_context: Dict[str, Any]):
        """
        LEARNING LOOP: Update Cognition Base based on processing results.

        Agents learn patterns:
        - Which relationship types are common in research papers
        - Which concept patterns indicate high-quality extraction
        - Narrative structures that appear repeatedly
        """
        # Record successful pattern
        if quality_score > 0.7:
            # High quality = good pattern
            pattern = {
                "pattern_type": "concept_relationship_extraction",
                "concepts_count": len(concepts),
                "relationships_extracted": len(relationships),
                "relationship_types": list(set(r['type'] for r in relationships)),
                "quality_score": quality_score,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

            self.cognition_base.record_successful_pattern(
                category="relationship_extraction",
                pattern=pattern
            )

            logger.info(f"📚 Cognition updated: Recorded successful pattern ({quality_score:.2f} quality)")

        # Learn from relationship types
        relationship_type_counts = {}
        for rel in relationships:
            rel_type = rel['type']
            relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1

        # If certain relationship types appear frequently, boost their priority
        for rel_type, count in relationship_type_counts.items():
            if count >= 3:  # Appears 3+ times in one document = important pattern
                self.cognition_base.boost_strategy_priority(
                    category="relationship_types",
                    strategy_name=rel_type,
                    boost_amount=0.1
                )

                logger.info(f"📈 Boosted priority for relationship type: {rel_type} (appeared {count} times)")

    def _create_concept_relationships(self, concepts: List[str], document_id: str, basin_context: Dict[str, Any] = None):
        """
        Extract and create semantic relationships between concepts.

        Relationship types:
        - CAUSES: Causality relationships
        - SYNONYM_OF: Synonymous concepts
        - COMPETES_WITH: Competing/conflicting concepts
        - PARENT_OF/CHILD_OF: Hierarchical relationships
        - REPLACED_BY: Outdated idea superseded by newer concept
        - MODIFIED_BY: Concept modified by another
        - REGULATED_BY: Concept regulated/controlled by another
        - NEUTRALIZED_BY: Concept neutralized/negated by another
        - CREATED_BY: Concept created/generated by another
        - ADVANCED_BY: Concept advanced/improved by another
        - DEPRECATED_BY: Concept deprecated in favor of another
        """
        if len(concepts) < 2:
            return

        # Use simple heuristics for relationship detection
        # In production, you'd use LLM or NLP to extract these
        relationships = self._analyze_concept_relationships(concepts)

        with self.neo4j.driver.session() as session:
            for rel in relationships:
                source_concept = rel['source']
                target_concept = rel['target']
                rel_type = rel['type']
                confidence = rel.get('confidence', 0.5)

                # Create relationship between concepts
                session.run(f"""
                    MATCH (c1:Concept), (c2:Concept)
                    WHERE c1.text = $source AND c2.text = $target
                    MERGE (c1)-[r:{rel_type}]->(c2)
                    SET r.confidence = $confidence,
                        r.document_id = $document_id,
                        r.created_at = $timestamp
                """,
                    source=source_concept,
                    target=target_concept,
                    confidence=confidence,
                    document_id=document_id,
                    timestamp=datetime.now().isoformat()
                )

        logger.info(f"Created {len(relationships)} concept relationships for document {document_id}")
        return relationships

    def _analyze_concept_relationships(self, concepts: List[str]) -> List[Dict[str, Any]]:
        """
        Use LLM to extract semantic relationships between concepts.

        The Neo4j graph accepts ANY relationship type dynamically.
        The LLM can create any relationship type it discovers.

        Returns list of relationships with DYNAMIC types.
        """
        if len(concepts) < 2:
            return []

        # Use Ollama local LLM for relationship extraction
        try:
            relationships = self._llm_extract_relationships(concepts)
            logger.info(f"LLM extracted {len(relationships)} relationships from {len(concepts)} concepts")
            return relationships
        except Exception as e:
            logger.error(f"LLM relationship extraction failed: {e}")
            logger.warning("Falling back to heuristic relationship detection")
            return self._heuristic_relationships(concepts)

    def _llm_extract_relationships(self, concepts: List[str]) -> List[Dict[str, Any]]:
        """
        Use Ollama to extract relationships between concepts.

        NO BULLSHIT. This uses actual LLM understanding.
        """
        prompt = f"""You are analyzing research concepts to build a knowledge graph.

Concepts:
{json.dumps(concepts, indent=2)}

Extract ALL semantic relationships between these concepts. Be thorough and precise.

For EACH pair of related concepts, identify:
1. The SOURCE concept
2. The TARGET concept
3. The RELATIONSHIP TYPE (use precise verbs: CAUSES, ENABLES, REQUIRES, EXTENDS, VALIDATES, CONTRADICTS, REPLACES, MODIFIES, REGULATES, etc.)
4. CONFIDENCE (0.0-1.0)

Rules:
- Use UPPERCASE_WITH_UNDERSCORES for relationship types (e.g., THEORETICALLY_EXTENDS, EMPIRICALLY_VALIDATES)
- Be specific: prefer "EMPIRICALLY_VALIDATES" over generic "RELATES_TO"
- Only include relationships you're confident about (>0.5 confidence)
- For research papers: look for theoretical extensions, empirical validations, contradictions, refinements

Return ONLY a JSON array, no other text:
[
  {{"source": "concept A", "target": "concept B", "type": "RELATIONSHIP_TYPE", "confidence": 0.85}},
  ...
]"""

        # Call Ollama
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'qwen2.5:14b',
                'prompt': prompt,
                'stream': False,
                'temperature': 0.3,  # Lower temperature for more factual extraction
                'format': 'json'
            },
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API returned {response.status_code}")

        result = response.json()
        llm_output = result.get('response', '').strip()

        # Parse JSON response
        try:
            relationships = json.loads(llm_output)

            # Validate structure
            if not isinstance(relationships, list):
                raise ValueError("LLM did not return a list")

            # Validate each relationship
            valid_relationships = []
            for rel in relationships:
                if all(k in rel for k in ['source', 'target', 'type', 'confidence']):
                    # Ensure relationship type is uppercase
                    rel['type'] = rel['type'].upper().replace(' ', '_')
                    valid_relationships.append(rel)
                else:
                    logger.warning(f"Skipping malformed relationship: {rel}")

            return valid_relationships

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON output: {e}")
            logger.error(f"LLM output was: {llm_output[:500]}")
            raise

    def _heuristic_relationships(self, concepts: List[str]) -> List[Dict[str, Any]]:
        """
        FALLBACK ONLY: Simple heuristic relationship detection.
        Only used if LLM fails.
        """
        relationships = []

        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i >= j:
                    continue

                concept1_lower = concept1.lower()
                concept2_lower = concept2.lower()

                # Detect relationship TYPE dynamically from text
                detected_rels = self._detect_relationship_type(concept1_lower, concept2_lower)

                for rel_type, confidence in detected_rels:
                    relationships.append({
                        'source': concept1,
                        'target': concept2,
                        'type': rel_type,
                        'confidence': confidence
                    })

        return relationships

    def _detect_relationship_type(self, concept1: str, concept2: str) -> List[tuple]:
        """
        Detect relationship type(s) between two concepts.

        Returns list of (relationship_type, confidence) tuples.
        Neo4j will create these relationship types dynamically.
        """
        detected = []

        # Pattern 1: Extract verbs/actions from concept text
        # If concept contains "X causes Y", create CAUSES relationship
        verbs_map = {
            'cause': 'CAUSES',
            'lead': 'LEADS_TO',
            'result': 'RESULTS_IN',
            'produce': 'PRODUCES',
            'enable': 'ENABLES',
            'prevent': 'PREVENTS',
            'require': 'REQUIRES',
            'depend': 'DEPENDS_ON',
            'influence': 'INFLUENCES',
            'affect': 'AFFECTS',
            'modify': 'MODIFIES',
            'change': 'CHANGES',
            'improve': 'IMPROVES',
            'enhance': 'ENHANCES',
            'replace': 'REPLACES',
            'supersede': 'SUPERSEDES',
            'regulate': 'REGULATES',
            'control': 'CONTROLS',
            'govern': 'GOVERNS',
            'manage': 'MANAGES',
            'neutralize': 'NEUTRALIZES',
            'negate': 'NEGATES',
            'cancel': 'CANCELS',
            'create': 'CREATES',
            'generate': 'GENERATES',
            'advance': 'ADVANCES',
            'develop': 'DEVELOPS',
            'deprecate': 'DEPRECATES',
        }

        for verb, rel_type in verbs_map.items():
            if verb in concept1 or verb in concept2:
                # Determine direction based on which concept has the verb
                detected.append((rel_type, 0.6))

        # Pattern 2: Semantic similarity = RELATED_TO
        if any(word in concept1 and word in concept2 for word in ['similar', 'same', 'like']):
            detected.append(('SYNONYM_OF', 0.7))

        # Pattern 3: Opposition = COMPETES_WITH
        if any(word in concept1 or word in concept2 for word in ['versus', 'vs', 'against', 'compete']):
            detected.append(('COMPETES_WITH', 0.6))

        # Pattern 4: Hierarchy detection
        if any(phrase in concept1 for phrase in ['type of', 'kind of', 'subset']):
            detected.append(('IS_A', 0.7))

        # Pattern 5: If no specific relationship found, create generic RELATED_TO
        # This ensures concepts are always connected if they co-occur
        if not detected:
            # Check for co-occurrence (both concepts in same document = related)
            detected.append(('RELATED_TO', 0.3))

        return detected

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
        """Clean up resources including Neo4j connection"""
        if self.neo4j and self.neo4j_connected:
            try:
                self.neo4j.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.warning(f"Error closing Neo4j connection: {e}")

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
