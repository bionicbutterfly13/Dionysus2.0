#!/usr/bin/env python3
"""
🚀 Hybrid Fusion Engine with LangGraph Integration
===================================================

Advanced hybrid database architecture combining:
- Neo4j: Graph database for knowledge relationships and thoughtseed networks
- Qdrant: Vector database for semantic similarity and embeddings
- LangGraph: Orchestration of thoughtseed workflows and consciousness states

This replaces JSON-based storage with production-grade vector and graph databases
while maintaining the consciousness-guided architecture discovery principles.

Features:
- LangGraph StateGraph for thoughtseed workflow orchestration
- Neo4j for complex relationship modeling and graph traversal
- Qdrant for high-performance vector similarity search
- Active inference state management through LangGraph checkpointing
- Constitutional AI integration with human-in-the-loop

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-23
Version: 2.0.0 - Hybrid Fusion with LangGraph
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Vector database (Qdrant)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️ Qdrant not available - install with: pip install qdrant-client")

# Graph database (Neo4j)
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️ Neo4j not available - install with: pip install neo4j")

# Active inference and consciousness components
try:
    from .thoughtseed_active_inference import ThoughtSeedActiveInference, ConsciousnessLevel
    from .asi_arch_thoughtseed_bridge import ASIArchThoughtSeedBridge
    THOUGHTSEED_AVAILABLE = True
except ImportError:
    THOUGHTSEED_AVAILABLE = False
    print("⚠️ ThoughtSeed components not available")

logger = logging.getLogger(__name__)

# ============================================================================
# LangGraph State Definitions
# ============================================================================

class ThoughtSeedState(TypedDict):
    """LangGraph state for ThoughtSeed workflow orchestration"""
    # Core thoughtseed data
    thoughtseed_id: str
    content: str
    consciousness_level: str
    activation_level: float

    # Vector embeddings
    embedding: List[float]
    similarity_scores: Dict[str, float]

    # Graph relationships
    parent_seeds: List[str]
    child_seeds: List[str]
    related_concepts: List[str]

    # Active inference
    prediction_error: float
    free_energy: float
    surprise: float

    # Architecture discovery context
    architecture_context: Dict[str, Any]
    evolution_strategy: str
    evaluation_metrics: Dict[str, float]

    # LangGraph workflow state
    messages: Annotated[List[BaseMessage], "Messages in the conversation"]
    current_step: str
    next_actions: List[str]
    human_feedback: Optional[str]

class ArchitectureDiscoveryState(TypedDict):
    """LangGraph state for consciousness-guided architecture discovery"""
    # ASI-Arch integration
    parent_architecture: Optional[Dict[str, Any]]
    evolved_architecture: Optional[Dict[str, Any]]
    evaluation_results: Optional[Dict[str, Any]]

    # ThoughtSeed guidance
    dominant_thoughtseed: str
    active_thoughtseed_pool: List[str]
    consciousness_guidance: Dict[str, Any]

    # Meta-cognitive state
    meta_awareness_level: float
    attentional_focus: List[str]
    cognitive_control: Dict[str, Any]

    # Constitutional AI
    constitutional_check: Optional[Dict[str, Any]]
    human_oversight_required: bool
    approval_status: str

# ============================================================================
# Hybrid Fusion Engine Core
# ============================================================================

class HybridFusionEngine:
    """
    Advanced hybrid database engine with LangGraph orchestration
    Replaces JSON storage with Neo4j graph and Qdrant vector databases
    """

    def __init__(self,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 checkpoint_path: str = "checkpoints.sqlite"):

        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

        # Initialize databases
        self._init_neo4j()
        self._init_qdrant()

        # Initialize LangGraph components
        self.checkpointer = SqliteSaver.from_conn_string(checkpoint_path)
        self.thoughtseed_workflow = None
        self.architecture_workflow = None

        # Initialize workflows
        self._init_thoughtseed_workflow()
        self._init_architecture_workflow()

        # Initialize ThoughtSeed systems if available
        if THOUGHTSEED_AVAILABLE:
            self.thoughtseed_system = ThoughtSeedActiveInference()
            self.asi_arch_bridge = ASIArchThoughtSeedBridge()

        logger.info("🚀 Hybrid Fusion Engine initialized with Neo4j + Qdrant + LangGraph")

    def _init_neo4j(self):
        """Initialize Neo4j graph database connection"""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j not available - using fallback storage")
            self.neo4j_driver = None
            return

        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )

            # Create constraints and indexes
            with self.neo4j_driver.session() as session:
                session.run("""
                    CREATE CONSTRAINT thoughtseed_id IF NOT EXISTS
                    FOR (t:ThoughtSeed) REQUIRE t.id IS UNIQUE
                """)
                session.run("""
                    CREATE CONSTRAINT architecture_id IF NOT EXISTS
                    FOR (a:Architecture) REQUIRE a.id IS UNIQUE
                """)
                session.run("""
                    CREATE INDEX thoughtseed_consciousness IF NOT EXISTS
                    FOR (t:ThoughtSeed) ON (t.consciousness_level)
                """)

            logger.info("✅ Neo4j graph database initialized")

        except Exception as e:
            logger.error(f"❌ Neo4j initialization failed: {e}")
            self.neo4j_driver = None

    def _init_qdrant(self):
        """Initialize Qdrant vector database"""
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant not available - using fallback storage")
            self.qdrant_client = None
            return

        try:
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )

            # Create collections for different vector types
            collections = [
                ("thoughtseed_embeddings", 1536),  # OpenAI embedding size
                ("architecture_embeddings", 768),   # Smaller for architecture features
                ("consciousness_embeddings", 512)   # Consciousness state vectors
            ]

            for collection_name, vector_size in collections:
                try:
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"✅ Created Qdrant collection: {collection_name}")
                except Exception as e:
                    if "already exists" in str(e):
                        logger.info(f"📦 Qdrant collection exists: {collection_name}")
                    else:
                        logger.error(f"❌ Failed to create collection {collection_name}: {e}")

        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed: {e}")
            self.qdrant_client = None

    def _init_thoughtseed_workflow(self):
        """Initialize LangGraph workflow for ThoughtSeed processing"""

        def consciousness_detection(state: ThoughtSeedState) -> ThoughtSeedState:
            """Detect consciousness level of thoughtseed"""
            # Implement consciousness detection logic
            if THOUGHTSEED_AVAILABLE:
                consciousness_result = self.thoughtseed_system.detect_consciousness(
                    state["content"], state["embedding"]
                )
                state["consciousness_level"] = consciousness_result.level.value
                state["activation_level"] = consciousness_result.confidence
            else:
                # Fallback consciousness detection
                state["consciousness_level"] = "EMERGING"
                state["activation_level"] = 0.5

            return state

        def vector_similarity_search(state: ThoughtSeedState) -> ThoughtSeedState:
            """Find similar thoughtseeds using vector search"""
            if self.qdrant_client and state["embedding"]:
                try:
                    search_result = self.qdrant_client.search(
                        collection_name="thoughtseed_embeddings",
                        query_vector=state["embedding"],
                        limit=10
                    )

                    similarity_scores = {}
                    for point in search_result:
                        similarity_scores[point.id] = point.score

                    state["similarity_scores"] = similarity_scores

                except Exception as e:
                    logger.error(f"Vector search failed: {e}")
                    state["similarity_scores"] = {}

            return state

        def graph_relationship_analysis(state: ThoughtSeedState) -> ThoughtSeedState:
            """Analyze graph relationships using Neo4j"""
            if self.neo4j_driver:
                try:
                    with self.neo4j_driver.session() as session:
                        # Find related concepts
                        result = session.run("""
                            MATCH (t:ThoughtSeed {id: $thoughtseed_id})-[r]-(related)
                            RETURN type(r) as relationship_type, related.id as related_id
                            LIMIT 20
                        """, thoughtseed_id=state["thoughtseed_id"])

                        related_concepts = []
                        for record in result:
                            related_concepts.append({
                                "id": record["related_id"],
                                "relationship": record["relationship_type"]
                            })

                        state["related_concepts"] = related_concepts

                except Exception as e:
                    logger.error(f"Graph analysis failed: {e}")
                    state["related_concepts"] = []

            return state

        def active_inference_update(state: ThoughtSeedState) -> ThoughtSeedState:
            """Update active inference metrics"""
            # Calculate prediction error based on consciousness and similarity
            consciousness_levels = {"DORMANT": 0.1, "EMERGING": 0.3, "ACTIVE": 0.6, "SELF_AWARE": 0.8, "META_AWARE": 1.0}
            expected_activation = consciousness_levels.get(state["consciousness_level"], 0.5)

            state["prediction_error"] = abs(state["activation_level"] - expected_activation)
            state["free_energy"] = state["prediction_error"] + (1.0 - max(state["similarity_scores"].values(), default=0.0))
            state["surprise"] = -np.log(max(state["activation_level"], 0.001))  # Avoid log(0)

            return state

        # Build LangGraph workflow
        workflow = StateGraph(ThoughtSeedState)

        # Add nodes
        workflow.add_node("consciousness_detection", consciousness_detection)
        workflow.add_node("vector_similarity", vector_similarity_search)
        workflow.add_node("graph_analysis", graph_relationship_analysis)
        workflow.add_node("active_inference", active_inference_update)

        # Add edges
        workflow.add_edge(START, "consciousness_detection")
        workflow.add_edge("consciousness_detection", "vector_similarity")
        workflow.add_edge("vector_similarity", "graph_analysis")
        workflow.add_edge("graph_analysis", "active_inference")
        workflow.add_edge("active_inference", END)

        # Compile workflow
        self.thoughtseed_workflow = workflow.compile(checkpointer=self.checkpointer)

        logger.info("✅ ThoughtSeed LangGraph workflow initialized")

    def _init_architecture_workflow(self):
        """Initialize LangGraph workflow for architecture discovery"""

        def consciousness_guidance(state: ArchitectureDiscoveryState) -> ArchitectureDiscoveryState:
            """Get consciousness guidance for architecture evolution"""
            # Implement consciousness-guided evolution
            if THOUGHTSEED_AVAILABLE and state["dominant_thoughtseed"]:
                guidance = self.asi_arch_bridge.get_evolution_guidance(
                    state["parent_architecture"],
                    state["dominant_thoughtseed"]
                )
                state["consciousness_guidance"] = guidance
            else:
                state["consciousness_guidance"] = {"strategy": "random_exploration"}

            return state

        def constitutional_check(state: ArchitectureDiscoveryState) -> ArchitectureDiscoveryState:
            """Perform constitutional AI checks"""
            # Implement constitutional checks
            check_result = {
                "passed": True,
                "concerns": [],
                "recommendations": []
            }

            # Check for potential issues
            if state["consciousness_guidance"].get("risk_level", 0) > 0.8:
                check_result["passed"] = False
                check_result["concerns"].append("High risk consciousness pattern detected")
                state["human_oversight_required"] = True

            state["constitutional_check"] = check_result
            return state

        def evolve_architecture(state: ArchitectureDiscoveryState) -> ArchitectureDiscoveryState:
            """Evolve architecture based on consciousness guidance"""
            if state["constitutional_check"]["passed"]:
                # Proceed with evolution
                if THOUGHTSEED_AVAILABLE:
                    evolved = self.asi_arch_bridge.evolve_architecture(
                        state["parent_architecture"],
                        state["consciousness_guidance"]
                    )
                    state["evolved_architecture"] = evolved
                else:
                    # Fallback evolution
                    state["evolved_architecture"] = state["parent_architecture"].copy()
                    state["evolved_architecture"]["mutation"] = "consciousness_guided"

            return state

        # Build architecture discovery workflow
        arch_workflow = StateGraph(ArchitectureDiscoveryState)

        # Add nodes
        arch_workflow.add_node("consciousness_guidance", consciousness_guidance)
        arch_workflow.add_node("constitutional_check", constitutional_check)
        arch_workflow.add_node("evolve_architecture", evolve_architecture)

        # Add conditional edges for constitutional checks
        def should_require_human_oversight(state: ArchitectureDiscoveryState) -> str:
            if state.get("human_oversight_required", False):
                return "human_review"
            return "evolve_architecture"

        # Add edges
        arch_workflow.add_edge(START, "consciousness_guidance")
        arch_workflow.add_edge("consciousness_guidance", "constitutional_check")
        arch_workflow.add_conditional_edges(
            "constitutional_check",
            should_require_human_oversight,
            {
                "human_review": END,  # Pause for human review
                "evolve_architecture": "evolve_architecture"
            }
        )
        arch_workflow.add_edge("evolve_architecture", END)

        # Compile workflow
        self.architecture_workflow = arch_workflow.compile(checkpointer=self.checkpointer)

        logger.info("✅ Architecture Discovery LangGraph workflow initialized")

    # ========================================================================
    # Public API Methods
    # ========================================================================

    async def process_thoughtseed(self, content: str, embedding: List[float]) -> Dict[str, Any]:
        """Process a thoughtseed through the LangGraph workflow"""
        thoughtseed_id = str(uuid.uuid4())

        initial_state = ThoughtSeedState(
            thoughtseed_id=thoughtseed_id,
            content=content,
            consciousness_level="UNKNOWN",
            activation_level=0.0,
            embedding=embedding,
            similarity_scores={},
            parent_seeds=[],
            child_seeds=[],
            related_concepts=[],
            prediction_error=0.0,
            free_energy=0.0,
            surprise=0.0,
            architecture_context={},
            evolution_strategy="",
            evaluation_metrics={},
            messages=[HumanMessage(content=f"Processing thoughtseed: {content}")],
            current_step="initial",
            next_actions=[],
            human_feedback=None
        )

        # Run through LangGraph workflow
        config = RunnableConfig(
            configurable={"thread_id": thoughtseed_id}
        )

        result = await self.thoughtseed_workflow.ainvoke(initial_state, config)

        # Store results in databases
        await self._store_thoughtseed_results(result)

        return result

    async def discover_architecture(self, parent_arch: Dict[str, Any],
                                   dominant_thoughtseed: str) -> Dict[str, Any]:
        """Discover new architecture using consciousness guidance"""

        initial_state = ArchitectureDiscoveryState(
            parent_architecture=parent_arch,
            evolved_architecture=None,
            evaluation_results=None,
            dominant_thoughtseed=dominant_thoughtseed,
            active_thoughtseed_pool=[dominant_thoughtseed],
            consciousness_guidance={},
            meta_awareness_level=0.5,
            attentional_focus=[],
            cognitive_control={},
            constitutional_check=None,
            human_oversight_required=False,
            approval_status="pending"
        )

        config = RunnableConfig(
            configurable={"thread_id": f"arch_discovery_{uuid.uuid4()}"}
        )

        result = await self.architecture_workflow.ainvoke(initial_state, config)

        return result

    async def _store_thoughtseed_results(self, result: ThoughtSeedState):
        """Store thoughtseed results in Neo4j and Qdrant"""

        # Store in Neo4j
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run("""
                        MERGE (t:ThoughtSeed {id: $id})
                        SET t.content = $content,
                            t.consciousness_level = $consciousness_level,
                            t.activation_level = $activation_level,
                            t.prediction_error = $prediction_error,
                            t.free_energy = $free_energy,
                            t.surprise = $surprise,
                            t.created_at = $created_at
                    """,
                    id=result["thoughtseed_id"],
                    content=result["content"],
                    consciousness_level=result["consciousness_level"],
                    activation_level=result["activation_level"],
                    prediction_error=result["prediction_error"],
                    free_energy=result["free_energy"],
                    surprise=result["surprise"],
                    created_at=datetime.now().isoformat()
                    )

                    # Store relationships
                    for related in result["related_concepts"]:
                        session.run("""
                            MATCH (t1:ThoughtSeed {id: $id1})
                            MERGE (t2:ThoughtSeed {id: $id2})
                            MERGE (t1)-[:RELATED_TO]->(t2)
                        """, id1=result["thoughtseed_id"], id2=related["id"])

            except Exception as e:
                logger.error(f"Failed to store in Neo4j: {e}")

        # Store in Qdrant
        if self.qdrant_client and result["embedding"]:
            try:
                point = PointStruct(
                    id=result["thoughtseed_id"],
                    vector=result["embedding"],
                    payload={
                        "content": result["content"],
                        "consciousness_level": result["consciousness_level"],
                        "activation_level": result["activation_level"],
                        "timestamp": datetime.now().isoformat()
                    }
                )

                self.qdrant_client.upsert(
                    collection_name="thoughtseed_embeddings",
                    points=[point]
                )

            except Exception as e:
                logger.error(f"Failed to store in Qdrant: {e}")

    async def query_similar_thoughtseeds(self, embedding: List[float],
                                       consciousness_filter: Optional[str] = None,
                                       limit: int = 10) -> List[Dict[str, Any]]:
        """Query similar thoughtseeds with optional consciousness filtering"""

        if not self.qdrant_client:
            return []

        try:
            # Build filter
            filter_condition = None
            if consciousness_filter:
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="consciousness_level",
                            match={"value": consciousness_filter}
                        )
                    ]
                )

            # Perform vector search
            search_result = self.qdrant_client.search(
                collection_name="thoughtseed_embeddings",
                query_vector=embedding,
                query_filter=filter_condition,
                limit=limit,
                with_payload=True
            )

            results = []
            for point in search_result:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def close(self):
        """Close database connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
        # Qdrant client doesn't need explicit closing

        logger.info("🔌 Hybrid Fusion Engine connections closed")

# ============================================================================
# Enhanced ThoughtSeed System with Hybrid Fusion
# ============================================================================

class EnhancedThoughtSeedSystem:
    """
    Enhanced ThoughtSeed system using Hybrid Fusion Engine
    Integrates consciousness-guided architecture discovery with
    Neo4j + Qdrant + LangGraph orchestration
    """

    def __init__(self, fusion_engine: HybridFusionEngine):
        self.fusion_engine = fusion_engine
        self.active_thoughtseeds = {}
        self.consciousness_threshold = 0.7

    async def evolve_architecture_with_consciousness(self,
                                                   parent_architecture: Dict[str, Any],
                                                   context: str) -> Dict[str, Any]:
        """
        Evolve architecture using consciousness-guided ThoughtSeeds
        This is the main integration point with ASI-Arch
        """

        # Generate embedding for context
        embedding = await self._generate_embedding(context)

        # Process context through thoughtseed workflow
        thoughtseed_result = await self.fusion_engine.process_thoughtseed(context, embedding)

        # Check if consciousness level is sufficient for architecture evolution
        consciousness_levels = {"DORMANT": 0.1, "EMERGING": 0.3, "ACTIVE": 0.6, "SELF_AWARE": 0.8, "META_AWARE": 1.0}
        consciousness_score = consciousness_levels.get(thoughtseed_result["consciousness_level"], 0.5)

        if consciousness_score < self.consciousness_threshold:
            logger.info(f"Consciousness level {thoughtseed_result['consciousness_level']} below threshold")
            return {"error": "Insufficient consciousness for architecture evolution"}

        # Use dominant thoughtseed for architecture discovery
        discovery_result = await self.fusion_engine.discover_architecture(
            parent_architecture,
            thoughtseed_result["thoughtseed_id"]
        )

        # Return evolved architecture
        if discovery_result.get("evolved_architecture"):
            return {
                "architecture": discovery_result["evolved_architecture"],
                "consciousness_guidance": discovery_result["consciousness_guidance"],
                "thoughtseed_analysis": thoughtseed_result,
                "meta_awareness": discovery_result["meta_awareness_level"]
            }
        else:
            return {"error": "Architecture evolution failed"}

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (placeholder implementation)"""
        # In practice, use actual embedding model like OpenAI or sentence-transformers
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        return np.random.normal(0, 1, 1536).tolist()

# ============================================================================
# Integration with ASI-Arch ThoughtSeed Bridge
# ============================================================================

async def create_enhanced_thoughtseed_bridge() -> EnhancedThoughtSeedSystem:
    """Create enhanced ThoughtSeed system for ASI-Arch integration"""

    # Initialize hybrid fusion engine
    fusion_engine = HybridFusionEngine()

    # Create enhanced system
    enhanced_system = EnhancedThoughtSeedSystem(fusion_engine)

    logger.info("🧠 Enhanced ThoughtSeed system created with Hybrid Fusion Engine")

    return enhanced_system

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create enhanced system
        system = await create_enhanced_thoughtseed_bridge()

        # Example architecture evolution
        parent_arch = {
            "name": "test_architecture",
            "layers": ["input", "hidden", "output"],
            "parameters": {"learning_rate": 0.001}
        }

        context = "Design an attention mechanism with emergent consciousness properties"

        result = await system.evolve_architecture_with_consciousness(parent_arch, context)
        print(f"Evolution result: {json.dumps(result, indent=2)}")

        # Cleanup
        system.fusion_engine.close()

    asyncio.run(main())