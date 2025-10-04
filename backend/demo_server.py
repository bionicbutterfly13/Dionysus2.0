#!/usr/bin/env python3
"""
Standalone Demo Server for CLAUSE Phase 2
No external dependencies (Neo4j, Redis) needed
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any
import time
import logging
import io
from PyPDF2 import PdfReader

# Import demo components
from src.services.demo.in_memory_graph import get_demo_graph, get_demo_embedder
from src.services.clause.path_navigator import PathNavigator
from src.services.clause.context_curator import ContextCurator
from src.services.clause.coordinator import LCMAPPOCoordinator
from src.models.clause.coordinator_models import (
    CoordinationRequest,
    BudgetAllocation,
    LambdaParameters
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(title="CLAUSE Phase 2 Demo")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "CLAUSE Phase 2 Demo Server", "status": "ready"}


@app.get("/api/stats/dashboard")
async def get_dashboard_stats() -> Dict[str, Any]:
    """Get dashboard statistics for Flux frontend"""
    graph = get_demo_graph()
    return {
        "documentsProcessed": 0,  # Could track this with a counter
        "conceptsExtracted": len(graph.nodes),
        "curiosityMissions": 0,
        "activeThoughtSeeds": 0,
        "mockData": True
    }


@app.get("/api/demo/graph-status")
async def get_graph_status() -> Dict[str, Any]:
    """Get current status of demo knowledge graph"""
    graph = get_demo_graph()
    return {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges),
        "available_concepts": list(graph.nodes.keys()),
        "sample_edges": graph.edges[:5],
    }


@app.post("/api/demo/process-document")
async def process_document_through_clause(file: UploadFile = File(...)):
    """
    Process document through complete CLAUSE pipeline

    Steps:
    1. Receive document
    2. Extract concepts and add to knowledge graph
    3. Run CLAUSE multi-agent coordination
    4. Return results with full trace
    """
    start_time = time.time()
    processing_stages = []

    # Stage 1: Read document
    stage_start = time.time()
    content = await file.read()

    # Check if it's a PDF
    if file.filename and file.filename.lower().endswith('.pdf'):
        try:
            # Extract text from PDF
            pdf_reader = PdfReader(io.BytesIO(content))
            document_text = ""
            for page in pdf_reader.pages:
                document_text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            document_text = "[PDF extraction failed]"
    else:
        # Try UTF-8 first, fallback to latin-1 if that fails
        try:
            document_text = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                document_text = content.decode('latin-1')
            except:
                # Last resort: ignore errors
                document_text = content.decode('utf-8', errors='ignore')

    processing_stages.append({
        "stage": 1,
        "name": "Document Upload",
        "result": f"Received {len(document_text)} characters",
        "duration_ms": (time.time() - stage_start) * 1000
    })

    # Stage 2: Extract concepts and update graph
    stage_start = time.time()
    graph = get_demo_graph()
    concepts = graph.add_document_concepts(document_text)
    processing_stages.append({
        "stage": 2,
        "name": "Concept Extraction",
        "result": f"Extracted {len(concepts)} concepts: {concepts}",
        "duration_ms": (time.time() - stage_start) * 1000
    })

    # If no concepts found, return early with helpful message
    if not concepts:
        return {
            "document_text": document_text[:500],
            "concepts_extracted": [],
            "clause_response": {
                "result": {"message": "No climate-related concepts found in document"},
                "agent_handoffs": [],
                "conflicts_resolved": 0,
                "performance": {"total_latency_ms": 0}
            },
            "processing_stages": processing_stages,
            "total_time_ms": (time.time() - start_time) * 1000,
            "note": "This demo currently only recognizes climate/energy keywords: climate, warming, greenhouse, carbon, CO2, temperature, weather, renewable, fossil"
        }

    # Stage 3: Initialize CLAUSE agents
    stage_start = time.time()
    embedder = get_demo_embedder()

    # Create navigator with patched async methods
    navigator = PathNavigator(embedding_service=embedder)

    async def get_node_text_async(node_id: str) -> str:
        return graph.get_node_text(node_id)

    async def get_node_degree_async(node_id: str) -> int:
        return graph.get_node_degree(node_id)

    async def get_neighbors_async(node_id: str):
        return graph.get_neighbors(node_id)

    async def get_candidate_hops_async(node_id: str):
        return graph.get_candidate_hops(node_id)

    navigator._get_node_text = get_node_text_async
    navigator._get_node_degree = get_node_degree_async
    navigator._get_neighbors = get_neighbors_async
    navigator._get_candidate_hops = get_candidate_hops_async
    navigator._embed_text = embedder.embed

    # Create curator with patched methods
    curator = ContextCurator(embedding_service=embedder)
    curator._embed_text = embedder.embed

    # Create coordinator
    coordinator = LCMAPPOCoordinator(
        path_navigator=navigator,
        context_curator=curator
    )

    processing_stages.append({
        "stage": 3,
        "name": "CLAUSE Agent Initialization",
        "result": "PathNavigator, ContextCurator, Coordinator ready",
        "duration_ms": (time.time() - stage_start) * 1000
    })

    # Stage 4: Execute CLAUSE coordination
    stage_start = time.time()

    # Build query from first concept
    query_text = f"What is {concepts[0]} and how does it relate to other concepts?"
    start_node = concepts[0]

    request = CoordinationRequest(
        query=query_text,
        budgets=BudgetAllocation(
            beta_edge=50,
            beta_step=10,
            beta_tok=2048
        ),
        lambdas=LambdaParameters(
            lambda_edge=0.1,
            lambda_step=0.05,
            lambda_tok=0.001
        )
    )

    clause_response = await coordinator.coordinate(request)

    processing_stages.append({
        "stage": 4,
        "name": "CLAUSE Multi-Agent Coordination",
        "result": f"Executed {len(clause_response.agent_handoffs)} agents",
        "duration_ms": (time.time() - stage_start) * 1000
    })

    total_time = (time.time() - start_time) * 1000

    return {
        "document_text": document_text[:500] + "..." if len(document_text) > 500 else document_text,
        "concepts_extracted": concepts,
        "clause_response": clause_response.model_dump(),
        "processing_stages": processing_stages,
        "total_time_ms": total_time,
    }


@app.post("/api/demo/simple-query")
async def simple_query(query: str, start_node: str = "climate_change"):
    """Simple query endpoint to test CLAUSE without document upload"""

    graph = get_demo_graph()
    embedder = get_demo_embedder()

    # Create navigator with patched async methods
    navigator = PathNavigator(embedding_service=embedder)

    async def get_node_text_async(node_id: str) -> str:
        return graph.get_node_text(node_id)

    async def get_node_degree_async(node_id: str) -> int:
        return graph.get_node_degree(node_id)

    async def get_neighbors_async(node_id: str):
        return graph.get_neighbors(node_id)

    async def get_candidate_hops_async(node_id: str):
        return graph.get_candidate_hops(node_id)

    navigator._get_node_text = get_node_text_async
    navigator._get_node_degree = get_node_degree_async
    navigator._get_neighbors = get_neighbors_async
    navigator._get_candidate_hops = get_candidate_hops_async
    navigator._embed_text = embedder.embed

    from src.models.clause.path_models import PathNavigationRequest

    request = PathNavigationRequest(
        query=query,
        start_node=start_node,
        step_budget=10
    )

    response = await navigator.navigate(request)

    return {
        "query": query,
        "start_node": start_node,
        "navigation_result": response.model_dump(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
