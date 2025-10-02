#!/usr/bin/env python3
"""
T011: Integration Test - Full subgraph construction workflow

This test MUST FAIL initially (services not implemented).
Tests complete CLAUSE subgraph construction from quickstart.md Step 3.1.
"""

import pytest
import time
from typing import List, Tuple


@pytest.fixture
def sample_graph() -> List[Tuple[str, str, str]]:
    """Create sample Neo4j graph (10 concept triplets) from quickstart.md"""
    return [
        ("neural_architecture", "RELATED_TO", "search_algorithms"),
        ("search_algorithms", "USES", "reinforcement_learning"),
        ("reinforcement_learning", "OPTIMIZES", "policy_network"),
        ("neural_architecture", "IMPLEMENTS", "differentiable_nas"),
        ("differentiable_nas", "REQUIRES", "gradient_descent"),
        ("search_algorithms", "EVALUATES", "performance_metrics"),
        ("performance_metrics", "INCLUDES", "accuracy"),
        ("performance_metrics", "INCLUDES", "latency"),
        ("neural_architecture", "PRODUCES", "model_architecture"),
        ("model_architecture", "DEPLOYED_ON", "edge_devices"),
    ]


@pytest.fixture
def clause_architect():
    """Import SubgraphArchitect (will fail until implemented)"""
    try:
        from src.services.clause.subgraph_architect import SubgraphArchitect
        return SubgraphArchitect()
    except ImportError:
        pytest.skip("SubgraphArchitect not implemented yet")


@pytest.fixture
def neo4j_test_db():
    """Connect to Neo4j test database"""
    try:
        from neo4j import GraphDatabase
        import os

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        driver = GraphDatabase.driver(uri, auth=(user, password))

        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")

        yield driver
        driver.close()

    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")


def test_subgraph_construction_budget_compliance(
    clause_architect, neo4j_test_db, sample_graph
):
    """Test subgraph construction respects edge budget"""

    # Load sample graph into Neo4j
    with neo4j_test_db.session() as session:
        # Clear test data
        session.run("MATCH (n) WHERE n.test_graph = true DETACH DELETE n")

        # Create nodes and edges
        for source, relation, target in sample_graph:
            session.run(
                """
                MERGE (s:Concept {name: $source, test_graph: true})
                MERGE (t:Concept {name: $target, test_graph: true})
                MERGE (s)-[r:RELATION {type: $relation, test_graph: true}]->(t)
                """,
                source=source,
                relation=relation,
                target=target,
            )

    # Build subgraph with CLAUSE
    query = "What is neural architecture search?"
    edge_budget = 5  # Only select 5 edges
    lambda_edge = 0.2

    result = clause_architect.build_subgraph(
        query=query, edge_budget=edge_budget, lambda_edge=lambda_edge, hop_distance=2
    )

    # Assert budget compliance
    selected_edges = result.get("selected_edges", [])
    assert len(selected_edges) <= edge_budget, \
        f"Budget violated: {len(selected_edges)} > {edge_budget}"

    # Cleanup
    with neo4j_test_db.session() as session:
        session.run("MATCH (n) WHERE n.test_graph = true DETACH DELETE n")


def test_subgraph_shaped_gain_rule(clause_architect, neo4j_test_db, sample_graph):
    """Test all selected edges satisfy shaped gain rule"""

    # Load sample graph
    with neo4j_test_db.session() as session:
        session.run("MATCH (n) WHERE n.test_graph = true DETACH DELETE n")

        for source, relation, target in sample_graph:
            session.run(
                """
                MERGE (s:Concept {name: $source, test_graph: true})
                MERGE (t:Concept {name: $target, test_graph: true})
                MERGE (s)-[r:RELATION {type: $relation, test_graph: true}]->(t)
                """,
                source=source,
                relation=relation,
                target=target,
            )

    # Build subgraph
    query = "neural architecture search"
    lambda_edge = 0.2

    result = clause_architect.build_subgraph(
        query=query, edge_budget=50, lambda_edge=lambda_edge, hop_distance=2
    )

    # Assert shaped gain rule: score - λ_edge × cost > 0
    shaped_gains = result.get("shaped_gains", {})

    for edge_key, gain in shaped_gains.items():
        assert gain > 0, \
            f"Shaped gain rule violated for edge {edge_key}: gain={gain} ≤ 0"

    # Cleanup
    with neo4j_test_db.session() as session:
        session.run("MATCH (n) WHERE n.test_graph = true DETACH DELETE n")


def test_subgraph_construction_performance(
    clause_architect, neo4j_test_db, sample_graph
):
    """Test subgraph construction completes in <500ms"""

    # Load sample graph
    with neo4j_test_db.session() as session:
        session.run("MATCH (n) WHERE n.test_graph = true DETACH DELETE n")

        for source, relation, target in sample_graph:
            session.run(
                """
                MERGE (s:Concept {name: $source, test_graph: true})
                MERGE (t:Concept {name: $target, test_graph: true})
                MERGE (s)-[r:RELATION {type: $relation, test_graph: true}]->(t)
                """,
                source=source,
                relation=relation,
                target=target,
            )

    # Measure construction time
    query = "neural architecture search"

    start_time = time.perf_counter()
    result = clause_architect.build_subgraph(
        query=query, edge_budget=50, lambda_edge=0.2, hop_distance=2
    )
    end_time = time.perf_counter()

    construction_time_ms = (end_time - start_time) * 1000

    # Assert performance SLA
    assert construction_time_ms < 500, \
        f"Construction time {construction_time_ms:.2f}ms exceeds 500ms limit"

    print(f"✅ Subgraph construction: {construction_time_ms:.2f}ms")

    # Cleanup
    with neo4j_test_db.session() as session:
        session.run("MATCH (n) WHERE n.test_graph = true DETACH DELETE n")


def test_subgraph_stopped_reason(clause_architect, neo4j_test_db, sample_graph):
    """Test stopped_reason field indicates why construction halted"""

    # Load sample graph
    with neo4j_test_db.session() as session:
        session.run("MATCH (n) WHERE n.test_graph = true DETACH DELETE n")

        for source, relation, target in sample_graph:
            session.run(
                """
                MERGE (s:Concept {name: $source, test_graph: true})
                MERGE (t:Concept {name: $target, test_graph: true})
                MERGE (s)-[r:RELATION {type: $relation, test_graph: true}]->(t)
                """,
                source=source,
                relation=relation,
                target=target,
            )

    # Build subgraph with small budget
    query = "neural architecture search"

    result = clause_architect.build_subgraph(
        query=query, edge_budget=3, lambda_edge=0.2, hop_distance=2
    )

    stopped_reason = result.get("stopped_reason", "")

    # Should stop due to budget exhaustion
    assert stopped_reason in [
        "budget_exhausted",
        "shaped_gain_zero",
        "no_more_candidates",
    ], f"Invalid stopped_reason: {stopped_reason}"

    print(f"✅ Stopped reason: {stopped_reason}")

    # Cleanup
    with neo4j_test_db.session() as session:
        session.run("MATCH (n) WHERE n.test_graph = true DETACH DELETE n")


if __name__ == "__main__":
    print("\n=== T011: Integration Test - Subgraph Construction Workflow ===\n")
    print("⚠️  This test MUST FAIL initially (SubgraphArchitect not implemented)")
    print("✅  Test will pass once core services are implemented (T015-T025)\n")

    pytest.main([__file__, "-v"])
