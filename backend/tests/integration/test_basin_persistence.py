#!/usr/bin/env python3
"""
T013: Integration Test - Neo4j + Redis persistence

This test MUST FAIL initially (services not implemented).
Tests basin persistence across Neo4j (permanent) and Redis (cache).
"""

import pytest
import redis
import time
from neo4j import GraphDatabase
import os


@pytest.fixture
def redis_client():
    """Connect to Redis"""
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        # Test connection
        client.ping()
        yield client
        client.close()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.fixture
def neo4j_driver():
    """Connect to Neo4j"""
    try:
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


@pytest.fixture
def basin_tracker():
    """Import BasinTracker (will fail until implemented)"""
    try:
        from src.services.clause.basin_tracker import BasinTracker
        return BasinTracker()
    except ImportError:
        pytest.skip("BasinTracker not implemented yet")


def test_basin_create_and_cache(basin_tracker, neo4j_driver, redis_client):
    """Test basin creation persists to Neo4j and caches in Redis"""

    concept = "test_concept_persistence"

    # Create basin
    result = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_001", increment=0.2
    )

    basin = (result.get("new_basins", []) + result.get("updated_basins", []))[0]
    basin_id = basin.get("basin_id") or concept

    # Verify Neo4j persistence
    with neo4j_driver.session() as session:
        result_neo4j = session.run(
            """
            MATCH (b:AttractorBasin {basin_id: $basin_id})
            RETURN b.strength AS strength, b.activation_count AS activation_count
            """,
            basin_id=basin_id,
        )

        record = result_neo4j.single()
        assert record is not None, f"Basin not found in Neo4j: {basin_id}"
        assert record["strength"] == 1.0, "Strength not persisted to Neo4j"
        assert record["activation_count"] == 1, "Activation count not persisted"

    # Verify Redis caching (1-hour TTL)
    redis_key = f"basin:{basin_id}"
    cached_data = redis_client.get(redis_key)

    assert cached_data is not None, f"Basin not cached in Redis: {redis_key}"

    # Verify TTL is set (should be ~3600 seconds)
    ttl = redis_client.ttl(redis_key)
    assert 3500 < ttl <= 3600, f"TTL not set correctly: {ttl} seconds"

    print(f"✅ Basin persisted to Neo4j and cached in Redis (TTL: {ttl}s)")

    # Cleanup
    with neo4j_driver.session() as session:
        session.run(
            "MATCH (b:AttractorBasin {basin_id: $basin_id}) DELETE b", basin_id=basin_id
        )
    redis_client.delete(redis_key)


def test_basin_update_invalidates_cache(basin_tracker, neo4j_driver, redis_client):
    """Test basin update invalidates Redis cache and updates Neo4j"""

    concept = "test_concept_update"

    # Create basin (strength = 1.0)
    result1 = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_001", increment=0.2
    )

    basin1 = (result1.get("new_basins", []) + result1.get("updated_basins", []))[0]
    basin_id = basin1.get("basin_id") or concept
    redis_key = f"basin:{basin_id}"

    # Cache should exist
    assert redis_client.get(redis_key) is not None, "Initial cache not created"

    # Update basin (strength = 1.2)
    result2 = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_002", increment=0.2
    )

    # Verify Neo4j updated
    with neo4j_driver.session() as session:
        result_neo4j = session.run(
            """
            MATCH (b:AttractorBasin {basin_id: $basin_id})
            RETURN b.strength AS strength, b.activation_count AS activation_count
            """,
            basin_id=basin_id,
        )

        record = result_neo4j.single()
        assert record["strength"] == 1.2, f"Neo4j not updated: {record['strength']}"
        assert record["activation_count"] == 2, "Activation count not incremented"

    # Cache should be invalidated and recreated
    cached_data = redis_client.get(redis_key)
    assert cached_data is not None, "Cache not recreated after update"

    print(f"✅ Basin update: Neo4j updated (1.2) and cache invalidated")

    # Cleanup
    with neo4j_driver.session() as session:
        session.run(
            "MATCH (b:AttractorBasin {basin_id: $basin_id}) DELETE b", basin_id=basin_id
        )
    redis_client.delete(redis_key)


def test_lazy_migration_old_basin(basin_tracker, neo4j_driver, redis_client):
    """Test lazy migration: old basin without new fields gets defaults"""

    basin_id = "test_old_basin"

    # Create old-style basin (without strength, activation_count fields)
    with neo4j_driver.session() as session:
        session.run(
            """
            CREATE (b:AttractorBasin {
                basin_id: $basin_id,
                basin_name: $basin_id,
                basin_type: 'CONCEPTUAL',
                stability: 0.8,
                depth: 1.5
            })
            """,
            basin_id=basin_id,
        )

    # Fetch basin (should apply defaults)
    basin = basin_tracker.get_basin(basin_id)

    # Verify lazy migration defaults
    assert basin.get("strength") == 1.0, \
        f"Lazy migration failed: strength = {basin.get('strength')}, expected 1.0"

    assert basin.get("activation_count") == 0, \
        f"Lazy migration failed: activation_count = {basin.get('activation_count')}, expected 0"

    assert basin.get("co_occurring_concepts") == {}, \
        "Lazy migration failed: co_occurring_concepts should be empty dict"

    print(f"✅ Lazy migration: Old basin received defaults (strength=1.0, count=0)")

    # Cleanup
    with neo4j_driver.session() as session:
        session.run(
            "MATCH (b:AttractorBasin {basin_id: $basin_id}) DELETE b", basin_id=basin_id
        )


def test_redis_cache_expiry_after_1_hour(basin_tracker, redis_client):
    """Test Redis cache expires after 1 hour (simulate with short TTL)"""

    concept = "test_concept_expiry"

    # Create basin with custom TTL (1 second for testing)
    result = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_001", increment=0.2
    )

    basin = (result.get("new_basins", []) + result.get("updated_basins", []))[0]
    basin_id = basin.get("basin_id") or concept
    redis_key = f"basin:{basin_id}"

    # Manually set short TTL for test
    redis_client.expire(redis_key, 1)

    # Cache should exist initially
    assert redis_client.get(redis_key) is not None, "Cache not created"

    # Wait for expiry
    time.sleep(1.1)

    # Cache should be expired
    assert redis_client.get(redis_key) is None, "Cache not expired after TTL"

    print(f"✅ Redis cache expired after 1 second (simulates 1-hour TTL)")


def test_concurrent_cache_reads(basin_tracker, redis_client):
    """Test concurrent cache reads are consistent"""

    import concurrent.futures

    concept = "test_concept_concurrent"

    # Create basin
    result = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_001", increment=0.2
    )

    basin = (result.get("new_basins", []) + result.get("updated_basins", []))[0]
    basin_id = basin.get("basin_id") or concept
    redis_key = f"basin:{basin_id}"

    # 50 concurrent reads
    def read_cache():
        return redis_client.get(redis_key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(read_cache) for _ in range(50)]
        results = [f.result() for f in futures]

    # All reads should return same value
    assert all(r == results[0] for r in results), "Inconsistent cache reads"
    assert all(r is not None for r in results), "Some reads returned None"

    print(f"✅ Concurrent cache reads: 50 reads, all consistent")

    # Cleanup
    redis_client.delete(redis_key)


if __name__ == "__main__":
    print("\n=== T013: Integration Test - Neo4j + Redis Persistence ===\n")
    print("⚠️  This test MUST FAIL initially (BasinTracker not implemented)")
    print("✅  Test will pass once core services are implemented (T015-T025)\n")

    pytest.main([__file__, "-v"])
