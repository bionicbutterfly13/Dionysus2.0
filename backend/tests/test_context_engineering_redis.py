#!/usr/bin/env python3
"""
T002: Context Engineering - Validate Redis Persistence for Basin Cache

Tests Redis connection and caching capabilities required for basin state persistence.
"""

import pytest
import redis
import json
import time
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_path))


def get_redis_connection():
    """Get Redis connection with timeout"""
    try:
        r = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2
        )
        # Test connection
        r.ping()
        return r
    except (redis.ConnectionError, redis.TimeoutError) as e:
        pytest.skip(f"Redis not available: {e}")


def test_redis_connection():
    """Verify Redis is accessible at localhost:6379"""
    r = get_redis_connection()

    # Test basic operations
    test_key = "test:connection"
    r.set(test_key, "connected")
    value = r.get(test_key)

    assert value == "connected", "Redis SET/GET failed"
    r.delete(test_key)

    print("✅ Redis connection established at localhost:6379")


def test_basin_cache_with_ttl():
    """Test basin state caching with 1-hour TTL"""
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence

    r = get_redis_connection()

    # Create test basin
    basin = AttractorBasin(
        basin_name="cached_concept",
        basin_type=BasinType.CONCEPTUAL,
        stability=0.8,
        depth=1.5,
        activation_threshold=0.5,
        neural_field_influence=NeuralFieldInfluence(
            field_contribution=0.6,
            spatial_extent=1.0,
            temporal_persistence=0.7
        )
    )

    # Cache basin with 1-hour TTL (3600 seconds)
    cache_key = f"basin:{basin.basin_id}"
    basin_json = basin.model_dump_json()  # Pydantic V2

    # Set with TTL
    r.setex(cache_key, 3600, basin_json)

    # Verify stored
    cached_json = r.get(cache_key)
    assert cached_json is not None, "Basin not cached"

    # Verify can deserialize
    cached_dict = json.loads(cached_json)
    restored_basin = AttractorBasin(**cached_dict)

    assert restored_basin.basin_id == basin.basin_id
    assert restored_basin.basin_name == basin.basin_name

    # Verify TTL was set
    ttl = r.ttl(cache_key)
    assert 3500 < ttl <= 3600, f"TTL not set correctly: {ttl}"

    # Cleanup
    r.delete(cache_key)

    print(f"✅ Basin caching with 1-hour TTL working (TTL: {ttl}s)")


def test_cache_invalidation():
    """Verify cache invalidation works"""
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence

    r = get_redis_connection()

    basin = AttractorBasin(
        basin_name="invalidation_test",
        basin_type=BasinType.SEMANTIC,
        stability=0.7,
        depth=1.2,
        activation_threshold=0.4,
        neural_field_influence=NeuralFieldInfluence(
            field_contribution=0.5,
            spatial_extent=0.9,
            temporal_persistence=0.6
        )
    )

    cache_key = f"basin:{basin.basin_id}"

    # Cache basin
    r.setex(cache_key, 3600, basin.model_dump_json())
    assert r.exists(cache_key) == 1, "Basin not cached"

    # Invalidate (delete)
    r.delete(cache_key)
    assert r.exists(cache_key) == 0, "Cache invalidation failed"

    print("✅ Cache invalidation working")


def test_concurrent_basin_reads():
    """Test 100+ concurrent basin lookups (performance requirement)"""
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence

    r = get_redis_connection()

    # Create test basins
    basin_ids = []
    for i in range(10):  # Cache 10 basins
        basin = AttractorBasin(
            basin_name=f"concurrent_test_{i}",
            basin_type=BasinType.CONCEPTUAL,
            stability=0.8,
            depth=1.5,
            activation_threshold=0.5,
            neural_field_influence=NeuralFieldInfluence(
                field_contribution=0.6,
                spatial_extent=1.0,
                temporal_persistence=0.7
            )
        )
        cache_key = f"basin:{basin.basin_id}"
        r.setex(cache_key, 3600, basin.model_dump_json())
        basin_ids.append(basin.basin_id)

    # Concurrent read test
    def read_basin(basin_id):
        """Read basin from cache"""
        cache_key = f"basin:{basin_id}"
        cached_json = r.get(cache_key)
        if cached_json:
            basin_dict = json.loads(cached_json)
            return AttractorBasin(**basin_dict)
        return None

    # Execute 120 concurrent reads (10 basins × 12 reads each)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for _ in range(12):  # 12 reads per basin
            for basin_id in basin_ids:
                future = executor.submit(read_basin, basin_id)
                futures.append(future)

        # Wait for all reads
        results = [f.result() for f in futures]

    elapsed_ms = (time.time() - start_time) * 1000

    # Verify all reads succeeded
    assert len(results) == 120, f"Expected 120 results, got {len(results)}"
    assert all(r is not None for r in results), "Some reads failed"

    # Cleanup
    for basin_id in basin_ids:
        r.delete(f"basin:{basin_id}")

    print(f"✅ 120 concurrent reads completed in {elapsed_ms:.2f}ms ({elapsed_ms/120:.2f}ms per read)")


def test_batch_cache_loading():
    """Test batch cache loading on startup (for T027)"""
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence

    r = get_redis_connection()

    # Simulate startup: load multiple basins to cache
    basins = []
    for i in range(50):
        basin = AttractorBasin(
            basin_name=f"batch_basin_{i}",
            basin_type=BasinType.CONCEPTUAL,
            stability=0.8,
            depth=1.5,
            activation_threshold=0.5,
            neural_field_influence=NeuralFieldInfluence(
                field_contribution=0.6,
                spatial_extent=1.0,
                temporal_persistence=0.7
            )
        )
        basins.append(basin)

    # Batch cache with pipeline (performance optimization)
    start_time = time.time()

    pipe = r.pipeline()
    for basin in basins:
        cache_key = f"basin:{basin.basin_id}"
        pipe.setex(cache_key, 3600, basin.model_dump_json())

    pipe.execute()

    elapsed_ms = (time.time() - start_time) * 1000

    # Verify all cached
    for basin in basins:
        assert r.exists(f"basin:{basin.basin_id}") == 1

    # Cleanup
    for basin in basins:
        r.delete(f"basin:{basin.basin_id}")

    print(f"✅ Batch cache loading: 50 basins in {elapsed_ms:.2f}ms ({elapsed_ms/50:.2f}ms per basin)")


def test_redis_memory_efficiency():
    """Verify Redis memory usage is reasonable for basin caching"""
    r = get_redis_connection()

    # Get memory usage before
    info_before = r.info('memory')
    mem_before = info_before['used_memory']

    # Cache 1000 basins
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence

    basin_ids = []
    for i in range(1000):
        basin = AttractorBasin(
            basin_name=f"memory_test_{i}",
            basin_type=BasinType.CONCEPTUAL,
            stability=0.8,
            depth=1.5,
            activation_threshold=0.5,
            neural_field_influence=NeuralFieldInfluence(
                field_contribution=0.6,
                spatial_extent=1.0,
                temporal_persistence=0.7
            )
        )
        cache_key = f"basin:{basin.basin_id}"
        r.setex(cache_key, 3600, basin.model_dump_json())
        basin_ids.append(basin.basin_id)

    # Get memory usage after
    info_after = r.info('memory')
    mem_after = info_after['used_memory']

    mem_used_mb = (mem_after - mem_before) / (1024 * 1024)

    # Cleanup
    for basin_id in basin_ids:
        r.delete(f"basin:{basin_id}")

    # Memory should be reasonable (< 50MB for 1000 basins)
    assert mem_used_mb < 50, f"Memory usage too high: {mem_used_mb:.2f}MB"

    print(f"✅ Memory efficient: 1000 basins used {mem_used_mb:.2f}MB (~{mem_used_mb/1000*1024:.1f}KB per basin)")


if __name__ == "__main__":
    print("\n=== T002: Context Engineering Redis Validation ===\n")

    test_redis_connection()
    test_basin_cache_with_ttl()
    test_cache_invalidation()
    test_concurrent_basin_reads()
    test_batch_cache_loading()
    test_redis_memory_efficiency()

    print("\n✅ T002 PASSED: Redis persistence ready for basin caching")
