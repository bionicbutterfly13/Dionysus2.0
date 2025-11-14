#!/usr/bin/env python3
"""
Consciousness Systems Startup Check

Checks the health of consciousness-related systems:
- Neo4j graph database (required for consciousness processing)
- Redis (required for neuronal packets and attractor basins)
- Skills database initialization status
"""
import sys
import asyncio
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from daedalus_gateway import get_graph_channel
    GRAPH_CHANNEL_AVAILABLE = True
except ImportError:
    GRAPH_CHANNEL_AVAILABLE = False


async def check_neo4j():
    """Check Neo4j connection and provide helpful guidance."""
    print("🔍 Checking Neo4j...")

    if not GRAPH_CHANNEL_AVAILABLE:
        print("  ⚠️  daedalus_gateway not available (Graph Channel)")
        print("  💡 Install with: pip install daedalus-gateway")
        return False

    try:
        channel = get_graph_channel()

        # Test connection using health check
        health = await channel.health_check()

        if health.get("connected"):
            print(f"  ✅ Neo4j connected (bolt://localhost:7687)")

            # Get total node count
            try:
                result = await channel.execute_read(
                    "MATCH (n) RETURN count(n) as node_count",
                    caller_service="check_script",
                    caller_function="check_neo4j"
                )

                records = result.get("records", [])
                if records:
                    node_count = records[0].get("node_count", 0)
                    print(f"  📊 Total nodes: {node_count}")

                    # Get consciousness-specific node counts
                    consciousness_query = """
                    MATCH (n)
                    WHERE n:ThoughtSeed OR n:AttractorBasin OR n:Concept
                    RETURN labels(n)[0] as type, count(*) as count
                    ORDER BY type
                    """
                    consciousness_result = await channel.execute_read(
                        consciousness_query,
                        caller_service="check_script",
                        caller_function="check_neo4j"
                    )

                    consciousness_records = consciousness_result.get("records", [])
                    if consciousness_records:
                        print("  🧠 Consciousness nodes:")
                        for record in consciousness_records:
                            print(f"     - {record['type']}: {record['count']}")
                    else:
                        print("  ℹ️  No consciousness nodes found yet")

            except Exception as e:
                print(f"  ⚠️  Could not query node counts: {e}")

            return True
        else:
            print("  ❌ Neo4j not connected")
            print()
            print("  💡 To start Neo4j:")
            print("     brew services start neo4j")
            return False

    except Exception as e:
        print(f"  ❌ Neo4j not responding: {e}")
        print()
        print("  💡 To start Neo4j:")
        print("     brew services start neo4j")
        print()
        print("  Or use Neo4j Aura (cloud):")
        print("     https://neo4j.com/cloud/aura/")
        print("     Then set environment variables:")
        print("       export NEO4J_URI=bolt://localhost:7687")
        print("       export NEO4J_USER=neo4j")
        print("       export NEO4J_PASSWORD=dionysus")
        return False


def check_redis():
    """Check Redis connection."""
    print("\n🔍 Checking Redis...")

    if not REDIS_AVAILABLE:
        print("  ⚠️  redis-py not available")
        print("  💡 Install with: pip install redis")
        return False

    try:
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()
        print("  ✅ Redis connected (localhost:6379)")
        return True

    except Exception as e:
        print(f"  ❌ Redis not responding: {e}")
        print()
        print("  💡 To start Redis:")
        print("     brew services start redis")
        return False


def check_skills_database():
    """Check if skills database has been initialized."""
    print("\n🔍 Checking Skills Database...")

    # Check if skills index exists
    skills_index = Path.home() / ".claude" / "skills" / "index.json"

    if skills_index.exists():
        import json
        try:
            with open(skills_index) as f:
                data = json.load(f)
                skill_count = len(data.get("skills", []))
                print(f"  ✅ Skills database initialized ({skill_count} skills indexed)")
                print(f"  📁 Index: {skills_index}")
        except Exception as e:
            print(f"  ⚠️  Skills index exists but couldn't read: {e}")
    else:
        print("  ℹ️  Skills database not initialized")
        print()
        print("  💡 To initialize:")
        print("     python backend/initialize_skills.py")
        print()
        print("  📖 Skills library: /Volumes/Asylum/skills-library/")


async def main():
    """Run all consciousness systems checks."""
    print("=" * 60)
    print("Dionysus Consciousness Systems Check")
    print("=" * 60)

    neo4j_ok = await check_neo4j()
    redis_ok = check_redis()
    check_skills_database()

    print("\n" + "=" * 60)

    if neo4j_ok and redis_ok:
        print("✅ All core systems operational")
        return 0
    else:
        print("⚠️  Some systems need attention (see above)")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
