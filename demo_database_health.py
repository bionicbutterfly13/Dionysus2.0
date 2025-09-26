#!/usr/bin/env python
"""
Demo script to show the database health checking system.
This shows the status of all database connections.
"""
import sys
sys.path.append('backend/src')

from services.database_health import DatabaseHealthService
import json


def demo_database_health_checks():
    """Demonstrate the database health checking system."""
    print("🏥 Testing Flux Database Health System")
    print("=" * 50)

    health_service = DatabaseHealthService()

    # Test individual database health checks
    print("\n🗄️ Individual Database Health Checks:")
    print("-" * 40)

    # Neo4j health
    print("\n📊 Neo4j Health Check:")
    neo4j_health = health_service.check_neo4j_health()
    print(f"  Status: {neo4j_health['status']}")
    print(f"  Message: {neo4j_health['message']}")
    print(f"  Response Time: {neo4j_health.get('response_time_ms', 'N/A')} ms")

    # Redis health
    print("\n⚡ Redis Health Check:")
    redis_health = health_service.check_redis_health()
    print(f"  Status: {redis_health['status']}")
    print(f"  Message: {redis_health['message']}")
    print(f"  Response Time: {redis_health.get('response_time_ms', 'N/A')} ms")

    # Qdrant health
    print("\n🔍 Qdrant Health Check:")
    qdrant_health = health_service.check_qdrant_health()
    print(f"  Status: {qdrant_health['status']}")
    print(f"  Message: {qdrant_health['message']}")
    print(f"  Response Time: {qdrant_health.get('response_time_ms', 'N/A')} ms")

    # Comprehensive health check
    print("\n🌐 Comprehensive Health Check:")
    print("-" * 40)
    all_health = health_service.check_all_databases()

    print(f"Overall Status: {all_health['overall_status']}")
    print(f"Healthy Databases: {all_health['healthy_count']}/{all_health['total_count']}")

    # Show status summary
    print("\n📋 Database Status Summary:")
    for db_name in ['neo4j', 'redis', 'qdrant']:
        status = all_health[db_name]['status']
        icon = "✅" if status == "healthy" else "❌" if status == "unhealthy" else "⚠️"
        print(f"  {icon} {db_name.capitalize()}: {status}")

    # Show JSON response (as API would return)
    print("\n📤 API Response Format:")
    print("-" * 40)
    print(json.dumps(all_health, indent=2))

    print("\n" + "=" * 50)
    print("✅ Database Health System Demo Complete!")
    print("The system:")
    print("  - Tests connections to Neo4j, Redis, and Qdrant")
    print("  - Measures response times for performance monitoring")
    print("  - Handles connection failures gracefully")
    print("  - Provides detailed status for each database")
    print("  - Returns comprehensive health overview")


if __name__ == "__main__":
    demo_database_health_checks()