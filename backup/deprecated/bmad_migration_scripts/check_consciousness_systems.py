#!/usr/bin/env python3
"""
Consciousness Systems Status Check

Run this at session startup to check:
1. Skills database initialization status
2. BMAD migration status
3. Dionysus consciousness system health

Usage:
    python backend/check_consciousness_systems.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.skills_manager import SkillsManager
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.WARNING)  # Quiet output


def check_skills_database() -> dict:
    """Check skills database status"""
    try:
        manager = SkillsManager()
        status = manager.get_skill_status()
        return status
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def check_bmad_migration() -> dict:
    """Check if BMAD nodes still exist in Neo4j"""
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "dionysus")
        )

        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE id(n) IN [5, 6]
                RETURN count(n) as bmad_nodes_remaining
            """)
            count = result.single()["bmad_nodes_remaining"]

        driver.close()

        if count > 0:
            return {
                "status": "not_migrated",
                "bmad_nodes_remaining": count,
                "action_required": "Run: python backend/migrate_bmad_to_consciousness.py"
            }
        else:
            return {
                "status": "migrated",
                "bmad_nodes_remaining": 0,
                "message": "BMAD data successfully migrated to consciousness system"
            }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def check_neo4j_health() -> dict:
    """Check Neo4j connection and schema"""
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "dionysus")
        )

        with driver.session() as session:
            # Get node counts
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """)
            node_counts = {record["label"]: record["count"] for record in result}

            # Check if consciousness nodes exist
            consciousness_nodes = {
                "Document": node_counts.get("Document", 0),
                "Concept": node_counts.get("Concept", 0),
                "AttractorBasin": node_counts.get("AttractorBasin", 0),
                "ThoughtSeed": node_counts.get("ThoughtSeed", 0)
            }

        driver.close()

        return {
            "status": "healthy",
            "total_nodes": sum(node_counts.values()),
            "consciousness_nodes": consciousness_nodes,
            "all_node_types": list(node_counts.keys())
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def print_status_report():
    """Print comprehensive status report"""
    print("\n" + "=" * 80)
    print("🧠 Dionysus Consciousness Systems Status")
    print("=" * 80)

    # Check Skills Database
    print("\n📚 Skills Database:")
    skills_status = check_skills_database()

    if skills_status["status"] == "initialized":
        print(f"  ✅ Initialized with {skills_status['skills_count']} skills")
        print(f"     Categories: {', '.join(skills_status['categories'])}")
        print(f"     Success rate: {skills_status['stats']['successful']}/{skills_status['skills_count']}")
        print(f"     Average quality: {skills_status['stats']['average_quality']:.2f}")
    elif skills_status["status"] == "not_initialized":
        print(f"  ⚠️  Not initialized")
        print(f"     Action: Run skills initialization")
        print(f"     Command: python backend/initialize_skills.py")
    else:
        print(f"  ❌ Error: {skills_status.get('error', 'Unknown')}")

    # Check BMAD Migration
    print("\n🔄 BMAD Migration:")
    bmad_status = check_bmad_migration()

    if bmad_status["status"] == "migrated":
        print(f"  ✅ {bmad_status['message']}")
    elif bmad_status["status"] == "not_migrated":
        print(f"  ⚠️  {bmad_status['bmad_nodes_remaining']} BMAD nodes remaining")
        print(f"     Action: {bmad_status['action_required']}")
    else:
        print(f"  ❌ Error: {bmad_status.get('error', 'Unknown')}")

    # Check Neo4j Health
    print("\n🗄️  Neo4j Knowledge Graph:")
    neo4j_status = check_neo4j_health()

    if neo4j_status["status"] == "healthy":
        print(f"  ✅ Healthy ({neo4j_status['total_nodes']} total nodes)")
        print(f"     Consciousness nodes:")
        for node_type, count in neo4j_status['consciousness_nodes'].items():
            print(f"       • {node_type}: {count}")
    else:
        print(f"  ❌ Error: {neo4j_status.get('error', 'Unknown')}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_status_report()
