#!/usr/bin/env python3
"""
Skills Database Initialization Script

Initializes the skills database by processing all skills through
the Dionysus consciousness system.

Usage:
    python backend/initialize_skills.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.skills_manager import SkillsManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """Initialize skills database"""
    print("\n" + "=" * 80)
    print("🚀 Dionysus Skills Database Initialization")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Scan /Volumes/Asylum/skills-library/")
    print("  2. Process each skill through Daedalus consciousness system")
    print("  3. Create ThoughtSeeds and AttractorBasins for skill relationships")
    print("  4. Store in Neo4j knowledge graph")
    print("  5. Create Claude Code skill index\n")

    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return 0

    try:
        manager = SkillsManager()
        result = manager.initialize_skills_database()

        print("\n" + "=" * 80)
        print("✅ Skills Database Initialized Successfully!")
        print("=" * 80)
        print(f"Skills discovered: {result['skills_discovered']}")
        print(f"Skills processed: {result['skills_processed']}")
        print(f"Categories scanned: {result['categories_scanned']}")
        print(f"Index created: {result['index_path']}")
        print("=" * 80)

        # Show index stats
        status = manager.get_skill_status()
        print(f"\nStats:")
        print(f"  • Successful: {status['stats']['successful']}")
        print(f"  • Failed: {status['stats']['failed']}")
        print(f"  • Total concepts extracted: {status['stats']['total_concepts']}")
        print(f"  • Average quality score: {status['stats']['average_quality']:.2f}")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
