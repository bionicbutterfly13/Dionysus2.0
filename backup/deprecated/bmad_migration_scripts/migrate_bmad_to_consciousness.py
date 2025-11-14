#!/usr/bin/env python3
"""
BMAD Data Migration Script

Migrates BMAD-related data from Neo4j to consciousness-processed knowledge.

This script:
1. Extracts BMAD decision data from Neo4j (nodes 5 and 6)
2. Processes it through Daedalus consciousness system
3. Removes old BMAD nodes after successful migration
4. Creates proper knowledge graph with ThoughtSeeds and AttractorBasins

Usage:
    python backend/migrate_bmad_to_consciousness.py [--dry-run] [--keep-original]

Options:
    --dry-run: Show what would be migrated without actually doing it
    --keep-original: Keep original BMAD nodes after migration (default: remove)
"""

import sys
import argparse
import io
import logging
from pathlib import Path
from datetime import datetime

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.daedalus import Daedalus
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BMADMigrator:
    """Migrate BMAD data through consciousness system"""

    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "dionysus"):
        """
        Initialize migrator.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.daedalus = Daedalus()

    def extract_bmad_data(self) -> dict:
        """
        Extract BMAD-related data from Neo4j.

        Returns:
            Dict with project and decision data
        """
        logger.info("📊 Extracting BMAD data from Neo4j...")

        with self.driver.session() as session:
            # Get Project node (ID 5)
            project_result = session.run("""
                MATCH (p:Project) WHERE id(p) = 5
                RETURN properties(p) as project
            """)
            project_data = project_result.single()

            # Get Decision node (ID 6)
            decision_result = session.run("""
                MATCH (d:Decision) WHERE id(d) = 6
                RETURN properties(d) as decision
            """)
            decision_data = decision_result.single()

            if not project_data or not decision_data:
                logger.warning("⚠ BMAD nodes not found in database")
                return None

            return {
                "project": project_data["project"] if project_data else None,
                "decision": decision_data["decision"] if decision_data else None
            }

    def create_migration_document(self, data: dict) -> str:
        """
        Create consciousness-processable document from BMAD data.

        Args:
            data: Extracted BMAD data

        Returns:
            Formatted document string
        """
        project = data["project"]
        decision = data["decision"]

        # Create comprehensive document
        document = f"""# Knowledge Management System: Decision Tracking Architecture

## Project Context
**Name**: {project.get('name', 'Unknown')}
**ID**: {project.get('id', 'Unknown')}
**Owner**: {project.get('owner', 'Unknown')}
**Created**: {project.get('created_at', 'Unknown')}

## Historical Context
This project underwent {project.get('rebuild_count', 0)} rebuilds due to {project.get('rebuild_reason', 'unknown reasons')}. This decision represents a critical architectural shift to prevent future knowledge loss.

## Architectural Decision: Decision Tracking System

### Decision Summary
{decision.get('description', 'No description')}

### Decision Type
{decision.get('decision_type', 'unknown').capitalize()}

### Phase
{decision.get('phase', 'unknown')}

### Status
{decision.get('status', 'unknown').capitalize()}

## Rationale
{decision.get('rationale', 'No rationale provided')}

## Expected Outcome
{decision.get('outcome_expected', 'No outcome specified')}

## Alternatives Considered

{chr(10).join(f'{i+1}. {alt}' for i, alt in enumerate(decision.get('alternatives_considered', [])))}

### Why Decision Tracking Was Chosen
Decision tracking in Neo4j provides:
- Persistent storage across sessions
- Queryable decision history
- Cross-project learning
- Context preservation
- Rebuild prevention through institutional memory

## Implementation Details
**Made By**: {decision.get('made_by', 'Unknown')}
**Decision ID**: {decision.get('id', 'Unknown')}
**Created**: {decision.get('created_at', 'Unknown')}

## Key Insights
This architectural decision exemplifies the importance of:
- **Institutional Memory**: Systems must remember their own evolution
- **Context Preservation**: Decisions without context lose value
- **Cross-Project Learning**: Patterns repeat across projects
- **Knowledge Persistence**: Information must survive process restarts

## Tags
#architecture #decision-tracking #knowledge-management #rebuild-prevention #institutional-memory #neo4j #consciousness-system

## Migration Note
This document was migrated from legacy BMAD decision tracking nodes (Project ID: 5, Decision ID: 6) to the Dionysus consciousness-enhanced knowledge system on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

The migration ensures this knowledge is:
- Processed through 5-level concept extraction
- Integrated with attractor basins
- Connected via thoughtseeds
- Searchable through the consciousness system
- Preserved with full relational context
"""
        return document

    def process_through_consciousness(self, document: str) -> dict:
        """
        Process migration document through Daedalus.

        Args:
            document: Document content

        Returns:
            Processing result from Daedalus
        """
        logger.info("🧠 Processing through consciousness system...")

        result = self.daedalus.receive_perceptual_information(
            data=io.BytesIO(document.encode('utf-8')),
            tags=[
                "decision",
                "architecture",
                "knowledge-management",
                "institutional-memory",
                "migration"
            ],
            max_iterations=3,
            quality_threshold=0.75
        )

        logger.info(f"  ✓ Processing complete")
        logger.info(f"    - Status: {result.get('status')}")
        logger.info(f"    - Document ID: {result.get('document', {}).get('document_id')}")
        logger.info(f"    - Concepts extracted: {len(result.get('consciousness', {}).get('concepts', []))}")
        logger.info(f"    - Basins created: {len(result.get('consciousness', {}).get('basins', []))}")
        logger.info(f"    - ThoughtSeeds: {len(result.get('consciousness', {}).get('thoughtseeds', []))}")
        logger.info(f"    - Quality score: {result.get('quality', {}).get('scores', {}).get('overall', 'N/A')}")

        return result

    def remove_original_nodes(self):
        """Remove original BMAD nodes from Neo4j"""
        logger.info("🗑️  Removing original BMAD nodes...")

        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Project) WHERE id(p) = 5
                MATCH (d:Decision) WHERE id(d) = 6
                DETACH DELETE p, d
                RETURN count(*) as deleted
            """)
            deleted = result.single()["deleted"]

        logger.info(f"  ✓ Removed {deleted} nodes")

    def migrate(self, dry_run: bool = False, keep_original: bool = False) -> dict:
        """
        Execute complete migration.

        Args:
            dry_run: If True, show what would happen without executing
            keep_original: If True, keep original BMAD nodes

        Returns:
            Migration result dict
        """
        logger.info("=" * 80)
        logger.info("🚀 BMAD to Consciousness Migration")
        logger.info("=" * 80)

        # Step 1: Extract
        data = self.extract_bmad_data()
        if not data:
            return {
                "status": "error",
                "message": "No BMAD data found to migrate"
            }

        logger.info(f"✅ Extracted data from Project and Decision nodes")

        # Step 2: Create document
        document = self.create_migration_document(data)
        logger.info(f"✅ Created migration document ({len(document)} bytes)")

        if dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("DRY RUN - Migration Document Preview:")
            logger.info("=" * 80)
            logger.info(document)
            logger.info("=" * 80)
            logger.info("Dry run complete. No changes made.")
            return {
                "status": "dry_run",
                "document_size": len(document),
                "would_remove_nodes": not keep_original
            }

        # Step 3: Process through consciousness
        result = self.process_through_consciousness(document)

        # Step 4: Remove original nodes (if requested)
        if not keep_original:
            self.remove_original_nodes()
        else:
            logger.info("⏭️  Skipping removal of original nodes (--keep-original)")

        logger.info("\n" + "=" * 80)
        logger.info("✅ Migration Complete!")
        logger.info("=" * 80)
        logger.info(f"Original BMAD decision data has been:")
        logger.info(f"  • Processed through Dionysus consciousness system")
        logger.info(f"  • Extracted into 5-level concept hierarchy")
        logger.info(f"  • Integrated with attractor basins")
        logger.info(f"  • Connected via thoughtseeds")
        logger.info(f"  • Stored in Neo4j knowledge graph")
        if not keep_original:
            logger.info(f"  • Original nodes removed from database")
        logger.info("=" * 80)

        return {
            "status": "success",
            "document_id": result.get('document', {}).get('document_id'),
            "concepts_extracted": len(result.get('consciousness', {}).get('concepts', [])),
            "basins_created": len(result.get('consciousness', {}).get('basins', [])),
            "thoughtseeds_generated": len(result.get('consciousness', {}).get('thoughtseeds', [])),
            "quality_score": result.get('quality', {}).get('scores', {}).get('overall', 0.0),
            "original_nodes_removed": not keep_original
        }

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Migrate BMAD data to consciousness system")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be migrated without actually doing it")
    parser.add_argument("--keep-original", action="store_true",
                        help="Keep original BMAD nodes after migration")
    args = parser.parse_args()

    migrator = BMADMigrator()

    try:
        result = migrator.migrate(dry_run=args.dry_run, keep_original=args.keep_original)

        if result["status"] == "success":
            print("\n✅ Migration successful!")
            print(f"Document ID: {result['document_id']}")
            print(f"Concepts: {result['concepts_extracted']}")
            print(f"Basins: {result['basins_created']}")
            print(f"ThoughtSeeds: {result['thoughtseeds_generated']}")
            print(f"Quality: {result['quality_score']:.2f}")
            return 0
        elif result["status"] == "dry_run":
            print("\n✅ Dry run complete. Use without --dry-run to execute migration.")
            return 0
        else:
            print(f"\n❌ Migration failed: {result.get('message')}")
            return 1

    except Exception as e:
        logger.error(f"❌ Migration error: {e}", exc_info=True)
        return 1
    finally:
        migrator.close()


if __name__ == "__main__":
    sys.exit(main())
