#!/usr/bin/env python3
"""
Neo4j Schema Initialization - Spec 054 T019-T020 + Spec 055 Agent 1

Creates uniqueness constraints and performance indexes for document persistence.

SPEC 055 AGENT 1 ENHANCEMENTS:
- UNIQUE constraint on Document.content_hash (deduplication)
- Index on content_hash for fast lookup

CONSTITUTIONAL COMPLIANCE (Spec 040):
- All operations via DaedalusGraphChannel
- NO direct neo4j imports

Author: Spec 054 + Spec 055 Agent 1 Implementation
Created: 2025-10-07
"""

import logging
from typing import Dict, List
import asyncio

# Constitutional compliance: Only Graph Channel import allowed
from daedalus_gateway import get_graph_channel

logger = logging.getLogger(__name__)


class Neo4jSchemaInitializer:
    """
    Initializes Neo4j schema for document persistence.

    Creates:
    - Uniqueness constraints (document_id, content_hash, concept_id, basin_id, seed_id)
    - Performance indexes (upload_timestamp, quality, tier, tags, level, salience)
    """

    def __init__(self):
        """Initialize with Graph Channel."""
        self.graph_channel = get_graph_channel()
        logger.info("Neo4jSchemaInitializer initialized with Graph Channel")

    async def initialize_schema(self) -> Dict[str, any]:
        """
        Create all constraints and indexes.

        Returns:
            Status dict with created constraints and indexes
        """
        logger.info("Initializing Neo4j schema for Spec 054...")

        constraints_created = await self._create_constraints()
        indexes_created = await self._create_indexes()

        result = {
            "status": "success",
            "constraints_created": constraints_created,
            "indexes_created": indexes_created
        }

        logger.info(f"Schema initialization complete: {result}")
        return result

    async def _create_constraints(self) -> List[str]:
        """
        Create uniqueness constraints.

        From plan.md lines 340-355.
        """
        constraints = [
            # Document constraints
            {
                "name": "document_id_unique",
                "query": """
                CREATE CONSTRAINT document_id_unique IF NOT EXISTS
                FOR (d:Document) REQUIRE d.document_id IS UNIQUE
                """
            },
            {
                "name": "content_hash_unique",
                "query": """
                CREATE CONSTRAINT content_hash_unique IF NOT EXISTS
                FOR (d:Document) REQUIRE d.content_hash IS UNIQUE
                """
            },
            # Concept constraint
            {
                "name": "concept_id_unique",
                "query": """
                CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
                FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE
                """
            },
            # AttractorBasin constraint
            {
                "name": "basin_id_unique",
                "query": """
                CREATE CONSTRAINT basin_id_unique IF NOT EXISTS
                FOR (b:AttractorBasin) REQUIRE b.basin_id IS UNIQUE
                """
            },
            # ThoughtSeed constraint
            {
                "name": "seed_id_unique",
                "query": """
                CREATE CONSTRAINT seed_id_unique IF NOT EXISTS
                FOR (t:ThoughtSeed) REQUIRE t.seed_id IS UNIQUE
                """
            },
            # Chunk constraint (Spec 056)
            {
                "name": "chunk_id_unique",
                "query": """
                CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
                """
            }
        ]

        created = []
        for constraint in constraints:
            try:
                await self.graph_channel.execute_write(
                    query=constraint["query"],
                    parameters={},
                    caller_service="neo4j_schema_init",
                    caller_function="_create_constraints"
                )
                created.append(constraint["name"])
                logger.info(f"✅ Created constraint: {constraint['name']}")
            except Exception as e:
                # Constraint may already exist
                logger.warning(f"Constraint {constraint['name']} may already exist: {e}")

        return created

    async def _create_indexes(self) -> List[str]:
        """
        Create performance indexes for filtering and sorting.

        From plan.md lines 357-375.
        """
        indexes = [
            # Document indexes for listing API
            {
                "name": "document_upload_timestamp",
                "query": """
                CREATE INDEX document_upload_timestamp IF NOT EXISTS
                FOR (d:Document) ON (d.upload_timestamp)
                """
            },
            {
                "name": "document_quality",
                "query": """
                CREATE INDEX document_quality IF NOT EXISTS
                FOR (d:Document) ON (d.quality_overall)
                """
            },
            {
                "name": "document_tier",
                "query": """
                CREATE INDEX document_tier IF NOT EXISTS
                FOR (d:Document) ON (d.tier)
                """
            },
            {
                "name": "document_tags",
                "query": """
                CREATE INDEX document_tags IF NOT EXISTS
                FOR (d:Document) ON (d.tags)
                """
            },
            # Concept indexes for querying
            {
                "name": "concept_level",
                "query": """
                CREATE INDEX concept_level IF NOT EXISTS
                FOR (c:Concept) ON (c.level)
                """
            },
            {
                "name": "concept_salience",
                "query": """
                CREATE INDEX concept_salience IF NOT EXISTS
                FOR (c:Concept) ON (c.salience)
                """
            },
            # SPEC 055 AGENT 1: Index on content_hash for fast duplicate lookup
            {
                "name": "document_content_hash",
                "query": """
                CREATE INDEX document_content_hash IF NOT EXISTS
                FOR (d:Document) ON (d.content_hash)
                """
            },
            # SPEC 056: Chunk indexes
            {
                "name": "chunk_position",
                "query": """
                CREATE INDEX chunk_position IF NOT EXISTS
                FOR (c:Chunk) ON (c.position)
                """
            },
            # SPEC 057: Document source_url index for filtering
            {
                "name": "document_source_url",
                "query": """
                CREATE INDEX document_source_url IF NOT EXISTS
                FOR (d:Document) ON (d.original_url)
                """
            }
        ]

        created = []
        for index in indexes:
            try:
                await self.graph_channel.execute_write(
                    query=index["query"],
                    parameters={},
                    caller_service="neo4j_schema_init",
                    caller_function="_create_indexes"
                )
                created.append(index["name"])
                logger.info(f"✅ Created index: {index['name']}")
            except Exception as e:
                # Index may already exist
                logger.warning(f"Index {index['name']} may already exist: {e}")

        return created

    async def verify_schema(self) -> Dict[str, bool]:
        """
        Verify schema is correctly initialized.

        Returns:
            Dict with verification results
        """
        logger.info("Verifying Neo4j schema...")

        # Check constraints exist
        constraints_query = "SHOW CONSTRAINTS"
        try:
            constraints_result = await self.graph_channel.execute_read(
                query=constraints_query,
                parameters={},
                caller_service="neo4j_schema_init",
                caller_function="verify_schema"
            )

            constraint_names = [rec.get("name", "") for rec in constraints_result.get("records", [])]

            # Check indexes exist
            indexes_query = "SHOW INDEXES"
            indexes_result = await self.graph_channel.execute_read(
                query=indexes_query,
                parameters={},
                caller_service="neo4j_schema_init",
                caller_function="verify_schema"
            )

            index_names = [rec.get("name", "") for rec in indexes_result.get("records", [])]

            verification = {
                "constraints_exist": len(constraint_names) > 0,
                "indexes_exist": len(index_names) > 0,
                "constraint_count": len(constraint_names),
                "index_count": len(index_names)
            }

            logger.info(f"Schema verification: {verification}")
            return verification

        except Exception as e:
            logger.error(f"Schema verification failed: {e}")
            return {
                "constraints_exist": False,
                "indexes_exist": False,
                "error": str(e)
            }


async def main():
    """
    CLI entry point for schema initialization.

    Usage:
        python -m backend.src.services.neo4j_schema_init
    """
    initializer = Neo4jSchemaInitializer()

    print("🔧 Initializing Neo4j schema for Spec 054...")
    result = await initializer.initialize_schema()

    print(f"\n✅ Schema initialization complete!")
    print(f"   - Constraints created: {len(result['constraints_created'])}")
    print(f"   - Indexes created: {len(result['indexes_created'])}")

    print("\n🔍 Verifying schema...")
    verification = await initializer.verify_schema()
    print(f"   - Constraints exist: {verification['constraints_exist']}")
    print(f"   - Indexes exist: {verification['indexes_exist']}")

    if verification['constraints_exist'] and verification['indexes_exist']:
        print("\n🎉 Schema ready for document persistence!")
    else:
        print("\n⚠️  Schema verification incomplete. Check logs.")


if __name__ == "__main__":
    asyncio.run(main())
