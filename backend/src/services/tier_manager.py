#!/usr/bin/env python3
"""
Tier Manager Service - Spec 054

Manages document tier transitions (warm → cool → cold) based on
hybrid age + access pattern rules.

CONSTITUTIONAL COMPLIANCE (Spec 040):
- All Neo4j access via DaedalusGraphChannel
- NO direct neo4j imports allowed

Author: Spec 054 Implementation
Created: 2025-10-07
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path

# Constitutional compliance: Only Graph Channel import allowed
from daedalus_gateway import get_graph_channel

logger = logging.getLogger(__name__)

# Archive configuration
ARCHIVE_DIR = os.getenv("ARCHIVE_DIR", "/tmp/dionysus_archive")


class TierManager:
    """
    Manages document storage tiers based on hybrid age + access patterns.

    Tier Rules (from clarifications):
    - Warm → Cool: age >= 30 days AND access_count <= 5 AND days_since_access >= 14
    - Cool → Cold: age >= 90 days AND access_count <= 2 AND days_since_access >= 60
    """

    def __init__(self):
        """Initialize tier manager with Graph Channel."""
        self.graph_channel = get_graph_channel()
        logger.info("TierManager initialized with Graph Channel")

    async def evaluate_tier_migrations(self) -> Dict[str, int]:
        """
        T035-T036: Evaluate and execute tier migrations for all documents.

        Hybrid age + access rules from clarifications:
        - Warm → Cool: age >= 30 days AND access_count <= 5 AND days_since_access >= 14
        - Cool → Cold: age >= 90 days AND access_count <= 2 AND days_since_access >= 60

        Returns:
            Migration counts by transition type
        """
        now = datetime.utcnow()
        migrations = {
            "warm_to_cool": 0,
            "cool_to_cold": 0
        }

        # T035: Find documents eligible for Warm → Cool
        warm_to_cool_query = """
        MATCH (d:Document {tier: "warm"})
        WHERE duration.between(d.upload_timestamp, datetime()).days >= 30
          AND d.access_count <= 5
          AND duration.between(d.last_accessed, datetime()).days >= 14
        RETURN d.document_id as document_id
        """

        warm_result = await self.graph_channel.execute_read(
            query=warm_to_cool_query,
            parameters={},
            caller_service="tier_manager",
            caller_function="evaluate_tier_migrations"
        )

        # Migrate warm → cool
        for record in warm_result.get("records", []):
            await self.update_tier(
                document_id=record["document_id"],
                new_tier="cool",
                reason="automated_age_access"
            )
            migrations["warm_to_cool"] += 1

        # T036: Find documents eligible for Cool → Cold
        cool_to_cold_query = """
        MATCH (d:Document {tier: "cool"})
        WHERE duration.between(d.upload_timestamp, datetime()).days >= 90
          AND d.access_count <= 2
          AND duration.between(d.last_accessed, datetime()).days >= 60
        RETURN d.document_id as document_id
        """

        cool_result = await self.graph_channel.execute_read(
            query=cool_to_cold_query,
            parameters={},
            caller_service="tier_manager",
            caller_function="evaluate_tier_migrations"
        )

        # Migrate cool → cold (with archival)
        for record in cool_result.get("records", []):
            await self.update_tier(
                document_id=record["document_id"],
                new_tier="cold",
                reason="automated_age_access"
            )
            migrations["cool_to_cold"] += 1

        logger.info(f"✅ Tier migrations complete: {migrations}")
        return migrations

    async def update_tier(
        self,
        document_id: str,
        new_tier: str,
        reason: str = "manual"
    ) -> Dict[str, Any]:
        """
        T037: Update document tier.

        Args:
            document_id: Document ID
            new_tier: New tier (warm, cool, cold)
            reason: Reason for tier change

        Returns:
            Tier update result
        """
        # Validate tier
        valid_tiers = ["warm", "cool", "cold"]
        if new_tier not in valid_tiers:
            raise ValueError(f"Invalid tier: {new_tier}. Must be one of {valid_tiers}")

        # Get current tier
        get_query = """
        MATCH (d:Document {document_id: $document_id})
        RETURN d.tier as old_tier
        """

        get_result = await self.graph_channel.execute_read(
            query=get_query,
            parameters={"document_id": document_id},
            caller_service="tier_manager",
            caller_function="update_tier"
        )

        if not get_result.get("records"):
            raise ValueError(f"Document not found: {document_id}")

        old_tier = get_result["records"][0]["old_tier"]

        # Update tier
        update_query = """
        MATCH (d:Document {document_id: $document_id})
        SET d.tier = $new_tier,
            d.tier_changed_at = datetime()
        RETURN d.tier as tier
        """

        await self.graph_channel.execute_write(
            query=update_query,
            parameters={
                "document_id": document_id,
                "new_tier": new_tier
            },
            caller_service="tier_manager",
            caller_function="update_tier"
        )

        # If moving to cold tier, archive the document (T038-T040)
        archive_location = None
        if new_tier == "cold":
            archive_result = await self.archive_to_cold_tier(document_id)
            archive_location = archive_result.get("archive_location")

        result = {
            "status": "success",
            "document_id": document_id,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "tier_changed_at": datetime.utcnow().isoformat(),
            "reason": reason
        }

        if archive_location:
            result["archive_location"] = archive_location

        logger.info(f"✅ Updated tier for {document_id}: {old_tier} → {new_tier} ({reason})")
        return result

    async def archive_to_cold_tier(self, document_id: str) -> Dict[str, Any]:
        """
        T038-T040: Archive document to cold tier storage (filesystem with S3 fallback).

        Args:
            document_id: Document ID

        Returns:
            Archival result with archive_location
        """
        # T038: Retrieve full document data
        query = """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (c:Concept)-[:EXTRACTED_FROM]->(d)
        OPTIONAL MATCH (b:AttractorBasin)-[:ATTRACTED_TO]->(d)
        OPTIONAL MATCH (t:ThoughtSeed)-[:GERMINATED_FROM]->(d)
        RETURN d, collect(DISTINCT c) as concepts, collect(DISTINCT b) as basins, collect(DISTINCT t) as thoughtseeds
        """

        result = await self.graph_channel.execute_read(
            query=query,
            parameters={"document_id": document_id},
            caller_service="tier_manager",
            caller_function="archive_to_cold_tier"
        )

        if not result.get("records"):
            raise ValueError(f"Document not found for archival: {document_id}")

        record = result["records"][0]
        doc_data = {
            "document": dict(record["d"]),
            "concepts": [dict(c) for c in record["concepts"] if c],
            "basins": [dict(b) for b in record["basins"] if b],
            "thoughtseeds": [dict(t) for t in record["thoughtseeds"] if t],
            "archived_at": datetime.utcnow().isoformat()
        }

        # T039: Write to archive
        archive_location = await self._write_to_archive(document_id, doc_data)

        # T040: Update document with archive location
        update_query = """
        MATCH (d:Document {document_id: $document_id})
        SET d.archive_location = $archive_location,
            d.archived_at = datetime()
        """

        await self.graph_channel.execute_write(
            query=update_query,
            parameters={
                "document_id": document_id,
                "archive_location": archive_location
            },
            caller_service="tier_manager",
            caller_function="archive_to_cold_tier"
        )

        logger.info(f"✅ Archived document {document_id} to {archive_location}")

        return {
            "status": "archived",
            "document_id": document_id,
            "archive_location": archive_location,
            "archived_at": datetime.utcnow().isoformat()
        }

    async def _write_to_archive(
        self,
        document_id: str,
        doc_data: Dict[str, Any]
    ) -> str:
        """
        T039: Write document data to cold tier storage (filesystem with S3 option).

        Args:
            document_id: Document ID
            doc_data: Complete document data

        Returns:
            archive_location (s3://bucket/key or /archive/path)
        """
        # Check for S3 configuration (future enhancement)
        use_s3 = os.getenv("USE_S3_ARCHIVE", "false").lower() == "true"

        if use_s3:
            # Future: S3 implementation with boto3
            # For now, fall back to filesystem
            logger.warning("S3 archival not yet implemented, using filesystem")

        # Filesystem archival (default)
        archive_dir = Path(ARCHIVE_DIR)
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Organize by date for easier management
        date_path = datetime.utcnow().strftime("%Y/%m/%d")
        doc_dir = archive_dir / date_path
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON file
        archive_file = doc_dir / f"{document_id}.json"
        with open(archive_file, "w") as f:
            json.dump(doc_data, f, indent=2, default=str)

        archive_location = str(archive_file)
        logger.debug(f"✅ Wrote archive to {archive_location}")

        return archive_location
