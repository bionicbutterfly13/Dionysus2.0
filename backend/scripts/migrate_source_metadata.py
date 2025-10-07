#!/usr/bin/env python3
"""
Source Metadata Migration Script - Spec 057

Backfills existing documents with source metadata:
- source_type = "uploaded_file" (default for existing docs)
- original_url = None
- connector_icon = inferred from mime_type
- download_metadata = None

This script is idempotent - can be run multiple times safely.

Author: Spec 057 Implementation
Created: 2025-10-07
"""

import asyncio
import logging
from typing import Dict, List
from datetime import datetime

# Constitutional compliance: Only Graph Channel import allowed
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daedalus_gateway import get_graph_channel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def infer_connector_icon_from_mime(mime_type: str) -> str:
    """
    Infer connector icon from mime_type.

    Args:
        mime_type: Document MIME type

    Returns:
        Icon hint string
    """
    mime_icon_map = {
        "application/pdf": "pdf",
        "text/html": "html",
        "text/plain": "text",
        "application/msword": "doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "doc",
        "text/markdown": "markdown",
        "application/json": "json"
    }

    return mime_icon_map.get(mime_type, "upload")


async def check_documents_needing_migration(graph_channel) -> List[Dict]:
    """
    Find documents that need source metadata migration.

    Documents need migration if they don't have source_type field.

    Args:
        graph_channel: DaedalusGraphChannel instance

    Returns:
        List of documents needing migration
    """
    query = """
    MATCH (d:Document)
    WHERE d.source_type IS NULL
    RETURN
        d.document_id as document_id,
        d.mime_type as mime_type,
        d.filename as filename
    """

    result = await graph_channel.execute_read(
        query=query,
        parameters={},
        caller_service="migration_script",
        caller_function="check_documents_needing_migration"
    )

    documents = []
    if result.get("records"):
        for record in result["records"]:
            documents.append({
                "document_id": record["document_id"],
                "mime_type": record.get("mime_type", "application/pdf"),
                "filename": record["filename"]
            })

    logger.info(f"Found {len(documents)} documents needing migration")
    return documents


async def migrate_document(graph_channel, document: Dict) -> bool:
    """
    Migrate a single document to add source metadata.

    Args:
        graph_channel: DaedalusGraphChannel instance
        document: Document to migrate

    Returns:
        True if successful, False otherwise
    """
    try:
        # Infer connector_icon from mime_type
        connector_icon = infer_connector_icon_from_mime(document["mime_type"])

        # Update document with source metadata
        update_query = """
        MATCH (d:Document {document_id: $document_id})
        SET
            d.source_type = $source_type,
            d.original_url = $original_url,
            d.connector_icon = $connector_icon,
            d.download_metadata = $download_metadata
        RETURN d.document_id as document_id
        """

        result = await graph_channel.execute_write(
            query=update_query,
            parameters={
                "document_id": document["document_id"],
                "source_type": "uploaded_file",  # Default for existing docs
                "original_url": None,
                "connector_icon": connector_icon,
                "download_metadata": None
            },
            caller_service="migration_script",
            caller_function="migrate_document"
        )

        if result.get("records"):
            logger.info(
                f"✅ Migrated {document['document_id']} "
                f"(icon: {connector_icon}, filename: {document['filename']})"
            )
            return True
        else:
            logger.warning(f"❌ Failed to migrate {document['document_id']}")
            return False

    except Exception as e:
        logger.error(f"❌ Error migrating {document['document_id']}: {e}")
        return False


async def migrate_existing_documents():
    """
    Main migration function.

    Backfills all existing documents with source metadata.
    """
    start_time = datetime.utcnow()
    logger.info("=" * 60)
    logger.info("Starting Source Metadata Migration - Spec 057")
    logger.info("=" * 60)

    try:
        # Get graph channel
        graph_channel = get_graph_channel()
        logger.info("✅ Connected to Neo4j via DaedalusGraphChannel")

        # Find documents needing migration
        documents = await check_documents_needing_migration(graph_channel)

        if not documents:
            logger.info("✅ No documents need migration - all up to date!")
            return {
                "status": "success",
                "documents_checked": 0,
                "documents_migrated": 0,
                "documents_failed": 0,
                "duration_seconds": 0
            }

        logger.info(f"📋 Starting migration of {len(documents)} documents...")

        # Migrate each document
        migrated_count = 0
        failed_count = 0

        for i, document in enumerate(documents, 1):
            logger.info(f"[{i}/{len(documents)}] Processing {document['document_id']}...")

            success = await migrate_document(graph_channel, document)

            if success:
                migrated_count += 1
            else:
                failed_count += 1

            # Progress update every 10 documents
            if i % 10 == 0:
                logger.info(
                    f"Progress: {i}/{len(documents)} "
                    f"(✅ {migrated_count} migrated, ❌ {failed_count} failed)"
                )

        # Final summary
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info("Migration Complete!")
        logger.info("=" * 60)
        logger.info(f"Documents checked: {len(documents)}")
        logger.info(f"Successfully migrated: {migrated_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("=" * 60)

        return {
            "status": "success",
            "documents_checked": len(documents),
            "documents_migrated": migrated_count,
            "documents_failed": failed_count,
            "duration_seconds": round(duration, 2)
        }

    except Exception as e:
        logger.error(f"❌ Migration failed with error: {e}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e)
        }


async def verify_migration():
    """
    Verify migration completed successfully.

    Checks that all documents have source metadata.
    """
    logger.info("=" * 60)
    logger.info("Verifying Migration...")
    logger.info("=" * 60)

    try:
        graph_channel = get_graph_channel()

        # Count documents with source metadata
        count_query = """
        MATCH (d:Document)
        RETURN
            count(d) as total_documents,
            count(d.source_type) as with_source_type,
            count(d.connector_icon) as with_connector_icon
        """

        result = await graph_channel.execute_read(
            query=count_query,
            parameters={},
            caller_service="migration_script",
            caller_function="verify_migration"
        )

        if result.get("records"):
            record = result["records"][0]
            total = record["total_documents"]
            with_source_type = record["with_source_type"]
            with_connector_icon = record["with_connector_icon"]

            logger.info(f"Total documents: {total}")
            logger.info(f"With source_type: {with_source_type}")
            logger.info(f"With connector_icon: {with_connector_icon}")

            if total == with_source_type == with_connector_icon:
                logger.info("✅ Verification PASSED - All documents have source metadata!")
                return True
            else:
                logger.warning(
                    f"⚠️ Verification INCOMPLETE - "
                    f"{total - with_source_type} documents missing source metadata"
                )
                return False

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}", exc_info=True)
        return False


async def main():
    """Run migration and verification."""
    # Run migration
    result = await migrate_existing_documents()

    # Verify if migration succeeded
    if result["status"] == "success":
        await verify_migration()


if __name__ == "__main__":
    asyncio.run(main())
