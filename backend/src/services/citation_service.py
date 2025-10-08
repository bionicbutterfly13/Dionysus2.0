#!/usr/bin/env python3
"""
Citation Service - Spec 058

Retrieves chunk, basin, and thoughtseed data for citation display.

CONSTITUTIONAL COMPLIANCE (Spec 040):
- All Neo4j access via DaedalusGraphChannel
- NO direct neo4j imports allowed
- Only: from daedalus_gateway import get_graph_channel

Author: Agent 058-Backend
Created: 2025-10-08
"""

from typing import Optional, Dict, Any
from datetime import datetime
import logging

# Constitutional compliance: Only Graph Channel import allowed
from daedalus_gateway import get_graph_channel

# Models
from ..models.citation import (
    CitationResponse,
    ChunkCitationData,
    BasinCitationData,
    ThoughtseedCitationData
)

logger = logging.getLogger(__name__)


class CitationService:
    """
    Service for retrieving citation data.

    Fetches chunk text, basin metadata, and thoughtseed information
    from Neo4j via DaedalusGraphChannel.
    """

    def __init__(self):
        """Initialize citation service with graph channel."""
        self.graph_channel = get_graph_channel()

    async def get_citation_data(
        self,
        document_id: str,
        chunk_id: str
    ) -> CitationResponse:
        """
        Retrieve citation data for a specific chunk.

        Args:
            document_id: Document identifier
            chunk_id: Chunk identifier

        Returns:
            CitationResponse with chunk, basin, and thoughtseed data

        Raises:
            ValueError: If document or chunk not found
            ValueError: If chunk doesn't belong to document
        """
        logger.info(f"Fetching citation data for document={document_id}, chunk={chunk_id}")

        # Neo4j query to fetch chunk, basin, and thoughtseed data
        query = """
        // Verify document exists
        MATCH (doc:Document {document_id: $document_id})

        // Find chunk belonging to this document
        MATCH (chunk:Chunk {chunk_id: $chunk_id})-[:BELONGS_TO]->(doc)

        // Optional: Find associated basin
        OPTIONAL MATCH (chunk)-[:ATTRACTED_TO]->(basin:Basin)

        // Optional: Get basin layer influences
        OPTIONAL MATCH (basin)-[inf:INFLUENCES_LAYER]->(layer)

        // Optional: Find associated thoughtseed
        OPTIONAL MATCH (chunk)-[:RESONATES_WITH]->(ts:ThoughtSeed)

        // Optional: Get thoughtseed concepts
        OPTIONAL MATCH (ts)-[:HAS_CONCEPT]->(concept:Concept)

        RETURN
            chunk,
            basin,
            collect(DISTINCT {layer: inf.layer_name, influence: inf.strength}) as layer_influences,
            ts,
            collect(DISTINCT concept.label) as concept_labels
        """

        params = {
            "document_id": document_id,
            "chunk_id": chunk_id
        }

        try:
            # Execute query via Graph Channel
            result = await self.graph_channel.execute_read(query, params)

            if not result or len(result) == 0:
                # Check if document exists
                doc_query = "MATCH (doc:Document {document_id: $document_id}) RETURN doc"
                doc_result = await self.graph_channel.execute_read(
                    doc_query,
                    {"document_id": document_id}
                )

                if not doc_result or len(doc_result) == 0:
                    raise ValueError(f"Document not found: {document_id}")
                else:
                    raise ValueError(f"Chunk not found or doesn't belong to document: {chunk_id}")

            # Extract data from result
            record = result[0]
            chunk_data = record["chunk"]
            basin_data = record.get("basin")
            layer_influences_raw = record.get("layer_influences", [])
            ts_data = record.get("ts")
            concept_labels = record.get("concept_labels", [])

            # Build chunk citation data
            chunk_citation = ChunkCitationData(
                chunk_id=chunk_data["chunk_id"],
                chunk_text=chunk_data["text"],
                chunk_index=chunk_data["index"],
                start_offset=chunk_data["start_offset"],
                end_offset=chunk_data["end_offset"],
                created_at=datetime.fromisoformat(chunk_data["created_at"])
            )

            # Build basin citation data (if exists)
            basin_citation = None
            if basin_data:
                # Convert layer influences to dict
                layer_influences_dict = {}
                for item in layer_influences_raw:
                    if item and "layer" in item and "influence" in item:
                        layer_influences_dict[item["layer"]] = float(item["influence"])

                basin_citation = BasinCitationData(
                    basin_id=basin_data["basin_id"],
                    basin_name=basin_data["name"],
                    stability=float(basin_data["stability"]),
                    attractor_strength=float(basin_data["activation_strength"]),
                    layer_influences=layer_influences_dict
                )

            # Build thoughtseed citation data (if exists)
            thoughtseed_citation = None
            if ts_data:
                # Filter out empty concept labels
                valid_concepts = [c for c in concept_labels if c]

                thoughtseed_citation = ThoughtseedCitationData(
                    thoughtseed_id=ts_data["seed_id"],
                    resonance_score=float(ts_data["resonance_score"]),
                    concept_labels=valid_concepts if valid_concepts else ["uncategorized"],
                    emergence_timestamp=datetime.fromisoformat(ts_data["germination_timestamp"])
                )

            # Build response
            response = CitationResponse(
                document_id=document_id,
                chunk=chunk_citation,
                basin=basin_citation,
                thoughtseed=thoughtseed_citation
            )

            logger.info(f"Successfully retrieved citation data: basin={basin_citation is not None}, "
                       f"thoughtseed={thoughtseed_citation is not None}")

            return response

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Error fetching citation data: {e}", exc_info=True)
            raise ValueError(f"Failed to retrieve citation data: {str(e)}")


# Singleton instance
_citation_service = None


def get_citation_service() -> CitationService:
    """Get singleton citation service instance."""
    global _citation_service
    if _citation_service is None:
        _citation_service = CitationService()
    return _citation_service
