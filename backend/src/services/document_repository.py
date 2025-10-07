#!/usr/bin/env python3
"""
Document Repository Service - Spec 054

Persists Daedalus LangGraph final_output to Neo4j via Graph Channel.

CONSTITUTIONAL COMPLIANCE (Spec 040):
- All Neo4j access via DaedalusGraphChannel
- NO direct neo4j imports allowed
- Only: from daedalus_gateway import get_graph_channel

Author: Spec 054 Implementation
Created: 2025-10-07
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import time
import asyncio
import redis

# Constitutional compliance: Only Graph Channel import allowed
from daedalus_gateway import get_graph_channel

# Models (using relative imports for compatibility)
from ..models.document_node import (
    DocumentNode, ConceptNode, ThoughtSeedNode, AttractorBasinNode,
    TierEnum, ProcessingStatus, ConceptLevel
)
from ..models.document_relationships import (
    ExtractedFromRel, AttractedToRel, GerminatedFromRel,
    InfluenceType
)

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Repository for document persistence and retrieval.

    All Neo4j operations go through DaedalusGraphChannel (Spec 040 compliance).
    """

    def __init__(self):
        """Initialize repository with Graph Channel."""
        self.graph_channel = get_graph_channel()

        # Redis for basin evolution tracking (optional, with fallback)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            self.redis_available = True
            logger.info("DocumentRepository initialized with Graph Channel + Redis")
        except Exception as e:
            self.redis_client = None
            self.redis_available = False
            logger.warning(f"Redis not available, basin evolution tracking disabled: {e}")

    async def persist_document(
        self,
        final_output: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Persist document with all processing artifacts to Neo4j.

        T024-T029 implementation from plan.md lines 615-672.

        Args:
            final_output: Daedalus LangGraph final_output
            metadata: Document metadata (filename, content_hash, tags, etc.)

        Returns:
            Persistence result with performance metrics
        """
        start_time = time.time()
        logger.info(f"Starting persistence for document {metadata.get('document_id')}")

        try:
            # T024: Validation and duplicate check
            await self._validate_and_check_duplicates(metadata)

            # T025: Create Document node
            nodes_created = 1
            await self._create_document_node(final_output, metadata)

            # T026: Persist 5-level concepts
            concepts_count = await self._persist_concepts(
                final_output.get("concepts", {}),
                metadata["document_id"]
            )
            nodes_created += concepts_count

            # T027: Persist attractor basins with Context Engineering
            basins_count = await self._persist_basins(
                final_output.get("basins", []),
                metadata["document_id"]
            )
            nodes_created += basins_count

            # T028: Persist thoughtseeds
            seeds_count = await self._persist_thoughtseeds(
                final_output.get("thoughtseeds", []),
                metadata["document_id"]
            )
            nodes_created += seeds_count

            # T029: Calculate performance metrics
            persistence_duration_ms = (time.time() - start_time) * 1000
            met_target = persistence_duration_ms < 2000  # <2s target

            result = {
                "status": "success",
                "document_id": metadata["document_id"],
                "persisted_at": datetime.utcnow().isoformat(),
                "tier": "warm",
                "nodes_created": nodes_created,
                "relationships_created": concepts_count + basins_count + seeds_count,
                "performance": {
                    "persistence_duration_ms": round(persistence_duration_ms, 2),
                    "met_target": met_target
                }
            }

            if not met_target:
                logger.warning(
                    f"Performance target missed: {persistence_duration_ms:.0f}ms (target: <2000ms)"
                )

            logger.info(f"✅ Document {metadata['document_id']} persisted successfully in {persistence_duration_ms:.0f}ms")
            return result

        except Exception as e:
            logger.error(f"❌ Document persistence failed: {e}", exc_info=True)
            raise

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        T030: Retrieve document detail with all artifacts and update access tracking.

        From plan.md lines 983-1075.

        Args:
            document_id: Document ID

        Returns:
            Complete document data or None if not found
        """
        # Single comprehensive query to minimize round trips
        query = """
        MATCH (d:Document {document_id: $document_id})

        // Get all concepts by level
        OPTIONAL MATCH (c:Concept)-[r_concept:EXTRACTED_FROM]->(d)
        WITH d, collect({
            concept_id: c.concept_id,
            name: c.name,
            level: c.level,
            salience: c.salience,
            definition: c.definition,
            source_concept: c.source_concept,
            target_concept: c.target_concept,
            components: c.components,
            domain: c.domain,
            storyline: c.storyline
        }) as concepts

        // Get all basins
        OPTIONAL MATCH (b:AttractorBasin)-[r_basin:ATTRACTED_TO]->(d)
        WITH d, concepts, collect({
            basin_id: b.basin_id,
            name: b.name,
            depth: b.depth,
            stability: b.stability,
            strength: b.strength,
            activation_strength: r_basin.activation_strength,
            influence_type: r_basin.influence_type
        }) as basins

        // Get all thoughtseeds
        OPTIONAL MATCH (t:ThoughtSeed)-[r_seed:GERMINATED_FROM]->(d)
        WITH d, concepts, basins, collect({
            seed_id: t.seed_id,
            content: t.content,
            germination_potential: t.germination_potential,
            resonance_score: t.resonance_score,
            field_resonance_energy: t.field_resonance_energy,
            field_resonance_phase: t.field_resonance_phase,
            field_resonance_pattern: t.field_resonance_pattern
        }) as thoughtseeds

        // Update access tracking
        SET d.last_accessed = datetime(),
            d.access_count = d.access_count + 1

        RETURN d, concepts, basins, thoughtseeds
        """

        result = await self.graph_channel.execute_write(  # Write to update access tracking
            query=query,
            parameters={"document_id": document_id},
            caller_service="document_repository",
            caller_function="get_document"
        )

        if not result.get("records"):
            logger.warning(f"Document not found: {document_id}")
            return None

        record = result["records"][0]
        doc = record["d"]

        # Organize concepts by level
        concepts_by_level = {
            "atomic": [],
            "relationship": [],
            "composite": [],
            "context": [],
            "narrative": []
        }
        for concept in record["concepts"]:
            if concept and concept.get("concept_id"):  # Filter null entries
                level = concept.get("level")
                if level in concepts_by_level:
                    concepts_by_level[level].append(concept)

        # Build response
        response = {
            "document_id": doc["document_id"],
            "metadata": {
                "filename": doc["filename"],
                "upload_timestamp": doc["upload_timestamp"],
                "file_size": doc["file_size"],
                "mime_type": doc["mime_type"],
                "tags": doc["tags"],
                "tier": doc["tier"],
                "last_accessed": doc["last_accessed"],
                "access_count": doc["access_count"]
            },
            "quality": {
                "overall": doc["quality_overall"],
                "coherence": doc.get("quality_coherence"),
                "novelty": doc.get("quality_novelty"),
                "depth": doc.get("quality_depth")
            },
            "concepts": concepts_by_level,
            "basins": [b for b in record["basins"] if b and b.get("basin_id")],
            "thoughtseeds": [s for s in record["thoughtseeds"] if s and s.get("seed_id")],
            "processing_timeline": []  # Can be added from metadata if stored
        }

        logger.info(f"✅ Retrieved document {document_id} (access_count: {doc['access_count']})")
        return response

    async def list_documents(
        self,
        page: int = 1,
        limit: int = 50,
        tags: Optional[List[str]] = None,
        quality_min: Optional[float] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sort: str = "upload_date",
        order: str = "desc",
        tier: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        T031-T033: List documents with pagination, filtering, sorting.

        From plan.md lines 864-982.

        Args:
            page: Page number (1-indexed)
            limit: Items per page
            tags: Filter by tags
            quality_min: Minimum quality score
            date_from: Start date (ISO 8601)
            date_to: End date (ISO 8601)
            sort: Sort field (upload_date, quality, curiosity)
            order: Sort order (asc, desc)
            tier: Filter by tier (warm, cool, cold)

        Returns:
            Documents list with pagination metadata
        """
        start_time = time.time()

        # T031: Build WHERE clause with filters
        where_clauses = []
        parameters = {}

        if tags:
            # ANY tag in the list matches
            where_clauses.append("ANY(tag IN $tags WHERE tag IN d.tags)")
            parameters["tags"] = tags

        if quality_min is not None:
            where_clauses.append("d.quality_overall >= $quality_min")
            parameters["quality_min"] = quality_min

        if date_from:
            where_clauses.append("d.upload_timestamp >= datetime($date_from)")
            parameters["date_from"] = date_from

        if date_to:
            where_clauses.append("d.upload_timestamp <= datetime($date_to)")
            parameters["date_to"] = date_to

        if tier:
            where_clauses.append("d.tier = $tier")
            parameters["tier"] = tier

        where_clause = " AND ".join(where_clauses) if where_clauses else "true"

        # T031: Build ORDER BY clause
        sort_field_map = {
            "upload_date": "d.upload_timestamp",
            "quality": "d.quality_overall",
            "curiosity": "d.curiosity_triggers"
        }
        sort_field = sort_field_map.get(sort, "d.upload_timestamp")
        order_direction = "DESC" if order.lower() == "desc" else "ASC"

        # T032: Calculate pagination
        skip = (page - 1) * limit
        parameters["skip"] = skip
        parameters["limit"] = limit

        # T033: Optimized query - split count from pagination for performance
        # Use indexes created in T020 for performance
        query = f"""
        MATCH (d:Document)
        WHERE {where_clause}

        WITH d
        ORDER BY {sort_field} {order_direction}
        SKIP $skip LIMIT $limit

        // Get artifact counts efficiently using pattern comprehension
        RETURN
            d.document_id as document_id,
            d.filename as filename,
            d.upload_timestamp as upload_timestamp,
            d.quality_overall as quality_overall,
            d.tags as tags,
            d.tier as tier,
            d.curiosity_triggers as curiosity_triggers,
            d.file_size as file_size,
            size([(c:Concept)-[:EXTRACTED_FROM]->(d) | c]) as concept_count,
            size([(b:AttractorBasin)-[:ATTRACTED_TO]->(d) | b]) as basin_count,
            size([(t:ThoughtSeed)-[:GERMINATED_FROM]->(d) | t]) as thoughtseed_count
        """

        # Separate count query for total (only run when needed)
        count_query = f"""
        MATCH (d:Document)
        WHERE {where_clause}
        RETURN count(d) as total
        """

        # Execute both queries in parallel for performance
        result, count_result = await asyncio.gather(
            self.graph_channel.execute_read(
                query=query,
                parameters=parameters,
                caller_service="document_repository",
                caller_function="list_documents"
            ),
            self.graph_channel.execute_read(
                query=count_query,
                parameters=parameters,
                caller_service="document_repository",
                caller_function="list_documents_count"
            )
        )

        query_duration_ms = (time.time() - start_time) * 1000

        # Process results
        documents = []
        total = count_result["records"][0]["total"] if count_result.get("records") else 0

        if result.get("records"):
            for record in result["records"]:
                documents.append({
                    "document_id": record["document_id"],
                    "filename": record["filename"],
                    "upload_timestamp": record["upload_timestamp"],
                    "quality_overall": record["quality_overall"],
                    "tags": record["tags"],
                    "tier": record["tier"],
                    "curiosity_triggers": record["curiosity_triggers"],
                    "file_size": record["file_size"],
                    "concept_count": record["concept_count"],
                    "basin_count": record["basin_count"],
                    "thoughtseed_count": record["thoughtseed_count"]
                })

        # T032: Calculate pagination metadata
        total_pages = (total + limit - 1) // limit if total > 0 else 0

        # T033: Performance validation
        met_target = query_duration_ms < 500  # <500ms target

        response = {
            "documents": documents,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": total_pages
            },
            "performance": {
                "query_duration_ms": round(query_duration_ms, 2),
                "met_target": met_target
            }
        }

        if not met_target:
            logger.warning(
                f"Performance target missed for list_documents: {query_duration_ms:.0f}ms (target: <500ms)"
            )

        logger.info(
            f"✅ Listed {len(documents)} documents (page {page}/{total_pages}, "
            f"total: {total}, {query_duration_ms:.0f}ms)"
        )

        return response

    async def update_tier(
        self,
        document_id: str,
        new_tier: str,
        reason: str = "manual"
    ) -> Dict[str, Any]:
        """
        Update document tier.

        Args:
            document_id: Document ID
            new_tier: New tier (warm, cool, cold)
            reason: Reason for tier change

        Returns:
            Tier update result
        """
        # TODO: Implement (T037)
        raise NotImplementedError("update_tier not yet implemented")

    # Private helper methods for persist_document()

    async def _validate_and_check_duplicates(self, metadata: Dict[str, Any]) -> None:
        """
        T024: Validate required fields and check for duplicates.

        From plan.md lines 221-226.
        """
        # Validate required fields
        required_fields = ["document_id", "filename", "content_hash"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required field: {field}")

        # Check for duplicate by content_hash
        duplicate_check_query = """
        MATCH (d:Document {content_hash: $content_hash})
        RETURN d.document_id as document_id, d.filename as filename, d.upload_timestamp as uploaded_at
        """

        result = await self.graph_channel.execute_read(
            query=duplicate_check_query,
            parameters={"content_hash": metadata["content_hash"]},
            caller_service="document_repository",
            caller_function="_validate_and_check_duplicates"
        )

        if result.get("records"):
            existing = result["records"][0]
            raise ValueError(
                f"Duplicate document detected. Content hash {metadata['content_hash']} "
                f"already exists as document {existing['document_id']}"
            )

    async def _create_document_node(
        self,
        final_output: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """
        T025: Create Document node in Neo4j.

        From plan.md lines 647-672.
        """
        quality = final_output.get("quality", {}).get("scores", {})
        research = final_output.get("research", {})

        create_query = """
        CREATE (d:Document {
            document_id: $document_id,
            filename: $filename,
            content_hash: $content_hash,
            upload_timestamp: datetime($upload_timestamp),
            file_size: $file_size,
            mime_type: $mime_type,
            tags: $tags,

            processed_at: datetime(),
            processing_duration_ms: $processing_duration_ms,
            processing_status: $processing_status,

            quality_overall: $quality_overall,
            quality_coherence: $quality_coherence,
            quality_novelty: $quality_novelty,
            quality_depth: $quality_depth,

            curiosity_triggers: $curiosity_triggers,
            research_questions: $research_questions,

            tier: "warm",
            last_accessed: datetime(),
            access_count: 0,
            tier_changed_at: datetime(),

            archive_location: null,
            archived_at: null
        })
        RETURN d.document_id as document_id
        """

        parameters = {
            "document_id": metadata["document_id"],
            "filename": metadata["filename"],
            "content_hash": metadata["content_hash"],
            "upload_timestamp": metadata.get("upload_timestamp", datetime.utcnow().isoformat()),
            "file_size": metadata.get("file_size", 0),
            "mime_type": metadata.get("mime_type", "application/pdf"),
            "tags": metadata.get("tags", []),
            "processing_duration_ms": final_output.get("processing_duration_ms", 0),
            "processing_status": "complete",
            "quality_overall": quality.get("overall", 0.0),
            "quality_coherence": quality.get("coherence"),
            "quality_novelty": quality.get("novelty"),
            "quality_depth": quality.get("depth"),
            "curiosity_triggers": research.get("curiosity_triggers", 0),
            "research_questions": research.get("research_questions", 0)
        }

        await self.graph_channel.execute_write(
            query=create_query,
            parameters=parameters,
            caller_service="document_repository",
            caller_function="_create_document_node"
        )

        logger.info(f"✅ Created Document node: {metadata['document_id']}")

    async def _persist_concepts(
        self,
        concepts_data: Dict[str, List[Dict]],
        document_id: str
    ) -> int:
        """
        T026: Persist 5-level concepts.

        From plan.md lines 689-740.
        """
        total_concepts = 0
        levels = ["atomic", "relationship", "composite", "context", "narrative"]

        for level in levels:
            concepts = concepts_data.get(level, [])
            for concept in concepts:
                # Create concept node
                create_concept_query = """
                MERGE (c:Concept {concept_id: $concept_id})
                ON CREATE SET
                    c.name = $name,
                    c.level = $level,
                    c.salience = $salience,
                    c.definition = $definition,
                    c.source_concept = $source_concept,
                    c.target_concept = $target_concept,
                    c.components = $components,
                    c.domain = $domain,
                    c.era = $era,
                    c.storyline = $storyline,
                    c.created_at = datetime()
                RETURN c.concept_id
                """

                await self.graph_channel.execute_write(
                    query=create_concept_query,
                    parameters={
                        "concept_id": concept["concept_id"],
                        "name": concept["name"],
                        "level": level,
                        "salience": concept.get("salience", 0.5),
                        "definition": concept.get("definition"),
                        "source_concept": concept.get("source_concept"),
                        "target_concept": concept.get("target_concept"),
                        "components": concept.get("components", []),
                        "domain": concept.get("domain"),
                        "era": concept.get("era"),
                        "storyline": concept.get("storyline")
                    },
                    caller_service="document_repository",
                    caller_function="_persist_concepts"
                )

                # Link to document
                link_query = """
                MATCH (c:Concept {concept_id: $concept_id})
                MATCH (d:Document {document_id: $document_id})
                CREATE (c)-[:EXTRACTED_FROM {
                    confidence: $confidence,
                    extraction_method: "AutoSchemaKG",
                    timestamp: datetime()
                }]->(d)
                """

                await self.graph_channel.execute_write(
                    query=link_query,
                    parameters={
                        "concept_id": concept["concept_id"],
                        "document_id": document_id,
                        "confidence": concept.get("confidence", 0.90)
                    },
                    caller_service="document_repository",
                    caller_function="_persist_concepts"
                )

                total_concepts += 1

        logger.info(f"✅ Persisted {total_concepts} concepts for document {document_id}")
        return total_concepts

    async def _persist_basins(
        self,
        basins_data: List[Dict],
        document_id: str
    ) -> int:
        """
        T027: Persist attractor basins with Context Engineering integration.

        From plan.md lines 760-823.
        """
        total_basins = 0

        for basin in basins_data:
            # Create/update basin node
            basin_query = """
            MERGE (b:AttractorBasin {basin_id: $basin_id})
            ON CREATE SET
                b.name = $name,
                b.depth = $depth,
                b.stability = $stability,
                b.strength = $strength,
                b.associated_concepts = $associated_concepts,
                b.created_at = datetime(),
                b.modification_count = 0
            ON MATCH SET
                b.depth = $depth,
                b.stability = $stability,
                b.strength = b.strength + $strength_delta,
                b.last_modified = datetime(),
                b.modification_count = b.modification_count + 1
            RETURN b.basin_id
            """

            await self.graph_channel.execute_write(
                query=basin_query,
                parameters={
                    "basin_id": basin["basin_id"],
                    "name": basin["name"],
                    "depth": basin["depth"],
                    "stability": basin["stability"],
                    "strength": basin.get("strength", 1.0),
                    "strength_delta": basin.get("strength_delta", 0.0),
                    "associated_concepts": basin.get("associated_concepts", [])
                },
                caller_service="document_repository",
                caller_function="_persist_basins"
            )

            # Link basin to document
            link_query = """
            MATCH (b:AttractorBasin {basin_id: $basin_id})
            MATCH (d:Document {document_id: $document_id})
            CREATE (b)-[:ATTRACTED_TO {
                activation_strength: $activation_strength,
                influence_type: $influence_type,
                strength_delta: $strength_delta,
                timestamp: datetime()
            }]->(d)
            """

            await self.graph_channel.execute_write(
                query=link_query,
                parameters={
                    "basin_id": basin["basin_id"],
                    "document_id": document_id,
                    "activation_strength": basin.get("activation_strength", 0.85),
                    "influence_type": basin.get("influence_type", "reinforcement"),
                    "strength_delta": basin.get("strength_delta", 0.10)
                },
                caller_service="document_repository",
                caller_function="_persist_basins"
            )

            # Update Redis basin evolution tracking (T046)
            await self._update_basin_manager(basin, document_id)

            total_basins += 1

        logger.info(f"✅ Persisted {total_basins} basins for document {document_id}")
        return total_basins

    async def _persist_thoughtseeds(
        self,
        thoughtseeds_data: List[Dict],
        document_id: str
    ) -> int:
        """
        T028: Persist thoughtseeds with neural field resonance.

        From plan.md lines 258-269.
        """
        total_seeds = 0

        for seed in thoughtseeds_data:
            # Create thoughtseed node
            seed_query = """
            CREATE (t:ThoughtSeed {
                seed_id: $seed_id,
                content: $content,
                germination_potential: $germination_potential,
                resonance_score: $resonance_score,
                field_resonance_energy: $field_resonance_energy,
                field_resonance_phase: $field_resonance_phase,
                field_resonance_pattern: $field_resonance_pattern,
                generated_at: datetime(),
                source_stage: $source_stage
            })
            """

            field_resonance = seed.get("field_resonance", {})

            await self.graph_channel.execute_write(
                query=seed_query,
                parameters={
                    "seed_id": seed["seed_id"],
                    "content": seed["content"],
                    "germination_potential": seed["germination_potential"],
                    "resonance_score": seed["resonance_score"],
                    "field_resonance_energy": field_resonance.get("energy"),
                    "field_resonance_phase": field_resonance.get("phase"),
                    "field_resonance_pattern": field_resonance.get("interference_pattern"),
                    "source_stage": seed.get("source_stage", "consciousness_processing")
                },
                caller_service="document_repository",
                caller_function="_persist_thoughtseeds"
            )

            # Link to document
            link_query = """
            MATCH (t:ThoughtSeed {seed_id: $seed_id})
            MATCH (d:Document {document_id: $document_id})
            CREATE (t)-[:GERMINATED_FROM {
                potential: $germination_potential,
                generation_stage: $source_stage,
                timestamp: datetime()
            }]->(d)
            """

            await self.graph_channel.execute_write(
                query=link_query,
                parameters={
                    "seed_id": seed["seed_id"],
                    "document_id": document_id,
                    "germination_potential": seed["germination_potential"],
                    "source_stage": seed.get("source_stage", "consciousness_processing")
                },
                caller_service="document_repository",
                caller_function="_persist_thoughtseeds"
            )

            total_seeds += 1

        logger.info(f"✅ Persisted {total_seeds} thoughtseeds for document {document_id}")
        return total_seeds

    async def _update_basin_manager(self, basin: Dict, document_id: str) -> None:
        """
        T046: Update basin evolution history in Redis.

        From plan.md lines 1149-1180.
        """
        if not self.redis_available:
            return

        try:
            import json

            # Store basin influence event
            redis_key = f"basin:evolution:{basin['basin_id']}"
            influence_event = {
                "document_id": document_id,
                "influence_type": basin.get("influence_type", "reinforcement"),
                "strength_delta": basin.get("strength_delta", 0.0),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Append to evolution history
            self.redis_client.rpush(redis_key, json.dumps(influence_event))

            # Set TTL: 90 days
            self.redis_client.expire(redis_key, 90 * 24 * 60 * 60)

            logger.debug(f"✅ Updated basin evolution in Redis: {basin['basin_id']}")

        except Exception as e:
            logger.warning(f"Failed to update basin manager in Redis: {e}")
