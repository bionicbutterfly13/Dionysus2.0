# Knowledge Graph

## Overview

The Knowledge Graph capability provides unified storage and retrieval of interconnected knowledge using Neo4j as the single source of truth. It consolidates graph relationships, vector embeddings, and full-text search into a single database system, enabling researchers and AI systems to explore complex relationships between neural architectures, consciousness states, episodes, and archetypal patterns.

This capability eliminates data duplication across multiple storage systems while preserving all historical information and enabling sophisticated queries that span multiple entity types and relationship dimensions.

---

## Purpose

Provide unified storage and retrieval of interconnected knowledge using Neo4j as the single source of truth, consolidating graph relationships, vector embeddings, and full-text search to enable sophisticated multi-dimensional queries without data duplication.

---

## Requirements

### Requirement: Unified Storage Architecture

The system MUST store all knowledge in Neo4j without requiring separate databases for graph, vector, or full-text operations.

#### Scenario: Single Database Query
**Given** neural architecture data with embeddings and narrative content
**When** a researcher queries for similar architectures with related consciousness patterns
**Then** the system returns results from a single Neo4j query combining graph traversal, vector similarity, and full-text search

#### Scenario: Data Consolidation
**Given** data currently split across MongoDB, FAISS, and SQLite systems
**When** the migration to unified storage completes
**Then** all entity types and relationships exist in Neo4j with zero data loss

---

### Requirement: Graph Relationship Preservation

The system MUST maintain all relationships between entities including evolution paths, episodic connections, archetypal associations, and consciousness development patterns.

#### Scenario: Evolution Path Tracking
**Given** an architecture with multiple ancestor and descendant relationships
**When** a researcher queries for the complete evolution lineage
**Then** the system returns the full parent-child relationship tree with transformation sequences

#### Scenario: Multi-Entity Relationships
**Given** a research question spanning architectures, episodes, consciousness states, and archetypes
**When** the researcher executes a relationship query
**Then** the system traverses connections across all entity types and returns the complete relationship graph

---

### Requirement: Vector Similarity Search

The system MUST support embedding-based similarity searches using Neo4j's native vector indexing capabilities.

#### Scenario: Architecture Similarity
**Given** a newly discovered neural architecture with a 512-dimensional embedding
**When** a researcher searches for similar architectures
**Then** the system returns top K architectures ranked by cosine similarity using Neo4j vector index

#### Scenario: Hybrid Search
**Given** a query combining semantic similarity and graph constraints
**When** the researcher searches for "architectures similar to X that evolved from transformer patterns"
**Then** the system executes a single Cypher query combining vector search with graph pattern matching

---

### Requirement: Full-Text Narrative Search

The system MUST enable full-text search across narrative content including architecture descriptions, episodic summaries, and archetypal pattern descriptions.

#### Scenario: Narrative Content Discovery
**Given** multiple architectures with rich narrative descriptions
**When** a researcher searches for "consciousness emergence through self-reflection"
**Then** the system returns architectures and episodes containing matching narrative patterns

#### Scenario: Cross-Entity Text Search
**Given** narrative content distributed across Document, Episode, and Archetype nodes
**When** a researcher performs a full-text query
**Then** the system searches all indexed narrative fields and returns deduplicated results ranked by relevance

---

### Requirement: Constitutional Graph Access

The system MUST enforce the Daedalus Graph Channel pattern where all Neo4j access flows exclusively through the Daedalus gateway with no direct driver imports in application code.

#### Scenario: Graph Channel Enforcement
**Given** a backend service requiring knowledge graph reads
**When** the service executes a query
**Then** the request flows through DaedalusGraphChannel with no direct neo4j imports in the calling module

#### Scenario: Migration Compliance
**Given** legacy services with direct Neo4j driver access
**When** the graph hardening phase completes
**Then** all services consume the Graph Channel facade exclusively with CI checks blocking direct driver imports

---

### Requirement: Schema and Index Optimization

The system MUST maintain optimized schema constraints and indexes to support sub-second query performance at scale (10,000+ nodes).

#### Scenario: Uniqueness Constraints
**Given** documents, concepts, attractor basins, and thoughtseeds being created
**When** duplicate identifiers are attempted
**Then** the system enforces uniqueness constraints preventing data corruption

#### Scenario: Performance Indexes
**Given** frequent queries on upload_timestamp, quality scores, tier levels, and consciousness depth
**When** researchers execute filtered searches
**Then** the system uses composite indexes to return results in under 100ms

#### Scenario: Vector Index Performance
**Given** 10,000+ architectures with 512-dimensional embeddings
**When** a researcher performs a similarity search for top 10 matches
**Then** the system returns results in under 100ms using HNSW vector indexing

---

### Requirement: Data Integrity and Validation

The system MUST ensure complete data preservation during migrations and maintain referential integrity across all entity relationships.

#### Scenario: Migration Validation
**Given** data being migrated from legacy systems
**When** the migration process completes
**Then** the system validates 100% count match, 99.9% content match (sample-based), and 100% functional query equivalence

#### Scenario: Relationship Integrity
**Given** architectures with evolution paths, episodes, and consciousness states
**When** entities are updated or deleted
**Then** the system maintains relationship consistency through constraint enforcement

---

### Requirement: Multi-Strategy Search

The system MUST support multiple search strategies that can be combined: full-text search, graph pattern matching, vector similarity, and relationship traversal.

#### Scenario: Combined Search Strategies
**Given** a complex research query
**When** the system executes the search
**Then** results are gathered from full-text indexes, graph patterns, vector similarity, and relationship traversal, then deduplicated and ranked by relevance

#### Scenario: Relevance Scoring
**Given** search results from multiple strategies (full-text score, connection count, vector similarity)
**When** the system aggregates results
**Then** each result receives a normalized relevance score between 0 and 1 for ranking

---

### Requirement: Temporal Analysis

The system MUST enable tracking of consciousness development, architecture evolution, and episodic progression over time.

#### Scenario: Consciousness Development Timeline
**Given** an architecture with multiple consciousness state snapshots over time
**When** a researcher queries for consciousness development patterns
**Then** the system returns the temporal sequence showing meta-cognitive depth evolution

#### Scenario: Population-Wide Temporal Analysis
**Given** the complete architecture population with timestamps
**When** a researcher analyzes consciousness emergence trends
**Then** the system aggregates temporal data across all architectures showing emergence patterns over time

---

### Requirement: Deduplication

The system MUST prevent duplicate data storage while preserving all unique historical information.

#### Scenario: Content Hash Deduplication
**Given** multiple document uploads with identical content
**When** the system processes the documents
**Then** only one Document node is created with content_hash uniqueness enforced

#### Scenario: Result Deduplication
**Given** multiple search strategies returning overlapping results
**When** the system aggregates results
**Then** duplicates are removed while preserving the highest relevance score for each unique result

---

## Key Entities

- **Document**: Uploaded files with extracted text, embeddings, metadata (filename, upload_timestamp, processing_status, content_hash)
- **Architecture**: Neural network designs with performance metrics, consciousness indicators, structural properties, and evolution lineage
- **Episode**: Narrative segments capturing architecture development with temporal boundaries and archetypal context
- **ConsciousnessState**: Records consciousness emergence indicators, self-awareness markers, meta-cognitive depth levels
- **Archetype**: Archetypal patterns (Hero, Sage, Creator, etc.) with resonance criteria and psychological markers
- **Concept**: Extracted concepts from document processing with center_concept, quality scores, tier levels
- **AttractorBasin**: Stable states in consciousness landscape with attraction dynamics and basin_id
- **Thoughtseed**: Germinal ideas with salience levels and developmental potential
- **EvolutionPath**: Parent-child relationships tracking transformations between architectures
- **NarrativePattern**: Recurring story elements and motifs appearing across multiple episodes
- **ContextStream**: Information flow dynamics in the river metaphor framework

---

## Success Criteria

- All knowledge graph operations execute through DaedalusGraphChannel with zero direct neo4j imports in application code
- Vector similarity searches return top 10 results in under 100ms for 10,000+ node graphs
- Full-text searches complete in under 200ms for narrative content
- Evolution path queries return complete lineages in under 50ms
- Multi-entity relationship queries execute in a single Cypher statement
- Schema constraints enforce uniqueness for all entity identifiers
- Migration validation shows 100% data count match and 99.9% content match
- CI pipeline blocks any new direct Neo4j driver imports outside the gateway
