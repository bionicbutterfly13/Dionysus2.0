# Spec 032: Emergent Pattern Detection and Entity Evolution Tracking

**Status**: DRAFT
**Priority**: HIGH
**Dependencies**: 028 (ThoughtSeeds), 029 (Curiosity Agents), 031 (Conflict Resolution)
**Created**: 2025-10-01

## Overview

Implement autonomous detection of **emergent patterns** and **emerging entities** in the agentic knowledge graph using iterative agent reasoning, burst detection, temporal tracking, and meta-analysis. The system captures emergence—new knowledge, relationships, or entities that weren't explicitly programmed—arising from multi-agent interaction and distributed reasoning.

## Problem Statement

Based on research into agentic knowledge graph capabilities for capturing emergent phenomena, current system lacks:

### Missing Capabilities

1. **No Emerging Entity Detection**
   - Papers mention "quantum neural networks" (new term) → System treats as unknown, doesn't track emergence
   - No frequency/burst detection to identify trending concepts
   - No temporal tracking of when entities first appeared

2. **No Emergent Pattern Recognition**
   - 10 papers independently discover same principle → Pattern not automatically detected
   - Cross-domain connections (biology + ML) → Not identified as emergent insight
   - Hub formation, bridge nodes → Not tracked or analyzed

3. **No Entity Evolution Tracking**
   - "Neural architecture search" evolves to "differentiable NAS" → No lineage tracking
   - Concept drift over time not captured
   - Historical states lost (no temporal versioning)

4. **No Meta-Agent Analysis**
   - Agent interaction patterns not analyzed
   - Collaboration networks not discovered
   - System-level insights not extracted

## Research-Backed Requirements

Based on your research on emergent phenomena in agentic knowledge graphs:

### FR1: Iterative Agent Reasoning (AgREE Pattern)
**Description**: Agents iteratively retrieve, reason, and create new graph triplets for emerging entities
**Research Source**: "Frameworks like AgREE enable agents to iteratively retrieve data and reason through multi-step queries, creating new graph triplets"
**Acceptance Criteria**:
- [ ] Agents can create new relationships for previously unseen entities
- [ ] Multi-hop reasoning generates novel connections
- [ ] New triplets merged into global graph after validation
- [ ] Reasoning traces stored for meta-analysis

### FR2: Emerging Entity Detection from Streaming Data
**Description**: Automated NER + burst detection identifies new entities in real-time
**Research Source**: "Use domain-adapted NER models to scan incoming data streams for new entity mentions"
**Acceptance Criteria**:
- [ ] NER models detect new entity mentions in each paper
- [ ] Frequency tracking identifies concept "bursts" (rapid mention increase)
- [ ] Context clustering spots unusual term co-occurrences
- [ ] Fuzzy matching filters variants before declaring new entity

**Implementation**:
```python
class EmergingEntityDetector:
    """Detect new entities from streaming document uploads"""

    def __init__(self, ner_model, entity_index):
        self.ner = ner_model  # spaCy or transformer-based
        self.entity_index = entity_index  # Known entities in KG
        self.mention_tracker = {}  # Frequency tracking

    def detect_emerging_entities(self, text: str, time_window: int = 3600) -> List[str]:
        """Detect entities not in KG with burst patterns"""

        # Step 1: Extract all entities
        extracted_entities = self.ner(text).ents

        # Step 2: Filter to unknown entities
        unknown_entities = [
            ent.text for ent in extracted_entities
            if ent.text not in self.entity_index
        ]

        # Step 3: Track mention frequency
        current_time = time.time()
        for entity in unknown_entities:
            if entity not in self.mention_tracker:
                self.mention_tracker[entity] = []
            self.mention_tracker[entity].append(current_time)

        # Step 4: Detect bursts (rapid frequency increase)
        emerging = []
        for entity, timestamps in self.mention_tracker.items():
            recent_mentions = [
                t for t in timestamps
                if current_time - t < time_window
            ]

            # Burst detected if 3+ mentions in time window
            if len(recent_mentions) >= 3:
                emerging.append({
                    "entity": entity,
                    "mention_count": len(recent_mentions),
                    "burst_score": len(recent_mentions) / time_window,
                    "first_seen": min(timestamps)
                })

        return emerging
```

### FR3: Temporal Knowledge Graph (Entity Evolution)
**Description**: Track when entities appeared, changed, or deprecated
**Research Source**: "Temporal knowledge graphs record not just that an entity exists, but when it was first seen, changed, or deprecated"
**Acceptance Criteria**:
- [ ] All entities have `first_seen`, `last_updated`, `deprecated_at` timestamps
- [ ] Time-based queries: "What concepts emerged in Q1 2025?"
- [ ] Lineage tracking: "neural architecture search" → "differentiable NAS" → "zero-cost NAS"
- [ ] Historical snapshots: Query graph state at any point in time

**Neo4j Schema**:
```cypher
// Temporal entity node
CREATE (e:Concept {
  name: "quantum neural networks",
  first_seen: "2025-10-01T10:30:00",
  last_updated: "2025-10-01T14:20:00",
  mention_count: 3,
  emergence_score: 0.85,  // Burst score
  status: "emerging"  // emerging/established/deprecated
})

// Temporal relationship with versioning
CREATE (e1:Concept)-[:EVOLVED_INTO {
  transition_date: "2025-10-05T12:00:00",
  evidence_papers: ["doc_123", "doc_456"],
  transition_type: "refinement"  // refinement/replacement/extension
}]->(e2:Concept)

// Historical state snapshots
CREATE (snapshot:GraphSnapshot {
  snapshot_id: "snapshot_2025-10-01",
  timestamp: "2025-10-01T23:59:59",
  entity_count: 1234,
  relationship_count: 5678,
  emerging_entities: ["quantum neural networks", "neuromorphic computing"]
})
```

### FR4: Emergent Pattern Recognition
**Description**: Detect novel patterns from agent interaction and multi-document analysis
**Research Source**: "Open-ended expansion: agents generate new concepts and links, creating self-organizing hubs, bridge nodes, and evolving centrality"
**Acceptance Criteria**:
- [ ] Hub detection: Identify concepts with >20 relationships (domain-central)
- [ ] Bridge node detection: Concepts connecting disconnected clusters
- [ ] Cross-domain emergence: Detect when concepts from different domains link
- [ ] Structural evolution metrics: Track modularity, clustering coefficient over time

**Implementation**:
```python
class EmergentPatternDetector:
    """Detect emergent patterns in knowledge graph structure"""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def detect_hub_formation(self, min_degree: int = 20) -> List[Dict]:
        """Identify emerging hub nodes (high centrality)"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Concept)-[r]-()
                WITH n, count(r) as degree
                WHERE degree >= $min_degree
                RETURN n.name as concept,
                       degree,
                       n.first_seen as first_seen,
                       n.emergence_score as emergence_score
                ORDER BY degree DESC
            """, min_degree=min_degree)

            hubs = [dict(record) for record in result]

            # Flag newly formed hubs (first_seen < 30 days ago)
            emerging_hubs = [
                hub for hub in hubs
                if self._is_recent(hub["first_seen"], days=30)
            ]

            logger.info(f"🌟 Detected {len(emerging_hubs)} emerging hub nodes")
            return emerging_hubs

    def detect_bridge_nodes(self) -> List[Dict]:
        """Identify concepts bridging disconnected clusters"""
        with self.driver.session() as session:
            # Use betweenness centrality to find bridges
            result = session.run("""
                CALL gds.betweenness.stream('knowledge-graph')
                YIELD nodeId, score
                WHERE score > 0.5
                MATCH (n) WHERE id(n) = nodeId
                RETURN n.name as concept,
                       score as betweenness,
                       n.first_seen as first_seen
                ORDER BY score DESC
                LIMIT 50
            """)

            bridges = [dict(record) for record in result]

            logger.info(f"🌉 Detected {len(bridges)} bridge nodes connecting clusters")
            return bridges

    def detect_cross_domain_emergence(self) -> List[Dict]:
        """Detect when concepts from different domains connect"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c1:Concept)-[r]-(c2:Concept)
                WHERE c1.domain <> c2.domain
                  AND r.created_at > datetime() - duration({days: 7})
                WITH c1.domain as domain1, c2.domain as domain2,
                     type(r) as relationship_type,
                     count(r) as connection_count
                WHERE connection_count >= 3
                RETURN domain1, domain2, relationship_type, connection_count
                ORDER BY connection_count DESC
            """)

            cross_domain = [dict(record) for record in result]

            logger.info(f"🔗 Detected {len(cross_domain)} cross-domain emergent patterns")
            return cross_domain
```

### FR5: Meta-Agent Analysis
**Description**: Meta-agent analyzes agent interaction patterns and system-level evolution
**Research Source**: "Meta-agents evaluate outcomes, refine rule sets, and help agents adapt strategies over time"
**Acceptance Criteria**:
- [ ] Meta-agent runs daily to analyze graph evolution
- [ ] Detects collaboration patterns between agents
- [ ] Identifies successful agent strategies (high-quality outputs)
- [ ] Recommends workflow optimizations
- [ ] Updates Cognition Base with learnings

**Implementation**:
```python
class MetaAnalysisAgent:
    """Analyze system-level patterns and guide agent evolution"""

    def __init__(self, neo4j_driver, cognition_base):
        self.driver = neo4j_driver
        self.cognition = cognition_base

    def analyze_graph_evolution(self, days: int = 7) -> Dict:
        """Daily meta-analysis of graph evolution"""

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "period_days": days,
            "emerging_entities": self._detect_emerging_entities(days),
            "emergent_hubs": self._detect_hub_formation(days),
            "bridge_nodes": self._detect_bridge_nodes(days),
            "cross_domain_patterns": self._detect_cross_domain_emergence(days),
            "agent_collaboration": self._analyze_agent_collaboration(days),
            "structural_metrics": self._calculate_structural_metrics(),
        }

        # Store analysis in Neo4j
        self._store_meta_analysis(analysis)

        # Update Cognition Base with learnings
        self._update_cognition_from_analysis(analysis)

        logger.info(f"🧠 Meta-analysis complete: {len(analysis['emerging_entities'])} emerging entities detected")
        return analysis

    def _analyze_agent_collaboration(self, days: int) -> List[Dict]:
        """Analyze which agents collaborate effectively"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (agent1:Agent)-[:CONTRIBUTED_TO]->(doc:Document)
                      <-[:CONTRIBUTED_TO]-(agent2:Agent)
                WHERE doc.created_at > datetime() - duration({days: $days})
                  AND agent1.agent_id <> agent2.agent_id
                WITH agent1, agent2, collect(doc) as shared_docs
                WHERE size(shared_docs) >= 5
                RETURN agent1.agent_id as agent_a,
                       agent2.agent_id as agent_b,
                       size(shared_docs) as collaboration_count,
                       avg([doc IN shared_docs | doc.quality_score]) as avg_quality
                ORDER BY collaboration_count DESC
            """, days=days)

            collaborations = [dict(record) for record in result]

            # Identify high-quality pairs
            successful_pairs = [
                c for c in collaborations
                if c["avg_quality"] > 0.8
            ]

            logger.info(f"🤝 Found {len(successful_pairs)} high-quality agent collaborations")
            return successful_pairs

    def _calculate_structural_metrics(self) -> Dict:
        """Calculate graph-level structural metrics"""
        with self.driver.session() as session:
            # Modularity (how clustered the graph is)
            modularity = session.run("""
                CALL gds.louvain.stream('knowledge-graph')
                YIELD communityId, nodeId
                RETURN count(DISTINCT communityId) as community_count,
                       avg(size(collect(nodeId))) as avg_community_size
            """).single()

            # Clustering coefficient
            clustering = session.run("""
                CALL gds.localClusteringCoefficient.stream('knowledge-graph')
                YIELD nodeId, localClusteringCoefficient
                RETURN avg(localClusteringCoefficient) as avg_clustering
            """).single()

            return {
                "modularity": {
                    "community_count": modularity["community_count"],
                    "avg_community_size": modularity["avg_community_size"]
                },
                "avg_clustering_coefficient": clustering["avg_clustering"]
            }

    def _update_cognition_from_analysis(self, analysis: Dict):
        """Update Cognition Base with meta-level learnings"""

        # Record successful agent collaboration patterns
        for collab in analysis["agent_collaboration"]:
            if collab["avg_quality"] > 0.8:
                self.cognition.record_successful_pattern(
                    "agent_collaboration",
                    {
                        "agents": [collab["agent_a"], collab["agent_b"]],
                        "collaboration_count": collab["collaboration_count"],
                        "avg_quality": collab["avg_quality"],
                        "recommendation": "Prioritize this agent pair for complex tasks"
                    }
                )

        # Record emerging topics for curiosity agents
        for entity in analysis["emerging_entities"]:
            if entity["emergence_score"] > 0.8:
                self.cognition.boost_strategy_priority(
                    "emerging_topics",
                    entity["entity"],
                    0.2  # Boost priority
                )

        logger.info("📚 Cognition Base updated with meta-analysis learnings")
```

### FR6: Provenance and Trust Signals
**Description**: Record comprehensive provenance metadata and trust signals for all emerging entities
**Research Source**: "The reliability and usefulness of new entities depend on recording strong provenance and trust signals"
**Acceptance Criteria**:
- [ ] **Source URI/Document**: Exact location where entity was extracted
- [ ] **Extraction Timestamp**: When entity was first observed
- [ ] **Extractor Identity**: Process, model, or agent responsible (e.g., "v1.4 NER pipeline", "Agent 7")
- [ ] **Supporting Evidence**: Direct snippets, quotes, or links supporting entity
- [ ] **History of Changes**: Complete audit trail of updates, corrections, merges
- [ ] **Source Type/Quality**: Primary, secondary, curated, crowdsourced + trustworthiness score
- [ ] **Verification Status**: Pass/fail for rule-based, ML, or manual validation
- [ ] **Cross-Source Corroboration**: Count and list of independent sources
- [ ] **Reputation Score**: Trust score based on historical reliability of source
- [ ] **Recency of Evidence**: How recently supporting sources were updated
- [ ] **Semantic Consistency**: Alignment with schema rules and graph topology

**Neo4j Schema for Provenance**:
```cypher
// Emerging entity with full provenance
CREATE (e:Concept:EmergingEntity {
  name: "quantum neural networks",

  // PROVENANCE METADATA
  source_uri: "file:///papers/qnn_2025.pdf",
  source_document_id: "doc_qnn_12345",
  extraction_timestamp: "2025-10-01T10:30:00",
  extractor_identity: "spaCy_NER_v3.7_Agent_Concept_Extractor",
  supporting_evidence: "...enabling quantum neural networks to leverage entanglement for...",

  // TRUST SIGNALS
  verification_status: "pending_review",  // pending_review/validated/rejected
  corroboration_count: 3,  // 3 independent papers mention this
  corroborating_sources: ["doc_123", "doc_456", "doc_789"],
  source_quality: "peer_reviewed",  // peer_reviewed/preprint/blog/crowdsourced
  reputation_score: 0.85,  // 0.0-1.0 based on journal impact factor
  recency_score: 0.95,  // 0.0-1.0 (1.0 = published today)
  semantic_consistency_score: 0.78,  // 0.0-1.0 alignment with schema

  // TEMPORAL TRACKING
  first_seen: "2025-10-01T10:30:00",
  last_updated: "2025-10-01T14:20:00",
  mention_count: 3,
  emergence_score: 0.85,
  status: "emerging"
})

// Provenance: Link to extraction agent
CREATE (e)-[:EXTRACTED_BY {
  timestamp: "2025-10-01T10:30:00",
  confidence: 0.92,
  method: "transformer_NER"
}]->(agent:Agent {agent_id: "concept_extractor_v1"})

// Provenance: Link to source document
CREATE (e)-[:EXTRACTED_FROM {
  timestamp: "2025-10-01T10:30:00",
  context: "...quantum neural networks leverage...",
  page_number: 3,
  position_in_text: 1234
}]->(doc:Document {document_id: "doc_qnn_12345"})

// Trust: Cross-source corroboration
CREATE (e)-[:CORROBORATED_BY {
  timestamp: "2025-10-01T11:00:00",
  similarity: 0.92
}]->(doc2:Document {document_id: "doc_456"})

// Trust: Verification event
CREATE (e)-[:VERIFIED_BY {
  timestamp: "2025-10-01T15:00:00",
  verification_method: "human_review",
  outcome: "validated",
  reviewer: "domain_expert_alice"
}]->(verification:VerificationEvent)

// History: Change audit trail
CREATE (e)-[:HAS_CHANGE_HISTORY]->(change:ChangeEvent {
  change_id: "change_001",
  timestamp: "2025-10-01T14:20:00",
  change_type: "property_update",
  field: "reputation_score",
  old_value: 0.80,
  new_value: 0.85,
  reason: "journal impact factor updated",
  changed_by: "meta_analysis_agent"
})
```

**Implementation**:
```python
class ProvenanceTracker:
    """Track provenance and trust signals for emerging entities"""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def create_emerging_entity_with_provenance(
        self,
        entity_name: str,
        source_document: str,
        extractor_agent: str,
        evidence_snippet: str,
        source_quality: str = "unknown"
    ) -> str:
        """Create entity with full provenance metadata"""

        with self.driver.session() as session:
            result = session.run("""
                // Create emerging entity
                CREATE (e:Concept:EmergingEntity {
                    name: $entity_name,

                    // Provenance
                    source_uri: $source_uri,
                    source_document_id: $source_document,
                    extraction_timestamp: datetime(),
                    extractor_identity: $extractor_agent,
                    supporting_evidence: $evidence_snippet,

                    // Trust signals (initial)
                    verification_status: 'pending_review',
                    corroboration_count: 1,
                    corroborating_sources: [$source_document],
                    source_quality: $source_quality,
                    reputation_score: $initial_reputation,
                    recency_score: 1.0,  // Just extracted
                    semantic_consistency_score: 0.0,  // Not yet calculated

                    // Temporal
                    first_seen: datetime(),
                    last_updated: datetime(),
                    mention_count: 1,
                    status: 'emerging'
                })

                // Link to source document
                MATCH (doc:Document {document_id: $source_document})
                CREATE (e)-[:EXTRACTED_FROM {
                    timestamp: datetime(),
                    context: $evidence_snippet,
                    extraction_method: $extractor_agent
                }]->(doc)

                // Link to extractor agent
                MERGE (agent:Agent {agent_id: $extractor_agent})
                CREATE (e)-[:EXTRACTED_BY {
                    timestamp: datetime(),
                    confidence: $extraction_confidence
                }]->(agent)

                RETURN e.name as entity_name, id(e) as entity_id
            """,
                entity_name=entity_name,
                source_uri=f"file:///{source_document}",
                source_document=source_document,
                extractor_agent=extractor_agent,
                evidence_snippet=evidence_snippet,
                source_quality=source_quality,
                initial_reputation=self._calculate_initial_reputation(source_quality),
                extraction_confidence=0.85  # Could be dynamic based on NER score
            )

            entity_id = result.single()["entity_id"]
            logger.info(f"✅ Created emerging entity '{entity_name}' with full provenance")
            return entity_id

    def add_corroboration(self, entity_name: str, corroborating_document: str, similarity: float):
        """Add cross-source corroboration for entity"""

        with self.driver.session() as session:
            session.run("""
                MATCH (e:EmergingEntity {name: $entity_name})
                MATCH (doc:Document {document_id: $corroborating_document})

                // Create corroboration link
                CREATE (e)-[:CORROBORATED_BY {
                    timestamp: datetime(),
                    similarity: $similarity
                }]->(doc)

                // Update corroboration count and trust
                SET e.corroboration_count = e.corroboration_count + 1,
                    e.corroborating_sources = e.corroborating_sources + [$corroborating_document],
                    e.reputation_score = e.reputation_score + 0.05,  // Boost trust
                    e.last_updated = datetime()

                // Log change
                CREATE (e)-[:HAS_CHANGE_HISTORY]->(change:ChangeEvent {
                    change_id: randomUUID(),
                    timestamp: datetime(),
                    change_type: 'corroboration_added',
                    field: 'corroboration_count',
                    old_value: e.corroboration_count - 1,
                    new_value: e.corroboration_count,
                    reason: 'Cross-source validation',
                    changed_by: 'provenance_tracker'
                })
            """,
                entity_name=entity_name,
                corroborating_document=corroborating_document,
                similarity=similarity
            )

            logger.info(f"📊 Added corroboration for '{entity_name}' from {corroborating_document}")

    def verify_entity(self, entity_name: str, verification_method: str, outcome: str, reviewer: str = None):
        """Record verification event for entity"""

        with self.driver.session() as session:
            session.run("""
                MATCH (e:EmergingEntity {name: $entity_name})

                // Create verification event
                CREATE (verification:VerificationEvent {
                    verification_id: randomUUID(),
                    timestamp: datetime(),
                    method: $verification_method,
                    outcome: $outcome,
                    reviewer: $reviewer
                })

                CREATE (e)-[:VERIFIED_BY]->(verification)

                // Update verification status
                SET e.verification_status = $outcome,
                    e.last_updated = datetime()

                // If validated, promote to established entity
                WITH e
                WHERE $outcome = 'validated'
                SET e.status = 'established'
                REMOVE e:EmergingEntity
                SET e:EstablishedEntity
            """,
                entity_name=entity_name,
                verification_method=verification_method,
                outcome=outcome,
                reviewer=reviewer
            )

            logger.info(f"✅ Verified entity '{entity_name}': {outcome}")

    def _calculate_initial_reputation(self, source_quality: str) -> float:
        """Calculate initial reputation score based on source quality"""
        reputation_map = {
            "peer_reviewed": 0.9,
            "preprint": 0.7,
            "conference": 0.8,
            "blog": 0.5,
            "crowdsourced": 0.4,
            "unknown": 0.5
        }
        return reputation_map.get(source_quality, 0.5)

    def calculate_semantic_consistency(self, entity_name: str) -> float:
        """Calculate how well entity fits existing graph topology"""

        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:EmergingEntity {name: $entity_name})

                // Check schema compliance
                OPTIONAL MATCH (e)-[r]->()
                WITH e, collect(type(r)) as rel_types

                // Compare to expected relationship types for Concept nodes
                WITH e, rel_types,
                     ['EXTENDS', 'RELATED_TO', 'PART_OF', 'ENABLES'] as expected_types

                // Calculate overlap
                WITH e, rel_types, expected_types,
                     size([rt IN rel_types WHERE rt IN expected_types]) as overlap,
                     size(rel_types) as total_rels

                RETURN CASE
                    WHEN total_rels = 0 THEN 0.5
                    ELSE toFloat(overlap) / total_rels
                END as consistency_score
            """, entity_name=entity_name)

            score = result.single()["consistency_score"]

            # Update entity
            session.run("""
                MATCH (e:EmergingEntity {name: $entity_name})
                SET e.semantic_consistency_score = $score,
                    e.last_updated = datetime()
            """, entity_name=entity_name, score=score)

            return score
```

### FR7: Context and Co-occurrence Analysis
**Description**: Detect unusual term clusters that signal new multi-word entities or trends
**Research Source**: "Use word embeddings or context clustering to spot novel groups of terms appearing together"
**Acceptance Criteria**:
- [ ] Track which terms frequently co-occur in documents
- [ ] Detect unusual co-occurrence patterns (new combinations)
- [ ] Identify multi-word entities ("neural architecture search")
- [ ] Flag semantic drift (existing entity in new context)

**Implementation**:
```python
class CooccurrenceAnalyzer:
    """Detect unusual co-occurrence patterns indicating emergence"""

    def __init__(self, embedding_model):
        self.embeddings = embedding_model
        self.cooccurrence_matrix = defaultdict(Counter)

    def track_cooccurrence(self, concepts: List[str], document_id: str):
        """Track which concepts appear together in documents"""
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                # Bidirectional tracking
                self.cooccurrence_matrix[concept_a][concept_b] += 1
                self.cooccurrence_matrix[concept_b][concept_a] += 1

    def detect_unusual_cooccurrences(self, min_count: int = 3) -> List[Dict]:
        """Find concept pairs that suddenly start co-occurring"""
        unusual = []

        for concept_a, partners in self.cooccurrence_matrix.items():
            for concept_b, count in partners.items():
                # Check if this is a recent pattern (not historical)
                if count >= min_count and self._is_recent_pattern(concept_a, concept_b):
                    # Calculate semantic surprise
                    surprise = self._calculate_semantic_surprise(concept_a, concept_b)

                    if surprise > 0.7:
                        unusual.append({
                            "concept_a": concept_a,
                            "concept_b": concept_b,
                            "cooccurrence_count": count,
                            "semantic_surprise": surprise,
                            "pattern_type": "emerging_multi_word_entity"
                        })

        logger.info(f"🔍 Detected {len(unusual)} unusual co-occurrence patterns")
        return unusual

    def _calculate_semantic_surprise(self, concept_a: str, concept_b: str) -> float:
        """How semantically distant are these concepts?"""
        emb_a = self.embeddings.encode(concept_a)
        emb_b = self.embeddings.encode(concept_b)

        # Cosine distance (1 - similarity)
        similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        surprise = 1.0 - similarity

        return float(surprise)
```

## Technical Design

### System Architecture

```
Emergent Pattern Detection System:

┌─────────────────────────────────────────────────────────┐
│  Document Upload Stream                                 │
│  - Paper 1, Paper 2, ..., Paper N                       │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │  Emerging Entity Detector    │
    │  - NER extraction            │
    │  - Frequency tracking        │
    │  - Burst detection           │
    └────────┬─────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │  Neo4j Temporal Knowledge Graph      │
    │  - Entities with timestamps          │
    │  - Evolution relationships           │
    │  - Historical snapshots              │
    └────────┬─────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │  Co-occurrence Analyzer              │
    │  - Track term pairs                  │
    │  - Detect unusual combinations       │
    │  - Semantic surprise scoring         │
    └────────┬─────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │  Emergent Pattern Detector           │
    │  - Hub formation detection           │
    │  - Bridge node identification        │
    │  - Cross-domain pattern discovery    │
    └────────┬─────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │  Meta-Analysis Agent (Daily)         │
    │  - Analyze graph evolution           │
    │  - Detect agent collaboration        │
    │  - Calculate structural metrics      │
    │  - Update Cognition Base             │
    └────────┬─────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │  Emergent Pattern Notifications      │
    │  - Alert on new hub formation        │
    │  - Flag cross-domain connections     │
    │  - Recommend papers to ingest        │
    │  - Suggest research directions       │
    └──────────────────────────────────────┘
```

### Integration with Existing Specs

#### With Spec 028 (ThoughtSeeds)
```python
# Extended Pattern Emergence Agent
class PatternEmergenceAgent:
    def detect_emergent_patterns(self, basin_id: str) -> Optional[Dict]:
        # EXISTING: Detect 5+ ThoughtSeeds clustering
        pattern = super().detect_emergent_patterns(basin_id)

        # NEW: Temporal analysis of pattern formation
        if pattern:
            pattern["formation_timeline"] = self._analyze_formation_timeline(basin_id)
            pattern["emergence_score"] = self._calculate_emergence_score(basin_id)
            pattern["cross_domain"] = self._detect_cross_domain(basin_id)

        return pattern
```

#### With Spec 029 (Curiosity Agents)
```python
# Extended Curiosity Detection Agent
class CuriosityDetectionAgent:
    def detect_curiosity_triggers(self, concepts: List[str]) -> List[CuriosityTrigger]:
        triggers = super().detect_curiosity_triggers(concepts)

        # NEW: Boost priority for emerging entities
        emerging_entities = self.emerging_detector.detect_emerging_entities(concepts)
        for trigger in triggers:
            if trigger.concept in [e["entity"] for e in emerging_entities]:
                trigger.priority *= 1.5  # 50% boost for emerging concepts
                trigger.knowledge_gap_type = "emerging_entity"

        return triggers
```

## Test Strategy

### Unit Tests

```python
def test_emerging_entity_detection_burst():
    """Test burst detection identifies rapid mention increases"""
    detector = EmergingEntityDetector(ner_model, entity_index)

    # Simulate 5 mentions in 1 hour
    for i in range(5):
        text = "Research on quantum neural networks shows..."
        detector.detect_emerging_entities(text, time_window=3600)

    emerging = detector.get_emerging_entities()

    assert len(emerging) == 1
    assert emerging[0]["entity"] == "quantum neural networks"
    assert emerging[0]["mention_count"] == 5

def test_temporal_entity_tracking():
    """Test temporal versioning of entity evolution"""
    with driver.session() as session:
        # Create initial entity
        session.run("""
            CREATE (e:Concept {
                name: 'neural architecture search',
                first_seen: datetime('2025-01-01T00:00:00'),
                status: 'emerging'
            })
        """)

        # Evolve to new entity
        session.run("""
            MATCH (e1:Concept {name: 'neural architecture search'})
            CREATE (e2:Concept {
                name: 'differentiable NAS',
                first_seen: datetime('2025-03-01T00:00:00'),
                status: 'emerging'
            })
            CREATE (e1)-[:EVOLVED_INTO {
                transition_date: datetime('2025-03-01'),
                transition_type: 'refinement'
            }]->(e2)
        """)

        # Query evolution chain
        result = session.run("""
            MATCH path = (start:Concept)-[:EVOLVED_INTO*]->(end:Concept)
            WHERE start.name = 'neural architecture search'
            RETURN [node IN nodes(path) | node.name] as evolution_chain
        """)

        chain = result.single()["evolution_chain"]
        assert chain == ["neural architecture search", "differentiable NAS"]

def test_hub_formation_detection():
    """Test detection of emerging hub nodes"""
    detector = EmergentPatternDetector(driver)

    # Create concept with 25 relationships (hub threshold: 20)
    # ... create test data ...

    hubs = detector.detect_hub_formation(min_degree=20)

    assert len(hubs) > 0
    assert hubs[0]["degree"] >= 20

def test_meta_agent_collaboration_analysis():
    """Test meta-agent identifies successful agent pairs"""
    meta_agent = MetaAnalysisAgent(driver, cognition_base)

    # Create test data: 2 agents collaborating on 6 high-quality docs
    # ... create test data ...

    analysis = meta_agent.analyze_graph_evolution(days=7)

    assert len(analysis["agent_collaboration"]) > 0
    assert analysis["agent_collaboration"][0]["collaboration_count"] >= 5
    assert analysis["agent_collaboration"][0]["avg_quality"] > 0.8
```

### Integration Tests

```python
def test_end_to_end_emergent_entity_workflow():
    """Test complete workflow: paper upload → emerging entity → meta-analysis"""

    # Step 1: Upload papers with new concept
    for i in range(5):
        graph.process_document(
            content=f"Paper {i} discusses quantum neural networks...",
            filename=f"qnn_paper_{i}.pdf"
        )

    # Step 2: Verify emerging entity detected
    detector = EmergingEntityDetector(ner_model, entity_index)
    emerging = detector.get_emerging_entities()
    assert any(e["entity"] == "quantum neural networks" for e in emerging)

    # Step 3: Run meta-analysis
    meta_agent = MetaAnalysisAgent(driver, cognition_base)
    analysis = meta_agent.analyze_graph_evolution(days=1)

    # Step 4: Verify meta-analysis captured emergence
    assert "quantum neural networks" in analysis["emerging_entities"]

    # Step 5: Verify Cognition Base updated
    assert cognition_base.has_strategy("emerging_topics", "quantum neural networks")
```

## Implementation Plan

### Phase 1: Emerging Entity Detection (3-4 hours)
1. Implement `EmergingEntityDetector` with NER
2. Add frequency tracking and burst detection
3. Integrate with document processing pipeline
4. Test burst detection

### Phase 2: Temporal Knowledge Graph (4-5 hours)
1. Extend Neo4j schema with temporal fields
2. Add `first_seen`, `last_updated`, `deprecated_at` timestamps
3. Implement evolution relationships (EVOLVED_INTO)
4. Create historical snapshot mechanism
5. Test time-based queries

### Phase 3: Pattern Detection (4-5 hours)
1. Implement `EmergentPatternDetector`
2. Add hub formation detection
3. Add bridge node detection
4. Add cross-domain pattern discovery
5. Test structural metrics

### Phase 4: Co-occurrence Analysis (3-4 hours)
1. Implement `CooccurrenceAnalyzer`
2. Track concept pairs
3. Calculate semantic surprise
4. Test unusual pattern detection

### Phase 5: Meta-Agent (4-5 hours)
1. Implement `MetaAnalysisAgent`
2. Add daily analysis workflow
3. Integrate with Cognition Base
4. Add agent collaboration analysis
5. Test meta-level learnings

### Phase 6: Integration & Testing (3-4 hours)
1. Integrate with Specs 028, 029
2. End-to-end tests
3. Performance validation
4. Documentation

**Total Estimated Time**: 21-27 hours

## Success Criteria

- [ ] Emerging entities detected with burst analysis
- [ ] Temporal graph tracks entity evolution
- [ ] Hub and bridge nodes identified
- [ ] Cross-domain patterns discovered
- [ ] Meta-agent runs daily analysis
- [ ] Agent collaboration patterns detected
- [ ] Cognition Base updated with learnings
- [ ] All tests passing (unit + integration)

## References

- Research: Emergent Phenomena in Agentic Knowledge Graphs
- Research: Detecting Emerging Entities from Streaming Data
- Research: Keeping Knowledge Graphs Up-to-Date
- AgREE Framework: Iterative Agent Reasoning
- Neo4j Temporal Graphs
- Spec 028: ThoughtSeed Generation
- Spec 029: Curiosity-Driven Background Agents
