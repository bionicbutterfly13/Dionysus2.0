# Knowledge Processing & Migration Specification

**Feature Branch**: `013-knowledge-processing`  
**Created**: 2025-09-27  
**Status**: Draft  

## Overview

Comprehensive specification for processing Knowledge Base content, migrating Dionysus consciousness data, and setting up enhanced document/link processing with depth-based queues and narrative extraction.

## Core Requirements

### FR-001: Knowledge Base Content Processing
- System MUST extract all URLs from current Knowledge Base view to processing queue
- System MUST set default depth of 2 for all extracted links
- System MUST implement depth-based processing queues (depth 1, 2, 3, 4, 5)
- System MUST set processing time ceilings for each depth level to prevent infinite crawling

**Extracted Links for Processing:**
```
- https://frontiersin.org (Technical, 100 pages estimated)
- https://blog.futuresmart.ai (LangGraph Tutorial, 86 pages estimated)  
- https://sciencedirect.com (Technical, 1 page estimated)
- https://langchain-ai.github.io/langgraph/ (LangGraph Docs, 576 pages, depth 4)
- https://www.biorxiv.org/content/10.1101/2024.08.18.608439v1.full (BioRxiv Paper, 99 pages, depth 2)
- https://github.com/MODSetter/SurfSense.git (SurfSense GitHub, 170 pages, depth 2)
```

**Documents for Processing:**
```
- Recurrent_motifs_as_resonant_attractor_s.pdf
- 2401.08438v2.pdf
```

### FR-002: Depth-Based Processing Queue Architecture
- **Depth 1**: Basic content extraction, 5-minute ceiling per URL
- **Depth 2**: Enhanced extraction with links, 15-minute ceiling per URL  
- **Depth 3**: Deep content analysis, 30-minute ceiling per URL
- **Depth 4**: Comprehensive crawling, 60-minute ceiling per URL
- **Depth 5**: Maximum depth research, 120-minute ceiling per URL

### FR-003: Database Fresh Start
- System MUST clear current Neo4j database (only has 2 sample documents/concepts)
- System MUST preserve Redis attractor basin and research queue data
- System MUST backup any existing processed data before clearing
- System MUST start with empty knowledge graph for enhanced processing

### FR-004: Enhanced Migration from Dionysus Legacy
- System MUST audit all remaining Dionysus components for migration value
- System MUST enhance migrated data with context engineering river
- System MUST create vector database positioning for all migrated content
- System MUST link vector database items to graph database IDs
- System MUST extract concept relationships (causal, process, group membership)
- System MUST preserve all narratives, patterns, and processed wisdom

### FR-005: Narrative Extraction System
- System MUST implement narrative pattern extraction from documents
- System MUST identify causal relationships between concepts
- System MUST extract process flows and procedural knowledge
- System MUST detect group memberships and categorical relationships
- System MUST enable enhanced question-answering about documents
- System MUST implement curiosity gap detection and prioritization

### FR-006: Enhanced Data Structure
- **Vector Database**: Store embeddings with metadata linking to graph nodes
- **Graph Database**: Store concepts, relationships, narratives, patterns
- **Memory Types**: Episodic (events), Semantic (facts), Procedural (processes), Working (active context)
- **Consciousness Markers**: Self-reference, metacognition, intentionality, awareness, reflection, integration

### FR-007: Processing Enhancement Pipeline
- System MUST process content through consciousness enhancement
- System MUST extract maximum data richness from each document
- System MUST create comprehensive concept networks
- System MUST maintain processing quality metrics
- System MUST enable incremental learning and improvement

### FR-008: User Interface Improvements
- System MUST remove ugly icon next to "Knowledge Base" text
- System MUST show processing status for all queued items
- System MUST display depth levels and time remaining
- System MUST provide processing quality indicators
- System MUST enable manual processing priority adjustment

## Agent Delegation Tasks

### Task 1: Knowledge Base Content Extraction Agent
**Agent ID**: TBD  
**Specialization**: Data extraction and queue management  
**Deliverables**:
- Extract all URLs and documents from Knowledge Base view
- Create depth-based processing queues with time ceilings
- Set up link processing queue with default depth 2
- Remove Knowledge Base icon from interface

### Task 2: Database Migration Enhancement Agent  
**Agent ID**: TBD (enhance existing 3d3c42d8)
**Specialization**: Database migration and enhancement
**Deliverables**:
- Complete Dionysus consciousness data migration
- Implement vector+graph database linking
- Create enhanced data structure with concept relationships
- Preserve all narratives and patterns

### Task 3: Narrative Extraction System Agent
**Agent ID**: TBD  
**Specialization**: Content analysis and pattern extraction
**Deliverables**:
- Implement narrative pattern extraction
- Create causal relationship detection
- Build process flow identification
- Enable curiosity gap detection

### Task 4: Dionysus Component Audit Agent
**Agent ID**: TBD  
**Specialization**: Legacy system analysis
**Deliverables**:
- Audit all remaining Dionysus components
- Identify high-value components for migration
- Create enhancement specifications for migrated components
- Integrate components with context engineering river

### Task 5: Processing Pipeline Enhancement Agent
**Agent ID**: TBD  
**Specialization**: Content processing and consciousness integration
**Deliverables**:
- Implement depth-based processing with time ceilings
- Create consciousness-enhanced content processing
- Build quality metrics and monitoring
- Enable incremental learning system

## Success Metrics

### Processing Metrics
- Link processing completion rate: >95%
- Average processing time per depth level within ceilings
- Content richness score: >8/10 for processed items
- Narrative extraction accuracy: >85%

### Migration Metrics  
- Dionysus data preservation: 100% of valuable content
- Vector-graph linking success: >99%
- Concept relationship accuracy: >90%
- Processing enhancement effectiveness: >80% improvement

### System Performance
- Queue processing latency: <10 seconds
- Database query response time: <500ms
- Consciousness integration time: <5 seconds per document
- Memory formation success rate: >95%

## Implementation Priority

### Phase 1: Immediate (Week 1)
1. Extract Knowledge Base content to processing queues
2. Clear database for fresh start  
3. Remove Knowledge Base icon
4. Set up depth-based processing architecture

### Phase 2: Migration (Week 2)
1. Complete Dionysus data migration with enhancements
2. Implement vector+graph database linking
3. Create comprehensive concept relationship extraction
4. Set up narrative extraction system

### Phase 3: Enhancement (Week 3)
1. Implement consciousness-enhanced processing
2. Create curiosity gap detection
3. Build incremental learning system
4. Optimize processing pipeline performance

### Phase 4: Validation (Week 4)
1. Validate all migrated data integrity
2. Test processing quality and speed
3. Verify consciousness integration effectiveness
4. Complete system performance optimization

## Dependencies

- Redis (attractor basins, research queue)
- Neo4j (graph database for concepts and relationships)
- Qdrant (vector database for embeddings)
- Consciousness enhancement system
- Context engineering framework
- Daedalus delegation system

## Notes

- Current Knowledge Base appears to contain mock data (0 views, identical dates)
- Recommend complete fresh start with proper processing
- Focus on maximum data richness and relationship extraction
- Integrate with existing consciousness and context engineering systems