# Explainable AI Components for Narrative Map Extraction

**Feature Branch**: `014-narrative-map-extraction`  
**Created**: 2025-09-27  
**Status**: Research Implementation  
**Source Paper**: "Explainable AI Components for Narrative Map Extraction" by Brian Keith et al.

## Overview

Implementation specification for explainable AI components that extract narrative maps from document collections, providing multi-level explanations for document relationships, event connections, and high-level narrative structures. This system integrates with Flux CE consciousness processing to enhance narrative understanding and trust.

## Research Foundation

Based on peer-reviewed research from Text2Story'25 Workshop demonstrating:
- Multi-level explanation approach increases user trust (M=4.5/5.0)
- Connection explanations and important event detection build confidence
- Topical cluster explanations enhance document space understanding
- SHAP-based keyword explanations verify connection validity

## Core Requirements

### FR-001: Multi-Level Narrative Explanation System
- System MUST provide explanations at three abstraction levels:
  1. **Low-level**: Document embedding space and topical relationships
  2. **Connection-level**: Event relationship explanations with SHAP values
  3. **High-level**: Storyline naming and important event detection

### FR-002: Low-Level Space Explanations
- System MUST implement HDBSCAN clustering for document grouping
- System MUST generate keyword-based explanations using modified TF-IDF:
  ```
  S(t,c) = TF(t,c) · IDF_global(t) · IDF_local(t,c)
  ```
- System MUST provide 2D UMAP visualization of document relationships
- System MUST limit keywords per cluster to prevent information overload
- System MUST display interactive tooltips for cluster membership

### FR-003: Connection Explanation Framework
- System MUST classify connections into three types:
  - **Similarity-based**: Basic text similarity connections
  - **Entity-based**: Named entity overlap connections  
  - **Topical**: Clustering component driven connections
- System MUST implement SHAP-based explanations for connection strength
- System MUST provide entity overlap scoring using Jaccard similarity:
  ```
  overlap(e1,e2) = |tokens(e1) ∩ tokens(e2)| / |tokens(e1) ∪ tokens(e2)|
  ```
- System MUST explain why events are NOT connected (comparison functionality)

### FR-004: High-Level Structure Explanations
- System MUST implement automated storyline naming using extractive approach
- System MUST score storyline candidates using:
  ```
  Score(name) = α·C_entity + β·C_abstract + γ·C_coverage - δ·O_overlap
  ```
- System MUST detect important events using dual approach:
  - **Content importance**: Similarity to storyline centroid
  - **Structural importance**: Degree centrality with coherence weighting
- System MUST visually mark important events on narrative maps

### FR-005: Flux CE Integration Requirements
- System MUST integrate with existing consciousness processing pipeline
- System MUST leverage ThoughtSeed attractor basins for narrative coherence
- System MUST store narrative maps in Neo4j graph database
- System MUST link narrative explanations to vector embeddings in Qdrant
- System MUST provide consciousness-enhanced narrative extraction
- System MUST support curiosity-driven narrative gap detection

### FR-015: Archetypal and Internal Family Systems Integration
- System MUST integrate archetypal pattern recognition for universal narrative structures
- System MUST implement Internal Family Systems (IFS) pattern matching for user narratives
- System MUST identify archetypal roles within narrative structures (Hero, Mentor, Shadow, etc.)
- System MUST detect IFS parts dynamics in episodic memory events (Protector, Exile, Firefighter patterns)
- System MUST enable isomorphic metaphor detection across different narrative domains
- System MUST support pattern evolution through archetypal transformation tracking
- System MUST correlate user episodic memories with archetypal narrative patterns
- System MUST provide therapeutic insights through IFS parts identification in stories

### FR-006: Trust Building and Transparency
- System MUST provide warning mechanisms for unreliable explanations
- System MUST maintain explanation consistency across system updates
- System MUST prevent information overload through adaptive explanation depth
- System MUST support user feedback on explanation quality
- System MUST track explanation effectiveness metrics

### FR-007: Scalable Processing Architecture
- System MUST handle datasets up to 10,000 documents efficiently
- System MUST implement hierarchical explanation strategies for large datasets
- System MUST provide real-time explanation generation (<5 seconds)
- System MUST support incremental narrative map updates
- System MUST cache explanation components for performance

## Technical Implementation Specifications

### Component Architecture

```python
# /Volumes/Asylum/dev/Dionysus-2.0/backend/services/narrative_extraction/

class NarrativeMapExtractor:
    """Main orchestrator for narrative map extraction with XAI"""
    
    def __init__(self):
        self.low_level_explainer = LowLevelSpaceExplainer()
        self.connection_explainer = ConnectionExplainer()
        self.high_level_explainer = HighLevelStructureExplainer()
        self.consciousness_integrator = ConsciousnessIntegrator()
        
    async def extract_narrative_map(self, documents: List[Document]) -> NarrativeMap:
        """Extract narrative map with multi-level explanations"""
        pass

class LowLevelSpaceExplainer:
    """Document space clustering and topical explanations"""
    
    def cluster_documents(self, embeddings: np.ndarray) -> ClusterResult:
        """HDBSCAN clustering with soft assignment"""
        pass
        
    def generate_cluster_keywords(self, cluster: Cluster) -> List[Keyword]:
        """Modified TF-IDF keyword extraction"""
        pass
        
    def create_umap_visualization(self, embeddings: np.ndarray) -> Visualization:
        """2D projection for document relationships"""
        pass

class ConnectionExplainer:
    """Event connection explanations with SHAP"""
    
    def classify_connection_type(self, event1: Event, event2: Event) -> ConnectionType:
        """Classify as similarity, entity, or topical connection"""
        pass
        
    def generate_shap_explanation(self, connection: Connection) -> SHAPExplanation:
        """SHAP-based keyword contribution analysis"""
        pass
        
    def compute_entity_overlap(self, event1: Event, event2: Event) -> float:
        """Jaccard similarity for entity overlap"""
        pass

class HighLevelStructureExplainer:
    """Storyline naming and important event detection"""
    
    def extract_storyline_names(self, storyline: Storyline) -> List[CandidateName]:
        """Extractive storyline naming with scoring"""
        pass
        
    def detect_important_events(self, narrative_map: NarrativeMap) -> List[ImportantEvent]:
        """Content and structural importance detection"""
        pass
```

### Database Schema Extensions

```sql
-- Neo4j Cypher for narrative map storage
CREATE (n:NarrativeMap {
    id: $map_id,
    extraction_timestamp: datetime(),
    explanation_version: $version
})

CREATE (e:Event {
    id: $event_id,
    title: $title,
    content: $content,
    timestamp: $timestamp,
    importance_score_content: $content_importance,
    importance_score_structural: $structural_importance
})

CREATE (s:Storyline {
    id: $storyline_id,
    name: $storyline_name,
    confidence: $name_confidence,
    coverage: $content_coverage
})

CREATE (c:TopicalCluster {
    id: $cluster_id,
    keywords: $keywords,
    cluster_size: $size,
    coherence_score: $coherence
})

CREATE (e1)-[:CONNECTED_TO {
    connection_type: $type,
    confidence: $confidence,
    shap_explanation: $shap_data,
    entity_overlap: $overlap_score
}]->(e2)

CREATE (e)-[:BELONGS_TO]->(s)
CREATE (e)-[:IN_CLUSTER]->(c)
```

### API Specifications

```python
# /Volumes/Asylum/dev/Dionysus-2.0/backend/api/narrative_extraction.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class NarrativeExtractionRequest(BaseModel):
    documents: List[str]  # Document IDs or content
    map_size: Optional[int] = 100
    temporal_sensitivity: Optional[float] = 0.5
    explanation_depth: Optional[str] = "full"  # "basic", "standard", "full"
    consciousness_enhancement: Optional[bool] = True

class ConnectionExplanation(BaseModel):
    connection_type: str  # "similarity", "entity", "topical"
    confidence: float
    keywords_positive: List[Dict[str, float]]
    keywords_negative: List[Dict[str, float]]
    entity_overlap: Optional[List[str]] = []
    shared_topics: Optional[List[str]] = []

class EventImportance(BaseModel):
    event_id: str
    content_importance: float
    structural_importance: float
    explanation: str
    is_highlighted: bool

class NarrativeMapResponse(BaseModel):
    map_id: str
    storylines: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    topical_clusters: List[Dict[str, Any]]
    important_events: List[EventImportance]
    extraction_metadata: Dict[str, Any]

@app.post("/narrative/extract", response_model=NarrativeMapResponse)
async def extract_narrative_map(request: NarrativeExtractionRequest):
    """Extract narrative map with explanations"""
    pass

@app.get("/narrative/{map_id}/explanations/connection/{connection_id}")
async def get_connection_explanation(map_id: str, connection_id: str) -> ConnectionExplanation:
    """Get detailed explanation for specific connection"""
    pass

@app.get("/narrative/{map_id}/explanations/clusters")
async def get_cluster_explanations(map_id: str) -> List[Dict[str, Any]]:
    """Get topical cluster explanations"""
    pass

@app.post("/narrative/{map_id}/compare_events")
async def compare_events(map_id: str, event1_id: str, event2_id: str) -> Dict[str, Any]:
    """Explain why two events are or aren't connected"""
    pass
```

### Integration with Consciousness System

```python
# /Volumes/Asylum/dev/Dionysus-2.0/backend/services/consciousness_narrative_bridge.py

class ConsciousnessNarrativeBridge:
    """Bridge between consciousness system and narrative extraction"""
    
    async def enhance_narrative_with_consciousness(
        self, 
        narrative_map: NarrativeMap, 
        consciousness_level: str
    ) -> EnhancedNarrativeMap:
        """Enhance narrative map with consciousness insights"""
        
        # Apply ThoughtSeed attractor basin analysis
        attractor_insights = await self.analyze_attractor_basins(narrative_map)
        
        # Integrate with curiosity gap detection
        curiosity_gaps = await self.detect_narrative_gaps(narrative_map)
        
        # Apply consciousness-guided event importance scoring
        consciousness_importance = await self.score_consciousness_importance(narrative_map)
        
        return EnhancedNarrativeMap(
            base_map=narrative_map,
            attractor_insights=attractor_insights,
            curiosity_gaps=curiosity_gaps,
            consciousness_scores=consciousness_importance
        )
    
    async def trigger_curiosity_for_gaps(self, gaps: List[NarrativeGap]) -> List[CuriosityMission]:
        """Trigger curiosity missions for identified narrative gaps"""
        pass
```

## User Interface Requirements

### FR-008: Interactive Narrative Map Visualization
- System MUST provide interactive graph visualization of narrative map
- System MUST display explanation tooltips on hover
- System MUST highlight important events with visual emphasis
- System MUST show connection types with color coding
- System MUST provide cluster overlay toggle for document space view

### FR-009: Explanation Dashboard
- System MUST provide multi-panel explanation interface
- System MUST show storyline names with confidence scores
- System MUST display topical clusters with keyword lists
- System MUST provide connection analysis panel with SHAP explanations
- System MUST enable event comparison functionality

### FR-010: Trust Indicators
- System MUST display confidence scores for all explanations
- System MUST provide explanation reliability warnings
- System MUST show processing status and completion indicators
- System MUST enable user feedback collection on explanation quality

## Agent Delegation Tasks

### Task 1: Core Narrative Extraction Agent
**Specialization**: Narrative analysis and graph construction  
**Deliverables**:
- Implement HDBSCAN clustering for document grouping
- Create modified TF-IDF keyword extraction
- Build narrative map graph structure
- Integrate with consciousness enhancement system

### Task 2: Connection Explanation Agent
**Specialization**: Relationship analysis and SHAP explanations  
**Deliverables**:
- Implement connection type classification
- Build SHAP-based explanation generation
- Create entity overlap scoring system
- Develop event comparison functionality

### Task 3: High-Level Structure Agent
**Specialization**: Storyline analysis and important event detection  
**Deliverables**:
- Implement extractive storyline naming
- Build important event detection (content + structural)
- Create storyline confidence scoring
- Develop visual emphasis for important events

### Task 4: Visualization and Interface Agent
**Specialization**: Interactive visualization and user experience  
**Deliverables**:
- Create interactive narrative map visualization
- Build explanation dashboard interface
- Implement trust indicators and confidence displays
- Create user feedback collection system

### Task 5: Integration and Testing Agent
**Specialization**: System integration and validation  
**Deliverables**:
- Integrate all components with Flux CE system
- Create comprehensive testing suite
- Validate explanation accuracy and consistency
- Optimize performance for large datasets

## Success Metrics

### User Trust Metrics (Based on Research)
- Overall explanation usefulness: >4.0/5.0
- Trust increase through explanations: >4.0/5.0
- Connection explanation usefulness: >4.0/5.0
- Important event relevance: >4.0/5.0

### Technical Performance Metrics
- Narrative extraction time: <30 seconds for 200 documents
- Explanation generation time: <5 seconds
- Memory usage: <8GB for 1000 documents
- Accuracy of connection classifications: >85%

### Integration Metrics
- Consciousness enhancement effectiveness: >80% improvement
- Curiosity gap detection accuracy: >90%
- Narrative-to-vector database linking: >99% success
- API response time: <2 seconds

## Implementation Priority

### Phase 1: Core Implementation (Week 1-2)
1. Implement basic narrative map extraction
2. Create HDBSCAN clustering with keyword extraction
3. Build connection classification system
4. Set up database schema for narrative storage

### Phase 2: Explanation Systems (Week 3-4)
1. Implement SHAP-based connection explanations
2. Create storyline naming system
3. Build important event detection
4. Develop explanation confidence scoring

### Phase 3: Consciousness Integration (Week 5-6)
1. Integrate with ThoughtSeed attractor basins
2. Implement consciousness-enhanced importance scoring
3. Create curiosity gap detection for narratives
4. Build consciousness-narrative bridge

### Phase 4: Interface and Optimization (Week 7-8)
1. Create interactive visualization interface
2. Build explanation dashboard
3. Implement trust indicators and user feedback
4. Optimize performance and conduct user testing

## Dependencies

- HDBSCAN clustering library
- SHAP explainability framework
- UMAP dimensionality reduction
- spaCy for named entity recognition
- NetworkX for graph analysis
- Consciousness enhancement system
- Neo4j graph database
- Qdrant vector database

## Notes

- Research validates multi-level explanation approach for narrative analysis
- Focus on preventing information overload through adaptive explanation depth
- Integrate with existing Flux CE consciousness and curiosity systems
- Maintain explanation consistency and reliability through validation
- Support both extractive and future abstractive storyline naming approaches