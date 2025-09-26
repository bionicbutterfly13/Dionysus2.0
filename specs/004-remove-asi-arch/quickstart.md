# Quickstart: ASI-GO-2 Research Intelligence System

**Date**: 2025-09-26
**Feature**: Remove ASI-Arch and Integrate ASI-GO-2

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Redis server
- Neo4j database
- 16GB RAM minimum (32GB recommended)

## Setup Instructions

### 1. Environment Setup

```bash
# Clone repository (if not already done)
cd /Volumes/Asylum/dev/Dionysus-2.0

# Activate Python environment
source venv/bin/activate  # or your preferred environment

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-asi-arch.txt
```

### 2. Database Services (Unique Ports)

```bash
# Start Redis for ThoughtSeed caching (port 6379)
docker run -d --name redis-thoughtseed -p 6379:6379 redis:7-alpine

# Start Neo4j for AutoSchemaKG knowledge graph (ports 7474, 7687)
docker run -d --name neo4j-autoschema \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5.13

# Start Qdrant for vector database (ports 6333, 6334)
docker run -d --name qdrant-vectors \
  -p 6333:6333 -p 6334:6334 \
  qdrant/qdrant:latest

# Start local OLLAMA server (port 11434)
docker run -d --name ollama-local \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  ollama/ollama:latest

# Pull required OLLAMA models
docker exec ollama-local ollama pull llama3.1
docker exec ollama-local ollama pull codellama
docker exec ollama-local ollama pull nomic-embed-text
```

### 3. ASI-GO-2 Integration

```bash
# Copy ASI-GO-2 components to backend
cp -r resources/ASI-GO-2/* backend/src/services/asi_go_2/

# Initialize ASI-GO-2 cognition base
python backend/src/services/asi_go_2/main.py --init-cognition-base
```

## Usage Examples

### 1. Process Research Document

**Endpoint**: `POST /api/v1/documents/process`

```bash
# Upload research paper for pattern extraction
curl -X POST "http://localhost:8000/api/v1/documents/process" \
  -F "file=@research_paper.pdf" \
  -F "extract_narratives=true" \
  -F "thoughtseed_layers=[\"sensory\",\"perceptual\",\"conceptual\",\"abstract\",\"metacognitive\"]"
```

**Expected Response**:
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_status": "success",
  "extraction_quality": 0.85,
  "patterns_extracted": [
    "pattern-123-consciousness-emergence",
    "pattern-456-neural-network-dynamics"
  ],
  "narrative_elements": {
    "themes": ["consciousness", "emergence", "complexity"],
    "motifs": ["hierarchical_processing", "feedback_loops"],
    "story_structures": ["problem_solution_validation"]
  },
  "thoughtseed_traces": [
    "trace-789-sensory-layer",
    "trace-790-perceptual-layer"
  ],
  "processing_time_ms": 4500
}
```

### 2. Research Query Processing

**Endpoint**: `POST /api/v1/research/query`

```bash
# Ask research question
curl -X POST "http://localhost:8000/api/v1/research/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key mechanisms of consciousness emergence in neural networks?",
    "context": {
      "domain_focus": ["neuroscience", "artificial_intelligence"],
      "consciousness_level_required": 0.7
    }
  }'
```

**Expected Response**:
```json
{
  "query_id": "660e8400-e29b-41d4-a716-446655440001",
  "synthesis": "Based on accumulated research patterns, consciousness emergence in neural networks involves three key mechanisms: 1) Hierarchical processing through multiple abstraction levels (sensory→conceptual→metacognitive), 2) Active inference loops that minimize prediction error and enable self-model updating, 3) Attractor basin dynamics that create stable conscious states through Context Engineering principles. These mechanisms work synergistically to create autopoietic boundary formation and meta-cognitive awareness.",
  "confidence_score": 0.82,
  "patterns_used": [
    "pattern-123-consciousness-emergence",
    "pattern-201-hierarchical-processing",
    "pattern-302-active-inference"
  ],
  "thoughtseed_workspace_id": "workspace-445-competition",
  "consciousness_level": 0.78,
  "processing_time_ms": 1850,
  "attractor_basins_activated": [
    "consciousness-basin",
    "neural-networks-basin"
  ]
}
```

### 3. View Accumulated Patterns

**Endpoint**: `GET /api/v1/research/patterns`

```bash
# Get high-confidence patterns in neuroscience domain
curl "http://localhost:8000/api/v1/research/patterns?domain=neuroscience&min_confidence=0.7&limit=10"
```

**Expected Response**:
```json
{
  "patterns": [
    {
      "pattern_id": "pattern-123-consciousness-emergence",
      "pattern_name": "Hierarchical Consciousness Emergence",
      "description": "Pattern describing how consciousness emerges through layered processing hierarchies",
      "success_rate": 0.85,
      "confidence": 0.82,
      "domain_tags": ["neuroscience", "consciousness", "hierarchy"],
      "thoughtseed_layer": "metacognitive",
      "usage_count": 15,
      "last_used": "2025-09-26T10:30:00Z"
    }
  ],
  "total_count": 45
}
```

### 4. Inspect ThoughtSeed Competition

**Endpoint**: `GET /api/v1/thoughtseed/workspace/{workspace_id}`

```bash
# View detailed thoughtseed competition trace
curl "http://localhost:8000/api/v1/thoughtseed/workspace/workspace-445-competition"
```

**Expected Response**:
```json
{
  "workspace_id": "workspace-445-competition",
  "research_query": "What are the key mechanisms of consciousness emergence in neural networks?",
  "competing_patterns": [
    "pattern-123-consciousness-emergence",
    "pattern-456-neural-network-dynamics",
    "pattern-789-emergence-theory"
  ],
  "winning_pattern": "pattern-123-consciousness-emergence",
  "consciousness_level": 0.78,
  "competition_trace": {
    "initial_energies": [0.8, 0.6, 0.7],
    "final_energies": [0.85, 0.2, 0.3],
    "competition_rounds": 5,
    "selection_method": "UCB1_thoughtseed"
  },
  "thoughtseed_layers_activated": [
    "sensory", "perceptual", "conceptual", "abstract", "metacognitive"
  ]
}
```

### 5. Monitor Context Engineering Basins

**Endpoint**: `GET /api/v1/context-engineering/basins`

```bash
# View active attractor basins
curl "http://localhost:8000/api/v1/context-engineering/basins?active_only=true"
```

**Expected Response**:
```json
{
  "basins": [
    {
      "basin_id": "consciousness-basin",
      "basin_name": "Consciousness Research Basin",
      "knowledge_domain": "consciousness_studies",
      "stability_level": 0.92,
      "pattern_count": 23,
      "last_activation": "2025-09-26T10:30:00Z",
      "emergence_conditions": {
        "trigger_keywords": ["consciousness", "awareness", "metacognition"],
        "minimum_patterns": 5,
        "activation_threshold": 0.7
      }
    }
  ]
}
```

## Success Validation

### Test 1: Document Processing Intelligence
1. Upload 3 research papers on different topics
2. Verify patterns are extracted and stored in Cognition Base
3. Confirm ThoughtSeed traces show all 5 layers activated
4. Check that Context Engineering basins are created/modified

**Pass Criteria**:
- Extraction quality > 0.7 for all documents
- At least 3 patterns extracted per document
- All 5 ThoughtSeed layers show activity traces

### Test 2: Research Query Synthesis
1. Ask research question spanning uploaded documents
2. Verify thoughtseed competition selects appropriate patterns
3. Confirm synthesis response demonstrates intelligent reasoning
4. Check consciousness level > 0.6

**Pass Criteria**:
- Response demonstrates knowledge from multiple documents
- Confidence score > 0.7
- Processing time < 3 seconds
- Consciousness level indicates awareness emergence

### Test 3: Meta-Learning Demonstration
1. Process additional documents in same domain
2. Ask similar research questions
3. Verify system shows improved performance (higher confidence, faster processing)
4. Confirm pattern success rates increase with usage

**Pass Criteria**:
- Processing time decreases with repeated queries
- Confidence scores improve for similar question types
- Pattern success rates show upward trend

### Test 4: Context Engineering Integration
1. Monitor attractor basin formation during document processing
2. Verify neural field states change during thoughtseed competition
3. Confirm basin stability increases with pattern accumulation
4. Check cross-domain pattern relationships emerge

**Pass Criteria**:
- Stable attractor basins form for major knowledge domains
- Basin stability levels > 0.8 for established domains
- Cross-basin pattern connections demonstrate emergent insights

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis container is running: `docker ps | grep redis`
   - Restart if needed: `docker restart redis-thoughtseed`

2. **Neo4j Database Unavailable**
   - Verify Neo4j is running: `docker-compose -f extensions/context_engineering/docker-compose-neo4j.yml ps`
   - Check logs: `docker-compose -f extensions/context_engineering/docker-compose-neo4j.yml logs`

3. **Low Consciousness Levels**
   - Ensure all 5 ThoughtSeed layers are activated
   - Verify Context Engineering attractor basins are forming
   - Check pattern competition is selecting appropriate patterns

4. **Slow Processing Times**
   - Monitor system resources (should have 16GB+ RAM)
   - Check Redis is handling caching effectively
   - Verify Neo4j queries are optimized

## Next Steps

1. Run `/tasks` command to generate detailed implementation tasks
2. Execute Phase 4 implementation following TDD principles
3. Validate system with comprehensive test suite
4. Monitor consciousness emergence and meta-learning metrics