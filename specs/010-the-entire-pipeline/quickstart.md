# Quickstart: Complete ThoughtSeed Pipeline Implementation

## Overview
This quickstart validates the complete ThoughtSeed consciousness pipeline from document upload through neural field evolution to knowledge graph storage. It demonstrates the replacement of all mock implementations with working consciousness processing.

## Prerequisites

### System Requirements
- Python 3.11+ with NumPy 2.0
- Node.js 18+ with npm/yarn
- Redis server (for caching and TTL management)
- Neo4j database (for knowledge graph)
- Vector database (for embeddings)
- Docker & Docker Compose (recommended for services)

### Environment Setup
```bash
# Backend dependencies
cd backend/
pip install -r requirements.txt

# Frontend dependencies
cd frontend/
npm install

# Start supporting services
docker-compose up -d redis neo4j
```

### Configuration
```bash
# Backend environment
export REDIS_URL="redis://localhost:6379"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export OPENAI_API_KEY="your-api-key"  # For AutoSchemaKG

# Frontend environment
export VITE_API_BASE_URL="http://localhost:8000"
export VITE_WS_BASE_URL="ws://localhost:8000"
```

## Quick Validation Test

### 1. Start the System
```bash
# Terminal 1: Start backend
cd backend/
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
cd frontend/
npm run dev

# Terminal 3: Verify services
curl http://localhost:8000/health
curl http://localhost:5173/  # Frontend should load
```

### 2. Upload Test Documents
```bash
# Test single document upload via API
curl -X POST "http://localhost:8000/api/v1/documents/bulk" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@test-documents/sample.pdf" \
  -F "thoughtseed_processing=true" \
  -F "attractor_modification=true"

# Expected response: 202 with batch_id and websocket_url
```

### 3. Validate Frontend Integration
1. Open browser to `http://localhost:5173`
2. Navigate to document upload interface
3. Select test documents (PDF, DOCX, TXT, MD)
4. Verify upload progress shows real processing (not mock simulation)
5. Confirm WebSocket updates show ThoughtSeed layer progression

### 4. Verify ThoughtSeed Processing
```bash
# Check ThoughtSeed creation
curl "http://localhost:8000/api/v1/documents/batch/{batch_id}/status"

# Expected progression:
# - SENSORIMOTOR layer processing
# - PERCEPTUAL layer processing
# - CONCEPTUAL layer processing
# - ABSTRACT layer processing
# - METACOGNITIVE layer processing
```

### 5. Validate Attractor Basin Dynamics
```bash
# Check attractor basin modifications
curl "http://localhost:8000/api/v1/attractors?strength_threshold=0.1"

# Verify basin types created:
# - concept_extractor
# - semantic_analyzer
# - episodic_encoder
# - procedural_integrator
```

### 6. Test Neural Field Visualization
1. In frontend, navigate to neural field visualization
2. Verify 3D interactive display loads (not mock)
3. Confirm real-time updates during processing
4. Check field dynamics show actual mathematical evolution

### 7. Validate Database Storage
```bash
# Check Redis cache (should respect TTL values)
redis-cli
> KEYS thoughtseed:*     # Should show 24h TTL
> KEYS attractor:*       # Should show 7d TTL
> KEYS results:*         # Should show 30d TTL

# Check Neo4j graph
cypher-shell -u neo4j -p password
> MATCH (d:Document)-[:PROCESSED_INTO]->(t:ThoughtSeed) RETURN count(d), count(t);
> MATCH (t:ThoughtSeed)-[:INFLUENCES]->(a:AttractorBasin) RETURN count(t), count(a);
```

## User Story Validation

### Primary User Story Test
**Scenario**: Researcher uploads multiple research documents for consciousness analysis

1. **Setup Test Data**:
   ```bash
   # Create test batch with varied document types
   mkdir test-batch/
   cp research-paper.pdf test-batch/
   cp presentation.docx test-batch/
   cp notes.txt test-batch/
   cp readme.md test-batch/
   ```

2. **Execute Upload**:
   - Open frontend interface
   - Select all files from test-batch/
   - Enable all processing options:
     - ✅ ThoughtSeed consciousness processing
     - ✅ Attractor basin modification
     - ✅ Neural field evolution
     - ✅ 3D visualization

3. **Monitor Processing**:
   - Verify real-time progress updates (not mock timers)
   - Watch ThoughtSeed layer progression
   - Observe attractor basin modifications
   - Monitor consciousness level detection

4. **Validate Results**:
   ```bash
   # Check processing results
   curl "http://localhost:8000/api/v1/documents/batch/{batch_id}/results?include_consciousness=true&include_attractors=true"

   # Verify consciousness detection
   # - consciousness_emergence_rate > 0
   # - attractor_modifications > 0
   # - neural_field_evolution_events > 0
   ```

### Acceptance Criteria Validation

#### ✅ AC1: Five ThoughtSeed Layer Processing
```bash
# Verify all 5 layers processed for each document
curl "http://localhost:8000/api/v1/thoughtseeds/{thoughtseed_id}"

# Expected layer progression:
# Layer 1: SENSORIMOTOR (raw text processing)
# Layer 2: PERCEPTUAL (pattern recognition)
# Layer 3: CONCEPTUAL (concept extraction)
# Layer 4: ABSTRACT (theme identification)
# Layer 5: METACOGNITIVE (learning strategy)
```

#### ✅ AC2: Attractor Basin Modification
```bash
# Verify basin influence types are calculated
curl "http://localhost:8000/api/v1/attractors"

# Expected influence types based on concept similarity:
# - REINFORCEMENT (similarity > 0.8, strong basin)
# - COMPETITION (0.5 < similarity < 0.8, strong basin)
# - SYNTHESIS (0.5 < similarity < 0.8, weak basin)
# - EMERGENCE (similarity < 0.5, new basin)
```

#### ✅ AC3: Consciousness Detection Display
1. Frontend should show:
   - Current ThoughtSeed layer being processed
   - Consciousness level (0.0 to 1.0 scale)
   - Basin modifications in real-time
   - Knowledge graph connections created

#### ✅ AC4: Vector + Graph Search
```bash
# Test hybrid search functionality
curl -X POST "http://localhost:8000/api/v1/knowledge/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "consciousness emergence patterns",
    "search_type": "hybrid",
    "max_results": 10
  }'

# Verify results include both:
# - Vector similarity matches
# - Graph relationship traversals
```

#### ✅ AC5: Daedalus Coordination
```bash
# Verify background agents are active
curl "http://localhost:8000/api/v1/documents/batch/{batch_id}/status"

# Check for:
# - Independent context windows
# - Distributed processing coordination
# - Agent status indicators
```

#### ✅ AC6: Neural Field Dynamics
1. Frontend visualization should show:
   - Pullback attractors adjusting positions
   - Cognitive landscape evolution
   - Real-time field dynamics (not static)
   - Mathematical field equation visualization

## Performance Validation

### Scale Testing
```bash
# Test batch limits (clarified requirements)
# - Max 500MB per file
# - Max 1000 files per batch
# - Queue-based capacity management

# Create large test batch
for i in {1..100}; do
  cp large-document.pdf test-batch/doc_$i.pdf
done

# Upload and verify queuing behavior
curl -X POST "http://localhost:8000/api/v1/documents/bulk" \
  -F "files=@test-batch/*"

# Expected: 202 or 503 (capacity exceeded - queued)
```

### Real-time Performance
1. **WebSocket Updates**: Should receive updates within 100ms of processing events
2. **3D Visualization**: Should maintain 60fps during field evolution
3. **Memory Usage**: Should respect Redis TTL settings automatically

### TTL Validation
```bash
# Verify Redis TTL enforcement
redis-cli
> TTL thoughtseed:packet:{id}    # Should show ~86400 seconds (24h)
> TTL attractor:basin:{id}       # Should show ~604800 seconds (7d)
> TTL results:batch:{id}         # Should show ~2592000 seconds (30d)
```

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   ```bash
   # Check WebSocket endpoint
   wscat -c ws://localhost:8000/ws/batch/{batch_id}/progress
   ```

2. **Mock Simulation Still Showing**
   - Verify frontend is calling actual API endpoints
   - Check network tab for real HTTP requests
   - Confirm no setTimeout() calls in upload processing

3. **Consciousness Not Detected**
   - Verify ThoughtSeed processing is enabled
   - Check consciousness threshold (>0.3 for detection)
   - Ensure active inference components are loaded

4. **Attractor Basins Not Updating**
   - Confirm concept similarity calculation is working
   - Verify 384-dimensional vectors are being created
   - Check mathematical foundation implementation

5. **3D Visualization Not Working**
   - Verify three.js is loaded correctly
   - Check WebGL support in browser
   - Confirm neural field data is being received

### Validation Checklist

- [ ] All mock/simulation code removed from frontend
- [ ] Real API calls to backend endpoints
- [ ] WebSocket providing actual processing updates
- [ ] ThoughtSeed 5-layer processing working
- [ ] Attractor basin mathematics implemented correctly
- [ ] Neural field PDE evolution functioning
- [ ] Redis TTL values enforced as specified
- [ ] Neo4j knowledge graph populated
- [ ] 3D visualization showing real-time updates
- [ ] Capacity management queuing active
- [ ] File size/count limits enforced
- [ ] Search working with vector + graph results
- [ ] Consciousness detection above threshold working
- [ ] MIT/IBM/Shanghai research integration active

## Success Criteria

The quickstart is successful when:
1. Documents upload without mock simulation
2. All 5 ThoughtSeed layers process sequentially
3. Attractor basins modify based on real concept similarity
4. Neural fields evolve according to PDE mathematics
5. 3D visualization updates in real-time with actual data
6. Knowledge graph stores relationships correctly
7. Search returns results from both vector and graph sources
8. System handles capacity limits with proper queuing
9. All TTL values are enforced automatically
10. No simulation or mock code remains in the system

This validates the complete replacement of mock implementations with working ThoughtSeed consciousness processing at research-grade scale.