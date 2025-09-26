# Quickstart: Flux Self-Teaching Consciousness Emulator

**Branch**: `002-i-m-building`  
**Spec**: [spec.md](spec.md)  
**Plan**: [plan.md](plan.md)

Flux is our self-teaching consciousness emulator that integrates Dionysus modules, SurfSense inspiration, and ASI-Arch pipelines into a local-first co-thinking system. This quickstart walks through environment setup, document ingestion, curiosity flows, and validation.

---

## 1. Prerequisites (Updated from Implementation Testing)
- macOS or Linux desktop (local-first priority)
- Python 3.11+ (confirmed working with existing `asi-arch-env`)
- Node.js 20+ (for React frontend)
- **Docker & Docker Compose** (for database services)

### 1.1 Critical Dependencies (Missing from Initial Implementation)
Based on testing, these dependencies are **required** but not yet installed:

```bash
# Activate the ASI-Arch environment
source asi-arch-env/bin/activate

# Install missing core dependencies
pip install neo4j>=5.15.0 redis>=5.0.1 aioredis>=2.0.1
pip install qdrant-client>=1.7.1 ollama>=0.1.7
pip install numpy>=1.24.4 pandas>=2.1.4
pip install sentence-transformers  # For embeddings

# Install LangGraph for workflow orchestration (already in codebase)
pip install langgraph>=0.0.40 langchain>=0.1.0

# Install AutoSchema KG dependencies (optional, for knowledge graph construction)
pip install atlas-rag  # If available, for AutoSchema integration
```

### 1.2 Service Dependencies
**IMPORTANT**: These services must be running before starting Flux:

- **Neo4j 5.x**: Graph database (canonical storage)
- **Redis 7.x**: Caching and curiosity signal streams
- **Qdrant**: Vector embeddings (or use built-in SQLite fallback)
- **Ollama**: Local LLM inference with LLaMA models

---

## 2. Repository Setup & Dependencies

### 2.1 Install Backend Dependencies
```bash
cd /Volumes/Asylum/dev/Dionysus-2.0
source asi-arch-env/bin/activate

# Install core FastAPI dependencies (tested working)
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv httpx

# Install database clients (currently missing)
pip install neo4j>=5.15.0 redis>=5.0.1 aioredis>=2.0.1 qdrant-client>=1.7.1

# Install ML and processing dependencies
pip install numpy>=1.24.4 pandas>=2.1.4 sentence-transformers ollama>=0.1.7

# Install authentication
pip install PyJWT

# Install from existing requirements (optional)
pip install -r backend/requirements.txt  # May have version conflicts
```

### 2.2 Install Frontend Dependencies
```bash
cd frontend
npm install  # Already configured with React, TypeScript, Vite
```

### 2.3 Start Required Services (Docker Recommended)

Create `docker-compose.yml` for local development:
```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.15
    environment:
      NEO4J_AUTH: neo4j/flux_password
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_default__database: flux
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  neo4j_data:
  redis_data:
  qdrant_data:
```

**Start services:**
```bash
# Start all database services
docker-compose up -d

# Install and start Ollama separately (for LLM inference)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1  # Or your preferred model

# Verify services are running
docker ps
curl http://localhost:6333/health  # Qdrant
redis-cli ping  # Redis
# Neo4j browser: http://localhost:7474
```

---

## 3. Launch Flux Services

### 3.1 Initialize Databases
**FIRST RUN ONLY**: Initialize database schemas and collections:
```bash
# Initialize Neo4j schema (creates constraints, indexes, node types)
curl -X POST http://localhost:8000/api/v1/system/databases/neo4j/schema

# Initialize Qdrant collections (creates embedding collections)
curl -X POST http://localhost:8000/api/v1/system/databases/qdrant/collections

# Initialize Redis streams (creates curiosity and consciousness streams)
curl -X POST http://localhost:8000/api/v1/system/databases/redis/streams
```

### 3.2 Backend (FastAPI)
```bash
# Ensure services are running first
docker-compose up -d

# Activate environment and start backend
source asi-arch-env/bin/activate
cd backend
python -c "from src.app_factory import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000)"

# Or use the main entry point
python src/main.py
```

**Backend Features Tested:**
- ✅ FastAPI app starts successfully
- ✅ Health check: `GET http://localhost:8000/health`
- ✅ API endpoints respond (with proper headers)
- ✅ Constitutional compliance middleware active
- ✅ Development mode bypasses auth with `X-Dev-Mode: true` header

### 3.2.1 Test Backend Health
```bash
# Basic health check
curl http://localhost:8000/health

# Test document endpoint (requires proper headers)
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: multipart/form-data" \
  -H "X-Dev-Mode: true" \
  -F "test=data"

# Test curiosity endpoint
curl -X POST http://localhost:8000/api/v1/curiosity/missions \
  -H "Content-Type: application/json" \
  -H "X-Dev-Mode: true" \
  -d '{}'

# Check database connectivity
curl http://localhost:8000/api/v1/system/databases/status
```

### 3.3 Frontend (React + Vite)
```bash
cd frontend
npm run dev
```
- **URL**: `http://localhost:3000`
- **Status**: Basic React app structure exists, components need implementation

### 3.4 ThoughtSeed Integration (Existing ASI-Arch Modules)
```bash
# Test existing ThoughtSeed enhanced pipeline
python extensions/context_engineering/thoughtseed_enhanced_pipeline.py

# Test AutoSchema KG integration
python extensions/context_engineering/autoschema_integration.py

# Test LangGraph hybrid fusion engine
python extensions/context_engineering/hybrid_fusion_langgraph_engine.py
```

---

## 4. Core Workflow Validation
### 4.1 Document Ingestion
1. Launch Flux UI (`localhost:3000`).
2. Upload sample documents (use real PDF/Markdown to avoid mock-only flows).
3. Monitor backend logs for ThoughtSeed activation.
4. Verify Neo4j nodes via Neo4j Browser:
   ```cypher
   MATCH (d:DocumentArtifact) RETURN d LIMIT 5;
   MATCH (t:ThoughtSeedTrace)-[:FEATURES_IN]->(c:ConceptNode) RETURN t, c LIMIT 5;
   ```
5. Confirm `EvaluationFrame` entries exist for ingestion session:
   ```cypher
   MATCH (e:EvaluationFrame {context_type: 'ingestion'}) RETURN e;
   ```

### 4.2 Explanation & Visualization
- Open Flux visualization dashboard → confirm graph & card stack updates.
- Check WebSocket logs for `VisualizationMessage` payloads (type `evaluation_frame` and `mosaic_state`).
- Ensure Mosaic dimensions reflect the Mosaic Systems LLC schema (values 0-1).

### 4.3 Curiosity Missions
1. Trigger curiosity by exploring concept gaps or enabling “increase curiosity” slider.
2. Review `/api/v1/curiosity/missions` response:
   ```bash
   curl http://localhost:8000/api/v1/curiosity/missions
   ```
3. Validate trust scoring and evaluation frames in mission updates.
4. Confirm replay scheduling occurs during idle/nightly periods (check logs for `replay_priority` handling).

### 4.4 Dreaming & Replay
- Wait for scheduled nightly replay or simulate via CLI:
  ```bash
  python dionysus-source/agents/dream_scheduler.py --run-once
  ```
- Ensure new dream insights are flagged in Neo4j (`consciousness_state = 'dreaming'`).

---

## 5. Constitutional Compliance Checks
- **Scientific Integrity**: Inspect generated summaries for provenance & no hype.
- **Mock Data Transparency**: If using mock inputs, confirm UI displays disclaimers and `mock_data = true` persisted in Neo4j.
- **Redundancy Safeguard**: Audit integration references in code (no duplicate Dionysus/SurfSense/ASI-Arch functionality).
- **ThoughtSeed Channels**: Verify all ingestion events create `ThoughtSeedTrace` relationships.
- **Evaluation Feedback**: Each processing stage should log `whats_good / whats_broken / works_but_shouldnt / pretends_but_doesnt`.

---

## 6. Common Troubleshooting
| Issue | Resolution |
|-------|------------|
| Neo4j auth errors | Update credentials in config `configs/flux.yaml` |
| WebSocket disconnects | Ensure backend at `localhost:8000` exposes `/ws/v1/visualizations` |
| Redis offline | Curiosity loops suspended automatically; restart Redis and requeue missions |
| Ollama model missing | Run `ollama pull llama3` or configure alternative local model |
| Mock data flagged | Replace sample inputs with real documents before production readiness |

---

## 7. Next Steps
- Execute `/tasks` after `/plan` Phase 1 completes to generate implementation tasks.
- Integrate migrated Dionysus modules into ASI-Arch `backend/src/` structure.
- Develop Flux UI components based on SurfSense inspirations with Flux branding and attribution.

This quickstart ensures Flux respects our constitution, maintains local-first privacy, and demonstrates the core co-thinking loop from document ingestion through curiosity-driven insight. Continue with `tasks.md` generation once plan artifacts are complete.
