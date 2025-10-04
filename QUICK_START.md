# CLAUSE Phase 2 - Quick Start for Flux

## ✅ Ready to Use

CLAUSE Phase 2 multi-agent system is running and ready for Flux integration.

---

## For Flux Developers

### 1. Upload a Document

```typescript
const formData = new FormData();
formData.append('file', fileBlob, 'document.txt');

const response = await fetch('http://localhost:8001/api/demo/process-document', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

### 2. Display Results

```typescript
// Extracted concepts
result.concepts_extracted
// ["climate_change", "greenhouse_gases", "CO2", ...]

// Agent execution timeline
result.clause_response.agent_handoffs
// [{agent: "SubgraphArchitect", latency_ms: 0.02}, ...]

// Total processing time
result.total_time_ms
// 32.26
```

### 3. Available Endpoints

| Endpoint | What it does |
|----------|-------------|
| `POST /api/demo/process-document` | Upload document → Extract concepts → Run CLAUSE |
| `GET /api/demo/graph-status` | See available concepts in graph |
| `POST /api/demo/simple-query` | Query graph directly |

---

## Test It Now

```bash
# Check server is running
curl http://localhost:8001/api/demo/graph-status

# Test document processing
echo "Climate change is caused by greenhouse gases." > test.txt
curl -X POST http://localhost:8001/api/demo/process-document -F "file=@test.txt"
```

---

## What Works

✅ Document upload (.txt files)
✅ Concept extraction (climate-related keywords)
✅ Multi-agent coordination (3 agents)
✅ Knowledge graph navigation
✅ Evidence curation with provenance
✅ Performance metrics (<50ms)
✅ No external dependencies needed

---

## Supported Concepts

The demo graph contains 8 climate concepts:
- climate_change
- greenhouse_gases
- CO2
- fossil_fuels
- global_warming
- sea_level_rise
- extreme_weather
- renewable_energy

Documents mentioning these keywords will be processed through the full CLAUSE pipeline.

---

## More Information

- **Integration Guide**: `FLUX_CLAUSE_INTEGRATION.md`
- **Technical Details**: `backend/DEMO_INSTRUCTIONS.md`
- **Complete Summary**: `CLAUSE_PHASE2_READY.md`

---

## Server Status

**Running at**: http://localhost:8001
**Demo endpoints**: `/api/demo/*`
**Status**: ✅ READY
