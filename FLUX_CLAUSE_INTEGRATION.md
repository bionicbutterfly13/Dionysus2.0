# Flux Integration with CLAUSE Phase 2

## Summary

CLAUSE Phase 2 multi-agent system is now available for Flux to use via the backend API at `/api/demo/*` endpoints.

**No Neo4j or Redis required** - runs entirely in-memory with a pre-populated climate change knowledge graph.

## Available Endpoints

### 1. Process Document (End-to-End Pipeline)

```typescript
// Upload document and run through complete CLAUSE pipeline
const formData = new FormData();
formData.append('file', fileBlob, 'document.txt');

const response = await fetch('http://localhost:8001/api/demo/process-document', {
  method: 'POST',
  body: formData
});

const result = await response.json();
/*
{
  "document_text": "Climate change is...",
  "concepts_extracted": ["climate_change", "greenhouse_gases", "CO2", ...],
  "clause_response": {
    "result": {
      "subgraph": {...},
      "path": {
        "nodes": ["climate_change", "greenhouse_gases", ...],
        "edges": [...],
        "steps": [...]
      },
      "evidence": [...]
    },
    "agent_handoffs": [
      {"step": 1, "agent": "SubgraphArchitect", ...},
      {"step": 2, "agent": "PathNavigator", ...},
      {"step": 3, "agent": "ContextCurator", ...}
    ],
    "conflicts_resolved": 0,
    "performance": {"total_latency_ms": 13.15, ...}
  },
  "processing_stages": [
    {"stage": 1, "name": "Document Upload", "duration_ms": 0.014},
    {"stage": 2, "name": "Concept Extraction", "duration_ms": 0.159},
    {"stage": 3, "name": "CLAUSE Multi-Agent Coordination", "duration_ms": 13.397},
    {"stage": 4, "name": "Result Extraction", "duration_ms": 0.002}
  ],
  "total_time_ms": 279.02
}
*/
```

### 2. Graph Status

```typescript
// Check what concepts are available in the knowledge graph
const response = await fetch('http://localhost:8001/api/demo/graph-status');
const status = await response.json();
/*
{
  "total_nodes": 8,
  "total_edges": 7,
  "available_concepts": [
    "climate_change",
    "greenhouse_gases",
    "CO2",
    "fossil_fuels",
    "global_warming",
    "sea_level_rise",
    "extreme_weather",
    "renewable_energy"
  ],
  "sample_edges": [
    {"from": "greenhouse_gases", "to": "climate_change", "relation": "causes"},
    {"from": "CO2", "to": "greenhouse_gases", "relation": "is_a"},
    ...
  ]
}
*/
```

### 3. Simple Query (Test CLAUSE Only)

```typescript
// Query the knowledge graph directly without uploading a document
const query = encodeURIComponent("What causes climate change?");
const startNode = "climate_change";

const response = await fetch(
  `http://localhost:8001/api/demo/simple-query?query=${query}&start_node=${startNode}`,
  { method: 'POST' }
);

const result = await response.json();
/*
{
  "query": "What causes climate change?",
  "start_node": "climate_change",
  "path": {
    "nodes": ["climate_change", "greenhouse_gases", ...],
    "edges": [...],
    "steps": [...]
  },
  "metadata": {...},
  "performance": {...}
}
*/
```

## React/TypeScript Integration Example

```typescript
import { useState } from 'react';

interface CLAUSEResult {
  concepts_extracted: string[];
  clause_response: {
    agent_handoffs: Array<{
      agent: string;
      latency_ms: number;
    }>;
    performance: {
      total_latency_ms: number;
    };
  };
  processing_stages: Array<{
    stage: number;
    name: string;
    duration_ms: number;
  }>;
}

function CLAUSEUploader() {
  const [result, setResult] = useState<CLAUSEResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async (file: File) => {
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8001/api/demo/process-document', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('CLAUSE processing failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept=".txt"
        onChange={(e) => {
          if (e.target.files?.[0]) {
            handleFileUpload(e.target.files[0]);
          }
        }}
      />

      {loading && <div>Processing through CLAUSE multi-agent system...</div>}

      {result && (
        <div>
          <h3>Concepts Extracted: {result.concepts_extracted.length}</h3>
          <ul>
            {result.concepts_extracted.map((concept) => (
              <li key={concept}>{concept}</li>
            ))}
          </ul>

          <h3>Agent Execution:</h3>
          {result.clause_response.agent_handoffs.map((handoff) => (
            <div key={handoff.agent}>
              {handoff.agent}: {handoff.latency_ms.toFixed(2)}ms
            </div>
          ))}

          <h3>Total Time: {result.clause_response.performance.total_latency_ms.toFixed(2)}ms</h3>
        </div>
      )}
    </div>
  );
}
```

## What Flux Can Do

### Document Analysis
1. **Upload any text document** (.txt files)
2. **Automatic concept extraction** (detects climate-related keywords)
3. **Knowledge graph integration** (concepts added to graph)
4. **Multi-agent processing** (3 agents execute sequentially)
5. **Structured results** with full provenance

### Knowledge Graph Exploration
1. **Query concepts directly** without document upload
2. **Navigate relationships** between concepts
3. **View evidence** collected by agents
4. **Track performance** of each agent

### Visualization Opportunities
- **Agent execution timeline** (show 3-agent handoff)
- **Knowledge graph visualization** (nodes + edges)
- **Navigation path display** (step-by-step traversal)
- **Performance metrics** (latency breakdown)
- **Evidence provenance** (trust signals, verification status)

## Demo Data

The in-memory graph contains 8 climate-related concepts:

**Concepts:**
- `climate_change`: Root concept for climate science
- `greenhouse_gases`: Atmospheric heat-trapping gases
- `CO2`: Primary greenhouse gas from human activity
- `fossil_fuels`: Energy source releasing CO2
- `global_warming`: Rising surface temperatures
- `sea_level_rise`: Effect of climate change
- `extreme_weather`: Effect of climate change
- `renewable_energy`: Low-carbon energy solution

**Relationships:**
- greenhouse_gases → causes → climate_change
- CO2 → is_a → greenhouse_gases
- fossil_fuels → produces → CO2
- climate_change → includes → global_warming
- climate_change → causes → sea_level_rise
- climate_change → causes → extreme_weather
- renewable_energy → reduces → greenhouse_gases

## Supported Document Content

For best results, documents should mention these keywords:
- **climate**, warming, greenhouse, carbon, CO2
- temperature, weather, renewable, fossil

Example test document:
```text
Climate change is primarily caused by greenhouse gas emissions from human activities.
Carbon dioxide (CO2) from burning fossil fuels is the main contributor to global warming.
Rising temperatures are causing extreme weather events and sea level rise around the world.
To address this crisis, we need to transition to renewable energy sources like solar and wind power.
Renewable energy technologies produce electricity without emitting greenhouse gases into the atmosphere.
```

## Performance Targets

- **Document upload**: < 1ms
- **Concept extraction**: < 1ms
- **CLAUSE coordination**: 10-20ms
- **Total end-to-end**: < 300ms

## Error Handling

```typescript
try {
  const response = await fetch('http://localhost:8001/api/demo/process-document', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    console.error('CLAUSE error:', error.detail);
    // Common errors:
    // - "No relevant concepts found in document" (HTTP 400)
    // - "Node not found" for invalid start_node (HTTP 400)
  }
} catch (error) {
  console.error('Network error:', error);
}
```

## Next Steps for Flux Integration

1. **Add file upload component** in Flux UI
2. **Display extracted concepts** as badges/chips
3. **Visualize agent execution** as timeline
4. **Show navigation path** as interactive graph
5. **Display evidence** with provenance metadata
6. **Add performance metrics** dashboard

## Production Migration Path

When ready to move from demo to production:

1. Replace `/api/demo/*` with `/api/clause/*`
2. Connect to Neo4j for full knowledge graph
3. Use sentence-transformers for semantic embeddings
4. Enable Redis for distributed caching
5. Add user authentication
6. Scale with horizontal deployment

## Technical Details

- **No external dependencies**: Runs entirely in-memory
- **Deterministic embeddings**: Hash-based (reproducible)
- **Fast performance**: Sub-300ms total processing
- **Full provenance**: Every piece of evidence tracked
- **Budget enforcement**: Edge, step, and token budgets respected
- **Multi-agent coordination**: LC-MAPPO algorithm

## Questions?

See `backend/DEMO_INSTRUCTIONS.md` for detailed technical documentation.
