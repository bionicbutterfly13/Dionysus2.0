# CLAUSE Phase 2 - Working Demo Instructions

## ✅ Demo Server is Ready

The standalone demo server is now running with a complete working pipeline.

### What's Working:
- ✅ In-memory knowledge graph with climate change data (8 nodes, 7 edges)
- ✅ Document upload and concept extraction
- ✅ Complete CLAUSE multi-agent coordination:
  - SubgraphArchitect
  - PathNavigator
  - ContextCurator
- ✅ Real-time processing visualization
- ✅ Performance metrics tracking

### Quick Start

#### 1. Demo Server is Running on Port 8001
```bash
# Server is already running at http://localhost:8001
# Check status:
curl http://localhost:8001/api/demo/graph-status
```

#### 2. Open the Demo Interface
```bash
# Open in browser:
open /Volumes/Asylum/dev/Dionysus-2.0/backend/demo_interface.html
```

#### 3. Upload a Test Document
The interface supports drag-and-drop or click to browse.

Sample test file is at: `/tmp/climate_test.txt`

Content:
```
Climate change is primarily caused by greenhouse gas emissions from human activities.
Carbon dioxide (CO2) from burning fossil fuels is the main contributor to global warming.
Rising temperatures are causing extreme weather events and sea level rise around the world.
To address this crisis, we need to transition to renewable energy sources like solar and wind power.
Renewable energy technologies produce electricity without emitting greenhouse gases into the atmosphere.
```

### API Endpoints

#### 1. Graph Status
```bash
curl http://localhost:8001/api/demo/graph-status
```

Returns:
- Total nodes: 8
- Total edges: 7
- Available concepts: climate_change, greenhouse_gases, CO2, fossil_fuels, etc.

#### 2. Process Document (Full Pipeline)
```bash
curl -X POST http://localhost:8001/api/demo/process-document \
  -F "file=@/tmp/climate_test.txt"
```

Returns:
- Document text (truncated to 500 chars)
- Concepts extracted: [climate_change, global_warming, greenhouse_gases, CO2, renewable_energy, fossil_fuels]
- CLAUSE response with:
  - Agent handoffs (3 agents)
  - Navigation path
  - Evidence curated
  - Performance metrics
- Processing stages (4 stages with timing)
- Total time: ~280ms

#### 3. Simple Query (Test CLAUSE Only)
```bash
curl -X POST "http://localhost:8001/api/demo/simple-query?query=What+causes+climate+change&start_node=climate_change"
```

### What Happens When You Upload a Document

**Stage 1: Document Upload (Daedalus Gateway)**
- Receives your uploaded file
- Validates file type (.txt)
- Extracts text content

**Stage 2: Concept Extraction**
- Scans document for keywords
- Maps keywords to knowledge graph concepts
- Extracts 6 concepts from the climate test document

**Stage 3: CLAUSE Agent Initialization**
- Creates PathNavigator with in-memory graph access
- Creates ContextCurator with GPT-4 tokenizer
- Creates LC-MAPPO Coordinator
- Patches methods to use demo graph

**Stage 4: CLAUSE Multi-Agent Coordination**
1. **SubgraphArchitect**: Builds relevant subgraph (0.009ms)
2. **PathNavigator**: Navigates knowledge graph (0.71ms)
   - State encoding: 1155-dim feature vector
   - Termination head: binary classifier
   - Budget-aware navigation
3. **ContextCurator**: Curates evidence (6.38ms)
   - Listwise scoring with diversity penalty
   - Token budget enforcement
   - Provenance tracking with trust signals

**Total Processing Time**: ~280ms per document

### Demo Interface Features

When you open `demo_interface.html`:

1. **Upload Area**
   - Drag and drop .txt files
   - Or click to browse

2. **Processing Pipeline Visualization**
   - Shows all 4 stages
   - Real-time duration tracking
   - Stage-by-stage results

3. **Results Display**
   - **Concepts Extracted**: Color-coded badges for each concept
   - **Navigation Path**: Visual flow chart (node → node → node)
   - **Performance Metrics**: 4-panel dashboard
     - Total time (ms)
     - Agents executed
     - Nodes visited
     - Conflicts resolved
   - **Full JSON Response**: Expandable raw data

### Known Limitations (Demo Mode)

- **In-Memory Graph**: Pre-populated with climate data only
- **Keyword Extraction**: Simple pattern matching (no NLP)
- **Hash Embeddings**: Deterministic but not semantic
- **No Persistence**: Data resets when server restarts
- **Single User**: Not designed for concurrent requests

### Production Differences

In production CLAUSE Phase 2:
- **Neo4j**: Full graph database with millions of nodes
- **Sentence-Transformers**: Real semantic embeddings
- **Redis**: Distributed caching and queuing
- **AutoSchemaKG**: Automatic knowledge graph construction
- **Multi-tenant**: Handles concurrent users

### Stopping the Demo

```bash
# Find and kill the demo server
lsof -ti:8001 | xargs kill -9
```

### Creating Your Own Test Documents

Create any .txt file with climate-related content:

```bash
cat > my_test.txt << 'EOF'
Renewable energy sources like solar and wind power can help reduce greenhouse gas emissions.
This is important for fighting climate change and global warming.
Fossil fuels release CO2 when burned, contributing to extreme weather and sea level rise.
EOF
```

Then upload via the demo interface!

### Troubleshooting

**Server Not Responding**
```bash
# Check if server is running
lsof -i:8001

# Restart server
python /Volumes/Asylum/dev/Dionysus-2.0/backend/demo_server.py
```

**CORS Errors in Browser**
- The server has CORS enabled for all origins
- Make sure you're opening the HTML file locally

**No Concepts Extracted**
- Demo only recognizes climate-related keywords
- Keywords: climate, warming, greenhouse, carbon, CO2, temperature, weather, renewable, fossil

### Next Steps

To test CLAUSE Phase 2 further:
1. Upload different climate-related documents
2. Watch the processing stages
3. Examine the navigation paths
4. Review performance metrics

For production deployment:
- Complete remaining tasks (T059-T068)
- Set up Neo4j database
- Configure Redis cache
- Deploy with gunicorn/nginx
