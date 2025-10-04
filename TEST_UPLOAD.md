# ✅ Test CLAUSE Document Upload in Flux

## Simple Instructions

### 1. Open Flux
Go to: **http://localhost:9244/upload**

### 2. Create a Test File

Save this as `climate_test.txt` on your computer:

```
Climate change is primarily caused by greenhouse gas emissions from human activities.
Carbon dioxide (CO2) from burning fossil fuels is the main contributor to global warming.
Rising temperatures are causing extreme weather events and sea level rise around the world.
To address this crisis, we need to transition to renewable energy sources like solar and wind power.
Renewable energy technologies produce electricity without emitting greenhouse gases into the atmosphere.
```

### 3. Upload the File

**Method 1: Drag and Drop**
- Drag the file into the upload area

**Method 2: Click to Browse**
- Click the upload area
- Select your file

### 4. Watch It Process

You'll see:
1. **Upload progress bar** (fast)
2. **Status changes to "Processing with consciousness emulator..."**
3. **Results appear showing**:
   - ✅ **Concepts**: 6 concepts extracted (climate_change, greenhouse_gases, CO2, fossil_fuels, global_warming, renewable_energy)
   - ✅ **Consciousness**: X basins • Y seeds
   - ✅ **Quality**: 95%
   - ✅ **Workflow messages**: 4 stages with timing

### 5. What You're Seeing

The file goes through:
1. **Document Upload** via Daedalus gateway
2. **Concept Extraction** finds climate keywords
3. **CLAUSE Multi-Agent Coordination**:
   - SubgraphArchitect builds relevant subgraph
   - PathNavigator explores knowledge graph
   - ContextCurator selects evidence
4. **Results** displayed in Flux UI

---

## Expected Results

After upload, you should see a card like this:

```
📄 climate_test.txt                                    ✓

Concepts: 6
  climate_change, greenhouse_gases, CO2, fossil_fuels,
  global_warming, renewable_energy

Consciousness: 🧠 6 basins • 3 seeds

Quality: 95%

Workflow:
  Stage 1: Document Upload (Daedalus) (0.0ms)
  Stage 2: Concept Extraction (0.2ms)
  Stage 3: CLAUSE Multi-Agent Coordination (13.4ms)
  Stage 4: Result Extraction (0.0ms)
```

---

## Troubleshooting

**Dashboard shows 404 errors?**
- The backend auto-reloaded - dashboard stats endpoint is now working
- Refresh the page: http://localhost:9244/

**Upload doesn't work?**
- Check both servers are running:
  - Backend: http://localhost:8001/ (should show "CLAUSE Phase 2 Demo Server")
  - Frontend: http://localhost:9244/ (should show Flux interface)

**No concepts extracted?**
- Make sure file mentions climate keywords: climate, warming, greenhouse, carbon, CO2, renewable, fossil

---

## What's Actually Happening

Behind the scenes:
1. Frontend sends file to `/api/demo/process-document`
2. Backend extracts text and finds climate keywords
3. CLAUSE agents process the concepts through the knowledge graph
4. Results are formatted and displayed in Flux UI

**You're seeing CLAUSE Phase 2 multi-agent system working through Flux!** 🎉
