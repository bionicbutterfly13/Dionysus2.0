# 🚀 Access Flux Frontend with CLAUSE Phase 2

## ✅ Both Services Running

### 1. **Flux Frontend** (React + Vite)
**URL**: http://localhost:9244/
**Status**: ✅ Running
**Started**: `npm run dev` in `frontend/` directory

### 2. **CLAUSE Backend** (FastAPI)
**URL**: http://localhost:8001/
**Status**: ✅ Running
**Started**: `python backend/demo_server.py`

---

## 🎯 How to Use CLAUSE from Flux

The frontend now proxies all `/api/*` requests to the CLAUSE backend on port 8001.

### Available CLAUSE Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/demo/process-document` | POST | Upload document → CLAUSE processing |
| `/api/demo/graph-status` | GET | View knowledge graph status |
| `/api/demo/simple-query` | POST | Query graph directly |

### Test from Browser Console

Open http://localhost:9244/ and try:

```javascript
// Test 1: Check graph status
fetch('/api/demo/graph-status')
  .then(r => r.json())
  .then(console.log)

// Test 2: Upload a document
const formData = new FormData();
const blob = new Blob(['Climate change is caused by greenhouse gases.'], {type: 'text/plain'});
formData.append('file', blob, 'test.txt');

fetch('/api/demo/process-document', {
  method: 'POST',
  body: formData
})
  .then(r => r.json())
  .then(console.log)

// Test 3: Simple query
fetch('/api/demo/simple-query?query=What+causes+climate+change&start_node=climate_change', {
  method: 'POST'
})
  .then(r => r.json())
  .then(console.log)
```

---

## 📁 Where to Add CLAUSE UI Components

To integrate CLAUSE into Flux, add components to the frontend:

```
frontend/
├── src/
│   ├── components/
│   │   └── clause/              ← Create this folder
│   │       ├── DocumentUpload.tsx
│   │       ├── ConceptDisplay.tsx
│   │       ├── AgentTimeline.tsx
│   │       └── GraphVisualization.tsx
│   └── pages/
│       └── CLAUSE.tsx           ← Create new page
```

---

## 🔧 Quick Component Example

Create `frontend/src/components/clause/DocumentUpload.tsx`:

```typescript
import { useState } from 'react';

export function DocumentUpload() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/demo/process-document', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('CLAUSE error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">CLAUSE Document Processing</h2>

      <input
        type="file"
        accept=".txt"
        onChange={handleUpload}
        className="mb-4"
      />

      {loading && <div>Processing through CLAUSE multi-agent system...</div>}

      {result && (
        <div className="mt-4 space-y-4">
          <div>
            <h3 className="font-bold">Concepts Extracted:</h3>
            <div className="flex gap-2 flex-wrap mt-2">
              {result.concepts_extracted.map((concept: string) => (
                <span key={concept} className="bg-blue-100 px-3 py-1 rounded">
                  {concept}
                </span>
              ))}
            </div>
          </div>

          <div>
            <h3 className="font-bold">Agent Execution:</h3>
            {result.clause_response.agent_handoffs.map((handoff: any) => (
              <div key={handoff.agent} className="flex justify-between py-1">
                <span>{handoff.agent}</span>
                <span>{handoff.latency_ms.toFixed(2)}ms</span>
              </div>
            ))}
          </div>

          <div>
            <h3 className="font-bold">Total Time:</h3>
            <span>{result.total_time_ms.toFixed(2)}ms</span>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## 🎨 Flux UI Integration Steps

1. **Create CLAUSE page**:
   ```bash
   # In frontend/src/pages/
   touch CLAUSE.tsx
   ```

2. **Add route** in `frontend/src/App.tsx`:
   ```typescript
   import { CLAUSEPage } from './pages/CLAUSE';

   // In routes:
   <Route path="/clause" element={<CLAUSEPage />} />
   ```

3. **Add navigation link** in your sidebar/nav:
   ```typescript
   <Link to="/clause">CLAUSE Phase 2</Link>
   ```

4. **Build the CLAUSE components**:
   - Document uploader
   - Concept badges
   - Agent execution timeline
   - Knowledge graph visualization (use Three.js or D3)
   - Performance metrics dashboard

---

## 🧪 Test Endpoints

```bash
# Test backend directly
curl http://localhost:8001/api/demo/graph-status

# Test through frontend proxy
curl http://localhost:9244/api/demo/graph-status
```

Both should return the same JSON response.

---

## 📊 What You Can Build

### 1. Document Processing Interface
- Drag-and-drop upload
- Real-time processing status
- Concept extraction display
- Agent execution timeline

### 2. Knowledge Graph Explorer
- Interactive node/edge visualization (Three.js)
- Click nodes to explore relationships
- Search/filter concepts
- Path navigation display

### 3. Analytics Dashboard
- Total documents processed
- Average processing time
- Most frequent concepts
- Agent performance metrics

---

## 🔄 Restart Services

**If you need to restart:**

```bash
# Kill frontend
lsof -ti:9244 | xargs kill -9

# Kill backend
lsof -ti:8001 | xargs kill -9

# Restart backend
python backend/demo_server.py

# Restart frontend
cd frontend && npm run dev
```

---

## 📚 Documentation

- **Integration Guide**: [FLUX_CLAUSE_INTEGRATION.md](FLUX_CLAUSE_INTEGRATION.md)
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Technical Details**: [CLAUSE_PHASE2_READY.md](CLAUSE_PHASE2_READY.md)
- **Backend Docs**: [backend/DEMO_INSTRUCTIONS.md](backend/DEMO_INSTRUCTIONS.md)

---

## ✅ Summary

**Flux Frontend**: http://localhost:9244/
**CLAUSE Backend**: http://localhost:8001/
**Status**: Both running and connected via Vite proxy

**Next**: Build UI components to integrate CLAUSE into Flux!
