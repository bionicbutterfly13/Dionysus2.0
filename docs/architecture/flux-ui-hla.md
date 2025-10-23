# Flux UI Enhancement - High-Level Architecture

**Document Version**: 1.0
**Date**: 2025-10-10
**Architect**: Winston (Architect) 🏗️
**Epic**: Flux UI Enhancement - Consciousness-Enhanced Interface
**Status**: Draft

---

## 1. Executive Summary

This document defines the technical architecture for transforming Flux from a basic document viewer into a consciousness-enhanced knowledge exploration workspace. The architecture supports real-time concept mapping, basin visualization, narrative extraction, hyper-bulk processing, and bidirectional knowledge management.

### Key Architectural Decisions

| Decision Area | Choice | Rationale |
|--------------|---------|-----------|
| **Real-Time Streaming** | Server-Sent Events (SSE) | Simpler than WebSockets, sufficient for unidirectional data flow, better browser compatibility, automatic reconnection |
| **State Management** | Zustand + React Query | Already in stack, lightweight, minimal boilerplate, React Query for server state caching |
| **Graph Visualization** | React Flow + Three.js hybrid | React Flow for 2D concept maps (<500 nodes), Three.js for 3D basin visualization (>500 nodes) |
| **Performance Strategy** | Progressive enhancement + Web Workers | Start simple, scale up. Offload heavy computation to workers |
| **Backend Integration** | REST + SSE (no GraphQL) | FastAPI REST for CRUD, SSE for real-time updates. Keep it simple. |
| **Type Safety** | Zod for schema validation | Runtime validation + TypeScript types from single source |

---

## 2. System Context

### 2.1 Current State Analysis

**Existing Stack** (from package.json):
- **Frontend**: React 18, Vite, TypeScript, Tailwind CSS
- **State**: Zustand 4.4.7, React Query 5.90.2
- **3D**: Three.js 0.158, React Three Fiber, Drei
- **Testing**: Jest, React Testing Library, Playwright
- **Already Installed**: `react-dropzone`, `websocket`, `axios`

**Backend** (from documents.py):
- **Framework**: FastAPI with async support
- **Processing**: LangGraph-based document processing pipeline
- **Storage**: In-memory + disk (uploads/) with Neo4j for knowledge graph
- **Consciousness**: DebugDocumentProcessor with basin activation

**Current Capabilities**:
- Document upload (single file)
- Debug pipeline with SSE streaming events
- Consciousness processing (basins, thoughtseeds, concepts)
- Basic document listing

### 2.2 Target Capabilities

**Phase 1 (Weeks 1-3)**: Foundation + Bulk Upload
- Lint-zero codebase with test framework
- Folder drag/drop with progress tracking
- Local processing toggle with batch handling
- Concept summary panel

**Phase 2 (Weeks 3-5)**: Core Visualization
- Interactive concept map (2D graph)
- Real-time basin activation
- Knowledge manager with bidirectional links
- Narrative extraction feed

**Phase 3 (Weeks 5-6)**: Advanced Features
- Generation controls (narrow/deeper/wider)
- Dual-concept analysis
- Zettelkasten integration
- Timeline visualization

---

## 3. Architectural Principles

### 3.1 Core Principles

**Boring Technology for Stability**
- Use existing stack (Zustand, React Query, Three.js)
- No new frameworks unless absolutely necessary
- Prefer libraries with strong ecosystem support

**Progressive Enhancement**
- Start with simple 2D visualizations
- Scale to 3D when complexity demands it
- Feature flags for experimental features

**Performance as Feature**
- <500ms for concept map rendering
- <100ms for real-time updates
- <2s for bulk upload (100 files)
- <200ms for knowledge manager search

**Developer Productivity**
- Shared types (Zod schemas)
- Clear component boundaries
- Comprehensive testing at all levels
- Hot reload for all UI changes

**Defense in Depth**
- Input validation (client + server)
- Rate limiting for expensive operations
- Graceful degradation when services unavailable
- Error boundaries for UI resilience

---

## 4. Real-Time Architecture

### 4.1 Decision: Server-Sent Events (SSE) over WebSockets

**Why SSE?**
- ✅ Unidirectional data flow (server → client) is sufficient
- ✅ Automatic reconnection built into EventSource API
- ✅ Simpler server implementation (FastAPI supports SSE natively)
- ✅ HTTP/2 multiplexing support
- ✅ Better firewall/proxy compatibility
- ✅ Already in use for debug_stream endpoint

**When WebSockets Would Be Needed**:
- ❌ Bidirectional real-time communication (not our use case)
- ❌ Binary data streaming (we use JSON)
- ❌ Gaming/collaborative editing (not our use case)

### 4.2 SSE Event Schema

**Event Channels**:
```typescript
// Using Zod for runtime validation + TypeScript types
import { z } from 'zod'

// Channel 1: Document Processing Events
const ProcessingEventSchema = z.object({
  type: z.enum([
    'processing_started',
    'extraction_complete',
    'basin_activated',
    'thoughtseed_created',
    'processing_complete',
    'processing_error'
  ]),
  document_id: z.string(),
  timestamp: z.string().datetime(),
  data: z.record(z.any()) // Flexible payload
})

// Channel 2: Concept Map Updates
const ConceptMapEventSchema = z.object({
  type: z.enum(['concept_added', 'concept_updated', 'link_created', 'basin_activated']),
  concept_id: z.string().optional(),
  basin_id: z.string().optional(),
  data: z.object({
    nodes: z.array(z.any()).optional(),
    edges: z.array(z.any()).optional(),
    basin_state: z.record(z.any()).optional()
  })
})

// Channel 3: Narrative Extraction Events
const NarrativeEventSchema = z.object({
  type: z.enum(['archetype_detected', 'sentiment_update', 'metaphor_generated', 'curiosity_prompt']),
  source_text: z.string().optional(),
  data: z.object({
    archetypes: z.array(z.string()).optional(),
    sentiment: z.number().optional(),
    metaphors: z.array(z.string()).optional(),
    curiosity_questions: z.array(z.string()).optional()
  })
})
```

**SSE Endpoint Structure**:
```
GET /api/stream/processing     # Document processing events
GET /api/stream/concept-map    # Concept map real-time updates
GET /api/stream/narrative      # Narrative extraction feed
```

### 4.3 Client-Side SSE Management

**Custom Hook for SSE Connection**:
```typescript
// hooks/useSSEStream.ts
import { useEffect, useState } from 'react'
import { z } from 'zod'

interface UseSSEStreamOptions<T> {
  url: string
  schema: z.ZodSchema<T>
  onMessage: (data: T) => void
  onError?: (error: Error) => void
  enabled?: boolean
}

export function useSSEStream<T>({ url, schema, onMessage, onError, enabled = true }: UseSSEStreamOptions<T>) {
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!enabled) return

    const eventSource = new EventSource(url)

    eventSource.onopen = () => {
      setIsConnected(true)
      setError(null)
    }

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        const validated = schema.parse(data)
        onMessage(validated)
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Invalid SSE data')
        setError(error)
        onError?.(error)
      }
    }

    eventSource.onerror = (err) => {
      setIsConnected(false)
      const error = new Error('SSE connection error')
      setError(error)
      onError?.(error)
    }

    return () => {
      eventSource.close()
    }
  }, [url, enabled])

  return { isConnected, error }
}
```

**Usage Pattern**:
```typescript
// In ConceptMapComponent
const { isConnected } = useSSEStream({
  url: '/api/stream/concept-map',
  schema: ConceptMapEventSchema,
  onMessage: (event) => {
    if (event.type === 'basin_activated') {
      updateConceptMap(event.data)
    }
  }
})
```

---

## 5. State Management Architecture

### 5.1 Decision: Zustand + React Query Hybrid

**State Ownership Model**:
```
┌─────────────────────────────────────┐
│         Client State                │
│    (Zustand Stores)                 │
│  - UI state (modals, selections)    │
│  - Local preferences                │
│  - Transient visualization state    │
└─────────────────────────────────────┘
              ↕
┌─────────────────────────────────────┐
│        Server State                 │
│    (React Query Cache)              │
│  - Documents                        │
│  - Concepts                         │
│  - Basins                           │
│  - Knowledge graph                  │
└─────────────────────────────────────┘
              ↕
┌─────────────────────────────────────┐
│      Real-Time Updates              │
│         (SSE Events)                │
│  - Invalidate React Query cache     │
│  - Update Zustand optimistically    │
└─────────────────────────────────────┘
```

### 5.2 Zustand Store Structure

**Store 1: UI State** (`stores/uiStore.ts`)
```typescript
import create from 'zustand'

interface UIState {
  // Modal states
  uploadModalOpen: boolean
  conceptDetailOpen: boolean

  // Selected items
  selectedConcept: string | null
  selectedBasin: string | null
  selectedDocuments: string[]

  // View preferences
  viewMode: '2d' | '3d'
  processingMode: 'local' | 'cloud'

  // Actions
  openUploadModal: () => void
  closeUploadModal: () => void
  selectConcept: (id: string | null) => void
  toggleViewMode: () => void
}

export const useUIStore = create<UIState>((set) => ({
  uploadModalOpen: false,
  conceptDetailOpen: false,
  selectedConcept: null,
  selectedBasin: null,
  selectedDocuments: [],
  viewMode: '2d',
  processingMode: 'local',

  openUploadModal: () => set({ uploadModalOpen: true }),
  closeUploadModal: () => set({ uploadModalOpen: false }),
  selectConcept: (id) => set({ selectedConcept: id }),
  toggleViewMode: () => set((state) => ({ viewMode: state.viewMode === '2d' ? '3d' : '2d' }))
}))
```

**Store 2: Concept Map State** (`stores/conceptMapStore.ts`)
```typescript
import create from 'zustand'

interface ConceptNode {
  id: string
  label: string
  type: 'concept' | 'basin' | 'thoughtseed'
  position: { x: number, y: number, z?: number }
  metadata: Record<string, any>
}

interface ConceptMapState {
  nodes: ConceptNode[]
  edges: Array<{ source: string, target: string, type: string }>

  // Optimistic updates from SSE
  addNode: (node: ConceptNode) => void
  updateNode: (id: string, updates: Partial<ConceptNode>) => void
  addEdge: (edge: { source: string, target: string, type: string }) => void

  // Reset for new data
  setGraph: (nodes: ConceptNode[], edges: any[]) => void
}

export const useConceptMapStore = create<ConceptMapState>((set) => ({
  nodes: [],
  edges: [],

  addNode: (node) => set((state) => ({ nodes: [...state.nodes, node] })),
  updateNode: (id, updates) => set((state) => ({
    nodes: state.nodes.map(n => n.id === id ? { ...n, ...updates } : n)
  })),
  addEdge: (edge) => set((state) => ({ edges: [...state.edges, edge] })),
  setGraph: (nodes, edges) => set({ nodes, edges })
}))
```

### 5.3 React Query Configuration

**Query Client Setup** (`lib/queryClient.ts`):
```typescript
import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      cacheTime: 1000 * 60 * 30, // 30 minutes
      refetchOnWindowFocus: false,
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000)
    },
    mutations: {
      retry: 1
    }
  }
})
```

**Query Hooks** (`hooks/useDocuments.ts`):
```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'

// Fetch documents with caching
export function useDocuments(topic?: string) {
  return useQuery({
    queryKey: ['documents', topic],
    queryFn: async () => {
      const { data } = await axios.get('/api/documents', { params: { topic } })
      return data
    }
  })
}

// Upload documents with optimistic updates
export function useUploadDocuments() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (files: File[]) => {
      const formData = new FormData()
      files.forEach(file => formData.append('files', file))
      const { data } = await axios.post('/api/documents', formData)
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['documents'])
    }
  })
}

// Fetch concept map data
export function useConceptMap(documentIds: string[]) {
  return useQuery({
    queryKey: ['concept-map', documentIds],
    queryFn: async () => {
      const { data } = await axios.get('/api/concepts/graph', { params: { documentIds } })
      return data
    },
    enabled: documentIds.length > 0
  })
}
```

### 5.4 SSE + React Query Integration

**Automatic Cache Invalidation from SSE**:
```typescript
// In component that listens to SSE
useSSEStream({
  url: '/api/stream/processing',
  schema: ProcessingEventSchema,
  onMessage: (event) => {
    if (event.type === 'processing_complete') {
      // Invalidate documents cache to trigger refetch
      queryClient.invalidateQueries(['documents'])

      // Invalidate concept map cache
      queryClient.invalidateQueries(['concept-map'])
    }
  }
})
```

---

## 6. Graph Visualization Architecture

### 6.1 Decision: React Flow + Three.js Hybrid

**Visualization Strategy**:
```
Concept Count < 500 nodes:
  └─> React Flow (2D)
      ├─> Fast rendering
      ├─> Built-in interactions (zoom, pan, select)
      ├─> Easy layout algorithms
      └─> Good accessibility

Concept Count >= 500 nodes:
  └─> Three.js + React Three Fiber (3D)
      ├─> WebGL rendering (60fps)
      ├─> Force-directed layout in 3D
      ├─> Instanced rendering for performance
      └─> Immersive exploration
```

### 6.2 React Flow Implementation (2D Concept Map)

**Component Structure**:
```typescript
// components/ConceptMap2D.tsx
import ReactFlow, { Node, Edge, Background, Controls } from 'reactflow'
import 'reactflow/dist/style.css'
import { useConceptMapStore } from '@/stores/conceptMapStore'
import { useConceptMap } from '@/hooks/useDocuments'

export function ConceptMap2D({ documentIds }: { documentIds: string[] }) {
  const { data: graphData } = useConceptMap(documentIds)
  const { nodes, edges, setGraph } = useConceptMapStore()

  // Sync React Query data to Zustand on load
  useEffect(() => {
    if (graphData) {
      setGraph(graphData.nodes, graphData.edges)
    }
  }, [graphData])

  // Convert to ReactFlow format
  const flowNodes: Node[] = nodes.map(n => ({
    id: n.id,
    position: { x: n.position.x, y: n.position.y },
    data: { label: n.label, ...n.metadata },
    type: getNodeType(n.type)
  }))

  const flowEdges: Edge[] = edges.map(e => ({
    id: `${e.source}-${e.target}`,
    source: e.source,
    target: e.target,
    type: e.type
  }))

  return (
    <div style={{ width: '100%', height: '600px' }}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        fitView
        attributionPosition="bottom-right"
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  )
}

function getNodeType(type: string) {
  switch (type) {
    case 'concept': return 'default'
    case 'basin': return 'output'
    case 'thoughtseed': return 'input'
    default: return 'default'
  }
}
```

**Custom Node Components**:
```typescript
// components/nodes/ConceptNode.tsx
import { Handle, Position } from 'reactflow'

export function ConceptNode({ data }: { data: any }) {
  return (
    <div className="concept-node">
      <Handle type="target" position={Position.Top} />
      <div className="concept-label">{data.label}</div>
      <div className="concept-meta">
        {data.document_count} docs
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}
```

### 6.3 Three.js Implementation (3D Basin Visualization)

**Component Structure**:
```typescript
// components/BasinVisualizer3D.tsx
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import { useConceptMapStore } from '@/stores/conceptMapStore'

export function BasinVisualizer3D() {
  const { nodes, edges } = useConceptMapStore()

  return (
    <Canvas camera={{ position: [0, 0, 100] }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />

      <ForceDirectedGraph nodes={nodes} edges={edges} />

      <OrbitControls />
    </Canvas>
  )
}

// Force-directed layout in 3D using Web Worker
function ForceDirectedGraph({ nodes, edges }: { nodes: any[], edges: any[] }) {
  const [positions, setPositions] = useState<Record<string, [number, number, number]>>({})

  useEffect(() => {
    // Offload force simulation to Web Worker
    const worker = new Worker(new URL('../workers/forceLayout.worker.ts', import.meta.url))

    worker.postMessage({ nodes, edges })

    worker.onmessage = (e) => {
      setPositions(e.data.positions)
    }

    return () => worker.terminate()
  }, [nodes, edges])

  return (
    <>
      {nodes.map(node => (
        <mesh key={node.id} position={positions[node.id] || [0, 0, 0]}>
          <sphereGeometry args={[1, 32, 32]} />
          <meshStandardMaterial color={getNodeColor(node.type)} />
          <Text position={[0, 2, 0]} fontSize={0.5}>
            {node.label}
          </Text>
        </mesh>
      ))}

      {edges.map((edge, i) => (
        <Line
          key={i}
          points={[
            positions[edge.source] || [0, 0, 0],
            positions[edge.target] || [0, 0, 0]
          ]}
          color="white"
          lineWidth={1}
        />
      ))}
    </>
  )
}
```

**Web Worker for Force Layout** (`workers/forceLayout.worker.ts`):
```typescript
// Force-directed simulation in worker to avoid blocking UI
import * as d3 from 'd3-force-3d'

self.onmessage = (e) => {
  const { nodes, edges } = e.data

  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(d => d.id))
    .force('charge', d3.forceManyBody().strength(-30))
    .force('center', d3.forceCenter(0, 0, 0))
    .force('collision', d3.forceCollide(2))

  simulation.on('tick', () => {
    const positions = nodes.reduce((acc, node) => {
      acc[node.id] = [node.x || 0, node.y || 0, node.z || 0]
      return acc
    }, {})

    self.postMessage({ positions })
  })

  simulation.tick(300) // Run simulation for 300 iterations
}
```

### 6.4 Performance Optimizations

**Instanced Rendering for Large Graphs**:
```typescript
// When rendering >1000 nodes
import { Instances, Instance } from '@react-three/drei'

function OptimizedNodeCloud({ nodes }: { nodes: any[] }) {
  return (
    <Instances limit={nodes.length}>
      <sphereGeometry args={[1, 16, 16]} />
      <meshStandardMaterial />

      {nodes.map(node => (
        <Instance key={node.id} position={[node.x, node.y, node.z]} />
      ))}
    </Instances>
  )
}
```

**Level of Detail (LOD)**:
```typescript
// Reduce detail for distant nodes
import { useLOD } from '@react-three/drei'

function AdaptiveNode({ node, distance }: { node: any, distance: number }) {
  const geometry = distance > 50
    ? <sphereGeometry args={[1, 8, 8]} />  // Low poly
    : <sphereGeometry args={[1, 32, 32]} /> // High poly

  return (
    <mesh position={[node.x, node.y, node.z]}>
      {geometry}
      <meshStandardMaterial />
    </mesh>
  )
}
```

---

## 7. Performance Optimization Strategy

### 7.1 Progressive Enhancement Levels

**Level 0: Baseline (All browsers)**
- Basic document list
- Simple 2D concept map (React Flow)
- Standard REST API calls

**Level 1: Enhanced (Modern browsers)**
- SSE for real-time updates
- Optimistic UI updates
- React Query caching

**Level 2: Advanced (High-performance devices)**
- 3D basin visualization (Three.js)
- Web Workers for heavy computation
- Virtual scrolling for large lists

**Level 3: Maximum (Desktop, powerful GPUs)**
- Instanced rendering (>1000 nodes)
- Advanced shader effects
- Real-time particle systems

### 7.2 Performance Budgets

**Critical Metrics**:
```typescript
// Define performance budgets
const PERFORMANCE_BUDGETS = {
  // Time budgets
  CONCEPT_MAP_RENDER: 500, // ms
  REAL_TIME_UPDATE: 100,   // ms
  BULK_UPLOAD_100_FILES: 2000, // ms
  KNOWLEDGE_SEARCH: 200,   // ms
  BASIN_ACTIVATION: 1000,  // ms

  // Resource budgets
  MAX_NODES_2D: 500,       // nodes before switching to 3D
  MAX_NODES_3D: 5000,      // nodes before LOD/culling
  MAX_CONCURRENT_UPLOADS: 10, // parallel file uploads

  // Bundle budgets
  INITIAL_JS_BUNDLE: 250,  // KB (gzipped)
  INITIAL_CSS_BUNDLE: 50,  // KB (gzipped)
  MAX_CHUNK_SIZE: 100      // KB (gzipped)
}
```

### 7.3 Web Worker Strategy

**Worker 1: Force Layout Computation**
```typescript
// workers/forceLayout.worker.ts
// Handles force-directed graph layout
// Input: nodes, edges
// Output: updated positions every tick
```

**Worker 2: Bulk File Processing**
```typescript
// workers/fileProcessor.worker.ts
// Handles client-side file parsing (optional local processing)
// Input: File objects
// Output: extracted text, metadata
```

**Worker 3: Search Index**
```typescript
// workers/searchIndex.worker.ts
// Maintains client-side search index for knowledge manager
// Input: documents, concepts
// Output: search results with ranking
```

### 7.4 Code Splitting Strategy

**Route-Based Splitting**:
```typescript
// App routing with lazy loading
import { lazy, Suspense } from 'react'

const DocumentUpload = lazy(() => import('./pages/DocumentUpload'))
const KnowledgeBase = lazy(() => import('./pages/KnowledgeBase'))
const ConceptExplorer = lazy(() => import('./pages/ConceptExplorer'))
const NotebookEditor = lazy(() => import('./pages/NotebookEditor'))

// Only load what's needed
function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/upload" element={<DocumentUpload />} />
        <Route path="/concepts" element={<ConceptExplorer />} />
        <Route path="/knowledge" element={<KnowledgeBase />} />
        <Route path="/notebook" element={<NotebookEditor />} />
      </Routes>
    </Suspense>
  )
}
```

**Component-Level Splitting**:
```typescript
// Heavy visualization components loaded on demand
const BasinVisualizer3D = lazy(() => import('./components/BasinVisualizer3D'))

function ConceptExplorer() {
  const [show3D, setShow3D] = useState(false)

  return (
    <>
      <ConceptMap2D />
      {show3D && (
        <Suspense fallback={<LoadingSpinner />}>
          <BasinVisualizer3D />
        </Suspense>
      )}
    </>
  )
}
```

### 7.5 Virtual Scrolling

**For Document Lists (>100 items)**:
```typescript
// components/VirtualDocumentList.tsx
import { useVirtualizer } from '@tanstack/react-virtual'

export function VirtualDocumentList({ documents }: { documents: any[] }) {
  const parentRef = useRef<HTMLDivElement>(null)

  const virtualizer = useVirtualizer({
    count: documents.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 80, // Estimated row height
    overscan: 5 // Render 5 extra items for smooth scrolling
  })

  return (
    <div ref={parentRef} style={{ height: '600px', overflow: 'auto' }}>
      <div style={{ height: `${virtualizer.getTotalSize()}px`, position: 'relative' }}>
        {virtualizer.getVirtualItems().map(virtualItem => (
          <div
            key={virtualItem.index}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${virtualItem.size}px`,
              transform: `translateY(${virtualItem.start}px)`
            }}
          >
            <DocumentCard document={documents[virtualItem.index]} />
          </div>
        ))}
      </div>
    </div>
  )
}
```

---

## 8. Backend Integration Patterns

### 8.1 REST API Endpoints

**Document Management**:
```
POST   /api/documents              # Upload documents (bulk)
GET    /api/documents              # List documents (with filters)
GET    /api/documents/{id}         # Get document details
DELETE /api/documents/{id}         # Delete document
```

**Concept & Basin Operations**:
```
GET    /api/concepts/graph         # Get concept graph data
GET    /api/concepts/{id}          # Get concept details
POST   /api/basins/activate        # Trigger basin activation
GET    /api/basins/{id}            # Get basin state
POST   /api/basins/{id}/strengthen # Strengthen basin connections
```

**Knowledge Manager**:
```
GET    /api/knowledge/timeline     # Get concept evolution timeline
GET    /api/knowledge/search       # Full-text search across distillations
POST   /api/knowledge/link         # Create bidirectional link
DELETE /api/knowledge/link/{id}    # Remove link
```

**Narrative Extraction**:
```
POST   /api/narrative/extract      # Extract archetypes/sentiment
POST   /api/narrative/metaphor     # Generate isomorphic metaphors
GET    /api/narrative/curiosity    # Get curiosity prompts
```

**Generation Controls**:
```
POST   /api/generation/narrow      # Narrow focus on specific concept
POST   /api/generation/deeper      # Deepen analysis
POST   /api/generation/wider       # Broaden exploration
POST   /api/generation/dual        # Dual-concept analysis
```

### 8.2 SSE Streaming Endpoints

**Processing Stream**:
```python
# backend/src/api/routes/stream.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import json

router = APIRouter()

@router.get("/stream/processing")
async def stream_processing():
    """Stream document processing events to frontend."""
    async def event_generator():
        while True:
            # Check for new events from processing queue
            event = await get_next_processing_event()
            if event:
                yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Concept Map Stream**:
```python
@router.get("/stream/concept-map")
async def stream_concept_map():
    """Stream real-time concept map updates."""
    async def event_generator():
        while True:
            event = await get_concept_map_update()
            if event:
                yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### 8.3 Error Handling & Retry Strategy

**Client-Side Retry Logic**:
```typescript
// lib/apiClient.ts
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000
})

// Retry logic for failed requests
api.interceptors.response.use(
  response => response,
  async error => {
    const config = error.config

    // Retry on network errors or 5xx
    if (!config.__retryCount && (error.code === 'ECONNABORTED' || error.response?.status >= 500)) {
      config.__retryCount = (config.__retryCount || 0) + 1

      if (config.__retryCount <= 3) {
        await new Promise(resolve => setTimeout(resolve, 1000 * config.__retryCount))
        return api(config)
      }
    }

    return Promise.reject(error)
  }
)
```

**SSE Reconnection Strategy**:
```typescript
// hooks/useSSEStream.ts (extended)
useEffect(() => {
  let reconnectAttempts = 0
  const maxReconnectAttempts = 5

  const connect = () => {
    const eventSource = new EventSource(url)

    eventSource.onerror = () => {
      eventSource.close()

      if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++
        const delay = Math.min(1000 * 2 ** reconnectAttempts, 30000)
        setTimeout(connect, delay)
      }
    }

    return eventSource
  }

  const eventSource = connect()
  return () => eventSource.close()
}, [url])
```

---

## 9. Component Architecture

### 9.1 Feature-Based Organization

```
frontend/src/
├── components/
│   ├── common/              # Shared UI components
│   │   ├── Button.tsx
│   │   ├── Modal.tsx
│   │   ├── LoadingSpinner.tsx
│   │   └── ErrorBoundary.tsx
│   │
│   ├── document/            # Document-related components
│   │   ├── DocumentList.tsx
│   │   ├── DocumentCard.tsx
│   │   ├── UploadModal.tsx
│   │   └── BulkUploadProgress.tsx
│   │
│   ├── concept/             # Concept visualization
│   │   ├── ConceptMap2D.tsx
│   │   ├── ConceptNode.tsx
│   │   ├── BasinVisualizer3D.tsx
│   │   └── ConceptDetailPanel.tsx
│   │
│   ├── knowledge/           # Knowledge manager
│   │   ├── KnowledgeManager.tsx
│   │   ├── TimelineView.tsx
│   │   ├── BidirectionalLinks.tsx
│   │   └── SearchInterface.tsx
│   │
│   ├── narrative/           # Narrative extraction
│   │   ├── NarrativeFeed.tsx
│   │   ├── ArchetypePanel.tsx
│   │   ├── MetaphorGenerator.tsx
│   │   └── CuriosityPrompts.tsx
│   │
│   └── synthesis/           # Synthesis workspace
│       ├── GenerationControls.tsx
│       ├── DualConceptAnalysis.tsx
│       └── ZettelkastenView.tsx
│
├── pages/
│   ├── Dashboard.tsx
│   ├── DocumentUpload.tsx
│   ├── KnowledgeBase.tsx
│   ├── ConceptExplorer.tsx
│   ├── NotebookEditor.tsx
│   └── SynthesisWorkspace.tsx
│
├── hooks/
│   ├── useSSEStream.ts
│   ├── useDocuments.ts
│   ├── useConceptMap.ts
│   ├── useKnowledge.ts
│   └── useNarrative.ts
│
├── stores/
│   ├── uiStore.ts
│   ├── conceptMapStore.ts
│   └── knowledgeStore.ts
│
├── workers/
│   ├── forceLayout.worker.ts
│   ├── fileProcessor.worker.ts
│   └── searchIndex.worker.ts
│
└── lib/
    ├── apiClient.ts
    ├── queryClient.ts
    └── schemas.ts (Zod schemas)
```

### 9.2 Component Composition Pattern

**Container/Presenter Pattern**:
```typescript
// Container: Handles data fetching and business logic
function ConceptExplorerContainer() {
  const { data: concepts, isLoading } = useConceptMap([])
  const { nodes, edges } = useConceptMapStore()

  if (isLoading) return <LoadingSpinner />

  return <ConceptExplorerPresenter nodes={nodes} edges={edges} />
}

// Presenter: Pure UI component
function ConceptExplorerPresenter({ nodes, edges }: { nodes: any[], edges: any[] }) {
  return (
    <div className="concept-explorer">
      {nodes.length < 500 ? (
        <ConceptMap2D nodes={nodes} edges={edges} />
      ) : (
        <BasinVisualizer3D nodes={nodes} edges={edges} />
      )}
    </div>
  )
}
```

### 9.3 Error Boundary Strategy

**Top-Level Error Boundary**:
```typescript
// components/common/ErrorBoundary.tsx
import { Component, ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('ErrorBoundary caught:', error, errorInfo)
    // Log to error tracking service
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="error-fallback">
          <h2>Something went wrong</h2>
          <button onClick={() => this.setState({ hasError: false, error: null })}>
            Try again
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
```

**Feature-Level Error Boundaries**:
```typescript
// Wrap each major feature
function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/upload" element={
            <ErrorBoundary fallback={<UploadError />}>
              <DocumentUpload />
            </ErrorBoundary>
          } />

          <Route path="/concepts" element={
            <ErrorBoundary fallback={<ConceptMapError />}>
              <ConceptExplorer />
            </ErrorBoundary>
          } />
        </Routes>
      </Router>
    </ErrorBoundary>
  )
}
```

---

## 10. Data Flow Diagrams

### 10.1 Bulk Upload Flow

```
┌──────────────────┐
│  User selects    │
│  folder (100+    │
│  files)          │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  BulkUploadModal                     │
│  - Validates file types              │
│  - Shows file count preview          │
│  - Checks processing mode toggle     │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Processing Mode Decision            │
│  ┌──────────┬──────────┐            │
│  │  Local   │  Cloud   │            │
│  └─────┬────┴─────┬────┘            │
└────────┼──────────┼──────────────────┘
         │          │
         │          ▼
         │  ┌──────────────────────┐
         │  │  Warn user:          │
         │  │  "Cloud processing   │
         │  │  may be slower"      │
         │  └──────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Upload Strategy                     │
│  - Batch files in chunks of 10       │
│  - Upload chunks in parallel         │
│  - Show progress per file            │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Backend: POST /api/documents        │
│  - Receives FormData with files      │
│  - Triggers LangGraph processor      │
│  - Emits SSE events for progress     │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  SSE Stream: /api/stream/processing  │
│  Events:                             │
│  - processing_started                │
│  - extraction_complete               │
│  - basin_activated                   │
│  - processing_complete               │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Frontend Update                     │
│  - Update progress bar per file      │
│  - Show concept count incrementally  │
│  - Display basin activation status   │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Post-Upload Summary Panel           │
│  - Total concepts extracted          │
│  - Basins created                    │
│  - Related documents found           │
│  - "View in Concept Explorer" button │
└──────────────────────────────────────┘
```

### 10.2 Real-Time Concept Map Update Flow

```
┌──────────────────┐
│  User types in   │
│  notebook or     │
│  selects text    │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Debounced Input Handler (300ms)    │
│  - Prevent excessive API calls      │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  POST /api/narrative/extract         │
│  Body: { text, context }             │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Backend: Consciousness Processing   │
│  - Extract concepts                  │
│  - Detect archetypes                 │
│  - Calculate sentiment               │
│  - Trigger basin activation          │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  SSE: /api/stream/concept-map        │
│  Event: concept_added                │
│  Data: { nodes, edges, basin_state } │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Zustand Store Update                │
│  conceptMapStore.addNode(newNode)    │
│  - Optimistic update                 │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  React Flow Re-render                │
│  - New node appears instantly        │
│  - Layout algorithm adjusts          │
│  - Smooth animation                  │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Narrative Feed Update               │
│  - Archetype panel shows new tags    │
│  - Sentiment meter updates           │
│  - Related concepts highlighted      │
└──────────────────────────────────────┘
```

### 10.3 Knowledge Manager Bidirectional Link Flow

```
┌──────────────────┐
│  User creates    │
│  link between    │
│  Concept A and   │
│  Distillation B  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  POST /api/knowledge/link            │
│  Body: {                             │
│    source_id: "concept_A",           │
│    target_id: "distillation_B",      │
│    link_type: "explains"             │
│  }                                   │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Backend: Create Bidirectional Link  │
│  - Store in Neo4j knowledge graph    │
│  - Create reverse link automatically │
│  - Update timeline                   │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  React Query: Invalidate Caches      │
│  - ['knowledge', conceptId]          │
│  - ['knowledge', distillationId]     │
│  - ['knowledge', 'timeline']         │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  UI Updates                          │
│  - Concept A shows "Explained by B"  │
│  - Distillation B shows "Explains A" │
│  - Timeline shows new link event     │
│  - Graph visualization updates       │
└──────────────────────────────────────┘
```

---

## 11. Testing Strategy

### 11.1 Test Pyramid

```
          ┌─────────────┐
          │    E2E      │  <-- Playwright (critical paths)
          │   Tests     │      10% of tests
          └─────────────┘
        ┌───────────────────┐
        │   Integration     │  <-- React Testing Library
        │      Tests        │      30% of tests
        └───────────────────┘
    ┌─────────────────────────────┐
    │      Unit Tests             │  <-- Jest + RTL
    │  (Components, Hooks, Utils) │      60% of tests
    └─────────────────────────────┘
```

### 11.2 Unit Testing Strategy

**Component Tests**:
```typescript
// __tests__/ConceptMap2D.test.tsx
import { render, screen } from '@testing-library/react'
import { ConceptMap2D } from '@/components/ConceptMap2D'

describe('ConceptMap2D', () => {
  it('renders nodes and edges', () => {
    const nodes = [
      { id: '1', label: 'Concept A', type: 'concept', position: { x: 0, y: 0 } }
    ]
    const edges = []

    render(<ConceptMap2D documentIds={['doc1']} />)

    expect(screen.getByText('Concept A')).toBeInTheDocument()
  })

  it('switches to 3D view when node count exceeds 500', () => {
    const nodes = Array.from({ length: 501 }, (_, i) => ({
      id: `${i}`,
      label: `Concept ${i}`,
      type: 'concept',
      position: { x: 0, y: 0 }
    }))

    render(<ConceptMap2D documentIds={['doc1']} />)

    expect(screen.getByTestId('basin-visualizer-3d')).toBeInTheDocument()
  })
})
```

**Hook Tests**:
```typescript
// __tests__/useSSEStream.test.tsx
import { renderHook, waitFor } from '@testing-library/react'
import { useSSEStream } from '@/hooks/useSSEStream'

describe('useSSEStream', () => {
  it('connects to SSE endpoint and receives messages', async () => {
    const onMessage = jest.fn()

    const { result } = renderHook(() => useSSEStream({
      url: '/api/stream/test',
      schema: z.object({ type: z.string() }),
      onMessage
    }))

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    })

    // Mock SSE message
    global.EventSource.mockSendMessage({ type: 'test' })

    expect(onMessage).toHaveBeenCalledWith({ type: 'test' })
  })
})
```

### 11.3 Integration Testing

**API Integration Tests**:
```typescript
// __tests__/integration/documentUpload.test.tsx
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DocumentUpload } from '@/pages/DocumentUpload'
import { server } from '@/mocks/server'

describe('Document Upload Flow', () => {
  beforeAll(() => server.listen())
  afterEach(() => server.resetHandlers())
  afterAll(() => server.close())

  it('uploads file and shows processing progress', async () => {
    render(<DocumentUpload />)

    const file = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
    const input = screen.getByLabelText(/upload/i)

    await userEvent.upload(input, file)

    expect(screen.getByText('Uploading test.pdf')).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText('Upload complete')).toBeInTheDocument()
    }, { timeout: 5000 })
  })
})
```

### 11.4 E2E Testing (Playwright)

**Critical User Journeys**:
```typescript
// e2e/conceptExplorer.spec.ts
import { test, expect } from '@playwright/test'

test('Concept Explorer: Select text triggers concept map update', async ({ page }) => {
  await page.goto('/concepts')

  // Wait for concept map to load
  await page.waitForSelector('[data-testid="concept-map"]')

  // Select text in notebook
  await page.fill('[data-testid="notebook-editor"]', 'machine learning neural networks')

  // Trigger concept extraction
  await page.click('[data-testid="extract-concepts"]')

  // Wait for SSE update
  await page.waitForSelector('[data-testid="concept-node-machine-learning"]')

  // Verify concept appears
  const conceptNode = page.locator('[data-testid="concept-node-machine-learning"]')
  await expect(conceptNode).toBeVisible()

  // Verify basin activation
  const basinPanel = page.locator('[data-testid="basin-panel"]')
  await expect(basinPanel).toContainText('Basin activated')
})

test('Bulk Upload: 100 files progress tracking', async ({ page }) => {
  await page.goto('/upload')

  // Generate 100 test files
  const files = Array.from({ length: 100 }, (_, i) => ({
    name: `test-${i}.txt`,
    mimeType: 'text/plain',
    buffer: Buffer.from(`Test content ${i}`)
  }))

  // Simulate folder drop
  await page.setInputFiles('[data-testid="file-input"]', files)

  // Verify progress bar
  const progress = page.locator('[data-testid="upload-progress"]')
  await expect(progress).toBeVisible()

  // Wait for completion (should be <2s per requirement)
  await expect(progress).toHaveText('100/100 files uploaded', { timeout: 5000 })

  // Verify concept summary panel
  const summary = page.locator('[data-testid="concept-summary"]')
  await expect(summary).toContainText('concepts extracted')
})
```

### 11.5 Performance Testing

**Lighthouse CI Integration**:
```yaml
# .github/workflows/lighthouse.yml
name: Lighthouse CI
on: [pull_request]

jobs:
  lighthouse:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm run build
      - run: npx @lhci/cli@0.12.x autorun
        env:
          LHCI_GITHUB_APP_TOKEN: ${{ secrets.LHCI_GITHUB_APP_TOKEN }}
```

**Performance Benchmarks**:
```typescript
// e2e/performance.spec.ts
import { test, expect } from '@playwright/test'

test('Concept map renders in <500ms', async ({ page }) => {
  await page.goto('/concepts')

  const start = Date.now()
  await page.waitForSelector('[data-testid="concept-map"]')
  const renderTime = Date.now() - start

  expect(renderTime).toBeLessThan(500)
})

test('Real-time update latency <100ms', async ({ page }) => {
  await page.goto('/concepts')

  // Trigger concept extraction
  const start = Date.now()
  await page.fill('[data-testid="notebook-editor"]', 'test')

  // Wait for SSE event and UI update
  await page.waitForSelector('[data-testid="new-concept"]')
  const latency = Date.now() - start

  expect(latency).toBeLessThan(100)
})
```

---

## 12. Security & Privacy

### 12.1 Input Validation

**Client-Side Validation**:
```typescript
// lib/validators.ts
import { z } from 'zod'

export const FileUploadSchema = z.object({
  files: z.array(z.instanceof(File))
    .min(1, 'At least one file required')
    .max(100, 'Maximum 100 files per upload')
    .refine(
      (files) => files.every(f => f.size < 50 * 1024 * 1024),
      'Each file must be less than 50MB'
    )
    .refine(
      (files) => files.every(f =>
        ['application/pdf', 'text/plain', 'text/markdown'].includes(f.type)
      ),
      'Only PDF, TXT, and MD files allowed'
    ),
  tags: z.array(z.string()).optional()
})
```

**Server-Side Validation** (already in place):
```python
# backend/src/api/routes/documents.py
# - File size limits
# - Content type validation
# - Empty file rejection
```

### 12.2 Rate Limiting

**Client-Side Throttling**:
```typescript
// hooks/useThrottle.ts
import { useCallback, useRef } from 'react'

export function useThrottle<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T {
  const lastRun = useRef(Date.now())

  return useCallback((...args: any[]) => {
    const now = Date.now()

    if (now - lastRun.current >= delay) {
      lastRun.current = now
      return callback(...args)
    }
  }, [callback, delay]) as T
}

// Usage: Limit narrative extraction to 1 req/second
const extractNarrative = useThrottle(async (text: string) => {
  await axios.post('/api/narrative/extract', { text })
}, 1000)
```

### 12.3 XSS Prevention

**Content Sanitization**:
```typescript
// lib/sanitize.ts
import DOMPurify from 'dompurify'

export function sanitizeHTML(html: string): string {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
    ALLOWED_ATTR: ['href', 'target']
  })
}

// Usage in components
function ConceptDescription({ html }: { html: string }) {
  return (
    <div dangerouslySetInnerHTML={{ __html: sanitizeHTML(html) }} />
  )
}
```

### 12.4 CORS Configuration

**Backend CORS Setup** (FastAPI):
```python
# backend/src/app_factory.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 13. Deployment & DevOps

### 13.1 Build Configuration

**Vite Production Build**:
```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    target: 'es2020',
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'visualization': ['reactflow', 'three', '@react-three/fiber'],
          'state': ['zustand', '@tanstack/react-query']
        }
      }
    },
    chunkSizeWarningLimit: 1000
  }
})
```

### 13.2 Environment Variables

**Frontend (.env)**:
```bash
VITE_API_URL=http://localhost:9127
VITE_SSE_URL=http://localhost:9127/api/stream
VITE_ENABLE_3D_VISUALIZATION=true
VITE_MAX_UPLOAD_FILES=100
```

**Backend (.env)**:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379
```

### 13.3 CI/CD Pipeline

**GitHub Actions**:
```yaml
# .github/workflows/test-and-deploy.yml
name: Test and Deploy
on: [push, pull_request]

jobs:
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: cd frontend && npm ci
      - run: npm run lint
      - run: npm test
      - run: npm run build

  e2e-tests:
    runs-on: ubuntu-latest
    needs: frontend-tests
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npx playwright install --with-deps
      - run: npm run test:e2e

  deploy:
    runs-on: ubuntu-latest
    needs: [frontend-tests, e2e-tests]
    if: github.ref == 'refs/heads/main'
    steps:
      - run: echo "Deploy to production"
```

---

## 14. Migration Path

### 14.1 Phase 0: Foundation (Week 1)

**Goal**: Stabilize existing codebase

**Tasks**:
- ✅ Complete lint cleanup (Story #1)
- ✅ Establish test framework (Jest + Playwright)
- ✅ Set up CI/CD pipeline
- ✅ Document API contracts

**Success Criteria**:
- `npm run lint` passes with 0 warnings
- `npm test` achieves >60% coverage
- All smoke tests passing

### 14.2 Phase 1: Real-Time Infrastructure (Weeks 1-2)

**Goal**: Implement SSE streaming and state management

**Tasks**:
- Implement SSE endpoints (`/api/stream/*`)
- Create `useSSEStream` hook
- Set up Zustand stores
- Configure React Query
- Write Zod schemas for type safety

**Success Criteria**:
- SSE connection stable for >5 minutes
- State updates propagate within 100ms
- Type safety enforced with Zod

### 14.3 Phase 2: Bulk Upload (Week 2-3)

**Goal**: Enable folder drag/drop with progress tracking

**Tasks**:
- Implement `BulkUploadModal` component
- Add folder input handling
- Create batch upload strategy (10 files/batch)
- Add progress bars and status indicators
- Implement local/cloud processing toggle

**Success Criteria**:
- 100 files upload in <2s
- Progress tracking accurate
- Concept summary panel shows results

### 14.4 Phase 3: Concept Visualization (Weeks 3-5)

**Goal**: Interactive concept map with real-time updates

**Tasks**:
- Integrate React Flow for 2D visualization
- Create custom node components
- Implement SSE-driven map updates
- Add Three.js for 3D (>500 nodes)
- Create force-directed layout worker

**Success Criteria**:
- Concept map renders in <500ms
- Real-time updates within 100ms
- Smooth transitions between 2D/3D

### 14.5 Phase 4: Knowledge Manager (Weeks 4-5)

**Goal**: Bidirectional links and timeline view

**Tasks**:
- Create knowledge manager UI
- Implement bidirectional link storage (Neo4j)
- Build timeline visualization
- Add full-text search
- Create Zettelkasten view

**Success Criteria**:
- Search results <200ms
- Timeline shows concept evolution
- Bidirectional links work both ways

### 14.6 Phase 5: Advanced Features (Weeks 5-6)

**Goal**: Narrative extraction and synthesis workspace

**Tasks**:
- Implement narrative extraction API
- Create archetype/sentiment panels
- Build metaphor generator
- Add generation controls (narrow/deeper/wider)
- Create dual-concept analysis

**Success Criteria**:
- Narrative updates in real-time
- Generation controls produce results
- Metaphors are contextually relevant

---

## 15. Risks & Mitigation

### 15.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **SSE connection instability** | High | Medium | Automatic reconnection, exponential backoff, fallback to polling |
| **Graph rendering performance** | High | Medium | Progressive enhancement (2D → 3D), LOD, instanced rendering, Web Workers |
| **Browser memory limits** | Medium | Low | Virtual scrolling, pagination, automatic cleanup, memory monitoring |
| **WebSocket scalability** | Low | Low | Using SSE instead (stateless, HTTP/2) |
| **Type safety gaps** | Medium | Medium | Zod runtime validation, strict TypeScript config |

### 15.2 UX Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Bulk upload too slow** | High | Medium | Parallel uploads (10/batch), progress feedback, local processing option |
| **Concept map overwhelming** | Medium | High | Start simple (2D), gradual complexity, filtering/search |
| **Real-time updates distracting** | Low | Low | Smooth animations, debouncing, user control (pause updates) |

### 15.3 Integration Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Backend API changes** | Medium | Low | Zod schema validation, versioned API endpoints |
| **Neo4j performance** | Medium | Medium | Indexing, query optimization, caching layer |
| **LangGraph processing slow** | High | Medium | Async processing, SSE updates, local processing mode |

---

## 16. Success Metrics

### 16.1 Performance Metrics

**Baseline Targets** (from Epic):
- Bulk upload: <2s for 100 files ✅
- Concept map render: <500ms ✅
- Real-time updates: <100ms latency ✅
- Basin activation: <1s ✅
- Knowledge search: <200ms ✅

**Monitoring Plan**:
- Lighthouse CI on every PR
- Real User Monitoring (RUM) in production
- Synthetic monitoring for critical paths

### 16.2 Quality Metrics

**Code Quality**:
- Lint: 0 warnings ✅
- Test coverage: >80% ✅
- E2E coverage: All critical paths ✅
- Zero console errors ✅

**Stability**:
- SSE uptime: >99%
- Error rate: <0.1%
- Mean time to recovery: <5 minutes

### 16.3 User Experience Metrics

**Engagement**:
- Average session duration
- Concepts explored per session
- Documents uploaded per user

**Satisfaction**:
- Task completion rate
- Error recovery rate
- Feature adoption rate

---

## 17. Appendices

### Appendix A: Technology Choices Comparison

#### Real-Time Communication

| Technology | Pros | Cons | Decision |
|------------|------|------|----------|
| **WebSockets** | Bidirectional, low latency | Complex server, stateful | ❌ Not needed |
| **SSE** | Simple, auto-reconnect, HTTP/2 | Unidirectional only | ✅ **CHOSEN** |
| **Long Polling** | Broad support | High overhead, inefficient | ❌ Legacy approach |

#### State Management

| Technology | Pros | Cons | Decision |
|------------|------|------|----------|
| **Redux** | Powerful, ecosystem | Boilerplate, complex | ❌ Overkill |
| **Zustand** | Simple, lightweight | Less ecosystem | ✅ **CHOSEN** |
| **Context API** | Built-in, no deps | Performance issues | ❌ Too limited |
| **React Query** | Server state caching | Not for UI state | ✅ **CHOSEN** (hybrid) |

#### Graph Visualization

| Technology | Pros | Cons | Decision |
|------------|------|------|----------|
| **D3.js** | Powerful, flexible | Steep learning curve | ❌ Too complex |
| **React Flow** | React-friendly, fast | 2D only | ✅ **CHOSEN** (2D) |
| **Cytoscape** | Network analysis | Heavy bundle | ❌ Too specialized |
| **Three.js** | 3D, WebGL | Complex for simple cases | ✅ **CHOSEN** (3D) |

### Appendix B: Glossary

- **Basin**: Attractor basin in consciousness processing system
- **ThoughtSeed**: Emergent cognitive pattern from document processing
- **Concept Map**: Graph visualization of concepts and relationships
- **Narrative Extraction**: Archetype/sentiment/metaphor detection
- **Bidirectional Link**: Two-way reference between concepts and distillations
- **Zettelkasten**: Note-taking method with interconnected knowledge cards
- **SSE**: Server-Sent Events (unidirectional HTTP streaming)
- **LOD**: Level of Detail (performance optimization technique)

### Appendix C: References

- [React Flow Documentation](https://reactflow.dev/)
- [Three.js Documentation](https://threejs.org/docs/)
- [Zustand Documentation](https://docs.pmnd.rs/zustand/)
- [TanStack Query (React Query)](https://tanstack.com/query/latest)
- [Server-Sent Events Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [Zod Schema Validation](https://zod.dev/)
- [FastAPI WebSockets/SSE](https://fastapi.tiangolo.com/advanced/websockets/)

---

## Document Approval

**Author**: Winston (Architect) 🏗️
**Reviewers**: Murat (TEA), John (PM), Bob (SM)
**Status**: Draft → Ready for Review
**Next Steps**:
1. Team review session
2. Finalize technology selections
3. Create detailed tech specs for each story
4. Begin Phase 0 implementation

**Questions/Feedback**: Document in Epic issue tracker

---

*End of High-Level Architecture Document*
