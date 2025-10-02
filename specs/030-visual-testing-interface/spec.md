# Spec 030: Visual Testing Interface for Paper Processing

**Status**: DRAFT
**Priority**: HIGH
**Dependencies**: 027 (Basin Strengthening), 028 (ThoughtSeeds), 029 (Curiosity Agents), 006 (Query System)
**Created**: 2025-10-01

## Overview

Create a real-time visual interface for testing and monitoring paper processing during bulk upload. The interface shows:

1. **Agent orchestration workflow** (handoff patterns, task delegation)
2. **Knowledge graph updates** (concepts, relationships, basins)
3. **ThoughtSeed propagation** (cross-document linking, pattern emergence)
4. **Curiosity triggers** (background agents, question generation)
5. **Quality metrics** (relationship extraction, basin strengthening)

**User Involvement**: User can upload a paper, watch the entire processing pipeline in real-time, and see agent decisions/handoffs as they happen.

## Problem Statement

Currently, document processing happens in a "black box":
- No visibility into which agents are active
- Can't see ThoughtSeeds being created/propagated
- Can't watch basins strengthen in real-time
- No way to validate relationship extraction quality
- Can't observe curiosity triggers spawning background agents

The user wants:
> "I'd like to see the interface even. I want to be involved in the testing iterations by putting a paper in and seeing what happens to it."

## Agent Orchestration Patterns (Following User Guidelines)

### Handoff Orchestration Pattern

The document processing pipeline uses **step-by-step handoff orchestration** where specialized agents pass control sequentially:

```
Upload → Daedalus → Extractor → Analyst → Synthesizer → Storage → Done
         [Gateway]   [Concepts]  [Quality]  [Relations]  [Neo4j]
```

**Handoff Protocol**:
1. **Explicit payload**: Each agent receives complete state (document, concepts, relationships, metadata)
2. **Versioned context**: State includes version number for tracking
3. **Transfer reason**: Logs why handoff occurred ("extraction complete", "quality validated", etc.)
4. **Single active agent**: Only one agent processes at a time for deterministic behavior

### Asynchronous Coordination Pattern

Background curiosity agents use **asynchronous coordination via shared graph**:
- Main pipeline continues processing documents
- Curiosity agents run independently, reading/writing Neo4j
- Redis task queue coordinates background work
- No direct communication between agents (graph is shared context)

### Graph-Based Multi-Agent Framework

Agents represented as nodes, tasks as edges:
```
(Daedalus)-[:HANDS_OFF_TO]->(Extractor)
(Extractor)-[:HANDS_OFF_TO]->(Analyst)
(Analyst)-[:DELEGATES_TO]->(CuriosityAgent)  [async, doesn't block]
```

## Requirements

### Functional Requirements

#### FR1: Real-Time Agent Orchestration Visualization
**Description**: Show active agent, handoffs, and task delegation
**Acceptance Criteria**:
- [ ] Timeline shows sequential agent activations
- [ ] Handoff events displayed: "Daedalus → Extractor (concepts ready)"
- [ ] Current active agent highlighted
- [ ] Background agents shown separately (curiosity agents)
- [ ] Agent state: idle/processing/complete/failed

#### FR2: Knowledge Graph Live Updates
**Description**: Watch concepts, relationships, and basins appear in real-time
**Acceptance Criteria**:
- [ ] Concept nodes appear as extracted
- [ ] Relationships drawn with confidence scores
- [ ] Basin nodes show strength increasing when concepts reappear
- [ ] Graph layout auto-updates (force-directed or hierarchical)
- [ ] User can click nodes to see details

#### FR3: ThoughtSeed Propagation Tracking
**Description**: Visualize ThoughtSeed creation and cross-document linking
**Acceptance Criteria**:
- [ ] ThoughtSeeds appear as green nodes when created
- [ ] Propagation paths shown: Concept → ThoughtSeed → Basin
- [ ] Cross-document links highlighted (different color)
- [ ] ThoughtSeed TTL countdown visible (24 hours)
- [ ] Emergent patterns triggered when 5+ ThoughtSeeds cluster

#### FR4: Curiosity Agent Monitoring
**Description**: Show curiosity triggers and background agent activity
**Acceptance Criteria**:
- [ ] Curiosity triggers appear as orange alerts
- [ ] Background agents shown in separate panel
- [ ] Questions generated displayed in real-time
- [ ] Recommendations appear when ready
- [ ] Resolution status updated (pending → resolved)

#### FR5: Quality Metrics Dashboard
**Description**: Track relationship extraction quality, basin health, agent performance
**Acceptance Criteria**:
- [ ] Overall quality score (0.0-1.0)
- [ ] Relationship extraction metrics (LLM-based vs heuristic)
- [ ] Basin strength distribution (histogram)
- [ ] Agent performance: time per stage, success rate
- [ ] Curiosity resolution rate (% of triggers resolved)

#### FR6: Interactive Testing Controls
**Description**: User can upload paper, pause/resume, inspect state
**Acceptance Criteria**:
- [ ] Upload button for single paper
- [ ] Bulk upload for multiple papers
- [ ] Pause/Resume processing
- [ ] Step-through mode (advance one agent at a time)
- [ ] Export results (JSON, CSV)

### Non-Functional Requirements

#### NFR1: Real-Time Performance
- Graph updates render in <100ms
- Agent state changes visible within 200ms
- Support 100 simultaneous ThoughtSeeds without lag

#### NFR2: Usability
- Interface loads in <2 seconds
- Responsive design (desktop + tablet)
- Clear visual hierarchy (primary flow vs background agents)

#### NFR3: Debugging Support
- Click any agent to see logs
- Click any concept to see extraction context
- Click any relationship to see LLM reasoning
- Replay processing from any checkpoint

## Technical Design

### Architecture

```
Frontend (React + Three.js):
┌──────────────────────────────────────────────────────┐
│  Visual Testing Interface                            │
│                                                      │
│  ┌────────────────────────────────────────────┐     │
│  │ Agent Orchestration Timeline               │     │
│  │ [Daedalus] → [Extractor] → [Analyst] → ... │     │
│  └────────────────────────────────────────────┘     │
│                                                      │
│  ┌────────────────────┬────────────────────────┐    │
│  │ Knowledge Graph    │  ThoughtSeed Panel     │    │
│  │ (3D Force Layout)  │  - Active ThoughtSeeds │    │
│  │                    │  - Cross-doc links     │    │
│  │  (Concept)         │  - Emergent patterns   │    │
│  │   ↓ EXTENDS        │                        │    │
│  │  (Concept)         │                        │    │
│  └────────────────────┴────────────────────────┘    │
│                                                      │
│  ┌────────────────────┬────────────────────────┐    │
│  │ Curiosity Agents   │  Quality Metrics       │    │
│  │ - [Agent 1] (🔬)   │  Overall: 0.87         │    │
│  │ - [Agent 2] (🔬)   │  Relations: 42/50      │    │
│  │ - [Agent 3] (🔬)   │  Basin Avg: 1.45       │    │
│  └────────────────────┴────────────────────────┘    │
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │ Controls                                 │       │
│  │ [Upload Paper] [Pause] [Resume] [Export] │       │
│  └──────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────┘
         │                    ▲
         │ WebSocket          │ Real-time updates
         ▼                    │
┌──────────────────────────────────────────────────────┐
│  Backend (FastAPI + WebSocket)                       │
│  - /api/documents/upload                             │
│  - /ws/processing-events (WebSocket)                 │
│  - Emit events: agent_handoff, concept_extracted,    │
│                 relationship_created, etc.           │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  Document Processing Graph (LangGraph)               │
│  - Each node emits WebSocket event                   │
│  - State changes broadcast to frontend               │
└──────────────────────────────────────────────────────┘
```

### WebSocket Event Schema

#### Agent Handoff Event
```json
{
  "event_type": "agent_handoff",
  "timestamp": "2025-10-01T10:30:15.234Z",
  "from_agent": "daedalus_gateway",
  "to_agent": "concept_extractor",
  "handoff_reason": "perceptual_information_received",
  "payload_size_kb": 1234,
  "state_version": 3
}
```

#### Concept Extracted Event
```json
{
  "event_type": "concept_extracted",
  "timestamp": "2025-10-01T10:30:16.456Z",
  "agent": "concept_extractor",
  "concept": "neural architecture search",
  "confidence": 0.95,
  "basin_match": true,
  "basin_id": "basin_nas_12345",
  "basin_strength_before": 1.6,
  "basin_strength_after": 1.8
}
```

#### Relationship Created Event
```json
{
  "event_type": "relationship_created",
  "timestamp": "2025-10-01T10:30:18.789Z",
  "agent": "relationship_extractor",
  "source_concept": "DARTS",
  "target_concept": "gradient-based optimization",
  "relationship_type": "THEORETICALLY_EXTENDS",
  "confidence": 0.88,
  "llm_reasoning": "DARTS uses continuous relaxation to enable gradient-based search..."
}
```

#### ThoughtSeed Created Event
```json
{
  "event_type": "thoughtseed_created",
  "timestamp": "2025-10-01T10:30:20.123Z",
  "agent": "thoughtseed_generator",
  "thoughtseed_id": "ts_67890",
  "concept": "one-shot NAS",
  "source_document": "doc_nas_2024.pdf",
  "target_basins": ["basin_nas_12345", "basin_meta_learning_67890"],
  "propagation_hops": 2
}
```

#### Curiosity Triggered Event
```json
{
  "event_type": "curiosity_triggered",
  "timestamp": "2025-10-01T10:30:22.456Z",
  "agent": "curiosity_detector",
  "concept": "quantum neural networks",
  "prediction_error": 0.85,
  "gap_type": "no_basin_match",
  "priority": 0.85,
  "background_agent_spawned": true,
  "background_agent_id": "bg_agent_1"
}
```

#### Background Agent Status Event
```json
{
  "event_type": "background_agent_status",
  "timestamp": "2025-10-01T10:30:35.789Z",
  "agent_id": "bg_agent_1",
  "status": "exploring",
  "concept": "quantum neural networks",
  "questions_generated": 4,
  "related_concepts_found": 12,
  "recommendations_generated": 2,
  "progress": 0.75
}
```

### Frontend Components (React)

#### AgentOrchestrationTimeline.tsx
```typescript
interface AgentHandoff {
  timestamp: string;
  fromAgent: string;
  toAgent: string;
  reason: string;
  stateVersion: number;
}

export const AgentOrchestrationTimeline: React.FC = () => {
  const [handoffs, setHandoffs] = useState<AgentHandoff[]>([]);
  const [activeAgent, setActiveAgent] = useState<string | null>(null);

  useEffect(() => {
    // WebSocket connection
    const ws = new WebSocket('ws://localhost:8000/ws/processing-events');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.event_type === 'agent_handoff') {
        setHandoffs(prev => [...prev, data]);
        setActiveAgent(data.to_agent);
      }
    };

    return () => ws.close();
  }, []);

  return (
    <div className="timeline">
      {handoffs.map((handoff, idx) => (
        <div key={idx} className="handoff-event">
          <div className="from-agent">{handoff.fromAgent}</div>
          <div className="arrow">→</div>
          <div className={`to-agent ${activeAgent === handoff.toAgent ? 'active' : ''}`}>
            {handoff.toAgent}
          </div>
          <div className="reason">{handoff.reason}</div>
        </div>
      ))}
    </div>
  );
};
```

#### KnowledgeGraphViz.tsx
```typescript
import { Canvas } from '@react-three/fiber';
import { ForceGraph3D } from 'react-force-graph';

interface GraphNode {
  id: string;
  type: 'concept' | 'thoughtseed' | 'basin' | 'pattern';
  label: string;
  strength?: number;
  color: string;
}

interface GraphLink {
  source: string;
  target: string;
  type: string;
  confidence: number;
}

export const KnowledgeGraphViz: React.FC = () => {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [links, setLinks] = useState<GraphLink[]>([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/processing-events');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // Add concept nodes
      if (data.event_type === 'concept_extracted') {
        setNodes(prev => [...prev, {
          id: data.concept,
          type: 'concept',
          label: data.concept,
          color: '#4CAF50'
        }]);
      }

      // Add relationships
      if (data.event_type === 'relationship_created') {
        setLinks(prev => [...prev, {
          source: data.source_concept,
          target: data.target_concept,
          type: data.relationship_type,
          confidence: data.confidence
        }]);
      }

      // Add ThoughtSeeds
      if (data.event_type === 'thoughtseed_created') {
        setNodes(prev => [...prev, {
          id: data.thoughtseed_id,
          type: 'thoughtseed',
          label: data.concept,
          color: '#8BC34A'
        }]);

        // Link to basins
        data.target_basins.forEach(basin => {
          setLinks(prev => [...prev, {
            source: data.thoughtseed_id,
            target: basin,
            type: 'ATTRACTED_TO',
            confidence: 1.0
          }]);
        });
      }
    };

    return () => ws.close();
  }, []);

  return (
    <ForceGraph3D
      graphData={{ nodes, links }}
      nodeLabel="label"
      nodeColor="color"
      linkLabel="type"
      linkWidth={link => link.confidence * 3}
      linkDirectionalArrowLength={6}
      onNodeClick={node => {
        // Show node details panel
        console.log('Node clicked:', node);
      }}
    />
  );
};
```

#### CuriosityAgentPanel.tsx
```typescript
interface BackgroundAgent {
  agentId: string;
  concept: string;
  status: 'exploring' | 'resolved' | 'failed';
  questionsGenerated: number;
  relatedConceptsFound: number;
  recommendationsGenerated: number;
  progress: number;
}

export const CuriosityAgentPanel: React.FC = () => {
  const [agents, setAgents] = useState<Map<string, BackgroundAgent>>(new Map());

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/processing-events');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.event_type === 'curiosity_triggered') {
        setAgents(prev => new Map(prev).set(data.background_agent_id, {
          agentId: data.background_agent_id,
          concept: data.concept,
          status: 'exploring',
          questionsGenerated: 0,
          relatedConceptsFound: 0,
          recommendationsGenerated: 0,
          progress: 0
        }));
      }

      if (data.event_type === 'background_agent_status') {
        setAgents(prev => {
          const updated = new Map(prev);
          updated.set(data.agent_id, {
            agentId: data.agent_id,
            concept: data.concept,
            status: data.status,
            questionsGenerated: data.questions_generated,
            relatedConceptsFound: data.related_concepts_found,
            recommendationsGenerated: data.recommendations_generated,
            progress: data.progress
          });
          return updated;
        });
      }
    };

    return () => ws.close();
  }, []);

  return (
    <div className="curiosity-panel">
      <h3>Background Curiosity Agents</h3>
      {Array.from(agents.values()).map(agent => (
        <div key={agent.agentId} className="agent-card">
          <div className="agent-header">
            <span className="agent-icon">🔬</span>
            <span className="concept">{agent.concept}</span>
            <span className={`status ${agent.status}`}>{agent.status}</span>
          </div>
          <div className="agent-progress">
            <div className="progress-bar" style={{ width: `${agent.progress * 100}%` }} />
          </div>
          <div className="agent-stats">
            <span>Questions: {agent.questionsGenerated}</span>
            <span>Concepts: {agent.relatedConceptsFound}</span>
            <span>Recommendations: {agent.recommendationsGenerated}</span>
          </div>
        </div>
      ))}
    </div>
  );
};
```

### Backend WebSocket Integration

```python
# backend/src/api/routes/documents.py

from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast event to all connected clients"""
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/processing-events")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Modify DocumentProcessingGraph to emit events

class DocumentProcessingGraph:
    def __init__(self, websocket_manager: ConnectionManager):
        self.ws_manager = websocket_manager
        # ... rest of init ...

    async def _concept_extraction_node(self, state):
        """Extract concepts and broadcast events"""
        concepts = self.concept_extractor.extract(state["content"])

        for concept in concepts:
            # Emit concept extracted event
            await self.ws_manager.broadcast({
                "event_type": "concept_extracted",
                "timestamp": datetime.now().isoformat(),
                "agent": "concept_extractor",
                "concept": concept.text,
                "confidence": concept.confidence,
                # ... rest of data ...
            })

        return state
```

## Test Strategy

### User Acceptance Testing

```gherkin
Feature: Visual Testing Interface

Scenario: User uploads paper and watches processing
  Given the visual interface is loaded
  When the user uploads "neural_architecture_search.pdf"
  Then they see agent handoff timeline
  And they see concepts appearing in graph
  And they see relationships being created
  And they see ThoughtSeeds propagating
  And they see curiosity agents spawning
  And they see quality metrics updating

Scenario: User inspects concept details
  Given processing is complete
  When the user clicks on "neural architecture search" concept node
  Then they see extraction context
  And they see basin strength history
  And they see related ThoughtSeeds
  And they see cross-document links

Scenario: User monitors curiosity agents
  Given 3 curiosity agents are active
  When the user views curiosity agent panel
  Then they see agent status (exploring/resolved)
  And they see questions generated
  And they see recommendations
  And they see progress bars
```

## Implementation Plan

### Phase 1: WebSocket Backend (3-4 hours)
1. Implement `ConnectionManager` for WebSocket connections
2. Add event emission to `DocumentProcessingGraph` nodes
3. Test WebSocket event broadcasting
4. Document event schemas

### Phase 2: Frontend Components (6-8 hours)
1. Implement `AgentOrchestrationTimeline`
2. Implement `KnowledgeGraphViz` (3D force-directed)
3. Implement `CuriosityAgentPanel`
4. Implement `QualityMetricsDashboard`
5. Wire up WebSocket connections

### Phase 3: Interactive Controls (2-3 hours)
1. Upload button + file handling
2. Pause/Resume processing
3. Step-through mode
4. Export functionality (JSON/CSV)

### Phase 4: Testing & Polish (2-3 hours)
1. User acceptance testing
2. Performance optimization
3. UI/UX polish
4. Documentation

**Total Estimated Time**: 13-18 hours

## Success Criteria

- [ ] Real-time agent handoffs visible in timeline
- [ ] Knowledge graph updates live as concepts extracted
- [ ] ThoughtSeed propagation animated
- [ ] Curiosity agents monitored in separate panel
- [ ] Quality metrics dashboard shows overall health
- [ ] User can pause/resume processing
- [ ] User can click nodes/agents for details
- [ ] WebSocket latency <200ms
- [ ] Interface supports 100+ concurrent graph nodes

## References

- Spec 027: Basin Frequency Strengthening
- Spec 028: ThoughtSeed Generation
- Spec 029: Curiosity-Driven Background Agents
- React Three Fiber: https://docs.pmnd.rs/react-three-fiber
- Force Graph: https://github.com/vasturiano/react-force-graph
- FastAPI WebSockets: https://fastapi.tiangolo.com/advanced/websockets/
