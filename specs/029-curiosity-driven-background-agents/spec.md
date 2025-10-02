# Spec 029: Curiosity-Driven Background Agents Using Local LLM

**Status**: DRAFT
**Priority**: HIGH
**Dependencies**: 027 (Basin Strengthening), 028 (ThoughtSeeds), Active Inference System
**Created**: 2025-10-01

## Overview

Implement autonomous background agents that spawn when prediction errors trigger curiosity, using **local Ollama LLM (free)** instead of expensive API calls (OpenAI/Anthropic). These agents run asynchronously during bulk document processing to:

1. Investigate knowledge gaps discovered during document processing
2. Generate research questions from high prediction-error concepts
3. Search existing knowledge graph for related information
4. Propose new documents to ingest based on curiosity
5. Run continuously in background without blocking document upload

**Key Constraint**: **ZERO COST** - All LLM inference uses local Ollama models (DeepSeek-R1, Qwen2.5, etc.)

## Problem Statement

During bulk document processing, the active inference system detects:
- **Prediction errors**: Concepts that don't match existing knowledge
- **Knowledge gaps**: Missing connections between concepts
- **Novel patterns**: Ideas that don't fit existing basins

Currently, these curiosity triggers are **ignored**. We want to:
1. **Capture curiosity signals** during document processing
2. **Spawn background agents** to investigate each curiosity trigger
3. **Use local LLM** (Ollama) to reason about gaps and generate questions
4. **Queue exploration tasks** for later processing
5. **Never use paid APIs** for curiosity exploration

## Agent Role Definition (Following User Guidelines)

### Specialized Agent Roles for Curiosity-Driven Exploration

#### 1. **Curiosity Detection Agent**
**Responsibility**: Monitor document processing for prediction errors and knowledge gaps
**Input**: Active inference state (beliefs, prediction errors, surprise signals)
**Output**: Curiosity trigger events with severity scores
**Tools**: Active inference engine, basin strength comparison
**Trigger Conditions**:
- Prediction error > 0.7 (high surprise)
- New concept with no basin match
- Contradictory relationships between concepts

#### 2. **Question Generation Agent** (Local LLM)
**Responsibility**: Generate research questions from curiosity triggers
**Input**: Curiosity trigger (concept, context, prediction error)
**Output**: 3-5 research questions to explore
**Tools**: Ollama (DeepSeek-R1 for reasoning)
**Example**:
```
Input: Concept "quantum neural networks" (prediction error: 0.85)
Output:
  - How do quantum neural networks differ from classical ANNs?
  - What are the current limitations of quantum computing for ML?
  - Which research groups are leading quantum ML?
```

#### 3. **Knowledge Graph Search Agent**
**Responsibility**: Search existing Neo4j graph for related information
**Input**: Research questions
**Output**: Related concepts, documents, relationships
**Tools**: Neo4j Cypher queries, semantic search
**Strategy**: 2-hop graph traversal from concept node

#### 4. **Document Recommendation Agent** (Local LLM)
**Responsibility**: Propose new documents to ingest based on gaps
**Input**: Search results, knowledge gaps
**Output**: Document titles, keywords, search queries
**Tools**: Ollama (Qwen2.5 for structured output)
**Example**:
```
Gap: "No information on quantum entanglement for ML"
Recommendations:
  - Search: "quantum entanglement machine learning 2024"
  - Authors: "Scott Aaronson", "Vedran Dunjko"
  - Conferences: ICML, NeurIPS quantum ML workshops
```

#### 5. **Exploration Task Queue Manager**
**Responsibility**: Manage background agent tasks, prioritize by curiosity strength
**Input**: Curiosity triggers from all processed documents
**Output**: Prioritized queue of exploration tasks
**Tools**: Redis task queue, priority scoring
**Queue Structure**: High priority (>0.8), Medium (0.6-0.8), Low (<0.6)

#### 6. **Background Agent Orchestrator**
**Responsibility**: Spawn and manage background agents asynchronously
**Input**: Exploration tasks from queue
**Output**: Completed explorations (research questions answered, recommendations generated)
**Tools**: Asyncio, LangGraph background workflows
**Constraint**: Never block document upload pipeline

**Best Practices Applied**:
- ✅ **Specialization**: Each agent focuses on one curiosity-related task
- ✅ **Modularity**: Agents can run independently in background
- ✅ **Bidirectional Graph Interaction**: All agents read/write Neo4j
- ✅ **Monitoring**: Track curiosity resolution rate, agent performance
- ✅ **Integration Layer**: Redis queue coordinates agent execution

## Requirements

### Functional Requirements

#### FR1: Curiosity Trigger Detection
**Description**: Detect curiosity signals during document processing
**Acceptance Criteria**:
- [ ] Active inference engine calculates prediction error for each concept
- [ ] Prediction errors >0.7 trigger curiosity events
- [ ] Knowledge gaps (missing relationships) trigger curiosity events
- [ ] Curiosity events stored in Redis queue with priority
- [ ] Example: "Concept 'quantum neural networks' has prediction error 0.85"

#### FR2: Background Agent Spawning
**Description**: Spawn agents in background without blocking document upload
**Acceptance Criteria**:
- [ ] Document processing continues while curiosity agents run
- [ ] Agents run asynchronously using asyncio
- [ ] Max 5 concurrent curiosity agents to prevent resource exhaustion
- [ ] Agents log start/completion times and results

#### FR3: Local LLM Question Generation (ZERO COST)
**Description**: Use Ollama to generate research questions from curiosity triggers
**Acceptance Criteria**:
- [ ] All LLM calls use `http://localhost:11434/api/generate` (Ollama)
- [ ] Default model: DeepSeek-R1 (reasoning) or Qwen2.5:14b (structured output)
- [ ] Generate 3-5 questions per curiosity trigger
- [ ] Questions stored in Neo4j with CURIOSITY_QUESTION relationship
- [ ] Example questions are specific and actionable

#### FR4: Knowledge Graph Exploration
**Description**: Search Neo4j for information related to curiosity questions
**Acceptance Criteria**:
- [ ] 2-hop Cypher queries from concept node
- [ ] Return related concepts, documents, relationships
- [ ] Semantic similarity search (>0.75 threshold)
- [ ] Results logged: "Found 12 related concepts, 3 documents"

#### FR5: Document Recommendation
**Description**: Suggest new documents to ingest based on knowledge gaps
**Acceptance Criteria**:
- [ ] Recommendations include: document title, keywords, authors, conferences
- [ ] Recommendations stored in Neo4j with RECOMMENDED_DOCUMENT node
- [ ] User can review recommendations via interface
- [ ] Example: "Recommended: 'Quantum ML Survey 2024' by Aaronson et al."

#### FR6: Curiosity Resolution Tracking
**Description**: Track which curiosity triggers were resolved
**Acceptance Criteria**:
- [ ] Each curiosity event has status: pending/exploring/resolved/unresolvable
- [ ] Resolution metrics: % of questions answered, documents found
- [ ] Unresolvable gaps flagged for user review
- [ ] Metrics logged: "85% of curiosity triggers resolved within 10 minutes"

### Non-Functional Requirements

#### NFR1: Zero Cost Constraint
- **ALL** LLM inference uses local Ollama (no OpenAI, Anthropic, etc.)
- Redis queue management (free/self-hosted)
- Neo4j Community Edition (free)
- Total cost: $0 for curiosity exploration

#### NFR2: Performance
- Curiosity detection: <50ms per concept
- Question generation (local LLM): <5 seconds per trigger
- Background agents: Process 100 curiosity triggers in <10 minutes
- Document upload pipeline: Zero blocking (agents run in background)

#### NFR3: Scalability
- Support 1000+ curiosity triggers in Redis queue
- Max 5 concurrent background agents
- Queue processing rate: 10-20 triggers per minute

## Technical Design

### Architecture

```
Curiosity-Driven Background Agent System:

Document Processing (Foreground):
┌─────────────────────────────────────────┐
│  Upload 100 Papers                      │
│  - Extract concepts                     │
│  - Calculate prediction errors          │
│  - Detect knowledge gaps                │
└──────────┬──────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────┐
    │  Active Inference Engine     │
    │  - Beliefs: {neural: 0.8}    │
    │  - Prediction: 0.6           │
    │  - Error: 0.85 ← HIGH!       │
    └──────────┬───────────────────┘
               │
               ▼ CURIOSITY TRIGGER
    ┌──────────────────────────────────────┐
    │  Curiosity Detection Agent           │
    │  - Create curiosity event            │
    │  - Priority: 0.85 (high)             │
    │  - Concept: "quantum neural nets"    │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Redis Curiosity Queue               │
    │  Key: curiosity_queue                │
    │                                      │
    │  [                                   │
    │    {                                 │
    │      "trigger_id": "ct_001",         │
    │      "concept": "quantum neural...", │
    │      "prediction_error": 0.85,       │
    │      "priority": 0.85,               │
    │      "status": "pending",            │
    │      "created_at": "2025-10-01..."   │
    │    },                                │
    │    ...                               │
    │  ]                                   │
    └──────────┬───────────────────────────┘
               │
               ▼ BACKGROUND PROCESSING (No blocking!)
    ┌──────────────────────────────────────┐
    │  Background Agent Orchestrator       │
    │  - Poll queue every 5 seconds        │
    │  - Spawn agents for high priority    │
    │  - Max 5 concurrent agents           │
    └──────────┬───────────────────────────┘
               │
      ┌────────┴────────┬──────────┐
      ▼                 ▼          ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Agent 1     │  │ Agent 2     │  │ Agent 3     │
│ Exploring   │  │ Exploring   │  │ Exploring   │
│ ct_001      │  │ ct_002      │  │ ct_003      │
└─────┬───────┘  └──────┬──────┘  └──────┬──────┘
      │                 │                 │
      ▼                 ▼                 ▼
┌─────────────────────────────────────────────┐
│  Question Generation Agent (LOCAL LLM)      │
│  - Ollama: DeepSeek-R1                      │
│  - Generate 3-5 questions                   │
│  - NO API COST!                             │
└─────────────┬───────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────┐
    │  Questions Generated:        │
    │  1. How do quantum NNs work? │
    │  2. Current limitations?     │
    │  3. Leading researchers?     │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Knowledge Graph Search Agent        │
    │  - Query Neo4j (2-hop search)        │
    │  - Find related concepts/docs        │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Search Results:                     │
    │  - 12 related concepts found         │
    │  - 3 relevant documents              │
    │  - 8 relationships                   │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Document Recommendation Agent       │
    │  (LOCAL LLM)                         │
    │  - Ollama: Qwen2.5:14b               │
    │  - Generate recommendations          │
    │  - NO API COST!                      │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Recommendations:                    │
    │  - "Quantum ML Survey 2024"          │
    │  - Authors: Scott Aaronson           │
    │  - Keywords: quantum entanglement ML │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Neo4j Knowledge Graph               │
    │  - Store curiosity questions         │
    │  - Store recommendations             │
    │  - Update curiosity status: resolved │
    └──────────────────────────────────────┘
```

### Data Model

#### Curiosity Trigger (Redis Queue Item)
```python
@dataclass
class CuriosityTrigger:
    trigger_id: str                         # Unique ID
    concept: str                            # Concept that triggered curiosity
    prediction_error: float                 # 0.0 - 1.0
    knowledge_gap_type: str                 # "no_basin_match" / "contradictory_relationship" / "novel_pattern"
    priority: float                         # Same as prediction_error
    status: str                             # pending/exploring/resolved/unresolvable
    created_at: str                         # Timestamp
    document_source: str                    # Where curiosity originated
    questions_generated: List[str] = []     # Research questions
    related_concepts: List[str] = []        # From graph search
    recommendations: List[Dict] = []        # Document recommendations
    resolution_time: Optional[str] = None   # When resolved
```

#### Neo4j Schema Extensions
```cypher
// Curiosity Question Node
CREATE (q:CuriosityQuestion {
  question_id: "q_12345",
  question_text: "How do quantum neural networks differ from classical ANNs?",
  concept: "quantum neural networks",
  prediction_error: 0.85,
  created_at: "2025-10-01T10:30:00",
  status: "resolved",
  answer_source: "doc_quantum_ml_survey.pdf"
})

// Concept triggered curiosity
CREATE (c:Concept)-[:TRIGGERED_CURIOSITY {
  prediction_error: 0.85,
  gap_type: "no_basin_match",
  timestamp: "2025-10-01T10:30:00"
}]->(q:CuriosityQuestion)

// Recommended Document
CREATE (rec:RecommendedDocument {
  rec_id: "rec_67890",
  title: "Quantum Machine Learning Survey 2024",
  authors: ["Scott Aaronson", "Vedran Dunjko"],
  keywords: ["quantum computing", "machine learning", "entanglement"],
  search_query: "quantum entanglement machine learning 2024",
  source_curiosity: "q_12345",
  created_at: "2025-10-01T10:35:00",
  status: "pending_review"
})

// Question led to recommendation
CREATE (q:CuriosityQuestion)-[:LED_TO_RECOMMENDATION]->(rec:RecommendedDocument)

// Document answered question
CREATE (doc:Document)-[:ANSWERED_QUESTION {
  satisfaction_score: 0.9,
  timestamp: "2025-10-01T12:00:00"
}]->(q:CuriosityQuestion)
```

### Agent Implementation

#### Curiosity Detection Agent
```python
class CuriosityDetectionAgent:
    """Monitor document processing for curiosity triggers"""

    def __init__(self, redis_client, active_inference_engine):
        self.redis = redis_client
        self.active_inference = active_inference_engine

    def detect_curiosity_triggers(self, concepts: List[str], document_id: str) -> List[CuriosityTrigger]:
        """Detect which concepts trigger curiosity"""
        triggers = []

        for concept in concepts:
            # Get active inference state for concept
            belief = self.active_inference.get_belief(concept)
            prediction = self.active_inference.predict(concept)
            prediction_error = abs(belief - prediction)

            # Check if curiosity triggered
            if prediction_error > 0.7:
                trigger = CuriosityTrigger(
                    trigger_id=f"ct_{uuid.uuid4().hex[:12]}",
                    concept=concept,
                    prediction_error=prediction_error,
                    knowledge_gap_type=self._classify_gap(concept, prediction_error),
                    priority=prediction_error,
                    status="pending",
                    created_at=datetime.now().isoformat(),
                    document_source=document_id
                )

                # Add to Redis queue (sorted set by priority)
                self.redis.zadd(
                    "curiosity_queue",
                    {json.dumps(asdict(trigger)): trigger.priority}
                )

                triggers.append(trigger)
                logger.info(f"🔍 CURIOSITY TRIGGERED: '{concept}' "
                           f"(error: {prediction_error:.3f})")

        return triggers

    def _classify_gap(self, concept: str, error: float) -> str:
        """Classify type of knowledge gap"""
        # Check if concept has basin
        basin = self.redis.get(f"attractor_basin:{concept}")

        if not basin:
            return "no_basin_match"
        elif error > 0.9:
            return "contradictory_relationship"
        else:
            return "novel_pattern"
```

#### Question Generation Agent (Local LLM - ZERO COST)
```python
class QuestionGenerationAgent:
    """Generate research questions using LOCAL Ollama LLM"""

    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url

    def generate_questions(self, trigger: CuriosityTrigger) -> List[str]:
        """Generate 3-5 research questions for curiosity trigger"""

        prompt = f"""
You are a research scientist investigating a knowledge gap.

Concept: {trigger.concept}
Knowledge Gap Type: {trigger.knowledge_gap_type}
Prediction Error: {trigger.prediction_error:.2f} (high surprise)

Generate 3-5 specific, actionable research questions to investigate this concept.
Questions should be:
- Specific and well-defined
- Answerable through literature review or knowledge graph search
- Focused on understanding the concept deeply

Return ONLY a JSON array of questions, no other text:
["Question 1", "Question 2", ...]
"""

        # Call LOCAL Ollama (ZERO COST)
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "deepseek-r1",  # FREE local model
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "format": "json"
            },
            timeout=30
        )

        result = response.json()
        questions = json.loads(result.get("response", "[]"))

        logger.info(f"🔬 Generated {len(questions)} questions for '{trigger.concept}' "
                   f"(LOCAL LLM - $0 cost)")

        return questions
```

#### Background Agent Orchestrator
```python
class BackgroundAgentOrchestrator:
    """Spawn and manage background curiosity agents"""

    def __init__(self, redis_client, neo4j_schema):
        self.redis = redis_client
        self.neo4j = neo4j_schema
        self.active_agents = []
        self.max_concurrent_agents = 5

    async def run_continuous_exploration(self):
        """Continuously process curiosity queue in background"""
        logger.info("🌌 Background agent orchestrator started")

        while True:
            try:
                # Check if can spawn new agent
                if len(self.active_agents) < self.max_concurrent_agents:
                    # Get highest priority trigger from queue
                    trigger_data = self.redis.zpopmax("curiosity_queue", 1)

                    if trigger_data:
                        trigger_json, priority = trigger_data[0]
                        trigger = CuriosityTrigger(**json.loads(trigger_json))

                        # Spawn background agent
                        agent_task = asyncio.create_task(
                            self._explore_curiosity_trigger(trigger)
                        )
                        self.active_agents.append(agent_task)

                        logger.info(f"🚀 Spawned agent for '{trigger.concept}' "
                                   f"(priority: {priority:.3f})")

                # Clean up completed agents
                self.active_agents = [
                    agent for agent in self.active_agents
                    if not agent.done()
                ]

                # Wait before next check
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Orchestrator error: {e}")
                await asyncio.sleep(10)

    async def _explore_curiosity_trigger(self, trigger: CuriosityTrigger):
        """Explore single curiosity trigger (background task)"""
        try:
            # 1. Generate questions (LOCAL LLM)
            question_agent = QuestionGenerationAgent()
            questions = question_agent.generate_questions(trigger)
            trigger.questions_generated = questions

            # 2. Search knowledge graph
            search_agent = KnowledgeGraphSearchAgent(self.neo4j)
            related = search_agent.search_related_concepts(trigger.concept)
            trigger.related_concepts = related

            # 3. Generate recommendations (LOCAL LLM)
            rec_agent = DocumentRecommendationAgent()
            recommendations = rec_agent.generate_recommendations(trigger, related)
            trigger.recommendations = recommendations

            # 4. Store results in Neo4j
            self._store_exploration_results(trigger)

            # 5. Update status
            trigger.status = "resolved" if related else "unresolvable"
            trigger.resolution_time = datetime.now().isoformat()

            logger.info(f"✅ Curiosity exploration complete: '{trigger.concept}' "
                       f"({len(questions)} questions, {len(related)} concepts, "
                       f"{len(recommendations)} recommendations)")

        except Exception as e:
            logger.error(f"Exploration failed for '{trigger.concept}': {e}")
            trigger.status = "unresolvable"

    def _store_exploration_results(self, trigger: CuriosityTrigger):
        """Store curiosity exploration results in Neo4j"""
        # Store questions
        for question in trigger.questions_generated:
            self.neo4j.create_curiosity_question_node(question, trigger)

        # Store recommendations
        for rec in trigger.recommendations:
            self.neo4j.create_recommended_document_node(rec, trigger)
```

### Test Strategy

```python
def test_curiosity_detection():
    """Test curiosity triggers are detected"""
    agent = CuriosityDetectionAgent(redis_client, active_inference_engine)

    # High prediction error concept
    concepts = ["quantum neural networks"]
    triggers = agent.detect_curiosity_triggers(concepts, "doc_123")

    assert len(triggers) > 0
    assert triggers[0].prediction_error > 0.7
    assert triggers[0].status == "pending"

def test_local_llm_question_generation():
    """Test LOCAL LLM generates questions (ZERO COST)"""
    agent = QuestionGenerationAgent()

    trigger = CuriosityTrigger(
        trigger_id="ct_001",
        concept="quantum neural networks",
        prediction_error=0.85,
        knowledge_gap_type="no_basin_match",
        priority=0.85,
        status="pending",
        created_at=datetime.now().isoformat(),
        document_source="doc_123"
    )

    questions = agent.generate_questions(trigger)

    assert len(questions) >= 3
    assert all(isinstance(q, str) for q in questions)
    assert any("quantum" in q.lower() for q in questions)
    # Verify NO API calls to OpenAI/Anthropic (check logs)

def test_background_agent_orchestration():
    """Test agents run in background without blocking"""
    orchestrator = BackgroundAgentOrchestrator(redis_client, neo4j_schema)

    # Add triggers to queue
    for i in range(10):
        trigger = create_curiosity_trigger(f"concept_{i}")
        redis_client.zadd("curiosity_queue", {json.dumps(asdict(trigger)): trigger.priority})

    # Start orchestrator
    asyncio.run(orchestrator.run_continuous_exploration())

    # Verify agents spawned
    assert len(orchestrator.active_agents) <= 5  # Max concurrent
```

## Implementation Plan

### Phase 1: Curiosity Detection (2-3 hours)
1. Implement `CuriosityDetectionAgent`
2. Integrate with active inference engine
3. Redis queue with priority
4. Test trigger detection

### Phase 2: Local LLM Integration (3-4 hours)
1. Implement `QuestionGenerationAgent` with Ollama
2. Implement `DocumentRecommendationAgent` with Ollama
3. Verify ZERO API cost
4. Test question quality

### Phase 3: Background Orchestration (3-4 hours)
1. Implement `BackgroundAgentOrchestrator`
2. Async agent spawning
3. Max concurrent agent limit
4. Queue processing loop

### Phase 4: Knowledge Graph Integration (2-3 hours)
1. Implement `KnowledgeGraphSearchAgent`
2. Store curiosity questions in Neo4j
3. Store recommendations in Neo4j
4. Track resolution status

### Phase 5: Testing & Documentation (2-3 hours)
1. Unit tests for all agents
2. Integration tests
3. Performance validation
4. Documentation

**Total Estimated Time**: 12-17 hours

## Success Criteria

- [ ] Curiosity triggers detected (prediction error >0.7)
- [ ] Background agents spawn asynchronously (max 5 concurrent)
- [ ] **ZERO COST**: All LLM calls use local Ollama
- [ ] Questions generated for each trigger (3-5 questions)
- [ ] Knowledge graph searched for related concepts
- [ ] Document recommendations generated
- [ ] Results stored in Neo4j with relationships
- [ ] All tests passing

## References

- Spec 027: Basin Frequency Strengthening
- Spec 028: ThoughtSeed Generation
- Active Inference System documentation
- Ollama API: http://localhost:11434/api/generate
