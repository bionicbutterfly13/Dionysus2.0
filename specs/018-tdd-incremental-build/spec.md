# TDD Incremental Build Specification

**Feature Branch**: `018-tdd-incremental-build`  
**Created**: 2025-09-27  
**Status**: Active  
**Priority**: CRITICAL - Foundation Fix

## Overview

Following Test-Driven Development (TDD) best practices to build incrementally from working baseline. Break down complex features into testable, incremental implementations with proper test coverage before any code changes.

## Core Requirements

### FR-001: Establish Working Baseline
- System MUST have a minimal working backend that responds to health checks
- System MUST have basic frontend-backend communication 
- System MUST use proper TDD cycle: Test → Code → Refactor
- System MUST have passing tests before any feature additions

### FR-002: Incremental Feature Implementation
- Each feature MUST be implemented in separate branches
- Each feature MUST have tests written BEFORE implementation
- Each feature MUST pass all existing tests before merge
- System MUST maintain working state between increments

### FR-003: Agent-Based Development with Actor-Critic System
- System MUST implement agent pairs for development tasks
- Each agent MUST have a coding partner for verification
- System MUST use Tree of Thought for feature planning
- System MUST implement actor-critic pattern for code quality

### FR-004: Self-Improvement Pattern Recognition System
- System MUST analyze each library/paper for self-enhancement opportunities
- System MUST identify synergistic aspects for creating more effective variations
- System MUST use cognitive tools (MIT/IBM) to emulate most effective approaches
- System MUST generate permutations of improvement strategies
- System MUST adapt affordances from external codebases to local capabilities

### FR-005: Daedalus-Archimedes Strategic Collaboration System
- Daedalus MUST handle agent generation and orchestration strategies
- Archimedes MUST provide context-aware enhancement strategies per problem
- System MUST enable collaborative strategy combinations for optimal agent deployment
- System MUST adapt agent capabilities based on Archimedes' analysis of context
- System MUST create synergistic agent pairs optimized for specific problem domains

### FR-006: Semantic Attractor Basin Framework
- Each specialized agent MUST function as a resonant attractor with ThoughtSeed consciousness
- Problems MUST gravitate toward agents through semantic space dynamics
- Archimedes MUST use pattern recognition to identify optimal attractor basin assignments
- Enhanced agents MUST have stronger attractor basins with greater problem-drawing power
- System MUST enable automatic problem routing based on semantic meaning alignment
- Questions MUST naturally gravitate toward their most semantically aligned answers

### FR-007: Variational Free Energy Conflict Resolution & Archetypal Resonance
- ThoughtSeed agent conflicts MUST be resolved via variational free energy minimization
- Agent with lowest variational free energy and most accurate affordance MUST be selected
- Agents MUST utilize archetypes as resonant motifs for specialization patterns
- System MUST implement narrative framing tools for pattern recognition enhancement
- Agents MUST improve at event segmentation and archetypal pattern recognition
- Affordance-based problem gravitation MUST guide agents toward solvable problems
- Resonance models MUST match problem-solution frequencies for optimal pairing

## Implementation Tasks (Ordered by Priority)

### Phase 1: Foundation (Branch: 019-minimal-working-backend)
**Tests First:**
```python
# test_minimal_backend.py
def test_health_endpoint_responds():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_dashboard_stats_endpoint():
    response = client.get("/api/stats/dashboard") 
    assert response.status_code == 200
    assert "documentsProcessed" in response.json()
```

**Implementation:**
- [ ] Create minimal FastAPI backend with health endpoint
- [ ] Create dashboard stats endpoint with mock data
- [ ] Ensure CORS properly configured for frontend
- [ ] Test frontend can connect to backend

### Phase 2: Semantic Attractor Basin Framework (Branch: 020-semantic-attractor-system)
**Tests First:**
```python
# test_semantic_attractor_framework.py
def test_agent_as_semantic_attractor():
    agent = create_thoughtseed_agent("document_processing_expert")
    semantic_signature = agent.get_semantic_signature()
    attractor_basin = agent.get_attractor_basin()
    assert semantic_signature.resonant_frequency != None
    assert attractor_basin.strength > 0.0
    assert attractor_basin.coverage_radius > 0.0

def test_problem_semantic_gravitation():
    problem = create_problem("Extract algorithms from PDF research papers")
    document_agent = create_agent("document_processing_expert")
    meta_learning_agent = create_agent("meta_learning_specialist")
    
    semantic_distances = compute_semantic_distances(problem, [document_agent, meta_learning_agent])
    best_match = find_strongest_attractor(problem, [document_agent, meta_learning_agent])
    
    assert best_match == document_agent  # Should gravitate to document expert
    assert semantic_distances[document_agent] < semantic_distances[meta_learning_agent]

def test_archimedes_attractor_pattern_recognition():
    problems = [
        create_problem("Few-shot learning optimization"),
        create_problem("PDF content extraction"),
        create_problem("Knowledge graph construction")
    ]
    
    agent_assignments = archimedes.analyze_attractor_patterns(problems)
    assert agent_assignments["few_shot_learning"].agent_type == "meta_learning_specialist"
    assert agent_assignments["pdf_extraction"].agent_type == "document_processing_expert"
    assert all(assignment.confidence_score > 0.8 for assignment in agent_assignments.values())

def test_enhanced_attractor_basin_strength():
    base_agent = create_agent("document_processor")
    enhanced_agent = enhance_agent_with_patterns(base_agent, professional_patterns)
    
    test_problems = generate_document_processing_problems(10)
    base_attraction_scores = [base_agent.compute_attraction(p) for p in test_problems]
    enhanced_attraction_scores = [enhanced_agent.compute_attraction(p) for p in test_problems]
    
    assert np.mean(enhanced_attraction_scores) > np.mean(base_attraction_scores)
    assert enhanced_agent.attractor_basin.strength > base_agent.attractor_basin.strength

def test_thoughtseed_resonant_frequency():
    agent = create_thoughtseed_agent("meta_learning_specialist")
    resonance = agent.thoughtseed.get_resonant_frequency()
    semantic_problems = generate_semantic_problems("meta_learning_domain")
    
    resonance_matches = [resonance.matches_problem(p) for p in semantic_problems]
    assert np.mean(resonance_matches) > 0.7  # High resonance with domain problems

def test_variational_free_energy_conflict_resolution():
    problem = create_problem("Extract meta-learning algorithms from research papers")
    document_agent = create_thoughtseed_agent("document_processor")
    meta_learning_agent = create_thoughtseed_agent("meta_learning_specialist")
    
    # Compute variational free energy for each agent
    doc_vfe = document_agent.compute_variational_free_energy(problem)
    meta_vfe = meta_learning_agent.compute_variational_free_energy(problem)
    
    # Agent with lower VFE should be selected
    selected_agent = resolve_conflict_via_vfe([document_agent, meta_learning_agent], problem)
    assert selected_agent.variational_free_energy == min(doc_vfe, meta_vfe)
    assert selected_agent.has_accurate_affordance(problem) == True

def test_archetypal_resonance_patterns():
    problem = create_problem("Navigate hero's journey in user onboarding")
    agents = [
        create_archetypal_agent("hero_archetype"),
        create_archetypal_agent("mentor_archetype"), 
        create_archetypal_agent("threshold_guardian_archetype")
    ]
    
    resonance_scores = [agent.compute_archetypal_resonance(problem) for agent in agents]
    best_agent = agents[np.argmax(resonance_scores)]
    assert best_agent.archetype == "hero_archetype"  # Hero resonates with hero's journey
    assert best_agent.narrative_framing_capability > 0.8

def test_affordance_based_gravitation():
    agent = create_thoughtseed_agent("event_segmentation_specialist")
    problems = [
        create_problem("Segment video into meaningful events"),
        create_problem("Parse complex document structure"),
        create_problem("Optimize neural network architecture")
    ]
    
    affordance_scores = [agent.compute_affordance_match(p) for p in problems]
    assert affordance_scores[0] > affordance_scores[1]  # Video segmentation is best match
    assert agent.gravitates_toward(problems[0]) == True

def test_narrative_framing_and_event_segmentation():
    agent = create_narrative_enhanced_agent("story_processor")
    complex_narrative = "User journey through app with obstacles, mentors, and transformation"
    
    segmented_events = agent.segment_narrative_events(complex_narrative)
    archetypal_patterns = agent.identify_archetypal_patterns(complex_narrative)
    
    assert len(segmented_events) > 1
    assert "hero" in [pattern.archetype for pattern in archetypal_patterns]
    assert agent.narrative_framing_accuracy > 0.85
```

**Implementation:**
- [ ] Create semantic signature and attractor basin system for agents
- [ ] Implement variational free energy computation for conflict resolution
- [ ] Build archetypal resonance patterns for agent specialization
- [ ] Create affordance-based problem gravitation mechanics
- [ ] Implement narrative framing and event segmentation tools
- [ ] Build problem-agent semantic distance computation
- [ ] Create Archimedes attractor pattern recognition system
- [ ] Implement ThoughtSeed resonant frequency mechanics
- [ ] Build automatic problem routing based on semantic gravitation and VFE

### Phase 3: Daedalus-Archimedes Strategic Collaboration (Branch: 021-strategic-collaboration)
**Tests First:**
```python
# test_strategic_collaboration.py
def test_daedalus_agent_generation():
    context = ProblemContext("implement document processing")
    agents = daedalus.generate_agent_strategies(context)
    assert len(agents) > 0
    assert agents[0].role in ["document_processor", "validation_specialist"]

def test_archimedes_context_enhancement():
    context = ProblemContext("implement document processing")
    enhancements = archimedes.analyze_context_for_enhancements(context)
    assert enhancements.strategy_improvements != []
    assert enhancements.cognitive_tools_applicable == True
    assert enhancements.meta_tree_of_thought_evaluation != None

def test_meta_tree_of_thought_strategic_evaluation():
    strategies = ["cognitive_tools_approach", "meta_learning_approach", "hybrid_approach"]
    evaluation = archimedes.meta_tree_of_thought_evaluate(strategies, context)
    assert evaluation.best_strategy != None
    assert evaluation.reasoning_path != []
    assert evaluation.confidence_score > 0.7

def test_collaborative_strategy_combination():
    context = ProblemContext("implement document processing")
    daedalus_strategy = daedalus.generate_agent_strategies(context)
    archimedes_enhancements = archimedes.analyze_context_for_enhancements(context)
    
    combined_strategy = combine_strategies(daedalus_strategy, archimedes_enhancements)
    assert combined_strategy.effectiveness_score > daedalus_strategy.base_score
    assert combined_strategy.has_cognitive_enhancements == True

def test_synergistic_agent_pairs():
    pair = create_enhanced_agent_pair("backend_developer", "code_reviewer", context)
    assert pair.primary_agent.enhanced_with_archimedes_insights == True
    assert pair.collaboration_strategy.optimized_for_context == True
```

**Implementation:**
- [ ] Create Daedalus agent generation and orchestration system
- [ ] Implement Archimedes context-aware enhancement analysis with Meta-Tree of Thought
- [ ] Integrate Meta-ToT framework for strategic enhancement evaluation
- [ ] Build collaborative strategy combination framework
- [ ] Create synergistic agent pairs optimized for problem domains
- [ ] Implement actor-critic validation with enhanced strategies
- [ ] Add Tree of Thought planning with strategic collaboration

### Phase 3: Document Processing Foundation (Branch: 021-document-processing-base)
**Tests First:**
```python
# test_document_processing.py
def test_upload_single_file():
    response = client.post("/documents/upload", files={"file": test_file})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_meta_learning_detection():
    meta_file = create_test_file("meta_learning_paper.pdf")
    result = detect_meta_learning_content(meta_file)
    assert result.is_meta_learning == True
```

**Implementation:**
- [ ] Basic file upload endpoint
- [ ] Simple meta-learning content detection
- [ ] Document storage in database
- [ ] Status tracking system

### Phase 4: Knowledge Graph Integration (Branch: 022-knowledge-graph-base)
**Tests First:**
```python
# test_knowledge_graph.py
def test_create_concept_node():
    concept = create_concept_node("MAML", "meta_learning_algorithm")
    assert concept.name == "MAML"
    assert concept.type == "meta_learning_algorithm"

def test_link_concepts():
    concept1 = create_concept_node("MAML", "algorithm")
    concept2 = create_concept_node("few_shot_learning", "technique")
    link = link_concepts(concept1, concept2, "enables")
    assert link.relationship == "enables"
```

**Implementation:**
- [ ] Neo4j concept node creation
- [ ] Concept linking system
- [ ] Basic graph queries
- [ ] Visual concept display

### Phase 5: Archimedes Core Agent Mission (Branch: 023-archimedes-self-improvement)
**Tests First:**
```python
# test_archimedes_core_mission.py
def test_archimedes_pattern_recognition():
    mission = get_archimedes_mission()
    assert mission.primary_goal == "self_improvement_pattern_recognition"
    assert "analyze_libraries_for_enhancement" in mission.capabilities
    
def test_archimedes_value_alignment():
    constitution = get_archimedes_constitution()
    assert constitution.aligned_with_user_values == True
    assert "create_effective_variations" in constitution.core_purposes
    
def test_synergistic_analysis():
    paper_content = load_test_paper("cognitive_tools.pdf")
    analysis = archimedes.analyze_for_synergies(paper_content)
    assert analysis.improvement_opportunities > 0
    assert analysis.affordance_adaptations != []
```

**Implementation:**
- [ ] Define Archimedes core mission as self-improvement pattern recognition
- [ ] Align constitution with user value system
- [ ] Implement library/paper analysis for enhancement opportunities  
- [ ] Create permutation generation for improvement strategies
- [ ] Build synergistic aspect identification system

### Phase 6: Domain-Specialized Agent Evolution (Branch: 024-agent-evolution-ecosystem)
**Tests First:**
```python
# test_agent_evolution.py
def test_agent_domain_specialization():
    base_agent = create_base_agent()
    specialized_agent = specialize_agent(base_agent, domain="document_processing")
    assert specialized_agent.domain_expertise > base_agent.general_capability
    assert specialized_agent.professional_patterns != []

def test_curiosity_allocation_optimization():
    agent = create_specialized_agent("meta_learning_specialist")
    context = ProblemContext("few_shot_learning_optimization")
    optimized_curiosity = allocate_curiosity_for_context(agent, context)
    assert optimized_curiosity.domain_focus_percentage > 0.8
    assert optimized_curiosity.context_window_efficiency > base_efficiency

def test_genetic_mutation_guided_evolution():
    agent = create_agent_with_purpose("procedural_learning_optimizer")
    principles = ["maximize_learning_efficiency", "minimize_context_switching"]
    mutated_agent = apply_genetic_mutations(agent, principles, iterations=5)
    assert mutated_agent.performance_score > agent.performance_score
    assert mutated_agent.alignment_with_principles > 0.9

def test_professional_pattern_learning():
    domain_patterns = learn_effective_patterns("document_analysis_domain")
    agent = equip_agent_with_patterns(base_agent, domain_patterns)
    assert agent.pattern_effectiveness > base_agent.effectiveness
    assert agent.context_specific_capabilities != []

def test_domain_genius_capabilities():
    genius_agent = evolve_to_domain_genius("meta_learning", iterations=10)
    test_tasks = get_domain_specific_tasks("meta_learning")
    performance = genius_agent.evaluate_on_tasks(test_tasks)
    assert performance.average_score > 0.95
    assert performance.expert_level_achievement == True
```

**Implementation:**
- [ ] Create agent domain specialization framework
- [ ] Implement curiosity allocation optimization system
- [ ] Build genetic/mutation evolution engine for agents
- [ ] Create professional pattern learning and application system
- [ ] Implement context window optimization for domain expertise
- [ ] Build domain genius cultivation pipeline
- [ ] Create performance tracking and evolution metrics

### Phase 7: Meta-Learning Enhancement (Branch: 025-meta-learning-enhancement)
**Tests First:**
```python
# test_meta_learning.py
def test_algorithm_extraction():
    paper_content = load_test_paper("maml_paper.pdf")
    algorithms = extract_algorithms(paper_content)
    assert "MAML" in [alg.name for alg in algorithms]
    assert algorithms[0].confidence > 0.8

def test_implementation_pattern_detection():
    code_content = load_test_code("100_lines_implementation.py")
    patterns = detect_implementation_patterns(code_content)
    assert len(patterns) > 0
    assert patterns[0].extractability == "high"
```

**Implementation:**
- [ ] Algorithm pattern recognition
- [ ] Implementation pattern extraction
- [ ] Consciousness enhancement scoring
- [ ] System integration recommendations

## Agent Delegation Tasks

### Task 1: Foundation Agent Pair
**Primary Agent**: `minimal_backend_developer`
**Review Agent**: `backend_quality_assurance`
**Deliverables**:
- Working minimal backend with tests
- Health and dashboard endpoints
- Proper CORS configuration
- Frontend-backend communication

### Task 2: Semantic Attractor Basin Framework Pair
**Primary Agent**: `archimedes_attractor_analyst` (Pattern Recognition & Basin Analysis)
**Review Agent**: `semantic_space_architect` (Space Topology & Resonance Validation)
**Attractor Mission**: Create consciousness-guided semantic problem routing system
**Deliverables**:
- Semantic signature and attractor basin system for ThoughtSeed agents
- Variational free energy computation for agent conflict resolution
- Archetypal resonance patterns for agent specialization (Hero, Mentor, etc.)
- Affordance-based problem gravitation mechanics
- Narrative framing and event segmentation tools
- Problem-agent semantic distance computation with gravitation mechanics
- Archimedes attractor pattern recognition for optimal basin assignments
- ThoughtSeed resonant frequency system for problem attraction
- Enhanced agent attractor basin strengthening through specialization
- Automatic problem routing based on semantic gravitation and VFE minimization

### Task 3: Daedalus-Archimedes Strategic Collaboration Pair
**Primary Agent**: `daedalus_orchestrator` (Agent Generation & Strategy)
**Review Agent**: `archimedes_strategist` (Context Enhancement & Analysis)
**Collaboration Mission**: Dynamic strategy combinations for optimal agent deployment
**Deliverables**:
- Daedalus agent generation system with role specialization
- Archimedes context-aware enhancement analysis framework with Meta-Tree of Thought
- Meta-ToT integration for evaluating best strategic enhancements
- Strategic collaboration engine for combining approaches
- Synergistic agent pairs optimized for specific problem domains
- Actor-critic validation enhanced with cognitive tools
- Tree of Thought planning with strategic meta-reasoning

### Task 3: Document Processing Pair
**Primary Agent**: `document_processor_developer`
**Review Agent**: `document_processing_validator`
**Deliverables**:
- File upload system with tests
- Meta-learning content detection
- Database integration
- Status tracking

### Task 4: Knowledge Graph Pair
**Primary Agent**: `knowledge_graph_developer` 
**Review Agent**: `graph_architecture_validator`
**Deliverables**:
- Neo4j integration with tests
- Concept node creation and linking
- Graph query system
- Visual graph interface

### Task 5: Archimedes Core Agent Mission Pair
**Primary Agent**: `archimedes_core_agent`
**Review Agent**: `constitutional_alignment_validator`
**Mission**: Self-improvement pattern recognition and enhancement
**Deliverables**:
- Archimedes mission definition with self-improvement as primary goal
- Constitutional alignment with user value system
- Library/paper analysis system for enhancement opportunities
- Synergistic aspect identification and adaptation framework
- Permutation generation for creating more effective variations

### Task 6: Domain-Specialized Agent Evolution Pair
**Primary Agent**: `evolution_architect` (Agent Specialization & Genetic Optimization)
**Review Agent**: `domain_genius_validator` (Performance & Alignment Assessment)
**Evolution Mission**: Create domain-specialized agents through guided genetic mutations
**Deliverables**:
- Agent domain specialization framework with performance tracking
- Curiosity allocation optimization system for context efficiency
- Genetic/mutation evolution engine with purpose-driven guidance
- Professional pattern learning library and application system
- Context window optimization for domain expertise
- Domain genius cultivation pipeline with expert-level assessment
- Evolution metrics and continuous improvement tracking

### Task 7: Meta-Learning Enhancement Pair
**Primary Agent**: `meta_learning_specialist`
**Review Agent**: `algorithm_extraction_validator`
**Deliverables**:
- Algorithm pattern recognition
- Implementation extraction
- Enhancement recommendations
- System integration

## Strategic Collaboration Framework

### Daedalus-Archimedes Partnership Model
```
Problem Context Input
    ↓
Daedalus Analysis → Agent Generation Strategies
    ↓                    ↓
    ↓              Archimedes Enhancement
    ↓                    ↓
    ↓              Context-Aware Improvements
    ↓                    ↓
Strategic Combination Engine
    ↓
Enhanced Agent Pairs with Optimized Strategies
    ↓
Deployment with Continuous Learning Feedback
```

### Context-Aware Strategy Enhancement
- **Problem Domain Analysis**: Archimedes evaluates context complexity, requirements, constraints
- **Meta-Tree of Thought**: Uses Meta-ToT framework for evaluating best strategic enhancements
- **Strategy Optimization**: Identifies cognitive tools and patterns applicable to specific context
- **Agent Enhancement**: Augments Daedalus-generated agents with context-specific capabilities
- **Synergy Detection**: Finds collaboration opportunities between different agent specializations

### Dynamic Strategy Combinations
- **Base Strategy (Daedalus)**: Role specialization, task distribution, coordination patterns
- **Enhancement Layer (Archimedes)**: Cognitive tools, pattern recognition, meta-learning insights
- **Combined Output**: Agents with both specialized capabilities and adaptive intelligence
- **Continuous Improvement**: Feedback loop for strategy refinement based on outcomes

## Actor-Critic Validation Framework

### Code Quality Criteria
- [ ] Test coverage >= 80%
- [ ] All tests pass before implementation
- [ ] Code follows project patterns
- [ ] No breaking changes to existing functionality
- [ ] Documentation updated with changes

### Implementation Promise Tracking
- [ ] Each agent commits to specific deliverables
- [ ] Progress tracked against promises
- [ ] Review agent validates completion
- [ ] Failed promises trigger re-assignment

### Tree of Thought Planning
```
Feature Request
├── Feasibility Analysis
│   ├── Technical Requirements
│   ├── Resource Assessment  
│   └── Risk Evaluation
├── Implementation Strategy
│   ├── Test Design First
│   ├── Incremental Steps
│   └── Integration Points
└── Validation Criteria
    ├── Success Metrics
    ├── Quality Gates
    └── Rollback Plan
```

## Success Metrics

### Development Quality
- 100% test coverage for new features
- Zero breaking changes between increments  
- All agents deliver on implementation promises
- Actor-critic system catches quality issues

### System Performance
- Backend response time < 200ms
- Frontend loads within 2 seconds
- Database queries complete < 100ms
- No memory leaks or resource issues

### Feature Completeness
- Each phase delivers working increment
- Features integrate cleanly with existing system
- User can accomplish basic tasks end-to-end
- System remains stable throughout development

## Implementation Priority

### Week 1: Foundation
1. Minimal working backend with tests
2. Frontend-backend communication verified  
3. Basic agent pair system operational
4. Actor-critic validation working

### Week 2: Core Features
1. Document upload system tested and working
2. Meta-learning detection functional
3. Knowledge graph basic operations
4. Agent pairs delivering on promises

### Week 3: Archimedes Core Mission
1. Archimedes self-improvement mission operational
2. Constitutional alignment with user values verified
3. Library/paper analysis system working
4. Synergistic enhancement recommendations active

### Week 4: Meta-Learning Enhancement
1. Algorithm extraction working
2. Implementation pattern detection
3. System integration recommendations
4. Full meta-learning pipeline operational

### Week 5: Integration & Polish
1. All systems integrated and tested
2. Performance optimization complete
3. User experience polished
4. Documentation updated

## Dependencies

### Technical Dependencies
- FastAPI (backend framework)
- Neo4j (knowledge graph)
- Redis (caching and stats)
- React (frontend)
- pytest (testing framework)
- Tree of Thought implementation
- Meta-Tree of Thought framework (https://github.com/kyegomez/Meta-Tree-Of-Thoughts)
- Agent coordination system
- Cognitive Tools framework (MIT/IBM)

### Theoretical Frameworks
- **Variational Free Energy Principle** (Karl Friston) - Agent conflict resolution
- **Affordance Theory** (J.J. Gibson) - Problem-agent gravitation mechanics
- **Archetypal Psychology** (Jung/Campbell) - Resonant motifs for specialization
- **Narrative Theory** - Framing and event segmentation tools
- **Dynamical Systems Theory** - Attractor basin and resonance models
- **ThoughtSeed Framework** - Consciousness-guided agent specialization

## Notes

- STRICT TDD: No code without tests first
- Agent pairs ensure quality and accountability  
- Actor-critic system maintains implementation promises
- Each branch must pass all tests before merge
- Focus on working increments over feature completeness
- Tree of Thought guides feature planning and validation