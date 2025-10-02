# Feature Specification: CLAUSE Phase 1 - Agentic Subgraph Architect with Basin Strengthening

**Feature Branch**: `034-clause-phase1-foundation`
**Created**: 2025-10-01
**Status**: Draft
**Input**: User description: "implement Phase 1" - CLAUSE Foundation with Basin Frequency Strengthening from comprehensive roadmap analysis

## Clarifications

### Session 2025-10-01
- Q: Should Phase 1 include Path Navigator and Context Curator agents? → A: No - Focus only on Subgraph Architect in Phase 1 for solid foundation
- Q: What edge budget should be used for production deployment? → A: Start with β_edge=50 (configurable), proven from CLAUSE paper experiments
- Q: Should basin strengthening use +0.2 increment or different value? → A: Use +0.2 as specified in Spec 027, validated approach
- Q: Integration with existing Neo4j schema or create new? → A: Extend existing AttractorBasin model, preserve backward compatibility

### Session 2025-10-02
- Q: What should the API do when Neo4j connection fails during subgraph construction? → A: Retry 3x with exponential backoff, then 503

## User Scenarios & Testing *(mandatory)*

### Primary User Story
When a document is uploaded and processed through the knowledge graph system, the CLAUSE Subgraph Architect should construct a compact, query-relevant subgraph using budget-aware edge selection, while simultaneously strengthening attractor basins for frequently appearing concepts, creating a self-improving system that gets smarter with each document.

### Acceptance Scenarios
1. **Given** a research paper with 50 unique concepts is uploaded, **When** the Subgraph Architect processes it with edge_budget=50, **Then** it selects the top 50 most relevant edges based on 5-signal scoring (entity match, relation match, neighborhood, degree, basin strength)

2. **Given** a concept "neural architecture search" appears in 5 documents, **When** basin strengthening activates, **Then** its basin strength increases to 2.0 (1.0 base + 5×0.2), making it prioritized in future edge selection

3. **Given** two concepts co-occur in 3 documents, **When** basin tracking runs, **Then** their co-occurrence count reaches 3, influencing future neighborhood scoring

4. **Given** a query requires subgraph construction, **When** budget is exhausted (shaped_gain ≤ 0), **Then** Architect stops edge addition and returns compact subgraph

5. **Given** Architect scores an edge with basin_strength=1.8, **When** combined with other signals, **Then** edge score = 0.25×φ_ent + 0.25×φ_rel + 0.20×φ_nbr + 0.15×φ_deg + 0.15×(1.8-1.0) = weighted_sum + 0.12

### Edge Cases
1. **Budget constraint violation**:
   - **Given** edge_budget=10 and 15 high-value edges exist, **When** Architect processes, **Then** exactly 10 edges are selected via shaped gain optimization
   - Covered by: Test suite for budget enforcement

2. **Basin overflow (strength > 2.0)**:
   - **Given** a concept appears in 10+ documents, **When** basin strength exceeds 2.0, **Then** it caps at 2.0 to prevent unbounded growth
   - Covered by: Basin strength normalization tests

3. **Zero basin strength for new concepts**:
   - **Given** a completely new concept with no prior basin, **When** edge scoring occurs, **Then** basin_strength=0.0 and edge is scored on remaining 4 signals only
   - Covered by: New concept handling tests

4. **Co-occurrence tracking memory growth**:
   - **Given** 1000+ concepts with co-occurrences, **When** memory usage is measured, **Then** it remains under 100MB via sparse storage
   - Covered by: Memory profiling tests

5. **Neo4j connection failure during subgraph construction**:
   - **Given** Neo4j becomes unavailable during a subgraph request, **When** connection fails, **Then** system retries 3x with exponential backoff (100ms, 200ms, 400ms), then returns 503 Service Unavailable with error details
   - Covered by: Neo4j integration resilience tests

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST implement CLAUSE Subgraph Architect with budget-aware edge selection using shaped gain rule: accept edge if s(e|q,G) - λ_edge × c_edge > 0
- **FR-002**: Edge scoring MUST use 5 signals with weights: φ_ent (0.25), φ_rel (0.25), φ_nbr (0.20), φ_deg (0.15), basin_strength (0.15)
- **FR-003**: Basin strength MUST increase by +0.2 for each concept reappearance, capped at 2.0 maximum
- **FR-004**: System MUST track co-occurring concepts with counts in basin.co_occurring_concepts dictionary
- **FR-005**: Architect MUST stop edge addition when edge_budget exhausted or shaped_gain ≤ 0
- **FR-006**: Basin strength MUST be normalized to 0.0-1.0 range for edge scoring: (strength - 1.0) / 1.0
- **FR-007**: System MUST integrate with existing Neo4j AttractorBasin model without breaking backward compatibility
- **FR-008**: All basin updates MUST be persisted to Neo4j with activation_history timestamps

### Non-Functional Requirements
- **NFR-001**: Edge scoring MUST complete in <10ms per edge for real-time subgraph construction
- **NFR-002**: Basin strength updates MUST be atomic to prevent race conditions during concurrent document processing
- **NFR-003**: System MUST support 100+ concurrent subgraph constructions without performance degradation
- **NFR-004**: Memory usage for basin tracking MUST not exceed 100MB for 10,000 concepts
- **NFR-005**: Neo4j connection failures MUST trigger 3 retry attempts with exponential backoff (100ms, 200ms, 400ms) before returning 503 error

### Key Entities *(include if feature involves data)*
- **CLAUSE Subgraph Architect**: Agent that constructs compact, query-specific subgraphs using budget-aware edge selection with 5-signal scoring
- **Attractor Basin**: Cognitive state representation with frequency strengthening (strength: 1.0-2.0, activation_count, co_occurring_concepts)
- **Edge Budget (β_edge)**: Maximum number of edges allowed in subgraph, enforced via shaped gain rule
- **Shaped Gain**: Edge selection metric = edge_score - λ_edge × edge_cost, where λ_edge is dual variable from LC-MAPPO
- **5-Signal Edge Score**: Weighted combination of entity match, relation match, neighborhood, degree, and basin strength signals
- **Co-occurrence Dictionary**: Tracks which concepts appear together: {concept_id: count} for basin neighborhood analysis

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain (resolved via clarification session 2025-10-01)
- [x] Requirements are testable and unambiguous (all 8 FRs + 4 NFRs have clear acceptance criteria)
- [x] Success criteria are measurable (edge budget compliance, basin strength ranges, performance SLAs)
- [x] Scope is clearly bounded (Phase 1 only: Architect + Basin, no Navigator/Curator)
- [x] Dependencies identified (existing Neo4j schema, AttractorBasin model, CLAUSE paper algorithms)

---

## Success Criteria

### Phase 1 Success Metrics
1. **Subgraph Quality**:
   - Precision@50: ≥85% of selected edges are query-relevant (manual eval on 100 test queries)
   - Compactness: Average subgraph size ≤ β_edge (strict budget compliance)

2. **Basin Evolution**:
   - Concept reappearance: Basin strength increases exactly +0.2 per occurrence
   - Co-occurrence accuracy: Tracked pairs match ground truth on 50 test document sets

3. **Performance**:
   - Edge scoring latency: <10ms per edge (p95)
   - Subgraph construction: <500ms total for 50-edge budget (p95)
   - Concurrent processing: 100+ documents/sec throughput

4. **Integration**:
   - Zero breaking changes to existing Neo4j queries
   - Backward compatible with AttractorBasin model v1.0
   - All existing tests pass after integration

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked and clarified
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---

## References

**CLAUSE Paper**:
- arXiv:2509.21035 "CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering"
- Algorithm 1 (Appendix): Subgraph Architect pseudocode
- Section 4.2: 5-signal edge scoring formula

**Dionysus Specs**:
- Spec 027: Basin Frequency Strengthening (30% complete, to be finished in Phase 1)
- CLAUSE_INTEGRATION_ANALYSIS.md: Detailed integration approach
- SELF_EVOLVING_KG_INTEGRATION.md: Unified architecture

**Related Specs** (Future Phases):
- Spec 028: ThoughtSeed Bulk Processing (Phase 3)
- Spec 031: Write Conflict Resolution (Phase 2)
- Spec 032: Emergent Pattern Detection (Phase 2)
- Spec 033: Causal Reasoning (Phase 3)
