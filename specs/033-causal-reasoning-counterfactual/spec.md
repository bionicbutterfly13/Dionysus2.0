# Spec 033: Causal Reasoning and Counterfactual Simulation for Agentic Knowledge Graphs

**Status**: DRAFT
**Priority**: HIGH
**Dependencies**: 032 (Emergent Patterns), 029 (Curiosity Agents), 028 (ThoughtSeeds)
**Created**: 2025-10-01

## Overview

Implement causal reasoning and counterfactual simulation capabilities in the agentic knowledge graph using **Causal Bayesian Networks**, **Structural Causal Models**, and **Counterfactual KG reasoning** frameworks (COULDD, DoWhy, causalgraph). Enable agents to simulate interventions, answer "what if" questions, perform root cause analysis, and plan actions based on causal effects.

## Problem Statement

Current agentic knowledge graph lacks causal reasoning:

### Missing Capabilities

1. **No Causal Relationships**
   - Relationships are associative ("RELATED_TO", "EXTENDS") not causal ("CAUSES", "ENABLES")
   - Can't answer: "If I apply technique A, what effect on metric B?"
   - No distinction between correlation and causation

2. **No Counterfactual Reasoning**
   - Can't simulate "what if" scenarios
   - Example: "What would have happened if we used algorithm X instead of Y?"
   - No hypothetical premise testing

3. **No Intervention Planning**
   - Agents can't simulate action outcomes before execution
   - Can't optimize action sequences based on causal chains
   - No "do-operations" (Pearl's causal calculus)

4. **No Root Cause Analysis**
   - Given observed outcome, can't trace back to probable root causes
   - No abductive reasoning over causal structure
   - No multi-factor causal analysis

## Research-Backed Requirements

Based on your research on causal models and counterfactual reasoning tools:

### FR1: Causal Bayesian Network Integration
**Description**: Represent probabilistic causal relationships among entities
**Research Source**: "Causal Bayesian Networks incorporate probabilistic causal relationships, allowing agents to simulate interventions and predict downstream effects"
**Acceptance Criteria**:
- [ ] Nodes represent concepts/entities with conditional probability tables
- [ ] Directed edges represent causal influences with weights
- [ ] "Do-operations" enable intervention simulation
- [ ] Agents can predict downstream effects of actions
- [ ] Uncertainty captured via probabilities

**Implementation**:
```python
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

class CausalBayesianNetworkBuilder:
    """Build Causal Bayesian Network from knowledge graph"""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.bn = None

    def build_from_causal_subgraph(self, root_concept: str, max_depth: int = 3):
        """Extract causal subgraph and build Bayesian Network"""

        # Step 1: Extract causal relationships from Neo4j
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (root:Concept {name: $root_concept})
                             -[:CAUSES|ENABLES|INHIBITS*1..$max_depth]->()
                WITH nodes(path) as nodes, relationships(path) as rels
                RETURN nodes, rels
            """, root_concept=root_concept, max_depth=max_depth)

            nodes = []
            edges = []
            for record in result:
                for node in record["nodes"]:
                    nodes.append(node["name"])
                for rel in record["rels"]:
                    edges.append((
                        rel.start_node["name"],
                        rel.end_node["name"],
                        rel.type
                    ))

        # Step 2: Build Bayesian Network
        self.bn = BayesianNetwork([(source, target) for source, target, _ in edges])

        # Step 3: Define CPDs (Conditional Probability Distributions)
        # In practice, learn from data or use domain knowledge
        for node in nodes:
            cpd = self._create_cpd_for_node(node)
            self.bn.add_cpds(cpd)

        # Step 4: Validate network
        assert self.bn.check_model()

        logger.info(f"🔬 Built Causal Bayesian Network: {len(nodes)} nodes, {len(edges)} edges")
        return self.bn

    def simulate_intervention(self, intervention: Dict[str, float], target: str) -> Dict:
        """Simulate 'do-operation' and predict effect on target"""

        # Pearl's do-calculus: P(target | do(intervention))
        inference = VariableElimination(self.bn)

        # Remove edges INTO intervention variables (graph surgery)
        modified_bn = self._apply_do_operator(intervention)

        # Query target variable
        result = inference.query(
            variables=[target],
            evidence=intervention
        )

        return {
            "intervention": intervention,
            "target": target,
            "probability_distribution": result.values,
            "expected_value": float(np.mean(result.values))
        }

    def _apply_do_operator(self, intervention: Dict) -> BayesianNetwork:
        """Apply Pearl's do-operator (graph surgery)"""
        modified_bn = self.bn.copy()

        for var in intervention.keys():
            # Remove all incoming edges to intervention variable
            parents = modified_bn.get_parents(var)
            for parent in parents:
                modified_bn.remove_edge(parent, var)

        return modified_bn
```

### FR2: Structural Causal Models (SCM)
**Description**: Use structural equations to define how variable changes propagate
**Research Source**: "SCMs define how variable changes propagate through the system, enabling counterfactual questions"
**Acceptance Criteria**:
- [ ] Structural equations defined for causal relationships
- [ ] Variable propagation calculated via equations
- [ ] Counterfactual queries supported ("What would have happened if...")
- [ ] Integration with Neo4j causal edges

**Neo4j Schema for Causal Relationships**:
```cypher
// Causal relationship with structural equation
CREATE (source:Concept {name: "training data size"})
       -[:CAUSES {
         relationship_id: "cause_001",
         structural_equation: "accuracy = 0.65 + 0.15 * log(data_size)",
         causal_strength: 0.85,  // 0.0-1.0
         confidence: 0.92,
         supporting_papers: ["doc_123", "doc_456"],
         mechanism: "More data reduces overfitting and improves generalization",
         evidence_type: "empirical"  // empirical/theoretical/experimental
       }]->(target:Concept {name: "model accuracy"})

// Alternative: Multi-factor causal relationship (hyper-relation)
CREATE (factor1:Concept {name: "learning rate"})
CREATE (factor2:Concept {name: "batch size"})
CREATE (outcome:Concept {name: "convergence speed"})

CREATE (causal_relation:CausalRelation {
  relation_id: "multi_cause_001",
  structural_equation: "convergence_time = 100 / (lr * sqrt(batch_size))",
  factors: ["learning rate", "batch size"],
  outcome: "convergence speed",
  interaction_type: "multiplicative",
  confidence: 0.88
})

CREATE (factor1)-[:CONTRIBUTES_TO]->(causal_relation)
CREATE (factor2)-[:CONTRIBUTES_TO]->(causal_relation)
CREATE (causal_relation)-[:DETERMINES]->(outcome)
```

**Implementation**:
```python
class StructuralCausalModel:
    """Structural Causal Model for variable propagation"""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.equations = {}

    def load_from_knowledge_graph(self, root_concept: str):
        """Load structural equations from Neo4j"""

        with self.driver.session() as session:
            result = session.run("""
                MATCH (source:Concept)-[r:CAUSES]->(target:Concept)
                WHERE source.name = $root_concept OR target.name = $root_concept
                RETURN source.name as source,
                       target.name as target,
                       r.structural_equation as equation,
                       r.causal_strength as strength
            """, root_concept=root_concept)

            for record in result:
                self.equations[record["target"]] = {
                    "source": record["source"],
                    "equation": record["equation"],
                    "strength": record["strength"]
                }

        logger.info(f"📐 Loaded {len(self.equations)} structural equations")

    def simulate_propagation(self, initial_values: Dict[str, float]) -> Dict:
        """Simulate how variable changes propagate through causal structure"""

        current_values = initial_values.copy()
        propagated_values = {}

        # Topological sort to process in causal order
        for variable in self._topological_sort():
            if variable in self.equations:
                # Evaluate structural equation
                equation = self.equations[variable]["equation"]
                source_var = self.equations[variable]["source"]

                try:
                    # Safe evaluation (in production, use sympy or AST parsing)
                    result = self._evaluate_equation(
                        equation,
                        {source_var: current_values[source_var]}
                    )
                    propagated_values[variable] = result
                    current_values[variable] = result
                except Exception as e:
                    logger.warning(f"Failed to evaluate equation for {variable}: {e}")

        return propagated_values

    def counterfactual_query(self, factual: Dict, counterfactual: Dict, target: str) -> Dict:
        """Answer 'What would have happened if...' questions"""

        # Step 1: Propagate with factual values
        factual_outcome = self.simulate_propagation(factual)

        # Step 2: Propagate with counterfactual values
        counterfactual_outcome = self.simulate_propagation(counterfactual)

        # Step 3: Compare
        return {
            "factual_target_value": factual_outcome.get(target),
            "counterfactual_target_value": counterfactual_outcome.get(target),
            "causal_effect": counterfactual_outcome.get(target, 0) - factual_outcome.get(target, 0)
        }
```

### FR3: COULDD Framework for Counterfactual KG Reasoning
**Description**: Implement counterfactual updates and scenario simulation on knowledge graphs
**Research Source**: "COULDD adapts KG embedding models to hypothetical premises, enabling reasoning about alternate realities"
**Acceptance Criteria**:
- [ ] Hypothetical premise injection into KG
- [ ] Counterfactual scenario simulation
- [ ] Plausible facts retained while affected edges updated
- [ ] Classification of fact validity given counterfactual scenario

**Implementation**:
```python
from couldd import CounterfactualKGReasoner  # Hypothetical import

class CounterfactualScenarioSimulator:
    """Simulate counterfactual scenarios using COULDD framework"""

    def __init__(self, kg_embeddings):
        self.kg_embeddings = kg_embeddings
        self.reasoner = CounterfactualKGReasoner(kg_embeddings)

    def simulate_scenario(self, hypothetical_premise: str, query_facts: List[Tuple]) -> Dict:
        """
        Simulate scenario with hypothetical premise.

        Example:
          hypothetical_premise: "What if BERT never existed?"
          query_facts: [
            ("transformer", "ENABLES", "language_model"),
            ("GPT", "EXTENDS", "BERT"),
            ("T5", "EXTENDS", "BERT")
          ]
        """

        # Step 1: Inject hypothetical premise
        updated_kg = self.reasoner.inject_hypothesis(hypothetical_premise)

        # Step 2: Query affected facts
        results = []
        for (subject, relation, object) in query_facts:
            plausibility = self.reasoner.query_fact_plausibility(
                subject, relation, object,
                context=updated_kg
            )

            results.append({
                "fact": (subject, relation, object),
                "plausibility_baseline": self._query_baseline(subject, relation, object),
                "plausibility_counterfactual": plausibility,
                "affected": abs(plausibility - self._query_baseline(subject, relation, object)) > 0.1
            })

        # Step 3: Identify cascade effects
        cascade = self._detect_cascade_effects(results)

        return {
            "hypothetical_premise": hypothetical_premise,
            "query_results": results,
            "cascade_effects": cascade,
            "summary": self._summarize_counterfactual(results)
        }

    def _detect_cascade_effects(self, results: List[Dict]) -> List[Dict]:
        """Detect cascading causal effects from counterfactual"""
        cascade = []

        for result in results:
            if result["affected"]:
                # Find downstream effects
                subject, relation, object = result["fact"]
                downstream = self.reasoner.find_downstream_effects(object)

                cascade.append({
                    "trigger": result["fact"],
                    "downstream_effects": downstream
                })

        return cascade
```

### FR4: DoWhy Integration for Causal Effect Estimation
**Description**: Use Microsoft's DoWhy for causal modeling and effect estimation
**Research Source**: "DoWhy offers graph-based causal modeling and counterfactual effect estimation with robustness checks"
**Acceptance Criteria**:
- [ ] Causal graph modeling from Neo4j
- [ ] Causal effect identification (backdoor, frontdoor criterion)
- [ ] Effect estimation via matching/instrumental variables
- [ ] Refutation with robustness checks

**Implementation**:
```python
import dowhy
from dowhy import CausalModel

class CausalEffectEstimator:
    """Estimate causal effects using DoWhy"""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def estimate_causal_effect(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str] = None
    ) -> Dict:
        """Estimate causal effect of treatment on outcome"""

        # Step 1: Build causal graph from Neo4j
        causal_graph = self._build_causal_graph(treatment, outcome, confounders)

        # Step 2: Create DoWhy causal model
        # (In practice, also need data - could come from papers, experiments)
        model = CausalModel(
            data=None,  # Would be dataframe in production
            treatment=treatment,
            outcome=outcome,
            graph=causal_graph
        )

        # Step 3: Identify causal effect (backdoor criterion)
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=False
        )

        logger.info(f"📊 Identified causal effect: {identified_estimand}")

        # Step 4: Estimate causal effect
        # (Requires data - placeholder for now)
        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_matching"
        )

        # Step 5: Refute estimate (robustness checks)
        refutation = model.refute_estimate(
            identified_estimand,
            causal_estimate,
            method_name="random_common_cause"
        )

        return {
            "treatment": treatment,
            "outcome": outcome,
            "identified_estimand": str(identified_estimand),
            "causal_effect": causal_estimate.value,
            "refutation": refutation.refutation_result
        }

    def _build_causal_graph(self, treatment, outcome, confounders):
        """Build causal graph from Neo4j for DoWhy"""

        with self.driver.session() as session:
            # Find causal paths between treatment and outcome
            result = session.run("""
                MATCH path = (t:Concept {name: $treatment})
                             -[:CAUSES*1..5]->(o:Concept {name: $outcome})
                RETURN nodes(path) as nodes, relationships(path) as rels
            """, treatment=treatment, outcome=outcome)

            # Convert to DoWhy graph format
            edges = []
            for record in result:
                rels = record["rels"]
                for rel in rels:
                    edges.append((rel.start_node["name"], rel.end_node["name"]))

            # Add confounders if provided
            if confounders:
                for confounder in confounders:
                    edges.append((confounder, treatment))
                    edges.append((confounder, outcome))

            # DoWhy graph format (DOT notation)
            graph_str = "digraph {" + ";".join([f'"{s}" -> "{t}"' for s, t in edges]) + "}"
            return graph_str
```

### FR5: Root Cause Analysis via Causal Tracing
**Description**: Trace from observed outcome to probable root causes
**Research Source**: "Agent traces from observed outcome to probable root causes using directed causal structure"
**Acceptance Criteria**:
- [ ] Given outcome, identify candidate root causes
- [ ] Rank causes by causal strength and evidence
- [ ] Support multi-factor causal analysis
- [ ] Provide explanation with supporting papers

**Implementation**:
```python
class RootCauseAnalyzer:
    """Perform root cause analysis via causal graph tracing"""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def identify_root_causes(self, observed_outcome: str, max_depth: int = 5) -> List[Dict]:
        """Trace backward from outcome to find root causes"""

        with self.driver.session() as session:
            # Traverse causal graph backward
            result = session.run("""
                MATCH path = (cause:Concept)
                             -[r:CAUSES|ENABLES*1..$max_depth]->(outcome:Concept {name: $outcome})
                WHERE NOT (cause)<-[:CAUSES|ENABLES]-()  // Root nodes (no incoming causal edges)

                WITH cause, path, relationships(path) as rels

                // Calculate aggregated causal strength along path
                WITH cause, path,
                     reduce(strength = 1.0, rel IN rels | strength * rel.causal_strength) as path_strength

                RETURN cause.name as root_cause,
                       path_strength,
                       length(path) as path_length,
                       [node IN nodes(path) | node.name] as causal_chain,
                       [rel IN relationships(path) | rel.mechanism] as mechanisms
                ORDER BY path_strength DESC
                LIMIT 10
            """, outcome=observed_outcome, max_depth=max_depth)

            root_causes = []
            for record in result:
                root_causes.append({
                    "root_cause": record["root_cause"],
                    "causal_strength": record["path_strength"],
                    "path_length": record["path_length"],
                    "causal_chain": record["causal_chain"],
                    "mechanisms": record["mechanisms"],
                    "confidence": self._calculate_confidence(record)
                })

        logger.info(f"🔍 Identified {len(root_causes)} root causes for '{observed_outcome}'")
        return root_causes

    def explain_root_cause(self, root_cause: str, outcome: str) -> str:
        """Generate natural language explanation of causal path"""

        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (cause:Concept {name: $root_cause})
                             -[rels:CAUSES|ENABLES*]->(outcome:Concept {name: $outcome})

                RETURN [rel IN relationships(path) |
                        rel.mechanism + ' (supported by: ' +
                        reduce(papers = '', paper IN rel.supporting_papers |
                               papers + paper + ', ') + ')'
                       ] as explanation_chain
                LIMIT 1
            """, root_cause=root_cause, outcome=outcome)

            explanation_chain = result.single()["explanation_chain"]

            # Build narrative
            narrative = f"Root cause: {root_cause}\n\n"
            narrative += "Causal chain:\n"
            for i, explanation in enumerate(explanation_chain, 1):
                narrative += f"{i}. {explanation}\n"

            return narrative
```

### FR6: Prescriptive Planning via Causal Optimization
**Description**: Select actions with maximal desired effects based on causal graph
**Research Source**: "Agent selects actions with maximal upward causal edges, optimizing over chain-of-effect weights"
**Acceptance Criteria**:
- [ ] Given desired outcome, identify optimal intervention points
- [ ] Rank interventions by expected causal effect
- [ ] Consider chain effects and feedback loops
- [ ] Simulate intervention before execution

**Implementation**:
```python
class PrescriptivePlanner:
    """Plan optimal interventions using causal graph"""

    def __init__(self, neo4j_driver, causal_bn_builder):
        self.driver = neo4j_driver
        self.bn_builder = causal_bn_builder

    def recommend_interventions(self, desired_outcome: str, target_value: float) -> List[Dict]:
        """Recommend interventions to achieve desired outcome"""

        # Step 1: Find all concepts that causally affect outcome
        with self.driver.session() as session:
            result = session.run("""
                MATCH (intervention:Concept)
                      -[path:CAUSES|ENABLES*1..3]->(outcome:Concept {name: $outcome})

                WITH intervention,
                     reduce(strength = 1.0, rel IN relationships(path) |
                            strength * rel.causal_strength) as total_strength,
                     length(path) as path_length

                WHERE total_strength > 0.5  // Only strong causal effects

                RETURN intervention.name as intervention,
                       total_strength as causal_strength,
                       path_length,
                       intervention.actionability as actionability  // How easy to intervene
                ORDER BY total_strength DESC, path_length ASC
            """, outcome=desired_outcome)

            candidates = [dict(record) for record in result]

        # Step 2: Simulate each intervention
        recommendations = []
        for candidate in candidates:
            # Build Bayesian Network
            bn = self.bn_builder.build_from_causal_subgraph(candidate["intervention"])

            # Simulate intervention effect
            simulated_effect = bn.simulate_intervention(
                intervention={candidate["intervention"]: 1.0},  # "Turn on" intervention
                target=desired_outcome
            )

            recommendations.append({
                "intervention": candidate["intervention"],
                "causal_strength": candidate["causal_strength"],
                "path_length": candidate["path_length"],
                "actionability": candidate.get("actionability", 0.5),
                "predicted_effect": simulated_effect["expected_value"],
                "meets_target": simulated_effect["expected_value"] >= target_value,
                "confidence": simulated_effect.get("confidence", 0.7)
            })

        # Step 3: Rank by predicted effect and actionability
        recommendations.sort(
            key=lambda r: (r["meets_target"], r["predicted_effect"], r["actionability"]),
            reverse=True
        )

        logger.info(f"💡 Generated {len(recommendations)} intervention recommendations")
        return recommendations
```

## Integration with Existing Specs

### With Spec 029 (Curiosity Agents)
```python
# Curiosity agents use causal reasoning to generate better questions
class CausalCuriosityAgent(CuriosityDetectionAgent):
    def detect_curiosity_triggers(self, concepts: List[str]) -> List[CuriosityTrigger]:
        triggers = super().detect_curiosity_triggers(concepts)

        # NEW: Add causal gap triggers
        for concept in concepts:
            # Check if concept is effect but cause unknown
            root_causes = self.root_cause_analyzer.identify_root_causes(concept)
            if len(root_causes) == 0:
                triggers.append(CuriosityTrigger(
                    concept=concept,
                    prediction_error=0.9,
                    knowledge_gap_type="unknown_cause",
                    priority=0.9,
                    causal_question=f"What causes {concept}?"
                ))

        return triggers
```

### With Spec 032 (Emergent Patterns)
```python
# Meta-agent uses causal analysis to understand pattern emergence
class CausalMetaAgent(MetaAnalysisAgent):
    def analyze_graph_evolution(self, days: int = 7) -> Dict:
        analysis = super().analyze_graph_evolution(days)

        # NEW: Causal analysis of emergence
        for emerging_entity in analysis["emerging_entities"]:
            # Identify what caused this entity to emerge
            root_causes = self.root_cause_analyzer.identify_root_causes(
                emerging_entity["entity"]
            )

            emerging_entity["emergence_causes"] = root_causes
            emerging_entity["causal_explanation"] = self.root_cause_analyzer.explain_root_cause(
                root_causes[0]["root_cause"] if root_causes else "unknown",
                emerging_entity["entity"]
            )

        return analysis
```

## Test Strategy

### Unit Tests

```python
def test_causal_bayesian_network_intervention():
    """Test intervention simulation with Bayesian Network"""
    builder = CausalBayesianNetworkBuilder(driver)

    bn = builder.build_from_causal_subgraph("training data size")

    # Simulate doubling data size
    result = bn.simulate_intervention(
        intervention={"training_data_size": 2.0},
        target="model_accuracy"
    )

    assert result["expected_value"] > 0.7  # Accuracy should improve

def test_structural_causal_model_propagation():
    """Test variable propagation through structural equations"""
    scm = StructuralCausalModel(driver)
    scm.load_from_knowledge_graph("learning_rate")

    propagated = scm.simulate_propagation({
        "learning_rate": 0.01,
        "batch_size": 32
    })

    assert "convergence_speed" in propagated

def test_counterfactual_scenario_simulation():
    """Test counterfactual 'what if' query"""
    scm = StructuralCausalModel(driver)
    scm.load_from_knowledge_graph("algorithm_choice")

    result = scm.counterfactual_query(
        factual={"algorithm": "SGD"},
        counterfactual={"algorithm": "Adam"},
        target="training_time"
    )

    assert result["causal_effect"] != 0

def test_root_cause_analysis():
    """Test root cause identification"""
    analyzer = RootCauseAnalyzer(driver)

    root_causes = analyzer.identify_root_causes("poor_model_performance")

    assert len(root_causes) > 0
    assert root_causes[0]["causal_strength"] > 0.5

def test_prescriptive_planning():
    """Test intervention recommendation"""
    planner = PrescriptivePlanner(driver, bn_builder)

    recommendations = planner.recommend_interventions(
        desired_outcome="high_accuracy",
        target_value=0.9
    )

    assert len(recommendations) > 0
    assert recommendations[0]["predicted_effect"] >= 0.9
```

## Implementation Plan

### Phase 1: Causal Neo4j Schema (3-4 hours)
1. Extend Neo4j with causal relationship types (CAUSES, ENABLES, INHIBITS)
2. Add structural equations and causal strength properties
3. Add supporting evidence links
4. Test causal graph construction

### Phase 2: Causal Bayesian Network (4-5 hours)
1. Implement `CausalBayesianNetworkBuilder`
2. Extract causal subgraphs from Neo4j
3. Build Bayesian Networks with pgmpy
4. Implement intervention simulation (do-operator)
5. Test with synthetic data

### Phase 3: Structural Causal Models (4-5 hours)
1. Implement `StructuralCausalModel`
2. Load equations from Neo4j
3. Implement variable propagation
4. Implement counterfactual queries
5. Test equation evaluation

### Phase 4: DoWhy Integration (3-4 hours)
1. Implement `CausalEffectEstimator`
2. Convert Neo4j causal graph to DoWhy format
3. Estimate causal effects
4. Run robustness checks
5. Test with example causal graphs

### Phase 5: Root Cause Analysis (3-4 hours)
1. Implement `RootCauseAnalyzer`
2. Backward causal graph traversal
3. Causal strength aggregation
4. Natural language explanation generation
5. Test with complex causal chains

### Phase 6: Prescriptive Planning (4-5 hours)
1. Implement `PrescriptivePlanner`
2. Identify intervention candidates
3. Simulate intervention effects
4. Rank by predicted outcome
5. Test recommendation quality

### Phase 7: Integration & Testing (3-4 hours)
1. Integrate with Specs 029, 032
2. End-to-end causal reasoning tests
3. Performance validation
4. Documentation

**Total Estimated Time**: 24-31 hours

## Success Criteria

- [ ] Causal relationships stored in Neo4j with structural equations
- [ ] Bayesian Networks built from causal subgraphs
- [ ] Intervention simulation functional (do-operations)
- [ ] Counterfactual queries answerable
- [ ] Root cause analysis identifies causes with >0.5 strength
- [ ] Prescriptive planning recommends interventions
- [ ] All tests passing (unit + integration)

## References

- Research: Causal Models for Agent Planning
- Research: Counterfactual Reasoning Tools (COULDD, DoWhy, causalgraph)
- pgmpy: Python Bayesian Network library
- DoWhy: Microsoft causal inference library
- COULDD Framework: Counterfactual KG Reasoning
- Spec 029: Curiosity-Driven Background Agents
- Spec 032: Emergent Pattern Detection
