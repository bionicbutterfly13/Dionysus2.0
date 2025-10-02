# Quickstart: Archimedes-Daedalus Synergistic System

## Prerequisites

### Constitutional Requirements
```bash
# Verify NumPy compliance
python -c "import numpy; assert numpy.__version__.startswith('1.'), 'NumPy 2.x violates constitution'"

# Check ThoughtSeed integration
python -c "import redis; r = redis.Redis(); print('Redis connected:', r.ping())"
```

### Environment Setup
```bash
# Activate constitutional environment
source asi-arch-frozen-env/bin/activate

# Install dependencies
pip install "numpy<2" --force-reinstall
pip install redis neo4j-driver fastapi uvicorn
```

## Quick Start Scenarios

### Scenario 1: Novel Problem Recognition (100ms requirement)
**User Story**: Submit a novel problem and verify rapid pattern recognition

```python
import requests
import time

# Submit novel problem
problem = {
    "problem_description": "How to optimize memory usage in recursive algorithms while maintaining readability",
    "domain": "computer_science",
    "complexity_level": "medium"
}

start_time = time.time()
response = requests.post("http://localhost:8000/api/archimedes/solve_novel_problem", json=problem)
processing_time = (time.time() - start_time) * 1000

# Verify requirements
assert response.status_code == 200
assert processing_time <= 100, f"Processing took {processing_time}ms, exceeds 100ms requirement"
assert response.json()["novelty_detected"] is True
print(f"✅ Novel problem recognized in {processing_time:.2f}ms")
```

### Scenario 2: Specialized Agent Creation (>90% accuracy requirement)
**User Story**: Create specialized agent for specific domain

```python
domain_spec = {
    "domain_specification": {
        "domain_name": "algorithm_optimization",
        "expertise_areas": ["memory_management", "recursion", "performance_analysis"],
        "complexity_level": "medium",
        "required_capabilities": ["code_analysis", "performance_profiling"]
    },
    "required_tools": ["memory_profiler", "code_analyzer", "performance_tester"],
    "context_requirements": {
        "expertise_level": "expert",
        "specialization_focus": "memory_optimization"
    }
}

response = requests.post("http://localhost:8000/api/daedalus/create_specialized_agent", json=domain_spec)

# Verify requirements
assert response.status_code == 201
agent = response.json()
assert agent["subspecialty_domain"] == "algorithm_optimization"
assert "memory_profiler" in agent["available_tools"]
print(f"✅ Specialized agent created: {agent['agent_id']}")
```

### Scenario 3: Semantic Agent Matching (500ms requirement)
**User Story**: Match problem to most suitable agents

```python
problem_data = {
    "problem": {
        "description": "Optimize memory usage in recursive algorithms",
        "domain": "computer_science",
        "complexity_level": "medium",
        "content_type": "text"
    },
    "matching_criteria": {
        "max_agents": 3,
        "min_similarity_threshold": 0.8,
        "include_reasoning": True
    }
}

start_time = time.time()
response = requests.post("http://localhost:8000/api/semantic/match_problem_to_agents", json=problem_data)
matching_time = (time.time() - start_time) * 1000

# Verify requirements
assert response.status_code == 200
assert matching_time <= 500, f"Matching took {matching_time}ms, exceeds 500ms requirement"
matches = response.json()["matches"]
assert len(matches) > 0
assert all(match["similarity_score"] >= 0.8 for match in matches)
print(f"✅ Agent matching completed in {matching_time:.2f}ms, found {len(matches)} suitable agents")
```

### Scenario 4: Committee Formation (2s requirement)
**User Story**: Form reasoning committee for complex problem

```python
# Use top agents from previous matching
agent_ids = [match["agent_id"] for match in matches[:3]]

committee_request = {
    "problem_context": {
        "problem_description": "Optimize memory usage in recursive algorithms",
        "domain": "computer_science",
        "complexity_level": "medium",
        "required_expertise": ["memory_management", "algorithm_optimization"]
    },
    "member_agents": agent_ids,
    "formation_strategy": "complementary_expertise",
    "coordination_protocol": "consensus_building"
}

start_time = time.time()
response = requests.post("http://localhost:8000/api/committees/form", json=committee_request)
formation_time = (time.time() - start_time) * 1000

# Verify requirements
assert response.status_code == 201
assert formation_time <= 2000, f"Committee formation took {formation_time}ms, exceeds 2s requirement"
committee = response.json()
assert len(committee["member_agents"]) == 3
print(f"✅ Committee formed in {formation_time:.2f}ms with {len(committee['member_agents'])} agents")
```

### Scenario 5: Personal Knowledge Base Interaction
**User Story**: Engage with autobiographical construct of ideas

```python
# Add knowledge to personal base
knowledge_entry = {
    "content": "Recursive algorithms can be optimized using memoization, tail recursion, or iterative approaches",
    "domain": "computer_science",
    "tags": ["algorithms", "optimization", "recursion"],
    "confidence_level": 0.9,
    "source": "personal_experience"
}

response = requests.post("http://localhost:8000/api/knowledge/add", json=knowledge_entry)
assert response.status_code == 201

# Query knowledge base
query = {
    "query": "How to optimize recursive algorithms?",
    "context": "memory_efficiency",
    "include_confidence": True
}

response = requests.post("http://localhost:8000/api/knowledge/query", json=query)
assert response.status_code == 200
results = response.json()
assert len(results["matches"]) > 0
assert all(result["confidence_level"] > 0.7 for result in results["matches"])
print(f"✅ Knowledge base query returned {len(results['matches'])} relevant entries")
```

### Scenario 6: Authentication & Session Management
**User Story**: Single-user authentication via GoHighLevel

```python
# Authenticate with passcode
auth_request = {
    "passcode": "user_specific_passcode",
    "membership_id": "ghl_membership_123"
}

response = requests.post("http://localhost:8000/api/auth/login", json=auth_request)
assert response.status_code == 200
session_token = response.json()["session_token"]

# Verify session
headers = {"Authorization": f"Bearer {session_token}"}
response = requests.get("http://localhost:8000/api/auth/verify", headers=headers)
assert response.status_code == 200
assert response.json()["valid"] is True
print("✅ Authentication and session management working")
```

### Scenario 7: Capacity Management
**User Story**: Auto-purge with user alerts when approaching limits

```python
# Check capacity status
response = requests.get("http://localhost:8000/api/system/capacity")
capacity = response.json()

if capacity["pattern_library_usage"] > 0.9:
    # Trigger capacity management
    management_request = {
        "purge_strategy": "least_recently_accessed",
        "preserve_high_performance": True,
        "user_alert": True
    }
    
    response = requests.post("http://localhost:8000/api/system/manage_capacity", json=management_request)
    assert response.status_code == 200
    result = response.json()
    assert "patterns_marked_for_removal" in result
    assert "user_notification_sent" in result
    print(f"✅ Capacity management triggered, {result['patterns_marked_for_removal']} patterns marked for removal")
```

### Scenario 8: No-Hallucination Policy Validation
**User Story**: System maintains strict confidence levels and indicates uncertainty

```python
# Query for information outside knowledge base
uncertain_query = {
    "query": "What is the solution to an extremely specific, novel problem never encountered before?",
    "domain": "hypothetical",
    "require_confidence_metrics": True
}

response = requests.post("http://localhost:8000/api/knowledge/query", json=uncertain_query)
assert response.status_code == 200
result = response.json()

# Verify no-hallucination policy
if result["confidence_level"] < 0.3:
    assert "I don't know" in result["response"] or result["response"] == "Insufficient information"
    assert result["confidence_level"] is not None
    assert "data_sources" in result
    print("✅ No-hallucination policy enforced - low confidence properly indicated")
```

## Performance Validation

### Load Testing (1000 concurrent sessions requirement)
```python
import asyncio
import aiohttp

async def concurrent_request():
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/api/archimedes/solve_novel_problem", 
                               json={"problem_description": "test", "domain": "test"}) as response:
            return response.status

async def load_test():
    tasks = [concurrent_request() for _ in range(1000)]
    results = await asyncio.gather(*tasks)
    success_rate = sum(1 for status in results if status == 200) / len(results)
    assert success_rate >= 0.95, f"Success rate {success_rate} below 95% threshold"
    print(f"✅ Load test passed with {success_rate*100:.1f}% success rate")

# Run load test
asyncio.run(load_test())
```

## Integration Testing

### ASI-Arch Compatibility Check
```python
# Verify ASI-Arch integration
response = requests.get("http://localhost:8000/api/integration/asi_arch/status")
assert response.status_code == 200
status = response.json()
assert status["asi_goto_preserved"] is True
assert status["thoughtseed_active"] is True
print("✅ ASI-Arch integration preserved")
```

### ThoughtSeed Integration Check
```python
# Verify consciousness detection
consciousness_request = {
    "architecture_state": "pattern_evolution_active",
    "problem_context": "novel_problem_solving"
}

response = requests.post("http://localhost:8000/api/integration/thoughtseed/consciousness", 
                        json=consciousness_request)
assert response.status_code == 200
result = response.json()
assert "consciousness_level" in result
assert 0 <= result["consciousness_level"] <= 1
print(f"✅ ThoughtSeed consciousness detection active: {result['consciousness_level']}")
```

## Success Criteria Validation

### Pattern Evolution Performance (10% improvement/month requirement)
```python
# Get baseline performance
response = requests.get("http://localhost:8000/api/metrics/pattern_evolution")
metrics = response.json()

baseline = metrics["baseline_performance"]
current = metrics["current_performance"]
improvement = (current - baseline) / baseline

assert improvement >= 0.1, f"Pattern evolution improvement {improvement*100:.1f}% below 10% requirement"
print(f"✅ Pattern evolution showing {improvement*100:.1f}% improvement")
```

### Agent Specialization Performance (25% better than general requirement)
```python
# Compare specialized vs general agent performance
response = requests.get("http://localhost:8000/api/metrics/agent_performance")
metrics = response.json()

specialized_performance = metrics["specialized_agents"]["average_performance"]
general_performance = metrics["general_agents"]["average_performance"]
improvement = (specialized_performance - general_performance) / general_performance

assert improvement >= 0.25, f"Specialized agent improvement {improvement*100:.1f}% below 25% requirement"
print(f"✅ Specialized agents performing {improvement*100:.1f}% better than general agents")
```

## Cleanup

```python
# Cleanup test data
requests.delete("http://localhost:8000/api/test/cleanup")
print("✅ Test cleanup completed")
```

## Manual Testing Checklist

- [ ] System starts without constitutional violations
- [ ] Pattern library loads with existing patterns
- [ ] Agent creation responds to real problem contexts
- [ ] Semantic matching provides meaningful results
- [ ] Committee formation creates diverse expertise groups
- [ ] Personal knowledge base accepts and retrieves user ideas
- [ ] Authentication integrates with GoHighLevel API
- [ ] Capacity management alerts before auto-purging
- [ ] No-hallucination policy prevents made-up responses
- [ ] ASI-Arch functionality remains fully accessible
- [ ] ThoughtSeed consciousness detection operates correctly
- [ ] Performance meets all specified requirements

**Success Threshold**: All automated tests pass AND all manual checklist items verified.