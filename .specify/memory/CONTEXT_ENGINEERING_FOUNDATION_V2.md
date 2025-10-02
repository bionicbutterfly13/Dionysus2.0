# Context Engineering: Quick Reference

## What It Is
Groups related concepts automatically using attractor basins.

## Examples

```python
# Example 1: Create basin
from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager

manager = AttractorBasinManager()
basin = manager.create_basin("ml_concepts", strength=1.5)
# Result: Concepts like "AI", "neural nets", "deep learning" cluster here

# Example 2: Find related
manager.find_related("transformers", min_similarity=0.7)
# Returns: basins containing related concepts

# Example 3: Search with basins
query_basin = manager.create_query_basin("semantic search")
results = manager.get_influenced_basins(query_basin)
# Returns: Documents in activated basins
```

## Basin Landscape (Visual)
```
ml_concepts [1.8] ←→ ai_research [1.5]
     ↓                    ↓
statistics [1.2]    transformers [1.6]
```

## When to Use
- Document search (finds related, not just exact matches)
- Tag recommendations (suggests similar tags)
- Concept clustering (groups related ideas)

## Files
- Implementation: `extensions/context_engineering/attractor_basin_dynamics.py`
- Tests: `backend/tests/test_context_engineering_spec_pipeline.py`
- Full theory: See davidkimai/Context-Engineering repo

**Token count**: ~200 tokens
