# Context Engineering Principles for Dionysus Development

**Source**: David Kimai's Context Engineering Framework
**Purpose**: ALL code/docs I write for this project MUST follow these principles
**Date**: 2025-10-01

---

## Core Principles (Internalized)

### 1. **Atomic Clarity** (Level 1 - Atoms)
```
Rule: Every instruction must be standalone and minimal
Application: [TASK] + [CONSTRAINTS] + [OUTPUT FORMAT]

✅ DO:
"Create basin for concept X. Return basin_id."

❌ DON'T:
"We need to think about creating a basin, considering various factors..."
```

### 2. **Few-Shot Learning** (Level 2 - Molecules)
```
Rule: Show 2-3 concrete examples, not abstract theory
Application: Demonstrate with runnable code

✅ DO:
# Example 1: Search
manager.create_basin("search", strength=1.0)

# Example 2: Tags
manager.create_basin("tags", strength=0.8)

❌ DON'T:
"Basins can be created for various purposes with configurable parameters..."
```

### 3. **Token Efficiency** (Universal)
```
Rule: Measure token cost vs value delivered
Application: Every doc/code addition must justify its tokens

BEFORE adding anything, ask:
1. What value does this provide?
2. What is the token cost?
3. Is there a more efficient way?
```

### 4. **Ruthless Pruning** (Universal)
```
Rule: Deletion beats padding
Application: Remove anything that doesn't directly contribute

Process:
1. Write full version
2. Cut in half
3. Cut in half again
4. Now you have the right amount
```

### 5. **Runnable Over Theory** (Level 3-4)
```
Rule: Prioritize executable code over explanations
Application: Show working examples first, explain second

✅ DO:
```python
# Working example
manager = AttractorBasinManager()
basin = manager.create_basin("ml_concepts")
# Output: basin_id = "ml_concepts_001"
```

❌ DON'T:
"The AttractorBasinManager class provides a sophisticated interface
for creating and managing attractor basins in the cognitive landscape..."
```

### 6. **Progressive Complexity** (Levels 1-6)
```
Rule: Start simple, build incrementally
Application: Atoms → Molecules → Cells → Organs → Systems → Fields

Level 1: Basic basin creation
Level 2: Basin with examples
Level 3: Basin with persistence
Level 4: Multi-basin workflows
Level 5: Basin reasoning patterns
Level 6: Field dynamics
```

### 7. **Semantic Anchoring** (Attractor Dynamics)
```
Rule: Create deliberate stable states (attractors)
Application: Design code so common patterns are easy to reach

In our context:
- "Search" basin is a strong attractor (frequently used)
- "Niche concept" basin is weak attractor (rarely used)
- System naturally gravitates toward strong attractors
```

### 8. **Context as Payload** (Core Definition)
```
Rule: Context = Complete information at inference time
Application: C = A(instructions, knowledge, tools, memory, state, query)

For every function:
1. What instructions does it need?
2. What knowledge (data)?
3. What tools (capabilities)?
4. What memory (state)?
5. What current query?
6. Is anything missing?
```

### 9. **Measure Everything** (Operational)
```
Rule: No guessing, only data
Application: Track token cost, latency, quality

Metrics to track:
- Token count per operation
- Execution time
- Memory usage
- Success rate
- User satisfaction
```

### 10. **Visualization Over Explanation** (Communication)
```
Rule: Diagrams > Paragraphs
Application: Show relationships visually

✅ DO:
```
Basin Landscape:
ml_concepts (1.8) ←→ ai_research (1.5)
     ↓
statistics (1.2)
```

❌ DON'T:
"The ml_concepts basin has a strength of 1.8 and is related to
the ai_research basin which has strength 1.5..."
```

---

## Application to Dionysus Code

### Every Python Function Must:

1. **Have atomic clarity** in docstring
   ```python
   def create_basin(concept: str) -> str:
       """Create basin for concept. Returns basin_id."""
   ```

2. **Include few-shot examples** in docstring
   ```python
   """
   Examples:
   >>> create_basin("search")
   'search_001'
   >>> create_basin("tags")
   'tags_001'
   """
   ```

3. **Optimize token usage**
   - Minimal docstring (3 lines max for simple functions)
   - No redundant parameters
   - Clear naming (no need to explain)

4. **Be runnable immediately**
   - No setup required beyond imports
   - Sensible defaults
   - Works with minimal context

### Every Markdown Doc Must:

1. **Start with atomic task**
   ```markdown
   # What This Does
   Creates attractor basins for concept clustering.

   # Quick Start
   manager = AttractorBasinManager()
   basin = manager.create_basin("my_concept")
   ```

2. **Show examples before theory**
   - First: 3 working examples
   - Then: How it works
   - Last: Theory/math

3. **Be pruned ruthlessly**
   - Max 200 lines for foundation docs
   - Max 50 lines for guides
   - Max 10 lines for quick refs

4. **Visualize when possible**
   - ASCII diagrams
   - Code examples
   - Flow charts

### Every Workflow Step Must:

1. **Have measurable value**
   ```
   Step: Show Context Engineering foundation
   Token cost: ~300 tokens
   Value: User understands basins
   Worth it? Only if user doesn't know basins already
   ```

2. **Be prunable**
   ```
   Can this step be skipped?
   YES → Make it optional
   NO → Keep it, but minimize
   ```

3. **Progress complexity**
   ```
   /specify: Level 1-2 (atoms, examples)
   /plan: Level 3-4 (memory, workflows)
   /tasks: Level 4-5 (workflows, reasoning)
   ```

---

## Checklist for ALL Future Code

Before I write ANY code or documentation:

- [ ] **Atomic**: Is the core instruction clear and minimal?
- [ ] **Examples**: Do I show 2-3 concrete working examples?
- [ ] **Token cost**: Have I measured/estimated token usage?
- [ ] **Pruned**: Can I cut this in half and still deliver value?
- [ ] **Runnable**: Can someone copy-paste and run this?
- [ ] **Progressive**: Does it build from simple to complex?
- [ ] **Measured**: How will I know if this works?
- [ ] **Visualized**: Can I show this with a diagram?

---

## What This Means for Today's Work

### Violations Found:

1. **CONTEXT_ENGINEERING_FOUNDATION.md** (9KB)
   - Violation: Not atomic, not pruned
   - Token cost: ~2,400 tokens
   - Should be: ~200 tokens (12x reduction needed)

2. **Slash command additions**
   - Violation: Added steps without measuring value
   - Token cost: Unknown
   - Should be: Measured and optional

3. **Test suite**
   - ✅ Good: Has runnable examples
   - ❌ Bad: Tests theory, not practical usage

4. **Documentation approach**
   - Violation: Theory-first, examples-last
   - Should be: Examples-first, theory-optional

### Required Changes:

1. **Prune foundation doc to 200 lines** (90% reduction)
2. **Add token cost measurement** to all workflow steps
3. **Rewrite as: Examples → Usage → Theory** (reverse order)
4. **Create visual diagram** of basin landscape
5. **Make all workflow steps optional** with clear value props

---

## Going Forward

**Every PR I submit must**:
1. Pass the 8-point checklist above
2. Include token cost analysis
3. Start with examples, end with theory
4. Be pruned to minimum viable size
5. Have visual diagrams where applicable

**Every explanation I give must**:
1. Use atomic, clear language
2. Show working code examples
3. Justify token usage
4. Build progressively from simple to complex

**Every decision I make must**:
1. Be measurable
2. Start with first principles
3. Optimize for token efficiency
4. Prioritize runnable over theoretical

---

**This is my operating system now.**
**All Dionysus code follows Context Engineering principles.**
**No exceptions.**
