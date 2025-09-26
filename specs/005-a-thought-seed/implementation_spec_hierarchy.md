# ThoughtSeed Watching Implementation Specification Hierarchy

**Date**: 2025-09-26
**Feature**: ThoughtSeed State Watching for ASI-GO-2 Architecture

## Critical Discovery: Dependency Cascade

### 🚨 **Blocker Identified**
The original specification assumed ASI-GO-2 was fully integrated. **It is not.**

### **Specification Hierarchy (Dependencies Bottom-Up)**

## Spec A: ASI-GO-2 Core Components (Foundation)
**Priority**: 🔴 **CRITICAL** - Blocks all other work
**Files Required**:
- `backend/src/services/asi_go_2/cognitive_core/memory_system.py`
- `backend/src/services/asi_go_2/cognitive_core/thoughtseed_competition.py`
- `backend/src/services/asi_go_2/cognitive_core/__init__.py`

**Implementation Requirements**:
1. **CognitionBase class** - Stores problem-solving patterns
2. **Pattern class** - Individual pattern entities with success_rate, confidence
3. **InnerWorkspace class** - ThoughtSeed competition arena
4. **ThoughtGenerator class** - Creates competing thoughts
5. **ThoughtType enum** - GOAL, ACTION, BELIEF, PERCEPTION types

**Success Criteria**:
- Researcher component can import and instantiate without errors
- Pattern competition runs and selects winning thoughts

## Spec B: ASI-GO-2 Service Integration (Orchestration)
**Priority**: 🔴 **HIGH** - Required for system functionality
**Dependencies**: Spec A complete

**Implementation Requirements**:
1. **Service Discovery** - How components find each other
2. **PROCESS_ANY_DOCUMENT Integration** - Document → Pattern extraction
3. **Memory Persistence** - Pattern storage and retrieval
4. **API Endpoints** - Research query processing

**Success Criteria**:
- Document upload creates patterns in CognitionBase
- Research queries trigger thoughtseed competition
- Patterns accumulate and improve over time

## Spec C: ThoughtSeed State Watching (Original Feature)
**Priority**: 🟡 **FEATURE** - The watching capability
**Dependencies**: Specs A & B complete

**Implementation Requirements**:
1. **StateWatcher Service** - Monitors InnerWorkspace dynamics
2. **Redis Logging Backend** - 10-minute TTL structured logs
3. **Watching Toggle API** - Enable/disable per ThoughtSeed instance
4. **Log View Integration** - Display watched states

**Success Criteria**:
- Toggle watching on specific ThoughtSeed instances
- Capture detailed state during competition cycles
- View logs through existing log interface

## **Current State Assessment**

### ✅ **What Exists:**
- ASI-GO-2 component shells (researcher.py, engineer.py, analyst.py)
- Basic InnerWorkspace concept in test files
- Integration specifications and documentation

### ❌ **Critical Missing:**
- All of `cognitive_core/` directory
- Actual ThoughtSeed competition implementation
- Memory system integration
- Document processing pipeline connection

## **Recommendation: Specification Priority Order**

1. **Spec A** (Foundation) - Implement core cognitive components
2. **Spec B** (Integration) - Connect ASI-GO-2 to document pipeline
3. **Spec C** (Watching) - Add watching capability to working system

**Estimated Timeline:**
- Spec A: 1-2 hours (core classes and competition logic)
- Spec B: 2-3 hours (service integration and persistence)
- Spec C: 1-2 hours (watching hooks and logging)

**Total**: ~5-7 hours of systematic implementation

## **Decision Point**
Do we:
1. **Build everything** - Complete ASI-GO-2 integration + watching feature
2. **Focus on watching only** - Mock the missing components for demo purposes
3. **ASI-GO-2 first** - Complete integration, watching feature in follow-up

**Recommendation**: Option 1 (Build everything) - The system needs ASI-GO-2 working anyway, and watching won't be meaningful without real ThoughtSeed competition.