# Context Engineering Test Specification

**Version**: 1.0.0  
**Status**: ACTIVE DEVELOPMENT  
**Last Updated**: 2025-09-22  
**Specification Type**: Test Strategy and Implementation Guide

## 🎯 Test Strategy Overview

This specification defines the comprehensive testing strategy for the Context Engineering System, ensuring all requirements from `CONTEXT_ENGINEERING_SPEC.md` are properly validated.

## 🏗️ Test Architecture

### Test Pyramid Structure

```
                    ┌─────────────────┐
                    │   E2E Tests     │  <- 10% (Integration scenarios)
                    │   (Slow)        │
                    └─────────────────┘
                  ┌───────────────────────┐
                  │  Integration Tests    │  <- 20% (Component interaction)
                  │  (Medium)             │
                  └───────────────────────┘
              ┌─────────────────────────────────┐
              │        Unit Tests               │  <- 70% (Individual functions)
              │        (Fast)                   │
              └─────────────────────────────────┘
```

### Test Categories

#### 🔬 Unit Tests (70% of test suite)
- **Purpose**: Test individual functions and classes in isolation
- **Speed**: <1ms per test
- **Coverage Target**: 95% for critical components, 85% overall
- **Framework**: pytest with async support

#### 🔗 Integration Tests (20% of test suite)  
- **Purpose**: Test component interactions and data flow
- **Speed**: <100ms per test
- **Coverage**: All major component interfaces
- **Framework**: pytest with test fixtures

#### 🌐 End-to-End Tests (10% of test suite)
- **Purpose**: Test complete user workflows
- **Speed**: <5s per test
- **Coverage**: Critical user journeys
- **Framework**: pytest with web driver for dashboard tests

## 📋 Test Implementation Specification

### 🧪 Unit Test Specifications

#### Consciousness Detection Tests

```python
# File: tests/unit/test_consciousness_detection.py

import pytest
from extensions.context_engineering.core_implementation import ConsciousnessDetector, ConsciousnessLevel

class TestConsciousnessDetector:
    
    @pytest.fixture
    def detector(self):
        return ConsciousnessDetector()
    
    @pytest.mark.asyncio
    async def test_detect_dormant_consciousness(self, detector):
        """Test detection of DORMANT consciousness level"""
        # Given: Architecture with basic patterns
        arch_data = {
            'name': 'basic_linear',
            'program': 'class BasicLinear(nn.Module): def forward(self, x): return x',
            'analysis': 'simple linear transformation',
            'motivation': 'baseline implementation'
        }
        
        # When: Consciousness detection runs
        level = await detector.detect_consciousness_level(arch_data)
        
        # Then: Returns DORMANT level
        assert level == ConsciousnessLevel.DORMANT
        assert level.value < 0.2
    
    @pytest.mark.asyncio
    async def test_detect_self_aware_consciousness(self, detector):
        """Test detection of SELF_AWARE consciousness level"""
        # Given: Architecture with meta-learning patterns
        arch_data = {
            'name': 'meta_transformer',
            'program': 'class MetaTransformer with self_attention and adaptive_behavior',
            'analysis': 'emergent self-awareness patterns with recursive processing',
            'motivation': 'develop meta-learning and adaptive capabilities'
        }
        
        # When: Consciousness detection runs
        level = await detector.detect_consciousness_level(arch_data)
        
        # Then: Returns SELF_AWARE level
        assert level in [ConsciousnessLevel.SELF_AWARE, ConsciousnessLevel.META_AWARE]
        assert level.value >= 0.6
    
    @pytest.mark.asyncio
    async def test_consciousness_detection_performance(self, detector):
        """Test consciousness detection performance requirements"""
        import time
        
        # Given: Architecture data
        arch_data = {
            'name': 'test_arch',
            'program': 'class TestArch(nn.Module): pass',
            'analysis': 'test analysis',
            'motivation': 'test motivation'
        }
        
        # When: Detection runs 100 times
        start_time = time.time()
        for _ in range(100):
            await detector.detect_consciousness_level(arch_data)
        end_time = time.time()
        
        # Then: Average time < 100ms per detection
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.1, f"Average detection time {avg_time:.3f}s exceeds 0.1s limit"
    
    def test_consciousness_indicators_analysis(self, detector):
        """Test consciousness indicators analysis"""
        # Given: Text with known indicators
        arch_data = {
            'program': 'attention self query key value',
            'analysis': 'meta learning adaptive emergent',
            'motivation': 'recursive feedback loop'
        }
        
        # When: Indicators analyzed
        indicators = asyncio.run(detector._analyze_consciousness_indicators(arch_data))
        
        # Then: Indicators properly detected
        assert 'self_attention' in indicators
        assert 'meta_learning' in indicators
        assert 'recursive_processing' in indicators
        assert all(0.0 <= score <= 1.0 for score in indicators.values())
```

#### River Metaphor Tests

```python
# File: tests/unit/test_river_metaphor.py

import pytest
from extensions.context_engineering.core_implementation import ContextStreamManager, FlowState

class TestContextStreamManager:
    
    @pytest.fixture
    def stream_manager(self):
        from extensions.context_engineering.hybrid_database import create_hybrid_database
        db = create_hybrid_database()
        return ContextStreamManager(db)
    
    @pytest.mark.asyncio
    async def test_create_stream_valid_data(self, stream_manager):
        """Test stream creation with valid architecture data"""
        # Given: Valid architecture data
        arch_names = ['test_arch_1']
        arch_data = [{
            'name': 'test_arch_1',
            'program': 'class TestArch(nn.Module): pass',
            'result': {'test': 'acc=0.85'},
            'analysis': 'good performance'
        }]
        
        # When: Stream creation called
        stream = await stream_manager.create_stream_from_architectures(arch_names, arch_data)
        
        # Then: Returns valid ContextStream
        assert stream.id is not None
        assert stream.source_architecture_names == arch_names
        assert isinstance(stream.flow_state, FlowState)
        assert 0.0 <= stream.flow_velocity <= 1.0
        assert 0.0 <= stream.information_density <= 1.0
    
    def test_flow_state_determination(self, stream_manager):
        """Test flow state determination logic"""
        # Given: Architecture data with performance trends
        improving_data = [
            {'result': {'test': 'acc=0.70'}},
            {'result': {'test': 'acc=0.75'}},
            {'result': {'test': 'acc=0.80'}}
        ]
        
        # When: Flow state determined
        flow_state = stream_manager._determine_flow_state(improving_data)
        
        # Then: Returns appropriate flow state
        assert flow_state in [FlowState.FLOWING, FlowState.STABLE]
    
    def test_flow_velocity_calculation(self, stream_manager):
        """Test flow velocity calculation"""
        # Given: Architecture data with innovation indicators
        innovative_data = [{
            'motivation': 'novel breakthrough innovative approach',
            'analysis': 'new paradigm with improved performance'
        }]
        
        # When: Flow velocity calculated
        velocity = stream_manager._calculate_flow_velocity(innovative_data)
        
        # Then: Returns appropriate velocity
        assert 0.1 <= velocity <= 1.0
```

#### Attractor Basin Tests

```python
# File: tests/unit/test_attractor_basins.py

import pytest
from extensions.context_engineering.core_implementation import AttractorBasinManager

class TestAttractorBasinManager:
    
    @pytest.fixture
    def basin_manager(self):
        from extensions.context_engineering.hybrid_database import create_hybrid_database
        db = create_hybrid_database()
        return AttractorBasinManager(db)
    
    @pytest.mark.asyncio
    async def test_identify_basins_from_similar_architectures(self, basin_manager):
        """Test basin identification from similar architectures"""
        # Given: 5 similar architectures
        similar_archs = []
        for i in range(5):
            similar_archs.append({
                'name': f'similar_arch_{i}',
                'result': {'test': f'acc={0.82 + i*0.01}'},  # Similar performance
                'program': 'class SimilarArch(nn.Module): pass'
            })
        
        # When: Basin identification runs
        basins = await basin_manager.identify_basins_from_architectures(similar_archs)
        
        # Then: At least 1 basin identified
        assert len(basins) >= 1
        
        # And: Basin contains multiple architectures
        if basins:
            assert len(basins[0].contained_architectures) >= 2
            assert 0.0 <= basins[0].attraction_strength <= 1.0
            assert basins[0].radius > 0.0
    
    def test_performance_similarity_calculation(self, basin_manager):
        """Test performance similarity calculation"""
        # Given: Two similar performance signatures
        sig1 = {'accuracy': 0.85, 'loss': 0.15}
        sig2 = {'accuracy': 0.87, 'loss': 0.13}
        
        # When: Similarity calculated
        similarity = basin_manager._calculate_performance_similarity(sig1, sig2)
        
        # Then: High similarity score
        assert 0.7 <= similarity <= 1.0
    
    def test_stability_metrics_calculation(self, basin_manager):
        """Test stability metrics calculation"""
        # Given: Performance data with low variance
        performances = [
            {'accuracy': 0.85, 'loss': 0.15},
            {'accuracy': 0.84, 'loss': 0.16},
            {'accuracy': 0.86, 'loss': 0.14}
        ]
        
        # When: Stability metrics calculated
        metrics = basin_manager._calculate_stability_metrics(performances)
        
        # Then: High consistency score
        assert 'consistency' in metrics
        assert 'spread' in metrics
        assert 'robustness' in metrics
        assert 0.0 <= metrics['consistency'] <= 1.0
```

### 🔗 Integration Test Specifications

#### Database Integration Tests

```python
# File: tests/integration/test_database_integration.py

import pytest
import tempfile
import os
from extensions.context_engineering.hybrid_database import HybridContextDatabase

class TestDatabaseIntegration:
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = HybridContextDatabase(temp_dir)
            yield db
    
    def test_architecture_storage_and_retrieval(self, temp_db):
        """Test storing and retrieving architecture data"""
        # Given: Architecture data
        arch_data = {
            'name': 'test_arch',
            'program': 'class TestArch(nn.Module): pass',
            'result': {'test': 'acc=0.85'},
            'motivation': 'test architecture',
            'analysis': 'good performance'
        }
        
        # When: Architecture stored
        temp_db.store_architecture(arch_data, 'ACTIVE', 0.65)
        
        # Then: Architecture can be retrieved
        top_archs = temp_db.get_top_performing_architectures(limit=1)
        assert len(top_archs) == 1
        assert top_archs[0]['name'] == 'test_arch'
        assert top_archs[0]['consciousness_level'] == 'ACTIVE'
    
    def test_graph_relationship_creation(self, temp_db):
        """Test creating and querying graph relationships"""
        # Given: Two architectures
        arch1_data = {'name': 'arch1', 'result': {'test': 'acc=0.80'}}
        arch2_data = {'name': 'arch2', 'result': {'test': 'acc=0.85'}}
        
        temp_db.store_architecture(arch1_data, 'EMERGING', 0.35)
        temp_db.store_architecture(arch2_data, 'ACTIVE', 0.65)
        
        # When: Relationship created
        temp_db.create_architecture_relationship('arch1', 'arch2', 'evolved_to', 0.8)
        
        # Then: Relationship exists in graph
        neighbors = temp_db.graph_db.get_neighbors('arch1')
        assert 'arch2' in neighbors
    
    def test_consciousness_evolution_path_finding(self, temp_db):
        """Test finding evolution paths to higher consciousness"""
        # Given: Chain of architectures with increasing consciousness
        architectures = [
            ({'name': 'arch1'}, 'DORMANT', 0.1),
            ({'name': 'arch2'}, 'EMERGING', 0.3), 
            ({'name': 'arch3'}, 'ACTIVE', 0.6)
        ]
        
        for arch_data, level, score in architectures:
            temp_db.store_architecture(arch_data, level, score)
        
        # Create evolution chain
        temp_db.create_architecture_relationship('arch1', 'arch2', 'evolved_to', 0.7)
        temp_db.create_architecture_relationship('arch2', 'arch3', 'evolved_to', 0.8)
        
        # When: Evolution path searched
        path = temp_db.find_consciousness_evolution_path('arch1', 'ACTIVE')
        
        # Then: Path found
        assert len(path) >= 2
        assert 'arch1' in path
        assert any('arch' in node for node in path)
    
    def test_database_performance_requirements(self, temp_db):
        """Test database performance meets requirements"""
        import time
        
        # Given: 100 architectures
        for i in range(100):
            arch_data = {'name': f'arch_{i}', 'result': {'test': f'acc={0.7 + i*0.001}'}}
            temp_db.store_architecture(arch_data, 'ACTIVE', 0.5 + i*0.001)
        
        # When: Queries executed
        start_time = time.time()
        for _ in range(100):
            temp_db.get_top_performing_architectures(limit=10)
        end_time = time.time()
        
        # Then: Average query time < 10ms
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01, f"Average query time {avg_time:.3f}s exceeds 0.01s limit"
```

#### ASI-Arch Integration Tests

```python
# File: tests/integration/test_asi_arch_integration.py

import pytest
from extensions.context_engineering.asi_arch_bridge import enhance_evolution_context, get_consciousness_level

class TestASIArchIntegration:
    
    class MockDataElement:
        """Mock ASI-Arch DataElement for testing"""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to_dict(self):
            return self.__dict__
    
    @pytest.mark.asyncio
    async def test_context_enhancement_integration(self):
        """Test context enhancement with ASI-Arch data"""
        # Given: Original context and parent data
        original_context = "Create an improved attention mechanism"
        parent_data = self.MockDataElement(
            name='parent_attention',
            program='class ParentAttention(nn.Module): pass',
            result={'test': 'acc=0.80'},
            motivation='baseline attention',
            analysis='good baseline performance'
        )
        
        # When: Context enhancement called
        enhanced_context = await enhance_evolution_context(original_context, parent_data)
        
        # Then: Context is enhanced with insights
        assert len(enhanced_context) > len(original_context)
        assert 'CONTEXT ENGINEERING INSIGHTS' in enhanced_context
        assert 'River Metaphor Analysis' in enhanced_context
        assert 'Consciousness Detection' in enhanced_context
    
    @pytest.mark.asyncio
    async def test_consciousness_level_detection_integration(self):
        """Test consciousness level detection with ASI-Arch data"""
        # Given: Architecture with self-awareness indicators
        arch_data = self.MockDataElement(
            name='self_aware_transformer',
            program='class SelfAwareTransformer with meta_learning and self_attention',
            result={'test': 'acc=0.90'},
            analysis='emergent self-awareness patterns observed'
        )
        
        # When: Consciousness detection called
        level_name, score = await get_consciousness_level(arch_data)
        
        # Then: Appropriate consciousness level detected
        assert level_name in ['EMERGING', 'ACTIVE', 'SELF_AWARE', 'META_AWARE']
        assert 0.0 <= score <= 1.0
        assert score > 0.2  # Should be above DORMANT for this architecture
    
    def test_integration_graceful_degradation(self):
        """Test that integration fails gracefully"""
        # Given: Invalid data
        invalid_data = None
        
        # When: Integration functions called with invalid data
        # Then: Should not raise exceptions (graceful degradation)
        import asyncio
        
        async def test_graceful():
            try:
                result = await enhance_evolution_context("test", invalid_data)
                assert result == "test"  # Should return original on failure
            except Exception as e:
                pytest.fail(f"Integration should handle invalid data gracefully: {e}")
        
        asyncio.run(test_graceful())
```

### 🌐 End-to-End Test Specifications

#### Complete Pipeline Tests

```python
# File: tests/e2e/test_complete_pipeline.py

import pytest
import asyncio
import time
from extensions.context_engineering.live_integration import ContextEngineeringLiveService

class TestCompletePipeline:
    
    @pytest.fixture
    async def live_service(self):
        """Create live service for testing"""
        service = ContextEngineeringLiveService(
            dashboard_port=8082,  # Use different port for testing
            enable_dashboard=False  # Disable dashboard for automated tests
        )
        service.start(start_dashboard=False)
        yield service
        service.stop()
    
    @pytest.mark.asyncio
    async def test_complete_architecture_experiment_processing(self, live_service):
        """Test complete architecture experiment processing"""
        # Given: Architecture experiment data
        parent_data = {
            'name': 'parent_arch',
            'program': 'class ParentArch(nn.Module): pass',
            'result': {'test': 'acc=0.80'},
            'motivation': 'baseline implementation',
            'analysis': 'good baseline performance'
        }
        
        result_data = {
            'name': 'evolved_arch',
            'program': 'class EvolvedArch with self_attention(nn.Module): pass',
            'result': {'test': 'acc=0.85'},
            'motivation': 'add self-attention mechanism',
            'analysis': 'improved performance with attention patterns'
        }
        
        # When: Complete experiment processing
        analysis = await live_service.process_architecture_experiment(
            context="Test evolution context",
            parent_data=parent_data,
            result_data=result_data
        )
        
        # Then: All components processed successfully
        assert analysis['context_engineering_active'] is True
        assert 'enhanced_context' in analysis
        assert 'consciousness_analysis' in analysis
        assert len(analysis['enhanced_context']) > len("Test evolution context")
        
        # And: Consciousness analysis completed
        consciousness_info = analysis['consciousness_analysis']
        assert 'level' in consciousness_info
        assert 'score' in consciousness_info
        assert consciousness_info['level'] in ['DORMANT', 'EMERGING', 'ACTIVE', 'SELF_AWARE', 'META_AWARE']
    
    @pytest.mark.asyncio
    async def test_multiple_experiments_basin_formation(self, live_service):
        """Test that multiple experiments lead to basin formation"""
        # Given: Multiple similar architectures
        experiments = []
        for i in range(6):  # Need 5+ for basin formation
            parent = {'name': f'parent_{i}', 'result': {'test': f'acc={0.78 + i*0.005}'}}
            result = {'name': f'evolved_{i}', 'result': {'test': f'acc={0.82 + i*0.005}'}}
            experiments.append((parent, result))
        
        # When: All experiments processed
        for parent_data, result_data in experiments:
            await live_service.process_architecture_experiment(
                f"Context {parent_data['name']}",
                parent_data=parent_data,
                result_data=result_data
            )
        
        # Then: Attractor basin should be identified
        # (Basin analysis happens every 5 experiments)
        # This is tested indirectly through the consciousness evolution tracking
        assert len(live_service.consciousness_evolution) == 6
    
    def test_system_performance_under_load(self, live_service):
        """Test system performance with multiple concurrent experiments"""
        async def run_load_test():
            # Given: 20 concurrent experiments
            tasks = []
            for i in range(20):
                task = live_service.process_architecture_experiment(
                    f"Load test context {i}",
                    parent_data={'name': f'load_parent_{i}', 'result': {'test': 'acc=0.80'}},
                    result_data={'name': f'load_result_{i}', 'result': {'test': 'acc=0.82'}}
                )
                tasks.append(task)
            
            # When: All experiments run concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Then: All complete successfully within reasonable time
            assert len(results) == 20
            assert all(r['context_engineering_active'] for r in results)
            
            # And: Total time reasonable (should be much faster than sequential)
            total_time = end_time - start_time
            assert total_time < 5.0, f"Load test took {total_time:.2f}s, should be < 5s"
        
        asyncio.run(run_load_test())
```

## 🚀 Test Execution Strategy

### 🏃‍♂️ Test Execution Pipeline

```bash
# Phase 1: Fast Tests (Unit Tests)
pytest tests/unit/ -v --cov=extensions.context_engineering --cov-report=term-missing

# Phase 2: Integration Tests  
pytest tests/integration/ -v --timeout=30

# Phase 3: End-to-End Tests
pytest tests/e2e/ -v --timeout=60

# Phase 4: Performance Tests
pytest tests/performance/ -v --benchmark-only

# Phase 5: Full Test Suite
pytest tests/ -v --cov=extensions.context_engineering --cov-report=html
```

### 📊 Test Coverage Requirements

#### Coverage Targets by Component
- **Core Implementation**: 95% coverage
- **ASI-Arch Bridge**: 85% coverage  
- **Hybrid Database**: 95% coverage
- **Dashboard**: 70% coverage (UI components)
- **Integration**: 90% coverage

#### Coverage Exclusions
- Third-party library integrations
- Error handling for external service failures
- Development-only debug code
- Platform-specific code paths

## ⚡ Performance Test Specifications

### 🏋️‍♂️ Load Testing Requirements

#### Consciousness Detection Load Test
```python
def test_consciousness_detection_load():
    """Test consciousness detection under load"""
    # Target: 1000 detections in <10 seconds
    # Memory: Stable under 100MB
    # Accuracy: Consistent results across all detections
```

#### Database Performance Test
```python
def test_database_performance():
    """Test database performance requirements"""
    # Target: 10,000 queries in <1 second
    # Memory: Linear growth with data size
    # Consistency: No data corruption under load
```

#### Dashboard Responsiveness Test
```python
def test_dashboard_responsiveness():
    """Test dashboard performance requirements"""
    # Target: Page load <3 seconds
    # Updates: Data refresh <5 seconds
    # Concurrent: 10 users without degradation
```

## 🔧 Test Infrastructure

### 🛠️ Test Setup and Configuration

#### Test Environment Configuration
```python
# conftest.py - Pytest configuration
import pytest
import asyncio
import tempfile
import os

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_database():
    """Provide temporary database for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_architecture_data():
    """Provide mock architecture data for testing"""
    return {
        'name': 'test_architecture',
        'program': 'class TestArch(nn.Module): pass',
        'result': {'test': 'acc=0.85'},
        'motivation': 'test motivation',
        'analysis': 'test analysis'
    }
```

#### Continuous Integration Configuration
```yaml
# .github/workflows/test.yml
name: Context Engineering Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov pytest-timeout
    
    - name: Run unit tests
      run: pytest tests/unit/ -v --cov=extensions.context_engineering
    
    - name: Run integration tests  
      run: pytest tests/integration/ -v --timeout=30
    
    - name: Run E2E tests
      run: pytest tests/e2e/ -v --timeout=60
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## ✅ Test Acceptance Criteria

### 🎯 Test Suite Acceptance

#### Functional Test Acceptance
- [ ] All unit tests pass (>95% coverage for critical components)
- [ ] All integration tests pass (>90% coverage for interfaces)
- [ ] All E2E tests pass (critical user journeys covered)
- [ ] Performance tests meet benchmarks
- [ ] Load tests demonstrate system stability

#### Quality Test Acceptance  
- [ ] Test suite runs in <5 minutes total
- [ ] Tests are deterministic (no flaky tests)
- [ ] Test code follows same quality standards as production code
- [ ] Test documentation is complete and accurate
- [ ] CI/CD pipeline executes all tests successfully

#### Coverage Test Acceptance
- [ ] Overall test coverage >85%
- [ ] Critical path coverage >95%
- [ ] No untested public interfaces
- [ ] Edge cases and error conditions tested
- [ ] Performance characteristics validated

---

**Test Specification Status**: READY FOR IMPLEMENTATION  
**Next Review Date**: 2025-10-01  
**Test Owner**: ASI-Arch Context Engineering Team
