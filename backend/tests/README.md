# Flux Backend Test Suite

Comprehensive test suite to prevent startup failures and validate system integration.

## Test Files

### 1. `test_startup.py` - Backend Startup Tests
**Purpose**: Verify all critical imports, initialization, and configuration work correctly.

**Test Coverage**:
- ✅ FastAPI and middleware imports
- ✅ Daedalus gateway initialization
- ✅ DocumentProcessingGraph creation
- ✅ ConsciousnessDocumentProcessor availability
- ✅ API route registration
- ✅ PyPDF2 dependency
- ✅ Port availability checking
- ✅ Flux configuration loading
- ✅ Environment variable handling
- ✅ Redis connection graceful failure

**Why These Tests Matter**:
These tests prevent the import errors and initialization failures we encountered:
- ModuleNotFoundError for missing dependencies
- Import path issues (relative vs absolute)
- Missing PyPDF2 causing PDF processing crashes
- Port conflicts blocking backend startup

**Run Command**:
```bash
pytest tests/test_startup.py -v
```

### 2. `test_consciousness_pipeline.py` - Consciousness Processing Tests
**Purpose**: Verify attractor basin creation, ThoughtSeed generation, and concept extraction.

**Test Coverage**:
- ✅ AttractorBasinManager initialization
- ✅ Basin influence types (REINFORCEMENT, COMPETITION, SYNTHESIS, EMERGENCE)
- ✅ ThoughtSeed model structure
- ✅ Concept extraction from text
- ✅ Consciousness processing through basin dynamics
- ✅ LangGraph workflow node structure
- ✅ Daedalus integration with consciousness pipeline
- ✅ PDF text extraction
- ✅ Error handling for invalid documents
- ✅ Graceful Redis unavailability handling

**Why These Tests Matter**:
These tests verify the core consciousness processing flow documented in `DAEDALUS_INFORMATION_FLOW.md`:
- Upload → Daedalus → LangGraph → Consciousness → AttractorBasins → Neo4j
- Ensures concepts are properly converted to basins
- Validates the 4 influence types work correctly
- Confirms ThoughtSeed integration

**Run Command**:
```bash
pytest tests/test_consciousness_pipeline.py -v
```

### 3. `test_document_upload.py` - Document Upload Integration Tests
**Purpose**: Test complete document upload flow through Daedalus gateway.

**Test Coverage**:
- ✅ `/api/v1/documents` endpoint registration
- ✅ Daedalus processes text files
- ✅ Daedalus handles PDF files
- ✅ Document metadata (tags, quality_threshold, max_iterations)
- ✅ Complete workflow integration
- ✅ Response structure validation
- ✅ Error handling (None data, empty files, large files, Unicode)
- ✅ All API routers import correctly
- ✅ End-to-end text processing
- ✅ Cognition summary functionality

**Why These Tests Matter**:
These tests ensure the Markov blanket architecture is preserved:
- All uploads MUST go through Daedalus gateway (no bypassing)
- Validates complete information flow from upload to processing
- Ensures API endpoints are properly connected
- Tests graceful error handling for edge cases

**Run Command**:
```bash
pytest tests/test_document_upload.py -v
```

### 4. `test_database_connections.py` - Database Connection Tests
**Purpose**: Test Neo4j, Redis connectivity with graceful degradation.

**Test Coverage**:
- ✅ Neo4j driver import and connection
- ✅ Neo4j graceful failure handling
- ✅ Redis connection with timeout
- ✅ Redis operations (set/get, hash, key patterns)
- ✅ AttractorBasinManager handles Redis unavailable
- ✅ ConsciousnessProcessor works without Redis
- ✅ Database health check endpoint
- ✅ Environment configuration (.env loading)
- ✅ System runs without Neo4j
- ✅ System runs without Redis
- ✅ App starts without any database connections

**Why These Tests Matter**:
These tests address the Neo4j authentication failures we encountered:
- Ensures system degrades gracefully when databases unavailable
- Tests both connection success and failure paths
- Validates Redis basin storage operations
- Confirms Neo4j query execution when available

**Run Command**:
```bash
pytest tests/test_database_connections.py -v
```

## Running All Tests

### Quick Run (All Tests)
```bash
cd /Volumes/Asylum/dev/Dionysus-2.0/backend
chmod +x run_tests.sh
./run_tests.sh
```

### Individual Test Suites
```bash
# Activate virtual environment
source flux-backend-env/bin/activate
export PYTHONPATH=/Volumes/Asylum/dev/Dionysus-2.0/backend:$PYTHONPATH

# Run specific test file
pytest tests/test_startup.py -v
pytest tests/test_consciousness_pipeline.py -v
pytest tests/test_document_upload.py -v
pytest tests/test_database_connections.py -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
```

### Run Specific Test Class or Method
```bash
# Run single test class
pytest tests/test_startup.py::TestCriticalImports -v

# Run single test method
pytest tests/test_startup.py::TestCriticalImports::test_fastapi_imports -v
```

## Test Results Interpretation

### ✅ PASSED
Test succeeded - functionality works as expected.

### ⚠️ SKIPPED
Test skipped due to missing optional dependency (e.g., Redis not running).
- System should still work with degraded functionality
- Not a failure - indicates graceful degradation working

### ❌ FAILED
Test failed - indicates a real problem that needs fixing.
- Check error message for root cause
- Review recent code changes
- Ensure dependencies installed

## Continuous Integration

These tests should be run:
1. **Before starting backend** - Verify environment is correct
2. **After code changes** - Ensure nothing broke
3. **Before commits** - Maintain code quality
4. **In CI/CD pipeline** - Automated validation

## Troubleshooting

### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/Volumes/Asylum/dev/Dionysus-2.0/backend:$PYTHONPATH

# Install missing dependencies
pip install -r requirements.txt
```

### Database Connection Failures
```bash
# Start Redis
docker start redis-consciousness
# OR
docker run -d --name redis-consciousness -p 6379:6379 redis:7-alpine

# Start Neo4j
docker start neo4j-flux
# OR check Neo4j credentials in .env
```

### PyPDF2 Missing
```bash
pip install PyPDF2 pypdf
```

## Test Coverage Goals

- **Startup Tests**: 100% of critical imports and initialization paths
- **Consciousness Tests**: 80%+ of consciousness processing logic
- **Upload Tests**: 100% of API endpoints and error paths
- **Database Tests**: 100% of connection handling (success + failure)

## Adding New Tests

When adding new functionality:
1. Write tests BEFORE implementing (TDD)
2. Test both success and failure paths
3. Test graceful degradation
4. Update this README with new test descriptions

## Related Documentation

- [Backend README](../README.md)
- [DAEDALUS_INFORMATION_FLOW.md](../docs/DAEDALUS_INFORMATION_FLOW.md)
- [Spec 021: Daedalus Gateway](../../specs/021-remove-all-that/)
