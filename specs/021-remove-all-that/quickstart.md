# Quickstart: Daedalus Perceptual Information Gateway

## Overview
The Daedalus class serves as the simplified perceptual information gateway, receiving uploaded data with a single, clean interface.

## Prerequisites
- Python 3.11+
- flux-backend-env virtual environment activated
- Backend services running

## Quick Test

### 1. Verify Daedalus Simplicity
```bash
# Check that Daedalus has minimal code
wc -l backend/src/services/daedalus.py
# Expected: ~30-40 lines total

# Inspect the class
cat backend/src/services/daedalus.py
```

**Expected**: Single method `receive_perceptual_information()`, no other functionality

### 2. Run Contract Tests
```bash
cd /Volumes/Asylum/dev/Dionysus-2.0
source backend/flux-backend-env/bin/activate

# Run Daedalus contract tests
pytest backend/tests/contract/test_daedalus_spec_021.py -v

# Expected: 11/11 tests passing
```

### 3. Run Integration Tests
```bash
# Run integration tests
pytest backend/tests/integration/test_daedalus_integration.py -v

# Expected: 4/4 tests passing
```

### 4. Test End-to-End Upload Flow
```bash
# Start backend server
cd backend
uvicorn main:app --reload

# In another terminal, test upload
curl -X POST http://localhost:8000/api/v1/documents \
  -F "files=@test_document.pdf" \
  -F "metadata={\"batch_name\":\"test\"}"

# Expected: 200 OK with batch_id returned
```

## User Flow Validation

### Primary User Story
1. ✅ File uploaded to system
2. ✅ Daedalus receives perceptual information
3. ✅ Information forwarded to LangGraph agents
4. ✅ No unnecessary complexity in Daedalus

### Acceptance Criteria
- [x] Daedalus receives upload data
- [x] Daedalus interfaces with LangGraph
- [x] Daedalus contains only essential functionality
- [x] All tests passing (15/15)

## Expected Results

### Code Metrics
- **Daedalus LOC**: <50 lines
- **Methods**: 1 (`receive_perceptual_information`)
- **Dependencies**: Minimal (typing, Optional, Dict, Any, BinaryIO)

### Test Results
```
backend/tests/contract/test_daedalus_spec_021.py ........... 11 passed
backend/tests/integration/test_daedalus_integration.py .... 4 passed

Total: 15/15 PASSING ✅
```

### Constitution Compliance
- ✅ TDD followed (tests before implementation)
- ✅ Single responsibility maintained
- ✅ No unnecessary complexity
- ✅ All constitutional requirements met

## Troubleshooting

### If tests fail
```bash
# Check Python path
echo $PYTHONPATH

# Ensure conftest.py exists
ls backend/tests/conftest.py

# Re-run with verbose output
pytest backend/tests/contract/test_daedalus_spec_021.py -vv
```

### If Daedalus has extra methods
The cleanup was incomplete - review the simplified implementation:
```python
class Daedalus:
    def __init__(self):
        self._is_gateway = True

    def receive_perceptual_information(self, data: Optional[BinaryIO]) -> Dict[str, Any]:
        """Single responsibility: receive perceptual information."""
        # Implementation here
```

## Success Criteria

✅ **Simplicity**: Daedalus has only `receive_perceptual_information()` method
✅ **Tests**: All 15 tests passing
✅ **Integration**: Works with upload endpoint
✅ **Constitution**: TDD and single responsibility followed
✅ **Archive**: Removed functionality preserved separately

## Next Steps

This feature is **COMPLETE**. Next possible actions:
1. Implement additional upload types (if needed)
2. Enhance LangGraph agent creation
3. Add monitoring/observability to Daedalus gateway
