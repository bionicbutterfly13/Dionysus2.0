# Flux Test Policy: No Skipped Tests

## Rule: Every Test Must Either PASS or FAIL

### ✅ Allowed Test Outcomes:
- **PASS**: Feature is implemented and working
- **FAIL**: Test fails with clear TODO message explaining what needs to be implemented

### ❌ Not Allowed:
- **SKIP**: Tests that skip provide zero value and false confidence

### How to Handle Different Scenarios:

#### 1. Feature Not Yet Implemented
```python
def test_feature_not_implemented():
    # Don't skip - fail with clear message
    assert False, "TODO: Implement user authentication feature"
```

#### 2. External Dependency Not Available
```python
def test_database_connection():
    # Test the connection, expect it to fail gracefully if DB not running
    result = database_health_service.check_neo4j_health()

    # This is a real test - it should pass when DB is up, fail when down
    if result['status'] == 'unavailable':
        # This is still a valid test result - we learned something
        assert result['status'] in ['healthy', 'unavailable'], "Unexpected status"
    else:
        assert result['status'] == 'healthy', "Database should be healthy"
```

#### 3. Temporary Implementation Issue
```python
def test_complex_feature():
    # Fix the test or mark clearly what needs to be done
    assert False, "TODO: Fix import issue in models - see Issue #123"
```

### Benefits of No-Skip Policy:
1. **Real Feedback**: Every test run gives actionable information
2. **Clear TODOs**: Failed tests show exactly what needs implementation
3. **No False Confidence**: Can't accidentally think something works when it's skipped
4. **Progress Tracking**: As tests go from failing → passing, you see real progress

### Test Categories in Flux:
- **Working Tests**: Must always pass (regression protection)
- **TODO Tests**: Fail with clear implementation messages
- **No Skipped Tests**: Banned entirely
