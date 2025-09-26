# Strict Test-Driven Development (TDD) Rules for Flux

## Core TDD Cycle: RED → GREEN → REFACTOR

### Phase 1: RED (Write a Failing Test)
- **Never write implementation code unless a test fails**
- For every new feature or bugfix, start by creating (or expanding) a test in the corresponding `test_*.py` file describing the behavior
- Test should fail for the right reason (not due to syntax errors)
- Test should be minimal and focused on one specific behavior
- Use descriptive test names that explain the expected behavior

### Phase 2: GREEN (Make the Test Pass)
- Use only enough production code to pass the current failing test
- Don't write more code than necessary
- It's okay to write "stupid" code that just makes the test pass
- Focus on making it work, not making it perfect

### Phase 3: REFACTOR (Improve the Code)
- Refactor at each stage, but only when tests are green
- Improve code structure, remove duplication, enhance readability
- Run tests after each refactor to ensure nothing breaks
- Both test and production code can be refactored

## Execution Rules

### Testing Requirements
- Run the entire test suite for each iteration; do not break existing tests
- Use pytest conventions and standard Python naming
- Follow AAA pattern: Arrange, Act, Assert
- One assertion per test when possible
- Use descriptive assertion messages

### Documentation Requirements
- Provide git diffs and brief rationales with each step
- Be explicit, granular, and do not skip steps
- After each test result, output:
  - Which test failed (or passed)
  - What minimal code you'll add/change
  - A unified diff of the code changes
  - Results of re-running all tests

### Code Quality Standards
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write self-documenting code with clear variable names
- Maintain separation of concerns
- Keep functions small and focused

## Test Categories

### Unit Tests
- Test individual functions/methods in isolation
- Mock external dependencies
- Fast execution (< 1ms per test)
- No network calls, no database connections, no file I/O

### Integration Tests
- Test interaction between components
- Can use real dependencies in controlled environment
- Slower execution acceptable
- Test realistic scenarios

### End-to-End Tests
- Test complete user workflows
- Use real services when possible
- Slowest execution
- Test critical business paths

## Pytest Best Practices

### Test Structure
```python
def test_specific_behavior_under_specific_condition():
    # Arrange
    setup_data = create_test_data()

    # Act
    result = function_under_test(setup_data)

    # Assert
    assert result.expected_property == expected_value, "Clear failure message"
```

### Fixtures and Parametrization
- Use fixtures for common test setup
- Parametrize tests to cover multiple scenarios
- Use appropriate fixture scopes (function, class, module, session)

### Test Organization
- Group related tests in classes
- Use descriptive class names (TestUserAuthentication)
- Organize tests in logical packages mirroring production code

## Anti-Patterns to Avoid

### Testing Anti-Patterns
- Don't write tests after implementation (breaks TDD)
- Don't test implementation details (test behavior)
- Don't create overly complex test setups
- Don't ignore failing tests
- Don't skip the refactor phase

### Implementation Anti-Patterns
- Don't write speculative code "just in case"
- Don't optimize prematurely
- Don't add features not driven by tests
- Don't break existing tests

## Command Reference

### Run Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest test_specific_module.py

# Run with coverage
pytest --cov=src

# Run in verbose mode
pytest -v

# Run only failed tests from last run
pytest --lf
```

### Pre-Test Checklist
1. [ ] Read and refresh these TDD rules
2. [ ] Understand the feature requirement
3. [ ] Write the smallest failing test
4. [ ] Verify test fails for the right reason
5. [ ] Write minimal code to pass
6. [ ] Run all tests to ensure no regressions
7. [ ] Refactor if needed while keeping tests green
8. [ ] Commit changes with descriptive message

## Success Criteria
- All tests pass
- Code coverage is comprehensive
- No duplication in production code
- Tests are readable and maintainable
- Implementation is simple and focused

## Completion Signal
When all features are implemented and all tests pass, state: **"TDD complete for this feature."**

---

*These rules must be refreshed before each TDD session to maintain context and ensure strict adherence to the methodology.*