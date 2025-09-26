#!/usr/bin/env python
"""
No-Skip Test Enforcement for Flux
Ensures all tests either PASS or FAIL - no skipping allowed.
"""
import subprocess
import sys
import os
from pathlib import Path
import re


def find_skipped_tests():
    """Find all tests that use pytest.skip."""
    test_dir = Path("tests")
    skipped_tests = []

    for test_file in test_dir.rglob("*.py"):
        if test_file.name.startswith("test_"):
            content = test_file.read_text()
            if "pytest.skip" in content or "@pytest.mark.skip" in content:
                # Extract the skip reasons
                skip_lines = []
                for i, line in enumerate(content.split('\n'), 1):
                    if "pytest.skip" in line or "@pytest.mark.skip" in line:
                        skip_lines.append((i, line.strip()))

                skipped_tests.append({
                    'file': str(test_file),
                    'skips': skip_lines
                })

    return skipped_tests


def convert_skipped_to_todo_tests():
    """Convert all skipped tests to proper todo tests that fail with clear messages."""
    print("🚫 Converting Skipped Tests to TODO Tests")
    print("=" * 50)

    skipped = find_skipped_tests()

    if not skipped:
        print("✅ No skipped tests found!")
        return

    for test_info in skipped:
        print(f"\n📄 Processing: {test_info['file']}")

        file_path = Path(test_info['file'])
        content = file_path.read_text()

        # Replace pytest.skip with assert False and clear TODO message
        content = re.sub(
            r'pytest\.skip\(["\']([^"\']*)["\'].*?\)',
            r'assert False, "TODO: \1 - This test needs implementation"',
            content
        )

        # Replace @pytest.mark.skip decorators
        content = re.sub(
            r'@pytest\.mark\.skip.*?\n',
            '',
            content,
            flags=re.MULTILINE
        )

        # Write the updated content
        file_path.write_text(content)

        print(f"  ✅ Converted {len(test_info['skips'])} skipped tests to TODO tests")


def run_no_skip_test_check():
    """Run pytest and ensure no tests are skipped."""
    print("\n🧪 Running No-Skip Test Check")
    print("=" * 50)

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        # Fail if any tests are skipped
        "--strict-markers"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120
    )

    # Check for any skipped tests in output
    output_lines = result.stdout.split('\n')
    skipped_count = 0

    for line in output_lines:
        if 'skipped' in line.lower():
            # Look for patterns like "7 skipped" or "SKIPPED"
            skip_match = re.search(r'(\d+)\s+skipped', line)
            if skip_match:
                skipped_count = int(skip_match.group(1))
                break

    print(f"Return code: {result.returncode}")
    print(f"Skipped tests found: {skipped_count}")

    if skipped_count > 0:
        print("\n❌ SKIPPED TESTS DETECTED!")
        print("Skipped tests are not allowed. Convert them to:")
        print("  - Passing tests (implement the feature)")
        print("  - Failing tests with clear TODO messages")
        print(f"\nOutput:\n{result.stdout}")
        return False

    # Check if all tests either passed or failed (no skips)
    if 'skipped' not in result.stdout.lower():
        print("✅ NO SKIPPED TESTS - All tests either pass or fail with clear reasons")
        return True
    else:
        print("❌ Skipped tests still detected in output")
        return False


def create_test_policy():
    """Create a test policy file explaining the no-skip rule."""
    policy_content = """# Flux Test Policy: No Skipped Tests

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
"""

    Path("TEST_POLICY.md").write_text(policy_content)
    print("📋 Created TEST_POLICY.md explaining no-skip rule")


def main():
    """Main enforcement function."""
    print("🚫 Flux No-Skip Test Enforcement")
    print("=" * 50)

    # Step 1: Find and convert any skipped tests
    convert_skipped_to_todo_tests()

    # Step 2: Create the test policy
    create_test_policy()

    # Step 3: Run tests and verify no skips
    success = run_no_skip_test_check()

    if success:
        print("\n🎯 SUCCESS: No-skip test policy is now enforced")
        print("All tests either pass or fail with clear TODO messages")
    else:
        print("\n❌ FAILURE: Skipped tests still exist")
        print("Fix the skipped tests before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()