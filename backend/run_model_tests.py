#!/usr/bin/env python3
"""
Model Test Runner for Dionysus 2.0
TDD compliance validation script

Runs all model tests and provides comprehensive reporting.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add backend src to path
backend_dir = Path(__file__).parent
src_dir = backend_dir / "src"
sys.path.insert(0, str(src_dir))

def run_tests():
    """Run all model tests with pytest"""

    print("🧪 Running Dionysus 2.0 Model Test Suite")
    print("=" * 50)

    test_dir = backend_dir / "tests" / "models"

    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False

    # Run pytest with detailed output
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",                    # Verbose output
        "--tb=short",           # Short traceback format
        "--color=yes",          # Colored output
        "--durations=10",       # Show 10 slowest tests
        "--capture=no",         # Don't capture print statements
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=backend_dir, check=False)

        print("-" * 50)
        if result.returncode == 0:
            print("✅ All model tests passed!")
            print("🎉 TDD compliance validated successfully")
        else:
            print(f"❌ Tests failed with exit code: {result.returncode}")
            print("🔧 Review test output above for issues")

        return result.returncode == 0

    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest:")
        print("   pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main test runner function"""

    print("Dionysus 2.0 - Model Test Suite")
    print("TDD Compliance Validation")
    print("=" * 50)

    # Check if we're in the right directory
    if not (backend_dir / "src" / "models").exists():
        print("❌ Models directory not found. Are you in the backend directory?")
        sys.exit(1)

    # Run the tests
    success = run_tests()

    if success:
        print("\n🎯 Next Steps:")
        print("   - All models are tested and working")
        print("   - Ready to implement services layer (T022-T048)")
        print("   - TDD compliance maintained ✅")
        sys.exit(0)
    else:
        print("\n🔧 Action Required:")
        print("   - Fix failing tests before proceeding")
        print("   - Ensure all imports are working correctly")
        print("   - Verify model implementations match test expectations")
        sys.exit(1)


if __name__ == "__main__":
    main()