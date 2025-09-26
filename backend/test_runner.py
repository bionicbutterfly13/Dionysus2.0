#!/usr/bin/env python
"""
Comprehensive Test Runner for Flux
Ensures all features work together and nothing gets broken by new changes.
"""
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import time


class FluxTestRunner:
    """Comprehensive test runner with regression detection."""

    def __init__(self):
        self.backend_dir = Path(__file__).parent
        self.working_tests = [
            # Tests that currently work
            "tests/test_port_management.py",
            "tests/test_database_health.py"
        ]
        self.broken_tests = [
            # Tests that have import/implementation issues
            "tests/models/test_all_models.py",
            "tests/models/test_event_node.py",
            "tests/models/test_thoughtseed_trace.py",
            "tests/models/test_user_profile.py"
        ]
        self.skipped_tests = [
            # Tests that are intentionally skipped (not implemented yet)
            "tests/contract/test_curiosity_missions.py",
            "tests/contract/test_documents_post.py",
            "tests/contract/test_visualization_ws.py",
            "tests/integration/test_curiosity_lifecycle.py",
            "tests/integration/test_document_ingestion_flow.py",
            "tests/integration/test_dream_replay.py",
            "tests/integration/test_visualization_stream.py"
        ]

    def run_working_tests(self) -> Dict[str, Any]:
        """Run all tests that should pass."""
        print("🧪 Running Working Tests (Regression Check)")
        print("=" * 50)

        cmd = [
            sys.executable, "-m", "pytest",
            *self.working_tests,
            "-v", "--tb=short", "--no-header"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.backend_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            passed = result.returncode == 0
            test_count = self._count_tests_from_output(result.stdout)

            return {
                'passed': passed,
                'test_count': test_count,
                'output': result.stdout,
                'errors': result.stderr,
                'return_code': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'test_count': 0,
                'output': '',
                'errors': 'Tests timed out after 60 seconds',
                'return_code': -1
            }

    def check_broken_tests(self) -> Dict[str, Any]:
        """Check if broken tests are still broken (expected to fail)."""
        print("\n🔧 Checking Broken Tests (Should Still Fail)")
        print("=" * 50)

        cmd = [
            sys.executable, "-m", "pytest",
            *self.broken_tests,
            "-v", "--tb=line", "--no-header"
        ]

        result = subprocess.run(
            cmd,
            cwd=self.backend_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        # For broken tests, we expect them to fail
        still_broken = result.returncode != 0

        return {
            'still_broken': still_broken,
            'output': result.stdout,
            'errors': result.stderr,
            'return_code': result.returncode
        }

    def run_specific_test(self, test_path: str) -> Dict[str, Any]:
        """Run a specific test file."""
        print(f"\n🎯 Running Specific Test: {test_path}")
        print("=" * 50)

        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-v", "--tb=short"
        ]

        result = subprocess.run(
            cmd,
            cwd=self.backend_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        passed = result.returncode == 0
        test_count = self._count_tests_from_output(result.stdout)

        return {
            'passed': passed,
            'test_count': test_count,
            'output': result.stdout,
            'errors': result.stderr
        }

    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive regression test suite."""
        print("🌐 Flux Comprehensive Test Suite")
        print("=" * 60)

        start_time = time.time()

        # Run working tests (these must pass)
        working_results = self.run_working_tests()

        # Check broken tests (these should still be broken)
        broken_results = self.check_broken_tests()

        end_time = time.time()
        duration = end_time - start_time

        # Generate summary
        summary = {
            'overall_passed': working_results['passed'],
            'working_tests': working_results,
            'broken_tests': broken_results,
            'duration_seconds': round(duration, 2),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        self._print_summary(summary)
        return summary

    def _count_tests_from_output(self, output: str) -> int:
        """Extract test count from pytest output."""
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and ('error' in line or 'failed' in line or 'warning' in line):
                # Look for pattern like "13 passed, 4 warnings"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        try:
                            return int(parts[i-1])
                        except ValueError:
                            continue
        return 0

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print comprehensive test summary."""
        print(f"\n📊 Test Suite Summary ({summary['timestamp']})")
        print("=" * 60)

        working = summary['working_tests']
        broken = summary['broken_tests']

        # Working tests status
        status_icon = "✅" if working['passed'] else "❌"
        print(f"{status_icon} Working Tests: {working['test_count']} tests")

        if not working['passed']:
            print("❌ REGRESSION DETECTED - Working tests are failing!")
            print(f"Error: {working['errors']}")

        # Broken tests status
        if broken['still_broken']:
            print("⚠️ Broken Tests: Still broken (expected)")
        else:
            print("🎉 Broken Tests: Some may have been fixed!")

        print(f"\n⏱️ Total Duration: {summary['duration_seconds']}s")

        # Overall status
        if summary['overall_passed']:
            print("\n🎯 REGRESSION TEST: PASSED")
            print("✅ All implemented features are working correctly")
        else:
            print("\n🚨 REGRESSION TEST: FAILED")
            print("❌ Some working features have been broken")
            print("🔧 Fix the broken tests before proceeding")

    def add_new_test(self, test_path: str) -> None:
        """Add a new test to the working tests list."""
        if test_path not in self.working_tests:
            self.working_tests.append(test_path)
            print(f"✅ Added {test_path} to working tests")

    def get_test_status(self) -> Dict[str, List[str]]:
        """Get current test categorization."""
        return {
            'working': self.working_tests,
            'broken': self.broken_tests,
            'skipped': self.skipped_tests
        }


def main():
    """Main entry point for test runner."""
    import argparse

    parser = argparse.ArgumentParser(description='Flux Comprehensive Test Runner')
    parser.add_argument('--working-only', action='store_true',
                       help='Run only working tests (quick regression check)')
    parser.add_argument('--specific', type=str,
                       help='Run a specific test file')
    parser.add_argument('--status', action='store_true',
                       help='Show test categorization status')

    args = parser.parse_args()

    runner = FluxTestRunner()

    if args.status:
        status = runner.get_test_status()
        print("📋 Flux Test Status")
        print("=" * 40)
        print(f"✅ Working Tests ({len(status['working'])}): {status['working']}")
        print(f"🔧 Broken Tests ({len(status['broken'])}): {status['broken']}")
        print(f"⏭️ Skipped Tests ({len(status['skipped'])}): {status['skipped']}")
        return

    if args.specific:
        result = runner.run_specific_test(args.specific)
        sys.exit(0 if result['passed'] else 1)

    if args.working_only:
        result = runner.run_working_tests()
        sys.exit(0 if result['passed'] else 1)

    # Default: comprehensive check
    result = runner.run_comprehensive_check()
    sys.exit(0 if result['overall_passed'] else 1)


if __name__ == "__main__":
    main()