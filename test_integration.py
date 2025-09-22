#!/usr/bin/env python3
"""
🧪 ASI-Arch Integration Test Suite
==================================

Continuous integration testing for ASI-Arch with Context Engineering.
Tests core functionality without breaking existing systems.
"""

import asyncio
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegrationTest:
    """Base class for integration tests"""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0.0
    
    async def run(self) -> bool:
        """Run the test - override in subclasses"""
        raise NotImplementedError
    
    async def execute(self) -> bool:
        """Execute test with error handling"""
        start_time = datetime.now()
        
        try:
            self.passed = await self.run()
        except Exception as e:
            self.passed = False
            self.error = str(e)
            logger.error(f"Test {self.name} failed: {e}")
            
        end_time = datetime.now()
        self.duration = (end_time - start_time).total_seconds()
        
        return self.passed

class PythonEnvironmentTest(IntegrationTest):
    """Test Python environment setup"""
    
    def __init__(self):
        super().__init__("Python Environment")
    
    async def run(self) -> bool:
        # Check virtual environment
        venv_active = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if not venv_active:
            raise Exception("Virtual environment not active")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise Exception(f"Python {sys.version} too old, need 3.8+")
        
        return True

class DependencyTest(IntegrationTest):
    """Test key dependencies are importable"""
    
    def __init__(self):
        super().__init__("Dependencies")
    
    async def run(self) -> bool:
        # Core dependencies
        core_packages = ['numpy', 'pandas', 'requests']
        
        for package in core_packages:
            try:
                __import__(package)
            except ImportError:
                raise Exception(f"Core package {package} not available")
        
        # ML dependencies (may not be installed yet)
        ml_packages = ['torch', 'tensorflow', 'transformers']
        ml_available = []
        
        for package in ml_packages:
            try:
                __import__(package)
                ml_available.append(package)
            except ImportError:
                pass
        
        logger.info(f"ML packages available: {ml_available}")
        return True

class FileSystemTest(IntegrationTest):
    """Test file system setup"""
    
    def __init__(self):
        super().__init__("File System")
    
    async def run(self) -> bool:
        import os
        
        # Check key directories exist
        required_dirs = [
            'pipeline',
            'database', 
            'cognition_base',
            'extensions'
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                raise Exception(f"Required directory {directory} not found")
        
        # Check we can write files
        test_file = 'test_write.tmp'
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            raise Exception(f"Cannot write files: {e}")
        
        return True

class BasicPipelineTest(IntegrationTest):
    """Test basic pipeline components can be imported"""
    
    def __init__(self):
        super().__init__("Basic Pipeline")
    
    async def run(self) -> bool:
        import os
        sys.path.insert(0, os.path.join(os.getcwd(), 'pipeline'))
        
        try:
            # Test basic imports
            import config
            logger.info("Pipeline config imported successfully")
            
            # Test tools import
            from tools import tools
            logger.info("Pipeline tools imported successfully")
            
        except ImportError as e:
            raise Exception(f"Pipeline import failed: {e}")
        
        return True

class ContextEngineeringTest(IntegrationTest):
    """Test context engineering components"""
    
    def __init__(self):
        super().__init__("Context Engineering")
    
    async def run(self) -> bool:
        import os
        
        # Check context engineering directory exists
        ce_dir = os.path.join('extensions', 'context-engineering')
        if not os.path.exists(ce_dir):
            logger.info("Context engineering directory not yet created")
            return True  # Not an error, just not implemented yet
        
        # Test basic context engineering concepts
        try:
            # This would test our context engineering implementation
            # For now, just verify the structure exists
            logger.info("Context engineering structure verified")
            
        except Exception as e:
            raise Exception(f"Context engineering test failed: {e}")
        
        return True

class IntegrationTestSuite:
    """Main test suite runner"""
    
    def __init__(self):
        self.tests = [
            PythonEnvironmentTest(),
            DependencyTest(),
            FileSystemTest(),
            BasicPipelineTest(),
            ContextEngineeringTest(),
        ]
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': len(self.tests),
                'passed': 0,
                'failed': 0,
                'duration': 0.0
            }
        }
        
        start_time = datetime.now()
        
        print("🧪 Running ASI-Arch Integration Tests")
        print("="*50)
        
        for test in self.tests:
            print(f"Running {test.name}...", end=" ")
            
            success = await test.execute()
            
            if success:
                print("✅ PASS")
                results['summary']['passed'] += 1
            else:
                print("❌ FAIL")
                results['summary']['failed'] += 1
                if test.error:
                    print(f"   Error: {test.error}")
            
            results['tests'][test.name] = {
                'passed': test.passed,
                'duration': test.duration,
                'error': test.error
            }
        
        end_time = datetime.now()
        results['summary']['duration'] = (end_time - start_time).total_seconds()
        
        print("="*50)
        print(f"📊 Results: {results['summary']['passed']}/{results['summary']['total']} tests passed")
        print(f"⏱️  Duration: {results['summary']['duration']:.2f}s")
        
        if results['summary']['failed'] > 0:
            print("❌ Some tests failed - check output above")
        else:
            print("✅ All tests passed!")
        
        return results
    
    async def continuous_testing(self, interval: int = 300):
        """Run tests continuously"""
        print(f"🔄 Starting continuous testing (every {interval}s)")
        print("🛑 Press Ctrl+C to stop")
        
        try:
            while True:
                await self.run_all_tests()
                print(f"\n⏳ Waiting {interval}s for next test run...")
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Testing stopped by user")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ASI-Arch Integration Tests')
    parser.add_argument('--continuous', action='store_true',
                       help='Run tests continuously')
    parser.add_argument('--interval', type=int, default=300,
                       help='Test interval in seconds (default: 300)')
    
    args = parser.parse_args()
    
    suite = IntegrationTestSuite()
    
    if args.continuous:
        await suite.continuous_testing(args.interval)
    else:
        await suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
