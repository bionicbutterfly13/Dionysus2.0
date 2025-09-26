#!/usr/bin/env python3
"""
Constitutional Compliance Checker
=================================

MANDATORY tool for all agents to verify constitutional compliance
before any operations. Prevents NumPy compatibility issues and ensures
system stability.

Constitutional Requirements:
- NumPy version MUST be < 2.0
- All ML packages MUST be compatible
- Environment isolation MUST be used
- Service conflicts MUST be avoided

Author: ASI-Arch Constitutional Enforcement
Date: 2025-09-24
Version: 1.0.0
"""

import sys
import subprocess
import importlib
import pkg_resources
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

class ConstitutionalViolation(Exception):
    """Raised when constitutional requirements are violated"""
    pass

class ConstitutionalComplianceChecker:
    """
    Enforces ASI-Arch Agent Constitution compliance
    
    Checks:
    1. NumPy version compliance (< 2.0)
    2. ML package compatibility
    3. Environment isolation
    4. Service conflict detection
    5. ThoughtSeed integration readiness
    """
    
    def __init__(self):
        self.violations: List[Dict[str, str]] = []
        self.warnings: List[Dict[str, str]] = []
        self.compliance_report: Dict[str, any] = {}
        
    def check_constitutional_compliance(self) -> bool:
        """
        Perform comprehensive constitutional compliance check
        
        Returns:
            bool: True if compliant, False if violations found
        """
        print("🏛️ ASI-Arch Constitutional Compliance Check")
        print("=" * 60)
        
        # Check 1: NumPy version compliance (CRITICAL)
        numpy_compliant = self._check_numpy_compliance()
        
        # Check 2: ML package compatibility
        ml_compliant = self._check_ml_package_compatibility()
        
        # Check 3: Environment isolation
        env_compliant = self._check_environment_isolation()
        
        # Check 4: Service conflict detection
        service_compliant = self._check_service_conflicts()
        
        # Check 5: ThoughtSeed integration readiness
        thoughtseed_compliant = self._check_thoughtseed_readiness()
        
        # Generate compliance report
        self._generate_compliance_report()
        
        # Determine overall compliance
        overall_compliant = all([
            numpy_compliant,
            ml_compliant,
            env_compliant,
            service_compliant,
            thoughtseed_compliant
        ])
        
        if overall_compliant:
            print("\n✅ CONSTITUTIONAL COMPLIANCE VERIFIED")
            print("🟢 All agents may proceed with operations")
            return True
        else:
            print("\n❌ CONSTITUTIONAL VIOLATIONS DETECTED")
            print("🔴 Operations must be suspended until violations are resolved")
            self._report_violations()
            return False
    
    def _check_numpy_compliance(self) -> bool:
        """Check NumPy version compliance (CRITICAL)"""
        print("\n1. 🔍 Checking NumPy Version Compliance...")
        
        try:
            import numpy
            numpy_version = numpy.__version__
            
            if numpy_version.startswith('2.'):
                violation = {
                    'type': 'CRITICAL',
                    'component': 'NumPy',
                    'issue': f'NumPy {numpy_version} violates constitution',
                    'requirement': 'NumPy version MUST be < 2.0',
                    'fix': 'pip install "numpy<2" --force-reinstall'
                }
                self.violations.append(violation)
                print(f"❌ CRITICAL VIOLATION: NumPy {numpy_version}")
                print("   🚨 CONSTITUTION VIOLATION: NumPy 2.x detected")
                return False
            else:
                print(f"✅ NumPy {numpy_version} compliant")
                return True
                
        except ImportError:
            violation = {
                'type': 'HIGH',
                'component': 'NumPy',
                'issue': 'NumPy not installed',
                'requirement': 'NumPy required for ML operations',
                'fix': 'pip install "numpy<2"'
            }
            self.violations.append(violation)
            print("❌ HIGH VIOLATION: NumPy not installed")
            return False
    
    def _check_ml_package_compatibility(self) -> bool:
        """Check ML package compatibility with NumPy"""
        print("\n2. 🔍 Checking ML Package Compatibility...")
        
        ml_packages = [
            'torch', 'tensorflow', 'transformers', 'sentence_transformers',
            'scikit-learn', 'pandas', 'matplotlib', 'seaborn'
        ]
        
        compatible_count = 0
        total_count = len(ml_packages)
        
        for package in ml_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"   ✅ {package}: {version}")
                compatible_count += 1
            except ImportError:
                warning = {
                    'type': 'MEDIUM',
                    'component': package,
                    'issue': f'{package} not installed',
                    'requirement': f'{package} recommended for ML operations',
                    'fix': f'pip install {package}'
                }
                self.warnings.append(warning)
                print(f"   ⚠️ {package}: not installed")
        
        compatibility_ratio = compatible_count / total_count
        if compatibility_ratio >= 0.8:  # 80% compatibility threshold
            print(f"✅ ML Package Compatibility: {compatible_count}/{total_count} ({compatibility_ratio:.1%})")
            return True
        else:
            print(f"❌ ML Package Compatibility: {compatible_count}/{total_count} ({compatibility_ratio:.1%})")
            return False
    
    def _check_environment_isolation(self) -> bool:
        """Check if environment is properly isolated"""
        print("\n3. 🔍 Checking Environment Isolation...")
        
        # Check if we're in a virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if in_venv:
            print("✅ Virtual environment detected")
            return True
        else:
            warning = {
                'type': 'MEDIUM',
                'component': 'Environment',
                'issue': 'Not in virtual environment',
                'requirement': 'Virtual environment recommended for isolation',
                'fix': 'python -m venv asi-arch-env && source asi-arch-env/bin/activate'
            }
            self.warnings.append(warning)
            print("⚠️ Not in virtual environment (recommended)")
            return True  # Not a critical violation
    
    def _check_service_conflicts(self) -> bool:
        """Check for service conflicts"""
        print("\n4. 🔍 Checking Service Conflicts...")
        
        # Check for running processes that might conflict
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = result.stdout
            
            # Check for conflicting services
            conflicting_services = []
            if 'rag_api.py' in processes:
                conflicting_services.append('RAG API')
            if 'thoughtseed_service.py' in processes:
                conflicting_services.append('ThoughtSeed Service')
            if 'mongodb' in processes:
                conflicting_services.append('MongoDB')
            
            if conflicting_services:
                warning = {
                    'type': 'LOW',
                    'component': 'Services',
                    'issue': f'Active services detected: {", ".join(conflicting_services)}',
                    'requirement': 'Coordinate with other agents',
                    'fix': 'Check with other agents before starting new services'
                }
                self.warnings.append(warning)
                print(f"⚠️ Active services: {', '.join(conflicting_services)}")
            
            print("✅ Service conflict check complete")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not check service conflicts: {e}")
            return True  # Not critical
    
    def _check_thoughtseed_readiness(self) -> bool:
        """Check ThoughtSeed integration readiness"""
        print("\n5. 🔍 Checking ThoughtSeed Integration Readiness...")
        
        try:
            # Check if ThoughtSeed models are available
            sys.path.append('backend/models')
            from thoughtseed_trace import ThoughtSeedTrace
            print("✅ ThoughtSeed Trace Model available")
            
            # Check if ThoughtSeed service is available
            sys.path.append('backend/services')
            from thoughtseed_service import ThoughtSeedService
            print("✅ ThoughtSeed Service available")
            
            # Check if context engineering is available
            sys.path.append('extensions/context_engineering')
            from core_implementation import ContextEngineeringService
            print("✅ Context Engineering Service available")
            
            print("✅ ThoughtSeed integration ready")
            return True
            
        except ImportError as e:
            violation = {
                'type': 'HIGH',
                'component': 'ThoughtSeed',
                'issue': f'ThoughtSeed integration not ready: {e}',
                'requirement': 'ThoughtSeed integration required',
                'fix': 'Ensure ThoughtSeed models and services are properly installed'
            }
            self.violations.append(violation)
            print(f"❌ ThoughtSeed integration not ready: {e}")
            return False
    
    def _generate_compliance_report(self):
        """Generate comprehensive compliance report"""
        self.compliance_report = {
            'timestamp': datetime.now().isoformat(),
            'violations': self.violations,
            'warnings': self.warnings,
            'compliance_score': self._calculate_compliance_score(),
            'recommendations': self._generate_recommendations()
        }
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score"""
        total_checks = 5
        violation_penalty = len(self.violations) * 0.2
        warning_penalty = len(self.warnings) * 0.05
        
        score = max(0.0, 1.0 - violation_penalty - warning_penalty)
        return score
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for compliance improvement"""
        recommendations = []
        
        if self.violations:
            recommendations.append("🚨 CRITICAL: Resolve constitutional violations immediately")
        
        if self.warnings:
            recommendations.append("⚠️ Address warnings to improve compliance")
        
        recommendations.extend([
            "📋 Use frozen requirements: pip install -r requirements-frozen.txt",
            "🛡️ Always use virtual environments for isolation",
            "🤝 Coordinate with other agents before starting services",
            "🧪 Run compliance check before any major operations"
        ])
        
        return recommendations
    
    def _report_violations(self):
        """Report constitutional violations"""
        print("\n🚨 CONSTITUTIONAL VIOLATIONS REPORT")
        print("=" * 50)
        
        for violation in self.violations:
            print(f"\n❌ {violation['type']} VIOLATION:")
            print(f"   Component: {violation['component']}")
            print(f"   Issue: {violation['issue']}")
            print(f"   Requirement: {violation['requirement']}")
            print(f"   Fix: {violation['fix']}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"   - {warning['component']}: {warning['issue']}")
    
    def save_compliance_report(self, filepath: str = "constitutional_compliance_report.json"):
        """Save compliance report to file"""
        with open(filepath, 'w') as f:
            json.dump(self.compliance_report, f, indent=2)
        print(f"📄 Compliance report saved to {filepath}")

def main():
    """Main compliance check function"""
    checker = ConstitutionalComplianceChecker()
    
    compliant = checker.check_constitutional_compliance()
    
    # Save report
    checker.save_compliance_report()
    
    if not compliant:
        print("\n🛑 OPERATIONS SUSPENDED")
        print("Resolve violations before proceeding")
        sys.exit(1)
    else:
        print("\n🚀 OPERATIONS AUTHORIZED")
        print("Constitutional compliance verified")

if __name__ == "__main__":
    main()
