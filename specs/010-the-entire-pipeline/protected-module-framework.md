# Protected Module Framework Specification

**Feature**: Protected Module Framework for Constitutional Compliance
**Branch**: 010-the-entire-pipeline
**Date**: 2025-09-27
**Constitutional Reference**: CONST_ARCH_2025 Section 5

## Overview

This specification defines a protective framework for critical system modules that must be shielded from accidental modification by code generators, CLIs, SDKs, or automated tools.

## Constitutional Requirement

Per the updated constitutional framework, certain classes and modules are **PROTECTED** from subsequent code generation unless explicitly approved by developers through a high-visibility warning system.

## Protected Module Categories

### 1. Core Markov Blanket Implementation
**Location**: `/dionysus-source/src/perceptual_core/_markov_blanket.py`
**Protection Level**: MAXIMUM
**Rationale**: Contains the foundational Markov blanket enforcement system that cannot be modified without breaking constitutional compliance.

### 2. Constitutional Gateway
**Location**: `/dionysus-source/constitutional_document_gateway.py`
**Protection Level**: MAXIMUM
**Rationale**: Enforces constitutional document processing pipeline.

### 3. Daedalus Core Controllers
**Location**: `/dionysus-source/agents/daedalus_*`
**Protection Level**: HIGH
**Rationale**: Core entry point controllers that manage all information intake.

### 4. ASI-GO-2 Research Components
**Location**: `/resources/ASI-GO-2/`
**Protection Level**: HIGH
**Rationale**: Research-validated components that require careful integration.

### 5. Memory Orchestrator Core
**Location**: Memory orchestrator implementation files
**Protection Level**: HIGH
**Rationale**: Triple-store memory system that requires consistency.

## Protection Implementation Strategy

### Warning System Requirements

#### 1. Bright Yellow Warning Display
```python
class ProtectedModuleWarning:
    """
    Generates BRIGHT YELLOW warnings for protected module access
    """

    @staticmethod
    def display_protection_warning(module_path: str, protection_level: str) -> bool:
        """
        Display bright yellow warning that cannot be missed

        Returns:
            bool: True if developer explicitly grants permission
        """
        warning_message = f"""
        ⚠️  🟨 PROTECTED MODULE ACCESS WARNING 🟨 ⚠️
        ╔════════════════════════════════════════════════════╗
        ║                CONSTITUTIONAL PROTECTION            ║
        ║                                                    ║
        ║  Module: {module_path:<45} ║
        ║  Protection Level: {protection_level:<35} ║
        ║                                                    ║
        ║  This module is CONSTITUTIONALLY PROTECTED         ║
        ║  Modification may break system compliance          ║
        ║                                                    ║
        ║  Do you EXPLICITLY grant permission? (yes/no)     ║
        ╚════════════════════════════════════════════════════╝
        """

        print("\033[93m" + warning_message + "\033[0m")  # Bright yellow
        response = input("Your explicit permission (type 'yes' exactly): ")
        return response.strip().lower() == "yes"
```

#### 2. Development-Time Exception System
```python
class ProtectedModuleException(Exception):
    """
    Exception thrown when protected modules are accessed without permission
    """
    def __init__(self, module_path: str, protection_level: str):
        self.module_path = module_path
        self.protection_level = protection_level
        super().__init__(
            f"CONSTITUTIONAL VIOLATION: Attempted access to protected module "
            f"{module_path} (Level: {protection_level}) without explicit permission"
        )
```

#### 3. Protection Decorator System
```python
def protected_module(protection_level: str = "HIGH"):
    """
    Decorator to mark classes/functions as constitutionally protected

    Args:
        protection_level: "MAXIMUM", "HIGH", "MEDIUM"
    """
    def decorator(cls_or_func):
        # Add protection metadata
        cls_or_func._constitutional_protection = {
            "level": protection_level,
            "protected": True,
            "module_path": cls_or_func.__module__
        }

        # Wrap class/function with protection check
        if isinstance(cls_or_func, type):  # Class
            original_init = cls_or_func.__init__

            def protected_init(self, *args, **kwargs):
                _check_protection_permission(cls_or_func)
                return original_init(self, *args, **kwargs)

            cls_or_func.__init__ = protected_init
            return cls_or_func

        else:  # Function
            def protected_wrapper(*args, **kwargs):
                _check_protection_permission(cls_or_func)
                return cls_or_func(*args, **kwargs)

            return protected_wrapper

    return decorator
```

### Integration with Development Tools

#### 1. Code Generator Integration
All code generators (CLI tools, SDKs, automated scripts) MUST:
- Check for `_constitutional_protection` metadata before modification
- Display protection warnings for any protected modules
- Log all protection bypass attempts with developer approval

#### 2. IDE Integration
Development environments SHOULD:
- Highlight protected modules with visual indicators
- Show protection warnings in tooltips
- Require explicit confirmation for protected module edits

#### 3. Version Control Hooks
Git hooks SHOULD:
- Scan commits for protected module changes
- Require additional approval for protected module modifications
- Log protection status in commit messages

## Protected Module Registry

### Registry File: `protected_modules.json`
```json
{
  "protected_modules": {
    "/dionysus-source/src/perceptual_core/_markov_blanket.py": {
      "protection_level": "MAXIMUM",
      "reason": "Core Markov blanket implementation",
      "constitutional_reference": "CONST_ARCH_2025",
      "last_approved_modification": null,
      "modification_history": []
    },
    "/dionysus-source/constitutional_document_gateway.py": {
      "protection_level": "MAXIMUM",
      "reason": "Constitutional processing enforcement",
      "constitutional_reference": "CONST_ARCH_2025",
      "last_approved_modification": null,
      "modification_history": []
    },
    "/resources/ASI-GO-2/main.py": {
      "protection_level": "HIGH",
      "reason": "Research-validated ASI-GO-2 orchestrator",
      "constitutional_reference": "CONST_ARCH_2025",
      "last_approved_modification": null,
      "modification_history": []
    }
  },
  "protection_config": {
    "warning_color": "yellow",
    "require_explicit_permission": true,
    "log_access_attempts": true,
    "development_exceptions_enabled": true
  }
}
```

## Implementation Phases

### Phase 1: Core Protection Framework
1. Implement `ProtectedModuleWarning` class
2. Implement `ProtectedModuleException` class
3. Create `@protected_module` decorator
4. Establish protected module registry

### Phase 2: Tool Integration
1. Integrate with existing CLI tools
2. Add protection checks to code generators
3. Implement development environment warnings
4. Set up logging and monitoring

### Phase 3: Validation & Testing
1. Test protection warnings with various tools
2. Validate constitutional compliance
3. Performance impact assessment
4. Developer workflow optimization

## Testing Requirements

### Test Cases
```python
def test_protected_module_warning_display():
    """Test that bright yellow warnings are properly displayed"""
    assert warning_system.display_warning() == True  # With approval
    assert warning_system.display_warning() == False  # Without approval

def test_protection_exception_in_development():
    """Test that exceptions are thrown in development mode"""
    with pytest.raises(ProtectedModuleException):
        access_protected_module_without_permission()

def test_decorator_protection():
    """Test that protected_module decorator prevents access"""
    @protected_module(protection_level="HIGH")
    class TestProtectedClass:
        pass

    # Should trigger warning/exception
    with protection_warning_context():
        instance = TestProtectedClass()

def test_registry_functionality():
    """Test protected module registry operations"""
    registry = ProtectedModuleRegistry()
    assert registry.is_protected("/path/to/module.py") == True
    assert registry.get_protection_level("/path/to/module.py") == "HIGH"
```

## Performance Considerations

### Minimal Overhead Design
- Protection checks only on module access/initialization
- Registry lookup optimized with caching
- Development-time checks only (no production overhead)
- Lazy loading of protection metadata

### Optimization Strategies
- Cache protection status after first check
- Use file path hashing for fast registry lookup
- Minimize warning display time impact
- Asynchronous logging for access attempts

## Security Implications

### Protection Bypass Prevention
- Warning system cannot be disabled programmatically
- Registry file requires elevated permissions to modify
- Protection metadata embedded in module objects
- Multiple verification layers for critical modules

### Audit Trail Requirements
- All protection bypass attempts logged
- Developer approval records maintained
- Modification history tracked per module
- Constitutional compliance reports generated

## Constitutional Compliance Verification

### Automated Checks
```python
def verify_constitutional_compliance():
    """
    Verify all constitutionally required modules are protected
    """
    required_protections = [
        ("/dionysus-source/src/perceptual_core/_markov_blanket.py", "MAXIMUM"),
        ("/dionysus-source/constitutional_document_gateway.py", "MAXIMUM"),
        # ... other required protections
    ]

    for module_path, expected_level in required_protections:
        actual_level = registry.get_protection_level(module_path)
        assert actual_level == expected_level, f"Protection violation: {module_path}"
```

### Compliance Reporting
- Daily constitutional compliance reports
- Protection status dashboard
- Modification approval tracking
- Violation detection and alerting

## Future Extensions

### Advanced Protection Features
- Role-based protection permissions
- Time-limited modification approvals
- Automatic protection level adjustment
- Integration with external security systems

### Research Integration
- Protection for research-validated implementations
- Version control for research component updates
- Collaboration workflow for protected modifications
- Academic citation requirements for protected modules

---

**Implementation Priority**: IMMEDIATE (Constitutional Requirement)
**Testing Required**: Comprehensive protection validation
**Documentation**: Update development workflow guides
**Training**: Developer education on protection system