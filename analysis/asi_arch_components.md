# ASI-Arch Components Analysis for Removal

**Date**: 2025-09-26
**Task**: T001 - Complete ASI-Arch removal documentation

## Components Identified for Removal

### 1. Core Pipeline Directory: `pipeline/`
**Location**: `/pipeline/`
**Purpose**: Neural architecture discovery pipeline (evolve, eval, analyse)
**Removal Rationale**: ASI-GO-2 replaces this functionality for research intelligence
**Components**:
- `pipeline/evolve/` - Architecture evolution logic
- `pipeline/eval/` - Architecture evaluation components
- `pipeline/analyse/` - Architecture analysis tools
- `pipeline/database/` - MongoDB-based storage for architecture data
- `pipeline/tools/` - Utility tools for pipeline operations
- `pipeline/config.py` - Pipeline configuration
- `pipeline/pipeline.py` - Main pipeline orchestrator

### 2. ASI-Arch Bridge Components
**Location**: `extensions/context_engineering/`
**Components**:
- `asi_arch_bridge.py` - Legacy ASI-Arch integration bridge
- `asi_arch_thoughtseed_bridge.py` - ASI-Arch to ThoughtSeed bridge
- `flux_asi_arch_interface.py` - Flux web interface to ASI-Arch
- `data/unified_asi_arch.db` - ASI-Arch database files

### 3. Test Integration Files
**Components**:
- `test_asi_arch_integration.py` - ASI-Arch integration tests
- `test_thoughtseed_asi_arch_integration.py` - ThoughtSeed-ASI-Arch integration tests

### 4. Requirements and Documentation
**Components**:
- `requirements-asi-arch.txt` - ASI-Arch specific dependencies
- `resources/ASI-Arch.pdf` - ASI-Arch documentation (keep for reference)
- `spec-management/ASI-Arch-Specs/` - ASI-Arch specifications (keep for reference)

### 5. Python Cache Files
**Components**:
- All `__pycache__` directories under `pipeline/`
- Compiled `.pyc` files for ASI-Arch components

## Import Dependencies Analysis

### Files Importing ASI-Arch Components
**Method**: Search for imports across codebase
**Key Import Patterns**:
- `from pipeline.` imports
- `import pipeline` statements
- `from .asi_arch_` imports
- ASI-Arch configuration references

## Configuration Dependencies

### Configuration Files Referencing ASI-Arch
- `configs/flux.yaml` - May contain ASI-Arch pipeline references
- Environment variables for ASI-Arch database connections
- Docker configurations for ASI-Arch services

## Database Schema Dependencies

### ASI-Arch Database Schemas
**Location**: `backend/src/models/` (if exists)
**Schema Types**:
- Architecture evolution models
- Neural network architecture representations
- Performance evaluation metrics
- MongoDB collections for architecture data

## Removal Plan

### Phase 1: Analysis and Backup
1. ✅ Document all components (this file)
2. Create backup of pipeline directory (optional)
3. Identify all import dependencies
4. Document configuration references

### Phase 2: Safe Removal
1. Remove import statements first (prevent import errors)
2. Remove pipeline directory
3. Remove ASI-Arch bridge files
4. Remove test files
5. Clean requirements files
6. Remove database schemas
7. Clean Python cache files

### Phase 3: Verification
1. Verify no broken imports remain
2. Confirm system starts without ASI-Arch
3. Test that existing functionality still works
4. Validate no ASI-Arch dependencies remain

## Risk Assessment

### Low Risk Components
- `pipeline/` directory (isolated)
- Test files (isolated)
- Documentation files (can be archived)
- Python cache files (auto-regenerated)

### Medium Risk Components
- ASI-Arch bridge files (may be referenced by other components)
- Requirements files (may affect dependency resolution)
- Configuration references (may break system startup)

### High Risk Areas
- Import statements in core system files
- Database schema references in models
- Configuration dependencies in startup code

## Post-Removal Validation Checklist
- [ ] System starts without errors
- [ ] No import errors in logs
- [ ] Existing document processing still works
- [ ] Context Engineering components function normally
- [ ] ThoughtSeed system operates independently
- [ ] No orphaned database connections
- [ ] Configuration files valid
- [ ] All tests pass (except ASI-Arch specific)

## Notes for Implementation Team
- Keep `resources/ASI-Arch.pdf` and `spec-management/ASI-Arch-Specs/` for historical reference
- Ensure ASI-GO-2 integration doesn't rely on any ASI-Arch components
- Verify Context Engineering can operate without ASI-Arch bridges
- Check that Daedalus delegation doesn't reference ASI-Arch components