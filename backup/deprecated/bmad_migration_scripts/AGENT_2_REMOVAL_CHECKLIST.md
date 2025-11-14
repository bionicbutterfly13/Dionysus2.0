# Agent 2: BMAD Removal Checklist (GREEN Phase)

**Mission**: Remove all BMAD references to make tests pass

**Test Command**: `pytest backend/tests/test_bmad_removal.py -v`

## Pre-Flight Check

Before starting, verify RED phase:
```bash
cd backend
pytest tests/test_bmad_removal.py -v
# Should show: 4 FAILED, 6 PASSED, 1 SKIPPED
```

## Removal Checklist

### ☐ Step 1: Archive Python Migration Script

```bash
# Create archive directory
mkdir -p backup/deprecated/bmad_migration

# Move migration script
mv backend/migrate_bmad_to_consciousness.py backup/deprecated/bmad_migration/

# Verify removal
ls backend/migrate_bmad_to_consciousness.py  # Should fail (file not found)
```

**Test to verify**: `test_no_bmad_migration_script` should PASS

### ☐ Step 2: Clean check_consciousness_systems.py

**File**: `backend/check_consciousness_systems.py`

**Remove these sections**:
1. `check_bmad_migration()` function (lines ~39-66)
2. BMAD Migration section in main() (lines ~139-146)
3. Any BMAD-related imports

**After editing, test**:
```bash
pytest backend/tests/test_bmad_removal.py::test_no_bmad_check_function -v
# Should PASS
```

### ☐ Step 3: Archive BMAD Documentation

```bash
# Create documentation archive
mkdir -p backup/deprecated/bmad_documentation

# Archive completion document
mv BMAD_REMOVAL_COMPLETE.md backup/deprecated/bmad_documentation/

# Verify
ls BMAD_REMOVAL_COMPLETE.md  # Should fail
```

### ☐ Step 4: Clean CONSCIOUSNESS_SYSTEMS_README.md

**File**: `CONSCIOUSNESS_SYSTEMS_README.md`

**Remove**:
- Any sections mentioning BMAD migration
- BMAD-related workflow instructions
- References to migrate_bmad_to_consciousness.py

**Keep**:
- All consciousness system documentation
- Active inference explanations
- System architecture descriptions

### ☐ Step 5: Clean Root CLAUDE.md

**File**: `CLAUDE.md` (project root)

**Remove**:
- "### ✅ BMAD Migration to OpenSpec + Archon" section
- Any BMAD migration workflow instructions
- References to check_bmad_migration()

**Keep**:
- OpenSpec integration instructions
- Archon workflow documentation
- All other project context

**Test after**:
```bash
pytest backend/tests/test_bmad_removal.py::test_claude_md_no_bmad_migration_docs -v
# Should PASS
```

### ☐ Step 6: Clean Backend CLAUDE.md

**File**: `backend/CLAUDE.md`

**Remove**:
- BMAD references in project overview
- BMAD migration documentation
- Any BMAD workflow sections

**Keep**:
- All architecture documentation
- API documentation
- Testing instructions

### ☐ Step 7: Archive Gemini Agent Configs

```bash
# Create Gemini agents archive
mkdir -p backup/deprecated/gemini_agents

# Move all agent configs
mv .gemini/commands/agents/*.toml backup/deprecated/gemini_agents/

# Verify directory is empty
ls .gemini/commands/agents/  # Should be empty or not exist
```

**Test after**:
```bash
pytest backend/tests/test_bmad_removal.py::test_gemini_agents_removed -v
# Should PASS
```

## Continuous Validation

After each step, run:
```bash
pytest backend/tests/test_bmad_removal.py -v
```

Watch the number of failures decrease:
- Start: 4 FAILED
- Target: 0 FAILED

## Final Verification

When all removals complete:

```bash
# Run full test suite
pytest backend/tests/test_bmad_removal.py -v

# Expected output:
# ============ 10 passed, 1 skipped, 202 warnings ============

# Run summary report
pytest backend/tests/test_bmad_removal.py::test_summary_report -v -s

# Expected output:
# ✅ NO BMAD REFERENCES FOUND - All tests should pass!
```

## Success Criteria

✅ All these tests PASS:
- [ ] `test_no_bmad_in_python_code`
- [ ] `test_no_bmad_in_typescript_code`
- [ ] `test_no_bmad_in_markdown_docs`
- [ ] `test_no_bmad_migration_script`
- [ ] `test_no_bmad_check_function`
- [ ] `test_gemini_agents_removed`
- [ ] `test_neo4j_schema_bmad_free`
- [ ] `test_claude_md_no_bmad_migration_docs`
- [ ] `test_no_bmad_in_config_files`
- [ ] `test_no_bmad_imports`
- [ ] `test_summary_report`

## Troubleshooting

### If tests still fail after removal:

1. **Check backup directories**:
   ```bash
   # Files in backup/ should NOT be searched
   ls backup/deprecated/bmad_migration/
   ls backup/deprecated/bmad_documentation/
   ls backup/deprecated/gemini_agents/
   ```

2. **Run summary report**:
   ```bash
   pytest backend/tests/test_bmad_removal.py::test_summary_report -v -s
   # Shows exactly which files still have BMAD references
   ```

3. **Check allowed exceptions**:
   - `backend/tests/test_bmad_removal.py` - OK to have BMAD (it's the test)
   - `backup/deprecated/*` - OK to have BMAD (archived)
   - Other files - NOT OK, must be cleaned

## Post-Removal

After GREEN phase complete:

1. **Create GREEN phase report**:
   - Document what was removed
   - Show before/after test results
   - List archived files

2. **Run related tests**:
   ```bash
   # Ensure system still works
   pytest backend/tests/ -v
   ```

3. **Ready for Agent 3** (REFACTOR phase):
   - Agent 3 will review test quality
   - Optimize test performance
   - Add edge case coverage

## Quick Reference

**Test file**: `backend/tests/test_bmad_removal.py`
**Report**: `TDD_BMAD_REMOVAL_REPORT.md`
**This checklist**: `AGENT_2_REMOVAL_CHECKLIST.md`

**Run all tests**:
```bash
pytest backend/tests/test_bmad_removal.py -v
```

**Run specific test**:
```bash
pytest backend/tests/test_bmad_removal.py::test_name -v
```

**Watch mode** (requires pytest-watch):
```bash
ptw backend/tests/test_bmad_removal.py -- -v
```

---

**Ready to start**: All tests are RED (failing) as expected
**Goal**: All tests GREEN (passing)
**Next Agent**: Agent 3 (REFACTOR phase)
