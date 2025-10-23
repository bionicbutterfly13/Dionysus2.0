# Week 1 Foundation - Desktop App Initialization

**Change ID**: week-1-foundation
**Date**: 2025-10-14
**Status**: In Progress
**Type**: New Capability

## Why

Flux needs a desktop foundation to evolve from web app to native application with deep OS integration. The current web-based architecture limits file system access, native menus, and desktop-specific features required for a Zettlr-inspired knowledge management system.

Tauri 2.0 provides a lightweight (3-10MB vs 85-100MB), fast (<0.5s startup vs 1-2s), and secure desktop framework that enables:
- Direct file system operations without web security constraints
- Native OS dialogs and menus
- Desktop-first user experience
- Offline-first architecture
- Platform-specific optimizations

## What Changes

### New Capabilities
- **Desktop Application Framework** (Tauri 2.0)
  - Rust backend for native operations
  - WebView frontend for React UI
  - Platform abstraction for future flexibility

- **File System Operations**
  - Read/write local markdown files
  - Watch file changes in real-time
  - Native file/folder dialogs

- **Workspace Management**
  - Open local folder as workspace
  - Persist workspace path across sessions
  - Auto-load last workspace on startup

- **CodeMirror 6 Integration**
  - Markdown editor with syntax highlighting
  - WYSIWYM editing experience
  - Extension system for WikiLinks and consciousness markers

- **Native Desktop Features**
  - Application menus
  - Window management
  - Shell command execution

### Modified Components
- **React Frontend**: Adapted from web (`BrowserRouter` → `HashRouter`)
- **Python Backend**: Connect via HTTP/WebSocket instead of same-origin
- **Build System**: Vite config for Tauri integration

### BREAKING CHANGES
- **Deployment Model**: From web app (browser URL) to desktop app (installer)
- **User Migration**: Users must download and install `.dmg` (macOS) or `.exe` (Windows)
- **File Access**: From server-side to client-side file operations

## Impact

### Affected Specs
- **NEW**: `desktop-core` - Core desktop application capabilities
- **MODIFIED**: None (new capability, no existing specs)

### Affected Code
- **NEW**: `flux-desktop/` - Entire Tauri project directory
- **REUSED**: `frontend/src/components/` - React components migrated to desktop
- **MODIFIED**: `frontend/src/App.tsx` - Routing adapted for desktop

### Migration Path
1. **Week 1**: Desktop foundation setup
2. **Week 2-3**: Migrate core React components
3. **Week 4**: Dual deployment (web + desktop) for transition period
4. **Week 5+**: Desktop-exclusive features (file system, workspace)

### User Impact
- **Installation Required**: One-time download and install
- **Local Files**: Direct access to markdown files (no upload needed)
- **Offline First**: Works without internet connection
- **Better Performance**: Native app speed vs web app latency

## Dependencies

### Technical Dependencies
- Rust toolchain v1.90.0+ (installed)
- Tauri CLI v2.8.4+ (installed)
- Node.js 18+ (existing)
- Tauri plugins: fs, dialog, shell, window

### Specification Dependencies
- None (foundational change)

### Implementation Dependencies
- Existing React frontend (to be migrated)
- Python backend (to be connected)
- Build pipeline (to be configured)

## Risks and Mitigations

### Risk 1: Platform Abstraction Complexity
**Risk**: Tight coupling to Tauri makes switching frameworks costly
**Severity**: Medium
**Mitigation**:
- Implement `DesktopPlatform` interface from Day 1
- Abstract all Tauri calls behind interface
- Document migration path to Electron if needed (~2 weeks vs 3 months)

### Risk 2: First-Time Rust Development
**Risk**: Team unfamiliar with Rust may slow development
**Severity**: Low
**Mitigation**:
- Tauri handles most Rust code via templates
- Use Tauri plugins for common operations
- Rust code limited to Tauri command layer

### Risk 3: User Adoption of Desktop App
**Risk**: Users resist downloading installer vs web access
**Severity**: Medium
**Mitigation**:
- Maintain web version for 4 weeks during transition
- Highlight desktop-exclusive features (file system, performance)
- Provide clear migration guide

## Success Criteria

### Technical Success
- ✅ App launches in <0.5 seconds
- ✅ Hot reload works during development
- ✅ Can read and write local files
- ✅ React components render correctly
- ✅ Python backend connects successfully
- ✅ Bundle size <15MB
- ✅ Memory usage <150MB

### User Success
- ✅ User can open Flux.app and see main window
- ✅ App feels responsive and native (not web-like)
- ✅ File operations work without lag
- ✅ App updates don't break user data

## Implementation Notes

### Week 1 Focus
This proposal covers ONLY Week 1 foundation:
- Project initialization ✅ (completed)
- Dependency setup ✅ (completed)
- Basic window rendering ⏳ (in progress)
- Component migration 🔲 (pending)
- Platform abstraction 🔲 (pending)

### Future Weeks
Subsequent weeks will have separate OpenSpec proposals:
- Week 2: File management and workspace
- Week 3: CodeMirror editor integration
- Week 4: State persistence and sync
- etc.

## Validation Checklist

- [x] Proposal describes clear motivation (Why)
- [x] Changes are well-defined (What Changes)
- [x] Impact assessed (Affected specs, code, users)
- [x] Dependencies identified
- [x] Risks documented with mitigations
- [x] Success criteria measurable
- [x] Breaking changes explicitly called out
