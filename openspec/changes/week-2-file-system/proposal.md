# Week 2: File System & Workspace Management

**Change ID**: week-2-file-system
**Status**: Proposed
**Priority**: CRITICAL
**Target Week**: Oct 19-25, 2025
**Dependencies**: week-1-foundation (complete)

---

## Problem Statement

Users need to open, browse, and manage local markdown files in their Flux desktop app. Currently, the app has no way to access the local file system or organize files into workspaces.

## Proposed Solution

Implement a complete file system and workspace management system using the platform abstraction layer built in Week 1. This will enable users to:

1. Select a workspace folder from their local drive
2. Browse files in a tree view
3. Open and read markdown files
4. Create, rename, and delete files
5. Switch between multiple workspaces
6. Persist workspace state across sessions

## Technical Approach

### 1. Workspace Management
- Use platform abstraction's `dialog.openFolder()` for folder selection
- Store workspace state in localStorage
- Track recent workspaces (last 10)
- Auto-restore last workspace on app launch

### 2. File System Operations
- Leverage `platform.fs.readDir()` for directory listings
- Recursive file tree scanning
- File watching for external changes (future)
- Error handling for permissions, missing files

### 3. File Tree UI Component
- Collapsible folder tree
- File/folder icons with Lucide React
- Right-click context menu
- Keyboard navigation (arrows, enter)
- Search/filter within workspace

### 4. File Operations
- Create new markdown file
- Rename file/folder
- Delete file/folder (with confirmation)
- Duplicate file
- Move file (drag & drop - future)

## Implementation Plan

### Phase 1: Workspace Core (Days 1-2)
- Create `WorkspaceManager` service
- Implement folder selection dialog
- Build workspace state management (Zustand)
- Add workspace switcher UI

### Phase 2: File Tree (Days 3-4)
- Build `<FileTree>` component
- Implement recursive directory reading
- Add expand/collapse functionality
- Style tree with proper indentation

### Phase 3: File Operations (Days 5-6)
- Add context menu component
- Implement CRUD operations
- Add confirmation dialogs
- Error handling and user feedback

### Phase 4: Integration & Testing (Day 7)
- Connect file tree to editor (future)
- Test on all platforms (macOS, Windows, Linux)
- Performance optimization for large workspaces
- Polish UI/UX

## Success Criteria

**Functional**:
- ✅ Can select and open a workspace folder
- ✅ File tree displays all .md files recursively
- ✅ Can expand/collapse folders
- ✅ Can create new markdown file
- ✅ Can rename and delete files
- ✅ Workspace persists across app restarts
- ✅ Recent workspaces list works

**Performance**:
- ✅ File tree renders <500ms for 1000 files
- ✅ Folder selection dialog opens instantly
- ✅ No UI freeze during file operations

**UX**:
- ✅ Clear visual feedback for all operations
- ✅ Keyboard shortcuts work
- ✅ Context menu accessible via right-click
- ✅ Error messages are helpful

## Impact Assessment

### User Value: HIGH
- Core functionality required for any markdown editor
- Enables local-first workflow
- Foundation for all document operations

### Technical Complexity: MEDIUM
- Platform abstraction already built
- Standard tree view patterns
- Well-understood file operations

### Risk Level: LOW
- Using proven platform abstraction
- File operations are sandboxed by Tauri
- No network dependencies

## Future Enhancements (Post-Week 2)

1. **File Watching**: Auto-refresh when files change externally
2. **Drag & Drop**: Move files between folders
3. **Bulk Operations**: Select multiple files for batch operations
4. **Smart Search**: Full-text search within workspace
5. **Git Integration**: Show git status in file tree
6. **Favorites**: Pin frequently used files

## Related Features

From ZETTLR_FEATURE_PARITY.md:

- **Category 2: File Management** (9 CRITICAL features)
  - #27: Create/rename/delete files
  - #28: Drag-drop file organization
  - #29: Quick open file (Cmd+P)
  - #30: Recent files list
  - #31: File properties panel
  - #32: File search in workspace
  - #33: Multi-file operations
  - #34: Auto-save
  - #35: File recovery

**This proposal implements**: #27 (files), #29 (quick open), #30 (recent), #32 (search foundations)

## Resources Required

- **Development Time**: 5-7 days
- **Dependencies**: None (platform abstraction ready)
- **New Dependencies**: None (using existing Tauri plugins)
- **Documentation**: Component usage, API reference

---

## Approval

**Created**: 2025-10-15
**Status**: Ready for implementation
**Next Steps**: Create tasks.md and begin Phase 1
