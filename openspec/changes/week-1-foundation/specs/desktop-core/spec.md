# Desktop Core Capability

**Capability**: desktop-core
**Version**: 1.0.0
**Status**: New (being added)
**Owner**: Flux Desktop Team

---

## Overview

The `desktop-core` capability provides foundational desktop application functionality for Flux, enabling native OS integration, file system operations, and workspace management. This capability transforms Flux from a web application to a native desktop experience.

---

## ADDED Requirements

### Requirement: Desktop Application Initialization
**ID**: DC-001
**Priority**: CRITICAL
**The system SHALL run as a native desktop application using Tauri 2.0.**

#### Scenario: First launch
- **GIVEN** user has installed Flux desktop app
- **WHEN** user double-clicks Flux.app (macOS) or Flux.exe (Windows)
- **THEN** application window opens in <0.5 seconds
- **AND** window displays Flux dashboard
- **AND** bundle size is <15MB
- **AND** memory usage is <150MB on startup

#### Scenario: Subsequent launches
- **GIVEN** user has used Flux before
- **WHEN** user launches app
- **THEN** app opens to last workspace
- **AND** restores previous window size and position
- **AND** startup time remains <0.5 seconds

---

### Requirement: File System Access
**ID**: DC-002
**Priority**: CRITICAL
**The system SHALL read and write local markdown files with proper error handling.**

#### Scenario: Read markdown file
- **GIVEN** user has selected a workspace folder
- **WHEN** user clicks on a .md file in the file tree
- **THEN** system reads file content from disk
- **AND** displays content in editor within 100ms
- **AND** handles read errors gracefully (shows error message, doesn't crash)

#### Scenario: Write markdown file
- **GIVEN** user has edited a file in the editor
- **WHEN** user saves (Cmd+S / Ctrl+S) or auto-save triggers
- **THEN** system writes content to disk
- **AND** confirms save with visual indicator
- **AND** handles write errors gracefully (shows error, allows retry)

#### Scenario: File not found
- **GIVEN** a file has been deleted outside the app
- **WHEN** app tries to read the file
- **THEN** system detects file is missing
- **AND** shows "File not found" error
- **AND** removes file from file tree
- **AND** closes editor tab if open

---

### Requirement: Workspace Management
**ID**: DC-003
**Priority**: CRITICAL
**The system SHALL manage local folder workspaces with persistence.**

#### Scenario: Open workspace
- **GIVEN** user wants to work with local markdown files
- **WHEN** user clicks "Open Workspace" in menu
- **THEN** system displays native folder picker dialog
- **AND** user selects a folder
- **AND** system loads all .md files from folder (recursively)
- **AND** displays file tree in sidebar
- **AND** persists workspace path for next session

#### Scenario: Switch workspace
- **GIVEN** user has workspace A open
- **WHEN** user selects "Open Workspace" and chooses folder B
- **THEN** system saves any unsaved files in workspace A
- **AND** closes workspace A
- **AND** loads workspace B
- **AND** updates recent workspaces list

#### Scenario: Recent workspaces
- **GIVEN** user has opened multiple workspaces
- **WHEN** user clicks "File → Recent Workspaces"
- **THEN** system displays last 10 workspaces
- **AND** clicking a workspace opens it
- **AND** unavailable workspaces are grayed out

---

### Requirement: File System Watching
**ID**: DC-004
**Priority**: HIGH
**The system SHALL detect external file changes and update the UI.**

#### Scenario: File modified externally
- **GIVEN** user has a file open in Flux
- **WHEN** file is modified by another application
- **THEN** Flux detects the change within 1 second
- **AND** shows "File changed externally" notification
- **AND** offers options: "Reload", "Keep Current", "Compare"

#### Scenario: File created externally
- **GIVEN** workspace is open in Flux
- **WHEN** new .md file is created in workspace folder
- **THEN** Flux detects new file within 1 second
- **AND** adds file to file tree
- **AND** sorts file tree alphabetically

#### Scenario: File deleted externally
- **GIVEN** file is open in Flux editor
- **WHEN** file is deleted by another application
- **THEN** Flux detects deletion within 1 second
- **AND** shows "File deleted externally" warning
- **AND** keeps editor content in memory
- **AND** offers "Save As" option

---

### Requirement: Native Dialogs
**ID**: DC-005
**Priority**: HIGH
**The system SHALL use native OS dialogs for file operations.**

#### Scenario: Open file dialog
- **GIVEN** user wants to open a specific file
- **WHEN** user clicks "File → Open File"
- **THEN** system displays native file picker
- **AND** filters to show only .md files
- **AND** respects OS-specific dialog appearance (macOS vs Windows)

#### Scenario: Save file dialog
- **GIVEN** user has created new untitled document
- **WHEN** user saves for first time
- **THEN** system displays native save dialog
- **AND** defaults to .md extension
- **AND** pre-fills with "Untitled.md"
- **AND** allows user to choose location

---

### Requirement: Application Menus
**ID**: DC-006
**Priority**: HIGH
**The system SHALL provide native application menus.**

#### Scenario: File menu
- **GIVEN** app is running
- **WHEN** user opens File menu
- **THEN** menu shows:
  - New File (Cmd+N)
  - Open File (Cmd+O)
  - Open Workspace (Cmd+Shift+O)
  - Save (Cmd+S)
  - Save As (Cmd+Shift+S)
  - Recent Workspaces →
  - Close Window (Cmd+W)

#### Scenario: Edit menu
- **GIVEN** app is running
- **WHEN** user opens Edit menu
- **THEN** menu shows standard edit operations:
  - Undo (Cmd+Z)
  - Redo (Cmd+Shift+Z)
  - Cut (Cmd+X)
  - Copy (Cmd+C)
  - Paste (Cmd+V)
  - Select All (Cmd+A)

---

### Requirement: Window Management
**ID**: DC-007
**Priority**: MEDIUM
**The system SHALL manage application windows.**

#### Scenario: Minimize window
- **GIVEN** app window is open
- **WHEN** user clicks minimize button
- **THEN** window minimizes to dock/taskbar
- **AND** app continues running in background

#### Scenario: Fullscreen mode
- **GIVEN** app window is in normal mode
- **WHEN** user clicks fullscreen button (macOS) or presses F11 (Windows)
- **THEN** app enters fullscreen mode
- **AND** hides OS menubar/taskbar
- **AND** pressing Esc exits fullscreen

#### Scenario: Window state persistence
- **GIVEN** user has resized and positioned window
- **WHEN** user closes and reopens app
- **THEN** window restores to previous size
- **AND** window restores to previous position
- **AND** fullscreen state is restored if was fullscreen

---

### Requirement: Auto-Save
**ID**: DC-008
**Priority**: HIGH
**The system SHALL automatically save file changes.**

#### Scenario: Auto-save on edit
- **GIVEN** user is editing a file
- **WHEN** 3 seconds pass without typing
- **THEN** system automatically saves file
- **AND** shows subtle save indicator (fades in/out)
- **AND** doesn't interrupt typing

#### Scenario: Save on app quit
- **GIVEN** user has unsaved changes
- **WHEN** user closes app window
- **THEN** system saves all modified files
- **AND** closes gracefully
- **AND** doesn't show "unsaved changes" warning

---

### Requirement: Error Handling
**ID**: DC-009
**Priority**: HIGH
**The system SHALL handle file system errors gracefully.**

#### Scenario: Permission denied
- **GIVEN** user tries to save to protected folder
- **WHEN** save operation fails due to permissions
- **THEN** system shows "Permission denied" error
- **AND** offers to save to different location
- **AND** doesn't crash or lose data

#### Scenario: Disk full
- **GIVEN** user saves file to nearly full disk
- **WHEN** save operation fails due to disk space
- **THEN** system shows "Disk full" error
- **AND** keeps content in memory
- **AND** allows retry after user frees space

---

### Requirement: Platform Abstraction
**ID**: DC-010
**Priority**: MEDIUM
**The system SHALL abstract desktop platform operations.**

#### Scenario: Switch desktop framework
- **GIVEN** project needs to migrate from Tauri to Electron
- **WHEN** developers implement ElectronPlatform adapter
- **THEN** all file operations work without changing UI code
- **AND** migration takes ~2 weeks instead of 3 months
- **AND** existing React components need no changes

---

## Non-Functional Requirements

### Performance
- **Startup Time**: <0.5 seconds from launch to UI visible
- **File Read**: <100ms for files up to 1MB
- **File Save**: <50ms for typical documents
- **File Watch**: Detect changes within 1 second
- **Memory**: <150MB base, <300MB with large workspace

### Reliability
- **Data Safety**: Never lose user data, even on crash
- **Auto-Recovery**: Restore unsaved work after crash
- **Graceful Degradation**: Show errors, don't crash

### Security
- **File Access**: Only access files user explicitly grants
- **Sandboxing**: Respect OS security boundaries
- **No Network**: File operations don't require internet

### Usability
- **Native Feel**: App behaves like native OS application
- **Keyboard Shortcuts**: Standard OS shortcuts work
- **Accessibility**: Dialogs are screen-reader accessible

---

## Open Questions

1. **File Encoding**: UTF-8 only or support other encodings?
   → **Decision Needed**: Week 1 Day 3

2. **File Size Limit**: Max file size to prevent UI freeze?
   → **Decision Needed**: Week 1 Day 4

3. **Hidden Files**: Show .dotfiles in file tree by default?
   → **Decision Needed**: Week 1 Day 3

---

## Validation

This specification can be validated through:
- ✅ All requirements have clear scenarios
- ✅ All scenarios follow WHEN/THEN format
- ✅ Success criteria are measurable
- ✅ Non-functional requirements quantified
- ✅ Open questions identified for resolution
