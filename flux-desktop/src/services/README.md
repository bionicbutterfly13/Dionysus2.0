# Platform Abstraction Layer

This directory contains the platform abstraction layer for Flux Desktop.

## Overview

The platform abstraction provides a unified interface for desktop operations that works across different frameworks (Tauri, Electron, etc.).

## Architecture

```
platform.ts          ← Interface definitions (framework-agnostic)
tauri-platform.ts    ← Tauri implementation
electron-platform.ts ← Future: Electron implementation (if needed)
```

## Usage

### Basic Usage

```typescript
import { platform } from './services/tauri-platform';

// Read a file
const content = await platform.fs.readTextFile('/path/to/file.md');

// Open file dialog
const path = await platform.dialog.openFile({
  title: 'Select a markdown file',
  filters: [{ name: 'Markdown', extensions: ['md'] }]
});

// Write a file
await platform.fs.writeTextFile('/path/to/output.md', 'Hello, World!');
```

### React Hook

```tsx
import { usePlatform } from '../hooks/usePlatform';

function DocumentUpload() {
  const platform = usePlatform();

  const handleSelectFolder = async () => {
    const path = await platform.dialog.openFolder({
      title: 'Select workspace folder'
    });

    if (path) {
      const files = await platform.fs.readDir(path);
      console.log('Found files:', files);
    }
  };

  return <button onClick={handleSelectFolder}>Select Folder</button>;
}
```

## API Reference

### FileSystem API

- `readTextFile(path)` - Read text file contents
- `writeTextFile(path, content)` - Write text to file
- `exists(path)` - Check if file/directory exists
- `remove(path)` - Delete file
- `createDir(path)` - Create directory
- `readDir(path)` - List directory contents
- `stat(path)` - Get file metadata

### Dialog API

- `openFile(options)` - Show file picker dialog
- `openFolder(options)` - Show folder picker dialog
- `saveFile(options)` - Show save file dialog
- `message(message, options)` - Show message dialog
- `confirm(message, options)` - Show confirmation dialog

### Shell API

- `execute(command, args)` - Execute shell command
- `openUrl(url)` - Open URL in default browser
- `openPath(path)` - Open file with default application

### Window API

- `setTitle(title)` - Set window title
- `minimize()` - Minimize window
- `maximize()` - Maximize/restore window
- `toggleFullscreen()` - Toggle fullscreen mode
- `close()` - Close window

### App API

- `getName()` - Get application name
- `getVersion()` - Get application version
- `getPlatform()` - Get platform (macos, windows, linux)
- `quit()` - Quit application

## Switching Frameworks

To switch from Tauri to Electron (or another framework):

1. Create `electron-platform.ts` implementing the `DesktopPlatform` interface
2. Update imports to use the new platform:
   ```typescript
   // Before
   import { platform } from './services/tauri-platform';

   // After
   import { platform } from './services/electron-platform';
   ```
3. All application code continues to work unchanged

**Estimated effort**: ~2 weeks vs 3 months for a complete rewrite.

## Design Principles

1. **Framework Agnostic** - Interface doesn't expose framework-specific details
2. **Type Safe** - Full TypeScript support with detailed types
3. **Promise Based** - All operations are async for consistency
4. **Simple Migration** - Switching frameworks requires minimal code changes
5. **Future Proof** - Easy to add new platform capabilities
