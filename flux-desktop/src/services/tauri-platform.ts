/**
 * Tauri Platform Implementation
 *
 * Concrete implementation of the DesktopPlatform interface using Tauri APIs.
 */

import {
  readTextFile as tauriReadTextFile,
  writeTextFile as tauriWriteTextFile,
  exists as tauriExists,
  remove as tauriRemove,
  create as tauriCreateDir,
  readDir as tauriReadDir,
  stat as tauriStat,
} from '@tauri-apps/plugin-fs';

import {
  open as tauriOpen,
  save as tauriSave,
  message as tauriMessage,
  confirm as tauriConfirm,
} from '@tauri-apps/plugin-dialog';

import { Command } from '@tauri-apps/plugin-shell';
import { open as openUrl } from '@tauri-apps/plugin-opener';

import type {
  DesktopPlatform,
  FileSystemAPI,
  DialogAPI,
  ShellAPI,
  WindowAPI,
  AppAPI,
  DirEntry,
  FileStats,
  OpenFileOptions,
  OpenFolderOptions,
  SaveFileOptions,
  MessageOptions,
  ConfirmOptions,
  ShellResult,
} from './platform';

class TauriFileSystem implements FileSystemAPI {
  async readTextFile(path: string): Promise<string> {
    return await tauriReadTextFile(path);
  }

  async writeTextFile(path: string, content: string): Promise<void> {
    await tauriWriteTextFile(path, content);
  }

  async exists(path: string): Promise<boolean> {
    return await tauriExists(path);
  }

  async remove(path: string): Promise<void> {
    await tauriRemove(path);
  }

  async createDir(path: string): Promise<void> {
    await tauriCreateDir(path);
  }

  async readDir(path: string): Promise<DirEntry[]> {
    const entries = await tauriReadDir(path);
    return entries.map((entry) => ({
      name: entry.name,
      path: `${path}/${entry.name}`,
      isDirectory: entry.isDirectory,
      isFile: entry.isFile,
    }));
  }

  async stat(path: string): Promise<FileStats> {
    const stats = await tauriStat(path);
    return {
      size: stats.size,
      modified: new Date(stats.mtime || 0),
      created: new Date(stats.birthtime || 0),
      accessed: new Date(stats.atime || 0),
      isDirectory: stats.isDirectory,
      isFile: stats.isFile,
    };
  }
}

class TauriDialog implements DialogAPI {
  async openFile(options: OpenFileOptions): Promise<string | null> {
    const result = await tauriOpen({
      title: options.title,
      filters: options.filters,
      defaultPath: options.defaultPath,
      multiple: options.multiple || false,
    });

    // Handle both single file and multiple files
    if (result === null) return null;
    if (Array.isArray(result)) return result[0] || null;
    return result;
  }

  async openFolder(options?: OpenFolderOptions): Promise<string | null> {
    const result = await tauriOpen({
      title: options?.title,
      defaultPath: options?.defaultPath,
      directory: true,
    });

    if (result === null) return null;
    if (Array.isArray(result)) return result[0] || null;
    return result;
  }

  async saveFile(options: SaveFileOptions): Promise<string | null> {
    return await tauriSave({
      title: options.title,
      filters: options.filters,
      defaultPath: options.defaultPath,
    });
  }

  async message(message: string, options?: MessageOptions): Promise<void> {
    await tauriMessage(message, {
      title: options?.title,
      kind: options?.type,
    });
  }

  async confirm(message: string, options?: ConfirmOptions): Promise<boolean> {
    return await tauriConfirm(message, {
      title: options?.title,
      okLabel: options?.okLabel,
      cancelLabel: options?.cancelLabel,
    });
  }
}

class TauriShell implements ShellAPI {
  async execute(command: string, args?: string[]): Promise<ShellResult> {
    const cmd = Command.create(command, args || []);
    const output = await cmd.execute();

    return {
      stdout: output.stdout,
      stderr: output.stderr,
      code: output.code,
    };
  }

  async openUrl(url: string): Promise<void> {
    await openUrl(url);
  }

  async openPath(path: string): Promise<void> {
    await openUrl(path);
  }
}

class TauriWindow implements WindowAPI {
  private async getCurrentWindow() {
    const { getCurrentWindow } = await import('@tauri-apps/api/window');
    return getCurrentWindow();
  }

  async setTitle(title: string): Promise<void> {
    const window = await this.getCurrentWindow();
    await window.setTitle(title);
  }

  async minimize(): Promise<void> {
    const window = await this.getCurrentWindow();
    await window.minimize();
  }

  async maximize(): Promise<void> {
    const window = await this.getCurrentWindow();
    await window.toggleMaximize();
  }

  async toggleFullscreen(): Promise<void> {
    const window = await this.getCurrentWindow();
    const isFullscreen = await window.isFullscreen();
    await window.setFullscreen(!isFullscreen);
  }

  async close(): Promise<void> {
    const window = await this.getCurrentWindow();
    await window.close();
  }
}

class TauriApp implements AppAPI {
  async getName(): Promise<string> {
    const { getName } = await import('@tauri-apps/api/app');
    return await getName();
  }

  async getVersion(): Promise<string> {
    const { getVersion } = await import('@tauri-apps/api/app');
    return await getVersion();
  }

  async getPlatform(): Promise<string> {
    const { platform } = await import('@tauri-apps/plugin-os');
    return platform();
  }

  async quit(): Promise<void> {
    const { exit } = await import('@tauri-apps/plugin-process');
    await exit(0);
  }
}

/**
 * Tauri platform implementation
 *
 * Use this as the concrete implementation of DesktopPlatform for Tauri apps.
 *
 * @example
 * ```ts
 * import { platform } from './services/tauri-platform';
 *
 * // Read a file
 * const content = await platform.fs.readTextFile('/path/to/file.txt');
 *
 * // Open a file dialog
 * const path = await platform.dialog.openFile({
 *   title: 'Select a markdown file',
 *   filters: [{ name: 'Markdown', extensions: ['md'] }]
 * });
 * ```
 */
export const tauriPlatform: DesktopPlatform = {
  fs: new TauriFileSystem(),
  dialog: new TauriDialog(),
  shell: new TauriShell(),
  window: new TauriWindow(),
  app: new TauriApp(),
};

// Export as default platform for convenience
export const platform = tauriPlatform;
