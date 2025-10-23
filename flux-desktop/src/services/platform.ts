/**
 * Platform Abstraction Layer
 *
 * Provides a unified interface for desktop platform operations.
 * Currently implemented for Tauri, but designed to be framework-agnostic.
 *
 * This abstraction allows switching to Electron or other frameworks
 * with minimal changes to application code (~2 week effort vs 3 month rewrite).
 */

export interface FileSystemAPI {
  /**
   * Read a file from the local filesystem
   */
  readTextFile(path: string): Promise<string>;

  /**
   * Write text content to a file
   */
  writeTextFile(path: string, content: string): Promise<void>;

  /**
   * Check if a file or directory exists
   */
  exists(path: string): Promise<boolean>;

  /**
   * Delete a file
   */
  remove(path: string): Promise<void>;

  /**
   * Create a directory
   */
  createDir(path: string): Promise<void>;

  /**
   * List files in a directory
   */
  readDir(path: string): Promise<DirEntry[]>;

  /**
   * Get file metadata
   */
  stat(path: string): Promise<FileStats>;
}

export interface DialogAPI {
  /**
   * Show a file picker dialog
   */
  openFile(options: OpenFileOptions): Promise<string | null>;

  /**
   * Show a folder picker dialog
   */
  openFolder(options?: OpenFolderOptions): Promise<string | null>;

  /**
   * Show a save file dialog
   */
  saveFile(options: SaveFileOptions): Promise<string | null>;

  /**
   * Show a message dialog
   */
  message(message: string, options?: MessageOptions): Promise<void>;

  /**
   * Show a confirmation dialog
   */
  confirm(message: string, options?: ConfirmOptions): Promise<boolean>;
}

export interface ShellAPI {
  /**
   * Execute a shell command
   */
  execute(command: string, args?: string[]): Promise<ShellResult>;

  /**
   * Open a URL in the default browser
   */
  openUrl(url: string): Promise<void>;

  /**
   * Open a file with the default application
   */
  openPath(path: string): Promise<void>;
}

export interface WindowAPI {
  /**
   * Set the window title
   */
  setTitle(title: string): Promise<void>;

  /**
   * Minimize the window
   */
  minimize(): Promise<void>;

  /**
   * Maximize the window
   */
  maximize(): Promise<void>;

  /**
   * Toggle fullscreen mode
   */
  toggleFullscreen(): Promise<void>;

  /**
   * Close the window
   */
  close(): Promise<void>;
}

export interface AppAPI {
  /**
   * Get the app name
   */
  getName(): Promise<string>;

  /**
   * Get the app version
   */
  getVersion(): Promise<string>;

  /**
   * Get the platform (macos, windows, linux)
   */
  getPlatform(): Promise<string>;

  /**
   * Quit the application
   */
  quit(): Promise<void>;
}

/**
 * Complete desktop platform interface
 */
export interface DesktopPlatform {
  fs: FileSystemAPI;
  dialog: DialogAPI;
  shell: ShellAPI;
  window: WindowAPI;
  app: AppAPI;
}

// Type definitions
export interface DirEntry {
  name: string;
  path: string;
  isDirectory: boolean;
  isFile: boolean;
}

export interface FileStats {
  size: number;
  modified: Date;
  created: Date;
  accessed: Date;
  isDirectory: boolean;
  isFile: boolean;
}

export interface OpenFileOptions {
  title?: string;
  filters?: FileFilter[];
  defaultPath?: string;
  multiple?: boolean;
}

export interface OpenFolderOptions {
  title?: string;
  defaultPath?: string;
}

export interface SaveFileOptions {
  title?: string;
  filters?: FileFilter[];
  defaultPath?: string;
}

export interface FileFilter {
  name: string;
  extensions: string[];
}

export interface MessageOptions {
  title?: string;
  type?: 'info' | 'warning' | 'error';
}

export interface ConfirmOptions {
  title?: string;
  okLabel?: string;
  cancelLabel?: string;
}

export interface ShellResult {
  stdout: string;
  stderr: string;
  code: number;
}
