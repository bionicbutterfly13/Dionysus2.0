/**
 * React Hook for Platform Access
 *
 * Provides easy access to platform APIs throughout the React application.
 */

import { platform } from '../services/tauri-platform';
import type { DesktopPlatform } from '../services/platform';

/**
 * Hook to access platform APIs
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const platform = usePlatform();
 *
 *   const handleOpenFile = async () => {
 *     const path = await platform.dialog.openFile({
 *       title: 'Select a markdown file',
 *       filters: [{ name: 'Markdown', extensions: ['md'] }]
 *     });
 *
 *     if (path) {
 *       const content = await platform.fs.readTextFile(path);
 *       console.log(content);
 *     }
 *   };
 *
 *   return <button onClick={handleOpenFile}>Open File</button>;
 * }
 * ```
 */
export function usePlatform(): DesktopPlatform {
  return platform;
}
