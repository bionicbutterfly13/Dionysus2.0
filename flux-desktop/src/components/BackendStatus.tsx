/**
 * Backend Connection Status Indicator
 *
 * Shows real-time connection status to the Python backend.
 */

import { useBackendConnection } from '../hooks/useBackendConnection';

export function BackendStatus() {
  const backend = useBackendConnection(30000); // Check every 30 seconds

  return (
    <div className="flex items-center gap-2 text-sm">
      {backend.isChecking ? (
        <span className="flex items-center gap-1 text-yellow-400">
          <div className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
          Checking backend...
        </span>
      ) : backend.isConnected ? (
        <span className="flex items-center gap-1 text-green-400">
          <div className="h-2 w-2 rounded-full bg-green-400" />
          Backend connected
        </span>
      ) : (
        <span className="flex items-center gap-1 text-red-400">
          <div className="h-2 w-2 rounded-full bg-red-400" />
          Backend offline
          {backend.error && (
            <span className="text-xs text-gray-400">({backend.error})</span>
          )}
        </span>
      )}

      {backend.lastCheck && (
        <span className="text-xs text-gray-500">
          Last check: {backend.lastCheck.toLocaleTimeString()}
        </span>
      )}
    </div>
  );
}
