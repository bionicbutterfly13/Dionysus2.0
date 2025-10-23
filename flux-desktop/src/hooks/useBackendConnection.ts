/**
 * Backend Connection Hook
 *
 * Monitors backend connectivity and provides connection status.
 */

import { useState, useEffect } from 'react';
import { checkBackendHealth } from '../services/api';

export interface BackendConnection {
  isConnected: boolean;
  isChecking: boolean;
  lastCheck: Date | null;
  error: string | null;
}

/**
 * Hook to monitor backend connection status
 *
 * @example
 * ```tsx
 * function Dashboard() {
 *   const backend = useBackendConnection();
 *
 *   return (
 *     <div>
 *       {backend.isConnected ? (
 *         <span>✅ Connected to backend</span>
 *       ) : (
 *         <span>⚠️ Backend offline</span>
 *       )}
 *     </div>
 *   );
 * }
 * ```
 */
export function useBackendConnection(checkInterval: number = 30000): BackendConnection {
  const [isConnected, setIsConnected] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

  const checkConnection = async () => {
    setIsChecking(true);
    setError(null);

    try {
      const healthy = await checkBackendHealth();
      setIsConnected(healthy);
      setLastCheck(new Date());

      if (!healthy) {
        setError('Backend health check failed');
      }
    } catch (err) {
      setIsConnected(false);
      setError(err instanceof Error ? err.message : 'Connection error');
      setLastCheck(new Date());
    } finally {
      setIsChecking(false);
    }
  };

  // Check on mount
  useEffect(() => {
    checkConnection();
  }, []);

  // Check periodically
  useEffect(() => {
    const interval = setInterval(checkConnection, checkInterval);
    return () => clearInterval(interval);
  }, [checkInterval]);

  return {
    isConnected,
    isChecking,
    lastCheck,
    error,
  };
}
