import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Switch } from './ui/switch';

interface Thought {
  id: string;
  content: string;
  type: string;
  energy: number;
  confidence: number;
  parent_ids: string[];
}

interface WorkspaceState {
  timestamp: string;
  workspace_id: string;
  thought_count: number;
  thoughts: Record<string, Thought>;
  dominant_thought_id: string | null;
}

interface StateLogEntry {
  phase: string;
  state: WorkspaceState;
  logged_at: string;
}

const ThoughtSeedDebugPanel: React.FC = () => {
  const [isWatching, setIsWatching] = useState(false);
  const [stateLogs, setStateLogs] = useState<StateLogEntry[]>([]);
  const [selectedWorkspace, setSelectedWorkspace] = useState<string>('');
  const [watchedWorkspaces, setWatchedWorkspaces] = useState<string[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Mock API endpoints - replace with actual backend calls
  const API_BASE = 'http://localhost:8001';

  const fetchWatchedWorkspaces = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/thoughtseed/watched`);
      const data = await response.json();
      setWatchedWorkspaces(data.workspaces || []);
    } catch (error) {
      console.error('Failed to fetch watched workspaces:', error);
    }
  };

  const fetchStateLogs = async (workspaceId?: string) => {
    try {
      const url = workspaceId
        ? `${API_BASE}/api/thoughtseed/logs/${workspaceId}`
        : `${API_BASE}/api/thoughtseed/logs`;

      const response = await fetch(url);
      const data = await response.json();
      setStateLogs(data.logs || []);
    } catch (error) {
      console.error('Failed to fetch state logs:', error);
    }
  };

  const toggleWatching = async (workspaceId: string) => {
    try {
      const method = isWatching ? 'DELETE' : 'POST';
      await fetch(`${API_BASE}/api/thoughtseed/watch/${workspaceId}`, {
        method,
        headers: { 'Content-Type': 'application/json' }
      });

      setIsWatching(!isWatching);
      await fetchWatchedWorkspaces();
    } catch (error) {
      console.error('Failed to toggle watching:', error);
    }
  };

  const clearLogs = async () => {
    try {
      await fetch(`${API_BASE}/api/thoughtseed/logs`, { method: 'DELETE' });
      setStateLogs([]);
    } catch (error) {
      console.error('Failed to clear logs:', error);
    }
  };

  // Auto-refresh logs
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchStateLogs(selectedWorkspace);
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, selectedWorkspace]);

  // Initial load
  useEffect(() => {
    fetchWatchedWorkspaces();
    fetchStateLogs();
  }, []);

  const renderThoughtCard = (thought: Thought, isDominant: boolean) => (
    <Card key={thought.id} className={`mb-2 ${isDominant ? 'border-green-500 bg-green-50' : ''}`}>
      <CardContent className="p-3">
        <div className="flex justify-between items-start mb-2">
          <span className="text-sm font-medium">{thought.content}</span>
          {isDominant && <Badge variant="default">Dominant</Badge>}
        </div>

        <div className="flex justify-between text-xs text-gray-600">
          <span>Type: {thought.type}</span>
          <span>Energy: {thought.energy.toFixed(2)}</span>
          <span>Confidence: {thought.confidence.toFixed(2)}</span>
        </div>

        {thought.parent_ids.length > 0 && (
          <div className="mt-1 text-xs text-gray-500">
            Parents: {thought.parent_ids.length}
          </div>
        )}
      </CardContent>
    </Card>
  );

  const renderWorkspaceState = (entry: StateLogEntry) => {
    const state = entry.state;
    const thoughts = Object.values(state.thoughts);

    return (
      <Card key={`${state.workspace_id}-${state.timestamp}-${entry.phase}`} className="mb-4">
        <CardHeader>
          <CardTitle className="text-sm flex justify-between">
            <span>Workspace: {state.workspace_id}</span>
            <Badge variant={entry.phase === 'pre_update' ? 'outline' : 'default'}>
              {entry.phase}
            </Badge>
          </CardTitle>
          <div className="text-xs text-gray-500">
            {new Date(state.timestamp).toLocaleTimeString()}
            ({state.thought_count} thoughts)
          </div>
        </CardHeader>

        <CardContent>
          <div className="space-y-2">
            {thoughts.map(thought =>
              renderThoughtCard(thought, thought.id === state.dominant_thought_id)
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="w-full h-full p-4 bg-gray-50">
      <div className="mb-4">
        <h2 className="text-xl font-bold mb-2">ThoughtSeed Workspace Debug Panel</h2>

        {/* Control Panel */}
        <Card className="mb-4">
          <CardHeader>
            <CardTitle className="text-lg">Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Switch
                  checked={autoRefresh}
                  onCheckedChange={setAutoRefresh}
                />
                <label className="text-sm">Auto-refresh (2s)</label>
              </div>

              <Button
                onClick={() => fetchStateLogs(selectedWorkspace)}
                variant="outline"
                size="sm"
              >
                Refresh Now
              </Button>

              <Button
                onClick={clearLogs}
                variant="destructive"
                size="sm"
              >
                Clear Logs
              </Button>
            </div>

            <div className="flex items-center space-x-2">
              <label className="text-sm">Watched Workspaces:</label>
              <div className="flex space-x-2">
                {watchedWorkspaces.length > 0 ? (
                  watchedWorkspaces.map(id => (
                    <Badge key={id} variant="default">{id}</Badge>
                  ))
                ) : (
                  <span className="text-sm text-gray-500">None</span>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* State Logs Display */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">
          Workspace States ({stateLogs.length} entries)
        </h3>

        {stateLogs.length === 0 ? (
          <Card>
            <CardContent className="p-8 text-center">
              <p className="text-gray-500">No state logs available</p>
              <p className="text-sm text-gray-400 mt-1">
                Enable watching on a workspace to see competition dynamics
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="max-h-96 overflow-y-auto space-y-4">
            {stateLogs.slice(-10).reverse().map(renderWorkspaceState)}
          </div>
        )}
      </div>
    </div>
  );
};

export default ThoughtSeedDebugPanel;