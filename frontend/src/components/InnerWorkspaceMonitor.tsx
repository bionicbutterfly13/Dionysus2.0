import React, { useState, useEffect, useRef } from 'react';
import { Brain, Zap, Eye, EyeOff, Play, Pause, RotateCcw } from 'lucide-react';

interface Thought {
  id: string;
  content: string;
  type: 'goal' | 'action' | 'belief' | 'perception';
  energy: number;
  confidence: number;
  parent_ids: string[];
  dominance_score?: number;
}

interface WorkspaceState {
  workspace_id: string;
  timestamp: string;
  thought_count: number;
  thoughts: Record<string, Thought>;
  dominant_thought_id: string | null;
  consciousness_level?: number;
}

interface CompetitionCycle {
  cycle: number;
  pre_update: WorkspaceState;
  post_update: WorkspaceState;
  timestamp: string;
}

const InnerWorkspaceMonitor: React.FC = () => {
  const [isWatching, setIsWatching] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [cycles, setCycles] = useState<CompetitionCycle[]>([]);
  const [currentCycle, setCurrentCycle] = useState<CompetitionCycle | null>(null);
  const [selectedThought, setSelectedThought] = useState<string | null>(null);
  const [workspaces, setWorkspaces] = useState<string[]>(['demo_workspace', 'researcher_workspace']);
  const [selectedWorkspace, setSelectedWorkspace] = useState('demo_workspace');
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Mock data generator for demonstration
  const generateMockThought = (id: string, content: string, type: Thought['type']): Thought => ({
    id,
    content,
    type,
    energy: Math.random() * 0.8 + 0.2,
    confidence: Math.random() * 0.6 + 0.4,
    parent_ids: [],
    dominance_score: Math.random()
  });

  const generateMockWorkspaceState = (timestamp: string): WorkspaceState => {
    const thoughts = {
      'thought_1': generateMockThought('thought_1', 'Analyze data systematically', 'action'),
      'thought_2': generateMockThought('thought_2', 'Use creative approaches', 'action'),
      'thought_3': generateMockThought('thought_3', 'Apply proven methods', 'action'),
    };

    // Make one dominant
    const thoughtIds = Object.keys(thoughts);
    const dominantId = thoughtIds[Math.floor(Math.random() * thoughtIds.length)];
    thoughts[dominantId].energy = Math.max(thoughts[dominantId].energy, 0.8);

    return {
      workspace_id: selectedWorkspace,
      timestamp,
      thought_count: Object.keys(thoughts).length,
      thoughts,
      dominant_thought_id: dominantId,
      consciousness_level: Math.random() * 0.8 + 0.2
    };
  };

  const runCompetitionCycle = () => {
    const timestamp = new Date().toISOString();
    const cycleNumber = cycles.length + 1;

    const pre_update = generateMockWorkspaceState(timestamp);

    // Simulate competition - slight changes in post-update
    const post_update = { ...pre_update };
    Object.values(post_update.thoughts).forEach(thought => {
      thought.energy += (Math.random() - 0.5) * 0.2;
      thought.energy = Math.max(0.1, Math.min(1.0, thought.energy));
    });

    const newCycle: CompetitionCycle = {
      cycle: cycleNumber,
      pre_update,
      post_update,
      timestamp
    };

    setCycles(prev => [...prev.slice(-9), newCycle]); // Keep last 10 cycles
    setCurrentCycle(newCycle);
  };

  const toggleWatching = () => {
    setIsWatching(!isWatching);
    if (!isWatching) {
      // Start watching - clear previous data
      setCycles([]);
      setCurrentCycle(null);
    }
  };

  const toggleCompetition = () => {
    if (isRunning) {
      // Stop
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setIsRunning(false);
    } else {
      // Start
      setIsRunning(true);
      intervalRef.current = setInterval(runCompetitionCycle, 2000);
    }
  };

  const resetWorkspace = () => {
    setCycles([]);
    setCurrentCycle(null);
    setSelectedThought(null);
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const getThoughtTypeColor = (type: Thought['type']) => {
    switch (type) {
      case 'goal': return 'text-blue-400 bg-blue-900/30';
      case 'action': return 'text-green-400 bg-green-900/30';
      case 'belief': return 'text-purple-400 bg-purple-900/30';
      case 'perception': return 'text-yellow-400 bg-yellow-900/30';
      default: return 'text-gray-400 bg-gray-900/30';
    }
  };

  const renderThoughtCard = (thought: Thought, isDominant: boolean, phase: 'pre' | 'post') => (
    <div
      key={`${thought.id}-${phase}`}
      onClick={() => setSelectedThought(thought.id)}
      className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
        isDominant
          ? 'border-gold-400 bg-yellow-900/20'
          : selectedThought === thought.id
            ? 'border-purple-400 bg-purple-900/20'
            : 'border-gray-600 bg-gray-800 hover:border-gray-500'
      }`}
    >
      <div className="flex items-start justify-between mb-2">
        <span className="text-sm font-medium text-white truncate pr-2">{thought.content}</span>
        {isDominant && <span className="text-xs text-yellow-400 font-bold">👑</span>}
      </div>

      <div className="flex justify-between items-center mb-2">
        <span className={`text-xs px-2 py-1 rounded ${getThoughtTypeColor(thought.type)}`}>
          {thought.type}
        </span>
        <span className="text-xs text-gray-400">
          D: {(thought.energy * thought.confidence).toFixed(2)}
        </span>
      </div>

      <div className="space-y-1">
        {/* Energy Bar */}
        <div className="flex items-center space-x-2 text-xs">
          <span className="text-gray-400 w-6">E:</span>
          <div className="flex-1 bg-gray-700 rounded-full h-2">
            <div
              className="bg-green-400 h-2 rounded-full transition-all"
              style={{ width: `${thought.energy * 100}%` }}
            />
          </div>
          <span className="text-gray-300 w-8">{thought.energy.toFixed(2)}</span>
        </div>

        {/* Confidence Bar */}
        <div className="flex items-center space-x-2 text-xs">
          <span className="text-gray-400 w-6">C:</span>
          <div className="flex-1 bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-400 h-2 rounded-full transition-all"
              style={{ width: `${thought.confidence * 100}%` }}
            />
          </div>
          <span className="text-gray-300 w-8">{thought.confidence.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="bg-gray-900 rounded-xl p-6 border border-gray-700 min-h-96">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Brain className="h-6 w-6 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">Inner Workspace Monitor</h3>
        </div>

        <div className="flex items-center space-x-2">
          <select
            value={selectedWorkspace}
            onChange={(e) => setSelectedWorkspace(e.target.value)}
            className="bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white"
          >
            {workspaces.map(ws => (
              <option key={ws} value={ws}>{ws}</option>
            ))}
          </select>

          <button
            onClick={toggleWatching}
            className={`px-3 py-1 rounded text-sm font-medium flex items-center space-x-1 ${
              isWatching ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300'
            }`}
          >
            {isWatching ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
            <span>{isWatching ? 'Watching' : 'Watch'}</span>
          </button>

          {isWatching && (
            <>
              <button
                onClick={toggleCompetition}
                className={`px-3 py-1 rounded text-sm font-medium flex items-center space-x-1 ${
                  isRunning ? 'bg-red-600 text-white' : 'bg-blue-600 text-white'
                }`}
              >
                {isRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                <span>{isRunning ? 'Pause' : 'Start'}</span>
              </button>

              <button
                onClick={resetWorkspace}
                className="px-3 py-1 rounded text-sm font-medium bg-gray-700 text-gray-300 flex items-center space-x-1"
              >
                <RotateCcw className="h-4 w-4" />
                <span>Reset</span>
              </button>
            </>
          )}
        </div>
      </div>

      {!isWatching ? (
        <div className="text-center py-12 text-gray-400">
          <Brain className="h-12 w-12 mx-auto mb-4 text-gray-600" />
          <p>Click "Watch" to monitor ThoughtSeed competition cycles</p>
          <p className="text-sm mt-2">Enable LogFire for enhanced observability</p>
        </div>
      ) : !currentCycle ? (
        <div className="text-center py-12 text-gray-400">
          <Zap className="h-12 w-12 mx-auto mb-4 text-gray-600 animate-pulse" />
          <p>Watching workspace: {selectedWorkspace}</p>
          <p className="text-sm mt-2">Click "Start" to begin competition simulation</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Cycle Info */}
          <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
            <div>
              <span className="text-white font-medium">Cycle {currentCycle.cycle}</span>
              <span className="text-gray-400 text-sm ml-3">
                {new Date(currentCycle.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-400">Consciousness Level</div>
              <div className={`font-bold ${
                (currentCycle.post_update.consciousness_level || 0) > 0.7
                  ? 'text-green-400'
                  : (currentCycle.post_update.consciousness_level || 0) > 0.4
                    ? 'text-yellow-400'
                    : 'text-red-400'
              }`}>
                {((currentCycle.post_update.consciousness_level || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Competition States */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Pre-Update */}
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-3 flex items-center space-x-2">
                <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                <span>Pre-Update State</span>
              </h4>
              <div className="space-y-2">
                {Object.values(currentCycle.pre_update.thoughts).map(thought =>
                  renderThoughtCard(
                    thought,
                    thought.id === currentCycle.pre_update.dominant_thought_id,
                    'pre'
                  )
                )}
              </div>
            </div>

            {/* Post-Update */}
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-3 flex items-center space-x-2">
                <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                <span>Post-Update State</span>
              </h4>
              <div className="space-y-2">
                {Object.values(currentCycle.post_update.thoughts).map(thought =>
                  renderThoughtCard(
                    thought,
                    thought.id === currentCycle.post_update.dominant_thought_id,
                    'post'
                  )
                )}
              </div>
            </div>
          </div>

          {/* Cycle History */}
          {cycles.length > 1 && (
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-3">Recent Cycles</h4>
              <div className="flex space-x-2 overflow-x-auto pb-2">
                {cycles.slice(-8).map((cycle) => (
                  <button
                    key={cycle.cycle}
                    onClick={() => setCurrentCycle(cycle)}
                    className={`px-3 py-2 rounded text-xs whitespace-nowrap ${
                      currentCycle?.cycle === cycle.cycle
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    Cycle {cycle.cycle}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Selected Thought Details */}
          {selectedThought && currentCycle && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-300 mb-3">Thought Details</h4>
              {/* Add detailed thought analysis here */}
              <div className="text-sm text-gray-400">
                Selected: {selectedThought}
                <br />
                Click another thought to compare states across cycles
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default InnerWorkspaceMonitor;