import React, { useState, useEffect } from 'react'
import { Brain, Play, Pause, RotateCcw, Target } from 'lucide-react'

interface ThoughtSeed {
  id: string
  content: string
  type: 'goal' | 'action' | 'belief' | 'perception'
  energy: number
  confidence: number
  dominance_score: number
  parent_ids: string[]
}

interface CompetitionCycle {
  cycle: number
  timestamp: string
  pre_update: {
    thoughts: Record<string, ThoughtSeed>
    dominant_thought_id: string
    consciousness_level: number
  }
  post_update: {
    thoughts: Record<string, ThoughtSeed>
    dominant_thought_id: string
    consciousness_level: number
  }
}

export default function ThoughtSeedMonitor() {
  const [isRunning, setIsRunning] = useState(false)
  const [currentCycle, setCurrentCycle] = useState<CompetitionCycle | null>(null)
  const [cycles, setCycles] = useState<CompetitionCycle[]>([])
  const [selectedThought, setSelectedThought] = useState<string | null>(null)

  // Mock data generation
  const generateMockThought = (id: string, content: string, type: ThoughtSeed['type']): ThoughtSeed => ({
    id,
    content,
    type,
    energy: Math.random() * 0.8 + 0.2,
    confidence: Math.random() * 0.6 + 0.4,
    dominance_score: Math.random(),
    parent_ids: []
  })

  const generateMockCycle = (): CompetitionCycle => {
    const timestamp = new Date().toISOString()
    const thoughts = {
      'thought_1': generateMockThought('thought_1', 'Analyze data systematically', 'action'),
      'thought_2': generateMockThought('thought_2', 'Use creative approaches', 'action'),
      'thought_3': generateMockThought('thought_3', 'Apply proven methods', 'action'),
      'thought_4': generateMockThought('thought_4', 'Focus on core objectives', 'goal'),
      'thought_5': generateMockThought('thought_5', 'Trust in established patterns', 'belief'),
    }

    const thoughtIds = Object.keys(thoughts)
    const dominantId = thoughtIds[Math.floor(Math.random() * thoughtIds.length)]
    thoughts[dominantId].energy = Math.max(thoughts[dominantId].energy, 0.8)

    const pre_update = {
      thoughts,
      dominant_thought_id: dominantId,
      consciousness_level: Math.random() * 0.8 + 0.2
    }

    // Simulate post-update changes
    const post_thoughts = { ...thoughts }
    Object.values(post_thoughts).forEach(thought => {
      thought.energy += (Math.random() - 0.5) * 0.2
      thought.energy = Math.max(0.1, Math.min(1.0, thought.energy))
    })

    const post_update = {
      thoughts: post_thoughts,
      dominant_thought_id: dominantId,
      consciousness_level: Math.random() * 0.8 + 0.2
    }

    return {
      cycle: cycles.length + 1,
      timestamp,
      pre_update,
      post_update
    }
  }

  const runCycle = () => {
    const newCycle = generateMockCycle()
    setCycles(prev => [...prev.slice(-9), newCycle])
    setCurrentCycle(newCycle)
  }

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    if (isRunning) {
      interval = setInterval(runCycle, 3000)
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isRunning, cycles.length])

  const getThoughtTypeColor = (type: ThoughtSeed['type']) => {
    switch (type) {
      case 'goal': return 'text-blue-400 bg-blue-900/30 border-blue-500/30'
      case 'action': return 'text-green-400 bg-green-900/30 border-green-500/30'
      case 'belief': return 'text-purple-400 bg-purple-900/30 border-purple-500/30'
      case 'perception': return 'text-yellow-400 bg-yellow-900/30 border-yellow-500/30'
      default: return 'text-gray-400 bg-gray-900/30 border-gray-500/30'
    }
  }

  const renderThoughtCard = (thought: ThoughtSeed, isDominant: boolean, phase: 'pre' | 'post') => (
    <div
      key={`${thought.id}-${phase}`}
      onClick={() => setSelectedThought(thought.id)}
      className={`p-4 rounded-lg border cursor-pointer transition-all ${
        isDominant
          ? 'border-yellow-400 bg-yellow-900/20 shadow-lg shadow-yellow-400/20'
          : selectedThought === thought.id
            ? 'border-blue-400 bg-blue-900/20'
            : 'border-gray-600 bg-gray-800/50 hover:border-gray-500'
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-sm font-medium text-white">{thought.content}</span>
        {isDominant && <span className="text-yellow-400 text-lg">👑</span>}
      </div>

      <div className="flex justify-between items-center mb-3">
        <span className={`text-xs px-2 py-1 rounded border ${getThoughtTypeColor(thought.type)}`}>
          {thought.type}
        </span>
        <span className="text-xs text-gray-400">
          Dom: {(thought.energy * thought.confidence).toFixed(2)}
        </span>
      </div>

      <div className="space-y-2">
        {/* Energy Bar */}
        <div className="flex items-center space-x-2 text-xs">
          <span className="text-gray-400 w-8">Energy</span>
          <div className="flex-1 bg-gray-700 rounded-full h-2">
            <div
              className="bg-green-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${thought.energy * 100}%` }}
            />
          </div>
          <span className="text-gray-300 w-12">{thought.energy.toFixed(2)}</span>
        </div>

        {/* Confidence Bar */}
        <div className="flex items-center space-x-2 text-xs">
          <span className="text-gray-400 w-8">Conf</span>
          <div className="flex-1 bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${thought.confidence * 100}%` }}
            />
          </div>
          <span className="text-gray-300 w-12">{thought.confidence.toFixed(2)}</span>
        </div>
      </div>
    </div>
  )

  return (
    <div className="min-h-screen bg-black text-white p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-3">
          <Brain className="h-8 w-8 text-blue-400" />
          <h1 className="text-2xl font-bold text-white">ThoughtSeed Monitor</h1>
        </div>

        <div className="flex items-center space-x-4">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center space-x-2 ${
              isRunning
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {isRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            <span>{isRunning ? 'Pause' : 'Start'} Competition</span>
          </button>

          <button
            onClick={() => {
              setCycles([])
              setCurrentCycle(null)
              setSelectedThought(null)
            }}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium flex items-center space-x-2"
          >
            <RotateCcw className="h-4 w-4" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Inner Screen Display */}
      <div className="mb-8">
        <div className="bg-gray-900/60 border border-blue-500/30 rounded-xl p-6">
          <div className="flex items-center space-x-3 mb-4">
            <Eye className="h-5 w-5 text-blue-400" />
            <h2 className="text-lg font-semibold text-white">Inner Screen</h2>
          </div>

          {!currentCycle ? (
            <div className="text-center py-12 text-gray-400">
              <Brain className="h-16 w-16 mx-auto mb-4 text-gray-600" />
              <p>Start competition to monitor inner workspace</p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Cycle Info */}
              <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
                <div>
                  <span className="text-white font-medium">Cycle {currentCycle.cycle}</span>
                  <span className="text-gray-400 text-sm ml-3">
                    {new Date(currentCycle.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-400">Consciousness Level</div>
                  <div className={`font-bold text-lg ${
                    currentCycle.post_update.consciousness_level > 0.7
                      ? 'text-green-400'
                      : currentCycle.post_update.consciousness_level > 0.4
                        ? 'text-yellow-400'
                        : 'text-red-400'
                  }`}>
                    {(currentCycle.post_update.consciousness_level * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Winning ThoughtSeed Display */}
              <div className="p-4 bg-gradient-to-r from-yellow-900/20 to-yellow-800/20 border border-yellow-400/30 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <Target className="h-5 w-5 text-yellow-400" />
                  <h3 className="font-semibold text-yellow-400">Dominant ThoughtSeed</h3>
                </div>
                {currentCycle.post_update.thoughts[currentCycle.post_update.dominant_thought_id] && (
                  <div className="text-white">
                    {currentCycle.post_update.thoughts[currentCycle.post_update.dominant_thought_id].content}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Competition States */}
      {currentCycle && (
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-white mb-6">ThoughtSeed Competition</h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Pre-Update State */}
            <div>
              <h3 className="text-lg font-medium text-gray-300 mb-4 flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
                <span>Pre-Update State</span>
              </h3>
              <div className="space-y-3">
                {Object.values(currentCycle.pre_update.thoughts).map(thought =>
                  renderThoughtCard(
                    thought,
                    thought.id === currentCycle.pre_update.dominant_thought_id,
                    'pre'
                  )
                )}
              </div>
            </div>

            {/* Post-Update State */}
            <div>
              <h3 className="text-lg font-medium text-gray-300 mb-4 flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                <span>Post-Update State</span>
              </h3>
              <div className="space-y-3">
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
        </div>
      )}

      {/* Cycle History */}
      {cycles.length > 1 && (
        <div className="mb-8">
          <h3 className="text-lg font-medium text-gray-300 mb-4">Recent Cycles</h3>
          <div className="flex space-x-2 overflow-x-auto pb-2">
            {cycles.slice(-10).map((cycle) => (
              <button
                key={cycle.cycle}
                onClick={() => setCurrentCycle(cycle)}
                className={`px-4 py-2 rounded text-sm whitespace-nowrap transition-all ${
                  currentCycle?.cycle === cycle.cycle
                    ? 'bg-blue-600 text-white'
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
        <div className="bg-gray-900/60 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-300 mb-4">ThoughtSeed Analysis</h3>
          <div className="text-sm text-gray-400">
            Selected: {selectedThought}
            <br />
            Click another thought to compare states across cycles
          </div>
        </div>
      )}
    </div>
  )
}