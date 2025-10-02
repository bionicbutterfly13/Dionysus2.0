import React, { useState, useEffect } from 'react'
import { Brain, Play, Pause, RotateCcw, Target, Eye, Cpu, Zap, Activity } from 'lucide-react'

interface ThoughtSeed {
  id: string
  content: string
  type: 'analytical' | 'creative' | 'synthetic' | 'critical' | 'integrative' | 'metacognitive'
  energy: number
  confidence: number
  dominance_score: number
  parent_ids: string[]
}

interface MACAnalysis {
  type: string
  q_value: number
  metacognitive_error: number
  can_detect_suboptimal: boolean
  error_magnitude: number
}

interface IWMTMetrics {
  spatial_coherence: number
  temporal_coherence: number
  causal_coherence: number
  embodied_selfhood: number
  counterfactual_capacity: number
  overall_consciousness: number
}

interface ConsciousnessState {
  success: boolean
  consciousness_level: string
  iwmt_consciousness: boolean
  thoughtseed_winner: string | null
  mac_analysis: MACAnalysis[]
  processing_time: number
  iwmt_metrics: IWMTMetrics
  pipeline_summary: any
  error?: string
}

interface CompetitionCycle {
  cycle: number
  timestamp: string
  consciousness_state: ConsciousnessState | null
  thoughts: Record<string, ThoughtSeed>
  dominant_thought_id: string
  consciousness_level: number
}

export default function ThoughtSeedMonitor() {
  const [isRunning, setIsRunning] = useState(false)
  const [currentCycle, setCurrentCycle] = useState<CompetitionCycle | null>(null)
  const [cycles, setCycles] = useState<CompetitionCycle[]>([])
  const [selectedThought, setSelectedThought] = useState<string | null>(null)
  const [consciousnessStatus, setConsciousnessStatus] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // API endpoints (using proxy through Vite dev server)
  const API_BASE = '/api/consciousness'

  // Fetch consciousness status
  const fetchConsciousnessStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/status`)
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching consciousness status:', error)
      return null
    }
  }

  // Process consciousness with sample content
  const processConsciousness = async () => {
    try {
      const response = await fetch(`${API_BASE}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: "Research on consciousness, active inference, and thoughtseed competition dynamics reveals emergent patterns in neural architecture discovery. The integration of MAC theory with IWMT provides comprehensive understanding of consciousness levels through spatial, temporal, and causal coherence analysis.",
          filename: "consciousness_research.md",
          agents_requested: ["analytical_agent", "creative_agent", "pattern_agent", "synthesis_agent", "metacognitive_agent"]
        })
      })
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error processing consciousness:', error)
      return null
    }
  }

  // Generate thoughts from consciousness response
  const generateThoughtsFromConsciousness = (consciousnessData: any): Record<string, ThoughtSeed> => {
    const thoughts: Record<string, ThoughtSeed> = {}
    
    if (consciousnessData?.mac_analysis) {
      consciousnessData.mac_analysis.forEach((analysis: MACAnalysis, index: number) => {
        const id = `thought_${index + 1}`
        thoughts[id] = {
          id,
          content: `${analysis.type}: Q-value ${analysis.q_value.toFixed(3)}, Error detection: ${analysis.can_detect_suboptimal ? 'Yes' : 'No'}`,
          type: analysis.type as ThoughtSeed['type'],
          energy: Math.min(1.0, analysis.q_value),
          confidence: 1.0 - analysis.metacognitive_error,
          dominance_score: analysis.q_value * (1.0 - analysis.metacognitive_error),
          parent_ids: []
        }
      })
    }

    // Fallback thoughts if no MAC analysis
    if (Object.keys(thoughts).length === 0) {
      thoughts['fallback_1'] = {
        id: 'fallback_1',
        content: 'Processing consciousness pipeline...',
        type: 'analytical',
        energy: 0.5,
        confidence: 0.6,
        dominance_score: 0.3,
        parent_ids: []
      }
    }

    return thoughts
  }

  const generateRealCycle = async (): Promise<CompetitionCycle | null> => {
    setIsLoading(true)
    setError(null)
    
    try {
      const consciousnessData = await processConsciousness()
      
      if (!consciousnessData || !consciousnessData.success) {
        throw new Error(consciousnessData?.error || 'Failed to process consciousness')
      }

      const timestamp = new Date().toISOString()
      const thoughts = generateThoughtsFromConsciousness(consciousnessData)
      const thoughtIds = Object.keys(thoughts)
      const dominantId = consciousnessData.thoughtseed_winner || thoughtIds[0]
      
      // Calculate consciousness level from IWMT metrics
      const iwmtMetrics = consciousnessData.iwmt_metrics || {}
      const consciousness_level = iwmtMetrics.overall_consciousness || 0.5

      const consciousness_state: ConsciousnessState = {
        success: consciousnessData.success,
        consciousness_level: consciousnessData.consciousness_level,
        iwmt_consciousness: consciousnessData.iwmt_consciousness,
        thoughtseed_winner: consciousnessData.thoughtseed_winner,
        mac_analysis: consciousnessData.mac_analysis,
        processing_time: consciousnessData.processing_time,
        iwmt_metrics: consciousnessData.iwmt_metrics,
        pipeline_summary: consciousnessData.pipeline_summary
      }

      return {
        cycle: cycles.length + 1,
        timestamp,
        consciousness_state,
        thoughts,
        dominant_thought_id: dominantId,
        consciousness_level
      }
    } catch (error) {
      console.error('Error generating real cycle:', error)
      setError(error instanceof Error ? error.message : 'Unknown error')
      return null
    } finally {
      setIsLoading(false)
    }
  }

  const runCycle = async () => {
    const newCycle = await generateRealCycle()
    if (newCycle) {
      setCycles(prev => [...prev.slice(-9), newCycle])
      setCurrentCycle(newCycle)
    }
  }

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    if (isRunning) {
      interval = setInterval(() => {
        runCycle()
      }, 5000) // Slightly longer interval for real processing
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isRunning, cycles.length])

  // Fetch consciousness status on component mount
  useEffect(() => {
    const loadConsciousnessStatus = async () => {
      const status = await fetchConsciousnessStatus()
      setConsciousnessStatus(status)
    }
    loadConsciousnessStatus()
  }, [])

  const getThoughtTypeColor = (type: ThoughtSeed['type']) => {
    switch (type) {
      case 'analytical': return 'text-blue-400 bg-blue-900/30 border-blue-500/30'
      case 'creative': return 'text-green-400 bg-green-900/30 border-green-500/30'
      case 'synthetic': return 'text-purple-400 bg-purple-900/30 border-purple-500/30'
      case 'critical': return 'text-red-400 bg-red-900/30 border-red-500/30'
      case 'integrative': return 'text-yellow-400 bg-yellow-900/30 border-yellow-500/30'
      case 'metacognitive': return 'text-indigo-400 bg-indigo-900/30 border-indigo-500/30'
      default: return 'text-gray-400 bg-gray-900/30 border-gray-500/30'
    }
  }

  const renderThoughtCard = (thought: ThoughtSeed, isDominant: boolean) => (
    <div
      key={thought.id}
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
          {/* Consciousness Status */}
          {consciousnessStatus && (
            <div className="text-sm">
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                consciousnessStatus.available 
                  ? 'bg-green-900/30 text-green-400 border border-green-500/30'
                  : 'bg-red-900/30 text-red-400 border border-red-500/30'
              }`}>
                {consciousnessStatus.available ? 'Consciousness Online' : 'Consciousness Offline'}
              </span>
            </div>
          )}

          <button
            onClick={runCycle}
            disabled={isLoading}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg font-medium flex items-center space-x-2"
          >
            <Brain className="h-4 w-4" />
            <span>{isLoading ? 'Processing...' : 'Run Cycle'}</span>
          </button>

          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center space-x-2 ${
              isRunning
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {isRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            <span>{isRunning ? 'Pause' : 'Start'} Auto</span>
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
                    currentCycle.consciousness_level > 0.7
                      ? 'text-green-400'
                      : currentCycle.consciousness_level > 0.4
                        ? 'text-yellow-400'
                        : 'text-red-400'
                  }`}>
                    {(currentCycle.consciousness_level * 100).toFixed(1)}%
                  </div>
                  {currentCycle.consciousness_state?.iwmt_consciousness && (
                    <div className="text-xs text-green-300">IWMT Conscious</div>
                  )}
                </div>
              </div>

              {/* Winning ThoughtSeed Display */}
              <div className="p-4 bg-gradient-to-r from-yellow-900/20 to-yellow-800/20 border border-yellow-400/30 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <Target className="h-5 w-5 text-yellow-400" />
                  <h3 className="font-semibold text-yellow-400">Dominant ThoughtSeed</h3>
                </div>
                {currentCycle.thoughts[currentCycle.dominant_thought_id] && (
                  <div className="text-white">
                    {currentCycle.thoughts[currentCycle.dominant_thought_id].content}
                  </div>
                )}
                {currentCycle.consciousness_state?.consciousness_level && (
                  <div className="text-xs text-gray-300 mt-2">
                    Level: {currentCycle.consciousness_state.consciousness_level}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Current ThoughtSeed Competition */}
      {currentCycle && (
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-white mb-6 flex items-center space-x-3">
            <Cpu className="h-6 w-6 text-blue-400" />
            <span>ThoughtSeed Competition</span>
            {error && (
              <span className="text-red-400 text-sm font-normal">Error: {error}</span>
            )}
          </h2>

          <div className="space-y-4">
            {/* ThoughtSeeds Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.values(currentCycle.thoughts).map(thought =>
                renderThoughtCard(
                  thought,
                  thought.id === currentCycle.dominant_thought_id
                )
              )}
            </div>

            {/* IWMT Metrics */}
            {currentCycle.consciousness_state?.iwmt_metrics && (
              <div className="p-4 bg-gray-800/50 rounded-lg">
                <h4 className="font-medium text-gray-300 mb-3 flex items-center space-x-2">
                  <Activity className="h-4 w-4" />
                  <span>IWMT Consciousness Metrics</span>
                </h4>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-gray-400">Spatial Coherence</span>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-700 rounded-full h-1.5">
                        <div
                          className="bg-blue-400 h-1.5 rounded-full transition-all"
                          style={{ width: `${(currentCycle.consciousness_state.iwmt_metrics.spatial_coherence || 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-gray-300 w-10">
                        {((currentCycle.consciousness_state.iwmt_metrics.spatial_coherence || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-400">Temporal Coherence</span>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-700 rounded-full h-1.5">
                        <div
                          className="bg-green-400 h-1.5 rounded-full transition-all"
                          style={{ width: `${(currentCycle.consciousness_state.iwmt_metrics.temporal_coherence || 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-gray-300 w-10">
                        {((currentCycle.consciousness_state.iwmt_metrics.temporal_coherence || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-400">Causal Coherence</span>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-700 rounded-full h-1.5">
                        <div
                          className="bg-purple-400 h-1.5 rounded-full transition-all"
                          style={{ width: `${(currentCycle.consciousness_state.iwmt_metrics.causal_coherence || 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-gray-300 w-10">
                        {((currentCycle.consciousness_state.iwmt_metrics.causal_coherence || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-400">Embodied Selfhood</span>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-700 rounded-full h-1.5">
                        <div
                          className="bg-yellow-400 h-1.5 rounded-full transition-all"
                          style={{ width: `${(currentCycle.consciousness_state.iwmt_metrics.embodied_selfhood || 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-gray-300 w-10">
                        {((currentCycle.consciousness_state.iwmt_metrics.embodied_selfhood || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* MAC Analysis Display */}
            {currentCycle.consciousness_state?.mac_analysis && (
              <div className="mb-8">
                <h3 className="text-lg font-medium text-gray-300 mb-4 flex items-center space-x-2">
                  <Zap className="h-5 w-5 text-yellow-400" />
                  <span>Metacognitive Actor-Critic Analysis</span>
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {currentCycle.consciousness_state.mac_analysis.map((analysis, index) => (
                    <div key={index} className="p-3 bg-gray-800/50 rounded border border-gray-600">
                      <div className="flex justify-between items-center mb-2">
                        <span className={`text-sm px-2 py-1 rounded ${getThoughtTypeColor(analysis.type as ThoughtSeed['type'])}`}>
                          {analysis.type}
                        </span>
                        <span className="text-xs text-gray-400">
                          Q: {analysis.q_value.toFixed(3)}
                        </span>
                      </div>
                      <div className="text-xs space-y-1">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Error:</span>
                          <span className="text-gray-300">{analysis.metacognitive_error.toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Detection:</span>
                          <span className={analysis.can_detect_suboptimal ? 'text-green-400' : 'text-red-400'}>
                            {analysis.can_detect_suboptimal ? 'Yes' : 'No'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Magnitude:</span>
                          <span className="text-gray-300">{analysis.error_magnitude.toFixed(3)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
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