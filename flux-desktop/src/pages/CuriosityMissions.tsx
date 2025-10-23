import { useState, useEffect } from 'react'
import { Search, Plus, Play, Pause, Globe, BookOpen, AlertCircle } from 'lucide-react'

interface CuriosityMission {
  id: string
  title: string
  description: string
  status: 'active' | 'paused' | 'completed' | 'failed'
  sourcesFound: number
  knowledgeGaps: string[]
  createdAt: string
  mockData: boolean
}

export default function CuriosityMissions() {
  const [missions, setMissions] = useState<CuriosityMission[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [newMissionPrompt, setNewMissionPrompt] = useState('')

  useEffect(() => {
    // TODO: Replace with actual API call
    setTimeout(() => {
      setMissions([
        {
          id: '1',
          title: 'Active Inference in Machine Learning',
          description: 'Exploring recent developments in active inference algorithms and their applications',
          status: 'active',
          sourcesFound: 12,
          knowledgeGaps: ['Bayesian optimization', 'Variational inference'],
          createdAt: '2024-01-15T10:30:00Z',
          mockData: true
        },
        {
          id: '2',
          title: 'Consciousness and Attention Mechanisms',
          description: 'Investigating the relationship between attention mechanisms in transformers and consciousness theories',
          status: 'completed',
          sourcesFound: 8,
          knowledgeGaps: [],
          createdAt: '2024-01-14T14:20:00Z',
          mockData: true
        },
        {
          id: '3',
          title: 'Memory Consolidation in Neural Networks',
          description: 'Researching how neural networks can implement memory consolidation similar to human sleep',
          status: 'paused',
          sourcesFound: 5,
          knowledgeGaps: ['Sleep-wake cycles', 'Hippocampal replay'],
          createdAt: '2024-01-13T09:15:00Z',
          mockData: true
        }
      ])
      setIsLoading(false)
    }, 1000)
  }, [])

  const getStatusColor = (status: CuriosityMission['status']) => {
    switch (status) {
      case 'active':
        return 'consciousness-curiosity'
      case 'completed':
        return 'consciousness-memory'
      case 'paused':
        return 'consciousness-thought'
      case 'failed':
        return 'bg-red-100 text-red-800'
    }
  }

  const getStatusIcon = (status: CuriosityMission['status']) => {
    switch (status) {
      case 'active':
        return <Play className="h-4 w-4" />
      case 'completed':
        return <BookOpen className="h-4 w-4" />
      case 'paused':
        return <Pause className="h-4 w-4" />
      case 'failed':
        return <AlertCircle className="h-4 w-4" />
    }
  }

  const handleCreateMission = () => {
    if (!newMissionPrompt.trim()) return

    const newMission: CuriosityMission = {
      id: Math.random().toString(36).substr(2, 9),
      title: newMissionPrompt,
      description: `Curiosity mission: ${newMissionPrompt}`,
      status: 'active',
      sourcesFound: 0,
      knowledgeGaps: [],
      createdAt: new Date().toISOString(),
      mockData: true
    }

    setMissions(prev => [newMission, ...prev])
    setNewMissionPrompt('')
  }

  const toggleMissionStatus = (missionId: string) => {
    setMissions(prev => 
      prev.map(mission => 
        mission.id === missionId 
          ? { 
              ...mission, 
              status: mission.status === 'active' ? 'paused' : 'active' 
            }
          : mission
      )
    )
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Curiosity Missions</h1>
          <p className="text-gray-600">Loading curiosity engine missions...</p>
        </div>
        <div className="flux-card">
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-flux-primary"></div>
            <span className="ml-3 text-gray-600">Initializing curiosity engine...</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Curiosity Missions</h1>
        <p className="text-gray-600">Explore knowledge gaps and discover new insights through curiosity-driven research</p>
      </div>

      {/* Create New Mission */}
      <div className="flux-card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Start New Curiosity Mission</h3>
        <div className="space-y-4">
          <div>
            <label htmlFor="mission-prompt" className="block text-sm font-medium text-gray-700 mb-2">
              What knowledge gap or topic interests you?
            </label>
            <textarea
              id="mission-prompt"
              rows={3}
              value={newMissionPrompt}
              onChange={(e) => setNewMissionPrompt(e.target.value)}
              placeholder="Describe a topic, question, or knowledge gap you'd like the curiosity engine to explore..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-flux-primary"
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-500">
              The curiosity engine will search reputable sources and identify related concepts
            </div>
            <button 
              onClick={handleCreateMission}
              className="flux-button-primary flex items-center"
            >
              <Plus className="h-4 w-4 mr-2" />
              Start Mission
            </button>
          </div>
        </div>
      </div>

      {/* Active Missions */}
      <div className="flux-card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Active Missions</h3>
        <div className="space-y-4">
          {missions.filter(m => m.status === 'active').map((mission) => (
            <div key={mission.id} className="p-4 border border-gray-200 rounded-lg">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900 mb-1">{mission.title}</h4>
                  <p className="text-sm text-gray-600">{mission.description}</p>
                </div>
                <div className="flex items-center space-x-2 ml-4">
                  <span className={`consciousness-indicator ${getStatusColor(mission.status)}`}>
                    {getStatusIcon(mission.status)}
                    <span className="ml-1 capitalize">{mission.status}</span>
                  </span>
                  <button
                    onClick={() => toggleMissionStatus(mission.id)}
                    className="p-1 text-gray-400 hover:text-gray-600"
                  >
                    <Pause className="h-4 w-4" />
                  </button>
                </div>
              </div>
              
              <div className="flex items-center justify-between text-sm text-gray-500">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center">
                    <Globe className="h-4 w-4 mr-1" />
                    <span>{mission.sourcesFound} sources found</span>
                  </div>
                  <div className="flex items-center">
                    <Search className="h-4 w-4 mr-1" />
                    <span>{mission.knowledgeGaps.length} knowledge gaps</span>
                  </div>
                </div>
                <div>
                  Started {new Date(mission.createdAt).toLocaleDateString()}
                </div>
              </div>

              {mission.knowledgeGaps.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <p className="text-xs text-gray-500 mb-2">Knowledge gaps identified:</p>
                  <div className="flex flex-wrap gap-1">
                    {mission.knowledgeGaps.map((gap, index) => (
                      <span key={index} className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs rounded">
                        {gap}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {mission.mockData && (
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <p className="text-xs text-yellow-600">Mock curiosity engine - simulated web search</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Mission History */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="flux-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Completed Missions</h3>
          <div className="space-y-3">
            {missions.filter(m => m.status === 'completed').map((mission) => (
              <div key={mission.id} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900 text-sm">{mission.title}</h4>
                  <span className={`consciousness-indicator ${getStatusColor(mission.status)}`}>
                    {getStatusIcon(mission.status)}
                    <span className="ml-1 capitalize">{mission.status}</span>
                  </span>
                </div>
                <div className="text-xs text-gray-500">
                  {mission.sourcesFound} sources • {new Date(mission.createdAt).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flux-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Paused Missions</h3>
          <div className="space-y-3">
            {missions.filter(m => m.status === 'paused').map((mission) => (
              <div key={mission.id} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900 text-sm">{mission.title}</h4>
                  <button
                    onClick={() => toggleMissionStatus(mission.id)}
                    className="p-1 text-gray-400 hover:text-gray-600"
                  >
                    <Play className="h-4 w-4" />
                  </button>
                </div>
                <div className="text-xs text-gray-500">
                  {mission.sourcesFound} sources • {mission.knowledgeGaps.length} gaps
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Curiosity Engine Info */}
      <div className="flux-card bg-blue-50 border-blue-200">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <Search className="h-5 w-5 text-blue-400" />
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">Curiosity Engine</h3>
            <p className="text-sm text-blue-700 mt-1">
              The curiosity engine identifies knowledge gaps, searches reputable sources, 
              and discovers new strategies when existing approaches fail. It operates 
              autonomously during idle times and integrates findings into your knowledge graph.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
