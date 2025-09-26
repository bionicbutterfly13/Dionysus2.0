import { useState, useEffect } from 'react'
import { Network, Search, Filter, Download } from 'lucide-react'

interface GraphNode {
  id: string
  label: string
  type: 'concept' | 'document' | 'thoughtseed'
  connections: number
  mockData: boolean
}

interface GraphConnection {
  from: string
  to: string
  strength: number
}

export default function KnowledgeGraph() {
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [connections, setConnections] = useState<GraphConnection[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedType, setSelectedType] = useState<string>('all')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // TODO: Replace with actual API call
    // Simulate loading knowledge graph data
    setTimeout(() => {
      setNodes([
        { id: '1', label: 'Active Inference', type: 'concept', connections: 8, mockData: true },
        { id: '2', label: 'Consciousness Theory', type: 'concept', connections: 12, mockData: true },
        { id: '3', label: 'research_paper_ai_ethics.pdf', type: 'document', connections: 5, mockData: true },
        { id: '4', label: 'ThoughtSeed_001', type: 'thoughtseed', connections: 3, mockData: true },
        { id: '5', label: 'Pattern Recognition', type: 'concept', connections: 6, mockData: true },
        { id: '6', label: 'Memory Consolidation', type: 'concept', connections: 4, mockData: true },
        { id: '7', label: 'consciousness_notes.md', type: 'document', connections: 7, mockData: true },
        { id: '8', label: 'Curiosity Engine', type: 'concept', connections: 9, mockData: true }
      ])

      setConnections([
        { from: '1', to: '2', strength: 0.8 },
        { from: '1', to: '5', strength: 0.9 },
        { from: '2', to: '3', strength: 0.7 },
        { from: '3', to: '4', strength: 0.6 },
        { from: '4', to: '6', strength: 0.5 },
        { from: '5', to: '8', strength: 0.8 },
        { from: '6', to: '7', strength: 0.7 },
        { from: '7', to: '8', strength: 0.6 }
      ])

      setIsLoading(false)
    }, 1500)
  }, [])

  const filteredNodes = nodes.filter(node => {
    const matchesSearch = node.label.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesType = selectedType === 'all' || node.type === selectedType
    return matchesSearch && matchesType
  })

  const getNodeTypeColor = (type: GraphNode['type']) => {
    switch (type) {
      case 'concept':
        return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'document':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'thoughtseed':
        return 'bg-purple-100 text-purple-800 border-purple-200'
    }
  }

  const getNodeTypeIcon = (type: GraphNode['type']) => {
    switch (type) {
      case 'concept':
        return '🧠'
      case 'document':
        return '📄'
      case 'thoughtseed':
        return '🌱'
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Knowledge Graph</h1>
          <p className="text-gray-600">Loading your consciousness network...</p>
        </div>
        <div className="flux-card">
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-flux-primary"></div>
            <span className="ml-3 text-gray-600">Building knowledge graph...</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Knowledge Graph</h1>
        <p className="text-gray-600">Explore the interconnected concepts, documents, and ThoughtSeeds</p>
      </div>

      {/* Controls */}
      <div className="flux-card">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search concepts, documents, or ThoughtSeeds..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-flux-primary"
              />
            </div>
          </div>
          <div className="flex gap-2">
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-flux-primary"
            >
              <option value="all">All Types</option>
              <option value="concept">Concepts</option>
              <option value="document">Documents</option>
              <option value="thoughtseed">ThoughtSeeds</option>
            </select>
            <button className="flux-button-secondary flex items-center">
              <Filter className="h-4 w-4 mr-2" />
              Filter
            </button>
            <button className="flux-button-secondary flex items-center">
              <Download className="h-4 w-4 mr-2" />
              Export
            </button>
          </div>
        </div>
      </div>

      {/* Graph Visualization Placeholder */}
      <div className="flux-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Network Visualization</h3>
          <div className="text-sm text-gray-500">
            {filteredNodes.length} nodes, {connections.length} connections
          </div>
        </div>
        
        {/* Placeholder for actual graph visualization */}
        <div className="h-96 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300 flex items-center justify-center">
          <div className="text-center">
            <Network className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 mb-2">Interactive Graph Visualization</p>
            <p className="text-sm text-gray-500">
              Graph rendering will be implemented with D3.js or React Flow
            </p>
          </div>
        </div>
      </div>

      {/* Node List */}
      <div className="flux-card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Knowledge Nodes</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredNodes.map((node) => (
            <div key={node.id} className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center">
                  <span className="text-lg mr-2">{getNodeTypeIcon(node.type)}</span>
                  <span className="font-medium text-gray-900">{node.label}</span>
                </div>
                <span className={`px-2 py-1 text-xs rounded-full border ${getNodeTypeColor(node.type)}`}>
                  {node.type}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                <p>{node.connections} connections</p>
                {node.mockData && (
                  <p className="text-yellow-600 mt-1">Mock Data</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Connection Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="flux-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Strongest Connections</h3>
          <div className="space-y-3">
            {connections
              .sort((a, b) => b.strength - a.strength)
              .slice(0, 5)
              .map((conn, index) => {
                const fromNode = nodes.find(n => n.id === conn.from)
                const toNode = nodes.find(n => n.id === conn.to)
                return (
                  <div key={index} className="flex items-center justify-between">
                    <div className="text-sm">
                      <span className="font-medium">{fromNode?.label}</span>
                      <span className="text-gray-400 mx-2">→</span>
                      <span className="font-medium">{toNode?.label}</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                        <div 
                          className="bg-flux-primary h-2 rounded-full" 
                          style={{ width: `${conn.strength * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500">{Math.round(conn.strength * 100)}%</span>
                    </div>
                  </div>
                )
              })}
          </div>
        </div>

        <div className="flux-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Graph Statistics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Total Nodes</span>
              <span className="text-sm font-medium">{nodes.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Total Connections</span>
              <span className="text-sm font-medium">{connections.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Concepts</span>
              <span className="text-sm font-medium">{nodes.filter(n => n.type === 'concept').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Documents</span>
              <span className="text-sm font-medium">{nodes.filter(n => n.type === 'document').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">ThoughtSeeds</span>
              <span className="text-sm font-medium">{nodes.filter(n => n.type === 'thoughtseed').length}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
