import { useState, useEffect } from 'react'
import { 
  FileText, 
  Brain, 
  Search, 
  Activity,
  TrendingUp,
  Clock
} from 'lucide-react'

interface DashboardStats {
  documentsProcessed: number
  conceptsExtracted: number
  curiosityMissions: number
  activeThoughtSeeds: number
  mockData: boolean
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    documentsProcessed: 0,
    conceptsExtracted: 0,
    curiosityMissions: 0,
    activeThoughtSeeds: 0,
    mockData: true
  })

  useEffect(() => {
    // Fetch real stats from backend
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/stats/dashboard')
        if (response.ok) {
          const data = await response.json()
          setStats(data)
        } else {
          throw new Error('Failed to fetch stats')
        }
      } catch (error) {
        console.error('Error fetching dashboard stats:', error)
        // Fallback to mock data
        setStats({
          documentsProcessed: 42,
          conceptsExtracted: 156,
          curiosityMissions: 8,
          activeThoughtSeeds: 12,
          mockData: true
        })
      }
    }

    fetchStats()
    // Refresh every 5 seconds
    const interval = setInterval(fetchStats, 5000)
    return () => clearInterval(interval)
  }, [])

  const statCards = [
    {
      title: 'Documents Processed',
      value: stats.documentsProcessed,
      icon: FileText,
      color: 'text-blue-400',
      bgColor: 'bg-blue-900/20'
    },
    {
      title: 'Concepts Extracted',
      value: stats.conceptsExtracted,
      icon: Brain,
      color: 'text-green-400',
      bgColor: 'bg-green-900/20'
    },
    {
      title: 'Curiosity Missions',
      value: stats.curiosityMissions,
      icon: Search,
      color: 'text-purple-400',
      bgColor: 'bg-purple-900/20'
    },
    {
      title: 'Active ThoughtSeeds',
      value: stats.activeThoughtSeeds,
      icon: Activity,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-900/20'
    }
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-gray-400">Overview of your consciousness emulator's learning progress</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((card) => (
          <div key={card.title} className="panel-glow p-6">
            <div className="flex items-center">
              <div className={`p-3 rounded-lg ${card.bgColor}`}>
                <card.icon className={`h-6 w-6 ${card.color}`} />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-400">{card.title}</p>
                <p className="text-2xl font-bold text-white">{card.value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="panel-glow p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Recent Document Processing</h3>
          <div className="space-y-3">
            <div className="flex items-center text-sm">
              <Clock className="h-4 w-4 text-gray-400 mr-2" />
              <span className="text-gray-300">research_paper_ai_ethics.pdf</span>
              <span className="ml-auto text-green-400 font-medium">Processed</span>
            </div>
            <div className="flex items-center text-sm">
              <Clock className="h-4 w-4 text-gray-400 mr-2" />
              <span className="text-gray-300">consciousness_theory_notes.md</span>
              <span className="ml-auto text-yellow-400 font-medium">Processing</span>
            </div>
            <div className="flex items-center text-sm">
              <Clock className="h-4 w-4 text-gray-400 mr-2" />
              <span className="text-gray-300">neuroscience_review.pdf</span>
              <span className="ml-auto text-purple-400 font-medium">Curious</span>
            </div>
          </div>
        </div>

        <div className="panel-glow p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Active ThoughtSeeds</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-300">Active Inference Learning</span>
              <TrendingUp className="h-4 w-4 text-green-400" />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-300">Pattern Recognition</span>
              <TrendingUp className="h-4 w-4 text-blue-400" />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-300">Memory Consolidation</span>
              <TrendingUp className="h-4 w-4 text-purple-400" />
            </div>
          </div>
        </div>
      </div>

      {/* Mock Data Notice */}
      {stats.mockData && (
        <div className="panel-glow-yellow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-200">
                <strong>Development Mode:</strong> Dashboard showing simulated data.
                Real data integration pending backend connection.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
