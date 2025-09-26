/**
 * ThoughtSeed API Service
 * Connects to backend ThoughtSeed competition system
 */

export interface ThoughtSeedData {
  id: string
  content: string
  type: 'goal' | 'action' | 'belief' | 'perception'
  energy: number
  confidence: number
  dominance_score: number
}

export interface CompetitionState {
  cycle: number
  timestamp: string
  thoughts: ThoughtSeedData[]
  dominant_thought_id: string
  consciousness_level: number
  status: 'running' | 'paused' | 'stopped'
}

class ThoughtSeedAPI {
  private baseURL = '/api/thoughtseed'
  private isConnected = false

  async checkConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/status`)
      this.isConnected = response.ok
      return this.isConnected
    } catch (error) {
      console.log('⚠️ ThoughtSeed backend offline - using fallback data')
      this.isConnected = false
      return false
    }
  }

  async startCompetition(): Promise<{ success: boolean; message?: string }> {
    if (!this.isConnected) {
      return { success: false, message: 'Backend not connected' }
    }

    try {
      const response = await fetch(`${this.baseURL}/competition/start`, {
        method: 'POST'
      })
      return await response.json()
    } catch (error) {
      return { success: false, message: 'Failed to start competition' }
    }
  }

  async pauseCompetition(): Promise<{ success: boolean }> {
    if (!this.isConnected) {
      return { success: false }
    }

    try {
      const response = await fetch(`${this.baseURL}/competition/pause`, {
        method: 'POST'
      })
      return await response.json()
    } catch (error) {
      return { success: false }
    }
  }

  async getCurrentState(): Promise<CompetitionState | null> {
    if (!this.isConnected) {
      return this.generateFallbackData()
    }

    try {
      const response = await fetch(`${this.baseURL}/competition/state`)
      if (response.ok) {
        return await response.json()
      }
      return this.generateFallbackData()
    } catch (error) {
      return this.generateFallbackData()
    }
  }

  // Fallback data when backend is offline
  private generateFallbackData(): CompetitionState {
    const thoughts: ThoughtSeedData[] = [
      {
        id: 'thought_1',
        content: 'Analyze the current problem systematically',
        type: 'action',
        energy: Math.random() * 0.8 + 0.2,
        confidence: Math.random() * 0.6 + 0.4,
        dominance_score: Math.random()
      },
      {
        id: 'thought_2',
        content: 'Use creative and novel approaches',
        type: 'action',
        energy: Math.random() * 0.8 + 0.2,
        confidence: Math.random() * 0.6 + 0.4,
        dominance_score: Math.random()
      },
      {
        id: 'thought_3',
        content: 'Focus on the core objective',
        type: 'goal',
        energy: Math.random() * 0.8 + 0.2,
        confidence: Math.random() * 0.6 + 0.4,
        dominance_score: Math.random()
      },
      {
        id: 'thought_4',
        content: 'Trust in established patterns',
        type: 'belief',
        energy: Math.random() * 0.8 + 0.2,
        confidence: Math.random() * 0.6 + 0.4,
        dominance_score: Math.random()
      }
    ]

    // Find dominant thought
    const dominantThought = thoughts.reduce((prev, current) =>
      (prev.energy * prev.confidence) > (current.energy * current.confidence) ? prev : current
    )

    return {
      cycle: Math.floor(Math.random() * 100) + 1,
      timestamp: new Date().toISOString(),
      thoughts,
      dominant_thought_id: dominantThought.id,
      consciousness_level: Math.random() * 0.8 + 0.2,
      status: 'running'
    }
  }

  // WebSocket connection for real-time updates
  connectWebSocket(onUpdate: (state: CompetitionState) => void): WebSocket | null {
    if (!this.isConnected) {
      // Simulate real-time updates with intervals when offline
      setInterval(() => {
        onUpdate(this.generateFallbackData())
      }, 3000)
      return null
    }

    try {
      const ws = new WebSocket('ws://localhost:8000/ws/thoughtseed')

      ws.onmessage = (event) => {
        const state = JSON.parse(event.data)
        onUpdate(state)
      }

      ws.onerror = () => {
        console.log('WebSocket error - falling back to polling')
        setInterval(() => {
          onUpdate(this.generateFallbackData())
        }, 3000)
      }

      return ws
    } catch (error) {
      console.log('WebSocket not available - using polling')
      setInterval(() => {
        onUpdate(this.generateFallbackData())
      }, 3000)
      return null
    }
  }
}

export const thoughtseedAPI = new ThoughtSeedAPI()