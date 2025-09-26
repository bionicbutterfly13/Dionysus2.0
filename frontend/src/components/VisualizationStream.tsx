import { useState, useEffect } from 'react'
import { Activity, Brain, Zap, Eye } from 'lucide-react'

interface VisualizationMessage {
  type: 'graph_update' | 'card_stack_update' | 'curiosity_signal' | 'evaluation_frame' | 'mosaic_state'
  timestamp: string
  data: any
  mockData: boolean
}

export default function VisualizationStream() {
  const [isConnected, setIsConnected] = useState(false)
  const [messages, setMessages] = useState<VisualizationMessage[]>([])
  const [isExpanded, setIsExpanded] = useState(false)

  useEffect(() => {
    // TODO: Replace with actual WebSocket connection
    // Simulate WebSocket connection for development
    const simulateConnection = () => {
      setIsConnected(true)
      
      // Simulate incoming messages
      const messageTypes = [
        'graph_update',
        'curiosity_signal', 
        'evaluation_frame',
        'mosaic_state'
      ] as const

      const interval = setInterval(() => {
        const randomType = messageTypes[Math.floor(Math.random() * messageTypes.length)]
        const newMessage: VisualizationMessage = {
          type: randomType,
          timestamp: new Date().toISOString(),
          data: { mock: true },
          mockData: true
        }
        
        setMessages(prev => [newMessage, ...prev.slice(0, 9)]) // Keep last 10 messages
      }, 3000)

      return () => clearInterval(interval)
    }

    const cleanup = simulateConnection()
    return cleanup
  }, [])

  const getMessageIcon = (type: VisualizationMessage['type']) => {
    switch (type) {
      case 'graph_update':
        return <Brain className="h-4 w-4" />
      case 'curiosity_signal':
        return <Zap className="h-4 w-4" />
      case 'evaluation_frame':
        return <Activity className="h-4 w-4" />
      case 'mosaic_state':
        return <Eye className="h-4 w-4" />
      default:
        return <Activity className="h-4 w-4" />
    }
  }

  const getMessageColor = (type: VisualizationMessage['type']) => {
    switch (type) {
      case 'graph_update':
        return 'text-blue-600 bg-blue-50'
      case 'curiosity_signal':
        return 'text-purple-600 bg-purple-50'
      case 'evaluation_frame':
        return 'text-green-600 bg-green-50'
      case 'mosaic_state':
        return 'text-orange-600 bg-orange-50'
      default:
        return 'text-gray-600 bg-gray-50'
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Connection Status */}
      <div className={`mb-2 px-3 py-1 rounded-full text-xs font-medium ${
        isConnected 
          ? 'bg-green-100 text-green-800' 
          : 'bg-red-100 text-red-800'
      }`}>
        {isConnected ? 'Consciousness Stream Active' : 'Stream Disconnected'}
      </div>

      {/* Stream Panel */}
      <div className={`bg-white rounded-lg shadow-lg border transition-all duration-300 ${
        isExpanded ? 'w-80 h-96' : 'w-16 h-16'
      }`}>
        {isExpanded ? (
          <div className="p-4 h-full flex flex-col">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-gray-900">Consciousness Stream</h3>
              <button
                onClick={() => setIsExpanded(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto space-y-2">
              {messages.length === 0 ? (
                <div className="text-center text-gray-500 text-sm py-8">
                  Waiting for consciousness signals...
                </div>
              ) : (
                messages.map((message, index) => (
                  <div key={index} className="flex items-start space-x-2 p-2 rounded">
                    <div className={`p-1 rounded ${getMessageColor(message.type)}`}>
                      {getMessageIcon(message.type)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium text-gray-900 capitalize">
                        {message.type.replace('_', ' ')}
                      </div>
                      <div className="text-xs text-gray-500">
                        {formatTimestamp(message.timestamp)}
                      </div>
                      {message.mockData && (
                        <div className="text-xs text-yellow-600 mt-1">
                          Mock Signal
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
            
            <div className="pt-3 border-t border-gray-200">
              <div className="text-xs text-gray-500 text-center">
                Real-time consciousness visualization
              </div>
            </div>
          </div>
        ) : (
          <button
            onClick={() => setIsExpanded(true)}
            className="w-full h-full flex items-center justify-center text-gray-600 hover:text-gray-900"
          >
            <Activity className="h-6 w-6" />
          </button>
        )}
      </div>
    </div>
  )
}
