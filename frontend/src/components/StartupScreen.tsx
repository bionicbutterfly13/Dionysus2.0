import { useState, useEffect } from 'react'
import { AlertTriangle, CheckCircle, Loader, XCircle } from 'lucide-react'

interface ServiceStatus {
  name: string
  port: number
  status: 'checking' | 'connected' | 'failed' | 'port_busy'
  progress: number
  error?: string
}

interface StartupScreenProps {
  onReady: () => void
}

export default function StartupScreen({ onReady }: StartupScreenProps) {
  const [services, setServices] = useState<ServiceStatus[]>([
    { name: 'Backend', port: 9127, status: 'checking', progress: 0 },
    { name: 'Frontend', port: 5173, status: 'checking', progress: 0 },
    { name: 'Neo4j', port: 7687, status: 'checking', progress: 0 },
    { name: 'Redis', port: 6379, status: 'checking', progress: 0 },
  ])

  const [showKillDialog, setShowKillDialog] = useState<ServiceStatus | null>(null)

  useEffect(() => {
    checkServices()
  }, [])

  const checkServices = async () => {
    for (const service of services) {
      try {
        setServices(prev =>
          prev.map(s =>
            s.port === service.port ? { ...s, status: 'checking', progress: 30 } : s
          )
        )

        const response = await fetch(`http://localhost:${service.port}/health`, {
          signal: AbortSignal.timeout(2000)
        })

        if (response.ok) {
          setServices(prev =>
            prev.map(s =>
              s.port === service.port ? { ...s, status: 'connected', progress: 100 } : s
            )
          )
        } else {
          throw new Error('Service unhealthy')
        }
      } catch (error) {
        // Check if port is busy or service is down
        const isBusy = await checkPortBusy(service.port)

        setServices(prev =>
          prev.map(s =>
            s.port === service.port
              ? {
                  ...s,
                  status: isBusy ? 'port_busy' : 'failed',
                  progress: 0,
                  error: error instanceof Error ? error.message : 'Connection failed'
                }
              : s
          )
        )
      }
    }

    // Check if all services are ready
    setTimeout(() => {
      const allReady = services.every(s => s.status === 'connected')
      if (allReady) {
        onReady()
      }
    }, 500)
  }

  const checkPortBusy = async (port: number): Promise<boolean> => {
    try {
      const response = await fetch(`/api/system/check-port/${port}`)
      const data = await response.json()
      return data.busy === true
    } catch {
      return false
    }
  }

  const killPort = async (service: ServiceStatus) => {
    try {
      const response = await fetch(`/api/system/kill-port/${service.port}`, {
        method: 'POST'
      })

      if (response.ok) {
        setShowKillDialog(null)
        // Retry connection after killing
        setTimeout(() => checkServices(), 1000)
      }
    } catch (error) {
      console.error('Failed to kill port:', error)
    }
  }

  const getStatusIcon = (status: ServiceStatus['status']) => {
    switch (status) {
      case 'checking':
        return <Loader className="w-4 h-4 animate-spin text-blue-400" />
      case 'connected':
        return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />
      case 'port_busy':
        return <AlertTriangle className="w-4 h-4 text-yellow-400" />
    }
  }

  const allFailed = services.every(s => s.status === 'failed' || s.status === 'port_busy')
  const diagnostic = allFailed
    ? `Services failed to connect:\n${services
        .filter(s => s.status !== 'connected')
        .map(s => `- ${s.name} (port ${s.port}): ${s.error || 'Unknown error'}`)
        .join('\n')}\n\nCopy this and paste into GPT for help.`
    : ''

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-black via-gray-900 to-black flex items-center justify-center">
      {/* Luminescent threads background */}
      <div className="absolute inset-0 overflow-hidden opacity-30">
        <svg className="absolute w-full h-full">
          <defs>
            <linearGradient id="thread1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style={{ stopColor: '#3b82f6', stopOpacity: 0.6 }} />
              <stop offset="100%" style={{ stopColor: '#8b5cf6', stopOpacity: 0 }} />
            </linearGradient>
            <linearGradient id="thread2" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style={{ stopColor: '#10b981', stopOpacity: 0.6 }} />
              <stop offset="100%" style={{ stopColor: '#3b82f6', stopOpacity: 0 }} />
            </linearGradient>
          </defs>
          <path
            d="M 0,100 Q 250,50 500,100 T 1000,100"
            stroke="url(#thread1)"
            strokeWidth="1"
            fill="none"
            className="animate-pulse"
          />
          <path
            d="M 0,200 Q 250,150 500,200 T 1000,200"
            stroke="url(#thread2)"
            strokeWidth="1"
            fill="none"
            className="animate-pulse"
            style={{ animationDelay: '0.5s' }}
          />
        </svg>
      </div>

      {/* Loading card */}
      <div className="relative bg-gray-800/80 backdrop-blur-sm border border-gray-700 rounded-lg shadow-2xl p-8 w-96">
        <h2 className="text-xl font-bold text-gray-100 mb-6 text-center">
          Loading Dionysus
        </h2>

        <div className="space-y-4">
          {services.map((service) => (
            <div key={service.port} className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  {getStatusIcon(service.status)}
                  <span className="text-gray-300">{service.name}</span>
                  <span className="text-gray-500 text-xs">(port {service.port})</span>
                </div>
                {service.status === 'port_busy' && (
                  <button
                    onClick={() => setShowKillDialog(service)}
                    className="text-xs text-yellow-400 hover:text-yellow-300 underline"
                  >
                    Kill
                  </button>
                )}
              </div>

              {/* Progress bar */}
              <div className="w-full bg-gray-700 rounded-full h-1">
                <div
                  className={`h-1 rounded-full transition-all duration-300 ${
                    service.status === 'connected'
                      ? 'bg-green-400'
                      : service.status === 'failed'
                      ? 'bg-red-400'
                      : service.status === 'port_busy'
                      ? 'bg-yellow-400'
                      : 'bg-blue-400'
                  }`}
                  style={{ width: `${service.progress}%` }}
                />
              </div>

              {service.error && (
                <p className="text-xs text-red-400 mt-1">{service.error}</p>
              )}
            </div>
          ))}
        </div>

        {allFailed && (
          <div className="mt-6 p-4 bg-red-900/20 border border-red-700 rounded-lg">
            <h3 className="text-sm font-semibold text-red-400 mb-2">Diagnostic Info</h3>
            <pre className="text-xs text-gray-300 whitespace-pre-wrap font-mono bg-gray-900/50 p-2 rounded max-h-40 overflow-y-auto">
              {diagnostic}
            </pre>
            <button
              onClick={() => navigator.clipboard.writeText(diagnostic)}
              className="mt-2 text-xs text-blue-400 hover:text-blue-300 underline"
            >
              Copy to clipboard
            </button>
          </div>
        )}

        <div className="mt-6 text-center text-sm text-gray-400">
          {services.every(s => s.status === 'connected')
            ? 'All services ready!'
            : services.some(s => s.status === 'checking')
            ? 'Connecting...'
            : 'Waiting for services...'}
        </div>
      </div>

      {/* Kill dialog */}
      {showKillDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 max-w-md">
            <h3 className="text-lg font-bold text-gray-100 mb-4">
              Backend is busy
            </h3>
            <p className="text-gray-300 mb-6">
              Port {showKillDialog.port} is already in use. Kill the existing process?
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowKillDialog(null)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={() => killPort(showKillDialog)}
                className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg"
              >
                Kill It
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
