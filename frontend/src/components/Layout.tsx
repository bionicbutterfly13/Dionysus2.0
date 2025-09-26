import { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  Home,
  Network,
  Search,
  Brain,
  Activity,
  Settings,
  Database,
  Zap
} from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Knowledge Base', href: '/knowledge-base', icon: Database },
  { name: 'Knowledge Graph', href: '/knowledge-graph', icon: Network },
  { name: 'ThoughtSeed', href: '/thoughtseed', icon: Zap },
  { name: 'Curiosity Missions', href: '/curiosity', icon: Search },
]

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="flex flex-col w-64 sidebar-dark">
        <div className="flex items-center h-16 px-6 border-b border-gray-700">
          <Brain className="h-8 w-8 text-blue-400" />
          <span className="ml-2 text-xl font-bold text-white">Flux</span>
        </div>
        
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`nav-button ${isActive ? 'active' : ''} flex items-center px-3 py-2 text-sm font-medium transition-colors text-gray-300 hover:text-white`}
              >
                <item.icon className="mr-3 h-5 w-5 glow-icon" />
                {item.name}
              </Link>
            )
          })}
        </nav>

        <div className="p-4 border-t border-gray-700 space-y-3">
          <div className="flex items-center text-sm text-gray-400">
            <Activity className="mr-2 h-4 w-4" />
            <span>Consciousness Active</span>
          </div>

          {/* Settings Link */}
          <Link
            to="/settings"
            className={`nav-button ${location.pathname === '/settings' ? 'active' : ''} flex items-center px-3 py-2 text-sm font-medium transition-colors text-gray-300 hover:text-white`}
          >
            <Settings className="mr-3 h-5 w-5 glow-icon" />
            Settings
          </Link>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
