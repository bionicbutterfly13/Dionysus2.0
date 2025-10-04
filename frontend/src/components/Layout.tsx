import { ReactNode, useState, useEffect } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import {
  Home,
  Network,
  Search,
  Brain,
  Activity,
  Settings,
  Database,
  Zap,
  FileText,
  ChevronDown,
  ChevronRight,
  X
} from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

interface Document {
  id: string
  title: string
  type: 'file' | 'web'
  uploaded_at: string
  extraction?: {
    concepts: string[]
  }
  quality?: {
    scores: {
      overall: number
    }
  }
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
  const navigate = useNavigate()
  const [documents, setDocuments] = useState<Document[]>([])
  const [expandedDocs, setExpandedDocs] = useState<Set<string>>(new Set())

  // Extract active document ID from URL
  const activeDocId = location.pathname.startsWith('/document/')
    ? location.pathname.split('/document/')[1]
    : null

  // Fetch documents from backend
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const response = await fetch('/api/v1/documents')
        if (response.ok) {
          const data = await response.json()
          setDocuments(data.documents || [])
        }
      } catch (error) {
        console.error('[DOCUMENTS] Failed to fetch:', error)
      }
    }

    fetchDocuments()
    // Poll every 10 seconds for new documents
    const interval = setInterval(fetchDocuments, 10000)
    return () => clearInterval(interval)
  }, [])

  const toggleDocument = (docId: string) => {
    const newExpanded = new Set(expandedDocs)
    if (newExpanded.has(docId)) {
      newExpanded.delete(docId)
    } else {
      newExpanded.add(docId)
    }
    setExpandedDocs(newExpanded)
  }

  const handleDeleteDocument = async (docId: string, docTitle: string, event: React.MouseEvent) => {
    event.stopPropagation() // Prevent triggering other click handlers

    if (!confirm(`Delete "${docTitle}"?\n\nThis will permanently remove the document and all associated data.`)) {
      return
    }

    try {
      const response = await fetch(`/api/v1/documents/${docId}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        // Remove from local state immediately
        setDocuments(documents.filter(d => d.id !== docId))

        // If viewing this document, navigate away
        if (activeDocId === docId) {
          navigate('/')
        }

        console.log(`[DELETE] Successfully deleted document: ${docTitle}`)
      } else {
        const error = await response.json()
        alert(`Failed to delete document: ${error.detail || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('[DELETE] Failed:', error)
      alert('Failed to delete document. Please try again.')
    }
  }

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="flex flex-col w-64 sidebar-dark">
        <div className="flex items-center h-16 px-6 border-b border-gray-700">
          <Brain className="h-8 w-8 text-blue-400" />
          <span className="ml-2 text-xl font-bold text-white">Flux</span>
        </div>
        
        <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
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

        {/* Documents Section - moved to bottom */}
        <div className="border-t border-gray-700">
          {/* Documents Section */}
          <div className="space-y-1">
            <div className="flex items-center px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
              <FileText className="mr-2 h-4 w-4" />
              Documents ({documents.length})
            </div>

            {documents.length === 0 ? (
              <div className="px-3 py-2 text-xs text-gray-500 italic">
                No documents yet
              </div>
            ) : (
              <div className="space-y-1">
                {documents.map((doc) => {
                  const isExpanded = expandedDocs.has(doc.id)
                  const isActive = activeDocId === doc.id
                  return (
                    <div key={doc.id} className={`text-sm border-b border-gray-800 last:border-b-0 ${isActive ? 'bg-blue-500/10 border-blue-500/30' : ''}`}>
                      {/* Accordion-style header - clickable */}
                      <div className="w-full flex items-start group/doc">
                        {/* Title - clicks to detail view */}
                        <button
                          onClick={() => navigate(`/document/${doc.id}`)}
                          className={`flex-1 min-w-0 px-4 py-3 text-left hover:bg-gray-800 transition-colors ${isActive ? 'text-blue-300' : 'text-gray-300'}`}
                        >
                          <div className={`${isActive ? 'text-blue-300' : 'text-gray-300 group-hover/doc:text-white'} truncate font-medium flex items-center`}>
                            <span className="mr-2">{doc.type === 'web' ? '🌐' : '📄'}</span>
                            {doc.title}
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            {new Date(doc.uploaded_at).toLocaleDateString()}
                          </div>
                        </button>

                        {/* Delete button - red X */}
                        <button
                          onClick={(e) => handleDeleteDocument(doc.id, doc.title, e)}
                          className="px-2 py-3 hover:bg-red-500/10 transition-colors group/delete"
                          title="Delete document"
                        >
                          <X className="h-4 w-4 text-gray-600 group-hover/delete:text-red-500 transition-colors" />
                        </button>

                        {/* Chevron - toggles accordion */}
                        <button
                          onClick={() => toggleDocument(doc.id)}
                          className="px-2 py-3 hover:bg-gray-800 transition-colors"
                        >
                          <div className={`text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}>
                            <ChevronDown className="h-4 w-4" />
                          </div>
                        </button>
                      </div>

                      {/* Accordion expanded details */}
                      {isExpanded && (
                        <div className="px-4 pb-3 text-xs space-y-2 bg-gray-800/30">
                          {/* Concepts */}
                          {doc.extraction?.concepts && doc.extraction.concepts.length > 0 && (
                            <div>
                              <div className="text-gray-400 font-medium mb-1">Concepts:</div>
                              <div className="flex flex-wrap gap-1">
                                {doc.extraction.concepts.slice(0, 5).map((concept, idx) => (
                                  <span
                                    key={idx}
                                    className="px-2 py-0.5 bg-blue-500/20 text-blue-300 rounded text-xs"
                                  >
                                    {concept}
                                  </span>
                                ))}
                                {doc.extraction.concepts.length > 5 && (
                                  <span className="text-gray-500 text-xs">
                                    +{doc.extraction.concepts.length - 5} more
                                  </span>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Quality */}
                          {doc.quality?.scores?.overall !== undefined && (
                            <div>
                              <div className="text-gray-400 font-medium mb-1">Quality:</div>
                              <div className="text-gray-300">
                                {(doc.quality.scores.overall * 100).toFixed(0)}%
                              </div>
                            </div>
                          )}

                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>

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
