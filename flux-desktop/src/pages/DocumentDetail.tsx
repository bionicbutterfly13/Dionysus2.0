import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, FileText, Calendar, BarChart3, Tag, Globe } from 'lucide-react'

interface DocumentDetail {
  id: string
  title: string
  type: 'file' | 'web'
  uploaded_at: string
  size?: number
  content_type?: string
  extraction?: {
    concepts: string[]
    content?: string
    summary?: string
  }
  quality?: {
    scores: {
      overall: number
      clarity?: number
      depth?: number
      novelty?: number
    }
  }
  research?: {
    curiosity_triggers?: Array<{
      question: string
      novelty_score: number
    }>
  }
}

export default function DocumentDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [document, setDocument] = useState<DocumentDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [relatedDocs, setRelatedDocs] = useState<{[topic: string]: DocumentDetail[]}>({})

  useEffect(() => {
    const fetchDocument = async () => {
      try {
        const response = await fetch(`/api/v1/documents/${id}`)
        if (response.ok) {
          const data = await response.json()
          setDocument(data)
        }
      } catch (error) {
        console.error('[DOCUMENT] Failed to fetch:', error)
      } finally {
        setLoading(false)
      }
    }

    if (id) {
      fetchDocument()
    }
  }, [id])

  const handleTopicClick = async (topic: string) => {
    try {
      const response = await fetch(`/api/v1/documents?topic=${encodeURIComponent(topic)}`)
      if (response.ok) {
        const data = await response.json()
        setRelatedDocs({
          ...relatedDocs,
          [topic]: data.documents.filter((d: DocumentDetail) => d.id !== id)
        })
      }
    } catch (error) {
      console.error('[RELATED] Failed to fetch:', error)
    }
  }

  const handleRelatedDocClick = (docId: string) => {
    navigate(`/document/${docId}`)
    setRelatedDocs({}) // Clear related docs when navigating
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400">Loading document...</div>
      </div>
    )
  }

  if (!document) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <FileText className="h-16 w-16 text-gray-600 mb-4" />
        <div className="text-gray-400 mb-4">Document not found</div>
        <button
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-blue-400 hover:bg-blue-500 text-black rounded-lg transition-colors font-medium"
        >
          <ArrowLeft className="inline h-4 w-4 mr-2" />
          Back to Dashboard
        </button>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={() => navigate('/')}
          className="flex items-center text-gray-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-5 w-5 mr-2" />
          Back
        </button>
        <div className="flex items-center space-x-2 text-sm text-gray-400">
          {document.type === 'web' ? <Globe className="h-4 w-4" /> : <FileText className="h-4 w-4" />}
          <span>{document.type === 'web' ? 'Web Crawl' : 'Document'}</span>
        </div>
      </div>

      {/* Document Title */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">{document.title}</h1>
        <div className="flex items-center space-x-4 text-sm text-gray-400">
          <div className="flex items-center">
            <Calendar className="h-4 w-4 mr-1" />
            {new Date(document.uploaded_at).toLocaleDateString('en-US', {
              year: 'numeric',
              month: 'long',
              day: 'numeric'
            })}
          </div>
          {document.size && (
            <div>
              {(document.size / 1024).toFixed(1)} KB
            </div>
          )}
          {document.quality?.scores?.overall !== undefined && (
            <div className="flex items-center">
              <BarChart3 className="h-4 w-4 mr-1" />
              Quality: {(document.quality.scores.overall * 100).toFixed(0)}%
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto space-y-6">
        {/* Summary */}
        {document.extraction?.summary && (
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold text-white mb-3">Summary</h2>
            <p className="text-gray-300 leading-relaxed">{document.extraction.summary}</p>
          </div>
        )}

        {/* Concepts/Topics */}
        {document.extraction?.concepts && document.extraction.concepts.length > 0 && (
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold text-white mb-3 flex items-center">
              <Tag className="h-5 w-5 mr-2" />
              Key Topics
            </h2>
            <div className="flex flex-wrap gap-2">
              {document.extraction.concepts.map((concept, idx) => (
                <div key={idx}>
                  <button
                    onClick={() => handleTopicClick(concept)}
                    className="px-3 py-1.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 rounded-lg text-sm transition-colors border border-blue-500/30 hover:border-blue-500/50"
                  >
                    {concept}
                  </button>

                  {/* Related Documents Dropdown */}
                  {relatedDocs[concept] && relatedDocs[concept].length > 0 && (
                    <div className="mt-2 ml-4 bg-gray-900/90 rounded-lg p-3 border border-blue-500/30">
                      <div className="text-xs text-gray-400 mb-2">
                        Related documents ({relatedDocs[concept].length}):
                      </div>
                      <div className="space-y-1">
                        {relatedDocs[concept].slice(0, 5).map((relDoc) => (
                          <button
                            key={relDoc.id}
                            onClick={() => handleRelatedDocClick(relDoc.id)}
                            className="block w-full text-left px-2 py-1 text-sm text-gray-300 hover:text-white hover:bg-gray-800 rounded transition-colors"
                          >
                            {relDoc.type === 'web' ? '🌐' : '📄'} {relDoc.title}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Quality Metrics */}
        {document.quality?.scores && (
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold text-white mb-3">Quality Metrics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-xs text-gray-400 mb-1">Overall</div>
                <div className="text-2xl font-bold text-white">
                  {(document.quality.scores.overall * 100).toFixed(0)}%
                </div>
              </div>
              {document.quality.scores.clarity !== undefined && (
                <div>
                  <div className="text-xs text-gray-400 mb-1">Clarity</div>
                  <div className="text-2xl font-bold text-white">
                    {(document.quality.scores.clarity * 100).toFixed(0)}%
                  </div>
                </div>
              )}
              {document.quality.scores.depth !== undefined && (
                <div>
                  <div className="text-xs text-gray-400 mb-1">Depth</div>
                  <div className="text-2xl font-bold text-white">
                    {(document.quality.scores.depth * 100).toFixed(0)}%
                  </div>
                </div>
              )}
              {document.quality.scores.novelty !== undefined && (
                <div>
                  <div className="text-xs text-gray-400 mb-1">Novelty</div>
                  <div className="text-2xl font-bold text-white">
                    {(document.quality.scores.novelty * 100).toFixed(0)}%
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Curiosity Triggers */}
        {document.research?.curiosity_triggers && document.research.curiosity_triggers.length > 0 && (
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold text-white mb-3">Research Questions</h2>
            <div className="space-y-3">
              {document.research.curiosity_triggers.slice(0, 5).map((trigger, idx) => (
                <div key={idx} className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500/20 text-blue-300 flex items-center justify-center text-xs">
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <p className="text-gray-300">{trigger.question}</p>
                    <div className="text-xs text-gray-500 mt-1">
                      Novelty: {(trigger.novelty_score * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Content Preview */}
        {document.extraction?.content && (
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold text-white mb-3">Content</h2>
            <div className="text-gray-300 leading-relaxed whitespace-pre-wrap max-h-96 overflow-y-auto">
              {document.extraction.content.slice(0, 2000)}
              {document.extraction.content.length > 2000 && (
                <span className="text-gray-500">... (truncated)</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
