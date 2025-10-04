import { useState, useCallback, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, Link, AlertCircle, CheckCircle, Globe, Brain, XCircle } from 'lucide-react'

interface UploadedFile {
  id: string
  name: string
  size: number
  status: 'uploading' | 'processing' | 'completed' | 'error'
  mockData: boolean
  progress?: number
  // Consciousness processing results
  extraction?: {
    concepts: string[]
    chunks: number
    summary?: any
  }
  consciousness?: {
    basins_created: number
    thoughtseeds_generated: number
    active_inference?: any
  }
  research?: {
    curiosity_triggers: Array<{
      concept: string
      prediction_error: number
      priority: string
    }>
    exploration_plan?: any
  }
  quality?: {
    scores: {
      overall: number
      concept_extraction?: number
      consciousness_integration?: number
    }
    insights?: any[]
  }
  workflow?: {
    iterations: number
    messages: string[]
  }
}

interface DocumentUploadProps {
  onClose?: () => void
}

interface HealthStatus {
  overall_status: string
  can_upload: boolean
  can_crawl: boolean
  errors: string[]
  services: Record<string, {
    name: string
    status: string
    message: string
  }>
}

export default function DocumentUpload({ onClose }: DocumentUploadProps = {}) {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [selectedMode, setSelectedMode] = useState<'crawl' | 'upload'>('crawl')
  const [crawlUrl, setCrawlUrl] = useState('')
  const [crawlDepth, setCrawlDepth] = useState(2)
  const [isCrawling, setIsCrawling] = useState(false)
  const [healthError, setHealthError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Check system health before upload
  const checkSystemHealth = async (): Promise<HealthStatus | null> => {
    try {
      const response = await fetch('/api/health')
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.statusText}`)
      }
      const health = await response.json()
      return health
    } catch (error) {
      console.error('[HEALTH] Check failed:', error)
      return null
    }
  }

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    // Clear previous health errors
    setHealthError(null)

    // Check system health BEFORE starting upload
    console.log('[HEALTH] Checking system health before upload...')
    const health = await checkSystemHealth()

    if (!health) {
      setHealthError('⚠️ Cannot connect to backend health check. Backend may be down.')
      return
    }

    if (!health.can_upload) {
      // Show specific error messages from health check
      const errorMsg = health.errors.join('\n')
      setHealthError(errorMsg)
      console.error('[HEALTH] Upload blocked:', health.errors)
      return
    }

    console.log('[HEALTH] ✅ All systems healthy, proceeding with upload')

    // Create file entries with uploading status
    const newFiles: UploadedFile[] = acceptedFiles.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      status: 'uploading',
      mockData: false,
      progress: 0
    }))

    setUploadedFiles(prev => [...prev, ...newFiles])

    // Upload files individually to show individual progress
    for (const file of acceptedFiles) {
      const fileEntry = newFiles.find(f => f.name === file.name)
      if (!fileEntry) continue

      try {
        const formData = new FormData()
        formData.append('files', file)  // Backend expects 'files' array

        // Simulate upload progress
        const progressInterval = setInterval(() => {
          setUploadedFiles(prev =>
            prev.map(f => f.id === fileEntry.id ?
              { ...f, progress: Math.min((f.progress || 0) + Math.random() * 30, 95) } : f
            )
          )
        }, 200)

        const response = await fetch('/api/v1/documents', {
          method: 'POST',
          body: formData,
        })

        clearInterval(progressInterval)

        if (response.ok) {
          const result = await response.json()
          console.log('[UPLOAD] Backend response:', result)
          console.log('[DAEDALUS] Processing result:', result.documents?.[0])

          const doc = result.documents?.[0]
          if (!doc) throw new Error('No document in response')

          // Map backend response to UI format
          const uploadData = {
            extraction: doc.extraction || { concepts: [], chunks: 0 },
            consciousness: doc.consciousness || { basins_created: 0, thoughtseeds_generated: 0 },
            research: doc.research || { curiosity_triggers: [] },
            quality: doc.quality || { scores: { overall: 0 } },
            workflow: doc.workflow || { iterations: 0, messages: [] }
          }

          // Complete upload progress and move to processing
          setUploadedFiles(prev =>
            prev.map(f => f.id === fileEntry.id ?
              { ...f, progress: 100, status: 'processing' } : f
            )
          )

          // Complete processing with upload data
          setTimeout(() => {
            setUploadedFiles(prev =>
              prev.map(f => f.id === fileEntry.id ?
                {
                  ...f,
                  status: 'completed',
                  extraction: uploadData.extraction,
                  consciousness: uploadData.consciousness,
                  research: uploadData.research,
                  quality: uploadData.quality,
                  workflow: uploadData.workflow
                } : f
              )
            )
          }, 1500)
        } else {
          throw new Error(`Upload failed: ${response.statusText}`)
        }
      } catch (error) {
        console.error('Upload error:', error)
        setUploadedFiles(prev => 
          prev.map(f => f.id === fileEntry.id ? 
            { ...f, status: 'error', progress: 0 } : f
          )
        )
      }
    }
  }, [])

  // Handle web crawl
  const handleStartCrawl = async () => {
    if (!crawlUrl.trim()) {
      setHealthError('⚠️ Please enter a valid URL to crawl')
      return
    }

    // Clear previous errors
    setHealthError(null)

    // Check system health BEFORE starting crawl
    console.log('[HEALTH] Checking system health before crawl...')
    const health = await checkSystemHealth()

    if (!health) {
      setHealthError('⚠️ Cannot connect to backend health check. Backend may be down.')
      return
    }

    if (!health.can_crawl) {
      const errorMsg = health.errors.join('\n')
      setHealthError(errorMsg)
      console.error('[HEALTH] Crawl blocked:', health.errors)
      return
    }

    console.log('[HEALTH] ✅ All systems healthy, proceeding with crawl')

    // Close modal immediately
    if (onClose) {
      onClose()
    }

    // Start crawl in background - fire and forget
    fetch('/api/crawl', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url: crawlUrl,
        max_depth: crawlDepth,
        tags: ['web_crawl', `depth_${crawlDepth}`]
      }),
    }).then(response => {
      if (response.ok) {
        console.log('[CRAWL] Started successfully - check sidebar for progress')
      } else {
        console.error('[CRAWL] Failed to start')
      }
    }).catch(error => {
      console.error('[CRAWL] Error:', error)
    })
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/markdown': ['.md'],
      'text/plain': ['.txt'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    multiple: true
  })

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'uploading':
        return <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
      case 'processing':
        return <div className="animate-pulse h-4 w-4 bg-yellow-400 rounded"></div>
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-600" />
    }
  }

  const getStatusText = (status: UploadedFile['status']) => {
    switch (status) {
      case 'uploading':
        return 'Uploading...'
      case 'processing':
        return 'Processing with consciousness emulator...'
      case 'completed':
        return 'Processed and stored in knowledge graph'
      case 'error':
        return 'Processing failed'
    }
  }

  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center">
      {/* Modal-style container exactly like Archon */}
      <div className="w-full max-w-2xl bg-gray-900 border border-blue-500 rounded-lg">
        {/* Header with blue gradient border */}
        <div className="border-b border-blue-500 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-medium text-blue-400 mb-2">Add Knowledge</h1>
              <p className="text-gray-400 text-sm">Crawl websites or upload documents to expand your knowledge base.</p>
            </div>
            {onClose && (
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            )}
          </div>
        </div>

        <div className="p-6">
          {/* Two main option buttons */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            {/* Crawl Website */}
            <button
              onClick={() => setSelectedMode('crawl')}
              className={`p-6 bg-gray-800 rounded-lg text-left transition-all ${
                selectedMode === 'crawl'
                  ? 'border-2 border-blue-500'
                  : 'border border-gray-600 hover:border-gray-500'
              }`}
            >
              <div className="flex items-center mb-2">
                <Globe className={`h-5 w-5 mr-2 ${selectedMode === 'crawl' ? 'text-blue-400' : 'text-gray-400'}`} />
                <span className="text-white font-medium">Crawl Websites</span>
              </div>
              <p className="text-gray-400 text-sm">Scan web pages</p>
            </button>

            {/* Upload Document */}
            <button
              onClick={() => setSelectedMode('upload')}
              className={`p-6 bg-gray-800 rounded-lg text-left transition-all ${
                selectedMode === 'upload'
                  ? 'border-2 border-blue-500'
                  : 'border border-gray-600 hover:border-gray-500'
              }`}
            >
              <div className="flex items-center mb-2">
                <Upload className={`h-5 w-5 mr-2 ${selectedMode === 'upload' ? 'text-blue-400' : 'text-gray-400'}`} />
                <span className="text-white font-medium">Upload Documents</span>
              </div>
              <p className="text-gray-400 text-sm">Add local files</p>
            </button>
          </div>

          {selectedMode === 'crawl' ? (
            <>
              {/* Health Error Alert */}
              {healthError && (
                <div className="mb-6 p-4 bg-red-900/30 border border-red-500 rounded-lg">
                  <div className="flex items-start">
                    <XCircle className="h-5 w-5 text-red-400 mr-3 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <h3 className="text-red-400 font-medium mb-2">Crawl Failed</h3>
                      <div className="text-red-300 text-sm whitespace-pre-line mb-3">
                        {healthError}
                      </div>
                      <button
                        onClick={() => setHealthError(null)}
                        className="mt-3 px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Website URL input */}
              <div className="mb-6">
                <label className="block text-white text-sm font-medium mb-2">Website URL</label>
                <input
                  type="url"
                  value={crawlUrl}
                  onChange={(e) => setCrawlUrl(e.target.value)}
                  placeholder="https://docs.example.com or https://github.com/..."
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:border-blue-400 focus:outline-none"
                  disabled={isCrawling}
                />
                <p className="text-gray-500 text-xs mt-2">Enter the URL of a website you want to crawl for knowledge</p>
              </div>

              {/* Crawl Depth */}
              <div className="mb-8">
                <div className="flex items-center mb-2">
                  <label className="text-white text-sm font-medium">Crawl Depth</label>
                  <span className="ml-2 text-gray-500 text-xs">ⓘ Higher levels crawl deeper</span>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {[1, 2, 3].map((depth) => (
                    <button
                      key={depth}
                      onClick={() => setCrawlDepth(depth)}
                      disabled={isCrawling}
                      className={`p-3 bg-gray-800 rounded text-center transition-all ${
                        crawlDepth === depth
                          ? 'border-2 border-blue-500'
                          : 'border border-gray-600 hover:border-gray-500'
                      } ${isCrawling ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <div className={`font-medium ${crawlDepth === depth ? 'text-blue-400' : 'text-white'}`}>
                        {depth}
                      </div>
                      <div className="text-gray-400 text-xs">{depth === 1 ? 'level' : 'levels'}</div>
                      {crawlDepth === depth && <span className="text-blue-400 text-xs">✓</span>}
                    </button>
                  ))}
                </div>
              </div>

              {/* Start Crawling Button */}
              <button
                onClick={handleStartCrawl}
                disabled={isCrawling || !crawlUrl.trim()}
                className={`w-full py-3 text-white font-medium rounded-lg transition-colors ${
                  isCrawling || !crawlUrl.trim()
                    ? 'bg-gray-600 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                <Globe className="inline h-4 w-4 mr-2" />
                {isCrawling ? 'Crawling...' : 'Start Crawling'}
              </button>
            </>
          ) : (
            <>
              {/* Health Error Alert */}
              {healthError && (
                <div className="mb-6 p-4 bg-red-900/30 border border-red-500 rounded-lg">
                  <div className="flex items-start">
                    <XCircle className="h-5 w-5 text-red-400 mr-3 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <h3 className="text-red-400 font-medium mb-2">System Health Check Failed</h3>
                      <div className="text-red-300 text-sm whitespace-pre-line mb-3">
                        {healthError}
                      </div>
                      <p className="text-gray-400 text-xs">
                        Copy this error and paste into your terminal or GPT for help resolving the issue.
                      </p>
                      <button
                        onClick={() => setHealthError(null)}
                        className="mt-3 px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Upload Document Interface */}
              <div className="mb-8">
                {uploadedFiles.length === 0 ? (
                  <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
                      isDragActive
                        ? 'border-blue-400 bg-blue-400/10'
                        : 'border-gray-600 hover:border-blue-400'
                    }`}
                  >
                    <input {...getInputProps()} />
                    <Upload className={`mx-auto h-16 w-16 mb-6 ${isDragActive ? 'text-blue-400' : 'text-gray-400'}`} />
                    <h2 className="text-xl font-medium text-white mb-4">
                      {isDragActive ? 'Drop files here' : 'Drag files here or click to browse'}
                    </h2>
                    <p className="text-gray-400 mb-6">
                      Supports multiple files: PDF, DOC, DOCX, TXT, MD files
                    </p>
                    <div className="text-gray-500 text-sm">
                      <span className="inline-block bg-gray-700 px-2 py-1 rounded mr-2 mb-2">.pdf</span>
                      <span className="inline-block bg-gray-700 px-2 py-1 rounded mr-2 mb-2">.doc</span>
                      <span className="inline-block bg-gray-700 px-2 py-1 rounded mr-2 mb-2">.docx</span>
                      <span className="inline-block bg-gray-700 px-2 py-1 rounded mr-2 mb-2">.txt</span>
                      <span className="inline-block bg-gray-700 px-2 py-1 rounded mr-2 mb-2">.md</span>
                    </div>
                  </div>
                ) : (
                  <div className="border-2 border-dashed border-gray-600 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-medium text-white">Upload Documents</h3>
                      <div
                        {...getRootProps()}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg cursor-pointer transition-colors"
                      >
                        <input {...getInputProps()} />
                        + Add More Files
                      </div>
                    </div>
                    <div className="space-y-3">
                      {uploadedFiles.map((file) => (
                        <div key={file.id} className="bg-gray-800 rounded-lg p-3">
                          {/* File header */}
                          <div className="flex items-center mb-2">
                            <FileText className="h-4 w-4 text-blue-400 mr-2 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <p className="text-white text-sm truncate">{file.name}</p>
                              {file.status === 'uploading' && file.progress !== undefined && (
                                <div className="w-full bg-gray-700 rounded-full h-1 mt-1">
                                  <div
                                    className="bg-blue-600 h-1 rounded-full transition-all duration-300"
                                    style={{ width: `${file.progress}%` }}
                                  ></div>
                                </div>
                              )}
                            </div>
                            <div className="ml-2 flex items-center">
                              {file.status === 'completed' ? (
                                <CheckCircle className="h-4 w-4 text-green-400" />
                              ) : file.status === 'error' ? (
                                <AlertCircle className="h-4 w-4 text-red-400" />
                              ) : (
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
                              )}
                              {/* Only show delete button when file is done (completed or error) */}
                              {(file.status === 'completed' || file.status === 'error') && (
                                <button
                                  onClick={() => setUploadedFiles(prev => prev.filter(f => f.id !== file.id))}
                                  className="ml-1 text-red-400 hover:text-red-300"
                                  title="Remove from list"
                                >
                                  ✕
                                </button>
                              )}
                            </div>
                          </div>

                          {/* Processing results - ONLY show when we have REAL data */}
                          {file.status === 'completed' && (
                            <div className="mt-3 pt-3 border-t border-gray-700 space-y-2">
                              {/* Only show concepts if we actually extracted some */}
                              {file.extraction?.concepts && file.extraction.concepts.length > 0 && (
                                <div className="flex items-start text-xs">
                                  <span className="text-gray-400 w-32 flex-shrink-0">Concepts:</span>
                                  <div className="flex-1">
                                    <span className="text-blue-400 font-medium">{file.extraction.concepts.length}</span>
                                    <p className="text-gray-500 mt-1">
                                      {file.extraction.concepts.slice(0, 5).join(', ')}
                                      {file.extraction.concepts.length > 5 && '...'}
                                    </p>
                                  </div>
                                </div>
                              )}

                              {/* Only show basins/seeds if they exist */}
                              {file.consciousness && (file.consciousness.basins_created > 0 || file.consciousness.thoughtseeds_generated > 0) && (
                                <div className="flex items-center text-xs">
                                  <span className="text-gray-400 w-32 flex-shrink-0">Consciousness:</span>
                                  <div className="flex items-center flex-1">
                                    <Brain className="h-3 w-3 text-purple-400 mr-1" />
                                    <span className="text-purple-400">
                                      {file.consciousness.basins_created} basins
                                    </span>
                                    <span className="mx-1 text-gray-600">•</span>
                                    <span className="text-purple-400">
                                      {file.consciousness.thoughtseeds_generated} seeds
                                    </span>
                                  </div>
                                </div>
                              )}

                              {/* Only show quality if > 0 */}
                              {file.quality?.scores?.overall && file.quality.scores.overall > 0 && (
                                <div className="flex items-center text-xs">
                                  <span className="text-gray-400 w-32 flex-shrink-0">Quality:</span>
                                  <span className={`font-medium ${
                                    file.quality.scores.overall >= 0.8 ? 'text-green-400' :
                                    file.quality.scores.overall >= 0.6 ? 'text-yellow-400' :
                                    'text-red-400'
                                  }`}>
                                    {(file.quality.scores.overall * 100).toFixed(0)}%
                                  </span>
                                </div>
                              )}

                              {/* Only show curiosity if triggers exist */}
                              {file.research?.curiosity_triggers && file.research.curiosity_triggers.length > 0 && (
                                <div className="flex items-start text-xs">
                                  <span className="text-gray-400 w-32 flex-shrink-0">Curiosity:</span>
                                  <div className="flex-1">
                                    <span className="text-orange-400">
                                      {file.research.curiosity_triggers.length} trigger{file.research.curiosity_triggers.length !== 1 ? 's' : ''}
                                    </span>
                                    <p className="text-gray-500 mt-1">
                                      {file.research.curiosity_triggers.slice(0, 2).map(t => t.concept).join(', ')}
                                      {file.research.curiosity_triggers.length > 2 && '...'}
                                    </p>
                                  </div>
                                </div>
                              )}

                              {/* Simple success message if no data yet */}
                              {(!file.extraction?.concepts || file.extraction.concepts.length === 0) && (
                                <div className="text-xs text-gray-400 italic">
                                  ✓ File uploaded successfully
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>


              {/* Close Button - can close even while uploading */}
              <button
                onClick={() => {
                  if (onClose) {
                    onClose()
                  }
                }}
                className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
              >
                {uploadedFiles.some(f => f.status === 'uploading' || f.status === 'processing') ? (
                  <>
                    <Upload className="inline h-4 w-4 mr-2 animate-pulse" />
                    Close (uploads continue in background)
                  </>
                ) : (
                  <>
                    <Upload className="inline h-4 w-4 mr-2" />
                    Close
                  </>
                )}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
