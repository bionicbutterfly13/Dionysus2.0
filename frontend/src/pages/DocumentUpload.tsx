import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, Link, AlertCircle, CheckCircle, Globe, Brain } from 'lucide-react'

interface UploadedFile {
  id: string
  name: string
  size: number
  status: 'uploading' | 'processing' | 'completed' | 'error'
  mockData: boolean
}

interface DocumentUploadProps {
  onClose?: () => void
}

export default function DocumentUpload({ onClose }: DocumentUploadProps = {}) {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [selectedMode, setSelectedMode] = useState<'crawl' | 'upload'>('crawl')

  const onDrop = useCallback((acceptedFiles: File[]) => {
    // Simulate file upload and processing
    const newFiles: UploadedFile[] = acceptedFiles.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      status: 'uploading',
      mockData: true
    }))

    setUploadedFiles(prev => [...prev, ...newFiles])

    // Simulate processing stages
    newFiles.forEach((file, index) => {
      setTimeout(() => {
        setUploadedFiles(prev => 
          prev.map(f => f.id === file.id ? { ...f, status: 'processing' } : f)
        )
      }, 1000 + index * 500)

      setTimeout(() => {
        setUploadedFiles(prev => 
          prev.map(f => f.id === file.id ? { ...f, status: 'completed' } : f)
        )
      }, 3000 + index * 500)
    })
  }, [])

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
                <span className="text-white font-medium">Crawl Website</span>
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
                <span className="text-white font-medium">Upload Document</span>
              </div>
              <p className="text-gray-400 text-sm">Add local files</p>
            </button>
          </div>

          {selectedMode === 'crawl' ? (
            <>
              {/* Website URL input */}
              <div className="mb-6">
                <label className="block text-white text-sm font-medium mb-2">Website URL</label>
                <input
                  type="url"
                  placeholder="https://docs.example.com or https://github.com/..."
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:border-blue-400 focus:outline-none"
                />
                <p className="text-gray-500 text-xs mt-2">Enter the URL of a website you want to crawl for knowledge</p>
              </div>

              {/* Knowledge Type */}
              <div className="mb-6">
                <label className="block text-white text-sm font-medium mb-2">Knowledge Type</label>
                <div className="grid grid-cols-2 gap-3">
                  {/* Technical - selected */}
                  <button className="p-4 bg-gray-800 border-2 border-blue-500 rounded-lg text-left">
                    <div className="flex items-center mb-1">
                      <span className="text-blue-400 mr-2">⚡</span>
                      <span className="text-blue-400 font-medium">Technical</span>
                      <span className="ml-auto text-blue-400">✓</span>
                    </div>
                    <p className="text-gray-400 text-xs">Code, APIs, dev docs</p>
                  </button>

                  {/* Business */}
                  <button className="p-4 bg-gray-800 border border-gray-600 rounded-lg text-left hover:border-gray-500">
                    <div className="flex items-center mb-1">
                      <span className="text-gray-400 mr-2">📊</span>
                      <span className="text-white font-medium">Business</span>
                    </div>
                    <p className="text-gray-400 text-xs">Guides, policies, general</p>
                  </button>
                </div>
                <p className="text-gray-500 text-xs mt-2">Choose the type that best describes your content</p>
              </div>

              {/* Crawl Depth */}
              <div className="mb-6">
                <div className="flex items-center mb-2">
                  <label className="text-white text-sm font-medium">Crawl Depth</label>
                  <span className="ml-2 text-gray-500 text-xs">ⓘ</span>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  <button className="p-3 bg-gray-800 border border-gray-600 rounded text-center hover:border-gray-500">
                    <div className="text-white font-medium">1</div>
                    <div className="text-gray-400 text-xs">level</div>
                  </button>
                  {/* Level 2 - selected */}
                  <button className="p-3 bg-gray-800 border-2 border-blue-500 rounded text-center">
                    <div className="text-blue-400 font-medium">2</div>
                    <div className="text-gray-400 text-xs">levels</div>
                    <span className="text-blue-400 text-xs">✓</span>
                  </button>
                  <button className="p-3 bg-gray-800 border border-gray-600 rounded text-center hover:border-gray-500">
                    <div className="text-white font-medium">3</div>
                    <div className="text-gray-400 text-xs">levels</div>
                  </button>
                  <button className="p-3 bg-gray-800 border border-gray-600 rounded text-center hover:border-gray-500">
                    <div className="text-white font-medium">5</div>
                    <div className="text-gray-400 text-xs">levels</div>
                  </button>
                </div>
                <p className="text-gray-500 text-xs mt-2">Higher levels crawl deeper into the website structure</p>
              </div>

              {/* Tags */}
              <div className="mb-8">
                <label className="block text-white text-sm font-medium mb-2">Tags</label>
                <input
                  type="text"
                  placeholder="Add tags like 'api', 'documentation', 'guide'..."
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:border-blue-400 focus:outline-none"
                />
                <p className="text-gray-500 text-xs mt-1">Press Enter or comma to add tags • Backspace to remove last tag</p>
                <p className="text-gray-500 text-xs">0/10 tags used</p>
              </div>

              {/* Start Crawling Button */}
              <button className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors">
                <Globe className="inline h-4 w-4 mr-2" />
                Start Crawling
              </button>
            </>
          ) : (
            <>
              {/* Upload Document Interface */}
              <div className="mb-8">
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
                    Supports PDF, DOC, TXT, MD files
                  </p>
                </div>
              </div>

              {/* Upload Button */}
              <button className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors">
                <Upload className="inline h-4 w-4 mr-2" />
                Upload Documents
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
