import React, { useState } from 'react'
import { Search, Plus, Grid3X3, List, Globe, FileText, ExternalLink, Tag, Clock, MoreHorizontal } from 'lucide-react'
import DocumentUpload from './DocumentUpload'

interface KnowledgeItem {
  id: string
  title: string
  type: 'webpage' | 'document'
  url?: string
  filename?: string
  tags: string[]
  updated: string
  pageCount?: number
  views: number
  category: 'Technical' | 'Business'
}

interface CrawlOperation {
  id: string
  title: string
  status: 'crawling' | 'completed'
  progress: number
  currentRange: string
  totalPages: number
  pagesCrawled: number
  url: string
  depth: number
}

export default function KnowledgeBase() {
  const [showAddKnowledgeModal, setShowAddKnowledgeModal] = useState(false)
  const [crawlOperations] = useState<CrawlOperation[]>([
    {
      id: '1',
      title: 'Crawling URLs 251-300 of 576 at depth 4',
      status: 'crawling',
      progress: 11,
      currentRange: '251-300',
      totalPages: 576,
      pagesCrawled: 549,
      url: 'https://langchain-ai.github.io/langgraph/',
      depth: 4
    },
    {
      id: '2',
      title: 'Crawling URLs 451-500 of 609 at depth 3',
      status: 'crawling',
      progress: 9,
      currentRange: '451-500',
      totalPages: 609,
      pagesCrawled: 458,
      url: 'https://arxiv.org/html/2312.07547v1',
      depth: 3
    },
    {
      id: '3',
      title: 'Crawling URLs 1-50 of 99 at depth 2',
      status: 'crawling',
      progress: 9,
      currentRange: '1-50',
      totalPages: 99,
      pagesCrawled: 1,
      url: 'https://www.biorxiv.org/content/10.1101/2024.08.18.608439v1.full',
      depth: 2
    },
    {
      id: '4',
      title: 'Crawling URLs 151-170 of 170 at depth 2',
      status: 'crawling',
      progress: 9,
      currentRange: '151-170',
      totalPages: 170,
      pagesCrawled: 151,
      url: 'https://github.com/MODSetter/SurfSense.git',
      depth: 2
    }
  ])

  const [knowledgeItems] = useState<KnowledgeItem[]>([
    {
      id: '1',
      title: 'Frontiersin',
      type: 'webpage',
      url: 'frontiersin.org',
      tags: [],
      updated: '9/25/2025',
      pageCount: 100,
      views: 0,
      category: 'Technical'
    },
    {
      id: '2',
      title: 'Blog Futuresmart - Langgraph-Tutorial-For-Beginners',
      type: 'webpage',
      url: 'blog.futuresmart.ai',
      tags: [],
      updated: '9/25/2025',
      pageCount: 86,
      views: 0,
      category: 'Technical'
    },
    {
      id: '3',
      title: 'Recurrent_motifs_as_resonant_attractor_s.pdf',
      type: 'document',
      filename: 'Recurrent_motifs_as_resonant_attractor_s.pdf',
      tags: [],
      updated: '9/25/2025',
      pageCount: 21,
      views: 0,
      category: 'Technical'
    },
    {
      id: '4',
      title: 'Recurrent_motifs_as_resonant_attractor_s.pdf',
      type: 'document',
      filename: 'Recurrent_motifs_as_resonant_attractor_s.pdf',
      tags: [],
      updated: '9/25/2025',
      pageCount: 0,
      views: 0,
      category: 'Technical'
    },
    {
      id: '5',
      title: 'Sciencedirect',
      type: 'webpage',
      url: 'sciencedirect.com',
      tags: [],
      updated: '9/25/2025',
      pageCount: 1,
      views: 0,
      category: 'Technical'
    },
    {
      id: '6',
      title: '2401.08438v2.pdf',
      type: 'document',
      filename: '2401.08438v2.pdf',
      tags: [],
      updated: '9/25/2025',
      pageCount: 9,
      views: 0,
      category: 'Technical'
    },
    {
      id: '7',
      title: 'Inpsy Fss Muni Cz',
      type: 'webpage',
      url: 'inpsy.fss.muni.cz',
      tags: [],
      updated: '9/25/2025',
      pageCount: 113,
      views: 0,
      category: 'Technical'
    },
    {
      id: '8',
      title: 'roots.pdf',
      type: 'document',
      filename: 'roots.pdf',
      tags: [],
      updated: '9/25/2025',
      pageCount: 300,
      views: 0,
      category: 'Technical'
    }
  ])

  return (
    <div className="min-h-screen bg-black text-white p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-purple-600 rounded flex items-center justify-center">
              <span className="text-white text-sm">📚</span>
            </div>
            <h1 className="text-xl font-medium text-white">Knowledge Base</h1>
            <span className="bg-gray-700 text-gray-300 px-2 py-1 rounded text-sm">23 items</span>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -y-1/2 h-4 w-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search knowledge base..."
              className="pl-10 pr-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none w-80"
            />
          </div>

          <button className="p-2 bg-gray-800 border border-gray-700 rounded hover:border-gray-600">
            <Plus className="h-4 w-4 text-gray-400" />
          </button>
          <button className="p-2 bg-gray-800 border border-gray-700 rounded hover:border-gray-600">
            <ExternalLink className="h-4 w-4 text-gray-400" />
          </button>
          <button className="p-2 bg-gray-800 border border-gray-700 rounded hover:border-gray-600">
            <Grid3X3 className="h-4 w-4 text-gray-400" />
          </button>
          <button className="p-2 bg-gray-800 border border-gray-700 rounded hover:border-gray-600">
            <List className="h-4 w-4 text-gray-400" />
          </button>

          <button
            onClick={() => setShowAddKnowledgeModal(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center space-x-2"
          >
            <Plus className="h-4 w-4" />
            <span>Add Knowledge</span>
          </button>
        </div>
      </div>

      {/* Active Operations */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium text-white">Active Operations (8)</h2>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            <span className="text-sm text-gray-400">Live Updates</span>
          </div>
        </div>

        <div className="space-y-3">
          {crawlOperations.map((operation) => (
            <div key={operation.id} className="bg-gray-900/60 border border-gray-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                  <h3 className="text-white font-medium">{operation.title}</h3>
                </div>
                <button className="px-3 py-1 bg-gray-800 border border-gray-600 rounded text-gray-300 text-sm hover:border-gray-500 flex items-center space-x-1">
                  <MoreHorizontal className="h-3 w-3" />
                  <span>Stop</span>
                </button>
              </div>

              <div className="flex items-center space-x-3 mb-4">
                <span className="px-2 py-1 bg-blue-900/50 text-blue-400 text-xs rounded">Crawling</span>
                <span className="px-2 py-1 bg-gray-800 text-gray-300 text-xs rounded">Web Crawl</span>
              </div>

              <div className="flex items-center space-x-6 mb-3">
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">Progress</span>
                    <span className="text-xs text-blue-400">{operation.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-1.5">
                    <div
                      className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                      style={{ width: `${operation.progress}%` }}
                    ></div>
                  </div>
                </div>

                <div className="text-center">
                  <div className="text-xl font-bold text-blue-400">{operation.pagesCrawled}</div>
                  <div className="text-xs text-gray-400">Pages</div>
                </div>
              </div>

              <div className="text-sm text-gray-500">
                URL: {operation.url}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Knowledge Items Grid */}
      <div className="grid grid-cols-4 gap-4">
        {knowledgeItems.map((item) => (
          <div
            key={item.id}
            className={`p-4 rounded-lg border ${
              item.type === 'webpage'
                ? 'bg-gray-900/60 border-blue-500/30'
                : 'bg-purple-900/20 border-purple-500/30'
            }`}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-2">
                {item.type === 'webpage' ? (
                  <Globe className="h-4 w-4 text-blue-400" />
                ) : (
                  <FileText className="h-4 w-4 text-purple-400" />
                )}
                <span className="text-xs text-gray-400 capitalize">{item.type}</span>
                <span className="text-xs text-blue-400">{item.category}</span>
              </div>
              <button className="text-gray-500 hover:text-gray-300">
                <MoreHorizontal className="h-4 w-4" />
              </button>
            </div>

            <h3 className="text-white text-sm font-medium mb-3 line-clamp-2">{item.title}</h3>

            {item.url && (
              <div className="flex items-center space-x-1 mb-3">
                <ExternalLink className="h-3 w-3 text-gray-500" />
                <span className="text-xs text-gray-500">{item.url}</span>
              </div>
            )}

            {item.filename && (
              <div className="flex items-center space-x-1 mb-3">
                <FileText className="h-3 w-3 text-gray-500" />
                <span className="text-xs text-gray-500">{item.filename}</span>
              </div>
            )}

            <button className="w-full mb-3">
              <div className="flex items-center space-x-1">
                <Tag className="h-3 w-3 text-gray-500" />
                <span className="text-xs text-gray-500">Tags</span>
              </div>
            </button>

            <div className="flex items-center justify-between text-xs text-gray-500">
              <div className="flex items-center space-x-1">
                <Clock className="h-3 w-3" />
                <span>Updated: {item.updated}</span>
              </div>
              <div className="flex items-center space-x-3">
                {item.pageCount > 0 && (
                  <div className="flex items-center space-x-1">
                    <FileText className="h-3 w-3 text-orange-400" />
                    <span className="text-orange-400">{item.pageCount}</span>
                  </div>
                )}
                <div className="flex items-center space-x-1">
                  <ExternalLink className="h-3 w-3" />
                  <span>{item.views}</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Add Knowledge Modal */}
      {showAddKnowledgeModal && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
          <DocumentUpload onClose={() => setShowAddKnowledgeModal(false)} />
        </div>
      )}
    </div>
  )
}