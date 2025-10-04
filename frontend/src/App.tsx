import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import DocumentUpload from './pages/DocumentUpload'
import DocumentDetail from './pages/DocumentDetail'
import KnowledgeGraph from './pages/KnowledgeGraph'
import KnowledgeBase from './pages/KnowledgeBase'
import CuriosityMissions from './pages/CuriosityMissions'
import ThoughtSeedMonitor from './pages/ThoughtSeedMonitor'
import Settings from './pages/Settings'
import VisualizationStream from './components/VisualizationStream'

function App() {
  return (
    <div className="min-h-screen">
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<DocumentUpload />} />
          <Route path="/document/:id" element={<DocumentDetail />} />
          <Route path="/knowledge-base" element={<KnowledgeBase />} />
          <Route path="/knowledge-graph" element={<KnowledgeGraph />} />
          <Route path="/thoughtseed" element={<ThoughtSeedMonitor />} />
          <Route path="/curiosity" element={<CuriosityMissions />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>

      <VisualizationStream />
    </div>
  )
}

export default App
