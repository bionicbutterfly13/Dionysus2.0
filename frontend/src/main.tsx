import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'
import './index.css'

// Load Flux configuration
async function loadFluxConfig() {
  try {
    const response = await fetch('/configs/flux.yaml')
    const yamlText = await response.text()
    // Parse YAML and apply settings
    console.log('Flux config loaded:', yamlText)
    return yamlText
  } catch (error) {
    console.error('Failed to load flux.yaml:', error)
    return null
  }
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

// Initialize app with flux config
loadFluxConfig().then(() => {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </QueryClientProvider>
    </React.StrictMode>,
  )
})
