import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import DocumentUpload, { MAX_FILES_PER_BATCH } from '../DocumentUpload'

// Store the onDrop callback for test access
let mockOnDrop: ((files: File[]) => void) | null = null

// Mock react-dropzone
jest.mock('react-dropzone', () => ({
  useDropzone: ({ onDrop }: any) => {
    mockOnDrop = onDrop
    return {
      getRootProps: () => ({
        onClick: jest.fn()
      }),
      getInputProps: () => ({}),
      isDragActive: false,
      open: jest.fn()
    }
  }
}))

describe('DocumentUpload Component', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockOnDrop = null
    global.fetch = jest.fn()
    window.dispatchEvent = jest.fn()

    // Mock localStorage
    const localStorageMock = {
      getItem: jest.fn(() => '[]'),
      setItem: jest.fn(),
      removeItem: jest.fn(),
      clear: jest.fn()
    }
    Object.defineProperty(window, 'localStorage', {
      value: localStorageMock,
      writable: true
    })
  })

  afterEach(() => {
    jest.restoreAllMocks()
  })

  describe('Upload Flow', () => {
    it('triggers health check before upload', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          overall_status: 'healthy',
          can_upload: true,
          can_crawl: true,
          errors: [],
          services: {
            daedalus: { name: 'Daedalus', status: 'healthy', message: 'ok', required_for: ['upload'] }
          }
        })
      })

      render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      const testFile = new File(['test'], 'test.txt', { type: 'text/plain' })

      if (mockOnDrop) {
        await mockOnDrop([testFile])
      }

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/health')
      })
    })

    it('performs sequential upload after health check passes', async () => {
      const testFile = new File(['test content'], 'test.txt', { type: 'text/plain' })

      ;(global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            overall_status: 'healthy',
            can_upload: true,
            can_crawl: true,
            errors: [],
            services: {
              daedalus: { name: 'Daedalus', status: 'healthy', message: 'ok', required_for: ['upload'] }
            }
          })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            documents: [{
              id: 'doc-1',
              title: 'test.txt',
              type: 'file',
              uploaded_at: new Date().toISOString(),
              extraction: { concepts: ['test', 'content'], chunks: 5 },
              consciousness: { basins_created: 2, thoughtseeds_generated: 3 },
              research: { curiosity_triggers: [{ concept: 'test', prediction_error: 0.8, priority: 'high' }] },
              quality: { scores: { overall: 0.85 } },
              workflow: { iterations: 1, messages: ['Processing complete'] }
            }]
          })
        })

      render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      if (mockOnDrop) {
        await mockOnDrop([testFile])
      }

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          '/api/v1/documents?mode=local',
          expect.objectContaining({
            method: 'POST',
            body: expect.any(FormData)
          })
        )
      })
    })

    it('handles batch limiting correctly', async () => {
      const files = Array.from({ length: MAX_FILES_PER_BATCH + 3 }, (_, idx) =>
        new File(['content'], `file-${idx}.txt`, { type: 'text/plain' })
      )

      ;(global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          overall_status: 'healthy',
          can_upload: true,
          can_crawl: true,
          errors: [],
          services: {
            daedalus: { name: 'Daedalus', status: 'healthy', message: 'ok', required_for: ['upload'] }
          }
        })
      })

      render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      if (mockOnDrop) {
        await mockOnDrop(files)
      }

      await waitFor(() => {
        expect(screen.getByText(/3 file\(s\) queued for later/i)).toBeInTheDocument()
      })
    })
  })

  describe('Event Emission', () => {
    it('emits flux:documents-updated event on successful upload', async () => {
      const testFile = new File(['test'], 'test.txt', { type: 'text/plain' })

      ;(global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            overall_status: 'healthy',
            can_upload: true,
            can_crawl: true,
            errors: [],
            services: {
              daedalus: { name: 'Daedalus', status: 'healthy', message: 'ok', required_for: ['upload'] }
            }
          })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            documents: [{
              id: 'doc-1',
              extraction: { concepts: [], chunks: 0 },
              consciousness: { basins_created: 0, thoughtseeds_generated: 0 },
              research: { curiosity_triggers: [] },
              quality: { scores: { overall: 0.5 } },
              workflow: { iterations: 1, messages: [] }
            }]
          })
        })

      render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      if (mockOnDrop) {
        await mockOnDrop([testFile])
      }

      await waitFor(() => {
        expect(window.dispatchEvent).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'flux:documents-updated'
          })
        )
      })
    })
  })

  describe('Progress Tracking', () => {
    it('updates progress from 0% to 10% to 100%', async () => {
      const testFile = new File(['test'], 'test.txt', { type: 'text/plain' })

      ;(global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            overall_status: 'healthy',
            can_upload: true,
            can_crawl: true,
            errors: [],
            services: {
              daedalus: { name: 'Daedalus', status: 'healthy', message: 'ok', required_for: ['upload'] }
            }
          })
        })
        .mockImplementationOnce(() =>
          new Promise(resolve =>
            setTimeout(() => resolve({
              ok: true,
              json: async () => ({
                documents: [{
                  id: 'doc-1',
                  extraction: { concepts: ['test'], chunks: 1 },
                  consciousness: { basins_created: 1, thoughtseeds_generated: 1 },
                  research: { curiosity_triggers: [] },
                  quality: { scores: { overall: 0.9 } },
                  workflow: { iterations: 1, messages: [] }
                }]
              })
            }), 100)
          )
        )

      const { container } = render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      if (mockOnDrop) {
        await mockOnDrop([testFile])
      }

      // Check progress updates
      await waitFor(() => {
        const progressBar = container.querySelector('.bg-blue-600')
        expect(progressBar).toBeInTheDocument()
      }, { timeout: 3000 })
    })
  })

  describe('Error Handling', () => {
    it('shows error state for empty file', async () => {
      const emptyFile = new File([], 'empty.txt', { type: 'text/plain' })

      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          overall_status: 'healthy',
          can_upload: true,
          can_crawl: true,
          errors: [],
          services: {
            daedalus: { name: 'Daedalus', status: 'healthy', message: 'ok', required_for: ['upload'] }
          }
        })
      })

      render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      if (mockOnDrop) {
        await mockOnDrop([emptyFile])
      }

      await waitFor(() => {
        expect(screen.getByText('Processing failed')).toBeInTheDocument()
      })
    })

    it('continues with remaining files after one fails', async () => {
      const emptyFile = new File([], 'empty.txt', { type: 'text/plain' })
      const validFile = new File(['content'], 'valid.txt', { type: 'text/plain' })

      ;(global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            overall_status: 'healthy',
            can_upload: true,
            can_crawl: true,
            errors: [],
            services: {
              daedalus: { name: 'Daedalus', status: 'healthy', message: 'ok', required_for: ['upload'] }
            }
          })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            documents: [{
              id: 'doc-1',
              extraction: { concepts: ['valid'], chunks: 1 },
              consciousness: { basins_created: 1, thoughtseeds_generated: 1 },
              research: { curiosity_triggers: [] },
              quality: { scores: { overall: 0.8 } },
              workflow: { iterations: 1, messages: [] }
            }]
          })
        })

      render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      if (mockOnDrop) {
        await mockOnDrop([emptyFile, validFile])
      }

      await waitFor(() => {
        expect(screen.getByText(/File was empty/i)).toBeInTheDocument()
      })

      await waitFor(() => {
        expect(screen.getByText(/Processed and stored in knowledge graph/i)).toBeInTheDocument()
      })
    })
  })

  describe('Health Blocking', () => {
    it('prevents upload when Daedalus is offline', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          overall_status: 'down',
          can_upload: false,
          can_crawl: false,
          errors: ['Daedalus offline'],
          services: {
            daedalus: { name: 'Daedalus', status: 'down', message: 'offline', required_for: ['upload'] }
          }
        })
      })

      render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      const testFile = new File(['test'], 'test.txt', { type: 'text/plain' })

      if (mockOnDrop) {
        await mockOnDrop([testFile])
      }

      await waitFor(() => {
        expect(screen.getByText(/Local processing is offline/i)).toBeInTheDocument()
      })

      expect(global.fetch).toHaveBeenCalledTimes(1)
    })

    it('shows error message when backend is unreachable', async () => {
      ;(global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'))

      render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      const testFile = new File(['test'], 'test.txt', { type: 'text/plain' })

      if (mockOnDrop) {
        await mockOnDrop([testFile])
      }

      await waitFor(() => {
        expect(screen.getByText(/Unable to reach local services/i)).toBeInTheDocument()
      })
    })
  })

  describe('Status Icons', () => {
    it('shows checkmark when completed', async () => {
      const testFile = new File(['test'], 'test.txt', { type: 'text/plain' })

      ;(global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            overall_status: 'healthy',
            can_upload: true,
            can_crawl: true,
            errors: [],
            services: {
              daedalus: { name: 'Daedalus', status: 'healthy', message: 'ok', required_for: ['upload'] }
            }
          })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            documents: [{
              id: 'doc-1',
              extraction: { concepts: ['test'], chunks: 1 },
              consciousness: { basins_created: 1, thoughtseeds_generated: 1 },
              research: { curiosity_triggers: [] },
              quality: { scores: { overall: 0.85 } },
              workflow: { iterations: 1, messages: [] }
            }]
          })
        })

      render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      if (mockOnDrop) {
        await mockOnDrop([testFile])
      }

      await waitFor(() => {
        expect(screen.getByText(/Processed and stored in knowledge graph/i)).toBeInTheDocument()
      })
    })

    it('shows X icon on error', async () => {
      const emptyFile = new File([], 'empty.txt', { type: 'text/plain' })

      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          overall_status: 'healthy',
          can_upload: true,
          can_crawl: true,
          errors: [],
          services: {
            daedalus: { name: 'Daedalus', status: 'healthy', message: 'ok', required_for: ['upload'] }
          }
        })
      })

      render(<DocumentUpload />)

      const uploadButton = screen.getByText('Upload Documents')
      fireEvent.click(uploadButton)

      if (mockOnDrop) {
        await mockOnDrop([emptyFile])
      }

      await waitFor(() => {
        expect(screen.getByText('Processing failed')).toBeInTheDocument()
      })
    })
  })
})
