import { deriveHealthBlocker, limitFilesForBatch, MAX_FILES_PER_BATCH } from '../../pages/DocumentUpload'

describe('DocumentUpload helpers', () => {
  describe('limitFilesForBatch', () => {
    it('limits batch to max files and reports overflow', () => {
      const files = Array.from({ length: MAX_FILES_PER_BATCH + 3 }, (_, idx) => new File(['content'], `file-${idx}.txt`, { type: 'text/plain' }))

      const { batch, overflow } = limitFilesForBatch(files)

      expect(batch).toHaveLength(MAX_FILES_PER_BATCH)
      expect(overflow).toBe(3)
    })

    it('handles files within batch limit', () => {
      const files = Array.from({ length: 5 }, (_, idx) => new File(['content'], `file-${idx}.txt`, { type: 'text/plain' }))

      const { batch, overflow } = limitFilesForBatch(files)

      expect(batch).toHaveLength(5)
      expect(overflow).toBe(0)
    })

    it('handles exactly max files', () => {
      const files = Array.from({ length: MAX_FILES_PER_BATCH }, (_, idx) => new File(['content'], `file-${idx}.txt`, { type: 'text/plain' }))

      const { batch, overflow } = limitFilesForBatch(files)

      expect(batch).toHaveLength(MAX_FILES_PER_BATCH)
      expect(overflow).toBe(0)
    })

    it('handles empty file array', () => {
      const { batch, overflow } = limitFilesForBatch([])

      expect(batch).toHaveLength(0)
      expect(overflow).toBe(0)
    })
  })

  describe('deriveHealthBlocker', () => {
    it('returns null when health is healthy', () => {
      const blocker = deriveHealthBlocker({
        overall_status: 'healthy',
        can_upload: true,
        can_crawl: true,
        errors: [],
        services: {
          daedalus: {
            name: 'Daedalus',
            status: 'healthy',
            message: 'ok',
            required_for: ['upload']
          }
        }
      })

      expect(blocker).toBeNull()
    })

    it('blocks when daedalus is down', () => {
      const blocker = deriveHealthBlocker({
        overall_status: 'down',
        can_upload: false,
        can_crawl: false,
        errors: ['Daedalus offline'],
        services: {
          daedalus: {
            name: 'Daedalus',
            status: 'down',
            message: 'offline',
            required_for: ['upload']
          }
        }
      })

      expect(blocker).toBe('Local processing is offline. Please try again later.')
    })

    it('handles null health result', () => {
      const blocker = deriveHealthBlocker(null)
      expect(blocker).toContain('Unable to reach local services')
    })

    it('blocks when can_upload is false', () => {
      const blocker = deriveHealthBlocker({
        overall_status: 'degraded',
        can_upload: false,
        can_crawl: true,
        errors: ['Upload service unavailable'],
        services: {
          daedalus: {
            name: 'Daedalus',
            status: 'healthy',
            message: 'ok',
            required_for: ['upload']
          }
        }
      })

      expect(blocker).toContain('Upload service unavailable')
    })

    it('blocks when daedalus is degraded', () => {
      const blocker = deriveHealthBlocker({
        overall_status: 'degraded',
        can_upload: true,
        can_crawl: true,
        errors: [],
        services: {
          daedalus: {
            name: 'Daedalus',
            status: 'degraded',
            message: 'running slowly',
            required_for: ['upload']
          }
        }
      })

      expect(blocker).toBe('Local processing is offline. Please try again later.')
    })
  })
})
