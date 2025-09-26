/**
 * TDD Tests for ThoughtSeed Debug Panel
 * ====================================
 *
 * These tests WILL FAIL initially - that's the TDD approach.
 * We write failing tests first, then implement to make them pass.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import ThoughtSeedDebugPanel from '../src/components/ThoughtSeedDebugPanel';

// Mock fetch globally
global.fetch = vi.fn();

describe('ThoughtSeed Debug Panel - TDD Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Component Rendering', () => {
    it('should render debug panel title', () => {
      render(<ThoughtSeedDebugPanel />);

      expect(screen.getByText('ThoughtSeed Workspace Debug Panel')).toBeInTheDocument();
    });

    it('should render control panel with toggle switches', () => {
      render(<ThoughtSeedDebugPanel />);

      expect(screen.getByText('Controls')).toBeInTheDocument();
      expect(screen.getByText('Auto-refresh (2s)')).toBeInTheDocument();
      expect(screen.getByText('Refresh Now')).toBeInTheDocument();
      expect(screen.getByText('Clear Logs')).toBeInTheDocument();
    });

    it('should show empty state when no logs available', () => {
      // Mock empty API response
      (fetch as any).mockResolvedValue({
        json: () => Promise.resolve({ logs: [] })
      });

      render(<ThoughtSeedDebugPanel />);

      expect(screen.getByText('No state logs available')).toBeInTheDocument();
    });
  });

  describe('API Integration', () => {
    it('should fetch watched workspaces on mount', async () => {
      const mockWorkspaces = { workspaces: ['workspace_1', 'workspace_2'] };
      (fetch as any).mockResolvedValue({
        json: () => Promise.resolve(mockWorkspaces)
      });

      render(<ThoughtSeedDebugPanel />);

      await waitFor(() => {
        expect(fetch).toHaveBeenCalledWith('http://localhost:8001/api/thoughtseed/watched');
      });
    });

    it('should fetch state logs on mount', async () => {
      const mockLogs = { logs: [] };
      (fetch as any).mockResolvedValue({
        json: () => Promise.resolve(mockLogs)
      });

      render(<ThoughtSeedDebugPanel />);

      await waitFor(() => {
        expect(fetch).toHaveBeenCalledWith('http://localhost:8001/api/thoughtseed/logs');
      });
    });

    it('should refresh logs when refresh button clicked', async () => {
      (fetch as any).mockResolvedValue({
        json: () => Promise.resolve({ logs: [] })
      });

      render(<ThoughtSeedDebugPanel />);

      const refreshButton = screen.getByText('Refresh Now');
      fireEvent.click(refreshButton);

      await waitFor(() => {
        expect(fetch).toHaveBeenCalledTimes(3); // initial + refresh
      });
    });
  });

  describe('State Display', () => {
    it('should display thought competition states', async () => {
      const mockStateLog = {
        logs: [{
          phase: 'pre_update',
          logged_at: '2025-09-26T10:30:45.123Z',
          state: {
            workspace_id: 'test_workspace',
            timestamp: '2025-09-26T10:30:45.123Z',
            thought_count: 2,
            thoughts: {
              'thought_1': {
                id: 'thought_1',
                content: 'Use analytical approach',
                type: 'action',
                energy: 0.8,
                confidence: 0.7,
                parent_ids: []
              },
              'thought_2': {
                id: 'thought_2',
                content: 'Apply creative solution',
                type: 'action',
                energy: 0.6,
                confidence: 0.9,
                parent_ids: []
              }
            },
            dominant_thought_id: 'thought_1'
          }
        }]
      };

      (fetch as any).mockResolvedValue({
        json: () => Promise.resolve(mockStateLog)
      });

      render(<ThoughtSeedDebugPanel />);

      await waitFor(() => {
        expect(screen.getByText('Use analytical approach')).toBeInTheDocument();
        expect(screen.getByText('Apply creative solution')).toBeInTheDocument();
        expect(screen.getByText('Dominant')).toBeInTheDocument();
      });
    });

    it('should highlight dominant thought', async () => {
      const mockStateLog = {
        logs: [{
          phase: 'post_update',
          logged_at: '2025-09-26T10:30:45.123Z',
          state: {
            workspace_id: 'test_workspace',
            timestamp: '2025-09-26T10:30:45.123Z',
            thought_count: 1,
            thoughts: {
              'dominant_thought': {
                id: 'dominant_thought',
                content: 'Winning thought',
                type: 'action',
                energy: 0.95,
                confidence: 0.9,
                parent_ids: []
              }
            },
            dominant_thought_id: 'dominant_thought'
          }
        }]
      };

      (fetch as any).mockResolvedValue({
        json: () => Promise.resolve(mockStateLog)
      });

      render(<ThoughtSeedDebugPanel />);

      await waitFor(() => {
        const dominantBadge = screen.getByText('Dominant');
        expect(dominantBadge).toBeInTheDocument();
      });
    });
  });

  describe('Auto-refresh Functionality', () => {
    it('should auto-refresh when toggle is enabled', async () => {
      vi.useFakeTimers();

      (fetch as any).mockResolvedValue({
        json: () => Promise.resolve({ logs: [] })
      });

      render(<ThoughtSeedDebugPanel />);

      // Fast-forward 2 seconds to trigger auto-refresh
      vi.advanceTimersByTime(2000);

      await waitFor(() => {
        expect(fetch).toHaveBeenCalledTimes(3); // mount + auto-refresh
      });

      vi.useRealTimers();
    });
  });

  describe('Error Handling', () => {
    it('should handle API errors gracefully', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      (fetch as any).mockRejectedValue(new Error('API Error'));

      render(<ThoughtSeedDebugPanel />);

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith('Failed to fetch watched workspaces:', expect.any(Error));
      });

      consoleSpy.mockRestore();
    });
  });
});

// Test data factories for reuse
export const createMockThought = (overrides = {}) => ({
  id: 'test_thought_id',
  content: 'Test thought content',
  type: 'action',
  energy: 0.5,
  confidence: 0.5,
  parent_ids: [],
  ...overrides
});

export const createMockWorkspaceState = (overrides = {}) => ({
  workspace_id: 'test_workspace',
  timestamp: '2025-09-26T10:30:45.123Z',
  thought_count: 1,
  thoughts: {
    'test_thought': createMockThought()
  },
  dominant_thought_id: 'test_thought',
  ...overrides
});

export const createMockStateLogEntry = (overrides = {}) => ({
  phase: 'pre_update',
  logged_at: '2025-09-26T10:30:45.123Z',
  state: createMockWorkspaceState(),
  ...overrides
});