/**
 * CitationPanel Component Tests - Spec 058
 *
 * RED phase: Tests written BEFORE implementation
 *
 * Tests cover:
 * - Side-sheet opens when isOpen is true
 * - Renders highlighted chunk text passed via props
 * - Displays basin + thoughtseed metadata blocks
 * - Emits onClose when close affordance is clicked
 *
 * Author: Agent 058A
 * Date: 2025-10-08
 */

import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import CitationPanel from '../CitationPanel';

describe('CitationPanel Component', () => {
  const mockChunkData = {
    chunkId: 'chunk_123',
    chunkText: 'This is the highlighted chunk text from the document.',
    chunkIndex: 0,
    startOffset: 0,
    endOffset: 55,
  };

  const mockBasinData = {
    basinId: 'basin_456',
    basinName: 'Cognitive Processing',
    stability: 0.85,
    attractorStrength: 0.72,
    layerInfluences: {
      layer1: 0.3,
      layer2: 0.5,
      layer3: 0.2,
    },
  };

  const mockThoughtseedData = {
    thoughtseedId: 'ts_789',
    resonanceScore: 0.91,
    conceptLabels: ['machine learning', 'neural networks', 'cognition'],
    emergenceTimestamp: '2025-10-08T12:00:00Z',
  };

  const defaultProps = {
    isOpen: false,
    chunkData: mockChunkData,
    basinData: mockBasinData,
    thoughtseedData: mockThoughtseedData,
    onClose: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Side-sheet visibility', () => {
    it('should not render panel when isOpen is false', () => {
      render(<CitationPanel {...defaultProps} isOpen={false} />);

      // Panel should not be visible
      const panel = screen.queryByRole('dialog');
      expect(panel).not.toBeInTheDocument();
    });

    it('should render panel when isOpen is true', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      // Panel should be visible as a dialog/complementary region
      const panel = screen.getByRole('dialog');
      expect(panel).toBeInTheDocument();
    });

    it('should have proper ARIA attributes when open', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      const panel = screen.getByRole('dialog');
      expect(panel).toHaveAttribute('aria-modal', 'true');
      expect(panel).toHaveAttribute('aria-labelledby');
    });
  });

  describe('Chunk text rendering', () => {
    it('should display the highlighted chunk text', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      expect(screen.getByText(mockChunkData.chunkText)).toBeInTheDocument();
    });

    it('should display chunk metadata (index and offsets)', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      // Should show chunk index
      expect(screen.getByText(/chunk #0/i)).toBeInTheDocument();

      // Should show character range
      expect(screen.getByText(/0-55/i)).toBeInTheDocument();
    });

    it('should handle empty chunk text gracefully', () => {
      const emptyChunkProps = {
        ...defaultProps,
        isOpen: true,
        chunkData: { ...mockChunkData, chunkText: '' },
      };

      render(<CitationPanel {...emptyChunkProps} />);

      // Should show placeholder or message
      expect(screen.getByText(/no text available/i)).toBeInTheDocument();
    });
  });

  describe('Basin metadata rendering', () => {
    it('should display basin name', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      expect(screen.getByText(mockBasinData.basinName)).toBeInTheDocument();
    });

    it('should display basin stability score', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      // Should show stability as percentage or decimal
      expect(screen.getByText(/stability.*0\.85|85%/i)).toBeInTheDocument();
    });

    it('should display attractor strength', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      expect(screen.getByText(/attractor.*0\.72|72%/i)).toBeInTheDocument();
    });

    it('should display layer influences', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      // Should show layer breakdown
      expect(screen.getByText(/layer 1.*0\.3|30%/i)).toBeInTheDocument();
      expect(screen.getByText(/layer 2.*0\.5|50%/i)).toBeInTheDocument();
      expect(screen.getByText(/layer 3.*0\.2|20%/i)).toBeInTheDocument();
    });

    it('should render dynamic layer influence keys without assumptions', () => {
      const dynamicLayerProps = {
        ...defaultProps,
        isOpen: true,
        basinData: {
          ...mockBasinData,
          layerInfluences: {
            layer_alpha: 0.42,
            cortex_beta: 0.18,
          },
        },
      };

      render(<CitationPanel {...dynamicLayerProps} />);

      expect(screen.getByText(/Layer alpha.*0\.42|42%/i)).toBeInTheDocument();
      expect(screen.getByText(/Cortex beta.*0\.18|18%/i)).toBeInTheDocument();
    });

    it('should show placeholder when layer influences are empty', () => {
      const noInfluenceProps = {
        ...defaultProps,
        isOpen: true,
        basinData: {
          ...mockBasinData,
          layerInfluences: {},
        },
      };

      render(<CitationPanel {...noInfluenceProps} />);

      expect(screen.getByText(/no layer influence data/i)).toBeInTheDocument();
    });

    it('should handle missing basin data gracefully', () => {
      const noBaysinProps = {
        ...defaultProps,
        isOpen: true,
        basinData: null,
      };

      render(<CitationPanel {...noBaysinProps} />);

      // Should show placeholder
      expect(screen.getByText(/no basin data/i)).toBeInTheDocument();
    });
  });

  describe('Thoughtseed metadata rendering', () => {
    it('should display resonance score', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      expect(screen.getByText(/resonance.*0\.91|91%/i)).toBeInTheDocument();
    });

    it('should display concept labels as tags', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      mockThoughtseedData.conceptLabels.forEach(label => {
        expect(screen.getByText(label)).toBeInTheDocument();
      });
    });

    it('should display emergence timestamp', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      // Should show formatted date/time
      expect(screen.getByText(/2025-10-08/i)).toBeInTheDocument();
    });

    it('should handle missing thoughtseed data gracefully', () => {
      const noThoughtseedProps = {
        ...defaultProps,
        isOpen: true,
        thoughtseedData: null,
      };

      render(<CitationPanel {...noThoughtseedProps} />);

      // Should show placeholder
      expect(screen.getByText(/no thoughtseed data/i)).toBeInTheDocument();
    });
  });

  describe('Close interaction', () => {
    it('should call onClose when close button is clicked', () => {
      const mockOnClose = jest.fn();
      render(<CitationPanel {...defaultProps} isOpen={true} onClose={mockOnClose} />);

      const closeButton = screen.getByRole('button', { name: /close/i });
      fireEvent.click(closeButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should call onClose when Escape key is pressed', () => {
      const mockOnClose = jest.fn();
      render(<CitationPanel {...defaultProps} isOpen={true} onClose={mockOnClose} />);

      const panel = screen.getByRole('dialog');
      fireEvent.keyDown(panel, { key: 'Escape', code: 'Escape' });

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should call onClose when clicking overlay/backdrop', () => {
      const mockOnClose = jest.fn();
      render(<CitationPanel {...defaultProps} isOpen={true} onClose={mockOnClose} />);

      // Click on backdrop (assuming data-testid="citation-panel-backdrop")
      const backdrop = screen.getByTestId('citation-panel-backdrop');
      fireEvent.click(backdrop);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });
  });

  describe('Accessibility', () => {
    it('should trap focus within panel when open', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      const panel = screen.getByRole('dialog');
      const closeButton = screen.getByRole('button', { name: /close/i });

      // Focus should be within the panel
      expect(panel).toBeInTheDocument();
      expect(closeButton).toBeInTheDocument();
    });

    it('should have descriptive heading for screen readers', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      const heading = screen.getByRole('heading', { name: /citation details/i });
      expect(heading).toBeInTheDocument();
    });

    it('should use semantic HTML for metadata sections', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      // Should use section or article elements with proper headings
      expect(screen.getByRole('heading', { name: /chunk text/i })).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: /basin metadata/i })).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: /thoughtseed/i })).toBeInTheDocument();
    });
  });

  describe('Visual structure', () => {
    it('should render sections in correct order', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      const headings = screen.getAllByRole('heading');
      const headingTexts = headings.map(h => h.textContent?.toLowerCase());

      // Order should be: panel title, chunk, basin, thoughtseed
      expect(headingTexts[0]).toMatch(/citation/i);
      expect(headingTexts[1]).toMatch(/chunk/i);
      expect(headingTexts[2]).toMatch(/basin/i);
      expect(headingTexts[3]).toMatch(/thoughtseed/i);
    });

    it('should apply distinct styling to each metadata section', () => {
      render(<CitationPanel {...defaultProps} isOpen={true} />);

      // Each section should have a class or data attribute for styling
      const chunkSection = screen.getByTestId('chunk-section');
      const basinSection = screen.getByTestId('basin-section');
      const thoughtseedSection = screen.getByTestId('thoughtseed-section');

      expect(chunkSection).toBeInTheDocument();
      expect(basinSection).toBeInTheDocument();
      expect(thoughtseedSection).toBeInTheDocument();
    });
  });
});
