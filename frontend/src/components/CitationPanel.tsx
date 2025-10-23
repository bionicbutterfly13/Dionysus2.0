/**
 * CitationPanel Component - Spec 058
 *
 * Side-sheet panel for displaying citation details including:
 * - Highlighted chunk text
 * - Basin metadata (stability, attractor strength, layer influences)
 * - Thoughtseed metadata (resonance, concept labels, emergence timestamp)
 *
 * Author: Implementation Agent (GREEN phase)
 * Date: 2025-10-08
 */

import React, { useEffect, useRef } from 'react';

// TypeScript interfaces based on test requirements
interface ChunkData {
  chunkId: string;
  chunkText: string;
  chunkIndex: number;
  startOffset: number;
  endOffset: number;
}

interface BasinData {
  basinId: string;
  basinName: string;
  stability: number;
  attractorStrength: number;
  layerInfluences: Record<string, number>;
}

interface ThoughtseedData {
  thoughtseedId: string;
  resonanceScore: number;
  conceptLabels: string[];
  emergenceTimestamp: string;
}

interface CitationPanelProps {
  isOpen: boolean;
  chunkData: ChunkData | null;
  basinData: BasinData | null;
  thoughtseedData: ThoughtseedData | null;
  onClose: () => void;
}

const CitationPanel: React.FC<CitationPanelProps> = ({
  isOpen,
  chunkData,
  basinData,
  thoughtseedData,
  onClose,
}) => {
  const panelRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  // Handle Escape key press
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isOpen) {
        event.stopPropagation();
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      // Focus close button when panel opens
      closeButtonRef.current?.focus();
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  // Don't render if not open
  if (!isOpen) {
    return null;
  }

  // Format percentage helper
  const formatPercent = (value: number): string => {
    return `${(value * 100).toFixed(0)}%`;
  };

  // Format decimal helper
  const formatDecimal = (value: number): string => {
    return value.toFixed(2);
  };

  // Format timestamp helper
  const formatTimestamp = (timestamp: string): string => {
    try {
      const date = new Date(timestamp);
      return date.toISOString().split('T')[0]; // Returns YYYY-MM-DD
    } catch {
      return timestamp;
    }
  };

  const formatLayerLabel = (label: string): string => {
    const withSpaces = label
      .replace(/[_-]+/g, ' ')
      .replace(/(\d+)/g, ' $1')
      .trim();
    return withSpaces.charAt(0).toUpperCase() + withSpaces.slice(1);
  };

  return (
    <>
      {/* Backdrop */}
      <div
        data-testid="citation-panel-backdrop"
        className="fixed inset-0 bg-black bg-opacity-50 z-40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="citation-panel-title"
        className="fixed right-0 top-0 bottom-0 w-96 bg-white shadow-xl z-50 overflow-y-auto"
      >
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 p-4 flex justify-between items-center">
          <h2 id="citation-panel-title" className="text-xl font-semibold">
            Citation Details
          </h2>
          <button
            ref={closeButtonRef}
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-6">
          {/* Chunk Text Section */}
          <section data-testid="chunk-section">
            <h3 className="text-lg font-semibold mb-2">Chunk Text</h3>
            {chunkData && chunkData.chunkText ? (
              <div className="space-y-2">
                <p className="text-gray-700 bg-gray-50 p-3 rounded">
                  {chunkData.chunkText}
                </p>
                <div className="text-sm text-gray-500">
                  <p>Chunk #{chunkData.chunkIndex}</p>
                  <p>Characters: {chunkData.startOffset}-{chunkData.endOffset}</p>
                </div>
              </div>
            ) : (
              <p className="text-gray-400 italic">No text available</p>
            )}
          </section>

          {/* Basin Metadata Section */}
          <section data-testid="basin-section">
            <h3 className="text-lg font-semibold mb-2">Basin Metadata</h3>
            {basinData ? (
              <div className="space-y-3 bg-blue-50 p-3 rounded">
                <p className="font-medium">{basinData.basinName}</p>

                <div className="space-y-1 text-sm">
                  <p>
                    <span className="font-medium">Stability:</span>{' '}
                    {formatDecimal(basinData.stability)} ({formatPercent(basinData.stability)})
                  </p>
                  <p>
                    <span className="font-medium">Attractor Strength:</span>{' '}
                    {formatDecimal(basinData.attractorStrength)} ({formatPercent(basinData.attractorStrength)})
                  </p>
                </div>

                <div className="mt-3">
                  <p className="font-medium text-sm mb-1">Layer Influences:</p>
                  {Object.entries(basinData.layerInfluences ?? {}).length > 0 ? (
                    <div className="space-y-1 text-sm pl-2">
                      {Object.entries(basinData.layerInfluences).map(([layer, influence]) => (
                        <p key={layer}>
                          {formatLayerLabel(layer)}: {formatDecimal(influence)} ({formatPercent(influence)})
                        </p>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500 italic">No layer influence data</p>
                  )}
                </div>
              </div>
            ) : (
              <p className="text-gray-400 italic">No basin data available</p>
            )}
          </section>

          {/* Thoughtseed Metadata Section */}
          <section data-testid="thoughtseed-section">
            <h3 className="text-lg font-semibold mb-2">Thoughtseed</h3>
            {thoughtseedData ? (
              <div className="space-y-3 bg-purple-50 p-3 rounded">
                <div className="space-y-1 text-sm">
                  <p>
                    <span className="font-medium">Resonance Score:</span>{' '}
                    {formatDecimal(thoughtseedData.resonanceScore)} ({formatPercent(thoughtseedData.resonanceScore)})
                  </p>
                  <p>
                    <span className="font-medium">Emergence:</span>{' '}
                    {formatTimestamp(thoughtseedData.emergenceTimestamp)}
                  </p>
                </div>

                <div>
                  <p className="font-medium text-sm mb-2">Concept Labels:</p>
                  <div className="flex flex-wrap gap-2">
                    {thoughtseedData.conceptLabels.map((label, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 bg-purple-200 text-purple-800 rounded text-xs"
                      >
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-gray-400 italic">No thoughtseed data available</p>
            )}
          </section>
        </div>
      </div>
    </>
  );
};

export default CitationPanel;
