import React, { useState, useEffect } from 'react';

interface QueuedDocument {
  id: string;
  filename: string;
  position: number;
  status: 'queued' | 'processing' | 'complete' | 'error';
  queued_at: string;
}

interface ProcessingEvent {
  type: string;
  document_id?: string;
  node?: string;
  timestamp: string;
  queue?: QueuedDocument[];
  active?: Record<string, ActiveDocument>;
  [key: string]: unknown;
}

interface ActiveDocument {
  filename: string;
  started_at: string;
  current_node: string | null;
  progress: number;
}

export function DebugPipelinePanel() {
  const [queue, setQueue] = useState<QueuedDocument[]>([]);
  const [active, setActive] = useState<Record<string, ActiveDocument>>({});
  const [events, setEvents] = useState<ProcessingEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect to debug stream
    const eventSource = new EventSource('/api/debug/stream');

    eventSource.onopen = () => {
      setIsConnected(true);
      console.log('🔌 Debug stream connected');
    };

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data) as ProcessingEvent;

      // Update queue status
      if (data.type === 'queue_status') {
        setQueue(data.queue || []);
        setActive(data.active || {});
      }

      // Add to event log
      setEvents(prev => [...prev.slice(-100), data]);  // Keep last 100 events
    };

    eventSource.onerror = () => {
      setIsConnected(false);
      console.error('❌ Debug stream disconnected');
    };

    return () => {
      eventSource.close();
    };
  }, []);

  const getNodeColor = (node: string | null): string => {
    const colors: Record<string, string> = {
      'extract_and_process': '#3b82f6',
      'generate_research_plan': '#8b5cf6',
      'consciousness_processing': '#ec4899',
      'analyze_results': '#10b981',
      'refine_processing': '#f59e0b',
      'finalize_output': '#06b6d4'
    };
    return colors[node || ''] || '#6b7280';
  };

  const getEventIcon = (type: string): string => {
    const icons: Record<string, string> = {
      'processing_start': '▶️',
      'node_start': '🔵',
      'node_complete': '✅',
      'concept_discovered': '💡',
      'basin_activated': '🌊',
      'thoughtseed_generated': '🌱',
      'quality_metric': '📊',
      'insight_discovered': '💭',
      'knowledge_graph_created': '🕸️',
      'processing_complete': '🎉',
      'processing_error': '❌'
    };
    return icons[type] || '•';
  };

  return (
    <div className="debug-pipeline-panel" style={{
      display: 'grid',
      gridTemplateColumns: '300px 1fr 400px',
      gap: '20px',
      padding: '20px',
      height: '100vh',
      backgroundColor: '#0f172a',
      color: '#f1f5f9',
      fontFamily: 'monospace'
    }}>
      {/* QUEUE COLUMN */}
      <div style={{
        backgroundColor: '#1e293b',
        borderRadius: '8px',
        padding: '15px',
        overflowY: 'auto'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '15px'
        }}>
          <h3 style={{ margin: 0 }}>📋 Queue ({queue.length})</h3>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: isConnected ? '#10b981' : '#ef4444'
          }} />
        </div>

        {queue.length === 0 ? (
          <div style={{ color: '#64748b', fontSize: '14px', textAlign: 'center', marginTop: '40px' }}>
            No documents queued
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {queue.map((doc) => (
              <div
                key={doc.id}
                style={{
                  backgroundColor: doc.status === 'processing' ? '#312e81' : '#1e293b',
                  border: doc.status === 'processing' ? '2px solid #6366f1' : '1px solid #334155',
                  borderRadius: '6px',
                  padding: '10px',
                  fontSize: '13px'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '5px' }}>
                  <span style={{ fontSize: '16px' }}>
                    {doc.status === 'processing' ? '⚙️' : '📄'}
                  </span>
                  <span style={{ fontWeight: 'bold', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {doc.filename}
                  </span>
                  <span style={{
                    backgroundColor: '#334155',
                    borderRadius: '4px',
                    padding: '2px 6px',
                    fontSize: '11px'
                  }}>
                    #{doc.position + 1}
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>
                  {doc.status === 'processing' ? (
                    <span style={{ color: '#10b981' }}>● Processing</span>
                  ) : (
                    <span>Queued {new Date(doc.queued_at).toLocaleTimeString()}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* PIPELINE COLUMN */}
      <div style={{
        backgroundColor: '#1e293b',
        borderRadius: '8px',
        padding: '15px',
        overflowY: 'auto'
      }}>
        <h3 style={{ margin: '0 0 15px 0' }}>🔄 Processing Pipeline</h3>

        {Object.keys(active).length === 0 ? (
          <div style={{ color: '#64748b', fontSize: '14px', textAlign: 'center', marginTop: '100px' }}>
            No active processing
          </div>
        ) : (
          Object.entries(active).map(([docId, doc]) => (
            <div key={docId} style={{ marginBottom: '30px' }}>
              {/* Document header */}
              <div style={{
                backgroundColor: '#312e81',
                borderRadius: '6px',
                padding: '12px',
                marginBottom: '15px'
              }}>
                <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
                  📄 {doc.filename}
                </div>
                <div style={{ fontSize: '12px', color: '#94a3b8' }}>
                  Started {new Date(doc.started_at).toLocaleTimeString()}
                </div>
              </div>

              {/* Progress bar */}
              <div style={{
                backgroundColor: '#334155',
                borderRadius: '8px',
                height: '8px',
                marginBottom: '20px',
                overflow: 'hidden'
              }}>
                <div style={{
                  backgroundColor: '#10b981',
                  height: '100%',
                  width: `${doc.progress * 100}%`,
                  transition: 'width 0.3s ease'
                }} />
              </div>

              {/* Pipeline nodes */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {[
                  { id: 'extract_and_process', label: 'Extract & Process', icon: '📝' },
                  { id: 'generate_research_plan', label: 'Research Plan', icon: '🔬' },
                  { id: 'consciousness_processing', label: 'Consciousness', icon: '🧠' },
                  { id: 'analyze_results', label: 'Analyze', icon: '📊' },
                  { id: 'refine_processing', label: 'Refine (optional)', icon: '🔄' },
                  { id: 'finalize_output', label: 'Finalize', icon: '✨' }
                ].map(node => (
                  <div
                    key={node.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '12px',
                      backgroundColor: doc.current_node === node.id ? '#312e81' : '#1e293b',
                      border: doc.current_node === node.id ? '2px solid #6366f1' : '1px solid #334155',
                      borderRadius: '6px',
                      opacity: doc.current_node === node.id ? 1 : 0.5,
                      transition: 'all 0.3s ease'
                    }}
                  >
                    <span style={{ fontSize: '20px' }}>{node.icon}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 'bold', fontSize: '14px' }}>{node.label}</div>
                      {doc.current_node === node.id && (
                        <div style={{ fontSize: '11px', color: '#10b981', marginTop: '2px' }}>
                          ● Processing...
                        </div>
                      )}
                    </div>
                    {doc.current_node === node.id && (
                      <div className="pulse" style={{
                        width: '10px',
                        height: '10px',
                        borderRadius: '50%',
                        backgroundColor: '#10b981',
                        animation: 'pulse 1.5s infinite'
                      }} />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      {/* EVENT LOG COLUMN */}
      <div style={{
        backgroundColor: '#1e293b',
        borderRadius: '8px',
        padding: '15px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <h3 style={{ margin: '0 0 15px 0' }}>📡 Event Stream</h3>

        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column-reverse',
          gap: '8px',
          fontSize: '12px'
        }}>
          {events.slice(-50).reverse().map((event, idx) => (
            <div
              key={idx}
              style={{
                backgroundColor: '#0f172a',
                borderRadius: '4px',
                padding: '8px',
                borderLeft: `3px solid ${getNodeColor(event.node || null)}`
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span>{getEventIcon(event.type)}</span>
                <span style={{ fontWeight: 'bold', color: '#f1f5f9' }}>
                  {event.type.replace(/_/g, ' ')}
                </span>
                <span style={{ marginLeft: 'auto', color: '#64748b', fontSize: '11px' }}>
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>
              </div>

              {event.type === 'concept_discovered' && (
                <div style={{ color: '#94a3b8', marginLeft: '28px' }}>
                  💡 {event.concept} ({event.index + 1}/{event.total})
                </div>
              )}

              {event.type === 'thoughtseed_generated' && (
                <div style={{ color: '#94a3b8', marginLeft: '28px' }}>
                  🌱 Concepts: {event.concepts?.join(', ') || 'N/A'}<br />
                  Resonance: {(event.resonance * 100).toFixed(0)}% | Basin: {event.basin}
                </div>
              )}

              {event.type === 'quality_metric' && (
                <div style={{ color: '#94a3b8', marginLeft: '28px' }}>
                  {event.metric}: {(event.score * 100).toFixed(1)}%
                </div>
              )}

              {event.type === 'insight_discovered' && (
                <div style={{ color: '#94a3b8', marginLeft: '28px' }}>
                  💭 {event.insight}
                </div>
              )}

              {event.type === 'knowledge_graph_created' && (
                <div style={{ color: '#94a3b8', marginLeft: '28px' }}>
                  🕸️ {event.nodes_created} nodes, {event.relationships_created} relationships
                </div>
              )}

              {event.node && (
                <div style={{ color: '#64748b', fontSize: '11px', marginLeft: '28px', marginTop: '2px' }}>
                  Node: {event.node}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .pulse {
          animation: pulse 1.5s infinite;
        }

        ::-webkit-scrollbar {
          width: 8px;
        }

        ::-webkit-scrollbar-track {
          background: #0f172a;
        }

        ::-webkit-scrollbar-thumb {
          background: #334155;
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: #475569;
        }
      `}</style>
    </div>
  );
}
