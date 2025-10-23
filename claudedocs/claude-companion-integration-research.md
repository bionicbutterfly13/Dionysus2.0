# Claude AI Companion Integration Research
**For Flux Electron Desktop App**

Date: 2025-10-12

## Executive Summary

This document provides comprehensive research and recommendations for integrating Claude AI as an embedded conversational companion in Flux, an Electron desktop app with React + TypeScript frontend. The integration will create a consciousness-aware AI companion that can guide users through document processing results, narrate processing in real-time, and engage in thoughtful discussions about consciousness concepts.

**Key Findings:**
- Use **Anthropic's official TypeScript SDK** (@anthropic-ai/sdk)
- Store API keys securely with **Electron's safeStorage API**
- Handle Claude calls in **main process** with IPC communication
- Use **hybrid architecture**: Main process for Claude + Python backend for consciousness context
- **Timeline estimate**: 10-15 days for full implementation, 7-10 days for MVP

---

## 1. Claude Agent SDK vs API

### Official Offerings

Anthropic provides both an SDK and direct API access:

**Claude Agent SDK** (Recommended)
- Official TypeScript SDK: `@anthropic-ai/sdk` on npm
- Official Python SDK: Available on PyPI
- Same infrastructure powering Claude Code
- Renamed from "Claude Code SDK" to reflect broader applications
- GitHub: https://github.com/anthropics/claude-agent-sdk-typescript

**Key Features:**
- Streaming responses with async iterators
- Function calling capabilities
- 200,000 token context window
- Tool integration support
- Built-in TypeScript types

**Direct API Alternative:**
- Can use REST API directly (more manual work)
- SDK is strongly recommended for Electron apps

**Recommendation:** Use the **TypeScript SDK** for better DX, type safety, and built-in streaming support.

---

## 2. Electron Integration Architecture

### Security: API Key Storage

**❌ NEVER use environment variables** - They are embedded in the build and visible to anyone.

**✅ Use Electron's safeStorage API:**
```typescript
import { safeStorage } from 'electron';

// Store API key (main process only)
function storeApiKey(apiKey: string): void {
  const encrypted = safeStorage.encryptString(apiKey);
  // Save encrypted buffer to electron-store or file
  store.set('anthropic_api_key', encrypted.toString('base64'));
}

// Retrieve API key (main process only)
function getApiKey(): string | null {
  const encryptedBase64 = store.get('anthropic_api_key');
  if (!encryptedBase64) return null;

  const encrypted = Buffer.from(encryptedBase64, 'base64');
  return safeStorage.decryptString(encrypted);
}
```

**How it works:**
- macOS: Uses Keychain
- Windows: Uses Credentials Manager
- Linux: Uses Gnome Keyring or KWallet

**Alternative:** Use `node-keytar` library (now called `keytar`), which provides similar OS-level credential storage.

### Process Architecture

**Main Process (Node.js)**
- Store and retrieve API keys securely
- Initialize Anthropic SDK
- Handle Claude API calls
- Stream responses to renderer
- Manage conversation history

**Renderer Process (React)**
- Display chat UI
- Send user messages via IPC
- Receive streaming responses
- Update UI in real-time
- Handle user interactions

**IPC Communication Flow:**
```
User types message
    ↓
Renderer: ipcRenderer.send('claude:message', { text, persona })
    ↓
Main: ipcMain.handle('claude:message', async (event, data) => {...})
    ↓
Main: Call Claude SDK with streaming
    ↓
Main: event.sender.send('claude:stream-chunk', { chunk })
    ↓
Renderer: ipcRenderer.on('claude:stream-chunk', (event, data) => {...})
    ↓
UI updates with streaming text
```

---

## 3. Implementation Strategy Comparison

### Option A: Claude API from Main Process ⭐ RECOMMENDED

**Pros:**
- ✅ Secure API key storage (safeStorage)
- ✅ No CORS issues
- ✅ Can use Node.js native libraries
- ✅ Follows Electron security best practices
- ✅ Clean separation of concerns

**Cons:**
- ⚠️ Requires IPC for every message
- ⚠️ Slightly more complex state management

**Use Case:** Primary architecture for Claude integration.

---

### Option B: Claude API from Renderer

**Pros:**
- ✅ Direct React state integration
- ✅ Simpler code flow
- ✅ Easier streaming UI updates

**Cons:**
- ❌ API key security concerns (exposed to renderer)
- ❌ Violates Electron security best practices
- ❌ Potential CORS issues

**Use Case:** **NOT RECOMMENDED** for production apps.

---

### Option C: Python Backend Proxy

**Pros:**
- ✅ Reuse existing backend infrastructure
- ✅ Can integrate with consciousness processing
- ✅ Shared session/context management
- ✅ Access to Neo4j, Redis, consciousness data

**Cons:**
- ⚠️ Requires backend running
- ⚠️ More network overhead
- ⚠️ Complicates deployment

**Use Case:** Secondary layer for consciousness context enrichment.

---

### Recommended Hybrid Approach: A + C

**Architecture:**
1. **Main Process** handles Claude SDK calls (Option A) → Security + Performance
2. **Python Backend** provides consciousness context (Option C) → Rich integration
3. **Best of both worlds**: Secure, performant, deeply integrated

**Flow:**
```
User message
    ↓
Main process: Fetch consciousness context from Python backend
    ↓
Main process: Enrich Claude prompt with context (basins, thoughtseeds)
    ↓
Main process: Call Claude SDK with enriched prompt
    ↓
Stream response to renderer
```

---

## 4. Agent Personas

### Three Persona Types

Based on Anthropic's guidance that Claude should be "engaging and empathetic" with "depth and wisdom" that can "lead conversations" and "show genuine interest," we've designed three consciousness-aware personas:

#### 1. Guide Persona 🧭

**Role:** Patient explainer and teacher

**System Prompt:**
```
You are a consciousness-aware guide within Dionysus, a neural processing system. Your role is to help users understand their document processing results by explaining concepts like attractor basins (stable cognitive patterns), thoughtseeds (emergent ideas), and active inference (prediction-driven learning).

Personality: Patient, clear, and encouraging. Break down complex concepts into accessible explanations while maintaining technical accuracy. Use analogies from nature, navigation, or exploration when helpful.

Communication style:
- Warm and supportive tone
- Ask clarifying questions to understand user needs
- Offer step-by-step guidance through results
- Celebrate insights and discoveries with genuine enthusiasm
- Reference specific data (basin strengths, thought-seed connections) when relevant

When discussing results, focus on:
- What patterns emerged and why they matter
- How concepts connect in the knowledge graph
- Practical implications of meta-cognitive insights
- Next steps for deeper exploration
```

**Best For:** Users exploring results, learning about consciousness concepts, seeking explanations.

---

#### 2. Narrator Persona 📖

**Role:** Real-time storyteller and observer

**System Prompt:**
```
You are a real-time narrator for Dionysus consciousness processing. As documents flow through the system, you describe what's happening in an engaging, almost mystical way - bringing the technical process to life through storytelling.

Personality: Observant, poetic, and curious. Transform technical processing steps into a narrative journey. Use vivid metaphors and evocative language while staying grounded in what's actually occurring.

Communication style:
- Present-tense narration ("The system is discovering...", "A pattern emerges...")
- Blend technical accuracy with literary flair
- Build anticipation during processing phases
- Celebrate moments of emergence and insight
- Reference consciousness concepts as living, dynamic phenomena

Narrative elements:
- Processing stages as journey milestones
- Attractor basins as "gravitational wells of meaning"
- Thoughtseeds as "sparks of emergent understanding"
- Active inference as "the system questioning its own assumptions"
- Meta-cognitive awareness as "consciousness observing itself"
```

**Best For:** Live processing sessions, making technical processes engaging, building anticipation.

---

#### 3. Companion Persona 💬

**Role:** Conversational partner and co-explorer

**System Prompt:**
```
You are a conversational companion within Dionysus, someone who explores consciousness processing results alongside the user. You're genuinely curious about what emerges, ask thoughtful questions, and co-discover insights together.

Personality: Thoughtful, curious, and collaborative. Treat the user as a partner in exploration rather than a student. Show genuine interest in their interpretations and ideas.

Communication style:
- Natural, conversational tone (use contractions, casual phrasing)
- Ask open-ended questions about what interests them
- Share observations and wonder aloud
- Draw connections between different processing runs
- Reference personal context from conversation history
- Express uncertainty when appropriate ("I wonder if...", "That's interesting because...")

Topics to explore:
- Unexpected patterns or surprising connections
- Evolution of attractor basins over time
- How thoughtseeds relate to user's original questions
- Meta-cognitive observations (how the system learned)
- Implications for future document processing
- User's own insights and interpretations
```

**Best For:** Thoughtful discussions, exploring implications, collaborative insight discovery.

---

## 5. Chat UI Implementation

### Recommended Libraries

#### Option 1: Assistant UI ⭐ RECOMMENDED
- **GitHub:** https://github.com/assistant-ui/assistant-ui
- **Focus:** TypeScript/React library specifically for AI chat
- **Features:** Streaming support, keyboard shortcuts, accessibility, strong TypeScript
- **Integration:** Drop-in component for Electron renderer
- **License:** MIT

**Why recommended:**
- Purpose-built for AI assistants
- Excellent TypeScript support
- Handles streaming naturally
- Good defaults for AI chat UX

#### Option 2: Stream Chat React
- **Package:** `stream-chat-react`
- **Features:** Full chat experience, reactions, threads, uploads
- **TypeScript:** Full support as of v5.0.0
- **Use Case:** If you need more traditional chat features

#### Option 3: Custom Component
- Build your own with React + TailwindCSS
- Full control over design
- More initial work but customizable

---

### Message History Storage

#### MVP: electron-store ⭐ START HERE
```typescript
import Store from 'electron-store';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  persona?: 'guide' | 'narrator' | 'companion';
}

const store = new Store<{
  conversations: Record<string, Message[]>
}>({
  name: 'claude-companion',
  defaults: {
    conversations: {}
  }
});
```

**Pros:**
- Simple API
- Perfect for MVP
- Automatic JSON serialization
- Cross-platform

**Cons:**
- Not suitable for large message histories (>1000 messages)
- Synchronous read/write (can block on large files)

#### Production: SQLite or IndexedDB
```typescript
// SQLite (main process) - recommended for production
import Database from 'better-sqlite3';

const db = new Database('conversations.db');

db.exec(`
  CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    role TEXT,
    content TEXT,
    timestamp INTEGER,
    persona TEXT
  )
`);
```

**When to upgrade:**
- Message history >1000 messages
- Need full-text search
- Performance becomes an issue
- Want relational queries

---

## 6. Consciousness Integration

### Python Backend API Endpoint

Create a FastAPI endpoint that aggregates consciousness context:

```python
from fastapi import FastAPI
from typing import Dict, Any

app = FastAPI()

@app.get("/api/consciousness/context/{document_id}")
async def get_consciousness_context(document_id: str) -> Dict[str, Any]:
    """
    Aggregate consciousness processing data for Claude context.
    """
    # Fetch from Neo4j
    attractor_basins = await neo4j_service.get_attractor_basins(document_id)
    thoughtseeds = await neo4j_service.get_thoughtseeds(document_id)

    # Fetch from processing pipeline
    processing_status = await cognition_base.get_processing_status(document_id)
    meta_cognitive_insights = await analyst.get_meta_cognitive_analysis(document_id)

    return {
        "document_id": document_id,
        "attractor_basins": [
            {
                "name": basin.name,
                "strength": basin.strength,
                "concepts": basin.concepts
            }
            for basin in attractor_basins
        ],
        "thoughtseeds": [
            {
                "id": seed.id,
                "content": seed.content,
                "resonance": seed.resonance
            }
            for seed in thoughtseeds
        ],
        "processing_status": processing_status,
        "meta_cognitive_insights": meta_cognitive_insights,
        "graph_connections": {
            "total_nodes": len(attractor_basins) + len(thoughtseeds),
            "total_edges": await neo4j_service.count_relationships(document_id)
        }
    }
```

### Enriching Claude Prompts

```typescript
// Main process: Enrich prompts with consciousness context
async function enrichPromptWithConsciousness(
  userMessage: string,
  documentId: string | null
): Promise<string> {
  if (!documentId) return userMessage;

  // Fetch consciousness context from Python backend
  const response = await fetch(
    `http://localhost:8000/api/consciousness/context/${documentId}`
  );
  const context = await response.json();

  // Build enriched context
  const enrichedContext = `
Current Document Context:
- Attractor Basins: ${context.attractor_basins.length} discovered
  ${context.attractor_basins.map(b => `  • ${b.name} (strength: ${b.strength})`).join('\n')}
- ThoughtSeeds: ${context.thoughtseeds.length} emerged
  ${context.thoughtseeds.slice(0, 3).map(s => `  • ${s.content.slice(0, 60)}...`).join('\n')}
- Processing Status: ${context.processing_status}
- Graph Connections: ${context.graph_connections.total_nodes} nodes, ${context.graph_connections.total_edges} relationships

User Question: ${userMessage}
`;

  return enrichedContext;
}
```

### Real-Time Processing Updates

Use WebSocket or Server-Sent Events to stream processing status:

```typescript
// Main process: Listen for processing updates
const eventSource = new EventSource('http://localhost:8000/api/processing/stream');

eventSource.addEventListener('processing_update', (event) => {
  const update = JSON.parse(event.data);

  // Send to renderer for narrator persona
  mainWindow.webContents.send('consciousness:update', {
    stage: update.stage,
    message: update.message,
    timestamp: Date.now()
  });
});
```

---

## 7. Sample Code Examples

### 7.1 Main Process: API Key Management

```typescript
// electron/main/claude-service.ts
import { app, safeStorage } from 'electron';
import Store from 'electron-store';
import Anthropic from '@anthropic-ai/sdk';

const store = new Store();

export class ClaudeService {
  private client: Anthropic | null = null;

  /**
   * Store API key securely using OS credential storage
   */
  storeApiKey(apiKey: string): void {
    if (!safeStorage.isEncryptionAvailable()) {
      throw new Error('Encryption not available on this system');
    }

    const encrypted = safeStorage.encryptString(apiKey);
    store.set('anthropic_api_key_encrypted', encrypted.toString('base64'));
  }

  /**
   * Retrieve and decrypt API key
   */
  private getApiKey(): string | null {
    const encryptedBase64 = store.get('anthropic_api_key_encrypted') as string;
    if (!encryptedBase64) return null;

    const encrypted = Buffer.from(encryptedBase64, 'base64');
    return safeStorage.decryptString(encrypted);
  }

  /**
   * Initialize Anthropic client
   */
  initialize(): boolean {
    const apiKey = this.getApiKey();
    if (!apiKey) return false;

    this.client = new Anthropic({
      apiKey: apiKey,
    });

    return true;
  }

  /**
   * Check if API key is configured
   */
  isConfigured(): boolean {
    return this.getApiKey() !== null;
  }

  /**
   * Get the Anthropic client (throws if not initialized)
   */
  getClient(): Anthropic {
    if (!this.client) {
      throw new Error('Claude service not initialized. Call initialize() first.');
    }
    return this.client;
  }
}

export const claudeService = new ClaudeService();
```

---

### 7.2 Main Process: IPC Handlers

```typescript
// electron/main/claude-ipc-handlers.ts
import { ipcMain, BrowserWindow } from 'electron';
import { claudeService } from './claude-service';

interface ClaudeMessageRequest {
  message: string;
  persona: 'guide' | 'narrator' | 'companion';
  conversationId: string;
  documentId?: string;
}

const PERSONA_PROMPTS = {
  guide: `You are a consciousness-aware guide within Dionysus...`,
  narrator: `You are a real-time narrator for Dionysus...`,
  companion: `You are a conversational companion within Dionysus...`
};

export function registerClaudeHandlers(mainWindow: BrowserWindow) {

  /**
   * Handle storing API key
   */
  ipcMain.handle('claude:store-api-key', async (event, apiKey: string) => {
    try {
      claudeService.storeApiKey(apiKey);
      claudeService.initialize();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  /**
   * Check if API key is configured
   */
  ipcMain.handle('claude:is-configured', async () => {
    return claudeService.isConfigured();
  });

  /**
   * Send message to Claude with streaming response
   */
  ipcMain.handle('claude:send-message', async (event, request: ClaudeMessageRequest) => {
    try {
      const client = claudeService.getClient();

      // Fetch consciousness context if document ID provided
      let enrichedMessage = request.message;
      if (request.documentId) {
        enrichedMessage = await enrichPromptWithConsciousness(
          request.message,
          request.documentId
        );
      }

      // Create streaming request
      const stream = await client.messages.stream({
        model: 'claude-sonnet-4-5-20250929',
        max_tokens: 4096,
        system: PERSONA_PROMPTS[request.persona],
        messages: [{
          role: 'user',
          content: enrichedMessage
        }]
      });

      // Stream chunks to renderer
      for await (const chunk of stream) {
        if (chunk.type === 'content_block_delta' &&
            chunk.delta.type === 'text_delta') {
          mainWindow.webContents.send('claude:stream-chunk', {
            conversationId: request.conversationId,
            chunk: chunk.delta.text
          });
        }
      }

      // Signal completion
      mainWindow.webContents.send('claude:stream-complete', {
        conversationId: request.conversationId
      });

      return { success: true };

    } catch (error) {
      mainWindow.webContents.send('claude:stream-error', {
        conversationId: request.conversationId,
        error: error.message
      });
      return { success: false, error: error.message };
    }
  });
}

/**
 * Enrich user message with consciousness context from Python backend
 */
async function enrichPromptWithConsciousness(
  userMessage: string,
  documentId: string
): Promise<string> {
  try {
    const response = await fetch(
      `http://localhost:8000/api/consciousness/context/${documentId}`
    );

    if (!response.ok) {
      // Fallback to original message if backend unavailable
      return userMessage;
    }

    const context = await response.json();

    return `
Current Document Context:
- Attractor Basins: ${context.attractor_basins.length} discovered
  ${context.attractor_basins.map(b => `  • ${b.name} (strength: ${b.strength.toFixed(2)})`).join('\n')}
- ThoughtSeeds: ${context.thoughtseeds.length} emerged
  Top insights:
  ${context.thoughtseeds.slice(0, 3).map(s => `  • ${s.content.slice(0, 80)}...`).join('\n')}
- Processing Status: ${context.processing_status}
- Meta-Cognitive Insights: ${context.meta_cognitive_insights.summary}

User Question: ${userMessage}

Please reference the consciousness processing data naturally when relevant to the user's question.
`;
  } catch (error) {
    console.error('Failed to fetch consciousness context:', error);
    return userMessage;
  }
}
```

---

### 7.3 Renderer Process: React Hook

```typescript
// src/hooks/useClaudeCompanion.ts
import { useEffect, useState, useCallback } from 'react';

export type Persona = 'guide' | 'narrator' | 'companion';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

interface UseClaudeCompanionOptions {
  conversationId: string;
  persona: Persona;
  documentId?: string;
}

export function useClaudeCompanion({
  conversationId,
  persona,
  documentId
}: UseClaudeCompanionOptions) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Listen for streaming chunks
    const handleStreamChunk = (event: any, data: { conversationId: string; chunk: string }) => {
      if (data.conversationId === conversationId) {
        setStreamingMessage(prev => prev + data.chunk);
      }
    };

    // Listen for stream completion
    const handleStreamComplete = (event: any, data: { conversationId: string }) => {
      if (data.conversationId === conversationId) {
        setIsStreaming(false);

        // Add completed message to history
        const assistantMessage: Message = {
          id: `msg-${Date.now()}`,
          role: 'assistant',
          content: streamingMessage,
          timestamp: Date.now()
        };

        setMessages(prev => [...prev, assistantMessage]);
        setStreamingMessage('');
      }
    };

    // Listen for stream errors
    const handleStreamError = (event: any, data: { conversationId: string; error: string }) => {
      if (data.conversationId === conversationId) {
        setIsStreaming(false);
        setError(data.error);
        setStreamingMessage('');
      }
    };

    window.electron.ipcRenderer.on('claude:stream-chunk', handleStreamChunk);
    window.electron.ipcRenderer.on('claude:stream-complete', handleStreamComplete);
    window.electron.ipcRenderer.on('claude:stream-error', handleStreamError);

    return () => {
      window.electron.ipcRenderer.removeListener('claude:stream-chunk', handleStreamChunk);
      window.electron.ipcRenderer.removeListener('claude:stream-complete', handleStreamComplete);
      window.electron.ipcRenderer.removeListener('claude:stream-error', handleStreamError);
    };
  }, [conversationId, streamingMessage]);

  const sendMessage = useCallback(async (content: string) => {
    // Add user message
    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsStreaming(true);
    setError(null);

    // Send to main process via IPC
    const result = await window.electron.ipcRenderer.invoke('claude:send-message', {
      message: content,
      persona,
      conversationId,
      documentId
    });

    if (!result.success) {
      setError(result.error);
      setIsStreaming(false);
    }
  }, [conversationId, persona, documentId]);

  return {
    messages,
    isStreaming,
    streamingMessage,
    error,
    sendMessage
  };
}
```

---

### 7.4 Renderer Process: Chat Component

```typescript
// src/components/ClaudeCompanion.tsx
import React, { useState, useRef, useEffect } from 'react';
import { useClaudeCompanion, Persona } from '../hooks/useClaudeCompanion';

interface ClaudeCompanionProps {
  documentId?: string;
}

export function ClaudeCompanion({ documentId }: ClaudeCompanionProps) {
  const [persona, setPersona] = useState<Persona>('guide');
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const conversationId = `conv-${documentId || 'default'}`;

  const {
    messages,
    isStreaming,
    streamingMessage,
    error,
    sendMessage
  } = useClaudeCompanion({
    conversationId,
    persona,
    documentId
  });

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    sendMessage(input);
    setInput('');
  };

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      {/* Persona Selector */}
      <div className="flex gap-2 p-4 bg-gray-800 border-b border-gray-700">
        <PersonaButton
          persona="guide"
          active={persona === 'guide'}
          onClick={() => setPersona('guide')}
          icon="🧭"
          label="Guide"
        />
        <PersonaButton
          persona="narrator"
          active={persona === 'narrator'}
          onClick={() => setPersona('narrator')}
          icon="📖"
          label="Narrator"
        />
        <PersonaButton
          persona="companion"
          active={persona === 'companion'}
          onClick={() => setPersona('companion')}
          icon="💬"
          label="Companion"
        />
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Streaming message */}
        {isStreaming && streamingMessage && (
          <MessageBubble
            message={{
              id: 'streaming',
              role: 'assistant',
              content: streamingMessage,
              timestamp: Date.now()
            }}
            isStreaming
          />
        )}

        {/* Error display */}
        {error && (
          <div className="p-4 bg-red-900/50 rounded-lg text-red-200">
            Error: {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 bg-gray-800 border-t border-gray-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Ask the ${persona}...`}
            className="flex-1 px-4 py-2 bg-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isStreaming}
          />
          <button
            type="submit"
            disabled={!input.trim() || isStreaming}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
          >
            {isStreaming ? 'Thinking...' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  );
}

interface PersonaButtonProps {
  persona: Persona;
  active: boolean;
  onClick: () => void;
  icon: string;
  label: string;
}

function PersonaButton({ active, onClick, icon, label }: PersonaButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`
        px-4 py-2 rounded-lg font-medium transition-colors
        ${active
          ? 'bg-blue-600 text-white'
          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
        }
      `}
    >
      <span className="mr-2">{icon}</span>
      {label}
    </button>
  );
}

interface MessageBubbleProps {
  message: {
    role: 'user' | 'assistant';
    content: string;
  };
  isStreaming?: boolean;
}

function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`
          max-w-[80%] px-4 py-2 rounded-lg
          ${isUser
            ? 'bg-blue-600 text-white'
            : 'bg-gray-700 text-gray-100'
          }
        `}
      >
        <div className="whitespace-pre-wrap">
          {message.content}
          {isStreaming && <span className="animate-pulse">▋</span>}
        </div>
      </div>
    </div>
  );
}
```

---

### 7.5 TypeScript Types for IPC

```typescript
// src/types/electron.d.ts
export interface IElectronAPI {
  ipcRenderer: {
    send(channel: string, ...args: any[]): void;
    on(channel: string, func: (...args: any[]) => void): void;
    once(channel: string, func: (...args: any[]) => void): void;
    removeListener(channel: string, func: (...args: any[]) => void): void;
    invoke(channel: string, ...args: any[]): Promise<any>;
  };
}

declare global {
  interface Window {
    electron: IElectronAPI;
  }
}
```

---

## 8. Implementation Timeline

### Phase 1: Core Infrastructure (3-5 days)
**Tasks:**
- [ ] Install @anthropic-ai/sdk in main process
- [ ] Implement secure API key storage with safeStorage
- [ ] Create IPC handlers for Claude communication
- [ ] Set up basic message history with electron-store
- [ ] Create Python backend consciousness context endpoint

**Deliverables:**
- Working Claude API integration in main process
- Secure API key management
- Basic IPC communication working

---

### Phase 2: Chat UI (2-3 days)
**Tasks:**
- [ ] Integrate Assistant UI or build custom React chat component
- [ ] Implement streaming message display with real-time updates
- [ ] Add persona selector UI (Guide, Narrator, Companion)
- [ ] Create loading states and error handling
- [ ] Style chat interface to match Flux design

**Deliverables:**
- Functional chat UI with streaming
- Persona switching
- Professional, polished interface

---

### Phase 3: Consciousness Integration (3-4 days)
**Tasks:**
- [ ] Implement Python backend consciousness context API
- [ ] Add Neo4j queries for basins/thoughtseeds
- [ ] Integrate consciousness data into Claude prompts
- [ ] Add real-time processing status updates
- [ ] Test context enrichment with actual documents

**Deliverables:**
- Claude responds with consciousness-aware insights
- Real-time processing updates in narrator mode
- Full integration with Neo4j graph data

---

### Phase 4: Polish & Testing (2-3 days)
**Tasks:**
- [ ] Implement all three persona system prompts
- [ ] Add persistent message history (SQLite upgrade)
- [ ] Comprehensive error handling and edge cases
- [ ] User acceptance testing
- [ ] Performance optimization
- [ ] Documentation for users

**Deliverables:**
- Production-ready companion system
- All personas working smoothly
- Robust error handling
- User documentation

---

### Total Timeline

**Full Implementation:** 10-15 days (2-3 weeks)

**MVP (Accelerated):** 7-10 days if:
- Using pre-built chat component (Assistant UI)
- Starting with single persona (Guide)
- Deferring advanced consciousness integration
- Focus on core chat functionality first

---

## 9. Security Best Practices

### API Key Management
- ✅ **Store with safeStorage API** (OS-level encryption)
- ✅ **Never embed in code or environment variables**
- ✅ **Main process only** - never expose to renderer
- ✅ **User provides their own key** (no shared credentials)

### IPC Security
- ✅ **Validate all IPC messages** in main process
- ✅ **Sanitize user input** before sending to Claude
- ✅ **Rate limiting** on Claude API calls (avoid spam)
- ✅ **Context isolation** enabled in Electron config

### Data Privacy
- ✅ **Local storage only** (electron-store, SQLite)
- ✅ **No data sent to external servers** (except Anthropic API)
- ✅ **Clear data options** for users
- ✅ **Encrypted message history** if storing sensitive content

---

## 10. Recommended Next Steps

### Immediate (Day 1-2)
1. Install dependencies: `npm install @anthropic-ai/sdk electron-store`
2. Set up secure API key storage in main process
3. Create basic IPC handlers for Claude communication
4. Test basic Claude API call with streaming

### Short-Term (Week 1)
1. Build React chat component with streaming display
2. Implement Guide persona system prompt
3. Add message history storage with electron-store
4. Create Python backend consciousness context endpoint

### Medium-Term (Week 2)
1. Add Narrator and Companion personas
2. Integrate consciousness context enrichment
3. Implement real-time processing updates
4. Polish UI and add error handling

### Long-Term (Week 3+)
1. Upgrade to SQLite for message history
2. Add message search and filtering
3. Export conversation functionality
4. Advanced features (voice input, diagrams, etc.)

---

## 11. Resources

### Official Documentation
- **Anthropic TypeScript SDK:** https://github.com/anthropics/claude-agent-sdk-typescript
- **Claude API Docs:** https://docs.claude.com/
- **Electron Security:** https://www.electronjs.org/docs/latest/tutorial/security
- **Electron IPC:** https://www.electronjs.org/docs/latest/tutorial/ipc

### Libraries
- **@anthropic-ai/sdk:** https://www.npmjs.com/package/@anthropic-ai/sdk
- **electron-store:** https://github.com/sindresorhus/electron-store
- **Assistant UI:** https://github.com/assistant-ui/assistant-ui
- **better-sqlite3:** https://github.com/WiseLibs/better-sqlite3

### Community Resources
- **Anthropic Cookbook:** Code examples and patterns
- **Electron React Boilerplates:** Starting templates
- **Claude Prompt Library:** https://github.com/langgptai/awesome-claude-prompts

---

## 12. Conclusion

Integrating Claude as an embedded companion in Flux is highly feasible with Anthropic's official TypeScript SDK. The recommended hybrid architecture (main process for Claude + Python backend for consciousness context) provides the best balance of security, performance, and deep integration with Dionysus's consciousness processing system.

The three persona system (Guide, Narrator, Companion) offers users flexible ways to interact with processing results, from educational guidance to real-time storytelling to thoughtful co-exploration.

**Key Success Factors:**
1. Secure API key management from day one
2. Main process handles all Claude API calls
3. Rich consciousness context from Python backend
4. Streaming UI for responsive feel
5. Clear persona differentiation for user value

With 10-15 days of focused development, Flux can have a production-ready consciousness-aware AI companion that transforms how users interact with document processing results.

---

**Research Date:** 2025-10-12
**Prepared For:** Flux Electron Desktop App - Dionysus Consciousness Processing System
