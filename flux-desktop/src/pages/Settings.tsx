import React, { useState } from 'react';
import { Settings as SettingsIcon, Database, Key, Zap, Monitor, GitBranch, Cpu } from 'lucide-react';
import InnerWorkspaceMonitor from '../components/InnerWorkspaceMonitor';

const SettingsPage: React.FC = () => {
  const [darkMode, setDarkMode] = useState(true);
  const [logfireEnabled, setLogfireEnabled] = useState(false);
  const [projectsEnabled, setProjectsEnabled] = useState(true);
  const [disconnectScreen, setDisconnectScreen] = useState(true);
  const [thoughtseedWatching, setThoughtseedWatching] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState('openai');

  const providers = [
    { id: 'openai', name: 'OpenAI', status: 'connected' },
    { id: 'google', name: 'Google', status: 'error' },
    { id: 'openrouter', name: 'OpenRouter', status: 'error' },
    { id: 'ollama', name: 'Ollama', status: 'error' },
    { id: 'anthropic', name: 'Anthropic', status: 'error' },
    { id: 'grok', name: 'Grok', status: 'error' }
  ];

  const ToggleCard = ({ title, description, enabled, onToggle, icon: Icon, testId }) => (
    <div
      className="bg-gray-900 rounded-xl p-4 border border-gray-700 hover:border-gray-600 transition-colors"
      data-testid={testId}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-3">
          <Icon className="h-5 w-5 text-gray-400" />
          <h3 className="font-medium text-white">{title}</h3>
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onToggle(e.target.checked)}
            className="sr-only peer"
            data-testid={`${testId}-toggle`}
            data-state={enabled ? 'checked' : 'unchecked'}
          />
          <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 dark:peer-focus:ring-purple-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-purple-600"></div>
        </label>
      </div>
      <p className="text-sm text-gray-400">{description}</p>
    </div>
  );

  return (
    <div className="min-h-screen bg-black text-white p-6" data-testid="settings-container">
      {/* Header */}
      <div className="flex items-center space-x-3 mb-8">
        <SettingsIcon className="h-8 w-8 text-purple-400" data-testid="settings-gear-icon" />
        <h1 className="text-3xl font-bold">Settings</h1>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8" data-testid="settings-grid">

        {/* Left Column */}
        <div className="space-y-8">

          {/* Features Section */}
          <div data-testid="features-section">
            <div className="flex items-center space-x-2 mb-6">
              <Zap className="h-5 w-5 text-purple-400" />
              <h2 className="text-xl font-semibold text-purple-400">Features</h2>
            </div>

            <div className="space-y-4">
              <ToggleCard
                title="Dark Mode"
                description="Switch between light and dark themes"
                enabled={darkMode}
                onToggle={setDarkMode}
                icon={Monitor}
                testId="dark-mode-card"
              />

              <ToggleCard
                title="Pydantic Logfire"
                description="Structured logging and observability platform"
                enabled={logfireEnabled}
                onToggle={setLogfireEnabled}
                icon={Zap}
                testId="logfire-card"
              />

              <ToggleCard
                title="Projects"
                description="Enable Projects and Tasks functionality"
                enabled={projectsEnabled}
                onToggle={setProjectsEnabled}
                icon={GitBranch}
                testId="projects-card"
              />

              <ToggleCard
                title="Disconnect Screen"
                description="Show disconnect screen when server disconnects"
                enabled={disconnectScreen}
                onToggle={setDisconnectScreen}
                icon={Monitor}
                testId="disconnect-screen-card"
              />

              <ToggleCard
                title="ThoughtSeed Watching"
                description="Enable detailed state monitoring for ThoughtSeed competition"
                enabled={thoughtseedWatching}
                onToggle={setThoughtseedWatching}
                icon={Cpu}
                testId="thoughtseed-panel"
              />
            </div>
          </div>

          {/* Version & Updates */}
          <div data-testid="version-section">
            <div className="flex items-center space-x-2 mb-6">
              <GitBranch className="h-5 w-5 text-blue-400" />
              <h2 className="text-xl font-semibold text-blue-400">Version & Updates</h2>
            </div>

            <div className="bg-gray-900 rounded-xl p-6 border border-gray-700" data-testid="version-info">
              <h3 className="font-medium text-white mb-4">Version Information</h3>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Current Version</span>
                  <span className="text-white">0.1.0</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Latest Version</span>
                  <span className="text-gray-500">No releases found</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Status</span>
                  <span className="text-green-400" data-testid="version-status">✓ Up to date</span>
                </div>
              </div>
            </div>
          </div>

          {/* Database Migrations */}
          <div data-testid="migrations-section">
            <div className="flex items-center space-x-2 mb-6">
              <Database className="h-5 w-5 text-purple-400" data-testid="database-icon" />
              <h2 className="text-xl font-semibold text-purple-400">Database Migrations</h2>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-8">

          {/* API Keys */}
          <div data-testid="api-keys-section">
            <div className="flex items-center space-x-2 mb-6">
              <Key className="h-5 w-5 text-pink-400" />
              <h2 className="text-xl font-semibold text-pink-400">API Keys</h2>
            </div>

            <div className="bg-gray-900 rounded-xl p-6 border border-gray-700">
              <p className="text-gray-400 mb-6">
                Manage your API keys and credentials for various services used by Archon.
              </p>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">KEY NAME</label>
                  <div className="text-gray-400 mb-2">OPENAI_API_KEY</div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">VALUE</label>
                  <div className="flex items-center space-x-3">
                    <input
                      type="password"
                      value="sk-1234567890abcdef"
                      className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-white"
                      data-testid="openai-api-key-input"
                      placeholder="OPENAI_API_KEY"
                    />
                    <span className="text-gray-500" data-testid="openai-key-masked">••••••••••••</span>
                  </div>
                </div>

                <button
                  className="bg-pink-600 hover:bg-pink-700 text-white px-4 py-2 rounded-lg text-sm font-medium"
                  data-testid="add-credential-btn"
                >
                  + Add Credential
                </button>

                <div className="flex items-start space-x-2 p-3 bg-gray-800 rounded-lg" data-testid="encryption-notice">
                  <div className="text-pink-400">🔒</div>
                  <p className="text-sm text-gray-400">
                    Encrypted credentials are masked after saving. Click on a masked credential to edit it - this allows you to change the value and encryption settings.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* RAG Settings */}
          <div data-testid="rag-settings-section">
            <div className="flex items-center space-x-2 mb-6">
              <Cpu className="h-5 w-5 text-green-400" />
              <h2 className="text-xl font-semibold text-green-400">RAG Settings</h2>
            </div>

            <div className="bg-gray-900 rounded-xl p-6 border border-gray-700">
              <p className="text-gray-400 mb-6">
                Configure Retrieval-Augmented Generation (RAG) strategies for optimal knowledge retrieval.
              </p>

              {/* LLM Provider */}
              <div className="mb-6" data-testid="llm-provider-section">
                <label className="block text-sm font-medium text-gray-300 mb-3">LLM Provider</label>
                <div className="grid grid-cols-3 gap-2">
                  {providers.map((provider) => (
                    <button
                      key={provider.id}
                      onClick={() => setSelectedProvider(provider.id)}
                      className={`p-3 rounded-lg border-2 transition-colors ${
                        selectedProvider === provider.id
                          ? 'border-green-500 bg-green-900/20'
                          : 'border-gray-600 bg-gray-800'
                      }`}
                      data-testid={`provider-${provider.id}`}
                    >
                      <div className="text-sm font-medium text-white">{provider.name}</div>
                      <div className="flex justify-center mt-1">
                        {provider.status === 'connected' ? (
                          <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                        ) : (
                          <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Model Selection */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Chat Model</label>
                  <select
                    className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-white"
                    data-testid="chat-model-select"
                  >
                    <option value="gpt-4.1-nano">gpt-4.1-nano</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Embedding Model</label>
                  <select
                    className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-white"
                    data-testid="embedding-model-select"
                  >
                    <option value="text-embedding-3-small">text-embedding-3-small</option>
                  </select>
                </div>
              </div>

              {/* Contextual Embeddings */}
              <div className="mb-6">
                <label className="flex items-center space-x-3">
                  <input type="checkbox" className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 rounded" />
                  <div>
                    <span className="text-white">Use Contextual Embeddings</span>
                    <p className="text-sm text-gray-400">Enhances embeddings with contextual information for better retrieval</p>
                  </div>
                </label>
              </div>

              <button
                className="bg-gray-700 hover:bg-gray-600 text-white px-6 py-2 rounded-lg font-medium"
                data-testid="save-settings-btn"
              >
                💾 Save Settings
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ThoughtSeed Inner Workspace Monitor (shown when watching enabled) */}
      {thoughtseedWatching && (
        <div className="mt-8" data-testid="workspace-viewer">
          <InnerWorkspaceMonitor />
        </div>
      )}
    </div>
  );
};

export default SettingsPage;