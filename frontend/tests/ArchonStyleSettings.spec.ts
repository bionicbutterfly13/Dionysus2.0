/**
 * Visual TDD Test for Archon-Style Settings Interface
 * ==================================================
 *
 * This test defines the visual layout and components that should match
 * the Archon interface design shown in the provided image.
 *
 * Test-First: These tests will FAIL until we build the interface.
 */

import { test, expect } from '@playwright/test';

test.describe('Archon-Style Settings Interface', () => {

  test.beforeEach(async ({ page }) => {
    // Navigate to our settings page (will need to create this route)
    await page.goto('http://localhost:3000/settings');
  });

  test('should have dark theme layout with proper structure', async ({ page }) => {
    // Main container should have dark background
    const mainContainer = page.locator('[data-testid="settings-container"]');
    await expect(mainContainer).toBeVisible();

    // Should have Settings title with gear icon
    const settingsTitle = page.locator('h1', { hasText: 'Settings' });
    await expect(settingsTitle).toBeVisible();

    // Should have gear icon next to title
    const gearIcon = page.locator('[data-testid="settings-gear-icon"]');
    await expect(gearIcon).toBeVisible();
  });

  test('should have left panel with Features section', async ({ page }) => {
    // Features section header
    const featuresSection = page.locator('[data-testid="features-section"]');
    await expect(featuresSection).toBeVisible();

    // Dark Mode toggle card
    const darkModeCard = page.locator('[data-testid="dark-mode-card"]');
    await expect(darkModeCard).toBeVisible();
    await expect(darkModeCard).toContainText('Dark Mode');
    await expect(darkModeCard).toContainText('Switch between light and dark themes');

    // Dark mode toggle should be ON (purple)
    const darkModeToggle = page.locator('[data-testid="dark-mode-card-toggle"]');
    await expect(darkModeToggle).toBeVisible();
    await expect(darkModeToggle).toHaveAttribute('data-state', 'checked');

    // Pydantic LogFire card
    const logfireCard = page.locator('[data-testid="logfire-card"]');
    await expect(logfireCard).toBeVisible();
    await expect(logfireCard).toContainText('Pydantic Logfire');
    await expect(logfireCard).toContainText('Structured logging and observability platform');

    // Projects card
    const projectsCard = page.locator('[data-testid="projects-card"]');
    await expect(projectsCard).toBeVisible();
    await expect(projectsCard).toContainText('Projects');
    await expect(projectsCard).toContainText('Enable Projects and Tasks functionality');

    // Disconnect Screen card
    const disconnectCard = page.locator('[data-testid="disconnect-screen-card"]');
    await expect(disconnectCard).toBeVisible();
    await expect(disconnectCard).toContainText('Disconnect Screen');
    await expect(disconnectCard).toContainText('Show disconnect screen when server disconnects');
  });

  test('should have Version & Updates section', async ({ page }) => {
    const versionSection = page.locator('[data-testid="version-section"]');
    await expect(versionSection).toBeVisible();

    // Version Information panel
    const versionInfo = page.locator('[data-testid="version-info"]');
    await expect(versionInfo).toBeVisible();
    await expect(versionInfo).toContainText('Version Information');

    // Should show current version
    await expect(versionInfo).toContainText('Current Version');
    await expect(versionInfo).toContainText('0.1.0');

    // Should show latest version status
    await expect(versionInfo).toContainText('Latest Version');
    await expect(versionInfo).toContainText('No releases found');

    // Should show up to date status with green indicator
    const statusIndicator = page.locator('[data-testid="version-status"]');
    await expect(statusIndicator).toContainText('Up to date');
    await expect(statusIndicator).toHaveClass(/text-green/);
  });

  test('should have Database Migrations section', async ({ page }) => {
    const migrationsSection = page.locator('[data-testid="migrations-section"]');
    await expect(migrationsSection).toBeVisible();
    await expect(migrationsSection).toContainText('Database Migrations');

    // Should have purple database icon
    const dbIcon = page.locator('[data-testid="database-icon"]');
    await expect(dbIcon).toBeVisible();
  });

  test('should have right panel with API Keys section', async ({ page }) => {
    const apiKeysSection = page.locator('[data-testid="api-keys-section"]');
    await expect(apiKeysSection).toBeVisible();
    await expect(apiKeysSection).toContainText('API Keys');

    // Should have description text
    await expect(apiKeysSection).toContainText('Manage your API keys and credentials for various services used by Archon');

    // Should have API key input for OPENAI_API_KEY
    const openaiKeyInput = page.locator('[data-testid="openai-api-key-input"]');
    await expect(openaiKeyInput).toBeVisible();
    await expect(openaiKeyInput).toHaveAttribute('placeholder', 'OPENAI_API_KEY');

    // Should have masked value display
    const maskedValue = page.locator('[data-testid="openai-key-masked"]');
    await expect(maskedValue).toContainText('••••••••••••');

    // Should have Add Credential button
    const addCredButton = page.locator('[data-testid="add-credential-btn"]');
    await expect(addCredButton).toBeVisible();
    await expect(addCredButton).toContainText('Add Credential');

    // Should have encryption notice
    const encryptionNotice = page.locator('[data-testid="encryption-notice"]');
    await expect(encryptionNotice).toBeVisible();
    await expect(encryptionNotice).toContainText('Encrypted credentials are masked after saving');
  });

  test('should have RAG Settings section', async ({ page }) => {
    const ragSection = page.locator('[data-testid="rag-settings-section"]');
    await expect(ragSection).toBeVisible();
    await expect(ragSection).toContainText('RAG Settings');

    // Should have description
    await expect(ragSection).toContainText('Configure Retrieval-Augmented Generation (RAG) strategies');

    // Should have LLM Provider selection
    const llmProviderSection = page.locator('[data-testid="llm-provider-section"]');
    await expect(llmProviderSection).toBeVisible();

    // Should show provider options with status indicators
    const providers = ['OpenAI', 'Google', 'OpenRouter', 'Ollama', 'Anthropic', 'Grok'];
    for (const provider of providers) {
      const providerBtn = page.locator(`[data-testid="provider-${provider.toLowerCase()}"]`);
      await expect(providerBtn).toBeVisible();
    }

    // OpenAI should be selected (green checkmark)
    const openaiProvider = page.locator('[data-testid="provider-openai"]');
    await expect(openaiProvider).toHaveClass(/border-green/);

    // Should have Chat Model and Embedding Model dropdowns
    const chatModel = page.locator('[data-testid="chat-model-select"]');
    await expect(chatModel).toBeVisible();
    await expect(chatModel).toContainText('gpt-4.1-nano');

    const embeddingModel = page.locator('[data-testid="embedding-model-select"]');
    await expect(embeddingModel).toBeVisible();
    await expect(embeddingModel).toContainText('text-embedding-3-small');

    // Should have Save Settings button
    const saveButton = page.locator('[data-testid="save-settings-btn"]');
    await expect(saveButton).toBeVisible();
    await expect(saveButton).toContainText('Save Settings');
  });

  test('should have proper card styling and hover effects', async ({ page }) => {
    const darkModeCard = page.locator('[data-testid="dark-mode-card"]');

    // Should have proper card styling
    await expect(darkModeCard).toHaveCSS('border-radius', '12px');
    await expect(darkModeCard).toHaveCSS('background-color', /rgb\(.*\)/); // Some dark color

    // Should have hover effect
    await darkModeCard.hover();
    // Card should have subtle hover state change
  });

  test('should have responsive grid layout', async ({ page }) => {
    // Should have two-column grid layout
    const settingsGrid = page.locator('[data-testid="settings-grid"]');
    await expect(settingsGrid).toHaveCSS('display', 'grid');
    // Grid uses responsive classes, so we just verify it's a grid
    await expect(settingsGrid).toHaveClass(/grid/);
  });

  test('should integrate ThoughtSeed watching panel', async ({ page }) => {
    // Our custom ThoughtSeed panel should be integrated
    const thoughtseedPanel = page.locator('[data-testid="thoughtseed-panel"]');
    await expect(thoughtseedPanel).toBeVisible();
    await expect(thoughtseedPanel).toContainText('ThoughtSeed Watching');

    // Should have toggle for watching
    const watchingToggle = page.locator('[data-testid="thoughtseed-panel-toggle"]');
    await expect(watchingToggle).toBeVisible();

    // Enable ThoughtSeed watching to show workspace viewer
    await page.locator('[data-testid="thoughtseed-panel"] label').click();

    // Should have workspace viewer after enabling watching
    const workspaceViewer = page.locator('[data-testid="workspace-viewer"]');
    await expect(workspaceViewer).toBeVisible();
  });
});

// Helper test for visual regression
test('should match visual design', async ({ page }) => {
  await page.goto('http://localhost:3000/settings');

  // Wait for all components to load
  await page.waitForSelector('[data-testid="settings-container"]');

  // Take screenshot and compare with reference
  await expect(page).toHaveScreenshot('archon-style-settings.png', {
    fullPage: true,
    threshold: 0.2 // Allow 20% difference for dynamic content
  });
});