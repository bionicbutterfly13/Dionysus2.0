/**
 * Flux Interface E2E Tests
 *
 * Per TDD process:
 * 1. All services MUST be online before tests run
 * 2. Tests use ACTUAL selectors from implementation
 * 3. Tests validate REAL functionality, not assumptions
 *
 * Run with: npx playwright test
 */

import { test, expect } from '@playwright/test';

const FRONTEND_URL = 'http://localhost:9243';
const BACKEND_URL = 'http://127.0.0.1:9127';

// Validate services before running any tests
test.beforeAll(async ({ request }) => {
  console.log('🔍 Validating required services...');

  try {
    const response = await request.get(`${BACKEND_URL}/api/health`);
    if (!response.ok()) {
      throw new Error('Backend health check failed');
    }

    const health = await response.json();

    if (health.overall_status === 'down') {
      console.error('❌ Services down:', health.errors);
      throw new Error(`Critical services offline: ${health.errors.join(', ')}`);
    }

    if (!health.can_upload) {
      console.warn('⚠️  Upload disabled:', health.errors);
    }

    console.log('✅ All services healthy');
    console.log(`   Neo4j: ${health.services.neo4j.status}`);
    console.log(`   Redis: ${health.services.redis.status}`);
    console.log(`   Daedalus: ${health.services.daedalus.status}`);

  } catch (error) {
    console.error('❌ Backend not responding');
    console.error('💡 Start backend with: ./START_FLUX.sh');
    throw error;
  }
});

test.describe('Flux Interface Critical Path', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
  });

  test('01: App loads with correct title and navigation', async ({ page }) => {
    // Actual title from frontend
    await expect(page).toHaveTitle(/Flux/);

    // Check sidebar navigation exists
    const sidebar = page.locator('.sidebar-dark');
    await expect(sidebar).toBeVisible();

    // Check for navigation links (actual routes from Layout.tsx)
    await expect(page.getByRole('link', { name: /dashboard/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /knowledge base/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /knowledge graph/i })).toBeVisible();
  });

  test('02: Knowledge Base page loads and shows add button', async ({ page }) => {
    // Navigate to Knowledge Base
    await page.getByRole('link', { name: /knowledge base/i }).click();
    await page.waitForLoadState('networkidle');

    // Verify URL changed
    expect(page.url()).toContain('/knowledge-base');

    // Check for "Add Knowledge" button (actual text from KnowledgeBase.tsx line 80)
    const addButton = page.getByRole('button', { name: /add knowledge/i });
    await expect(addButton).toBeVisible();
  });

  test('03: Upload modal opens when Add Knowledge clicked', async ({ page }) => {
    await page.getByRole('link', { name: /knowledge base/i }).click();
    await page.waitForLoadState('networkidle');

    // Click "Add Knowledge" button
    const addButton = page.getByRole('button', { name: /add knowledge/i });
    await addButton.click();

    // Modal should appear (DocumentUpload component)
    // Wait for modal to be visible
    await page.waitForTimeout(500);

    // Look for crawl/upload mode selector or file dropzone
    const modal = page.locator('[class*="modal"], [role="dialog"], .fixed').first();
    await expect(modal).toBeVisible({ timeout: 5000 });
  });

  test('04: Backend health endpoint responds correctly', async ({ page }) => {
    const response = await page.request.get(`${BACKEND_URL}/api/health`);
    expect(response.ok()).toBeTruthy();

    const health = await response.json();
    expect(health).toHaveProperty('overall_status');
    expect(health).toHaveProperty('can_upload');
    expect(health.services).toHaveProperty('neo4j');
    expect(health.services).toHaveProperty('redis');
    expect(health.services).toHaveProperty('daedalus');
  });

  test('05: Backend documents endpoint responds', async ({ page }) => {
    const response = await page.request.get(`${BACKEND_URL}/api/v1/documents`);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty('documents');
    expect(Array.isArray(data.documents)).toBeTruthy();
  });

  test('06: Navigation between pages works', async ({ page }) => {
    // Start on Dashboard
    expect(page.url()).toBe(`${FRONTEND_URL}/`);

    // Navigate to Knowledge Base
    await page.getByRole('link', { name: /knowledge base/i }).click();
    await page.waitForLoadState('networkidle');
    expect(page.url()).toContain('/knowledge-base');

    // Navigate to Knowledge Graph
    await page.getByRole('link', { name: /knowledge graph/i }).click();
    await page.waitForLoadState('networkidle');
    expect(page.url()).toContain('/knowledge-graph');

    // Navigate back to Dashboard
    await page.getByRole('link', { name: /dashboard/i }).click();
    await page.waitForLoadState('networkidle');
    expect(page.url()).toBe(`${FRONTEND_URL}/`);
  });

  test('07: Document sidebar shows documents count', async ({ page }) => {
    // Wait for documents to load
    await page.waitForTimeout(2000);

    // Check for "Documents" section in sidebar (Layout.tsx line 149)
    const documentsSection = page.locator('text=Documents (');
    await expect(documentsSection).toBeVisible();
  });

  test('08: Dashboard page renders without errors', async ({ page }) => {
    // Should already be on dashboard from beforeEach
    expect(page.url()).toBe(`${FRONTEND_URL}/`);

    // Check main content area exists
    const main = page.locator('main');
    await expect(main).toBeVisible();

    // No console errors (check via page.on('pageerror'))
    const errors: Error[] = [];
    page.on('pageerror', error => errors.push(error));

    await page.waitForTimeout(1000);
    expect(errors).toHaveLength(0);
  });

  test('09: Consciousness status is shown in sidebar', async ({ page }) => {
    // Check for "Consciousness Active" in sidebar (Layout.tsx line 247)
    const consciousnessStatus = page.locator('text=Consciousness Active');
    await expect(consciousnessStatus).toBeVisible();
  });

  test('10: Settings link exists and works', async ({ page }) => {
    // Find Settings link (Layout.tsx line 251-257)
    const settingsLink = page.getByRole('link', { name: /settings/i });
    await expect(settingsLink).toBeVisible();

    // Click it
    await settingsLink.click();
    await page.waitForLoadState('networkidle');

    // Verify URL changed
    expect(page.url()).toContain('/settings');
  });
});

test.describe('Document Management', () => {
  test('Document click navigates to detail page', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');

    // Wait for documents to load
    await page.waitForTimeout(2000);

    // Try to find first document in sidebar (Layout.tsx lines 158-177)
    const firstDoc = page.locator('[class*="border-gray-800"]').first();
    const docCount = await firstDoc.count();

    if (docCount > 0) {
      // Click the document title button
      const docButton = firstDoc.locator('button').first();
      await docButton.click();
      await page.waitForLoadState('networkidle');

      // Should navigate to /document/:id
      expect(page.url()).toContain('/document/');
    } else {
      console.log('⚠️  No documents found - skipping detail navigation test');
    }
  });
});
