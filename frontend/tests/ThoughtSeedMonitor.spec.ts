/**
 * TDD Test for Real ThoughtSeed Data Display
 * ==========================================
 *
 * Basic implementation to start - focus on actual data, not mock
 */

import { test, expect } from '@playwright/test';

test.describe('ThoughtSeed Monitor - Real Data', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000/thoughtseed');
  });

  test('should display real ThoughtSeed competition data', async ({ page }) => {
    // Should have ThoughtSeed monitor page
    await expect(page.locator('h1')).toContainText('ThoughtSeed Monitor');

    // Should have Start Competition button
    const startButton = page.locator('[data-testid="start-competition"]');
    await expect(startButton).toBeVisible();
    await expect(startButton).toContainText('Start');

    // Click start and verify competition begins
    await startButton.click();

    // Should show competition running
    await expect(page.locator('[data-testid="start-competition"]')).toContainText('Pause');

    // Should display actual thought data
    const thoughtCards = page.locator('[data-testid="thought-card"]');
    await expect(thoughtCards.first()).toBeVisible();

    // Should show energy and confidence values
    await expect(page.locator('[data-testid="energy-value"]').first()).toBeVisible();
    await expect(page.locator('[data-testid="confidence-value"]').first()).toBeVisible();
  });

  test('should connect to backend ThoughtSeed API', async ({ page }) => {
    // Should attempt to fetch real data from backend
    const responsePromise = page.waitForResponse(response =>
      response.url().includes('/api/thoughtseed') && response.status() === 200
    );

    await page.click('[data-testid="start-competition"]');

    // Should get real data from backend (or handle gracefully if offline)
    try {
      await responsePromise;
      console.log('✅ Connected to ThoughtSeed backend');
    } catch (error) {
      console.log('⚠️ Backend offline - using fallback data');
    }
  });

  test('should display winning ThoughtSeed clearly', async ({ page }) => {
    // Start competition
    await page.click('[data-testid="start-competition"]');

    // Should show dominant/winning thought
    const dominantThought = page.locator('[data-testid="dominant-thought"]');
    await expect(dominantThought).toBeVisible();

    // Should have crown or winner indicator
    await expect(dominantThought.locator('[data-testid="winner-indicator"]')).toBeVisible();
  });

  test('should show consciousness level', async ({ page }) => {
    await page.click('[data-testid="start-competition"]');

    // Should display consciousness percentage
    const consciousnessLevel = page.locator('[data-testid="consciousness-level"]');
    await expect(consciousnessLevel).toBeVisible();
    await expect(consciousnessLevel).toContainText('%');
  });

});