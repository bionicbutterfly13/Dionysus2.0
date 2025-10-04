# Spec 042: Playwright UI Validation Tests

## Overview
Comprehensive Playwright tests that validate the actual UI elements exist and work correctly. Tests must match the real implementation, not assumptions.

## Problem Statement
Current Playwright tests fail because:
- Tests expect "Dionysus" but app title is "Flux"
- Tests look for upload buttons that don't exist at those selectors
- Tests assume UI structure that doesn't match implementation
- Tests don't validate services before running

## Requirements

### NFR-001: Test Real UI Elements
- Tests MUST use actual selectors from implementation
- Tests MUST verify against actual page titles
- Tests MUST NOT assume UI structure
- Tests MUST inspect DOM to find correct selectors

### NFR-002: Service Validation
- Tests MUST check `/api/health` before running
- Tests MUST skip if services unavailable
- Tests MUST NOT launch browser if backend down

### NFR-003: Comprehensive Coverage
- Test ALL critical user paths:
  - Document upload flow
  - Document list display
  - Document detail view
  - Navigation between pages
  - Dashboard stats display
  - Query interface
- Test ALL critical buttons and controls

### NFR-004: Maintainability
- Use data-testid attributes for stable selectors
- Avoid brittle CSS selectors
- Group related tests
- Clear test descriptions

## User Stories

### US-001: Upload Flow
**As a** user
**I want** to upload a document and see it processed
**So that** I can query it later

**Test Steps:**
1. Navigate to upload page
2. Click upload trigger (button, icon, or drop zone)
3. Select file from file picker
4. Verify upload starts (progress indicator)
5. Verify upload completes (success message)
6. Verify document appears in list

### US-002: Document List
**As a** user
**I want** to see all my uploaded documents
**So that** I can select one to view

**Test Steps:**
1. Navigate to documents page
2. Verify document list renders
3. Verify each document shows: title, date, tags
4. Click first document
5. Verify detail page loads

### US-003: Dashboard
**As a** user
**I want** to see system statistics
**So that** I can monitor usage

**Test Steps:**
1. Navigate to dashboard
2. Verify stats cards render
3. Verify stats show numbers (not loading state)
4. Verify graphs/charts render

## Technical Design

### Test Setup
```typescript
// tests/setup/global-setup.ts
import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🔍 Validating services...');

  // Check backend health
  try {
    const response = await fetch('http://127.0.0.1:9127/api/health');
    const health = await response.json();

    if (health.overall_status === 'down') {
      console.error('❌ Services down:', health.errors.join(', '));
      console.error('💡 Start services with: ./START_FLUX.sh');
      process.exit(1);
    }

    if (!health.can_upload) {
      console.warn('⚠️  Upload disabled:', health.errors.join(', '));
    }

    console.log('✅ All services healthy');
  } catch (error) {
    console.error('❌ Backend not responding');
    console.error('💡 Start backend with: ./START_FLUX.sh');
    process.exit(1);
  }

  // Inspect UI to find actual selectors
  const browser = await chromium.launch();
  const page = await browser.newPage();

  try {
    await page.goto('http://localhost:9243');
    const title = await page.title();
    console.log(`📄 Page title: "${title}"`);

    // Find upload trigger
    const uploadSelectors = [
      'button:has-text("Upload")',
      '[data-testid="upload-button"]',
      '[aria-label*="upload" i]',
      '.upload-button',
      'input[type="file"]'
    ];

    for (const selector of uploadSelectors) {
      const found = await page.locator(selector).count();
      if (found > 0) {
        console.log(`✅ Upload trigger found: ${selector}`);
        break;
      }
    }
  } finally {
    await browser.close();
  }
}

export default globalSetup;
```

### Updated Tests
```typescript
// tests/interface/upload-flow.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Document Upload Flow', () => {
  test('Complete upload workflow', async ({ page }) => {
    await page.goto('http://localhost:9243');

    // Actual title from inspection
    await expect(page).toHaveTitle(/Flux/);

    // Find upload trigger (inspect actual DOM)
    const uploadTrigger = page.locator('[data-testid="upload-trigger"], button:has-text("Upload"), input[type="file"]').first();
    await expect(uploadTrigger).toBeVisible({ timeout: 10000 });

    // Click or attach file depending on element type
    const tagName = await uploadTrigger.evaluate(el => el.tagName);
    if (tagName === 'INPUT') {
      // Direct file input
      await uploadTrigger.setInputFiles('tests/fixtures/sample.pdf');
    } else {
      // Button that opens file picker
      await uploadTrigger.click();
      // Handle file picker dialog
    }

    // Verify upload progress
    const progressIndicator = page.locator('[data-testid="upload-progress"], .upload-progress, [role="progressbar"]');
    await expect(progressIndicator).toBeVisible({ timeout: 5000 });

    // Verify upload success
    const successMessage = page.locator('[data-testid="upload-success"], .success-message, :has-text("Upload complete")');
    await expect(successMessage).toBeVisible({ timeout: 30000 });

    // Verify document in list
    await page.goto('http://localhost:9243/documents');
    const documentItem = page.locator('[data-testid="document-item"]').first();
    await expect(documentItem).toBeVisible();
  });
});
```

## Test Plan

### Phase 1: UI Inspection (Manual)
1. Open http://localhost:9243 in browser
2. Open DevTools → Elements
3. Document actual selectors:
   - Upload button/input selector
   - Document list container
   - Document item selector
   - Navigation links
   - Dashboard stats cards
4. Create selector mapping document

### Phase 2: Add Test IDs (Code Changes)
1. Add `data-testid` attributes to critical elements:
   ```typescript
   <button data-testid="upload-button">Upload</button>
   <div data-testid="documents-list">...</div>
   <div data-testid="document-item">...</div>
   <nav data-testid="main-nav">...</nav>
   ```

### Phase 3: Write Tests (TDD)
1. Write test for upload flow (expect to fail)
2. Run test, observe failure
3. Fix selectors based on actual DOM
4. Test passes
5. Repeat for each user story

### Phase 4: CI Integration
1. Add global setup validation
2. Configure test artifacts (screenshots, traces)
3. Add to GitHub Actions workflow

## Expected Test Results

### With All Services Up
```
✅ 01: App loads and shows navigation
✅ 02: Documents page loads
✅ 03: Upload flow works end-to-end
✅ 04: Dashboard shows stats
✅ 05: Document detail page loads
✅ 06: Navigation works
✅ 07: Backend endpoints respond
```

### With Neo4j Down
```
⏭️  All tests skipped - Neo4j not available
💡 Start Neo4j with: docker start neo4j-memory
```

### With Frontend Down
```
❌ Tests failed - Frontend not responding
💡 Start frontend with: cd frontend && npm run dev
```

## Dependencies
- Frontend running at localhost:9243
- Backend running at 127.0.0.1:9127
- Neo4j running (for upload tests)
- Playwright installed with browsers

## Success Criteria
1. ✅ Tests use actual selectors from implementation
2. ✅ Tests skip gracefully when services down
3. ✅ All critical user paths covered
4. ✅ Tests pass consistently on clean system
5. ✅ Test failures provide actionable error messages
