import { chromium } from 'playwright';

async function takeScreenshots() {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  // Set viewport size for desktop
  await page.setViewportSize({ width: 1920, height: 1080 });

  // Navigate to the dev server
  await page.goto('http://localhost:1420/', { waitUntil: 'networkidle' });

  // Wait a bit for any loading
  await page.waitForTimeout(2000);

  // Take full page screenshot
  await page.screenshot({
    path: 'desktop-app-screenshot.png',
    fullPage: true
  });

  console.log('✅ Screenshot saved to desktop-app-screenshot.png');

  await browser.close();
}

takeScreenshots().catch(console.error);
