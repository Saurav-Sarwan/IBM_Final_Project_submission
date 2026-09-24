const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// Absolute paths
const dashboardPath = path.resolve(__dirname, '../../frontend/dashboard.html');
const outputDir = path.resolve(__dirname, '../../outputs');

const tabs = [
  { id: 'executive', btn: 0, file: 'screenshot_executive.png',  label: 'Executive Summary' },
  { id: 'insights',  btn: 1, file: 'screenshot_insights.png',   label: 'Risks & Opportunities' },
  { id: 'actions',   btn: 2, file: 'screenshot_actions.png',    label: 'Recommended Actions' },
];

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-web-security', '--allow-file-access-from-files'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1.5 });

  // Use file:// URL and allow local file access
  const url = 'file:///' + dashboardPath.replace(/\\/g, '/');
  console.log('Loading:', url);
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });

  // Wait for loading overlay to disappear OR 8 seconds
  try {
    await page.waitForSelector('#loading', { hidden: true, timeout: 8000 });
  } catch(e) {
    console.log('Loading overlay still visible — proceeding anyway');
  }
  // Extra render time for Chart.js
  await new Promise(r => setTimeout(r, 2500));

  for (const tab of tabs) {
    // Click the tab button by index
    const btns = await page.$$('.tab-btn');
    if (btns[tab.btn]) {
      await btns[tab.btn].click();
      await new Promise(r => setTimeout(r, 1200));
    }

    // Full page height
    const bodyH = await page.evaluate(() => document.body.scrollHeight);
    await page.setViewport({ width: 1440, height: Math.min(bodyH, 2400), deviceScaleFactor: 1.5 });
    await new Promise(r => setTimeout(r, 400));

    const outPath = path.join(outputDir, tab.file);
    await page.screenshot({ path: outPath, fullPage: true });
    console.log(`[saved] ${outPath}  (${tab.label})`);

    // Reset viewport
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1.5 });
  }

  await browser.close();
  console.log('All screenshots saved.');
})();
