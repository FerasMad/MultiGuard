const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1320, height: 1200, deviceScaleFactor: 2 });

  const url = 'file://' + path.resolve(__dirname, 'Algorithms.html');
  await page.goto(url, { waitUntil: 'networkidle0' });

  // Give JS time to inject diagrams and fonts to settle.
  await new Promise(r => setTimeout(r, 1500));

  await page.pdf({
    path: path.resolve(__dirname, 'Algorithms.pdf'),
    width: '1320px',
    height: '900px',
    printBackground: true,
    margin: { top: '24px', right: '24px', bottom: '24px', left: '24px' },
    preferCSSPageSize: false,
  });

  await browser.close();
  console.log('PDF written to Algorithms.pdf');
})();
