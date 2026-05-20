const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1320, height: 1200, deviceScaleFactor: 2 });

  const url = 'file://' + path.resolve(__dirname, 'Algorithms.html');
  await page.goto(url, { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1500));

  const figs = await page.$$('figure.diag');
  for (let i = 0; i < figs.length; i++) {
    const out = path.resolve(__dirname, `Figure_${i + 1}.png`);
    await figs[i].screenshot({ path: out, omitBackground: false });
    console.log('Saved', out);
  }

  await browser.close();
})();
