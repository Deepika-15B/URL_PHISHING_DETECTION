const { chromium } = require('playwright');

(async () => {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    let hasErrors = false;

    page.on('pageerror', err => {
        console.error('Uncaught exception in browser:', err);
        hasErrors = true;
    });

    page.on('console', msg => {
        if (msg.type() === 'error') {
            console.error('Browser console error:', msg.text());
            if (msg.text().includes('Cannot read properties')) {
                hasErrors = true;
            }
        }
    });

    await page.goto('http://127.0.0.1:5000');

    async function testUrl(url) {
        console.log(`\nTesting URL: ${url}`);
        await page.fill('#urlInput', url);
        await page.click('#analyzeBtn');
        
        try {
            await page.waitForSelector('#loadingState', { state: 'hidden', timeout: 45000 });
            await page.waitForTimeout(500); // give it a moment to render
        } catch (e) {
            console.log("Timeout waiting for loading state to hide.");
        }

        const isUnreachable = await page.isVisible('#unreachableCard');
        const isPred = await page.isVisible('#predictionCard');
        
        let label = "Unknown";
        if (isUnreachable) {
             label = await page.innerText('#unreachableCard .badge-prediction');
             console.log(`Warning Card Title: ${await page.innerText('#unreachableCard .card-title')}`);
        } else if (isPred) {
             label = await page.innerText('#predictionResult');
        }
        
        console.log(`Result Label: ${label}`);
        console.log(`Errors so far: ${hasErrors}`);
    }

    // 1. Legitimate site
    await testUrl('https://www.youtube.com');
    // 2. Unreachable site
    await testUrl('https://www.irctc.co.in');
    // 3. Partial Extraction (Amazon)
    await testUrl('https://www.amazon.in');
    // 4. Bot Protection (StackOverflow)
    await testUrl('https://stackoverflow.com');

    await browser.close();
    if (hasErrors) {
        console.log("\nTEST FAILED: Frontend errors detected.");
        process.exit(1);
    } else {
        console.log("\nTEST PASSED: No frontend errors detected.");
    }
})();
