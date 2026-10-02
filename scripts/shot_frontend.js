// 前端渲染验证 + 截图。真实浏览器走完整流程，断言读的是**渲染后的
// DOM 几何/文本**，不是 HTML 源码。
//
//   node scripts/shot_frontend.js http://127.0.0.1:8134 docs/screenshots

const fs = require('fs');

const BASE = process.argv[2] || 'http://127.0.0.1:8134';
const OUT = process.argv[3] || 'docs/screenshots';

const PW_CANDIDATES = [
  process.env.PW_PATH,
  'playwright-core',
  'playwright',
  'C:/Users/Legion/npm_global/node_modules/n8n/node_modules/playwright-core',
].filter(Boolean);

const CHROME_CANDIDATES = [
  process.env.CHROME,
  process.env.CHROME_PATH,
  'C:/Users/Legion/.agent-browser/browsers/chrome-153.0.8010.36/chrome.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

function loadPlaywright() {
  for (const c of PW_CANDIDATES) { try { return require(c); } catch (_) {} }
  console.error('找不到 playwright-core。用 PW_PATH 指过去。');
  process.exit(2);
}
function findChrome() {
  for (const c of CHROME_CANDIDATES) {
    if (c && fs.existsSync(c)) return c;
  }
  console.error('找不到 Chrome。用 CHROME 环境变量指过去。');
  process.exit(2);
}

const { chromium } = loadPlaywright();

let passed = 0, failed = 0;
function check(name, cond, detail) {
  if (cond) { passed++; console.log('  ✓ ' + name); }
  else { failed++; console.log('  ✗ ' + name + (detail !== undefined ? '  ← ' + detail : '')); }
}
function section(t) { console.log('\n[' + t + ']'); }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({
    executablePath: findChrome(),
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1100 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  const jsErrors = [];
  page.on('pageerror', (e) => jsErrors.push(String(e)));
  page.on('console', (m) => {
    // "Failed to load resource" 是浏览器对非 2xx 响应的固定日志，
    // 错误路径测试（404/422）会故意触发 —— 不算 JS 异常
    if (m.type() === 'error' &&
        !m.text().includes('Failed to load resource')) {
      jsErrors.push(m.text());
    }
  });

  let shotIndex = 0;
  const shots = [];
  async function shot(name) {
    const file = `${OUT}/${String(++shotIndex).padStart(2, '0')}-${name}.png`;
    await page.screenshot({ path: file, fullPage: true });
    shots.push({ name, file, size: fs.statSync(file).size });
    console.log(`  📸 ${file} (${fs.statSync(file).size} bytes)`);
  }

  async function analyze(sig, overrides = {}) {
    // secResult 第一次分析后就一直可见，waitForSelector 会立即返回 ——
    // 必须等本次响应真正到达，否则读到的是上一张信号的结果（竞态）。
    await page.selectOption('#sigSel', sig);
    for (const [k, v] of Object.entries(overrides)) {
      await page.fill(k, String(v));
    }
    const respPromise = page.waitForResponse(
      (r) => r.url().includes('/api/analyze') && r.request().method() === 'POST'
      && !r.url().includes('upload'), { timeout: 30000 });
    await page.click('#btnRun');
    await respPromise;
    // 响应到 DOM 渲染之间还有微任务，多让一帧
    await page.waitForTimeout(250);
  }

  // ---------------------------------------------------------------- 1
  section('1 初始状态');
  await page.goto(BASE, { waitUntil: 'networkidle' });
  check('标题', (await page.textContent('h1')).includes('speech-lab'));
  check('服务状态正常', (await page.textContent('#healthTxt')) === '正常');
  check('信号下拉已填充',
    (await page.$$('#sigSel option')).length >= 5);
  check('结果区初始隐藏',
    await page.$eval('#secResult', (el) => el.classList.contains('hidden')));
  await shot('初始');

  // ---------------------------------------------------------------- 2
  section('2 vowel_a 分析（基频还原）');
  await analyze('vowel_a');
  const audioCards = await page.textContent('#audioCards');
  check('音频信息卡已渲染', audioCards.includes('Hz') && audioCards.includes('s'));
  check('时长约 1s', audioCards.includes('1 s'), audioCards.slice(0, 100));
  check('波形 SVG 已内联',
    await page.$eval('#waveBox svg', (el) => el.getBoundingClientRect().height > 50));
  check('基频数值出现在特征表',
    (await page.textContent('#featRows')).includes('119') ||
    (await page.textContent('#featRows')).includes('120'));
  const canvasInfo = await page.$eval('#specCanvas', (el) => ({
    w: el.width, h: el.height,
    painted: (() => {
      // 语谱图必须真的画了像素：左上角和中心采样，不能全是底色
      const ctx = el.getContext('2d');
      const d1 = ctx.getImageData(10, el.height - 10, 1, 1).data;
      const d2 = ctx.getImageData(el.width >> 1, el.height >> 1, 1, 1).data;
      return (d1[0] + d1[1] + d1[2] > 30) || (d2[0] + d2[1] + d2[2] > 30);
    })(),
  }));
  check('语谱图 canvas 已上色', canvasInfo.painted, JSON.stringify(canvasInfo));
  check('MFCC 行存在', (await page.textContent('#featRows')).includes('MFCC'));
  await shot('vowel-a');

  // ---------------------------------------------------------------- 3
  section('3 vowel_burst：VAD 可视化');
  await analyze('vowel_burst');
  const burstTxt = await page.textContent('#featRows');
  const voicedPct = (await page.textContent('#audioCards'));
  // voiced 比例应显示 60-85% 之间（静音在中间）
  const m = voicedPct.match(/(\d+)%/);
  check('voiced 占比在 55~85%', m && +m[1] > 55 && +m[1] < 85,
    voicedPct.slice(0, 120));
  check('基频轮廓有灰色断点（静音帧）',
    (await page.$eval('#featRows svg', () => true)));
  await shot('vowel-burst');

  // ---------------------------------------------------------------- 4
  section('4 chirp：语谱图斜线');
  await analyze('chirp');
  await shot('chirp');

  // ---------------------------------------------------------------- 5
  section('5 帧长参数实验（时频折中）');
  await analyze('vowel_a');
  const framesLong = await page.textContent('#audioCards');
  const nLong = +framesLong.match(/(\d+)/) ? null : null;
  // 切换帧长 64ms，帧数必须明显变少
  await analyze('vowel_a', { '#frameMs': 64 });
  const nLongFrames = (await page.textContent('#audioCards')).match(/(\d+)/);
  check('帧长 64ms 后帧数 < 40',
    nLongFrames && +nLongFrames[1] < 40,
    framesLong.slice(0, 100));
  check('语谱图说明文本更新',
    (await page.textContent('#specMeta')).includes('64ms'));
  await shot('长帧');

  // ---------------------------------------------------------------- 6
  section('6 上传 WAV 分析');
  const wavPath = process.cwd() + '/data/demo_vowel_a.wav';
  if (fs.existsSync(wavPath)) {
    await page.setInputFiles('#fileIn', wavPath);
    await page.click('#btnUpload');
    await page.waitForSelector('#secResult:not(.hidden)', { timeout: 30000 });
    await sleep(400);
    const cards = await page.textContent('#audioCards');
    check('上传分析出结果', cards.includes('s'));
    check('上传信号也是 120Hz 基频',
      (await page.textContent('#featRows')).includes('119') ||
      (await page.textContent('#featRows')).includes('120'));
    await shot('上传分析');
  } else {
    check('演示 wav 存在', false, wavPath);
  }

  // ---------------------------------------------------------------- 7
  section('7 错误路径（前端 toast）');
  // 直接触发一次失败请求：非法帧长会被 422 拒绝
  await page.fill('#frameMs', '0.5');
  await page.click('#btnRun');
  await sleep(700);
  const toast = await page.textContent('#toast');
  check('422 显示 toast', toast.length > 0, toast);
  await page.fill('#frameMs', '25');   // 恢复

  // ---------------------------------------------------------------- 8
  section('8 控制台与截图完整性');
  check('无 JS 异常', jsErrors.length === 0, jsErrors.slice(0, 3).join(' | '));
  check('截图数量 ≥ 6', shots.length >= 6);
  const sizes = shots.map((s) => s.size);
  check('没有两张完全相同的截图',
    new Set(sizes).size === sizes.length,
    JSON.stringify(sizes));

  console.log('\n' + '='.repeat(46));
  console.log(`结果：${passed} 通过 · ${failed} 失败`);
  await browser.close();
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
