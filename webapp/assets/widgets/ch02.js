(function () {
  'use strict';

  var ERAS = {
    origin: { label: '偏好 RL 起源', color: 'var(--accent)',   range: [2007.4, 2018.95] },
    lm:     { label: '語言模型時代', color: 'var(--accent-2)', range: [2018.95, 2022.72] },
    gpt:    { label: 'ChatGPT 時代', color: 'var(--link)',     range: [2022.72, 2026.5] }
  };

  var EVENTS = [
    { t: 2008.3, year: '2008', era: 'origin', title: 'TAMER：人類評分訓練代理人',
      desc: '由人類反覆為代理人（agent）的行動評分，先學出一個獎勵模型（reward model），再用它學習行動策略。這種「先建模人類回饋、再做最佳化」的兩段式設計，正是日後 RLHF 流程的雛形。' },
    { t: 2017.05, year: '2017', era: 'origin', title: 'COACH：回饋調整優勢函數',
      desc: '演員-評論家（actor-critic）演算法 COACH 利用人類的正向與負向回饋來調整優勢函數。它說明除了「先學獎勵模型」之外，人類訊號還有其他嵌入 RL 更新的方式。' },
    { t: 2017.55, year: '2017', era: 'origin', title: 'Christiano et al.：引入 RLHF',
      desc: '現代 RLHF 最主要的參考文獻：讓人類在 Atari 代理人的軌跡（trajectories）之間表達偏好，非同步訓練獎勵預測器，代理人再最大化預測出的獎勵。它證明在某些領域，讓人類比較軌跡比直接與環境互動更有效。' },
    { t: 2018.4, year: '2018', era: 'origin', title: '獎勵建模轉向對齊研究',
      desc: 'TAMER 被擴展到神經網路（Deep TAMER），獎勵建模研究進一步延伸 Christiano 的方法。更關鍵的轉變是：獎勵模型開始被提出作為研究對齊（alignment）的一般性方法，而不再只是解決 RL 問題的工具。' },
    { t: 2019.5, year: '2019', era: 'lm', title: '微調語言模型：GPT-2 上的偏好 RL',
      desc: '《Fine-Tuning Language Models from Human Preferences》首次把人類偏好 RL 搬上語言模型。學習獎勵模型、KL 距離、回饋流程圖等經典概念都在此正式確立，與現代 RLHF 有驚人的相似之處。' },
    { t: 2020.6, year: '2020', era: 'lm', title: '學習摘要：RLHF 的首個殺手級任務',
      desc: '把 RLHF 應用於一般摘要，證明以人類偏好訓練的模型能在真實語言任務上超越監督式基準，後續更延伸到書籍的遞迴式摘要。RLHF 從遊戲走進了自然語言處理。' },
    { t: 2021.65, year: '2021', era: 'lm', title: 'WebGPT 等：走向助理行為',
      desc: 'RLHF 被應用到瀏覽器輔助問答（WebGPT）、附引用來源的回答（GopherCite）與一般對話（Sparrow）。RLHF 從單一任務最佳化，走向訓練「有用且可查證」的助理行為。' },
    { t: 2022.12, year: '2022', era: 'lm', title: 'InstructGPT：指令遵循三階段',
      desc: '把 RLHF 用於指令遵循，確立「監督微調 → 獎勵模型 → RL 最佳化」的流程，是 ChatGPT 的直接前身。同期研究也定義了獎勵模型過度最佳化與紅隊測試（red teaming）等關鍵議題。' },
    { t: 2022.5, year: '2022', era: 'lm', title: 'Anthropic 早期 Claude 大量採用 RLHF',
      desc: 'Anthropic 在 Claude 的早期版本中持續大量使用 RLHF，訓練有幫助且無害的對話助理；早期的 RLHF 開源工具也隨之出現。把 RLHF 精煉並應用於聊天模型的工作全面展開。' },
    { t: 2022.92, year: '2022', era: 'gpt', title: 'ChatGPT：RLHF 走入大眾視野',
      desc: '發布公告明確說明：採用與 InstructGPT 相同的 RLHF 方法訓練，僅資料蒐集設定略有差異。RLHF 一夕之間從研究技術變成家喻戶曉產品背後的關鍵配方，也是本書聚焦的時代轉折點。' },
    { t: 2023.35, year: '2023', era: 'gpt', title: '憲法式 AI 與 Claude',
      desc: 'Anthropic 以一組「憲法」原則讓 AI 自己產生回饋來訓練 Claude，減少對人工標註的依賴。它展示了人類回饋可以被 AI 回饋放大甚至部分取代，開啟 RLAIF 一系列研究。' },
    { t: 2023.8, year: '2023', era: 'gpt', title: 'DPO：直接偏好最佳化',
      desc: '直接偏好最佳化（Direct Preference Optimization, DPO）跳過獨立的獎勵模型與 RL 最佳化器，直接用偏好資料最佳化策略，催生出一整族直接對齊演算法，大幅降低偏好微調的門檻。' },
    { t: 2024.55, year: '2024', era: 'gpt', title: 'Llama 3、Tülu 3：配方公開化',
      desc: 'Meta 的 Llama 2/3、Nvidia 的 Nemotron、Ai2 的 Tülu 3 等公開權重模型都採用 RLHF 與偏好微調，完整訓練配方逐漸透明。RLHF 成長為更廣泛的偏好微調（PreFT）領域。' },
    { t: 2025.2, year: '2025', era: 'gpt', title: 'DeepSeek R1 與推理模型',
      desc: '受 OpenAI o1 啟發的線上推理方法興起：針對中間推理步驟的過程獎勵、從程式碼與數學的執行回饋中學習。RLHF 的技術棧被延伸到推理訓練，領域仍在快速演進。' }
  ];

  var DEFAULT_IDX = 9; // ChatGPT：時代轉折點
  var SVG_NS = 'http://www.w3.org/2000/svg';
  var W = 740, H = 132, AXIS_Y = 92, PAD_L = 26, PAD_R = 18;
  var YEAR_MIN = 2007.4, YEAR_MAX = 2026.5;

  function x(t) { return PAD_L + (t - YEAR_MIN) / (YEAR_MAX - YEAR_MIN) * (W - PAD_L - PAD_R); }

  function svgEl(tag, attrs) {
    var n = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }

  function render(rootEl) {
    var state = { idx: DEFAULT_IDX, filter: 'all' };

    // --- era 篩選按鈕列 ---
    var filterRow = document.createElement('div');
    filterRow.className = 'widget-row';
    filterRow.style.cssText = 'margin-bottom:.75rem;gap:.5rem;';
    var filterBtns = {};
    [['all', '全部'], ['origin', ERAS.origin.label], ['lm', ERAS.lm.label], ['gpt', ERAS.gpt.label]].forEach(function (f) {
      var b = document.createElement('button');
      if (f[0] !== 'all') {
        var swatch = document.createElement('span');
        swatch.style.cssText = 'display:inline-block;width:.6em;height:.6em;border-radius:50%;margin-right:.4em;background:' + ERAS[f[0]].color + ';';
        b.appendChild(swatch);
      }
      b.appendChild(document.createTextNode(f[1]));
      b.addEventListener('click', function () {
        state.filter = f[0];
        var vis = visibleEvents();
        if (vis.indexOf(EVENTS[state.idx]) === -1) state.idx = EVENTS.indexOf(vis[0]);
        update();
      });
      filterBtns[f[0]] = b;
      filterRow.appendChild(b);
    });

    // --- SVG 時間軸 ---
    var svgWrap = document.createElement('div');
    svgWrap.style.cssText = 'overflow-x:auto;padding-bottom:.25rem;';
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'RLHF 發展時間軸' });
    svg.style.cssText = 'display:block;width:100%;min-width:620px;height:auto;';
    svgWrap.appendChild(svg);

    Object.keys(ERAS).forEach(function (key) { // 時代底色與標籤
      var e = ERAS[key];
      var x0 = x(e.range[0]), x1 = x(e.range[1]);
      var band = svgEl('rect', { x: x0, y: 12, width: x1 - x0, height: AXIS_Y - 4, rx: 4, fill: e.color, opacity: '0.08' });
      var lbl = svgEl('text', { x: x0 + 6, y: 26, 'font-size': '10.5', fill: e.color });
      lbl.textContent = e.label;
      svg.appendChild(band); svg.appendChild(lbl);
    });

    svg.appendChild(svgEl('line', { x1: x(YEAR_MIN), y1: AXIS_Y, x2: x(YEAR_MAX), y2: AXIS_Y, stroke: 'var(--border)', 'stroke-width': '1.5' }));
    for (var yr = 2008; yr <= 2026; yr += 2) { // 年份刻度
      svg.appendChild(svgEl('line', { x1: x(yr), y1: AXIS_Y, x2: x(yr), y2: AXIS_Y + 6, stroke: 'var(--border)', 'stroke-width': '1' }));
      var t = svgEl('text', { x: x(yr), y: AXIS_Y + 20, 'font-size': '10', 'text-anchor': 'middle', fill: 'var(--fg-muted)' });
      t.textContent = yr;
      svg.appendChild(t);
    }

    var dots = EVENTS.map(function (ev, i) { // 事件圓點
      var g = svgEl('g', {});
      g.style.cursor = 'pointer';
      var ring = svgEl('circle', { cx: x(ev.t), cy: AXIS_Y, r: 11.5, fill: 'none', stroke: ERAS[ev.era].color, 'stroke-width': '1.5', visibility: 'hidden' });
      var dot = svgEl('circle', { cx: x(ev.t), cy: AXIS_Y, r: 6, fill: ERAS[ev.era].color, stroke: 'var(--bg)', 'stroke-width': '1.5' });
      var hit = svgEl('circle', { cx: x(ev.t), cy: AXIS_Y, r: 13, fill: 'transparent' });
      var tip = svgEl('title', {});
      tip.textContent = ev.year + '　' + ev.title;
      g.appendChild(ring); g.appendChild(dot); g.appendChild(hit); g.appendChild(tip);
      g.addEventListener('click', function () {
        state.idx = i;
        if (state.filter !== 'all' && ev.era !== state.filter) state.filter = 'all';
        update();
      });
      svg.appendChild(g);
      return { g: g, dot: dot, ring: ring };
    });

    // --- 上一個／下一個導覽列 ---
    var navRow = document.createElement('div');
    navRow.className = 'widget-row';
    navRow.style.cssText = 'margin:.5rem 0 .75rem;justify-content:space-between;';
    var prevBtn = document.createElement('button'); prevBtn.textContent = '← 上一個';
    var nextBtn = document.createElement('button'); nextBtn.textContent = '下一個 →';
    var counter = document.createElement('span');
    counter.style.cssText = 'color:var(--fg-muted);font-size:.9rem;';
    prevBtn.addEventListener('click', function () { step(-1); });
    nextBtn.addEventListener('click', function () { step(1); });
    navRow.appendChild(prevBtn); navRow.appendChild(counter); navRow.appendChild(nextBtn);

    // --- 事件卡片 ---
    var card = document.createElement('div');
    card.className = 'widget-panel';
    card.style.minHeight = '8em';
    var badge = document.createElement('span');
    badge.style.cssText = 'display:inline-block;padding:.1em .7em;border-radius:999px;font-size:.75rem;border:1px solid;margin-bottom:.5rem;';
    var heading = document.createElement('div');
    heading.style.cssText = 'font-weight:700;font-size:1.05rem;color:var(--fg);margin-bottom:.35rem;';
    var desc = document.createElement('p');
    desc.style.cssText = 'margin:0;line-height:1.75;color:var(--fg);';
    card.appendChild(badge); card.appendChild(heading); card.appendChild(desc);

    function visibleEvents() {
      return state.filter === 'all' ? EVENTS : EVENTS.filter(function (ev) { return ev.era === state.filter; });
    }

    function step(dir) {
      var vis = visibleEvents();
      var pos = vis.indexOf(EVENTS[state.idx]);
      pos = pos === -1 ? 0 : (pos + dir + vis.length) % vis.length; // 循環瀏覽
      state.idx = EVENTS.indexOf(vis[pos]);
      update();
    }

    function update() {
      var vis = visibleEvents();
      EVENTS.forEach(function (ev, i) {
        var selected = i === state.idx;
        var matched = state.filter === 'all' || ev.era === state.filter;
        dots[i].g.setAttribute('opacity', matched ? '1' : '0.18');
        dots[i].dot.setAttribute('r', selected ? '8.5' : '6');
        dots[i].ring.setAttribute('visibility', selected ? 'visible' : 'hidden');
      });
      Object.keys(filterBtns).forEach(function (k) {
        var active = k === state.filter;
        filterBtns[k].style.outline = active ? '2px solid var(--accent)' : '';
        filterBtns[k].style.background = active ? 'var(--accent-soft)' : '';
        filterBtns[k].setAttribute('aria-pressed', active);
      });
      var ev = EVENTS[state.idx];
      var color = ERAS[ev.era].color;
      badge.textContent = ERAS[ev.era].label;
      badge.style.color = color;
      badge.style.borderColor = color;
      heading.textContent = ev.year + '　' + ev.title;
      desc.textContent = ev.desc;
      counter.textContent = (vis.indexOf(ev) + 1) + ' / ' + vis.length;
    }

    rootEl.appendChild(filterRow);
    rootEl.appendChild(svgWrap);
    rootEl.appendChild(navRow);
    rootEl.appendChild(card);
    update();
  }

  window.ChapterWidget = {
    title: 'RLHF 發展互動時間軸',
    intro: '點擊時間軸上的圓點，或用「上一個／下一個」按鈕，走一遍 RLHF 從偏好 RL 起源到 ChatGPT 時代的 14 個里程碑；也可以用時代按鈕篩選。',
    render: render
  };
})();
