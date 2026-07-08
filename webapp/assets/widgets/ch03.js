/* 第 3 章互動元件：溫控器 RL 模擬器（呼應 3.1.1 agent/environment/state/action/reward 形式化） */
(function () {
  'use strict';

  var N = 100;          // 回合步數（時間視野 T）
  var HEAT = 1.0;       // 暖氣加熱功率（°C／步）
  var LEAK = 0.05;      // 散熱係數
  var NOISE = 0.25;     // 環境隨機噪聲幅度
  var SVG_NS = 'http://www.w3.org/2000/svg';

  // 可調參數（滑桿）
  var params = { target: 22, outside: 10, eps: 2.0, dead: 0.5 };
  var sim = null;       // 最近一次模擬結果 {T, A, R, G, switches, inBand}
  var rafId = 0;

  // 需在多個函式間共用的 DOM 參照
  var ui = {};

  /* ---------- 環境與策略 ---------- */
  function policy(T, prevOn) { // bang-bang（帶死區的遲滯控制）
    if (T < params.target - params.dead) return true;
    if (T > params.target + params.dead) return false;
    return prevOn;
  }
  function rewardFn(T) { // |T−T*|<ε 得 +1，否則按偏差比例懲罰
    var d = Math.abs(T - params.target);
    return d < params.eps ? 1 : -0.5 * d;
  }
  function simulate() {
    var T = [params.outside], A = [], R = [], prevOn = false, G = 0, sw = 0, inBand = 0;
    for (var t = 0; t < N; t++) {
      var on = policy(T[t], prevOn);
      if (t > 0 && on !== prevOn) sw++;
      prevOn = on;
      A.push(on);
      var r = rewardFn(T[t]);
      R.push(r); G += r;
      if (r > 0) inBand++;
      var noise = (Math.random() * 2 - 1) * NOISE;
      T.push(T[t] + (on ? HEAT : 0) - LEAK * (T[t] - params.outside) + noise);
    }
    sim = { T: T, A: A, R: R, G: G, switches: sw, inBand: inBand / N };
  }

  /* ---------- 小工具 ---------- */
  function el(tag, style, text) {
    var e = document.createElement(tag);
    if (style) e.style.cssText = style;
    if (text != null) e.textContent = text;
    return e;
  }
  function svg(tag, attrs, parent) {
    var e = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function fmt(x, d) { return x.toFixed(d == null ? 1 : d); }

  /* ---------- 圖表 ---------- */
  var PL = 46, PR = 710, PT = 16, PB = 208; // 溫度圖繪圖區
  var RZ = 272, RUNIT = 18;                 // 獎勵條零線與單位高度
  var scaleY = null;

  function xOf(t) { return PL + (t / N) * (PR - PL); }

  function drawStatic() { // 依當前模擬決定溫度軸範圍，畫座標軸／容忍帶／目標線
    var g = ui.gStatic;
    while (g.firstChild) g.removeChild(g.firstChild);
    var lo = Math.min(Math.min.apply(null, sim.T), params.outside, params.target - params.eps) - 1;
    var hi = Math.max(Math.max.apply(null, sim.T), params.target + params.eps) + 1;
    if (hi - lo < 8) { var c = (hi + lo) / 2; lo = c - 4; hi = c + 4; }
    scaleY = function (v) { return PB - (v - lo) / (hi - lo) * (PB - PT); };

    // ±ε 容忍帶與目標線
    svg('rect', { x: PL, y: scaleY(params.target + params.eps), width: PR - PL,
      height: scaleY(params.target - params.eps) - scaleY(params.target + params.eps),
      fill: 'var(--accent-soft)' }, g);
    svg('line', { x1: PL, x2: PR, y1: scaleY(params.target), y2: scaleY(params.target),
      stroke: 'var(--accent-2)', 'stroke-width': 1.5, 'stroke-dasharray': '6 4' }, g);
    svg('text', { x: PR - 4, y: scaleY(params.target) - 5, 'text-anchor': 'end',
      'font-size': 11, fill: 'var(--accent-2)' }, g).textContent = '目標 T* ± ε';

    // 溫度軸刻度
    var span = hi - lo, step = span > 28 ? 10 : span > 14 ? 5 : 2;
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      var y = scaleY(v);
      svg('line', { x1: PL, x2: PR, y1: y, y2: y, stroke: 'var(--border)', 'stroke-width': 1 }, g);
      svg('text', { x: PL - 6, y: y + 4, 'text-anchor': 'end', 'font-size': 10,
        fill: 'var(--fg-muted)' }, g).textContent = v + '°';
    }
    // 時間軸刻度
    for (var t = 0; t <= N; t += 25) {
      svg('text', { x: xOf(t), y: 236, 'text-anchor': 'middle', 'font-size': 10,
        fill: 'var(--fg-muted)' }, g).textContent = 't=' + t;
    }
    svg('text', { x: PL, y: PT - 4, 'font-size': 11, fill: 'var(--fg-muted)' }, g)
      .textContent = '室溫（°C）　橘色細條＝暖氣開啟';
    // 獎勵區零線與標題
    svg('line', { x1: PL, x2: PR, y1: RZ, y2: RZ, stroke: 'var(--border)', 'stroke-width': 1 }, g);
    svg('text', { x: PL, y: RZ - 24, 'font-size': 11, fill: 'var(--fg-muted)' }, g)
      .textContent = '每步獎勵 r（綠＝+1、橘＝懲罰）';
  }

  function drawFrame(upTo) { // 畫出前 upTo 步的軌跡、暖氣狀態條與獎勵條
    var g = ui.gDyn;
    while (g.firstChild) g.removeChild(g.firstChild);
    var pts = [];
    for (var t = 0; t <= upTo; t++) pts.push(fmt(xOf(t), 1) + ',' + fmt(scaleY(sim.T[t]), 1));
    svg('polyline', { points: pts.join(' '), fill: 'none', stroke: 'var(--accent)',
      'stroke-width': 2, 'stroke-linejoin': 'round' }, g);
    var barW = (PR - PL) / N * 0.7;
    for (t = 0; t < upTo; t++) {
      if (sim.A[t]) svg('rect', { x: xOf(t), y: 214, width: (PR - PL) / N + 0.5, height: 6,
        fill: 'var(--accent-2)', opacity: 0.65 }, g);
      var r = Math.max(sim.R[t], -2.5), h = Math.abs(r) * RUNIT;
      svg('rect', { x: xOf(t) - barW / 2, y: r > 0 ? RZ - h : RZ, width: barW, height: h,
        fill: r > 0 ? 'var(--accent)' : 'var(--accent-2)', opacity: 0.85 }, g);
    }
    if (upTo > 0) svg('circle', { cx: xOf(upTo - 1), cy: scaleY(sim.T[upTo - 1]), r: 4,
      fill: 'var(--accent)', stroke: 'var(--bg)', 'stroke-width': 1.5 }, g);
  }

  /* ---------- 側欄（RL 迴圈即時狀態）與解讀 ---------- */
  function updateLoopPanel(t) {
    var i = Math.max(0, Math.min(t, N - 1));
    var s = sim.T[i], a = sim.A[i], r = sim.R[i], G = 0;
    for (var k = 0; k <= i; k++) G += sim.R[k];
    ui.stepEl.textContent = '第 t = ' + i + ' 步（共 ' + N + ' 步）';
    ui.sarEl.innerHTML = '狀態 s = <b>' + fmt(s) + '°C</b> → 動作 a = <b style="color:var(--accent-2)">' +
      (a ? '開啟暖氣' : '關閉暖氣') + '</b> → 獎勵 r = <b style="color:var(--accent)">' +
      (r > 0 ? '+1' : fmt(r, 2)) + '</b>';
    ui.retEl.textContent = fmt(G);
  }

  function interpret() {
    var maxT = params.outside + HEAT / LEAK;
    var pct = Math.round(sim.inBand * 100);
    var parts = [];
    if (maxT < params.target - 0.5) {
      parts.push('暖氣全開的平衡溫度只有約 ' + fmt(maxT) + '°C，低於目標 ' + fmt(params.target) +
        '°C——環境的轉移動態不在代理人的控制範圍內，再好的策略也拿不到高報酬。');
    } else {
      parts.push(pct >= 75
        ? '有 ' + pct + '% 的時間落在 ±ε 容忍帶內，控制良好，累積回報 G = ' + fmt(sim.G) + '。'
        : '只有 ' + pct + '% 的時間落在容忍帶內（含開頭升溫的試誤成本），累積回報 G = ' + fmt(sim.G) + '。');
      if (params.eps <= 1.0) parts.push('ε 調小後獎勵函數更嚴格，同樣的行為累積回報下降——獎勵設計直接影響我們對行為的評價。');
      else if (params.eps >= 2.8) parts.push('ε 很大時獎勵過於寬鬆，幾乎每一步都得 +1，難以分辨策略好壞。');
      if (params.dead <= 0.2) parts.push('死區很小：室溫緊貼目標，但暖氣切換了 ' + sim.switches + ' 次——精準控制的代價是頻繁動作。');
      else if (params.dead >= 1.4) parts.push('死區較大：只切換 ' + sim.switches + ' 次，但溫度擺盪變大，可能盪出容忍帶。');
    }
    ui.noteEl.textContent = parts.join(' ');
  }

  /* ---------- 執行一次模擬（含播放動畫） ---------- */
  function run() {
    cancelAnimationFrame(rafId);
    simulate();
    drawStatic();
    interpret();
    var cur = 0;
    (function tick() {
      cur = Math.min(cur + 2, N);
      drawFrame(cur);
      updateLoopPanel(cur - 1);
      if (cur < N) rafId = requestAnimationFrame(tick);
    })();
  }

  /* ---------- 版面 ---------- */
  function makeSlider(box, label, unit, min, max, stepV, key) {
    var wrap = el('div', 'min-width:140px;flex:1 1 150px;');
    var lab = el('label', 'display:block;margin-bottom:.15rem;');
    var val = el('b', 'color:var(--fg);');
    lab.appendChild(document.createTextNode(label + '　'));
    lab.appendChild(val);
    var input = document.createElement('input');
    input.type = 'range'; input.min = min; input.max = max; input.step = stepV;
    input.value = params[key];
    var show = function () { val.textContent = fmt(+input.value) + unit; };
    input.addEventListener('input', function () { params[key] = +input.value; show(); run(); });
    show();
    wrap.appendChild(lab); wrap.appendChild(input);
    box.appendChild(wrap);
  }

  function render(root) {
    // 控制面板
    var ctrl = el('div');
    ctrl.className = 'widget-panel';
    ctrl.style.marginBottom = '1rem';
    var row = el('div');
    row.className = 'widget-row';
    makeSlider(row, '目標溫度 T*', '°C', 16, 28, 0.5, 'target');
    makeSlider(row, '室外溫度', '°C', -5, 25, 1, 'outside');
    makeSlider(row, '獎勵容忍帶 ε', '°C', 0.3, 3, 0.1, 'eps');
    makeSlider(row, '策略死區', '°C', 0, 2, 0.1, 'dead');
    ctrl.appendChild(row);
    var row2 = el('div', 'display:flex;flex-wrap:wrap;gap:1rem;align-items:center;margin-top:.8rem;');
    var btn = el('button', '', '重新模擬 ▶');
    btn.addEventListener('click', run);
    var formula = el('span', 'font-size:.95rem;');
    if (window.katex) {
      window.katex.render(
        'r_t = \\begin{cases} +1 & |T_t - T^*| < \\varepsilon \\\\ -0.5\\,|T_t - T^*| & \\text{otherwise} \\end{cases}',
        formula, { throwOnError: false });
    } else {
      formula.textContent = 'r = +1 若 |T−T*|<ε，否則 −0.5·|T−T*|';
    }
    row2.appendChild(btn); row2.appendChild(formula);
    ctrl.appendChild(row2);
    root.appendChild(ctrl);

    // 圖表＋側欄
    var flex = el('div', 'display:flex;flex-wrap:wrap;gap:1rem;align-items:stretch;');
    var chartPanel = el('div', 'flex:2 1 340px;min-width:0;');
    chartPanel.className = 'widget-panel';
    var s = svg('svg', { viewBox: '0 0 720 330', width: '100%',
      role: 'img', 'aria-label': '室溫軌跡與每步獎勵圖' });
    s.style.display = 'block';
    ui.gStatic = svg('g', {}, s);
    ui.gDyn = svg('g', {}, s);
    chartPanel.appendChild(s);
    flex.appendChild(chartPanel);

    var side = el('div', 'flex:1 1 200px;display:flex;flex-direction:column;gap:.55rem;font-size:.85rem;');
    side.className = 'widget-panel';
    side.appendChild(el('div', 'font-weight:700;color:var(--accent);letter-spacing:.05em;', 'RL 迴圈即時狀態'));
    ui.stepEl = el('div', 'color:var(--fg-muted);');
    ui.sarEl = el('div', 'line-height:1.8;');
    side.appendChild(ui.stepEl);
    side.appendChild(ui.sarEl);
    var retBox = el('div', 'margin-top:.2rem;padding:.55rem .8rem;background:var(--code-bg);border:1px solid var(--border);border-radius:8px;');
    retBox.appendChild(el('div', 'font-size:.75rem;color:var(--fg-muted);', '累積回報 G = Σ r（γ = 1）'));
    ui.retEl = el('div', 'font-size:1.5rem;font-weight:700;color:var(--accent);', '0.0');
    retBox.appendChild(ui.retEl);
    side.appendChild(retBox);
    ui.noteEl = el('div', 'margin-top:auto;padding-top:.5rem;border-top:1px dashed var(--border);color:var(--fg-muted);line-height:1.8;');
    side.appendChild(ui.noteEl);
    flex.appendChild(side);
    root.appendChild(flex);

    run();
  }

  window.ChapterWidget = {
    title: '溫控器 RL 模擬器',
    intro: '書中 3.1.1 節以溫控器說明 RL 的形式化：代理人觀察狀態（室溫 s）、依策略選動作（開／關暖氣 a）、從環境獲得獎勵 r，並累積回報 G。拖動滑桿改變環境（室外溫度）、獎勵設計（容忍帶 ε）與策略（死區），觀察同一個迴圈如何產生不同的行為與報酬。',
    render: render
  };
})();
