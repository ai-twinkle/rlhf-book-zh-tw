/* 第 12 章互動元件：知識蒸餾軟標籤實驗室 */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  var TOKENS = ['玉山', '雪山', '秀姑巒山', '合歡山', '阿里山', '喜馬拉雅山', '富士山', '台北101'];
  var LOGITS = [6.0, 3.6, 2.8, 2.0, 1.8, 0.2, -0.2, -1.8]; // 教師 logits（寫死）
  var CORRECT = 0; // 玉山
  var STEPS = 60, LR = 0.9;

  var state = { tau: 1.0, mode: 'soft', studentLogits: null, losses: [], timer: null, trained: false };
  var ui = {};

  function el(tag, style, html) {
    var n = document.createElement(tag);
    if (style) n.style.cssText = style;
    if (html !== undefined) n.innerHTML = html;
    return n;
  }
  function svgEl(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }
  function tex(node, src) {
    if (window.katex) window.katex.render(src, node, { throwOnError: false, displayMode: true });
    else node.textContent = src;
  }
  function softmax(z, tau) {
    var m = -Infinity, i;
    for (i = 0; i < z.length; i++) m = Math.max(m, z[i] / tau);
    var s = 0, out = [];
    for (i = 0; i < z.length; i++) { out.push(Math.exp(z[i] / tau - m)); s += out[i]; }
    for (i = 0; i < z.length; i++) out[i] /= s;
    return out;
  }
  function targetDist() {
    if (state.mode === 'hard') {
      return TOKENS.map(function (_, i) { return i === CORRECT ? 1 : 0; });
    }
    return softmax(LOGITS, state.tau);
  }
  function studentDist() { return softmax(state.studentLogits, 1); }
  function klQP(q, p) { // KL(teacher‖student)，即前向 KL
    var s = 0;
    for (var i = 0; i < q.length; i++) if (q[i] > 0) s += q[i] * Math.log(q[i] / p[i]);
    return s;
  }
  function resetStudent() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.studentLogits = TOKENS.map(function () { return 0; }); // 均勻分布
    state.losses = [];
    state.trained = false;
  }

  /* ---------- 條狀圖 ---------- */
  function drawBars() {
    var svg = ui.barSvg;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var q = targetDist(), p = studentDist();
    var W = 640, x0 = 34, plotW = W - x0 - 6, top = 14, base = 188, slot = plotW / TOKENS.length;
    var yMax = Math.min(1, Math.max(0.05, Math.max.apply(null, q.concat(p))) * 1.15);
    var i, g;
    for (i = 1; i <= 3; i++) { // 水平格線
      var gv = yMax * i / 4, gy = base - (base - top) * (i / 4);
      svg.appendChild(svgEl('line', { x1: x0, y1: gy, x2: W - 6, y2: gy, stroke: 'var(--border)', 'stroke-dasharray': '3 3', 'stroke-width': 1 }));
      g = svgEl('text', { x: x0 - 4, y: gy + 3.5, 'text-anchor': 'end', 'font-size': 10, fill: 'var(--fg-muted)' });
      g.textContent = (gv * 100).toFixed(0) + '%';
      svg.appendChild(g);
    }
    svg.appendChild(svgEl('line', { x1: x0, y1: base, x2: W - 6, y2: base, stroke: 'var(--border)', 'stroke-width': 1 }));
    for (i = 0; i < TOKENS.length; i++) {
      var cx = x0 + slot * (i + 0.5);
      var hq = (base - top) * Math.min(1, q[i] / yMax);
      var hp = (base - top) * Math.min(1, p[i] / yMax);
      svg.appendChild(svgEl('rect', { x: cx - 22, y: base - hq, width: 44, height: Math.max(hq, 0.5), rx: 3, fill: 'var(--accent)', opacity: 0.9 })); // 訓練目標
      svg.appendChild(svgEl('rect', { x: cx - 12, y: base - hp, width: 24, height: Math.max(hp, 0.5), rx: 3, fill: 'var(--accent-2)', opacity: 0.55, stroke: 'var(--accent-2)', 'stroke-width': 1 })); // 學生
      if (q[i] >= 0.005) {
        g = svgEl('text', { x: cx, y: base - hq - 4, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--fg-muted)' });
        g.textContent = (q[i] * 100) < 1 ? (q[i] * 100).toFixed(1) + '%' : (q[i] * 100).toFixed(0) + '%';
        svg.appendChild(g);
      }
      g = svgEl('text', { x: cx, y: base + 15, 'text-anchor': 'middle', 'font-size': 11, fill: i === CORRECT ? 'var(--fg)' : 'var(--fg-muted)', 'font-weight': i === CORRECT ? 700 : 400 });
      g.textContent = TOKENS[i];
      svg.appendChild(g);
    }
  }

  /* ---------- 損失曲線 ---------- */
  function drawLoss() {
    var svg = ui.lossSvg;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var W = 320, H = 120, x0 = 36, y0 = 12, x1 = W - 8, y1 = H - 22;
    svg.appendChild(svgEl('line', { x1: x0, y1: y1, x2: x1, y2: y1, stroke: 'var(--border)', 'stroke-width': 1 }));
    svg.appendChild(svgEl('line', { x1: x0, y1: y0, x2: x0, y2: y1, stroke: 'var(--border)', 'stroke-width': 1 }));
    var lx = svgEl('text', { x: (x0 + x1) / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--fg-muted)' });
    lx.textContent = '梯度下降步數（0–' + STEPS + '）';
    svg.appendChild(lx);
    var ly = svgEl('text', { x: 10, y: (y0 + y1) / 2, 'font-size': 10, fill: 'var(--fg-muted)', transform: 'rotate(-90 10 ' + ((y0 + y1) / 2) + ')', 'text-anchor': 'middle' });
    ly.textContent = 'KD 損失';
    svg.appendChild(ly);
    if (state.losses.length < 2) {
      var hint = svgEl('text', { x: (x0 + x1) / 2, y: (y0 + y1) / 2, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--fg-muted)' });
      hint.textContent = '按「訓練學生」開始最小化 KL(q‖p)';
      svg.appendChild(hint);
      return;
    }
    var maxL = Math.max.apply(null, state.losses) || 1;
    var d = '';
    for (var i = 0; i < state.losses.length; i++) {
      var px = x0 + (x1 - x0) * i / STEPS;
      var py = y1 - (y1 - y0) * (state.losses[i] / maxL);
      d += (i ? 'L' : 'M') + px.toFixed(1) + ' ' + py.toFixed(1);
    }
    svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 2 }));
  }

  /* ---------- 文案更新 ---------- */
  function updateText() {
    ui.tauVal.textContent = 'τ = ' + state.tau.toFixed(1);
    var t;
    if (state.mode === 'hard') {
      t = '<strong>硬標籤：</strong>one-hot 只告訴學生「正解是玉山」，其他七個選項一律得 0——' +
        '教師知道「雪山遠比台北101 合理」這件事完全沒有傳遞出去，溫度 τ 此時也不起作用。';
    } else if (state.tau <= 0.4) {
      t = '<strong>軟標籤（τ 低）：</strong>分布趨近 one-hot，幾乎退化成硬標籤，次要選項的資訊被壓掉了。';
    } else if (state.tau >= 2) {
      t = '<strong>軟標籤（τ 高）：</strong>「暗知識」顯現——雪山、秀姑巒山等次要選項的相對機率被放大，' +
        '學生能學到教師對相似選項的排序，而干擾項台北101 依然墊底。';
    } else {
      t = '<strong>軟標籤：</strong>教師分布傳遞了對相似選項的相對判斷（雪山＞合歡山＞富士山＞台北101）——' +
        '這正是蒸餾比單純模仿正解更有效的原因。試著調高 τ 讓暗知識更明顯。';
    }
    ui.interp.innerHTML = t;
    var q = targetDist(), kl = klQP(q, studentDist());
    ui.klVal.innerHTML = '目前 KL(q‖p) = <strong>' + kl.toFixed(4) + '</strong>' +
      (state.mode === 'hard' ? '（one-hot 時即 −log p(玉山)）' : '') +
      (state.trained && !state.timer ? '　學生已收斂到訓練目標。' : '');
  }
  function redraw() { drawBars(); drawLoss(); updateText(); }

  /* ---------- 訓練 ---------- */
  function train() {
    resetStudent();
    ui.trainBtn.disabled = true;
    var q = targetDist(), step = 0;
    state.losses.push(klQP(q, studentDist()));
    state.timer = setInterval(function () {
      var p = studentDist();
      for (var i = 0; i < TOKENS.length; i++) state.studentLogits[i] -= LR * (p[i] - q[i]); // ∂H(q,p)/∂w = p − q
      state.losses.push(klQP(q, studentDist()));
      step++;
      if (step >= STEPS) { clearInterval(state.timer); state.timer = null; state.trained = true; ui.trainBtn.disabled = false; }
      redraw();
    }, 45);
  }
  function setMode(m) {
    state.mode = m;
    resetStudent();
    ui.trainBtn.disabled = false;
    ui.softBtn.style.cssText = modeBtnCss(m === 'soft');
    ui.hardBtn.style.cssText = modeBtnCss(m === 'hard');
    redraw();
  }
  function modeBtnCss(active) {
    return active ? 'border-color:var(--accent);background:var(--accent-soft);font-weight:600;' : '';
  }
  function cardCol(title, color, rows) {
    var col = el('div', 'border:1px solid var(--border);border-top:3px solid ' + color + ';border-radius:8px;background:var(--panel-2);padding:.6rem .75rem;');
    col.appendChild(el('div', 'font-weight:600;margin-bottom:.4rem;', title));
    rows.forEach(function (r) {
      col.appendChild(el('div', 'font-size:.78em;color:var(--fg-muted);margin-top:.35rem;', r[0]));
      col.appendChild(el('div', 'font-size:.88em;', r[1]));
    });
    return col;
  }

  /* ---------- 版面 ---------- */
  function render(rootEl) {
    rootEl.innerHTML = '';
    resetStudent();
    state.tau = 1.0; state.mode = 'soft';

    var lab = el('div'); lab.className = 'widget-panel';
    lab.appendChild(el('div', 'font-size:.85em;color:var(--fg-muted);margin-bottom:.2rem;', '玩具情境：教師模型的下一個 token 分布'));
    lab.appendChild(el('div', 'font-weight:600;margin-bottom:.6rem;', '「台灣最高的山是＿」（8 個候選 token，教師 logits 固定）'));

    var row = el('div'); row.className = 'widget-row';
    row.style.cssText = 'display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:.5rem;';
    row.appendChild(el('span', 'font-size:.9em;', '溫度'));
    var slider = document.createElement('input');
    slider.type = 'range'; slider.min = '0.1'; slider.max = '5'; slider.step = '0.1'; slider.value = '1';
    slider.style.flex = '1 1 120px';
    slider.setAttribute('aria-label', '溫度 τ');
    slider.addEventListener('input', function () {
      state.tau = parseFloat(slider.value);
      if (state.trained || state.timer) resetStudent();
      ui.trainBtn.disabled = false;
      redraw();
    });
    row.appendChild(slider);
    ui.tauVal = el('span', 'font-variant-numeric:tabular-nums;min-width:4.5em;font-size:.9em;');
    row.appendChild(ui.tauVal);
    ui.softBtn = document.createElement('button'); ui.softBtn.textContent = '軟標籤（教師分布）';
    ui.hardBtn = document.createElement('button'); ui.hardBtn.textContent = '硬標籤（one-hot）';
    ui.softBtn.addEventListener('click', function () { setMode('soft'); });
    ui.hardBtn.addEventListener('click', function () { setMode('hard'); });
    row.appendChild(ui.softBtn); row.appendChild(ui.hardBtn);
    lab.appendChild(row);

    ui.barSvg = svgEl('svg', { viewBox: '0 0 640 208', width: '100%', role: 'img', 'aria-label': '訓練目標與學生分布條狀圖' });
    lab.appendChild(ui.barSvg);
    var legend = el('div', 'display:flex;gap:1rem;flex-wrap:wrap;font-size:.82em;color:var(--fg-muted);margin:.25rem 0 .5rem;');
    legend.appendChild(el('span', '', '<span style="display:inline-block;width:.85em;height:.85em;border-radius:3px;background:var(--accent);vertical-align:-.1em;"></span> 訓練目標 q（軟標籤時為 softmax(z/τ)）'));
    legend.appendChild(el('span', '', '<span style="display:inline-block;width:.85em;height:.85em;border-radius:3px;background:var(--accent-2);opacity:.55;vertical-align:-.1em;"></span> 學生分布 p（初始均勻）'));
    legend.appendChild(el('span', '', 'y 軸自動縮放'));
    lab.appendChild(legend);
    ui.interp = el('div', 'font-size:.9em;border-left:3px solid var(--accent);background:var(--panel-2);border-radius:6px;padding:.5rem .75rem;margin-bottom:.6rem;');
    lab.appendChild(ui.interp);

    var trainRow = el('div', 'display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;');
    ui.trainBtn = document.createElement('button'); ui.trainBtn.textContent = '訓練學生（60 步梯度下降）';
    ui.trainBtn.addEventListener('click', train);
    var resetBtn = document.createElement('button'); resetBtn.textContent = '重設學生';
    resetBtn.addEventListener('click', function () { resetStudent(); ui.trainBtn.disabled = false; redraw(); });
    trainRow.appendChild(ui.trainBtn); trainRow.appendChild(resetBtn);
    ui.klVal = el('span', 'font-size:.88em;color:var(--fg-muted);');
    trainRow.appendChild(ui.klVal);
    lab.appendChild(trainRow);
    ui.lossSvg = svgEl('svg', { viewBox: '0 0 320 120', width: '100%', style: 'max-width:420px;margin-top:.5rem;', role: 'img', 'aria-label': 'KD 損失下降曲線' });
    lab.appendChild(ui.lossSvg);
    rootEl.appendChild(lab);

    /* 公式面板（對齊式 127–130） */
    var fPanel = el('div'); fPanel.className = 'widget-panel';
    fPanel.appendChild(el('div', 'font-weight:600;margin-bottom:.3rem;', '這個實驗最小化的目標（式 127／129–130 的單一位置版本）'));
    var eq1 = el('div', 'overflow-x:auto;');
    tex(eq1, '\\mathcal{L}_{\\text{KD}} = H(q,p) = -\\sum_{k=1}^{|\\mathcal{V}|} q(u_j = k \\mid s, u_{<j})\\,\\log p(u_j = k \\mid s, u_{<j}),\\qquad q = \\mathrm{softmax}(z/\\tau)');
    fPanel.appendChild(eq1);
    var eq2 = el('div', 'overflow-x:auto;');
    tex(eq2, 'H(q,p) = H(q) + D_{\\text{KL}}(q \\,\\|\\, p)');
    fPanel.appendChild(eq2);
    fPanel.appendChild(el('div', 'font-size:.88em;color:var(--fg-muted);',
      '教師分布 q 固定時，H(q) 是常數，最小化交叉熵等價於最小化前向 KL：D<sub>KL</sub>(q‖p)——' +
      '這正是離線 KD 與 SFT 式訓練所用的 KL 方向；動畫中的梯度就是 ∂H(q,p)/∂w = p − q。' +
      '若換成硬標籤（q 為 one-hot），損失退化為一般的交叉熵 −log p(玉山)。'));
    rootEl.appendChild(fPanel);

    /* 離線 vs on-policy 概念卡（12.3.2） */
    var card = el('div'); card.className = 'widget-panel';
    card.appendChild(el('div', 'font-weight:600;margin-bottom:.5rem;', '概念卡：離線蒸餾 vs on-policy 蒸餾（OPD）'));
    var grid = el('div', 'display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.6rem;');
    grid.appendChild(cardCol('離線 KD（WORD-KD／SEQ-KD）', 'var(--accent)', [
      ['誰生成序列', '教師：訓練前就先產生好序列 u ∼ π<sub>T</sub>，屬前向 KL D<sub>KL</sub>(π<sub>T</sub>‖π<sub>θ</sub>)。'],
      ['優點', '資料可事先產生並重複使用，實作簡單、訓練穩定，銜接 SFT 流程。'],
      ['缺點', '訓練與推論的前綴分布不匹配；前向 KL 可能讓學生高估教師的低機率區域，誤差沿長序列累積可達 O(εL²)。']
    ]));
    grid.appendChild(cardCol('on-policy 蒸餾（OPD）', 'var(--accent-2)', [
      ['誰生成序列', '學生：訓練中即時取樣 a ∼ π<sub>θ</sub>，教師在學生造訪的狀態上給分布監督，屬反向 KL D<sub>KL</sub>(π<sub>θ</sub>‖π<sub>T</sub>)。'],
      ['優點', '學生面對自己的錯誤、學會恢復行為，誤差累積降至 O(εL)；逐 token 的教師訊號稠密。'],
      ['缺點', '需要線上取樣與 RL 式基礎設施，訓練成本較高、實作較複雜。']
    ]));
    card.appendChild(grid);
    card.appendChild(el('div', 'font-size:.88em;margin-top:.6rem;border-left:3px solid var(--accent-2);background:var(--panel-2);border-radius:6px;padding:.5rem .75rem;',
      '<strong>曝露偏差（exposure bias）一句話：</strong>學生訓練時只看過教師寫的前綴，推論時卻要面對自己（可能出錯）生成的前綴——訓練與測試的狀態造訪分布不一致。'));
    rootEl.appendChild(card);

    setMode('soft');
  }

  window.ChapterWidget = {
    title: '知識蒸餾軟標籤實驗室',
    intro: '以「台灣最高的山是＿」的下一個 token 分布為玩具情境：調整溫度 τ 觀察教師軟標籤中的「暗知識」，' +
      '對照硬標籤（one-hot）與軟標籤兩種訓練目標，再按下「訓練學生」，看學生分布如何以梯度下降最小化前向 KL 收斂到教師分布。',
    render: render
  };
})();
