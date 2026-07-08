/* 第 14 章：獎勵過度最佳化模擬器（Goodhart 定律） */
(function () {
  'use strict';

  // ===== 模型（Gao et al. 過度最佳化縮放律的簡化形式）=====
  const D_MAX = 3;                                   // 橫軸範圍：d = √KL
  const gold = (d, a, b) => d * (a - b * d);         // 真實獎勵（金標）：先升後降
  const proxy = (d, a) => d * a;                     // 代理獎勵：近似單調上升
  const peakD = (a, b) => a / (2 * b);               // 真實獎勵峰值（最佳停止點）
  // KL 懲罰下的收斂點：最大化 α·d − λ·d²（KL = d²）→ d_stop = α / 2λ
  const stopD = (a, lam) => Math.min(a / (2 * lam), D_MAX);
  const fmt = (v) => v.toFixed(2);

  // ===== SVG 佈局 =====
  const W = 700, H = 330, ML = 50, MR = 16, MT = 30, MB = 44;
  const PW = W - ML - MR, PH = H - MT - MB;
  const NS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function render(rootEl) {
    const state = { a: 1.6, b: 0.5, lam: 0.35, pos: 0, raf: 0 };
    state.pos = stopD(state.a, state.lam);

    // ---- 控制列 ----
    const panel = document.createElement('div');
    panel.className = 'widget-panel';
    const row = document.createElement('div');
    row.className = 'widget-row';
    panel.appendChild(row);
    function slider(labelHtml, min, max, step, val, key) {
      const box = document.createElement('label');
      box.style.cssText = 'flex:1 1 150px;min-width:140px;font-size:.85rem;color:var(--fg-muted)';
      const cap = document.createElement('div');
      cap.innerHTML = labelHtml + '：<strong style="color:var(--fg)">' + fmt(val) + '</strong>';
      const inp = document.createElement('input');
      inp.type = 'range'; inp.min = min; inp.max = max; inp.step = step; inp.value = val;
      inp.style.width = '100%';
      inp.addEventListener('input', () => {
        state[key] = parseFloat(inp.value);
        cap.innerHTML = labelHtml + '：<strong style="color:var(--fg)">' + fmt(state[key]) + '</strong>';
        stopAnim();
        state.pos = stopD(state.a, state.lam);
        drawAll();
      });
      box.appendChild(cap); box.appendChild(inp); row.appendChild(box);
    }
    slider('α（初始改善速率）', 0.8, 2.4, 0.05, state.a, 'a');
    slider('β（過度最佳化速率）', 0.3, 1.2, 0.05, state.b, 'b');
    slider('λ（KL 懲罰係數）', 0.15, 1.6, 0.05, state.lam, 'lam');
    const playBtn = document.createElement('button');
    playBtn.textContent = '▶ 播放訓練';
    playBtn.style.flex = '0 0 auto';
    row.appendChild(playBtn);
    rootEl.appendChild(panel);

    // ---- 圖例 ----
    const legend = document.createElement('div');
    legend.className = 'widget-row';
    legend.style.cssText = 'margin:.8rem 0 .3rem;font-size:.83rem;color:var(--fg-muted);gap:1.4rem';
    legend.innerHTML =
      '<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="var(--accent-2)" stroke-width="2" stroke-dasharray="5 4"/></svg> 代理獎勵（獎勵模型分數）</span>' +
      '<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="var(--accent)" stroke-width="2"/></svg> 真實獎勵（金標）</span>';
    rootEl.appendChild(legend);

    // ---- 主圖 ----
    const svg = el('svg', {
      viewBox: '0 0 ' + W + ' ' + H, width: '100%', role: 'img',
      'aria-label': '代理獎勵與真實獎勵隨最佳化距離 d 變化的曲線圖'
    });
    svg.style.cssText = 'display:block;background:var(--panel-2);border:1px solid var(--border);border-radius:10px';
    rootEl.appendChild(svg);
    const defs = el('defs', {}, svg);
    el('rect', { x: ML, y: MT, width: PW, height: PH },
      el('clipPath', { id: 'ch14-plot' }, defs));
    const progRect = el('rect', { x: ML, y: MT, width: 0, height: PH },
      el('clipPath', { id: 'ch14-prog' }, defs));

    const gGrid = el('g', {}, svg);
    const gPlot = el('g', { 'clip-path': 'url(#ch14-plot)' }, svg);
    // 尚未走過的曲線：淡色鋪底
    const pGoldDim = el('path', { fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, opacity: 0.22 }, gPlot);
    const pProxyDim = el('path', { fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 2, 'stroke-dasharray': '6 5', opacity: 0.22 }, gPlot);
    // 已走過的部分：全彩（progress 裁切）
    const gProg = el('g', { 'clip-path': 'url(#ch14-prog)' }, gPlot);
    const pGold = el('path', { fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.4 }, gProg);
    const pProxy = el('path', { fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 2.4, 'stroke-dasharray': '6 5' }, gProg);
    // 訓練停止豎直標線 + 兩曲線上的當前點
    const stopLine = el('line', { stroke: 'var(--fg-muted)', 'stroke-width': 1.5, 'stroke-dasharray': '3 4' }, gPlot);
    const dotGold = el('circle', { r: 5, fill: 'var(--accent)', stroke: 'var(--panel-2)', 'stroke-width': 2 }, gPlot);
    const dotProxy = el('circle', { r: 5, fill: 'var(--accent-2)', stroke: 'var(--panel-2)', 'stroke-width': 2 }, gPlot);
    const gTop = el('g', {}, svg); // 標籤層（不裁切）
    const peakDot = el('circle', { r: 5.5, fill: 'var(--accent)', stroke: 'var(--panel)', 'stroke-width': 2 }, gTop);
    const peakLbl = el('text', { 'font-size': 12, 'font-weight': 600, fill: 'var(--fg)', 'text-anchor': 'middle' }, gTop);
    peakLbl.textContent = '最佳停止';
    const stopLbl = el('text', { 'font-size': 11.5, fill: 'var(--fg-muted)' }, gTop);
    stopLbl.textContent = '訓練停在這裡（由 λ 決定）';

    // ---- 公式（KaTeX）與即時讀數 ----
    const eq = document.createElement('div');
    eq.style.cssText = 'margin:.9rem 0 .4rem;font-size:.95rem;overflow-x:auto';
    rootEl.appendChild(eq);
    const stats = document.createElement('div');
    stats.className = 'widget-row';
    stats.style.cssText = 'font-size:.85rem;color:var(--fg-muted);font-variant-numeric:tabular-nums;gap:1.4rem';
    rootEl.appendChild(stats);
    const readout = document.createElement('div');
    readout.className = 'widget-panel';
    readout.style.cssText = 'margin-top:.8rem;font-size:.9rem;line-height:1.7';
    rootEl.appendChild(readout);

    // ---- 座標換算 ----
    let yMin = 0, yMax = 1;
    const X = (d) => ML + (d / D_MAX) * PW;
    const Y = (r) => MT + (yMax - r) / (yMax - yMin) * PH;
    function curvePath(fn) {
      let s = '';
      for (let i = 0; i <= 120; i++) {
        const d = (i / 120) * D_MAX;
        s += (i ? 'L' : 'M') + X(d).toFixed(1) + ',' + Y(fn(d)).toFixed(1);
      }
      return s;
    }

    // ---- 靜態重繪（參數改變時）----
    function drawStatic() {
      const { a, b } = state;
      yMax = a * D_MAX * 1.06;
      yMin = -0.3 * yMax;
      gGrid.textContent = '';
      const step = yMax > 4.5 ? 2 : 1;
      for (let v = Math.ceil(yMin / step) * step; v <= yMax; v += step) {
        const y = Y(v), zero = Math.abs(v) < 1e-9;
        el('line', { x1: ML, x2: ML + PW, y1: y, y2: y, stroke: 'var(--border)', 'stroke-width': zero ? 1.6 : 1 }, gGrid);
        const t = el('text', { x: ML - 7, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--fg-muted)' }, gGrid);
        t.textContent = v;
      }
      for (let d = 0; d <= D_MAX; d += 1) {
        const t = el('text', { x: X(d), y: MT + PH + 17, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--fg-muted)' }, gGrid);
        t.textContent = d;
      }
      const xl = el('text', { x: ML + PW / 2, y: H - 8, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--fg-muted)' }, gGrid);
      xl.textContent = 'd = √KL(π‖π₀)（最佳化距離）';
      const yl = el('text', { x: 14, y: MT + PH / 2, 'font-size': 12, fill: 'var(--fg-muted)', 'text-anchor': 'middle', transform: 'rotate(-90 14 ' + (MT + PH / 2) + ')' }, gGrid);
      yl.textContent = '獎勵';

      const path1 = curvePath((d) => gold(d, a, b));
      const path2 = curvePath((d) => proxy(d, a));
      pGoldDim.setAttribute('d', path1); pGold.setAttribute('d', path1);
      pProxyDim.setAttribute('d', path2); pProxy.setAttribute('d', path2);

      const dp = Math.min(peakD(a, b), D_MAX);
      peakDot.setAttribute('cx', X(dp)); peakDot.setAttribute('cy', Y(gold(dp, a, b)));
      peakLbl.setAttribute('x', Math.min(X(dp), ML + PW - 34));
      peakLbl.setAttribute('y', Y(gold(dp, a, b)) - 12);

      const tex = 'R_{\\text{gold}} = d(\\alpha - \\beta d) = d(' + fmt(a) + ' - ' + fmt(state.b) +
        '\\,d),\\quad d=\\sqrt{D_{\\mathrm{KL}}(\\pi\\,\\|\\,\\pi_0)},\\quad d_{\\text{stop}}=\\tfrac{\\alpha}{2\\lambda}=' + fmt(stopD(a, state.lam));
      if (window.katex) window.katex.render(tex, eq, { throwOnError: false });
      else eq.textContent = 'R_gold = d(α − βd) = d(' + fmt(a) + ' − ' + fmt(b) + '·d)，d_stop = α/2λ = ' + fmt(stopD(a, state.lam));
    }

    // ---- 動態重繪（標線位置改變時）----
    function drawPos() {
      const { a, b, pos } = state;
      const x = X(pos);
      progRect.setAttribute('width', Math.max(0, x - ML));
      stopLine.setAttribute('x1', x); stopLine.setAttribute('x2', x);
      stopLine.setAttribute('y1', MT); stopLine.setAttribute('y2', MT + PH);
      const flip = x > ML + PW * 0.72;
      stopLbl.setAttribute('x', flip ? x - 7 : x + 7);
      stopLbl.setAttribute('y', MT + 14);
      stopLbl.setAttribute('text-anchor', flip ? 'end' : 'start');
      const rg = gold(pos, a, b), rp = proxy(pos, a);
      dotGold.setAttribute('cx', x); dotGold.setAttribute('cy', Y(rg));
      dotProxy.setAttribute('cx', x); dotProxy.setAttribute('cy', Y(rp));
      stats.innerHTML =
        '<span>d = <strong style="color:var(--fg)">' + fmt(pos) + '</strong></span>' +
        '<span>代理獎勵 = <strong style="color:var(--fg)">' + fmt(rp) + '</strong></span>' +
        '<span>真實獎勵 = <strong style="color:var(--fg)">' + fmt(rg) + '</strong></span>' +
        '<span>峰值 d* = α/2β = <strong style="color:var(--fg)">' + fmt(peakD(a, b)) + '</strong></span>';

      const dp = peakD(a, b), rPeak = gold(Math.min(dp, D_MAX), a, b);
      let msg;
      if (pos < dp * 0.92) {
        msg = '<strong style="color:var(--accent)">✅ 安全區</strong>：代理與真實獎勵仍同向上升——繼續最佳化仍有真實收益，距最佳停止點還有 Δd = ' + fmt(dp - pos) + '。';
      } else if (pos <= dp * 1.08) {
        msg = '<strong style="color:var(--accent)">⏸ 接近最佳停止點</strong>：真實獎勵正處峰值附近（R* ≈ ' + fmt(rPeak) + '）。此刻停止訓練（early stopping）能拿到最好的模型。';
      } else {
        const drop = rg <= 0 ? '甚至跌到負值' : '已自峰值下滑 ' + Math.round((1 - rg / rPeak) * 100) + '%';
        msg = '<strong style="color:var(--accent-2)">⚠️ 已進入過度最佳化</strong>：代理獎勵仍在上升（獎勵模型分數看起來一切健康），但真實獎勵' + drop +
          '——這就是 Goodhart 定律：「當一個衡量指標成為目標，它就不再是個好指標。」';
      }
      msg += '<br><span style="color:var(--fg-muted);font-size:.85em">提示：KL 懲罰讓訓練收斂在 d = α/2λ；試著把 λ 調到等於 β（= ' + fmt(b) + '），標線會恰好停在最佳停止點。</span>';
      readout.innerHTML = msg;
    }
    function drawAll() { drawStatic(); drawPos(); }

    // ---- 播放訓練動畫 ----
    function stopAnim() {
      if (state.raf) { cancelAnimationFrame(state.raf); state.raf = 0; }
      playBtn.textContent = '▶ 播放訓練';
    }
    playBtn.addEventListener('click', () => {
      if (state.raf) { stopAnim(); return; }
      const target = stopD(state.a, state.lam);
      const dur = 900 + 2400 * (target / D_MAX);
      const t0 = performance.now();
      playBtn.textContent = '⏹ 停止';
      const tick = (now) => {
        const t = Math.min((now - t0) / dur, 1);
        state.pos = target * t;
        drawPos();
        if (t < 1) state.raf = requestAnimationFrame(tick);
        else stopAnim();
      };
      state.pos = 0;
      state.raf = requestAnimationFrame(tick);
    });

    drawAll();
  }

  window.ChapterWidget = {
    title: '獎勵過度最佳化模擬器（Goodhart 定律）',
    intro: '對照本章圖 41／42：橫軸是最佳化距離 d = √KL(π‖π₀)。代理獎勵（獎勵模型分數）一路上升，真實獎勵（金標）卻先升後降。調整 α、β 觀察曲線形狀，調整 KL 懲罰係數 λ 看訓練會停在哪裡，再按「播放訓練」體驗「分數還在漲、模型卻變差」的瞬間。',
    render: render
  };
})();
