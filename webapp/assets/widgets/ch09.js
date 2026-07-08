/* 第 9 章：拒絕採樣模擬器（Rejection Sampling Simulator） */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  var MAX_N = 64, MAX_M = 10;

  function gauss() {
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  }
  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function div(cls, parent, text) {
    var e = document.createElement('div');
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function fmt(v, d) { return v.toFixed(d == null ? 2 : d); }
  function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : 0; }

  /* E[max of N 個標準常態樣本]，以蒙地卡羅（前綴最大值）估計一次，之後查表 */
  var EMAX = (function () {
    var trials = 4000, tbl = new Float64Array(MAX_N + 1);
    for (var t = 0; t < trials; t++) {
      var m = -Infinity;
      for (var n = 1; n <= MAX_N; n++) { var z = gauss(); if (z > m) m = z; tbl[n] += m; }
    }
    for (var i = 1; i <= MAX_N; i++) tbl[i] /= trials;
    tbl[1] = 0; // E[max of 1] = 0（理論值），去除取樣雜訊
    return tbl;
  })();

  var state = { N: 8, M: 6, sigma: 0.6, mode: 'per', mu: [], z: [], jit: [] };

  function resample() {
    state.mu = []; state.z = []; state.jit = [];
    for (var i = 0; i < MAX_M; i++) {
      state.mu.push(1 + gauss() * 0.3); // 每個提示的 μ_i 略有不同
      var row = [], jrow = [];
      for (var j = 0; j < MAX_N; j++) { row.push(gauss()); jrow.push(Math.random() - 0.5); }
      state.z.push(row); state.jit.push(jrow);
    }
  }

  /* 依當前 N、M、σ 取出獎勵矩陣與選取結果 */
  function compute() {
    var R = [], flat = [];
    for (var i = 0; i < state.M; i++) {
      var row = [];
      for (var j = 0; j < state.N; j++) {
        var r = state.mu[i] + state.z[i][j] * state.sigma;
        row.push(r); flat.push({ i: i, j: j, r: r });
      }
      R.push(row);
    }
    var chosen = {};
    if (state.mode === 'per') { // 每提示取最佳：S(R) = [argmax_j r_{i,j}]
      for (var p = 0; p < state.M; p++) {
        var best = 0;
        for (var q = 1; q < state.N; q++) if (R[p][q] > R[p][best]) best = q;
        chosen[p + ',' + best] = true;
      }
    } else { // 整體前 M 名：攤平後取 K = M 個最大值
      flat.slice().sort(function (a, b) { return b.r - a.r; }).slice(0, state.M)
        .forEach(function (c) { chosen[c.i + ',' + c.j] = true; });
    }
    return { R: R, flat: flat, chosen: chosen };
  }

  function axis(svg, x0, x1, y0, y1) { // 底部與左側軸線
    svgEl('line', { x1: x0, y1: y1, x2: x1, y2: y1, stroke: 'var(--border)' }, svg);
    svgEl('line', { x1: x0, y1: y0, x2: x0, y2: y1, stroke: 'var(--border)' }, svg);
  }
  function label(svg, x, y, text, anchor, size) {
    var t = svgEl('text', { x: x, y: y, fill: 'var(--fg-muted)', 'font-size': size || 11, 'text-anchor': anchor || 'middle' }, svg);
    t.textContent = text;
    return t;
  }

  function drawScatter(box, d) {
    box.innerHTML = '';
    var W = 720, H = 250, L = 46, R0 = 14, T = 14, B = 32;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%' }, box);
    var rs = d.flat.map(function (c) { return c.r; });
    var lo = Math.min.apply(null, rs), hi = Math.max.apply(null, rs);
    var pad = Math.max((hi - lo) * 0.1, 0.1); lo -= pad; hi += pad;
    var sx = function (i, j) { return L + (i + 0.5 + state.jit[i][j] * 0.72) * (W - L - R0) / state.M; };
    var sy = function (r) { return T + (hi - r) / (hi - lo) * (H - T - B); };
    axis(svg, L, W - R0, T, H - B);
    for (var g = 0; g <= 4; g++) {
      var rv = lo + (hi - lo) * g / 4, yy = sy(rv);
      svgEl('line', { x1: L, y1: yy, x2: W - R0, y2: yy, stroke: 'var(--border)', 'stroke-dasharray': '3 4', opacity: 0.5 }, svg);
      label(svg, L - 6, yy + 4, fmt(rv, 1), 'end', 10);
    }
    for (var i = 0; i < state.M; i++) label(svg, L + (i + 0.5) * (W - L - R0) / state.M, H - B + 16, '提示 ' + (i + 1), 'middle', 10);
    label(svg, L + (W - L - R0) / 2, H - 2, '提示編號 i（每欄 N 個補全的 reward）', 'middle', 11);
    d.flat.forEach(function (c) {
      var x = sx(c.i, c.j), y = sy(c.r);
      if (d.chosen[c.i + ',' + c.j]) {
        svgEl('circle', { cx: x, cy: y, r: 7, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.5 }, svg);
        svgEl('circle', { cx: x, cy: y, r: 3.2, fill: 'var(--accent)' }, svg);
      } else {
        svgEl('circle', { cx: x, cy: y, r: 3, fill: 'var(--fg-muted)', 'fill-opacity': 0.4 }, svg);
      }
    });
  }

  function drawHist(box, d) {
    box.innerHTML = '';
    var W = 720, H = 190, L = 46, R0 = 14, T = 26, B = 30, BINS = 26;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%' }, box);
    var all = d.flat.map(function (c) { return c.r; });
    var sel = d.flat.filter(function (c) { return d.chosen[c.i + ',' + c.j]; }).map(function (c) { return c.r; });
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all) + 1e-9;
    var count = function (arr) {
      var h = new Array(BINS).fill(0);
      arr.forEach(function (r) { h[Math.min(BINS - 1, Math.floor((r - lo) / (hi - lo) * BINS))] += 1 / arr.length; });
      return h;
    };
    var hAll = count(all), hSel = count(sel);
    var top = Math.max(Math.max.apply(null, hAll), Math.max.apply(null, hSel));
    var bw = (W - L - R0) / BINS;
    var sy = function (f) { return T + (1 - f / top) * (H - T - B); };
    axis(svg, L, W - R0, T, H - B);
    var bars = function (h, color, op) {
      for (var b = 0; b < BINS; b++) if (h[b] > 0)
        svgEl('rect', { x: L + b * bw + 0.5, y: sy(h[b]), width: bw - 1, height: H - B - sy(h[b]), fill: color, 'fill-opacity': op }, svg);
    };
    bars(hAll, 'var(--fg-muted)', 0.35);
    bars(hSel, 'var(--accent)', 0.55);
    var mAll = mean(all), mSel = mean(sel);
    var mx = function (r) { return L + (r - lo) / (hi - lo) * (W - L - R0); };
    svgEl('line', { x1: mx(mAll), y1: T, x2: mx(mAll), y2: H - B, stroke: 'var(--fg-muted)', 'stroke-dasharray': '4 3' }, svg);
    svgEl('line', { x1: mx(mSel), y1: T, x2: mx(mSel), y2: H - B, stroke: 'var(--accent)', 'stroke-dasharray': '4 3' }, svg);
    label(svg, L, T - 10, '全部樣本（灰）均值 ' + fmt(mAll) + ' ｜ 被選中（強調色）均值 ' + fmt(mSel) + ' ｜ 位移 +' + fmt(mSel - mAll), 'start', 11)
      .setAttribute('fill', 'var(--fg)');
    label(svg, L + (W - L - R0) / 2, H - 4, 'reward 分布（各自的相對頻率）', 'middle', 11);
    for (var g = 0; g <= 4; g++) label(svg, mx(lo + (hi - lo) * g / 4), H - B + 14, fmt(lo + (hi - lo) * g / 4, 1), 'middle', 10);
    return mSel - mAll;
  }

  function drawCurve(box) {
    box.innerHTML = '';
    var W = 720, H = 200, L = 46, R0 = 14, T = 24, B = 32;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%' }, box);
    var top = EMAX[MAX_N] * state.sigma * 1.12 + 1e-9;
    var sx = function (n) { return L + (n - 1) / (MAX_N - 1) * (W - L - R0); };
    var sy = function (v) { return T + (1 - v / top) * (H - T - B); };
    axis(svg, L, W - R0, T, H - B);
    [1, 8, 16, 32, 48, 64].forEach(function (n) { label(svg, sx(n), H - B + 14, String(n), 'middle', 10); });
    for (var g = 0; g <= 3; g++) {
      label(svg, L - 6, sy(top * g / 3) + 4, fmt(top * g / 3, 1), 'end', 10);
      svgEl('line', { x1: L, y1: sy(top * g / 3), x2: W - R0, y2: sy(top * g / 3), stroke: 'var(--border)', 'stroke-dasharray': '3 4', opacity: 0.5 }, svg);
    }
    var pts = [];
    for (var n = 1; n <= MAX_N; n++) pts.push(fmt(sx(n), 1) + ',' + fmt(sy(EMAX[n] * state.sigma), 1));
    svgEl('polyline', { points: pts.join(' '), fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 2 }, svg);
    var cx = sx(state.N), cy = sy(EMAX[state.N] * state.sigma);
    svgEl('line', { x1: cx, y1: cy, x2: cx, y2: H - B, stroke: 'var(--accent)', 'stroke-dasharray': '3 3' }, svg);
    svgEl('circle', { cx: cx, cy: cy, r: 5, fill: 'var(--accent)' }, svg);
    label(svg, Math.min(cx + 8, W - 90), cy - 8, 'N=' + state.N + '（+' + fmt(EMAX[state.N] * state.sigma) + '）', cx > W - 170 ? 'end' : 'start', 11)
      .setAttribute('fill', 'var(--fg)');
    label(svg, L, T - 10, 'Best-of-N 期望提升 E[max]−μ ≈ σ·E[max of N 標準常態]', 'start', 11);
    label(svg, L + (W - L - R0) / 2, H - 4, '每提示樣本數 N', 'middle', 11);
  }

  window.ChapterWidget = {
    title: '拒絕採樣模擬器',
    intro: '模擬「生成 N 個補全、用獎勵模型評分、只留最好的」流程：比較 9.1.2 節的兩種選取函數（每提示取最佳 vs 整體前 M 名），並觀察 Best-of-N 的報酬遞減。',
    render: function (root) {
      root.innerHTML = '';
      var panel = div('widget-panel', root);
      var row = div('widget-row', panel);
      function slider(text, min, max, step, key, fmtV) {
        var wrap = document.createElement('label');
        wrap.style.cssText = 'display:flex;flex-direction:column;gap:.25rem;font-size:.85rem;min-width:150px;flex:1';
        var cap = document.createElement('span');
        var inp = document.createElement('input');
        inp.type = 'range'; inp.min = min; inp.max = max; inp.step = step; inp.value = state[key];
        var upd = function () { cap.textContent = text + '：' + fmtV(state[key]); };
        inp.addEventListener('input', function () { state[key] = +inp.value; upd(); redraw(); });
        upd(); wrap.appendChild(cap); wrap.appendChild(inp); row.appendChild(wrap);
        return upd;
      }
      slider('每提示樣本數 N', 1, MAX_N, 1, 'N', function (v) { return v; });
      slider('提示數 M', 4, MAX_M, 1, 'M', function (v) { return v; });
      slider('獎勵標準差 σ', 0.2, 1.5, 0.05, 'sigma', function (v) { return fmt(v); });

      var row2 = div('widget-row', panel);
      row2.style.marginTop = '.6rem';
      var sel = document.createElement('select');
      [['per', '每提示取最佳（Top Per Prompt）'], ['overall', '整體前 M 名（Top Overall Pairs, K=M）']]
        .forEach(function (o) { var op = document.createElement('option'); op.value = o[0]; op.textContent = o[1]; sel.appendChild(op); });
      sel.addEventListener('change', function () { state.mode = sel.value; redraw(); });
      row2.appendChild(sel);
      var btn = document.createElement('button');
      btn.textContent = '重新採樣';
      btn.addEventListener('click', function () { resample(); redraw(); });
      row2.appendChild(btn);
      var formula = document.createElement('span');
      formula.style.cssText = 'font-size:.95rem;overflow-x:auto;max-width:100%';
      row2.appendChild(formula);

      var scatterBox = div('widget-panel', root); scatterBox.style.marginTop = '.8rem';
      var histBox = div('widget-panel', root); histBox.style.marginTop = '.8rem';
      var curveBox = div('widget-panel', root); curveBox.style.marginTop = '.8rem';
      var note = div('', root);
      note.style.cssText = 'margin-top:.8rem;padding:.7rem 1rem;border-left:3px solid var(--accent);background:var(--accent-soft);border-radius:0 8px 8px 0;font-size:.9rem;color:var(--fg);line-height:1.7';

      function redraw() {
        var d = compute();
        drawScatter(scatterBox, d);
        var shift = drawHist(histBox, d);
        drawCurve(curveBox);
        if (window.katex) {
          window.katex.render(state.mode === 'per'
            ? 'S(R)=\\big[\\arg\\max_j r_{i,j}\\big]_{i=1}^{M}'
            : 'S_K(R_{flat})=\\text{argsort}(R_{flat})[-M{:}]', formula, { throwOnError: false });
        }
        var n2 = Math.min(state.N * 2, MAX_N);
        var gainDouble = (EMAX[n2] - EMAX[state.N]) * state.sigma;
        var gainFirst = (EMAX[2] - EMAX[1]) * state.sigma;
        var covered = {};
        for (var k in d.chosen) covered[k.split(',')[0]] = true;
        var nCov = Object.keys(covered).length;
        var txt = '目前 N = ' + state.N + '：被選中樣本的平均 reward 比全體高 ' + fmt(shift) + '。';
        if (n2 > state.N) {
          txt += '理論上把 N 從 ' + state.N + ' 加倍到 ' + n2 + '，期望最大值只再提升約 ' + fmt(gainDouble) +
            '（對照：N 從 1→2 就提升了 ' + fmt(gainFirst) + '）——報酬遞減，這正是實務上 N 取 10～30 即已足夠的原因。';
        } else {
          txt += 'N 已達上限 64；曲線後段幾乎平坦——再加倍樣本數也換不到多少期望提升，報酬遞減。';
        }
        if (state.mode === 'overall') {
          txt += ' 整體前 M 名模式下，' + state.M + ' 個提示只有 ' + nCov + ' 個被覆蓋' +
            (nCov < state.M ? '：高 μ 的提示壟斷名額，低 μ 的提示可能一個補全都選不上，訓練資料會偏向「容易拿高獎勵」的提示。' : '，本次恰好每個提示都有補全入選。');
        } else {
          txt += ' 每提示取最佳模式保證每個提示各留一個補全，資料覆蓋均勻。';
        }
        note.textContent = txt;
      }
      resample();
      redraw();
    }
  };
})();
