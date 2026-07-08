(function () {
  'use strict';

  // 表 9（Olmo 3）：各評測在多次推論執行之間的標準差
  var BENCH = [
    { name: 'GPQA', sigma: 1.48, group: 'high' },
    { name: 'AlpacaEval 3', sigma: 1.24, group: 'high' },
    { name: 'IFEval', sigma: 0.88, group: 'high' },
    { name: 'ZebraLogic', sigma: 0.56, group: 'stable' },
    { name: 'Omega', sigma: 0.56, group: 'stable' },
    { name: 'AIME 24（Avg@32）', sigma: 0.54, group: 'stable' },
    { name: 'HumanEvalPlus', sigma: 0.46, group: 'stable' },
    { name: 'AgiEval', sigma: 0.43, group: 'stable' },
    { name: 'BigBenchHard', sigma: 0.39, group: 'stable' },
    { name: 'LiveCodeBench（Avg@10）', sigma: 0.29, group: 'vstable' },
    { name: 'MBPPPlus', sigma: 0.27, group: 'vstable' },
    { name: 'MATH', sigma: 0.25, group: 'vstable' },
    { name: 'MMLU', sigma: 0.22, group: 'vstable' },
    { name: 'PopQA', sigma: 0.16, group: 'vstable' }
  ];
  var GROUP = {
    high: { label: '高變異', color: 'var(--accent-2)' },
    stable: { label: '穩定', color: 'var(--accent)' },
    vstable: { label: '非常穩定', color: 'var(--link)' }
  };

  function h(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'text') node.textContent = attrs[k];
      else if (k === 'style') node.setAttribute('style', attrs[k]);
      else node[k] = attrs[k];
    });
    for (var i = 2; i < arguments.length; i++) node.appendChild(arguments[i]);
    return node;
  }
  function svg(tag, attrs) {
    var node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }
  // 標準常態分布 CDF（Abramowitz–Stegun 近似）
  function phi(x) {
    var t = 1 / (1 + 0.2316419 * Math.abs(x));
    var d = 0.3989423 * Math.exp(-x * x / 2);
    var p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x >= 0 ? 1 - p : p;
  }

  // 兩座鐘形曲線 + 重疊區示意圖
  function drawCurves(box, a, b, sigma) {
    box.textContent = '';
    var W = 600, H = 200, top = 26, base = H - 34;
    var lo = Math.min(a, b) - 3.5 * sigma, hi = Math.max(a, b) + 3.5 * sigma;
    var X = function (v) { return (v - lo) / (hi - lo) * (W - 20) + 10; };
    var Y = function (g) { return base - g * (base - top); }; // g: 0~1 正規化密度
    var pdf = function (v, mu) { return Math.exp(-((v - mu) * (v - mu)) / (2 * sigma * sigma)); };
    var el = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', role: 'img',
      'aria-label': '兩模型分數分布重疊示意圖' });
    var path = function (mu, both) {
      var d = 'M ' + X(lo) + ' ' + base;
      for (var i = 0; i <= 120; i++) {
        var v = lo + (hi - lo) * i / 120;
        var g = both ? Math.min(pdf(v, a), pdf(v, b)) : pdf(v, mu);
        d += ' L ' + X(v).toFixed(1) + ' ' + Y(g).toFixed(1);
      }
      return d + ' L ' + X(hi) + ' ' + base + ' Z';
    };
    el.appendChild(svg('path', { d: path(0, true), fill: 'var(--accent-soft)', stroke: 'none' }));
    el.appendChild(svg('path', { d: path(a), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2 }));
    el.appendChild(svg('path', { d: path(b), fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 2 }));
    el.appendChild(svg('line', { x1: 10, y1: base, x2: W - 10, y2: base, stroke: 'var(--border)' }));
    [[a, '模型 A', 'var(--accent)'], [b, '模型 B', 'var(--accent-2)']].forEach(function (m) {
      el.appendChild(svg('line', { x1: X(m[0]), y1: top, x2: X(m[0]), y2: base,
        stroke: m[2], 'stroke-dasharray': '4 3', 'stroke-width': 1 }));
      var left = m[0] <= (a + b) / 2 && a !== b;
      var t = svg('text', { x: X(m[0]) + (left ? -6 : 6), y: top - 8,
        'text-anchor': a === b ? 'middle' : (left ? 'end' : 'start'),
        'font-size': 12, fill: m[2] });
      t.textContent = m[1] + '｜' + m[0];
      el.appendChild(t);
    });
    [lo, (lo + hi) / 2, hi].forEach(function (v) {
      var t = svg('text', { x: X(v), y: H - 16, 'text-anchor': 'middle',
        'font-size': 11, fill: 'var(--fg-muted)' });
      t.textContent = v.toFixed(1);
      el.appendChild(t);
    });
    box.appendChild(el);
  }

  // σ 排序橫向條狀圖
  function buildBars(selectedIdx) {
    var wrap = h('div');
    var maxS = 1.48;
    BENCH.slice().sort(function (x, y) { return y.sigma - x.sigma; }).forEach(function (bm) {
      var sel = BENCH.indexOf(bm) === selectedIdx;
      var row = h('div', { style: 'display:grid;grid-template-columns:minmax(6.5em,10.5em) 1fr 2.6em;' +
        'gap:.5em;align-items:center;padding:.14em .3em;border-radius:4px;' +
        (sel ? 'background:var(--accent-soft);' : '') },
        h('span', { text: bm.name, title: bm.name, style: 'font-size:.74rem;color:var(--fg);overflow:hidden;' +
          'text-overflow:ellipsis;white-space:nowrap;' + (sel ? 'font-weight:700;' : '') }),
        h('div', { style: 'background:var(--panel);border:1px solid var(--border);border-radius:4px;height:.85em;' },
          h('div', { style: 'width:' + (bm.sigma / maxS * 100).toFixed(1) + '%;height:100%;border-radius:3px;' +
            'background:' + GROUP[bm.group].color + ';' })),
        h('span', { text: bm.sigma.toFixed(2), style: 'font-size:.74rem;color:var(--fg-muted);text-align:right;' }));
      wrap.appendChild(row);
    });
    var legend = h('div', { className: 'widget-row', style: 'margin-top:.6em;gap:1em;font-size:.76rem;color:var(--fg-muted);' });
    Object.keys(GROUP).forEach(function (g) {
      legend.appendChild(h('span', {},
        h('span', { style: 'display:inline-block;width:.75em;height:.75em;border-radius:2px;vertical-align:-.08em;' +
          'margin-right:.35em;background:' + GROUP[g].color + ';' }),
        document.createTextNode(GROUP[g].label)));
    });
    wrap.appendChild(legend);
    return wrap;
  }

  window.ChapterWidget = {
    title: '「分數差有意義嗎？」評測變異查核器',
    intro: '表 9 顯示：同一個模型重複評測，分數本來就會抖動。選一個評測、輸入兩個模型的分數，' +
      '看看這個差距是超出雜訊範圍的真實差異，還是單次執行的抽樣運氣。',
    render: function (root) {
      var panel = h('div', { className: 'widget-panel' });
      var select = h('select', { 'aria-label': '選擇評測基準' });
      BENCH.forEach(function (bm, i) {
        select.appendChild(h('option', { value: String(i),
          text: bm.name + '（σ = ' + bm.sigma.toFixed(2) + '，' + GROUP[bm.group].label + '）' }));
      });
      var inA = h('input', { type: 'number', step: '0.1', value: '76.5', style: 'width:6.5em;', 'aria-label': '模型 A 分數' });
      var inB = h('input', { type: 'number', step: '0.1', value: '77.7', style: 'width:6.5em;', 'aria-label': '模型 B 分數' });
      var btn = h('button', { type: 'button', text: '查核' });
      var lbl = function (t, c) { return h('label', { className: 'widget-row', style: 'gap:.4em;' }, h('span', { text: t, style: 'font-size:.85rem;color:var(--fg-muted);' }), c); };
      panel.appendChild(h('div', { className: 'widget-row' },
        lbl('評測', select), lbl('模型 A 分數', inA), lbl('模型 B 分數', inB), btn));

      var card = h('div', { style: 'margin-top:1rem;' });
      var svgBox = h('div', { style: 'margin-top:.8rem;' });
      panel.appendChild(card);
      panel.appendChild(svgBox);

      var barsPanel = h('div', { className: 'widget-panel', style: 'margin-top:1rem;' },
        h('div', { text: '表 9 全覽：各評測的標準差 σ（由高到低）', style: 'font-weight:700;margin-bottom:.6em;' }));
      var barsBox = h('div');
      barsPanel.appendChild(barsBox);
      barsPanel.appendChild(h('p', { text: '注意：LiveCodeBench 與 AIME 24 之所以落在穩定區，是因為表中數字已是 Avg@10／Avg@32' +
        '（重複執行取平均）後的結果——單次執行其實吵得多。降噪可以買，但成本很容易迅速膨脹（見 C.2）。',
        style: 'font-size:.8rem;color:var(--fg-muted);margin:.7em 0 0;' }));

      function check() {
        var bm = BENCH[Number(select.value)];
        var a = parseFloat(inA.value), b = parseFloat(inB.value);
        barsBox.textContent = '';
        barsBox.appendChild(buildBars(Number(select.value)));
        if (!isFinite(a) || !isFinite(b)) {
          card.innerHTML = '';
          card.appendChild(h('p', { text: '請先輸入兩個模型的分數。', style: 'color:var(--accent-2);margin:0;' }));
          svgBox.textContent = '';
          return;
        }
        var diff = Math.abs(a - b), k = diff / bm.sigma;
        var overlap = 2 * phi(-diff / (2 * bm.sigma)) * 100;
        var verdict, tone, advice;
        if (diff === 0) {
          verdict = '兩模型分數相同——沒有差距可以查核'; tone = 'var(--fg-muted)';
          advice = '分數打平不代表能力相同：在 ' + bm.name + ' 上單次執行本來就會抖動約 ±' + bm.sigma.toFixed(2) +
            ' 分。如 C.3 所建議，重要決策要多跑幾個 seed 取平均再比較。';
        } else if (k < 1) {
          verdict = '差 ' + diff.toFixed(1) + ' 分 = ' + k.toFixed(1) + 'σ——很可能只是雜訊'; tone = 'var(--accent-2)';
          advice = '這個差距完全落在 ' + bm.name + ' 單次執行的正常抖動範圍內，換個隨機種子重跑可能就反轉了。' +
            '如 C.3 所建議：重要決策要多跑幾個 seed 取平均，別憑單次分數選模型。';
        } else if (k < 2) {
          verdict = '差 ' + diff.toFixed(1) + ' 分 = ' + k.toFixed(1) + 'σ——灰色地帶，不足以下結論'; tone = 'var(--accent-2)';
          advice = '差距介於 1σ 與 2σ 之間：可能是真實進步，也可能是抽樣運氣。依 C.3 的建議，' +
            '重要決策要多跑幾個 seed 取平均，看差距是否穩定維持後再拍板。';
        } else {
          verdict = '差 ' + diff.toFixed(1) + ' 分 = ' + k.toFixed(1) + 'σ——大概率是真實差異'; tone = 'var(--accent)';
          advice = '差距遠超出 ' + bm.name + ' 的單次抖動（σ = ' + bm.sigma.toFixed(2) + '），大概率反映真實能力差異。' +
            '即便如此，C.3 仍提醒：重要決策要多跑幾個 seed 取平均，確認不是撿到正向離群值。';
        }
        card.innerHTML = '';
        card.appendChild(h('div', { style: 'border:1px solid ' + tone + ';border-left:5px solid ' + tone +
          ';border-radius:var(--radius, 8px);padding:.8em 1em;background:var(--panel);' },
          h('div', { text: verdict, style: 'font-weight:700;color:' + tone + ';' }),
          h('div', { text: bm.name + '：σ = ' + bm.sigma.toFixed(2) + '（' + GROUP[bm.group].label + '）｜兩模型分數分布重疊約 ' +
            overlap.toFixed(0) + '%', style: 'font-size:.82rem;color:var(--fg-muted);margin-top:.35em;' }),
          h('p', { text: advice, style: 'margin:.5em 0 0;font-size:.88rem;color:var(--fg);' })));
        drawCurves(svgBox, a, b, bm.sigma);
      }
      btn.addEventListener('click', check);
      select.addEventListener('change', check);
      root.appendChild(panel);
      root.appendChild(barsPanel);
      check();
    }
  };
})();
