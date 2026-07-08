/* 第 10 章「偏好的本質」互動元件：偏好彙總悖論實驗（Condorcet 循環） */
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';
  var PERMS = [['A', 'B', 'C'], ['A', 'C', 'B'], ['B', 'A', 'C'], ['B', 'C', 'A'], ['C', 'A', 'B'], ['C', 'B', 'A']];
  var PAIRS = [['A', 'B'], ['B', 'C'], ['A', 'C']];
  var POS = { A: { x: 160, y: 48 }, B: { x: 54, y: 208 }, C: { x: 266, y: 208 } };
  var CENTER = { x: 160, y: 158 };
  var R = 24;

  function el(tag, attrs, children) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'text') n.textContent = attrs[k];
      else if (k === 'class') n.className = attrs[k];
      else if (k === 'style') n.style.cssText = attrs[k];
      else n.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { n.appendChild(c); });
    return n;
  }
  function svgEl(tag, attrs) {
    var n = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'text') n.textContent = attrs[k];
      else n.setAttribute(k, attrs[k]);
    });
    return n;
  }
  function key(p) { return p.join(''); }
  function label(p) { return p.join(' > '); }

  /* 成對多數決：對每一組 (x, y) 統計偏好 x 的票數，回傳勝者與票數 */
  function tally(rankings) {
    return PAIRS.map(function (pair) {
      var x = pair[0], y = pair[1], vx = 0;
      rankings.forEach(function (r) { if (r.indexOf(x) < r.indexOf(y)) vx++; });
      var vy = rankings.length - vx;
      return vx > vy ? { a: x, b: y, winner: x, loser: y, w: vx, l: vy }
                     : { a: x, b: y, winner: y, loser: x, w: vy, l: vx };
    });
  }
  /* 三人完整排序不會平手；當 A、B、C 各恰好勝一場時即為循環 */
  function isCycle(results) {
    var wins = { A: 0, B: 0, C: 0 };
    results.forEach(function (r) { wins[r.winner]++; });
    return wins.A === 1 && wins.B === 1 && wins.C === 1;
  }
  function randPerm() { return PERMS[Math.floor(Math.random() * PERMS.length)]; }
  /* 產生會（或不會）循環、且與目前不同的標註組合 */
  function randomProfile(wantCycle, current) {
    var cur = current.map(key).join('|');
    for (var t = 0; t < 400; t++) {
      var prof;
      if (wantCycle) {
        var b = randPerm(); /* 三個「輪轉」排序必然構成循環 */
        prof = [b, [b[1], b[2], b[0]], [b[2], b[0], b[1]]];
        prof.sort(function () { return Math.random() - 0.5; });
      } else {
        prof = [randPerm(), randPerm(), randPerm()];
        if (isCycle(tally(prof))) continue;
      }
      if (prof.map(key).join('|') !== cur) return prof;
    }
    return current;
  }

  function render(rootEl) {
    /* 預設為經典循環組合 */
    var state = [['A', 'B', 'C'], ['B', 'C', 'A'], ['C', 'A', 'B']];

    /* --- 標註者卡片 --- */
    var selects = [];
    var cards = state.map(function (perm, i) {
      var sel = el('select', { 'aria-label': '標註者 ' + (i + 1) + ' 的排序', style: 'width:100%;margin-top:6px;' });
      PERMS.forEach(function (p) { sel.appendChild(el('option', { value: key(p), text: label(p) })); });
      sel.value = key(perm);
      sel.addEventListener('change', function () {
        state[i] = PERMS.filter(function (p) { return key(p) === sel.value; })[0];
        refresh();
      });
      selects.push(sel);
      return el('div', { class: 'widget-panel', style: 'flex:1 1 150px;min-width:150px;' }, [
        el('div', { text: '標註者 ' + (i + 1), style: 'font-weight:600;' }),
        el('div', { text: '偏好排序（好 → 差）', style: 'font-size:.8em;color:var(--fg-muted);margin-top:2px;' }),
        sel
      ]);
    });

    /* --- 成對多數決結果 --- */
    var chips = PAIRS.map(function () {
      return el('div', { style: 'flex:1 1 140px;background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:6px 10px;font-size:.9em;text-align:center;' });
    });

    /* --- SVG 三角圖 --- */
    var svg = svgEl('svg', { viewBox: '0 0 320 256', role: 'img', 'aria-label': '成對多數決三角圖', style: 'width:100%;max-width:400px;height:auto;display:block;margin:0 auto;' });
    var defs = svgEl('defs');
    [['chw10-arr', 'var(--accent)'], ['chw10-arr-cyc', 'var(--accent-2)']].forEach(function (m) {
      var mk = svgEl('marker', { id: m[0], markerWidth: 10, markerHeight: 8, refX: 9, refY: 4, orient: 'auto', markerUnits: 'userSpaceOnUse' });
      mk.appendChild(svgEl('path', { d: 'M0,0 L10,4 L0,8 Z', fill: m[1] }));
      defs.appendChild(mk);
    });
    var gEdges = svgEl('g');
    var gNodes = svgEl('g');
    ['A', 'B', 'C'].forEach(function (o) {
      gNodes.appendChild(svgEl('circle', { cx: POS[o].x, cy: POS[o].y, r: R, fill: 'var(--panel-2)', stroke: 'var(--border)', 'stroke-width': 1.5 }));
      gNodes.appendChild(svgEl('text', { x: POS[o].x, y: POS[o].y + 1, 'text-anchor': 'middle', 'dominant-baseline': 'middle', fill: 'var(--fg)', 'font-size': 15, 'font-weight': 600, text: '回覆 ' + o }));
    });
    svg.appendChild(defs); svg.appendChild(gEdges); svg.appendChild(gNodes);

    var warn = el('div', { style: 'display:none;margin:10px auto 0;max-width:400px;border:1px solid var(--accent-2);color:var(--accent-2);border-radius:8px;padding:8px 12px;font-weight:600;text-align:center;', text: '偏好循環！不存在一致的「最佳回覆」' });
    var legend = el('div', { text: '箭頭 X → Y 表示多數標註者認為 X 優於 Y', style: 'text-align:center;font-size:.8em;color:var(--fg-muted);margin-top:8px;' });

    /* --- 按鈕與動態解讀 --- */
    function preset(wantCycle) {
      state = randomProfile(wantCycle, state);
      selects.forEach(function (sel, i) { sel.value = key(state[i]); });
      refresh();
    }
    var btnCyc = el('button', { type: 'button', text: '換一組會循環的例子' });
    var btnAcy = el('button', { type: 'button', text: '換一組不循環的例子' });
    btnCyc.addEventListener('click', function () { preset(true); });
    btnAcy.addEventListener('click', function () { preset(false); });

    var interp = el('div', { class: 'widget-panel', style: 'font-size:.92em;line-height:1.7;' });

    function drawEdge(res, cycle) {
      var p1 = POS[res.winner], p2 = POS[res.loser];
      var dx = p2.x - p1.x, dy = p2.y - p1.y, len = Math.sqrt(dx * dx + dy * dy);
      var ux = dx / len, uy = dy / len;
      var x1 = p1.x + ux * (R + 4), y1 = p1.y + uy * (R + 4);
      var x2 = p2.x - ux * (R + 6), y2 = p2.y - uy * (R + 6);
      gEdges.appendChild(svgEl('line', {
        x1: x1, y1: y1, x2: x2, y2: y2,
        stroke: cycle ? 'var(--accent-2)' : 'var(--accent)', 'stroke-width': cycle ? 3 : 2,
        'marker-end': 'url(#' + (cycle ? 'chw10-arr-cyc' : 'chw10-arr') + ')'
      }));
      var mx = (x1 + x2) / 2, my = (y1 + y2) / 2; /* 票數標籤放在三角形外側 */
      var ox = mx - CENTER.x, oy = my - CENTER.y, ol = Math.sqrt(ox * ox + oy * oy) || 1;
      gEdges.appendChild(svgEl('text', {
        x: mx + (ox / ol) * 16, y: my + (oy / ol) * 16, 'text-anchor': 'middle', 'dominant-baseline': 'middle',
        fill: 'var(--fg-muted)', 'font-size': 11, text: res.w + ':' + res.l
      }));
    }

    function refresh() {
      var results = tally(state);
      var cycle = isCycle(results);
      results.forEach(function (r, i) {
        chips[i].textContent = '';
        chips[i].appendChild(el('span', { text: r.a + ' vs ' + r.b + '：', style: 'color:var(--fg-muted);' }));
        chips[i].appendChild(el('strong', { text: r.winner + ' 勝', style: 'color:var(--accent);' }));
        chips[i].appendChild(el('span', { text: '（' + r.w + ':' + r.l + '）' }));
      });
      while (gEdges.firstChild) gEdges.removeChild(gEdges.firstChild);
      results.forEach(function (r) { drawEdge(r, cycle); });
      warn.style.display = cycle ? 'block' : 'none';

      interp.textContent = '';
      if (cycle) {
        var beats = {};
        results.forEach(function (r) { beats[r.winner] = r.loser; });
        var c0 = 'A', c1 = beats[c0], c2 = beats[c1];
        interp.appendChild(el('p', { text: '三位標註者各自的排序都完全理性（具遞移性），但多數決彙總後卻得到 ' + c0 + ' 勝 ' + c1 + '、' + c1 + ' 勝 ' + c2 + '、' + c2 + ' 又回頭勝 ' + c0 + '——群體偏好不具遞移性，選不出 Condorcet 贏家。' }));
        interp.appendChild(el('p', { text: '若獎勵模型必須為 A、B、C 各給一個純量分數，任何一組分數都至少違背其中一項多數意見。這正是社會選擇理論中 Arrow 不可能定理的直觀展示：把多元的人類偏好壓縮成單一效用（獎勵）函數，必然有所損失——而 RLHF 的獎勵模型，做的正是這種彙總。', style: 'margin-top:8px;' }));
      } else {
        var wins = { A: 0, B: 0, C: 0 };
        results.forEach(function (r) { wins[r.winner]++; });
        var order = ['A', 'B', 'C'].sort(function (x, y) { return wins[y] - wins[x]; });
        interp.appendChild(el('p', { text: '目前的組合存在 Condorcet 贏家：回覆 ' + order[0] + ' 在所有成對比較中勝出，群體排序 ' + order.join(' > ') + ' 是一致（具遞移性）的，可以用單一純量獎勵函數忠實表示，例如 r(' + order[0] + ') > r(' + order[1] + ') > r(' + order[2] + ')。' }));
        interp.appendChild(el('p', { text: '但這種一致性相當脆弱：往往只要一位標註者改變心意，循環就會出現。試著調整上方的下拉選單，或按「換一組會循環的例子」，看看單一獎勵函數的假設如何失效。', style: 'margin-top:8px;' }));
      }
    }

    rootEl.appendChild(el('div', {}, [
      el('div', { class: 'widget-row', style: 'display:flex;flex-wrap:wrap;gap:10px;' }, cards),
      el('div', { class: 'widget-row', style: 'display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;' }, chips),
      el('div', { class: 'widget-panel', style: 'margin-top:10px;' }, [svg, warn, legend]),
      el('div', { class: 'widget-row', style: 'display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;' }, [btnCyc, btnAcy]),
      el('div', { style: 'margin-top:10px;' }, [interp])
    ]));
    refresh();
  }

  window.ChapterWidget = {
    title: '偏好彙總悖論：三位標註者與 Condorcet 循環',
    intro: '三位標註者對同一提示的三個回覆 A、B、C 給出完整排序，再以成對多數決彙總成「群體偏好」。即使每個人的偏好都完全理性（具遞移性），彙總結果仍可能形成循環、找不到一致的最佳回覆——這正是獎勵模型把多元人類偏好壓成單一純量時面臨的根本困難。',
    render: render
  };
})();
