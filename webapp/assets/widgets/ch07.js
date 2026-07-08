/* 第 7 章互動元件：推論時擴展模擬器（pass@k 與多數決） */
(function () {
  'use strict';

  // ---------- 數學工具 ----------
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const fmtPct = (v) => (v * 100).toFixed(1) + '%';
  // 對數階乘表（k 最大 256），用於數值穩定的二項分布計算
  const LOG_FACT = [0];
  for (let i = 1; i <= 256; i++) LOG_FACT[i] = LOG_FACT[i - 1] + Math.log(i);
  const logC = (n, k) => LOG_FACT[n] - LOG_FACT[k] - LOG_FACT[n - k];

  // pass@k = 1 − (1−p)^k：k 次取樣中「至少一次正確」（需要驗證器挑出對的那次）
  const passAt = (k, p) => 1 - Math.pow(1 - p, k);
  // maj@k：多數決答對的機率。X ~ Bin(k, p)，P(X > k/2) + ½·P(X = k/2)（平手擲硬幣）
  // 保守假設：每次作答獨立，且所有錯誤答案彼此相同（錯誤票不分散）
  function majAt(k, p) {
    const lp = Math.log(p), lq = Math.log(1 - p);
    let s = 0;
    for (let i = Math.ceil(k / 2); i <= k; i++) {
      const prob = Math.exp(logC(k, i) + i * lp + (k - i) * lq);
      s += (2 * i === k) ? 0.5 * prob : prob;
    }
    return clamp(s, 0, 1);
  }

  // ---------- DOM 小工具 ----------
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'text') n.textContent = attrs[k];
      else if (k === 'style') n.style.cssText = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    (children || []).forEach((c) => n.appendChild(c));
    return n;
  }
  const SVG_NS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs) {
    const n = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }
  function tex(node, src) {
    if (window.katex) window.katex.render(src, node, { throwOnError: false, displayMode: true });
    else node.textContent = src;
  }

  // ---------- 圖表座標 ----------
  const W = 560, H = 320, PAD = { l: 48, r: 16, t: 16, b: 44 };
  const IW = W - PAD.l - PAD.r, IH = H - PAD.t - PAD.b;
  const sy = (y) => PAD.t + (1 - clamp(y, 0, 1)) * IH;

  // ---------- 元件本體 ----------
  window.ChapterWidget = {
    title: '推論時擴展模擬器：pass@k 與多數決',
    intro: '拖動單次作答正確率 p 與樣本數上限 K，比較「有驗證器」的 pass@k 與「沒有驗證器、靠自我一致性」的多數決 maj@k 如何隨取樣次數 k 擴展。在曲線上拖動或點擊，可讀出任一 k 的數值。',
    render(root) {
      const state = { p: 0.3, expK: 6, k: 16 }; // K = 2^expK
      const K = () => Math.pow(2, state.expK);
      const sx = (k) => PAD.l + (Math.log2(k) / state.expK) * IW; // x 軸：log2 刻度

      // --- 公式面板 ---
      const eqMain = el('div');
      tex(eqMain,
        '\\text{pass@}k = 1 - (1-p)^k,\\qquad ' +
        '\\text{maj@}k = \\Pr\\!\\big[X > \\tfrac{k}{2}\\big] + \\tfrac{1}{2}\\Pr\\!\\big[X = \\tfrac{k}{2}\\big],\\quad X \\sim \\mathrm{Bin}(k,\\,p)');
      const eqNote = el('div', { style: 'font-size:.82rem; color:var(--fg-muted); margin-top:.3rem;',
        text: '假設：每次作答獨立、正確率皆為 p，且所有錯誤答案彼此相同（多數決的保守下界；平手時擲硬幣）。' });
      const eqPanel = el('div', { class: 'widget-panel', style: 'overflow-x:auto;' }, [eqMain, eqNote]);

      // --- 滑桿列 ---
      function makeSlider(labelText, min, max, step, value, oninput) {
        const lab = el('label', { style: 'flex:1 1 220px; min-width:200px; font-size:.9rem;' });
        const head = el('div', { style: 'display:flex; justify-content:space-between; margin-bottom:.15rem;' }, [
          el('span', { text: labelText }),
          el('span', { style: 'color:var(--accent); font-variant-numeric:tabular-nums; font-weight:600;' }),
        ]);
        const input = el('input', { type: 'range', min, max, step, value, style: 'width:100%;' });
        input.addEventListener('input', () => { oninput(parseFloat(input.value)); update(); });
        lab.appendChild(head); lab.appendChild(input);
        return { lab, input, val: head.lastChild };
      }
      const sP = makeSlider('單次作答正確率 p', 0.05, 0.95, 0.01, state.p, (v) => { state.p = v; });
      const sK = makeSlider('樣本數上限 K（log₂ 刻度）', 3, 8, 1, state.expK, (v) => { state.expK = v; state.k = Math.min(state.k, K()); });
      const controls = el('div', { class: 'widget-panel' }, [el('div', { class: 'widget-row' }, [sP.lab, sK.lab])]);

      // --- 數值面板（游標所在 k 的三個數值） ---
      function statBox(label, color) {
        const v = el('div', { style: 'font-size:1.15rem; font-weight:700; color:' + color + '; font-variant-numeric:tabular-nums;' });
        const box = el('div', { style: 'flex:1 1 130px; background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:.55rem .8rem;' }, [
          el('div', { text: label, style: 'font-size:.8rem; color:var(--fg-muted);' }), v,
        ]);
        return { box, v };
      }
      const stK = statBox('取樣次數 k', 'var(--fg)');
      const stPass = statBox('pass@k（有驗證器）', 'var(--accent)');
      const stMaj = statBox('maj@k（多數決）', 'var(--accent-2)');
      const stP = statBox('單次作答 p（基準）', 'var(--fg-muted)');
      const gapLine = el('div', { style: 'margin-top:.5rem; font-size:.85rem; color:var(--fg-muted); font-variant-numeric:tabular-nums;' });
      const stats = el('div', { class: 'widget-panel' }, [
        el('div', { class: 'widget-row' }, [stK.box, stPass.box, stMaj.box, stP.box]), gapLine,
      ]);

      // --- SVG 曲線圖 ---
      const svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, style: 'width:100%; height:auto; display:block; touch-action:none; cursor:crosshair;' });
      // y 軸格線（固定 0～1）
      for (let y = 0; y <= 1.001; y += 0.25) {
        svg.appendChild(svgEl('line', { x1: PAD.l, y1: sy(y), x2: W - PAD.r, y2: sy(y), stroke: 'var(--border)', 'stroke-width': 1 }));
        const t = svgEl('text', { x: PAD.l - 7, y: sy(y) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--fg-muted)' });
        t.textContent = (y * 100).toFixed(0) + '%'; svg.appendChild(t);
      }
      const yLab = svgEl('text', { x: 13, y: sy(0.5), 'font-size': 12, fill: 'var(--fg)', transform: 'rotate(-90 13 ' + sy(0.5) + ')', 'text-anchor': 'middle' });
      yLab.textContent = '答對機率'; svg.appendChild(yLab);
      const xLab = svgEl('text', { x: PAD.l + IW / 2, y: H - 4, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--fg)' });
      xLab.textContent = 'k（取樣次數，log₂ 刻度）'; svg.appendChild(xLab);
      const gX = svgEl('g'); svg.appendChild(gX);                    // x 軸刻度（隨 K 重建）
      const baseLine = svgEl('line', { stroke: 'var(--fg-muted)', 'stroke-width': 1.5, 'stroke-dasharray': '6 4' }); svg.appendChild(baseLine);
      const baseTag = svgEl('text', { 'font-size': 11, fill: 'var(--fg-muted)' }); svg.appendChild(baseTag);
      const majPath = svgEl('path', { fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 2.5 }); svg.appendChild(majPath);
      const passPath = svgEl('path', { fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5 }); svg.appendChild(passPath);
      const guide = svgEl('line', { stroke: 'var(--fg-muted)', 'stroke-width': 1, 'stroke-dasharray': '3 3' }); svg.appendChild(guide);
      const dotPass = svgEl('circle', { r: 5.5, fill: 'var(--accent)', stroke: 'var(--panel)', 'stroke-width': 2 }); svg.appendChild(dotPass);
      const dotMaj = svgEl('circle', { r: 5.5, fill: 'var(--accent-2)', stroke: 'var(--panel)', 'stroke-width': 2 }); svg.appendChild(dotMaj);

      function drawXAxis() {
        gX.textContent = '';
        for (let e = 0; e <= state.expK; e++) {
          const k = Math.pow(2, e);
          gX.appendChild(svgEl('line', { x1: sx(k), y1: PAD.t, x2: sx(k), y2: PAD.t + IH, stroke: 'var(--border)', 'stroke-width': 1 }));
          const t = svgEl('text', { x: sx(k), y: H - PAD.b + 16, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--fg-muted)' });
          t.textContent = k; gX.appendChild(t);
        }
      }
      function curvePath(fn) {
        const pts = [];
        for (let k = 1; k <= K(); k++) pts.push((k === 1 ? 'M' : 'L') + sx(k).toFixed(1) + ',' + sy(fn(k, state.p)).toFixed(1));
        return pts.join(' ');
      }

      // 游標拖動／點擊 → 選取 k（貼齊最近的整數 k）
      function pickK(ev) {
        const r = svg.getBoundingClientRect();
        const px = (ev.clientX - r.left) * (W / r.width);
        const v = ((px - PAD.l) / IW) * state.expK;             // log2(k)
        return clamp(Math.round(Math.pow(2, v)), 1, K());
      }
      let dragging = false;
      svg.addEventListener('pointerdown', (ev) => { dragging = true; svg.setPointerCapture(ev.pointerId); state.k = pickK(ev); update(); });
      svg.addEventListener('pointerup', () => { dragging = false; });
      svg.addEventListener('pointermove', (ev) => { if (dragging || ev.buttons === 0) { state.k = pickK(ev); update(); } });

      // --- 圖例 ---
      function legendItem(color, dash, label) {
        const sw = svgEl('svg', { viewBox: '0 0 26 10', style: 'width:26px; height:10px; flex:0 0 auto;' });
        sw.appendChild(svgEl('line', { x1: 1, y1: 5, x2: 25, y2: 5, stroke: color, 'stroke-width': 2.5, 'stroke-dasharray': dash }));
        return el('span', { style: 'display:inline-flex; align-items:center; gap:.35rem; font-size:.82rem; color:var(--fg-muted); margin-right:1rem;' },
          [sw, el('span', { text: label })]);
      }
      const legend = el('div', { style: 'margin-top:.4rem; display:flex; flex-wrap:wrap; row-gap:.3rem;' }, [
        legendItem('var(--accent)', 'none', 'pass@k：只要有一次對就算對——有驗證器時的上限'),
        legendItem('var(--accent-2)', 'none', 'maj@k：沒有驗證器時靠自我一致性（多數決）'),
        legendItem('var(--fg-muted)', '6 4', '單次作答 p（基準線）'),
      ]);
      const chartPanel = el('div', { class: 'widget-panel' }, [svg, legend]);

      // --- 動態解讀與對照卡 ---
      const interp = el('div', { class: 'widget-panel', style: 'font-size:.92rem; line-height:1.7;' });
      function card(title, body) {
        return el('div', { style: 'flex:1 1 240px; background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:.65rem .85rem;' }, [
          el('div', { text: title, style: 'font-weight:700; margin-bottom:.25rem; color:var(--accent);' }),
          el('div', { text: body, style: 'font-size:.85rem; color:var(--fg-muted); line-height:1.6;' }),
        ]);
      }
      const compare = el('div', { class: 'widget-panel' }, [
        el('div', { text: 'RL 訓練 vs 推論時擴展（§7.2.2）', style: 'font-weight:700; margin-bottom:.5rem;' }),
        el('div', { class: 'widget-row' }, [
          card('RL 訓練（RLVR）', '把「偶爾才對」的行為強化成穩健能力、內化進模型權重——直接抬高單次正確率 p，等於整條曲線的起點被墊高。'),
          card('推論時擴展', '在生成時花更多運算（更長推理鏈、取樣 k 次）換取表現——沿著曲線往右走，能力不變、只是用計算把它變現。'),
        ]),
      ]);

      // --- 更新 ---
      function update() {
        const { p, k } = state, Kv = K();
        sP.val.textContent = 'p = ' + fmtPct(p);
        sK.val.textContent = 'K = ' + Kv;
        drawXAxis();
        passPath.setAttribute('d', curvePath(passAt));
        majPath.setAttribute('d', curvePath(majAt));
        baseLine.setAttribute('x1', PAD.l); baseLine.setAttribute('x2', W - PAD.r);
        baseLine.setAttribute('y1', sy(p)); baseLine.setAttribute('y2', sy(p));
        baseTag.setAttribute('x', W - PAD.r - 4); baseTag.setAttribute('text-anchor', 'end');
        baseTag.setAttribute('y', sy(p) + (p > 0.9 ? 14 : -6)); baseTag.textContent = 'p = ' + fmtPct(p);
        const pk = passAt(k, p), mk = majAt(k, p);
        guide.setAttribute('x1', sx(k)); guide.setAttribute('x2', sx(k));
        guide.setAttribute('y1', PAD.t); guide.setAttribute('y2', PAD.t + IH);
        dotPass.setAttribute('cx', sx(k)); dotPass.setAttribute('cy', sy(pk));
        dotMaj.setAttribute('cx', sx(k)); dotMaj.setAttribute('cy', sy(mk));
        stK.v.textContent = 'k = ' + k;
        stPass.v.textContent = fmtPct(pk); stMaj.v.textContent = fmtPct(mk); stP.v.textContent = fmtPct(p);
        gapLine.textContent = '在 k = ' + k + ' 時，pass@k − maj@k = ' + fmtPct(pk - mk) + '——這段差距就是「有沒有驗證器」的價值。';
        // 動態解讀（核心教學）
        const passK = passAt(Kv, p), majK = majAt(Kv, p);
        let msg;
        if (p < 0.45) {
          msg = '目前 p = ' + fmtPct(p) + ' < 50%：多數決<strong>不升反降</strong>——錯誤答案佔多數的機率隨 k 放大，maj@' + Kv + ' ≈ ' +
            fmtPct(majK) + '，一路趨向 0；pass@' + Kv + ' 卻仍升到 ' + fmtPct(passK) +
            '。沒有驗證器時，「自我一致性」只會忠實放大模型最常見的答案——不夠常對，就是最常見地錯。此時取樣再多也救不了多數決，' +
            '要嘛先用 RLVR 訓練把 p 拉高，要嘛引入驗證器（正確答案抽取、單元測試）把 pass@k 的上限變現。';
        } else if (p <= 0.55) {
          msg = '目前 p = ' + fmtPct(p) + ' ≈ 50%：多數決近似擲硬幣，maj@k 幾乎黏在 50% 附近不動（maj@' + Kv + ' ≈ ' + fmtPct(majK) +
            '）；pass@k 依然穩定上升到 ' + fmtPct(passK) + '。這是分水嶺——自我一致性失效，但驗證器仍然有效。';
        } else {
          msg = '目前 p = ' + fmtPct(p) + ' > 50%：兩條曲線都上升——模型「夠常對」，多數決靠自我一致性就能收斂（maj@' + Kv + ' ≈ ' +
            fmtPct(majK) + '），不需要驗證器；pass@' + Kv + ' ≈ ' + fmtPct(passK) +
            ' 升得更快，兩者的差距就是驗證器額外榨出的表現。';
        }
        interp.innerHTML = '<strong>解讀：</strong>' + msg +
          '<br><strong>為什麼互補：</strong>推論時擴展只有在模型「夠常對」（p > 50%，多數決可用）或「有驗證器」（pass@k 可變現）時才有效。' +
          '這正是 RLVR 訓練與推論擴展互補的原因：RLVR 抬高 p、讓每一次取樣更值錢，推論擴展再沿著曲線用計算換表現。';
      }

      root.appendChild(el('div', { style: 'display:flex; flex-direction:column; gap:1rem;' }, [eqPanel, controls, stats, chartPanel, interp, compare]));
      update();
    },
  };
})();
