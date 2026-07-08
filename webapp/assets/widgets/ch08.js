/* 第 8 章互動元件：DPO 損失探索器 */
(function () {
  'use strict';

  // ---------- 數學工具 ----------
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const softplus = (x) => (x > 30 ? x : Math.log1p(Math.exp(x))); // 數值穩定
  const dpoLoss = (beta, dz) => softplus(-beta * dz);             // -log σ(β·Δz)
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const fmt = (v, d) => (Object.is(v, -0) ? 0 : v).toFixed(d == null ? 3 : d);
  const fmtBeta = (b) => (b < 0.1 ? b.toFixed(3) : b < 1 ? b.toFixed(2) : b.toFixed(1));

  // β 滑桿使用對數刻度：t ∈ [0,100] → β ∈ [0.01, 5]
  const B_MIN = 0.01, B_MAX = 5;
  const tToBeta = (t) => B_MIN * Math.pow(B_MAX / B_MIN, t / 100);
  const betaToT = (b) => 100 * Math.log(b / B_MIN) / Math.log(B_MAX / B_MIN);

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
  const W = 560, H = 300, PAD = { l: 46, r: 14, t: 14, b: 40 };
  const X_MIN = -10, X_MAX = 10, Y_MAX = 5;
  const sx = (x) => PAD.l + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - PAD.l - PAD.r);
  const sy = (y) => PAD.t + (1 - clamp(y, 0, Y_MAX) / Y_MAX) * (H - PAD.t - PAD.b);
  const invX = (px) => X_MIN + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (X_MAX - X_MIN);

  function curvePath(beta) {
    const pts = [];
    for (let i = 0; i <= 180; i++) {
      const x = X_MIN + (i / 180) * (X_MAX - X_MIN);
      pts.push((i ? 'L' : 'M') + sx(x).toFixed(1) + ',' + sy(dpoLoss(beta, x)).toFixed(1));
    }
    return pts.join(' ');
  }

  // ---------- 元件本體 ----------
  window.ChapterWidget = {
    title: 'DPO 損失探索器',
    intro: '拖動 β 與被選／被拒回覆的 log-ratio，觀察隱含獎勵差（margin）、DPO 損失與梯度權重 σ(−margin) 如何變化。曲線顯示不同 β 之下損失對 (z_c − z_r) 的形狀。',
    render(root) {
      const state = { beta: 0.1, zc: 0.5, zr: -0.5 };

      // --- 公式面板 ---
      const eqMain = el('div');
      const eqNow = el('div', { style: 'margin-top:.35rem;' });
      tex(eqMain,
        '\\mathcal{L}_{\\mathrm{DPO}} = -\\log \\sigma\\big(\\underbrace{\\beta\\,(z_c - z_r)}_{\\text{margin}}\\big),\\qquad ' +
        'z = \\log \\tfrac{\\pi_\\theta(y\\mid x)}{\\pi_{\\mathrm{ref}}(y\\mid x)},\\qquad ' +
        'w = \\sigma(-\\text{margin})');
      const eqPanel = el('div', { class: 'widget-panel', style: 'overflow-x:auto;' }, [eqMain, eqNow]);

      // --- 滑桿列 ---
      function makeSlider(labelHTML, min, max, step, value, oninput) {
        const lab = el('label', { style: 'flex:1 1 200px; min-width:180px; font-size:.9rem;' });
        const head = el('div', { style: 'display:flex; justify-content:space-between; margin-bottom:.15rem;' });
        const name = el('span'); name.innerHTML = labelHTML;
        const val = el('span', { style: 'color:var(--accent); font-variant-numeric:tabular-nums; font-weight:600;' });
        head.appendChild(name); head.appendChild(val);
        const input = el('input', { type: 'range', min, max, step, value, style: 'width:100%;' });
        input.addEventListener('input', () => { oninput(parseFloat(input.value)); update(); });
        lab.appendChild(head); lab.appendChild(input);
        return { lab, input, val };
      }
      const sBeta = makeSlider('β（KL 約束強度，log 刻度）', 0, 100, 0.5, betaToT(state.beta), (t) => { state.beta = tToBeta(t); });
      const sZc = makeSlider('被選 z<sub>c</sub> = log(π/π<sub>ref</sub>)(y<sub>c</sub>)', -5, 5, 0.05, state.zc, (v) => { state.zc = v; });
      const sZr = makeSlider('被拒 z<sub>r</sub> = log(π/π<sub>ref</sub>)(y<sub>r</sub>)', -5, 5, 0.05, state.zr, (v) => { state.zr = v; });
      const resetBtn = el('button', { type: 'button', text: '重設預設值（β=0.1, z_c=0.5, z_r=−0.5）' });
      resetBtn.addEventListener('click', () => {
        state.beta = 0.1; state.zc = 0.5; state.zr = -0.5;
        sBeta.input.value = betaToT(0.1); sZc.input.value = 0.5; sZr.input.value = -0.5;
        update();
      });
      const controls = el('div', { class: 'widget-panel' }, [
        el('div', { class: 'widget-row' }, [sBeta.lab, sZc.lab, sZr.lab]),
        el('div', { style: 'margin-top:.6rem;' }, [resetBtn]),
      ]);

      // --- 數值面板 ---
      function statBox(label) {
        const v = el('div', { style: 'font-size:1.15rem; font-weight:700; color:var(--accent); font-variant-numeric:tabular-nums;' });
        const box = el('div', { style: 'flex:1 1 140px; background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:.55rem .8rem;' }, [
          el('div', { text: label, style: 'font-size:.8rem; color:var(--fg-muted);' }), v,
        ]);
        return { box, v };
      }
      const stMargin = statBox('隱含獎勵差 margin = β(z_c − z_r)');
      const stLoss = statBox('DPO 損失 −log σ(margin)');
      const stW = statBox('梯度權重 σ(−margin)');
      const rewardLine = el('div', { style: 'margin-top:.5rem; font-size:.85rem; color:var(--fg-muted); font-variant-numeric:tabular-nums;' });
      const stats = el('div', { class: 'widget-panel' }, [
        el('div', { class: 'widget-row' }, [stMargin.box, stLoss.box, stW.box]), rewardLine,
      ]);

      // --- SVG 曲線圖 ---
      const svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, style: 'width:100%; height:auto; display:block; touch-action:none; cursor:crosshair;' });
      // 格線與座標軸
      for (let x = X_MIN; x <= X_MAX; x += 5) {
        svg.appendChild(svgEl('line', { x1: sx(x), y1: sy(0), x2: sx(x), y2: sy(Y_MAX), stroke: 'var(--border)', 'stroke-width': x === 0 ? 1.5 : 1 }));
        const t = svgEl('text', { x: sx(x), y: H - PAD.b + 16, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--fg-muted)' });
        t.textContent = x; svg.appendChild(t);
      }
      for (let y = 0; y <= Y_MAX; y++) {
        svg.appendChild(svgEl('line', { x1: sx(X_MIN), y1: sy(y), x2: sx(X_MAX), y2: sy(y), stroke: 'var(--border)', 'stroke-width': 1 }));
        const t = svgEl('text', { x: PAD.l - 7, y: sy(y) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--fg-muted)' });
        t.textContent = y; svg.appendChild(t);
      }
      const xLab = svgEl('text', { x: sx(0), y: H - 4, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--fg)' });
      xLab.textContent = 'z_c − z_r（log-ratio 差）'; svg.appendChild(xLab);
      const yLab = svgEl('text', { x: 14, y: sy(Y_MAX / 2), 'font-size': 12, fill: 'var(--fg)', transform: 'rotate(-90 14 ' + sy(Y_MAX / 2) + ')', 'text-anchor': 'middle' });
      yLab.textContent = 'DPO 損失'; svg.appendChild(yLab);
      // 對照虛線（β 固定組）＋當前曲線＋操作點
      const REF_BETAS = [0.05, 0.5, 2];
      const refGroup = svgEl('g'); svg.appendChild(refGroup);
      const mainPath = svgEl('path', { fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5 }); svg.appendChild(mainPath);
      const guide = svgEl('line', { stroke: 'var(--accent-2)', 'stroke-width': 1, 'stroke-dasharray': '3 3' }); svg.appendChild(guide);
      const dot = svgEl('circle', { r: 6, fill: 'var(--accent-2)', stroke: 'var(--panel)', 'stroke-width': 2 }); svg.appendChild(dot);
      const hoverDot = svgEl('circle', { r: 4, fill: 'none', stroke: 'var(--link)', 'stroke-width': 1.5, visibility: 'hidden' }); svg.appendChild(hoverDot);
      const hoverText = svgEl('text', { 'font-size': 11, fill: 'var(--fg-muted)', visibility: 'hidden' }); svg.appendChild(hoverText);
      const betaTag = svgEl('text', { 'font-size': 11, 'font-weight': 700, fill: 'var(--accent)' }); svg.appendChild(betaTag);

      function drawRefCurves() {
        refGroup.textContent = '';
        REF_BETAS.forEach((b) => {
          if (Math.abs(Math.log(b / state.beta)) < 0.12) return; // 與當前 β 太接近就略過
          refGroup.appendChild(svgEl('path', { d: curvePath(b), fill: 'none', stroke: 'var(--fg-muted)', 'stroke-width': 1.3, 'stroke-dasharray': '5 4', opacity: 0.65 }));
          const lx = clamp(-Math.log(Math.expm1(Y_MAX * 0.78)) / b, X_MIN + 0.6, X_MAX - 1.4); // 標注放在曲線約 0.78·Y_MAX 高度處
          const t = svgEl('text', { x: sx(lx) + 6, y: sy(dpoLoss(b, lx)) - 6, 'font-size': 11, fill: 'var(--fg-muted)' });
          t.textContent = 'β=' + fmtBeta(b); refGroup.appendChild(t);
        });
      }

      // 拖動／hover：以 Δz 對稱設定 z_c = Δz/2、z_r = −Δz/2
      function eventDz(ev) {
        const r = svg.getBoundingClientRect();
        return clamp(invX((ev.clientX - r.left) * (W / r.width)), X_MIN, X_MAX);
      }
      function applyDz(dz) {
        state.zc = clamp(dz / 2, -5, 5); state.zr = clamp(-dz / 2, -5, 5);
        sZc.input.value = state.zc; sZr.input.value = state.zr; update();
      }
      let dragging = false;
      svg.addEventListener('pointerdown', (ev) => { dragging = true; svg.setPointerCapture(ev.pointerId); applyDz(eventDz(ev)); });
      svg.addEventListener('pointerup', () => { dragging = false; });
      svg.addEventListener('pointermove', (ev) => {
        const dz = eventDz(ev);
        if (dragging) { applyDz(dz); hoverDot.setAttribute('visibility', 'hidden'); hoverText.setAttribute('visibility', 'hidden'); return; }
        const loss = dpoLoss(state.beta, dz);
        hoverDot.setAttribute('cx', sx(dz)); hoverDot.setAttribute('cy', sy(loss));
        hoverText.setAttribute('x', clamp(sx(dz) + 8, PAD.l, W - 150)); hoverText.setAttribute('y', clamp(sy(loss) - 8, 24, H - PAD.b));
        hoverText.textContent = 'Δz=' + fmt(dz, 2) + '，損失=' + fmt(loss, 3) + '（點擊/拖動移動操作點）';
        hoverDot.setAttribute('visibility', 'visible'); hoverText.setAttribute('visibility', 'visible');
      });
      svg.addEventListener('pointerleave', () => { hoverDot.setAttribute('visibility', 'hidden'); hoverText.setAttribute('visibility', 'hidden'); });

      const chartPanel = el('div', { class: 'widget-panel' }, [svg]);
      const interp = el('div', { class: 'widget-panel', style: 'font-size:.92rem; line-height:1.7;' });

      // --- 更新 ---
      function update() {
        const { beta, zc, zr } = state;
        const dz = zc - zr, margin = beta * dz;
        const loss = softplus(-margin), w = sigmoid(-margin);
        sBeta.val.textContent = 'β = ' + fmtBeta(beta);
        sZc.val.textContent = fmt(zc, 2); sZr.val.textContent = fmt(zr, 2);
        stMargin.v.textContent = fmt(margin); stLoss.v.textContent = fmt(loss); stW.v.textContent = fmt(w);
        rewardLine.textContent = '隱含獎勵：r_c = βz_c = ' + fmt(beta * zc) + '，r_r = βz_r = ' + fmt(beta * zr) + '；z_c − z_r = ' + fmt(dz, 2);
        tex(eqNow, '\\text{目前：}\\ \\text{margin} = ' + fmtBeta(beta) + '\\times(' + fmt(zc, 2) + ' - (' + fmt(zr, 2) + ')) = ' + fmt(margin) +
          ',\\quad \\mathcal{L}_{\\mathrm{DPO}} = ' + fmt(loss) + ',\\quad w = \\sigma(-\\text{margin}) = ' + fmt(w));
        mainPath.setAttribute('d', curvePath(beta));
        drawRefCurves();
        dot.setAttribute('cx', sx(dz)); dot.setAttribute('cy', sy(loss));
        guide.setAttribute('x1', sx(dz)); guide.setAttribute('y1', sy(0)); guide.setAttribute('x2', sx(dz)); guide.setAttribute('y2', sy(loss));
        const tagX = clamp(-1.1 / beta, X_MIN + 0.5, -0.8);
        betaTag.setAttribute('x', sx(tagX) - 6); betaTag.setAttribute('y', sy(dpoLoss(beta, tagX)) - 8);
        betaTag.textContent = 'β=' + fmtBeta(beta) + '（當前）';
        // 動態解讀
        let mMsg;
        if (margin > 2) mMsg = 'margin 已為正且大：模型早已把這對樣本排對、差距充分，梯度權重 σ(−margin) ≈ ' + fmt(w) + '，這筆樣本的梯度貢獻趨近 0，最佳化器會把力氣留給其他還沒排對的樣本。';
        else if (margin > 0.2) mMsg = 'margin 為正：策略已偏好被選回覆，但差距尚未拉開，梯度權重 ' + fmt(w) + ' 仍會温和地推動繼續擴大被選與被拒的相對對數機率差。';
        else if (margin >= -0.2) mMsg = 'margin ≈ 0：策略對兩個回覆幾乎不分軒輊（相對參考模型的偏移相同），梯度權重約 0.5，損失約 log 2 ≈ 0.693——這正是訓練起點的典型狀態。';
        else if (margin >= -2) mMsg = 'margin 為負：隱含獎勵把被拒回覆排在前面（排序錯誤），梯度權重 ' + fmt(w) + ' > 0.5，這筆樣本會收到較大的更新，拉高 y_c、壓低 y_r 的似然。';
        else mMsg = 'margin 為負且大：獎勵估計嚴重錯誤，梯度權重 σ(−margin) ≈ ' + fmt(w) + '，接近上限 1——正如式 82：估計越錯，權重越大，更新越猛。';
        let bMsg;
        if (beta >= 1) bMsg = '目前 β = ' + fmtBeta(beta) + ' 相當大：同樣的 log-ratio 差會被放大成很大的 margin，損失曲線陡峭，模型偏離參考模型的懲罰越強（KL 約束緊），些微偏移就足以飽和。';
        else if (beta >= 0.05) bMsg = '目前 β = ' + fmtBeta(beta) + '（DPO 論文預設約 0.1）：曲線斜率適中，需要可觀的 log-ratio 差才能把損失壓低，在「排對偏好」與「不要偏離參考模型太遠」之間取得平衡。';
        else bMsg = '目前 β = ' + fmtBeta(beta) + ' 非常小：KL 約束鬆，曲線平緩，策略得大幅偏離參考模型才能降低損失——自由度高，但也更容易過度最佳化。';
        interp.innerHTML = '<strong>解讀：</strong>' + mMsg + '<br><strong>β 的角色：</strong>' + bMsg;
      }

      root.appendChild(el('div', { style: 'display:flex; flex-direction:column; gap:1rem;' }, [eqPanel, controls, stats, chartPanel, interp]));
      update();
    },
  };
})();
