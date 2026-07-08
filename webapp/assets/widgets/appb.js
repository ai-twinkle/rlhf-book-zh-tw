/* 附錄 B 互動元件：話多的平衡（Chattiness Balance）體驗器 */
(function () {
  'use strict';

  var LEVELS = ['超簡短', '簡短', '適中', '詳盡條列', '過度鋪陳'];

  var TEXTS = [
    'RLHF（基於人類回饋的強化學習）是一種利用人類偏好訊號來微調語言模型的訓練方法。',

    'RLHF 是「基於人類回饋的強化學習」：先讓標註者比較模型的多個回覆並選出較好的那個，' +
    '用這些偏好資料訓練一個獎勵模型，再以強化學習讓語言模型往人類偏好的方向更新。' +
    'ChatGPT 等聊天助手「好聊」的特質，多半來自這一步。',

    '基於人類回饋的強化學習（RLHF）是機器學習的一個子領域，目標是把人類的判斷納入模型的訓練迴圈。' +
    '流程通常分三步：先以監督式微調讓模型學會遵循指令；接著收集人類對成對回覆的偏好比較，' +
    '訓練出一個獎勵模型來近似人類的喜好；最後用 PPO 或 DPO 等演算法，' +
    '讓語言模型在不偏離原模型太遠的前提下最大化這個獎勵。\n\n' +
    '相較於手工設計獎勵函數，這種做法能捕捉「什麼樣的回答比較好」這類難以明文定義的主觀標準，' +
    '但也面臨標註成本高、回饋不一致與獎勵駭客（reward hacking）等挑戰。',

    '基於人類回饋的強化學習（**RLHF**）是強化學習的一個子集，其中引導學習的獎勵由人類提供，' +
    '而非預先指定的數值獎勵函數。這在難以設計獎勵函數、或期望的表現涉及主觀判斷時特別有用。\n\n' +
    '**核心組成：**\n\n' +
    '1. **人類輸入：** 回饋來自人類評估者的定性判斷，形式包含數值評分、二元偏好、自然語言回饋或示範。\n' +
    '2. **獎勵模型：** 原始人類回饋存在變異與偏誤，需先訓練獎勵模型，把偏好轉換成穩定的學習訊號。\n' +
    '3. **策略最佳化：** 以 PPO、DPO 等演算法迭代更新模型，使其行為逐步貼近人類偏好。\n\n' +
    '**應用領域：**\n\n' +
    '- **對話助手：** 讓回覆更有幫助、無害且誠實。\n' +
    '- **機器人學：** 教機器人執行難以用獎勵函數描述的任務。\n' +
    '- **推薦系統：** 用人類回饋調整並改善推薦品質。\n\n' +
    '**主要挑戰：**\n\n' +
    '1. **可擴展性：** 收集人類回饋既耗時又昂貴。\n' +
    '2. **偏誤與雜訊：** 人類回饋不一致，可能導致次佳的學習結果。\n' +
    '3. **對齊與獎勵駭客：** 模型可能只學會討好表面訊號，而非真正的人類偏好。',

    '這是一個非常好的問題！基於人類回饋的強化學習（RLHF，Reinforcement Learning from Human ' +
    'Feedback）是一個內容相當豐富的主題，以下我將盡可能完整地為您說明。\n\n' +
    '首先，需要說明的是，RLHF 是一種利用人類偏好來訓練語言模型的方法。換句話說，' +
    '它讓人類的回饋參與模型的學習過程。也就是說，模型的訓練訊號來自人類的判斷，' +
    '而不是預先定義的獎勵函數。\n\n' +
    '**詳細流程如下：**\n\n' +
    '1. **監督式微調：** 首先，模型會先經過監督式微調。這一步非常重要，因為它是後續一切訓練的基礎；' +
    '沒有這一步，後面的步驟將難以進行。\n' +
    '2. **偏好收集：** 接著會收集人類的偏好資料。值得一提的是，這些資料的品質會直接影響最終效果，' +
    '因此資料品質非常關鍵，務必重視資料品質。\n' +
    '3. **獎勵模型訓練：** 然後訓練獎勵模型。如前所述，獎勵模型的作用是把人類偏好轉換成訓練訊號，' +
    '這正是第 2 步收集偏好資料的目的所在。\n' +
    '4. **強化學習：** 最後進行強化學習最佳化。這一步會用到前面所有步驟的成果，' +
    '也就是說，它整合了前述的一切。\n\n' +
    '**不過，需要注意的是**，實際效果可能因模型規模、資料品質與具體應用情境而異，以上內容僅供參考，' +
    '不構成任何工程決策建議。在實際部署前，建議諮詢相關領域的專業人士，並進行充分的評估與測試。' +
    '同時也要提醒您，AI 領域發展迅速，本說明可能無法涵蓋最新的進展。\n\n' +
    '總結來說，RLHF 就是用人類回饋來訓練模型的方法——正如開頭所說，' +
    '它讓人類的判斷參與模型的學習過程。希望以上說明對您有幫助！' +
    '如果您還有任何其他問題，歡迎隨時提出，我很樂意為您進一步說明！'
  ];

  /* 「適中」檔的 DPO 對照版：同樣的內容，更長、更結構化（呼應 Tülu 3 SFT vs DPO） */
  var DPO_TEXT =
    '基於人類回饋的強化學習（**RLHF**）是讓語言模型對齊人類偏好的核心技術：' +
    '引導學習的獎勵訊號來自人類判斷，而非預先寫定的獎勵函數。\n\n' +
    '典型流程包含三個階段：\n\n' +
    '1. **監督式微調（SFT）：** 以人工示範資料訓練模型遵循指令，作為後續訓練的起點。\n' +
    '2. **獎勵模型訓練：** 收集標註者對成對回覆的偏好比較（被選 ≻ 被拒），' +
    '訓練一個能為任意回覆打分的獎勵模型。\n' +
    '3. **強化學習最佳化：** 以 PPO 或 DPO 等演算法更新策略，在最大化獎勵的同時，' +
    '以 KL 懲罰約束模型不要偏離參考模型太遠。\n\n' +
    '**為什麼重要：** 這種做法能捕捉「什麼樣的回答比較好」這類難以明文定義的主觀標準，' +
    '是 ChatGPT 等助手好用的關鍵。\n\n' +
    '**主要挑戰：** 標註成本高、人類回饋充滿雜訊與偏誤，' +
    '以及模型可能學會鑽獎勵的漏洞（獎勵駭客）。';

  var INTERPS = [
    '一句話回答的資訊密度最高，但在成對比較中，評審常覺得它「不夠周全」——精簡的答案很容易輸給看起來更完整的對手。',
    '簡短但有內容。Starling Beta 這類模型示範了「長得剛好」：平均回應變長了，但增加的方式確實幫助到人類評分者。',
    '接近甜蜜點。評審（人類與 LLM）平均偏好較長、較完整的回覆——這是 RLHF 模型越來越囉唆的原因之一。試試切換上方的 SFT／DPO 對照。',
    '偏好微調的典型風格：粗體、編號、條列清單。勝率仍高，但每個字承載的資訊已開始稀釋——「話多」不只是長度，也包含格式編排。',
    '超過甜蜜點後，冗長反而傷害體驗，但自動評測不一定罰得到——這正是 AlpacaEval 與 WildBench 加入線性長度校正機制的原因。'
  ];
  var INTERP_DPO =
    '同樣的內容，DPO 版多了結構、粗體與編號，長度也明顯增加——長度與格式編排正是偏好微調最一致的「指紋」，且已被反覆證明與評審偏好相關。';

  var CSS = '' +
    '.appbw-q{font-weight:700;color:var(--fg);}' +
    '.appbw-q code{background:var(--code-bg);border-radius:6px;padding:2px 8px;font-size:.9em;}' +
    '.appbw-toggle{margin-left:auto;display:flex;gap:6px;}' +
    '.appbw-toggle button.is-active{background:var(--accent-soft);border-color:var(--accent);color:var(--fg);}' +
    '.appbw-card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 16px;margin:12px 0;}' +
    '.appbw-meta{display:flex;align-items:center;gap:8px;margin-bottom:10px;font-size:.78rem;color:var(--fg-muted);}' +
    '.appbw-badge{padding:2px 10px;border-radius:999px;background:var(--accent-soft);color:var(--accent-2);border:1px solid var(--border);}' +
    '.appbw-count{margin-left:auto;}' +
    '.appbw-text{color:var(--fg);font-size:.92rem;line-height:1.85;max-height:300px;overflow-y:auto;transition:opacity .18s ease;}' +
    '.appbw-text p{margin:0 0 10px;white-space:pre-wrap;}' +
    '.appbw-text p:last-child{margin-bottom:0;}' +
    '.appbw-note{margin-top:10px;padding-top:8px;border-top:1px dashed var(--border);color:var(--fg-muted);font-size:.82rem;}' +
    '.appbw-sliderrow{align-items:center;margin-top:16px;}' +
    '.appbw-sliderrow label{color:var(--fg);font-size:.9rem;white-space:nowrap;}' +
    '.appbw-sliderrow input[type=range]{flex:1;min-width:120px;}' +
    '.appbw-levels{display:flex;justify-content:space-between;font-size:.76rem;color:var(--fg-muted);margin-top:4px;}' +
    '.appbw-levels span.is-active{color:var(--accent);font-weight:700;}' +
    '.appbw-legend{display:flex;flex-wrap:wrap;gap:16px;font-size:.8rem;color:var(--fg-muted);margin:16px 0 6px;}' +
    '.appbw-legend i{display:inline-block;width:18px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px;}' +
    '.appbw-chart svg{width:100%;height:auto;display:block;}' +
    '.appbw-interp{background:var(--accent-soft);border:1px solid var(--border);border-radius:10px;padding:10px 14px;margin-top:12px;color:var(--fg);font-size:.88rem;line-height:1.7;}';

  var SVG_NS = 'http://www.w3.org/2000/svg';

  function el(tag, cls, parent) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (parent) parent.appendChild(node);
    return node;
  }
  function svg(tag, attrs, parent) {
    var node = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  /* 依 \n\n 分段、支援 **粗體** 的極簡渲染 */
  function renderRich(target, text) {
    target.textContent = '';
    text.split('\n\n').forEach(function (block) {
      var p = el('p', '', target);
      var parts = block.split(/\*\*([^*]+)\*\*/g);
      for (var i = 0; i < parts.length; i++) {
        if (!parts[i]) continue;
        if (i % 2 === 1) el('strong', '', p).textContent = parts[i];
        else p.appendChild(document.createTextNode(parts[i]));
      }
    });
  }

  /* 曲線：人類偏好勝率（倒 U，峰值中間偏右）與資訊密度（單調下降） */
  function fPref(t) { return 0.16 + 0.74 * Math.exp(-Math.pow(t - 0.58, 2) / (2 * 0.2 * 0.2)); }
  function fDens(t) { return 0.10 + 0.85 * Math.exp(-2.1 * t); }

  function render(rootEl) {
    var state = { value: 45, mode: 'sft' };

    var style = document.createElement('style');
    style.textContent = CSS;
    rootEl.appendChild(style);

    /* ── 頂列：提問 + SFT/DPO 切換 ── */
    var top = el('div', 'widget-row', rootEl);
    top.style.alignItems = 'center';
    var q = el('div', 'appbw-q', top);
    q.appendChild(document.createTextNode('提問：'));
    el('code', '', q).textContent = '什麼是 RLHF？';
    var toggle = el('div', 'appbw-toggle', top);
    var btnSft = el('button', 'is-active', toggle); btnSft.type = 'button'; btnSft.textContent = 'SFT 模型';
    var btnDpo = el('button', '', toggle); btnDpo.type = 'button'; btnDpo.textContent = 'DPO 模型';

    /* ── 展示卡 ── */
    var card = el('div', 'appbw-card', rootEl);
    var meta = el('div', 'appbw-meta', card);
    var badge = el('span', 'appbw-badge', meta);
    var levelTag = el('span', '', meta);
    var count = el('span', 'appbw-count', meta);
    var textEl = el('div', 'appbw-text', card);
    var note = el('div', 'appbw-note', card);

    /* ── 滑桿 ── */
    var row = el('div', 'widget-row appbw-sliderrow', rootEl);
    var lab = el('label', '', row); lab.textContent = '回覆詳盡度';
    lab.htmlFor = 'appbw-range';
    var range = el('input', '', row);
    range.type = 'range'; range.id = 'appbw-range';
    range.min = '0'; range.max = '100'; range.value = String(state.value);
    var levelsBar = el('div', 'appbw-levels', rootEl);
    var levelSpans = LEVELS.map(function (name) {
      var s = el('span', '', levelsBar); s.textContent = name; return s;
    });

    /* ── SVG 曲線圖 ── */
    var legend = el('div', 'appbw-legend', rootEl);
    var l1 = el('span', '', legend); var i1 = el('i', '', l1); i1.style.background = 'var(--accent)';
    l1.appendChild(document.createTextNode('人類偏好勝率'));
    var l2 = el('span', '', legend); var i2 = el('i', '', l2); i2.style.background = 'var(--accent-2)';
    l2.appendChild(document.createTextNode('資訊密度'));

    var W = 560, PL = 16, PR = 16, PT = 18, BOT = 188;
    function tx(t) { return PL + t * (W - PL - PR); }
    function ty(v) { return BOT - v * (BOT - PT); }
    function path(fn) {
      var d = '';
      for (var i = 0; i <= 60; i++) {
        var t = i / 60;
        d += (i ? 'L' : 'M') + tx(t).toFixed(1) + ',' + ty(fn(t)).toFixed(1);
      }
      return d;
    }
    var chart = el('div', 'appbw-chart', rootEl);
    var s = svg('svg', { viewBox: '0 0 560 232', role: 'img', 'aria-label': '回覆長度與偏好勝率、資訊密度的關係圖' }, chart);
    /* 甜蜜點區帶 */
    svg('rect', { x: tx(0.48), y: PT, width: tx(0.68) - tx(0.48), height: BOT - PT, fill: 'var(--accent-soft)' }, s);
    svg('text', { x: tx(0.58), y: PT + 12, 'text-anchor': 'middle', 'font-size': '11', fill: 'var(--fg-muted)' }, s)
      .textContent = '甜蜜點';
    /* 座標軸與檔位刻度 */
    svg('line', { x1: PL, y1: BOT, x2: W - PR, y2: BOT, stroke: 'var(--border)', 'stroke-width': 1.5 }, s);
    LEVELS.forEach(function (name, i) {
      var x = tx(0.1 + i * 0.2);
      svg('line', { x1: x, y1: BOT, x2: x, y2: BOT + 5, stroke: 'var(--border)' }, s);
      svg('text', { x: x, y: BOT + 18, 'text-anchor': 'middle', 'font-size': '10.5', fill: 'var(--fg-muted)' }, s)
        .textContent = name;
    });
    svg('text', { x: W - PR, y: BOT + 36, 'text-anchor': 'end', 'font-size': '11', fill: 'var(--fg-muted)' }, s)
      .textContent = '回覆長度 →';
    /* 兩條曲線 */
    svg('path', { d: path(fPref), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5, 'stroke-linecap': 'round' }, s);
    svg('path', { d: path(fDens), fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 2.5, 'stroke-linecap': 'round' }, s);
    /* 當前位置標記 */
    var marker = svg('line', { y1: PT, y2: BOT, stroke: 'var(--fg-muted)', 'stroke-dasharray': '4 3' }, s);
    var dotPref = svg('circle', { r: 4.5, fill: 'var(--accent)', stroke: 'var(--bg)', 'stroke-width': 1.5 }, s);
    var dotDens = svg('circle', { r: 4.5, fill: 'var(--accent-2)', stroke: 'var(--bg)', 'stroke-width': 1.5 }, s);

    /* ── 動態解讀 ── */
    var interp = el('div', 'appbw-interp', rootEl);

    function levelOf(v) { return Math.min(4, Math.floor(v / 20)); }
    function currentText(lv) {
      return (state.mode === 'dpo' && lv === 2) ? DPO_TEXT : TEXTS[lv];
    }

    var lastKey = null, fadeTimer = null;
    function updateCard(lv) {
      var key = lv + '-' + state.mode;
      if (key === lastKey) return;
      lastKey = key;
      var text = currentText(lv);
      textEl.style.opacity = '0';
      if (fadeTimer) clearTimeout(fadeTimer);
      fadeTimer = setTimeout(function () {
        renderRich(textEl, text);
        textEl.scrollTop = 0;
        textEl.style.opacity = '1';
      }, 180);
      badge.textContent = state.mode === 'dpo' ? 'DPO 模型' : 'SFT 模型';
      levelTag.textContent = '檔位：' + LEVELS[lv];
      count.textContent = '約 ' + text.replace(/\*\*/g, '').replace(/\s/g, '').length + ' 字';
      levelSpans.forEach(function (sp, i) { sp.classList.toggle('is-active', i === lv); });
      interp.textContent = (state.mode === 'dpo' && lv === 2) ? INTERP_DPO : INTERPS[lv];
      if (state.mode === 'dpo' && lv !== 2) {
        note.style.display = '';
        note.textContent = 'SFT／DPO 對照文本以「適中」檔最完整——把滑桿移到中段，看 DPO 如何把同樣的內容變得更長、更結構化。';
      } else {
        note.style.display = 'none';
      }
    }
    function updateMarker() {
      var x = tx(state.value / 100).toFixed(1);
      marker.setAttribute('x1', x); marker.setAttribute('x2', x);
      dotPref.setAttribute('cx', x); dotPref.setAttribute('cy', ty(fPref(state.value / 100)).toFixed(1));
      dotDens.setAttribute('cx', x); dotDens.setAttribute('cy', ty(fDens(state.value / 100)).toFixed(1));
    }

    range.addEventListener('input', function () {
      state.value = Number(range.value);
      updateMarker();
      updateCard(levelOf(state.value));
    });
    btnSft.addEventListener('click', function () {
      state.mode = 'sft';
      btnSft.classList.add('is-active'); btnDpo.classList.remove('is-active');
      updateCard(levelOf(state.value));
    });
    btnDpo.addEventListener('click', function () {
      state.mode = 'dpo';
      btnDpo.classList.add('is-active'); btnSft.classList.remove('is-active');
      updateCard(levelOf(state.value));
    });

    updateMarker();
    updateCard(levelOf(state.value));
  }

  window.ChapterWidget = {
    title: '話多的平衡體驗器',
    intro: '拖動「回覆詳盡度」滑桿，觀察同一個提問在五種詳盡度下的示範回覆，並對照曲線圖上的人類偏好勝率（倒 U 型）與資訊密度（單調下降）；再切換 SFT／DPO 模型，體會偏好微調如何讓回覆變得更長、更結構化。',
    render: render
  };
})();
