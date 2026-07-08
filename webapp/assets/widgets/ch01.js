(function () {
  'use strict';

  // 管線項目依序排列：偶數索引為節點（階段），奇數索引為箭頭（資料流）
  var ITEMS = [
    { kind: 'node', name: '基礎模型', en: 'Base Model',
      what: '經過大規模自迴歸預訓練（下一詞元預測）的語言模型。它只會「補完文字」，還不會以問答格式回應使用者。',
      input: '以網頁為主的大規模預訓練資料（數兆詞元）', output: '蘊藏知識與潛力、但沒有助理行為的基礎模型',
      chapter: '第 1 章 1.4 節（後訓練的直覺）、第 3 章（訓練概觀）',
      insight: '引出詮釋：基礎模型就像 F1 賽車的底盤，決定了最終模型絕大部分的潛力；後訓練的工作是把潛力全部引出來。' },
    { kind: 'edge', name: '指令資料 ~10K',
      what: '餵入精心打造的問答格式範例。模型「只」看到問答格式的資料，因此從補完文字轉為以助理人格回答查詢。',
      input: '約 1 萬筆（~10K）高品質指令—回應對', output: '訓練訊號：下一詞元預測損失',
      chapter: '第 4 章（指令微調）、第 12 章（合成資料）',
      insight: '量級小但品質關鍵：幾千到上萬筆樣本就能大幅改變模型的格式與人格，但這不代表「指令微調就足以實現對齊」。' },
    { kind: 'node', name: 'SFT 模型', en: 'SFT Model',
      what: '指令微調／監督式微調（IFT/SFT）：用同樣的下一詞元預測損失在問答資料上訓練，教導模型格式，打下遵循指令能力的基礎。',
      input: '指令資料，量級約 1 萬筆（~10K）', output: '能以問答格式可靠回應的指令遵循模型',
      chapter: '第 4 章（指令微調）',
      insight: 'SFT 學的是語言中的「特徵」：逐詞元（per-token）更新。它也是 RLHF 需要的強大起點——沒有好的 SFT，RLHF 難以奏效。' },
    { kind: 'edge', name: '偏好資料 ~100K',
      what: '標註者在同一提示詞的多個模型補全之間表達偏好（如「A 比 B 好」），構成訓練獎勵模型的成對比較資料集。',
      input: '約 10 萬筆（~100K）人類偏好比較', output: '訓練訊號：對比式（contrastive）損失',
      chapter: '第 10 章（偏好的本質）、第 11 章（偏好資料）',
      insight: '偏好資料遠比指令資料昂貴——10 萬至 100 萬美元等級的資料預算，曾是開放社群做 RLHF 的最大門檻。' },
    { kind: 'node', name: '獎勵模型', en: 'Reward Model',
      what: '通常以 SFT 模型為起點，在偏好資料上微調，學習「怎樣的回答比較好」，能為任何一段文字輸出好壞分數。',
      input: '偏好資料，量級約 10 萬筆（~100K）', output: '作為人類偏好代理（proxy）的獎勵模型',
      chapter: '第 5 章（獎勵模型建構）',
      insight: '獎勵模型充其量只是真實目標的代理，且資料雜訊較多——這正是 RL 階段容易「過度最佳化」、需要正則化的根源。' },
    { kind: 'edge', name: '獎勵訊號',
      what: '在 RL 期間，獎勵模型為策略取樣出的每個補全評分，把「人類覺得好不好」轉換成 RL 最佳化器可以使用的數字。',
      input: '語言模型生成的補全結果（completions）', output: '純量（scalar）獎勵分數',
      chapter: '第 5 章（獎勵模型建構）、第 6 章（強化學習）',
      insight: '這個純量訊號是整條管線的樞紐：它讓「難以明確定義」的人類偏好，變成可以被最佳化的目標。' },
    { kind: 'node', name: 'RL 最佳化', en: 'RL Optimizer',
      what: '取一批提示詞，讓模型生成補全，由獎勵模型評分，再用任選的 RL 最佳化器更新參數，讓好的詞元更可能出現。',
      input: '提示詞集合＋獎勵模型的純量訊號', output: '朝人類偏好迭代更新的模型參數',
      chapter: '第 3 章（訓練概觀）、第 6 章（強化學習）',
      insight: '與 SFT 的逐詞元更新不同，RLHF 在「整體回應」層級學習：告訴模型更好的回應長什麼樣、又該避免哪些回應。' },
    { kind: 'edge', name: '迭代更新',
      what: 'RL 最佳化器推導更新規則，把好壞歸因到模型參數上，並以迭代方式進行，以維持初始模型的一般能力。',
      input: '梯度更新（搭配 KL 等正則化約束）', output: '效能飽和後的最終模型',
      chapter: '第 6 章（強化學習）、第 15 章（正則化）',
      insight: '一旦 RL 完成且效能達到飽和，這通常就是提供給使用者的最終模型。' },
    { kind: 'node', name: '對齊模型', en: 'Aligned Model',
      what: 'RLHF 完成後的最終模型：以可靠、溫暖且引人入勝的風格回答問題，同時做到有幫助（helpful）且無害（harmless）。',
      input: '—（整條管線的最終產物）', output: '對齊人類偏好的助理模型（如 ChatGPT、Tülu 3）',
      chapter: '第 14–16 章（過度最佳化、正則化、評估）',
      insight: 'RLHF 比指令微調更能跨領域泛化，是催生現代後訓練的源頭。往下用對照卡感受它前後的差異。' }
  ];

  var DEFAULT_SEL = 2; // 預設高亮 SFT 模型
  var SVG_NS = 'http://www.w3.org/2000/svg';
  var W = 780, H = 96, NW = 112, NH = 46, NY = 32, GAP = 52, CY = NY + NH / 2;

  var COMPARE = {
    prompt: 'The president of the united states in 2006 was（2006 年的美國總統是）',
    base: { btn: '基礎模型（Base）', model: 'Llama 3.1 405B Base（僅預訓練）',
      text: 'George W. Bush（喬治·W·布希），2006 年的佛羅里達州州長是 Jeb Bush（傑布·布希），而 John McCain（約翰·馬侃）在 2006 年是亞利桑那州參議員——他後來敗給了 obama（歐巴馬）。9 月 1 日——美國總統布希簽署一項行政命令，讓信仰型組織在提供由聯邦資金支付的社會服務時擁有更多選項。1 月 1 日——目前由美國國會在《非法網路賭博執法法》（UIGEA）中設定的期限。search：1961 年《保護植物新品種國際公約》的修正案要求植物育種者權利納入農民特權。2009 年是格里曆中以星期四開始的平年……',
      note: '基礎模型只是在「補完句子」，還混入常見的網路中繼資料——內容語無倫次，因為它從未學過以問答格式回應。' },
    rlhf: { btn: 'RLHF 後（對齊）', model: 'Tülu 3 405B（完整後訓練）',
      text: 'George W. Bush（喬治·W·布希）是 2006 年的美國總統。他在任兩屆，任期自 2001 年 1 月 20 日至 2009 年 1 月 20 日。',
      note: '對齊模型簡潔、直接地回答同一個提示。這種風格上的轉變，正是 RLHF 與其他後訓練方法帶來的結果。' }
  };

  function svgEl(tag, attrs) {
    var n = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }

  function ensureStyle() {
    if (document.getElementById('ch01w-style')) return;
    var st = document.createElement('style');
    st.id = 'ch01w-style';
    st.textContent = '@keyframes ch01w-dash{to{stroke-dashoffset:-14;}}' +
      '.ch01w-flow{stroke-dasharray:7 7;animation:ch01w-dash .55s linear infinite;}';
    document.head.appendChild(st);
  }

  function render(rootEl) {
    ensureStyle();
    var state = { sel: DEFAULT_SEL, playing: false, timer: null, compare: 'base' };

    // --- 播放控制列 ---
    var ctrlRow = document.createElement('div');
    ctrlRow.className = 'widget-row';
    ctrlRow.style.cssText = 'margin-bottom:.6rem;align-items:center;gap:.75rem;';
    var playBtn = document.createElement('button');
    playBtn.textContent = '播放整條管線';
    var hint = document.createElement('span');
    hint.style.cssText = 'color:var(--fg-muted);font-size:.85rem;';
    hint.textContent = '或直接點擊圖中的節點與箭頭';
    ctrlRow.appendChild(playBtn); ctrlRow.appendChild(hint);

    // --- SVG 流程圖（呼應書中圖 1）---
    var svgWrap = document.createElement('div');
    svgWrap.style.cssText = 'overflow-x:auto;padding-bottom:.25rem;';
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'RLHF 三步驟管線流程圖' });
    svg.style.cssText = 'display:block;width:100%;min-width:640px;height:auto;';
    svgWrap.appendChild(svg);

    var refs = ITEMS.map(function (item, i) {
      var g = svgEl('g', {});
      g.style.cursor = 'pointer';
      var r = {};
      if (item.kind === 'node') {
        var nx = 6 + (i / 2) * (NW + GAP);
        r.rect = svgEl('rect', { x: nx, y: NY, width: NW, height: NH, rx: 10, fill: 'var(--panel-2)', stroke: 'var(--border)', 'stroke-width': '1.5' });
        var t1 = svgEl('text', { x: nx + NW / 2, y: NY + 20, 'text-anchor': 'middle', 'font-size': '12.5', 'font-weight': '700', fill: 'var(--fg)' });
        t1.textContent = item.name;
        var t2 = svgEl('text', { x: nx + NW / 2, y: NY + 36, 'text-anchor': 'middle', 'font-size': '9', fill: 'var(--fg-muted)' });
        t2.textContent = item.en;
        g.appendChild(r.rect); g.appendChild(t1); g.appendChild(t2);
      } else {
        var x0 = 6 + ((i - 1) / 2) * (NW + GAP) + NW, x1 = x0 + GAP, mid = (x0 + x1) / 2;
        r.line = svgEl('line', { x1: x0 + 3, y1: CY, x2: x1 - 10, y2: CY, stroke: 'var(--fg-muted)', 'stroke-width': '2' });
        r.head = svgEl('polygon', { points: (x1 - 10) + ',' + (CY - 4.5) + ' ' + (x1 - 1) + ',' + CY + ' ' + (x1 - 10) + ',' + (CY + 4.5), fill: 'var(--fg-muted)' });
        r.label = svgEl('text', { x: mid, y: 23, 'text-anchor': 'middle', 'font-size': '10', fill: 'var(--fg-muted)' });
        r.label.textContent = item.name;
        var hit = svgEl('rect', { x: x0, y: 12, width: GAP, height: NH + 28, fill: 'transparent' });
        g.appendChild(r.line); g.appendChild(r.head); g.appendChild(r.label); g.appendChild(hit);
      }
      var tip = svgEl('title', {});
      tip.textContent = item.name;
      g.appendChild(tip);
      g.addEventListener('click', function () { stopPlay(); state.sel = i; update(); });
      svg.appendChild(g);
      return r;
    });

    // --- 詳情卡 ---
    var card = document.createElement('div');
    card.className = 'widget-panel';
    var badge = document.createElement('span');
    badge.style.cssText = 'display:inline-block;padding:.1em .7em;border-radius:999px;font-size:.75rem;border:1px solid;margin-bottom:.45rem;';
    var heading = document.createElement('div');
    heading.style.cssText = 'font-weight:700;font-size:1.05rem;color:var(--fg);margin-bottom:.35rem;';
    var what = document.createElement('p');
    what.style.cssText = 'margin:0 0 .55rem;line-height:1.75;color:var(--fg);';
    var fields = document.createElement('div');
    fields.style.cssText = 'font-size:.9rem;line-height:1.9;color:var(--fg);';
    var fieldEls = ['輸入與量級', '產物', '對應章節'].map(function (lbl) {
      var row = document.createElement('div');
      var tag = document.createElement('span');
      tag.textContent = lbl;
      tag.style.cssText = 'display:inline-block;min-width:5.5em;color:var(--fg-muted);';
      var val = document.createElement('span');
      row.appendChild(tag); row.appendChild(val); fields.appendChild(row);
      return val;
    });
    var insight = document.createElement('p'); // 動態解讀
    insight.style.cssText = 'margin:.65rem 0 0;padding:.5em .8em;border-left:3px solid var(--accent);background:var(--accent-soft);border-radius:0 8px 8px 0;font-size:.9rem;line-height:1.7;color:var(--fg);';
    card.appendChild(badge); card.appendChild(heading); card.appendChild(what); card.appendChild(fields); card.appendChild(insight);

    // --- Base vs RLHF 後 對照卡 ---
    var cmp = document.createElement('div');
    cmp.className = 'widget-panel';
    cmp.style.marginTop = '.75rem';
    var cmpTitle = document.createElement('div');
    cmpTitle.style.cssText = 'font-weight:700;color:var(--fg);margin-bottom:.5rem;';
    cmpTitle.textContent = 'RLHF 做了什麼？同一提示的兩種回答';
    var promptBox = document.createElement('div');
    promptBox.style.cssText = 'background:var(--code-bg);border:1px solid var(--border);border-radius:8px;padding:.5em .8em;font-size:.85rem;color:var(--fg);margin-bottom:.6rem;';
    promptBox.textContent = '提示詞：' + COMPARE.prompt;
    var cmpRow = document.createElement('div');
    cmpRow.className = 'widget-row';
    cmpRow.style.cssText = 'gap:.5rem;margin-bottom:.6rem;';
    var cmpBtns = {};
    ['base', 'rlhf'].forEach(function (k) {
      var b = document.createElement('button');
      b.textContent = COMPARE[k].btn;
      b.addEventListener('click', function () { state.compare = k; update(); });
      cmpBtns[k] = b; cmpRow.appendChild(b);
    });
    var modelTag = document.createElement('div');
    modelTag.style.cssText = 'font-size:.8rem;color:var(--fg-muted);margin-bottom:.35rem;';
    var output = document.createElement('div');
    output.style.cssText = 'background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:.7em .9em;font-size:.88rem;line-height:1.8;color:var(--fg);min-height:6.5em;';
    var cmpNote = document.createElement('p');
    cmpNote.style.cssText = 'margin:.55rem 0 0;font-size:.85rem;color:var(--fg-muted);line-height:1.7;';
    cmp.appendChild(cmpTitle); cmp.appendChild(promptBox); cmp.appendChild(cmpRow);
    cmp.appendChild(modelTag); cmp.appendChild(output); cmp.appendChild(cmpNote);

    // --- 播放（每步約 1.5 秒，箭頭有流動動畫）---
    function stopPlay() {
      state.playing = false;
      if (state.timer) { clearTimeout(state.timer); state.timer = null; }
      playBtn.textContent = '播放整條管線';
    }
    function playStep(i) {
      if (!state.playing || !rootEl.isConnected) return stopPlay();
      if (i >= ITEMS.length) { stopPlay(); update(); return; }
      state.sel = i;
      update();
      state.timer = setTimeout(function () { playStep(i + 1); }, ITEMS[i].kind === 'edge' ? 1100 : 1500);
    }
    playBtn.addEventListener('click', function () {
      if (state.playing) { stopPlay(); update(); return; }
      state.playing = true;
      playBtn.textContent = '停止播放';
      playStep(0);
    });

    function update() {
      ITEMS.forEach(function (item, i) {
        var sel = i === state.sel, r = refs[i];
        if (item.kind === 'node') {
          r.rect.setAttribute('fill', sel ? 'var(--accent-soft)' : 'var(--panel-2)');
          r.rect.setAttribute('stroke', sel ? 'var(--accent)' : 'var(--border)');
          r.rect.setAttribute('stroke-width', sel ? '2.5' : '1.5');
        } else {
          var c = sel ? 'var(--accent)' : 'var(--fg-muted)';
          r.line.setAttribute('stroke', c);
          r.line.setAttribute('stroke-width', sel ? '3' : '2');
          r.head.setAttribute('fill', c);
          r.label.setAttribute('fill', sel ? 'var(--accent)' : 'var(--fg-muted)');
          r.line.setAttribute('class', sel && state.playing ? 'ch01w-flow' : '');
        }
      });
      var item = ITEMS[state.sel];
      var isNode = item.kind === 'node';
      var color = isNode ? 'var(--accent)' : 'var(--accent-2)';
      badge.textContent = isNode ? '階段' : '資料流';
      badge.style.color = color; badge.style.borderColor = color;
      heading.textContent = item.name + (item.en ? '　' + item.en : '');
      what.textContent = item.what;
      fieldEls[0].textContent = item.input;
      fieldEls[1].textContent = item.output;
      fieldEls[2].textContent = item.chapter;
      insight.textContent = '解讀：' + item.insight;
      ['base', 'rlhf'].forEach(function (k) {
        var active = state.compare === k;
        cmpBtns[k].style.outline = active ? '2px solid var(--accent)' : '';
        cmpBtns[k].style.background = active ? 'var(--accent-soft)' : '';
        cmpBtns[k].setAttribute('aria-pressed', active);
      });
      var cData = COMPARE[state.compare];
      modelTag.textContent = '模型：' + cData.model;
      output.textContent = cData.text;
      cmpNote.textContent = cData.note;
    }

    rootEl.appendChild(ctrlRow);
    rootEl.appendChild(svgWrap);
    rootEl.appendChild(card);
    rootEl.appendChild(cmp);
    update();
  }

  window.ChapterWidget = {
    title: 'RLHF 三步驟互動管線',
    intro: '點擊管線中的節點或箭頭，查看每個階段做什麼、輸入資料的量級、產物與對應章節；按「播放整條管線」看資料如何從基礎模型一路流向對齊模型，再用下方對照卡感受 RLHF 前後的差異。',
    render: render
  };
})();
