/* 附錄 A 互動元件：RLHF 詞彙抽認卡 */
(function () {
  'use strict';

  var TERMS = [
    { zh: 'KL 散度', en: 'Kullback-Leibler Divergence', cat: 'ML',
      def: '衡量定義在同一機率空間上的兩個機率分布 $P$ 與 $Q$ 之間差異的度量：$\\mathcal{D}_{\\mathrm{KL}}(P\\|Q)=\\sum_{x}P(x)\\log\\left(\\frac{P(x)}{Q(x)}\\right)$。' },
    { zh: '蒸餾', en: 'Distillation', cat: 'ML',
      def: '以較強模型的輸出來訓練另一個模型的通用做法，是一種已知能打造強大小型模型的合成資料類型。此詞如今用法已超載，與機器學習文獻中的特定技術定義有所出入。' },
    { zh: '（師生式）知識蒸餾', en: 'Knowledge Distillation', cat: 'ML',
      def: '蒸餾的特定型態，也是此術語的起源：透過修改損失函數，讓學生模型 $P_\\theta$ 從教師模型 $P_\\phi$ 在多個潛在詞元／對數機率上的分布中學習，而非直接學習單一選定的輸出。' },
    { zh: '合成資料', en: 'Synthetic Data', cat: 'ML',
      def: '任何由另一個 AI 系統的輸出所構成的訓練資料，範圍從模型針對開放式提示詞生成的文字，到模型對既有內容的改寫。' },
    { zh: '提示詞', en: 'Prompt', cat: 'NLP',
      def: '提供給語言模型、用以生成回應或補全的輸入文字，記作 $x$。' },
    { zh: '補全', en: 'Completion', cat: 'NLP',
      def: '語言模型針對某個提示詞所生成的輸出文字，常記作 $y \\mid x$；獎勵值常以 $r(y \\mid x)$ 的形式計算。' },
    { zh: '被選補全', en: 'Chosen Completion', cat: 'NLP',
      def: '相較於其他候選項而被挑選或偏好的補全，記作 $y_c$，也常記作 $y_{chosen}$。' },
    { zh: '被拒補全', en: 'Rejected Completion', cat: 'NLP',
      def: '在成對比較（pairwise）情境中不受偏好的那個補全，記作 $y_r$。' },
    { zh: '偏好關係', en: 'Preference Relation', cat: 'NLP',
      def: '表示某一補全優於另一補全的符號，例如 $y_{chosen} \\succ y_{rejected}$；獎勵模型會預測此關係成立的機率 $P(y_c \\succ y_r \\mid x)$。' },
    { zh: '思維鏈', en: 'Chain-of-Thought (CoT)', cat: 'NLP',
      def: '語言模型的一種特定行為，指模型被引導以逐步拆解問題的形式作答，最初的版本透過提示詞「Let’s think step by step」實現。' },
    { zh: '情境內學習', en: 'In-context Learning (ICL)', cat: 'NLP',
      def: '利用語言模型上下文視窗內任何資訊（通常是加到提示詞中的內容）的學習方式；最簡單的形式是在提示詞前加入形式相似的範例。' },
    { zh: '策略', en: 'Policy', cat: 'RL',
      def: '以 $\\theta$ 為參數、定義在所有可能補全上的機率分布 $\\pi_\\theta(y \\mid x)$；在強化學習中是代理人決定動作的規則 $\\pi(a \\mid s)$，RLHF 中也稱策略模型。' },
    { zh: '獎勵', en: 'Reward', cat: 'RL',
      def: '表示某個動作或狀態之可取程度（desirability）的純量值，通常記作 $r$。' },
    { zh: '優勢函數', en: 'Advantage Function', cat: 'RL',
      def: '量化在狀態 $s$ 下採取動作 $a$ 相較於平均動作的相對效益：$A(s,a)=Q(s,a)-V(s)$，可依附於特定策略記作 $A^\\pi(s,a)$。' },
    { zh: '價值函數', en: 'Value Function', cat: 'RL',
      def: '估計從給定狀態出發的期望累積獎勵的函數：$V(s)=\\mathbb{E}\\left[\\sum_{t=0}^{\\infty}\\gamma^t r_t \\mid s_0=s\\right]$。' },
    { zh: 'Q 函數', en: 'Q-Function', cat: 'RL',
      def: '估計在給定狀態下採取特定動作後的期望累積獎勵：$Q(s,a)=\\mathbb{E}\\left[\\sum_{t=0}^{\\infty}\\gamma^t r_t \\mid s_0=s, a_0=a\\right]$。' },
    { zh: '軌跡', en: 'Trajectory', cat: 'RL',
      def: '代理人所經歷的一連串狀態、動作與獎勵：$\\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\\ldots,s_T,a_T,r_T)$。' },
    { zh: '折扣因子', en: 'Discount Factor', cat: 'RL',
      def: '滿足 $0 \\le \\gamma < 1$ 的純量，用來在回報中對未來獎勵進行指數式降權，在即時性與長期收益之間取得權衡，並保證無限時域加總的收斂性。' },
    { zh: '同策略', en: 'On-policy', cat: 'RL',
      def: '強化學習中指資料恰好由代理人當前形式的模型所生成；在偏好微調文獻中意義被放寬為「由該版本模型生成」，例如偏好微調前的指令微調檢查點。' },
    { zh: '參考模型', en: 'Reference Model', cat: 'RLHF',
      def: 'RLHF 中使用的一組保存下來的參數 $\\pi_{\\mathrm{ref}}$，其輸出被用來對最佳化過程進行正則化（regularize）。' }
  ];
  var CATS = ['ML', 'NLP', 'RL', 'RLHF'];

  var CSS = '' +
    '.appaw-tabs{display:flex;gap:8px;margin-bottom:14px;}' +
    '.appaw-tabs button.is-active{background:var(--accent-soft);border-color:var(--accent);color:var(--fg);}' +
    '.appaw-list{display:grid;gap:10px;}' +
    '.appaw-card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 14px;}' +
    '.appaw-card h4{margin:0 0 6px;color:var(--fg);font-size:1rem;}' +
    '.appaw-en{color:var(--fg-muted);font-weight:normal;font-size:.85em;margin-left:6px;}' +
    '.appaw-badge{float:right;font-size:.72rem;padding:2px 8px;border-radius:999px;background:var(--accent-soft);color:var(--accent-2);border:1px solid var(--border);}' +
    '.appaw-def{color:var(--fg-muted);font-size:.92rem;line-height:1.7;}' +
    '.appaw-empty{color:var(--fg-muted);text-align:center;padding:24px 0;}' +
    '.appaw-flip{perspective:1000px;cursor:pointer;margin:14px 0;}' +
    '.appaw-inner{position:relative;height:250px;transition:transform .45s ease;transform-style:preserve-3d;}' +
    '.appaw-flip.is-flipped .appaw-inner{transform:rotateY(180deg);}' +
    '.appaw-face{position:absolute;inset:0;-webkit-backface-visibility:hidden;backface-visibility:hidden;border:1px solid var(--border);border-radius:14px;padding:20px;}' +
    '.appaw-front{background:var(--panel);display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;}' +
    '.appaw-front .appaw-zh{font-size:1.6rem;font-weight:700;color:var(--fg);}' +
    '.appaw-front .appaw-en2{margin-top:8px;color:var(--fg-muted);}' +
    '.appaw-back{background:var(--accent-soft);transform:rotateY(180deg);overflow:auto;display:flex;}' +
    '.appaw-back .appaw-def{margin:auto;color:var(--fg);text-align:center;}' +
    '.appaw-hint{text-align:center;color:var(--fg-muted);font-size:.82rem;margin-top:10px;}' +
    '.appaw-nav{justify-content:center;margin-top:12px;}' +
    '.appaw-prog{color:var(--fg-muted);min-width:64px;text-align:center;}';

  /* 將含 $...$ 行內式的字串渲染進元素 */
  function renderRich(el, text) {
    el.textContent = '';
    var parts = text.split(/\$([^$]+)\$/g);
    for (var i = 0; i < parts.length; i++) {
      if (i % 2 === 1 && window.katex) {
        var span = document.createElement('span');
        window.katex.render(parts[i], span, { throwOnError: false });
        el.appendChild(span);
      } else if (parts[i]) {
        el.appendChild(document.createTextNode(parts[i]));
      }
    }
  }

  function el(tag, cls, parent) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (parent) parent.appendChild(node);
    return node;
  }

  function render(rootEl) {
    var state = { mode: 'browse', query: '', cat: 'all', order: TERMS.map(function (_, i) { return i; }), idx: 0, flipped: false };

    var style = document.createElement('style');
    style.textContent = CSS;
    rootEl.appendChild(style);

    var tabs = el('div', 'appaw-tabs', rootEl);
    var btnBrowse = el('button', '', tabs); btnBrowse.type = 'button'; btnBrowse.textContent = '瀏覽';
    var btnFlash = el('button', '', tabs); btnFlash.type = 'button'; btnFlash.textContent = '抽認卡';
    var body = el('div', '', rootEl);

    function setMode(mode) {
      state.mode = mode;
      btnBrowse.classList.toggle('is-active', mode === 'browse');
      btnFlash.classList.toggle('is-active', mode === 'flash');
      body.textContent = '';
      if (mode === 'browse') renderBrowse(); else renderFlash();
    }
    btnBrowse.addEventListener('click', function () { setMode('browse'); });
    btnFlash.addEventListener('click', function () { setMode('flash'); });

    /* ── 模式一：瀏覽 ── */
    function renderBrowse() {
      var bar = el('div', 'widget-row', body);
      var search = el('input', '', bar);
      search.type = 'text'; search.placeholder = '搜尋中英名稱或定義…'; search.value = state.query;
      search.style.flex = '1'; search.style.minWidth = '160px';
      var sel = el('select', '', bar);
      var optAll = el('option', '', sel); optAll.value = 'all'; optAll.textContent = '全部分類';
      CATS.forEach(function (c) {
        var o = el('option', '', sel); o.value = c; o.textContent = c;
      });
      sel.value = state.cat;
      var list = el('div', 'appaw-list', body);
      list.style.marginTop = '12px';

      function refresh() {
        list.textContent = '';
        var q = state.query.trim().toLowerCase();
        var shown = TERMS.filter(function (t) {
          if (state.cat !== 'all' && t.cat !== state.cat) return false;
          if (!q) return true;
          return (t.zh + ' ' + t.en + ' ' + t.def).toLowerCase().indexOf(q) !== -1;
        });
        if (!shown.length) {
          el('div', 'appaw-empty', list).textContent = '沒有符合條件的詞彙。';
          return;
        }
        shown.forEach(function (t) {
          var card = el('div', 'appaw-card', list);
          el('span', 'appaw-badge', card).textContent = t.cat;
          var h = el('h4', '', card);
          h.appendChild(document.createTextNode(t.zh));
          el('span', 'appaw-en', h).textContent = t.en;
          renderRich(el('div', 'appaw-def', card), t.def);
        });
      }
      search.addEventListener('input', function () { state.query = search.value; refresh(); });
      sel.addEventListener('change', function () { state.cat = sel.value; refresh(); });
      refresh();
    }

    /* ── 模式二：抽認卡 ── */
    function renderFlash() {
      var flip = el('div', 'appaw-flip', body);
      flip.setAttribute('role', 'button');
      flip.tabIndex = 0;
      var inner = el('div', 'appaw-inner', flip);
      var front = el('div', 'appaw-face appaw-front', inner);
      var back = el('div', 'appaw-face appaw-back', inner);
      var backDef = el('div', 'appaw-def', back);
      el('div', 'appaw-hint', body).textContent = '點擊卡片可翻面查看定義';
      var nav = el('div', 'widget-row appaw-nav', body);
      var prev = el('button', '', nav); prev.type = 'button'; prev.textContent = '上一張';
      var prog = el('span', 'appaw-prog', nav);
      var next = el('button', '', nav); next.type = 'button'; next.textContent = '下一張';
      var shuf = el('button', '', nav); shuf.type = 'button'; shuf.textContent = '隨機洗牌';

      function show() {
        var t = TERMS[state.order[state.idx]];
        front.textContent = '';
        el('div', 'appaw-zh', front).textContent = t.zh;
        el('div', 'appaw-en2', front).textContent = t.en + '　·　' + t.cat;
        renderRich(backDef, t.def);
        flip.classList.toggle('is-flipped', state.flipped);
        prog.textContent = (state.idx + 1) + ' / ' + TERMS.length;
        flip.setAttribute('aria-label', '抽認卡：' + t.zh + '（點擊翻面）');
      }
      function go(delta) {
        state.idx = (state.idx + delta + TERMS.length) % TERMS.length;
        state.flipped = false;
        show();
      }
      flip.addEventListener('click', function () { state.flipped = !state.flipped; show(); });
      flip.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); state.flipped = !state.flipped; show(); }
      });
      prev.addEventListener('click', function () { go(-1); });
      next.addEventListener('click', function () { go(1); });
      shuf.addEventListener('click', function () {
        for (var i = state.order.length - 1; i > 0; i--) {
          var j = Math.floor(Math.random() * (i + 1));
          var tmp = state.order[i]; state.order[i] = state.order[j]; state.order[j] = tmp;
        }
        state.idx = 0; state.flipped = false; show();
      });
      show();
    }

    setMode('browse');
  }

  window.ChapterWidget = {
    title: 'RLHF 詞彙抽認卡',
    intro: '收錄附錄 A 的 20 條核心詞彙。可在「瀏覽」模式搜尋與依分類篩選，或切換到「抽認卡」模式逐張翻面複習中英術語與定義。',
    render: render
  };
})();
