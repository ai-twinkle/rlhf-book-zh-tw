/* 第 4 章互動元件：聊天模板建構器（Chat Template Playground） */
(function () {
  'use strict';

  function sp(t) { return { t: t, special: true }; }
  function pl(t) { return { t: t, special: false }; }

  // 三種聊天模板的序列化規則（格式依本章翻譯稿）
  var TEMPLATES = {
    chatml: {
      label: 'ChatML',
      note: 'ChatML（Chat Markup Language）源自 OpenAI，是早期標準化訊息格式的嘗試：每則訊息以 <|im_start|>role 開頭、<|im_end|> 收尾。',
      head: function (role) { return [sp('<|im_start|>'), pl(role + '\n')]; },
      tail: function () { return [sp('<|im_end|>'), pl('\n')]; },
      gen: function () { return [sp('<|im_start|>'), pl('assistant\n')]; }
    },
    zephyr: {
      label: 'Zephyr',
      note: 'Zephyr 模板出自 Hugging Face H4 團隊的 Zephyr 模型：以 <|system|>、<|user|>、<|assistant|> 標記角色，每則訊息以 </s> 收尾。',
      head: function (role) { return [sp('<|' + role + '|>'), pl('\n')]; },
      tail: function () { return [sp('</s>'), pl('\n')]; },
      gen: function () { return [sp('<|assistant|>'), pl('\n')]; }
    },
    tulu: {
      label: 'Tülu',
      note: 'Tülu 模板出自 AI2（Allen Institute for AI）的 Tülu 系列：格式極簡，角色標籤後換行直接接內容，僅 assistant 回覆以 <|endoftext|> 收尾。',
      head: function (role) { return [sp('<|' + role + '|>'), pl('\n')]; },
      tail: function (role) { return role === 'assistant' ? [sp('<|endoftext|>'), pl('\n')] : [pl('\n')]; },
      gen: function () { return [sp('<|assistant|>'), pl('\n')]; }
    }
  };

  var state = {
    template: 'chatml',
    showMask: false,
    genPrompt: false,
    system: '你是一位友善的聊天機器人，總是以海盜的口吻回答。',
    rounds: [
      { user: '一個人一餐能吃下幾架直升機？', assistant: '哎呀，就 6 架而已，夥計！' },
      { user: '你確定嗎？', assistant: '哈哈，當然是玩笑話——直升機可吃不得，夥計！' }
    ]
  };

  // 把目前對話狀態序列化成帶標記的片段串列
  function serialize() {
    var tpl = TEMPLATES[state.template];
    var msgs = [];
    if (state.system.trim() !== '') msgs.push({ role: 'system', content: state.system });
    state.rounds.forEach(function (r) {
      msgs.push({ role: 'user', content: r.user });
      msgs.push({ role: 'assistant', content: r.assistant });
    });
    var out = [];
    msgs.forEach(function (m) {
      // 角色開頭標籤屬於提示結構，一律遮罩
      tpl.head(m.role).forEach(function (s) { s.loss = false; out.push(s); });
      // 訊息內容：只有 assistant 回覆計算損失
      out.push({ t: m.content, special: false, loss: m.role === 'assistant' });
      // 結尾詞元跟隨所屬訊息：assistant 的 EOS 也要學（模型才知道何時停）
      tpl.tail(m.role).forEach(function (s) { s.loss = m.role === 'assistant'; out.push(s); });
    });
    if (state.genPrompt) tpl.gen().forEach(function (s) { s.loss = false; out.push(s); });
    return out;
  }

  function el(tag, styleText, text) {
    var n = document.createElement(tag);
    if (styleText) n.style.cssText = styleText;
    if (text != null) n.textContent = text;
    return n;
  }

  var refs = {}; // 持有需要更新的 DOM 節點

  function renderOutput() {
    var pre = refs.pre;
    pre.textContent = '';
    pre.style.background = state.showMask ? 'var(--panel)' : 'var(--code-bg)';
    serialize().forEach(function (s) {
      var span = document.createElement('span');
      span.textContent = s.t;
      if (s.special) span.style.cssText = 'color:var(--accent);font-weight:600;';
      if (state.showMask) {
        span.style.background = s.loss ? 'var(--accent-soft)' : 'var(--border)';
        span.style.borderRadius = '3px';
      }
      pre.appendChild(span);
    });
    refs.legend.style.display = state.showMask ? '' : 'none';
    refs.note.textContent = '模板出處：' + TEMPLATES[state.template].note;
  }

  function textarea(value, onInput) {
    var ta = document.createElement('textarea');
    ta.rows = 2;
    ta.value = value;
    ta.style.cssText = 'width:100%;resize:vertical;';
    ta.addEventListener('input', function () { onInput(ta.value); renderOutput(); });
    return ta;
  }

  function roleLabel(text) {
    return el('div', 'font-size:.78rem;font-weight:600;color:var(--fg-muted);margin:.4rem 0 .2rem;', text);
  }

  // 重建對話編輯區（新增／刪除輪次時呼叫）
  function renderEditor() {
    var box = refs.editor;
    box.textContent = '';
    box.appendChild(roleLabel('system（系統提示，留空則省略）'));
    box.appendChild(textarea(state.system, function (v) { state.system = v; }));

    state.rounds.forEach(function (round, i) {
      var card = el('div', 'border:1px solid var(--border);border-radius:8px;padding:.6rem .8rem;margin-top:.8rem;background:var(--panel);');
      var bar = el('div', 'display:flex;justify-content:space-between;align-items:center;gap:.5rem;');
      bar.appendChild(el('strong', 'font-size:.85rem;', '第 ' + (i + 1) + ' 輪'));
      var del = el('button', 'font-size:.78rem;', '刪除');
      del.disabled = state.rounds.length <= 1;
      del.addEventListener('click', function () {
        state.rounds.splice(i, 1);
        renderEditor();
        renderOutput();
      });
      bar.appendChild(del);
      card.appendChild(bar);
      card.appendChild(roleLabel('user（使用者）'));
      card.appendChild(textarea(round.user, function (v) { round.user = v; }));
      card.appendChild(roleLabel('assistant（助理）'));
      card.appendChild(textarea(round.assistant, function (v) { round.assistant = v; }));
      box.appendChild(card);
    });

    var add = el('button', 'margin-top:.8rem;', '＋ 新增一輪');
    add.addEventListener('click', function () {
      state.rounds.push({ user: '（輸入新的使用者訊息）', assistant: '（輸入新的助理回覆）' });
      renderEditor();
      renderOutput();
    });
    box.appendChild(add);
  }

  function legendChip(bg, text) {
    var wrap = el('span', 'display:inline-flex;align-items:center;gap:.35rem;font-size:.8rem;color:var(--fg-muted);');
    wrap.appendChild(el('span', 'width:.9rem;height:.9rem;border-radius:3px;border:1px solid var(--border);background:' + bg + ';'));
    wrap.appendChild(document.createTextNode(text));
    return wrap;
  }

  function render(rootEl) {
    // 面板一：對話編輯
    var panel1 = el('div', null); panel1.className = 'widget-panel';
    panel1.appendChild(el('h4', 'margin:.1rem 0 .5rem;font-size:.95rem;', '① 編輯對話'));
    refs.editor = el('div', null);
    panel1.appendChild(refs.editor);
    rootEl.appendChild(panel1);

    // 面板二：模板與序列化輸出
    var panel2 = el('div', 'margin-top:1rem;'); panel2.className = 'widget-panel';
    panel2.appendChild(el('h4', 'margin:.1rem 0 .5rem;font-size:.95rem;', '② 套用聊天模板'));

    var row = el('div', null); row.className = 'widget-row';
    var selLabel = el('label', 'display:inline-flex;align-items:center;gap:.4rem;font-size:.88rem;', '模板：');
    var sel = document.createElement('select');
    Object.keys(TEMPLATES).forEach(function (key) {
      var opt = document.createElement('option');
      opt.value = key;
      opt.textContent = TEMPLATES[key].label;
      sel.appendChild(opt);
    });
    sel.value = state.template;
    sel.addEventListener('change', function () { state.template = sel.value; renderOutput(); });
    selLabel.appendChild(sel);
    row.appendChild(selLabel);

    [['showMask', '顯示損失遮罩'], ['genPrompt', '附加生成提示（add_generation_prompt）']].forEach(function (pair) {
      var lab = el('label', 'display:inline-flex;align-items:center;gap:.35rem;font-size:.88rem;cursor:pointer;');
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = state[pair[0]];
      cb.addEventListener('change', function () { state[pair[0]] = cb.checked; renderOutput(); });
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(pair[1]));
      row.appendChild(lab);
    });
    panel2.appendChild(row);

    // 模板出處的一句話解讀
    refs.note = el('p', 'margin:.6rem 0 0;font-size:.85rem;color:var(--fg-muted);');
    panel2.appendChild(refs.note);

    // 序列化輸出
    refs.pre = el('pre', 'margin:.7rem 0 0;padding:.9rem 1rem;border:1px solid var(--border);border-radius:8px;' +
      'font-family:var(--mono,monospace);font-size:.84rem;line-height:1.85;white-space:pre-wrap;word-break:break-word;overflow-x:auto;color:var(--fg);');
    panel2.appendChild(refs.pre);

    // 圖例：損失遮罩說明
    refs.legend = el('div', 'display:flex;flex-wrap:wrap;gap:.8rem;align-items:center;margin-top:.6rem;');
    refs.legend.appendChild(legendChip('var(--border)', 'prompt／user 部分（遮罩，不計損失）'));
    refs.legend.appendChild(legendChip('var(--accent-soft)', 'assistant 回覆（計算損失）'));
    refs.legend.appendChild(el('span', 'font-size:.8rem;color:var(--fg-muted);flex-basis:100%;',
      'SFT 只對助理 token 計算損失（prompt masking）：模型不學著預測使用者查詢，只學習生成回應與結尾詞元。'));
    panel2.appendChild(refs.legend);

    rootEl.appendChild(panel2);
    renderEditor();
    renderOutput();
  }

  window.ChapterWidget = {
    title: '聊天模板建構器',
    intro: '編輯多輪對話並切換 ChatML／Zephyr／Tülu 模板，即時觀察訊息如何被序列化成詞元序列；開啟損失遮罩，看看 SFT 為什麼「只學回應、不學提示」。',
    render: render
  };
})();
