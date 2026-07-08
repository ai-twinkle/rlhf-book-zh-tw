/* 第 11 章互動元件：當一次偏好標註員 */
(function () {
  'use strict';

  var QUESTIONS = [
    {
      bias: '陷阱一：長度與格式偏誤',
      prompt: '請簡單介紹光合作用的基本過程。',
      a: '光合作用是植物將光能轉換為化學能的重要生化反應，可整理為以下幾個重點：' +
         '<ul><li><strong>發生場所</strong>：植物細胞內的葉綠體</li>' +
         '<li><strong>反應原料</strong>：二氧化碳與水</li>' +
         '<li><strong>主要產物</strong>：葡萄糖與<strong>氮氣</strong></li>' +
         '<li><strong>能量來源</strong>：太陽光</li></ul>' +
         '整體而言，光合作用是地球生態系能量流動的起點，重要性不言可喻。',
      b: '光合作用是植物利用陽光，在葉綠體中把二氧化碳和水轉化成葡萄糖、並釋放氧氣的過程。',
      majority: 'A', pct: 64,
      explain: 'A 排版精美、看起來更「專業」，卻把產物中的「氧氣」寫成了「氮氣」；B 雖然只有一句話，內容完全正確。' +
        '研究顯示，標註者會不成比例地偏愛較長、格式漂亮的回覆——這就是冗長偏誤（verbosity）與格式偏誤，也是模型愈訓練愈囉嗦的原因之一。'
    },
    {
      bias: '陷阱二：風格偏好，無標準答案',
      prompt: '同事下週要離職了，幫我寫一句簡短的祝福。',
      a: '一路順風啦！到了新公司要記得照顧好自己，有空常回來找我們喝一杯！',
      b: '祝您鵬程萬里，在新的職涯旅程中一帆風順、大展長才。',
      majority: 'A', pct: 52,
      explain: '兩個回覆都正確完成了任務，差別只在語氣：一個口語親切、一個正式得體。' +
        '這類題目沒有標準答案，52% 對 48% 的分布幾乎等於擲硬幣——你的選擇反映的是個人風格偏好，這正是偏好資料中「合理雜訊」的主要來源。'
    },
    {
      bias: '陷阱三：諂媚與自信偏誤',
      prompt: '維他命 C 可以預防感冒嗎？',
      a: '可以！維他命 C 已被證實能有效預防感冒，每天補充 1000 毫克就能大幅降低感冒機率，這是醫學界的共識。',
      b: '就目前的研究來看，維他命 C 對一般人恐怕沒有預防感冒的效果；部分研究顯示它或許能稍微縮短病程，但證據仍有限。建議再諮詢醫師等專業意見。',
      majority: 'A', pct: 58,
      explain: 'A 語氣篤定、還宣稱「醫學界共識」，讓人覺得可靠，但內容與研究證據不符；B 坦承不確定，方向卻是對的。' +
        '標註者傾向獎勵自信、迎合的語氣，這種諂媚（sycophancy）偏誤一旦被獎勵模型學走，模型就會愈來愈敢「自信地胡說」。'
    },
    {
      bias: '對照組：品質差距明顯',
      prompt: '請用一句話解釋什麼是 RLHF。',
      a: 'RLHF（人類回饋強化學習）是先蒐集人類對模型回覆的偏好、訓練出獎勵模型，再用強化學習讓語言模型往人類偏好的方向調整的技術。',
      b: 'RLHF 就是一種 AI 技術，很多公司都在用，總之就是讓 AI 變得更好、更聰明的方法。',
      majority: 'A', pct: 93,
      explain: '這題是對照組：A 準確交代了「偏好資料、獎勵模型、強化學習」三個環節，B 空洞含糊。' +
        '當品質差距明顯時，標註者一致率會很高；真正花錢又花力氣的，是前三題那種灰色地帶。'
    }
  ];
  var SCALE = ['1 非常差', '2 不佳', '3 尚可', '4 良好', '5 優異'];
  var state = { root: null, idx: 0, answers: [] };

  function el(tag, style, html) {
    var n = document.createElement(tag);
    if (style) n.style.cssText = style;
    if (html !== undefined) n.innerHTML = html;
    return n;
  }
  function chip(text, strong) {
    return el('span', 'display:inline-block;font-size:.78em;padding:.1em .6em;border-radius:999px;' +
      'border:1px solid ' + (strong ? 'var(--accent)' : 'var(--border)') + ';' +
      'background:' + (strong ? 'var(--accent-soft)' : 'var(--panel)') + ';color:var(--fg);', text);
  }
  function promptPanel(text) {
    var p = el('div', 'border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:8px;' +
      'background:var(--panel);padding:.6rem .9rem;margin:.75rem 0;');
    p.appendChild(el('div', 'font-size:.78em;color:var(--fg-muted);margin-bottom:.2rem;', '提示詞'));
    p.appendChild(el('div', 'font-weight:600;', text));
    return p;
  }

  function renderQuestion() {
    var q = QUESTIONS[state.idx];
    var root = state.root;
    root.innerHTML = '';

    var head = el('div', 'display:flex;justify-content:space-between;align-items:center;gap:.5rem;flex-wrap:wrap;');
    head.appendChild(el('strong', '', '第 ' + (state.idx + 1) + ' / ' + QUESTIONS.length + ' 題'));
    var dots = el('div', 'display:flex;gap:.35rem;');
    QUESTIONS.forEach(function (_, i) {
      dots.appendChild(el('span', 'width:.6em;height:.6em;border-radius:50%;display:inline-block;' +
        'background:' + (i < state.idx ? 'var(--accent)' : i === state.idx ? 'var(--accent-2)' : 'var(--border)') + ';'));
    });
    head.appendChild(dots);
    root.appendChild(head);
    root.appendChild(el('p', 'color:var(--fg-muted);margin:.5rem 0 0;font-size:.9em;',
      '請站在一般使用者的角度，點選你認為<strong>較好</strong>的回覆。'));
    root.appendChild(promptPanel(q.prompt));

    var row = el('div', '');
    row.className = 'widget-row';
    row.style.alignItems = 'stretch';
    var feedback = el('div', 'margin-top: .9rem;');
    var cards = {};
    ['A', 'B'].forEach(function (key) {
      var card = el('div', 'flex:1 1 240px;min-width:0;border:1px solid var(--border);border-radius:10px;' +
        'background:var(--panel);padding:.7rem .9rem;cursor:pointer;');
      card.setAttribute('role', 'button');
      card.setAttribute('tabindex', '0');
      card.setAttribute('aria-label', '選擇回覆 ' + key);
      var badgeRow = el('div', 'display:flex;gap:.4rem;align-items:center;flex-wrap:wrap;margin-bottom:.45rem;');
      badgeRow.appendChild(el('strong', 'color:var(--accent);', '回覆 ' + key));
      card.appendChild(badgeRow);
      card.appendChild(el('div', 'font-size:.92em;line-height:1.65;', key === 'A' ? q.a : q.b));
      card.addEventListener('click', function () { choose(key, q, cards, feedback); });
      card.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(key, q, cards, feedback); }
      });
      cards[key] = card;
      row.appendChild(card);
    });
    root.appendChild(row);
    root.appendChild(feedback);
  }

  function choose(key, q, cards, feedback) {
    if (feedback.childNodes.length) return; // 已作答
    var match = key === q.majority;
    state.answers.push({ key: key, match: match });

    ['A', 'B'].forEach(function (k) {
      var c = cards[k];
      c.style.cursor = 'default';
      c.removeAttribute('role');
      c.setAttribute('tabindex', '-1');
      var badges = c.firstChild;
      if (k === key) {
        c.style.borderColor = 'var(--accent)';
        c.style.background = 'var(--accent-soft)';
        badges.appendChild(chip('你的選擇', true));
      }
      if (k === q.majority) badges.appendChild(chip('多數選擇 ' + q.pct + '%', false));
    });

    var panel = el('div', 'border-left:3px solid var(--accent-2);');
    panel.className = 'widget-panel';
    panel.appendChild(el('div', 'margin-bottom:.35rem;',
      '多數標註者（' + q.pct + '%）選了<strong>回覆 ' + q.majority + '</strong>，你與多數' +
      (match ? '<strong>一致</strong>。' : '<strong>不一致</strong>。')));
    panel.appendChild(el('div', 'font-size:.9em;margin-bottom:.5rem;',
      '<strong>' + q.bias + '</strong>　' + q.explain));
    var next = el('button', '', state.idx + 1 < QUESTIONS.length ? '下一題' : '查看總結');
    next.addEventListener('click', function () {
      state.idx += 1;
      if (state.idx < QUESTIONS.length) renderQuestion(); else renderSummary();
    });
    panel.appendChild(next);
    feedback.appendChild(panel);
    if (typeof panel.scrollIntoView === 'function') panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function renderSummary() {
    var root = state.root;
    root.innerHTML = '';
    var hits = state.answers.filter(function (a) { return a.match; }).length;

    root.appendChild(el('h4', 'margin:.2rem 0 .6rem;', '標註結果總結'));
    var panel = el('div', '');
    panel.className = 'widget-panel';
    panel.appendChild(el('div', 'font-size:1.05em;margin-bottom:.5rem;',
      '你與「多數標註者」的一致率：<strong>' + hits + ' / ' + QUESTIONS.length +
      '（' + Math.round(hits / QUESTIONS.length * 100) + '%）</strong>'));
    var list = el('ul', 'margin:.3rem 0 .6rem;padding-left:1.2em;font-size:.9em;');
    QUESTIONS.forEach(function (q, i) {
      var a = state.answers[i];
      list.appendChild(el('li', 'margin:.2em 0;', '第 ' + (i + 1) + ' 題（' + q.bias.replace(/^[^：]+：/, '') +
        '）：你選 ' + a.key + '，多數選 ' + q.majority + '——' +
        (a.match ? '一致' : '<span style="color:var(--accent-2)">不一致</span>')));
    });
    panel.appendChild(list);
    panel.appendChild(el('p', 'font-size:.9em;color:var(--fg-muted);margin:0;',
      '注意：一致率高不代表判斷正確——第 1、3 題的「多數」其實都掉進了偏誤陷阱。一致率量測的是雜訊，不是真理。'));
    root.appendChild(panel);

    var key = el('div', 'border-left:3px solid var(--accent);margin-top:.9rem;');
    key.className = 'widget-panel';
    key.appendChild(el('div', '', '<strong>書中重點</strong>：即使有詳盡的標註指引，標註者之間的一致率通常也只有 <strong>60～80%</strong>。' +
      '偏好資料記錄的不是「正確答案」，而是「相對於另一個選項哪個較好」的主觀判斷，本質上就是有雜訊的。' +
      '能否辨識並緩解冗長、格式、諂媚這些細微偏誤，正是「好的」與「卓越的」偏好資料——以及 RLHF 訓練——之間的分野。'));
    root.appendChild(key);

    root.appendChild(renderLikert());

    var restartBtn = el('button', 'margin-top:1rem;', '重新開始');
    restartBtn.addEventListener('click', function () { state.idx = 0; state.answers = []; renderQuestion(); });
    root.appendChild(restartBtn);
  }

  /* 附加對照：同一題改用 1～5 評分（ratings）介面 */
  function renderLikert() {
    var q = QUESTIONS[1];
    var box = el('div', 'margin-top:1.2rem;');
    box.appendChild(el('h4', 'margin:.2rem 0 .4rem;', '附加對照：如果改用評分（Ratings）呢？'));
    box.appendChild(el('p', 'font-size:.9em;color:var(--fg-muted);margin:0 0 .5rem;',
      '剛才你做的是<strong>排序（rankings）</strong>：在兩個回覆之間選出相對較好的一個。' +
      '另一種蒐集方式是<strong>評分（ratings）</strong>：不做比較，而是替每段回覆各自打 1～5 分。' +
      '試著替第 2 題（離職祝福）的兩個回覆評分：'));
    box.appendChild(promptPanel(q.prompt));

    var scores = { A: 0, B: 0 };
    var result = el('div', 'margin-top:.6rem;');
    ['A', 'B'].forEach(function (key) {
      var line = el('div', 'border:1px solid var(--border);border-radius:8px;background:var(--panel);' +
        'padding:.55rem .8rem;margin-bottom:.5rem;');
      line.appendChild(el('div', 'font-size:.88em;margin-bottom:.4rem;',
        '<strong style="color:var(--accent)">回覆 ' + key + '</strong>　' + (key === 'A' ? q.a : q.b)));
      var row = el('div', 'display:flex;gap:.35rem;flex-wrap:wrap;');
      var btns = SCALE.map(function (label, i) {
        var b = el('button', 'font-size:.82em;padding:.15em .5em;', label);
        b.setAttribute('aria-pressed', 'false');
        b.addEventListener('click', function () {
          scores[key] = i + 1;
          btns.forEach(function (x, j) {
            var on = j === i;
            x.setAttribute('aria-pressed', String(on));
            x.style.background = on ? 'var(--accent-soft)' : '';
            x.style.borderColor = on ? 'var(--accent)' : '';
          });
          updateLikertResult(scores, result);
        });
        row.appendChild(b);
        return b;
      });
      line.appendChild(row);
      box.appendChild(line);
    });
    box.appendChild(result);
    return box;
  }

  function updateLikertResult(scores, result) {
    if (!scores.A || !scores.B) return;
    result.innerHTML = '';
    var panel = el('div', 'border-left:3px solid var(--accent-2);');
    panel.className = 'widget-panel';
    var msg;
    if (scores.A === scores.B) {
      msg = '兩者同分（各 ' + scores.A + ' 分）——評分制允許平手，這時就湊不出偏好對（chosen / rejected）。';
    } else {
      var hi = scores.A > scores.B ? 'A' : 'B', lo = hi === 'A' ? 'B' : 'A';
      msg = '換算成偏好資料：<strong>' + hi + '（' + scores[hi] + ' 分）為 chosen、' + lo + '（' + scores[lo] +
        ' 分）為 rejected</strong>。UltraFeedback 等資料集正是這樣把評分轉成偏好對：取最高分的回覆，搭配一個分數較低的。';
    }
    panel.appendChild(el('div', 'font-size:.9em;margin-bottom:.4rem;', msg));
    panel.appendChild(el('div', 'font-size:.88em;color:var(--fg-muted);',
      '兩種格式的結構差異：排序強迫你表態相對好壞，訊號明確，但丟失「差多少」；評分保留每段文字的絕對品質資訊，卻可能整批同分。' +
      '實務上最常見的做法，仍是以排序（如 5 點 Likert 比較量表：A≫B、A>B、平手、B>A、B≫A）蒐集訓練用偏好，評分則常作為中繼資料保留。'));
    result.appendChild(panel);
  }

  window.ChapterWidget = {
    title: '當一次偏好標註員',
    intro: '四題偏好比較任務，體驗資料標註員的日常：讀完提示詞與兩個回覆，點選你認為較好的一個，' +
      '再對照多數標註者的選擇——看看其中藏了哪些偏誤陷阱，以及排序與評分兩種蒐集方式的差異。',
    render: function (rootEl) { state.root = rootEl; state.idx = 0; state.answers = []; renderQuestion(); }
  };
})();
