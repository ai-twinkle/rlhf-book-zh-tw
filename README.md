# RLHF 中文版 — 從人類回饋中強化學習（互動版）

> **非官方社群翻譯（Unofficial Community Translation）**
> 本專案為 Nathan Lambert《[Reinforcement Learning from Human Feedback](https://rlhfbook.com)》（2026-07-01 版）的繁體中文（zh-TW）全譯本，由台灣 [Twinkle AI Community](https://github.com/ai-twinkle) 翻譯維護，已獲原作者知悉（[rlhf-book#472](https://github.com/natolambert/rlhf-book/issues/472)）。依 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hant) 授權，僅供學習研究、不得作商業用途。支持原作者請購買[實體書](https://rlhfbook.com)。

**📖 線上閱讀：<https://apps.twinkleai.tw/rlhf-book-zh-tw/>**

除了全書翻譯，每一章都配備一個**互動實驗**，讓讀者能動手操作該章的核心概念——從 Bradley-Terry 偏好機率、PPO 裁剪目標、GRPO 群組優勢，到 DPO 損失曲面與前向／反向 KL 的收斂行為。

## 互動實驗一覽

| 章節 | 實驗 |
|---|---|
| 第 1 章 導論 | RLHF 三步驟互動管線 |
| 第 2 章 RLHF 簡史 | RLHF 發展互動時間軸 |
| 第 3 章 訓練總覽 | 溫控器 RL 模擬器 |
| 第 4 章 指令微調 | 聊天模板建構器（ChatML／Zephyr／Tülu＋損失遮罩） |
| 第 5 章 獎勵模型 | Bradley-Terry 獎勵模型探索器 |
| 第 6 章 強化學習 | PPO 裁剪目標／GRPO 群組優勢遊樂場 |
| 第 7 章 推理與推論時擴展 | pass@k vs 多數決模擬 |
| 第 8 章 直接對齊演算法 | DPO 損失探索器 |
| 第 9 章 拒絕採樣 | 拒絕採樣與 Best-of-N 模擬器 |
| 第 10 章 偏好的本質 | 偏好聚合悖論（Condorcet 循環） |
| 第 11 章 偏好資料 | 當一次偏好標註員（偏誤陷阱體驗） |
| 第 12 章 合成資料與蒸餾 | 知識蒸餾軟標籤實驗室 |
| 第 13 章 工具使用 | 工具呼叫流程模擬器 |
| 第 14 章 過度最佳化 | 獎勵過度最佳化模擬（Goodhart 定律） |
| 第 15 章 正則化 | KL 散度探索器（前向 vs 反向） |
| 第 16 章 評估 | 評估雜訊模擬器 |
| 第 17 章 模型性格 | Persona 向量調音台 |
| 附錄 A／B／C | 詞彙抽認卡／話多平衡體驗器／評測變異查核器 |

## 目錄結構

```
├── content/                     # 逐章翻譯的獨立 Markdown（17 章 + 3 附錄 + 參考文獻）
├── webapp/                      # 互動式網站（純靜態、完全離線可用）
│   ├── index.html               # 目錄首頁
│   ├── chapters/*.html          # 每章一頁（內嵌翻譯內容）
│   └── assets/                  # 樣式、渲染器、互動元件、插圖、本地函式庫
└── build.py                     # 建置腳本：content/*.md → webapp 頁面
```

## 本地使用

```bash
cd webapp && python3 -m http.server 8642   # 瀏覽 http://localhost:8642
```

修改 `content/*.md` 或新增 `webapp/assets/widgets/*.js` 後重新執行 `python3 build.py`。

## 翻譯慣例

- 專有名詞第一次出現採「中文（English）」，常用縮寫（RLHF、PPO、DPO、SFT…）保留英文
- 數學式以 LaTeX 轉錄，公式編號沿用原書（`\tag{n}`）
- 文獻引用標記 [N] 對應 `content/bibliography.md`（保留原文）
- 程式碼區塊保留原文，註解翻譯

發現翻譯問題歡迎開 issue 或 PR。

---

## About (English)

This is an **unofficial Traditional Chinese (zh-TW) community translation** of *[Reinforcement Learning from Human Feedback](https://rlhfbook.com)* by Nathan Lambert (2026-07-01 edition), maintained by the [Twinkle AI Community](https://github.com/ai-twinkle) in Taiwan, acknowledged by the author in [rlhf-book#472](https://github.com/natolambert/rlhf-book/issues/472).

Beyond the full translation, every chapter ships with an **interactive lab** — a hands-on demo of the chapter's core concept (Bradley-Terry explorer, PPO clipping / GRPO group-advantage playground, DPO loss explorer, forward/reverse KL visualizer, and more).

**Read online: <https://apps.twinkleai.tw/rlhf-book-zh-tw/>**

Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), same as the original chapters. Non-commercial; full attribution to the original author. Please support the author by purchasing the [print edition](https://rlhfbook.com).

## Citation

Please cite the original book:

```bibtex
@book{rlhf2026lambert,
  author       = {Nathan Lambert},
  title        = {Reinforcement Learning from Human Feedback},
  year         = {2026},
  publisher    = {Online},
  url          = {https://rlhfbook.com},
}
```
