# 附錄 A　定義（Definitions）

> 譯自 Nathan Lambert, *Reinforcement Learning from Human Feedback*（rlhfbook.com），2026-07-01 版，原文第 227–230 頁。

本附錄收錄 RLHF 流程中經常使用的所有定義、符號與運算，並簡要概述語言模型——語言模型是貫穿本書的核心應用。

## A.1 語言建模總覽（Language Modeling Overview）

大多數現代語言模型的訓練目標，是以自迴歸（autoregressive）方式學習詞元（token；可能是單詞、子詞或字元）序列的聯合機率分布。所謂自迴歸，就是指每一個下一步的預測都取決於序列中先前的元素。給定詞元序列 $x = (x_1, x_2, \ldots, x_T)$，模型會將整個序列的機率分解為一連串條件分布的乘積：

$$
P_\theta(x) = \prod_{t=1}^{T} P_\theta(x_t \mid x_1, \ldots, x_{t-1}). \tag{153}
$$

為了訓練出能準確預測此分布的模型，目標通常是最大化當前模型對訓練資料的預測似然（likelihood）。為此，我們可以最小化負對數似然（negative log-likelihood, NLL）損失：

$$
\mathcal{L}_{\mathrm{LM}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \log P_\theta\left(x_t \mid x_{<t}\right) \right]. \tag{154}
$$

實務上，我們會對每一個下一詞元（next-token）預測使用交叉熵（cross-entropy）損失，計算方式是將序列中的真實詞元與模型的預測結果進行比較。

語言模型有許多不同架構，在知識、速度與其他效能特性上各有取捨。現代語言模型（包括 ChatGPT、Claude、Gemini 等）最常使用**僅解碼器 Transformer（decoder-only Transformers）** [404]。Transformer 的核心創新在於大量運用**自注意力（self-attention）** [405] 機制，讓模型能直接關注（attend to）上下文中的概念並學習複雜的映射。在本書中，特別是第 5 章討論獎勵模型時，我們會探討如何為 Transformer 加入新的頭（head），或修改其語言建模（language modeling, LM）頭。LM 頭是最後一層線性投影層，負責將模型的內部嵌入空間映射到分詞器空間（亦即詞彙表，vocabulary）。本書會展示語言模型不同的「頭」可以用來將模型微調至不同用途——在 RLHF 中最常見的情境就是訓練獎勵模型，這在第 5 章有重點說明。

## A.2 機器學習（Machine Learning）

- **Kullback-Leibler 散度（Kullback-Leibler (KL) divergence，$\mathcal{D}_{\mathrm{KL}}(P\|Q)$）**：又稱 KL 散度（KL divergence），是衡量兩個機率分布之間差異的度量。對於定義在同一機率空間 $\mathcal{X}$ 上的離散機率分布 $P$ 與 $Q$，從 $Q$ 到 $P$ 的 KL 距離定義為：

$$
\mathcal{D}_{\mathrm{KL}}(P\|Q) = \sum_{x \in \mathcal{X}} P(x) \log \left( \frac{P(x)}{Q(x)} \right) \tag{155}
$$

## A.3 自然語言處理（Natural Language Processing）

- **被選補全（Chosen Completion，$y_c$）**：相較於其他候選項而被挑選或偏好的補全，也常記作 $y_{chosen}$。
- **補全（Completion，$y$）**：語言模型針對某個提示詞所生成的輸出文字。補全常記作 $y \mid x$。獎勵值與其他數值常以 $r(y \mid x)$ 或 $P(y \mid x)$ 的形式計算。
- **策略（Policy，$\pi$）**：以 $\theta$ 為參數、定義在所有可能補全上的機率分布：$\pi_\theta(y \mid x)$。
- **偏好關係（Preference Relation，$\succ$）**：表示某一補全優於另一補全的符號，例如 $y_{chosen} \succ y_{rejected}$。舉例來說，獎勵模型會預測偏好關係成立的機率 $P(y_c \succ y_r \mid x)$。
- **提示詞（Prompt，$x$）**：提供給語言模型、用以生成回應或補全的輸入文字。
- **被拒補全（Rejected Completion，$y_r$）**：在成對比較（pairwise）情境中不受偏好的那個補全。

## A.4 強化學習（Reinforcement Learning）

- **動作（Action，$a$）**：代理人（agent）在環境中做出的決策或行動，常記作 $a \in A$，其中 $A$ 是所有可能動作的集合。
- **優勢函數（Advantage Function，$A$）**：優勢函數 $A(s, a)$ 量化在狀態 $s$ 下採取動作 $a$，相較於平均動作所帶來的相對效益。其定義為 $A(s, a) = Q(s, a) - V(s)$。優勢函數（以及價值函數）可以依附於特定策略，記作 $A^\pi(s, a)$。
- **折扣因子（Discount Factor，$\gamma$）**：一個滿足 $0 \le \gamma < 1$ 的純量，用來在回報（return）中對未來獎勵進行指數式降權，在「即時性」與「長期收益」之間取得權衡，並保證無限時域（infinite-horizon）加總的收斂性。有時也會不使用折扣，等價於 $\gamma = 1$。
- **獎勵最佳化的期望值（Expectation of Reward Optimization）**：強化學習的首要目標，即最大化期望累積獎勵：

$$
\max_\theta \, \mathbb{E}_{s \sim \rho_\pi,\, a \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right] \tag{156}
$$

  其中 $\rho_\pi$ 是策略 $\pi$ 之下的狀態分布，$\gamma$ 是折扣因子。

- **有限時域獎勵（Finite Horizon Reward，$J(\pi_\theta)$）**：以 $\theta$ 為參數的策略 $\pi_\theta$ 的期望有限時域折扣回報，定義為：

$$
J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right] \tag{157}
$$

  其中 $\tau \sim \pi_\theta$ 表示依循策略 $\pi_\theta$ 取樣得到的軌跡，$T$ 是有限時域（finite horizon）。

- **同策略（On-policy）**：在 RLHF 中，特別是在強化學習與直接對齊演算法（Direct Alignment Algorithms）的論辯裡，經常討論**同策略（on-policy）**資料。在強化學習文獻中，on-policy 指資料*恰好*由代理人當前形式的模型所生成；但在一般的偏好微調（preference-tuning）文獻中，on-policy 的意義被擴大為「由該版本模型生成」——例如在進行任何偏好微調之前的指令微調檢查點（instruction-tuned checkpoint）。在此脈絡下，off-policy 可以指由後訓練（post-training）過程中所使用的任何其他語言模型生成的資料。
- **策略（Policy，$\pi$）**：在 RLHF 中也稱為**策略模型（policy model）**。在強化學習中，策略是代理人用來決定在給定狀態下採取何種動作的策略或規則：$\pi(a \mid s)$。
- **依策略而定的數值（Policy-conditioned Values，$[]^{\pi(\cdot)}$）**：在各種強化學習的推導與實作中，一個關鍵環節是收集依附於（conditioned on）特定策略的資料或數值。本書中我們會在較簡潔的價值函數記號（$V, A, Q, G$）與其對應的策略條件版本（$V^\pi, A^\pi, Q^\pi$）之間切換。此外，在計算期望值時，另一個關鍵是從依附於特定策略的資料 $d$ 中取樣，記作 $d_\pi$（例如在估計 $\mathbb{E}_{s \sim d_\pi,\, a \sim \pi(\cdot \mid s)}[A^\pi(s, a)]$ 時，$s \sim d_\pi$ 且 $a \sim \pi(\cdot \mid s)$）。
- **Q 函數（Q-Function，$Q$）**：估計在給定狀態下採取特定動作後，期望累積獎勵的函數：$Q(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a\right]$。
- **獎勵（Reward，$r$）**：表示某個動作或狀態之可取程度（desirability）的純量值，通常記作 $r$。
- **狀態（State，$s$）**：環境當前的組態或情境，通常記作 $s \in S$，其中 $S$ 是狀態空間。
- **軌跡（Trajectory，$\tau$）**：軌跡 $\tau$ 是代理人所經歷的一連串狀態、動作與獎勵：$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$。
- **軌跡分布（Trajectory Distribution，$(\tau \mid \pi)$）**：在策略 $\pi$ 之下某條軌跡出現的機率為 $P(\tau \mid \pi) = p(s_0) \prod_{t=0}^{T} \pi(a_t \mid s_t)\, p(s_{t+1} \mid s_t, a_t)$，其中 $p(s_0)$ 是初始狀態分布（prior state distribution），$p(s_{t+1} \mid s_t, a_t)$ 是狀態轉移機率。
- **價值函數（Value Function，$V$）**：估計從給定狀態出發的期望累積獎勵的函數：$V(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]$。

## A.5 RLHF 專屬（RLHF-Only）

- **參考模型（Reference Model，$\pi_{\mathrm{ref}}$）**：RLHF 中使用的一組保存下來的參數，其輸出被用來對最佳化過程進行正則化（regularize）。

## A.6 延伸詞彙表（Extended Glossary）

- **思維鏈（Chain-of-Thought, CoT）**：思維鏈是語言模型的一種特定行為，指模型被引導以逐步拆解問題的形式作答。其最初的版本是透過提示詞「Let's think step by step」（讓我們一步一步思考）[374] 實現的。
- **蒸餾（Distillation）**：蒸餾是訓練 AI 模型的一類通用做法，指以較強模型的輸出來訓練另一個模型。這是一種已知能打造強大小型模型的合成資料類型。大多數模型會明確規範蒸餾相關的規則：開放權重（open-weight）模型透過授權條款，僅能經由 API 存取的模型則透過服務條款。「蒸餾」一詞如今的用法已經超載（overloaded），與機器學習文獻中的特定技術定義有所出入。
- **情境內學習（In-context Learning, ICL）**：這裡的「in-context」指語言模型上下文視窗（context window）內的任何資訊，通常是加到提示詞中的內容。情境內學習最簡單的形式是在提示詞前加入形式相似的範例；進階版本則能針對特定使用情境學習該納入哪些資訊。
- **（師生式）知識蒸餾（(Teacher-student) Knowledge Distillation）**：從特定教師模型到學生模型的知識蒸餾是上述蒸餾的一種特定型態，也是此術語的起源。它是一種特定的深度學習方法，透過修改神經網路損失函數，讓學生模型從教師模型在多個潛在詞元／對數機率（logits）上的對數機率中學習，而不是直接從單一選定的輸出學習 [276]。以知識蒸餾訓練的現代模型系列，例如 Gemma 2 [406] 或 Gemma 3。在語言建模的設定下，下一詞元損失函數可修改如下 [287]，其中學生模型 $P_\theta$ 向教師分布 $P_\phi$ 學習：

$$
\mathcal{L}_{\mathrm{KD}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{t=1}^{T} P_\phi(x_t \mid x_{<t}) \log P_\theta(x_t \mid x_{<t}) \right]. \tag{158}
$$

- **合成資料（Synthetic Data）**：指任何由另一個 AI 系統的輸出所構成的 AI 模型訓練資料。範圍很廣，從模型針對開放式提示詞生成的文字，到模型對既有內容的改寫，都屬於合成資料。
