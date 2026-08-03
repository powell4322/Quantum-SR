# 相关工作与防撞对照（RELATED_WORK）

> **用途（采纳 GPT 审稿建议 2026-08-02）**：防止撞 idea——重点确认 **dynamic density state 是否已经有人做过**。这是决定论文能否投的关键核查。
> 方法：按三个方向逐条调研，填"代表工作 / 核心做法 / 是否做了 dynamic density state / 与我们差异"。**未核实的工作不要标记为"已查"。**

---

## 0. 结论（2026-08-03 更新：发现关键邻近先例，需精读）
- 🔴 **重要发现**：DBLP 命中 **"Density Matrix Based Preference Evolution Networks for E-Commerce Recommendation"（DASFAA 2019, DOI 10.1007/978-3-030-18579-4_22）**——标题同时含 **density matrix + preference evolution**，疑似直接先例。**"空白点"结论降级为"待精读确认差异"，不能默认首创。**
- [ ] **待办（写论文前必做）**：获取该文全文（学校 Springer 权限），确认：① 是否序列 / next-item 任务？② 演化机制是否 PSD+trace 保合法凸组合？③ 打分方式？④ 与我们的差异——若无差异，定位从"首次"改为"显著扩展 + 理论强化（合法性 + 序列级 + 新打分）"。
- [x] 其余方向未发现直接先例（arXiv + DBLP 首轮）：density matrix 用于推荐（ConQAR 静态评分、WWW 2026 静态 CF）；Gaussian/distributional（非算子态、无硬约束）；interest drift（向量/RNN 门控）。
- ⚠️ 论文写作前必须先精读 DASFAA 2019 并明确差异；否则有撞 idea 风险。

---

## 1. 方向 A：Quantum / Density-operator Recommendation

| 代表工作 | 核心做法 | 静态 or 动态状态 | 是否覆盖"序列演化" | 与我们差异 |
|---|---|---|---|---|
| 🔴 **DASFAA 2019: Density Matrix Based Preference Evolution Networks**（DOI 10.1007/978-3-030-18579-4_22） | density matrix 表示 + 偏好演化（e-commerce） | **⚠️ 待精读确认是否动态** | **⚠️ 疑似是** | **最需核查**；若为动态 density 演化，定位从"首次"改为"扩展 + 理论强化" |
| **ConQAR**（arXiv 1912.11720, ICTIR'19 WS） | quantum-like **density matrix layer** 捕获卷积特征交互，用于**评论评分预测（rating）** | 静态（用户/物品各一表示） | ❌ 非序列、非动态 | 我们做**序列 next-item + 动态演化**；它是基于文本评论的评分 |
| WWW 2026 *Quantum-enhanced Repr. & Matching for Rec* | 用户表示为一阶密度算子 + 量子匹配（static CF） | **静态** $\rho_u$ | ❌ 无序列/时间维 | 我们引入 $\rho_1\to\rho_T$ 动态演化 + 保合法性转移 |
| Quantum-inspired algorithms（Kerenidis-Prakash / E. Tang 等） | 低秩矩阵近似、采样加速（复杂度） | 非表示学习 | ❌ | 属算法加速，非表示/序列建模 |
| VBAE（arXiv 2105.07597） | 高斯潜变量 + **quantum-inspired uncertainty 度量**（hybrid CF） | 静态 | ❌ | 不确定性度量启发，非 density-state 序列演化 |
| Quantum-theory-inspired rec（arXiv 1601.06035） | 量子模型 / PSD 因子化做 item 推荐 | 静态 | ❌ | 理论动机启发，非序列状态演化 |

---

## 2. 方向 B：Uncertainty-aware / Distributional Sequential Recommendation

| 代表工作 | 核心做法 | 是否动态 | 状态约束（PSD/trace） | 与我们差异 |
|---|---|---|---|---|
| **CGE**（IJCAI 2019, arXiv 2006.10932） | Gaussian embedding（$\mu,\Sigma$）+ MC 采样打分 | 静态 | ❌ 仅协方差正定，非 density 算子 | 我们用 density operator（二阶含相关性）+ 演化保合法 |
| **W-GAT**（IEEE TCSS, arXiv 2404.05962） | GCN 中 Gaussian embedding，Wasserstein 距离（CF） | 静态 | ❌ | 分布相似度，无状态演化 |
| **G-STO**（arXiv 2306.14314） | **序列**推荐 + Gaussian embedding 表示意图 + 图正则 | 有（Transformer 编码序列） | ❌ | 表示是 Gaussian 而非 density；演化是 attention 隐式，无 PSD+trace 硬约束 |
| NEAT（arXiv 2202.05456） | Gaussian embedding 建模互补购买噪声 | 静态 | ❌ | 非序列状态 |
| uncertainty-aware sequential（arXiv 2508.07210） | 不确定性作用于 **LLM 解码/logit**，非表示层 | 有 | ❌ | 我们是不确定性在**状态表示层** + 演化 |

> ⚠️ 该方向是**最大风险**：必须回答"我们与 Gaussian/distributional 的本质区别"——**受约束（PSD+trace）的密度算子状态 + 合法性保持演化**，不是"把向量换成高斯分布"。

---

## 3. 方向 C：状态演化 / State-Space / 偏好漂移建模

| 代表工作 | 核心做法 | 与我们差异 |
|---|---|---|
| **HCRNN**（AAAI 2019, arXiv 1904.12674） | 层次 RNN + interest drift 假设，序列推荐 | 漂移用 RNN 门控；我们用 density 状态 + 凸组合（保合法） |
| **PERIS**（CIKM 2022, arXiv 2209.06644） | 个性化兴趣持续性，建模 interest drift | 向量表示；无算子态约束 |
| **TAI2Vec**（UMAP 2026, arXiv 2604.15581） | 时间感知 item embedding，区分短期/长期演化 | 表示是向量；非密度状态 |
| UniRec（CIKM 2024, arXiv 2406.18470）、TGODE（arXiv 2511.18347） | 时间/频域增强偏好演化 | 均为向量/ODE 状态，非 density operator |
| PaperFlow（arXiv 2606.07454） | 论文推荐中按天更新用户状态建模漂移 | 状态更新无 PSD+trace 约束 |
| state-space / Bayesian filtering（Kalman） | 状态估计演化 | 我们的状态是密度算子 + 凸组合（EMA/贝叶斯合并语义） |

---

## 4. 调研来源建议（写作时可引）
- WWW / KDD / SIGIR / CIKM 近 3 年：quantum recommendation、uncertainty-aware recommendation、distributional embedding、interest drift。
- arXiv：quantum machine learning + recommendation；density matrix embedding。

---

## 5. 引用维护（发现新文献即追加）
```markdown
[N] {作者}. "{标题}." *{会议}*, {年份}.
> 核心工作：______
> 引用位置/原因：______
> 与我们差异：______
```

---

## 6. 首轮检索记录（2026-08-03，arXiv API）

| 检索式 | 命中数 | 相关命中的代表 |
|---|---|---|
| all:"density matrix" AND all:recommendation | 5 | ConQAR（唯一相关）；其余为物理（DMRG/电离） |
| all:"quantum-inspired" AND all:recommendation | 15 | quantum-inspired algorithms（Tang 等）、VBAE、量子理论启发推荐 |
| all:"quantum" AND all:recommendation | 462 | 噪声大，多为物理/密码；无序列密度状态 |
| all:"uncertainty-aware" AND all:"sequential recommendation" | 1 | LLM 解码不确定性（2508.07210） |
| all:"distributional" AND all:"sequential recommendation" | 97 | 多为生成式/LLM/图方法，非 density 状态 |
| all:"Gaussian embedding" AND all:recommendation | 5 | CGE、W-GAT、G-STO、NEAT |
| all:"interest drift" AND all:recommendation | 10 | HCRNN、PERIS、TAI2Vec、UniRec、PaperFlow |

### 第二轮（2026-08-03，DBLP + Semantic Scholar）

| 检索式 | 命中 | 相关代表 |
|---|---|---|
| DBLP: "quantum recommendation" | 多 | WWW 2026（已知）、tSVD-based quantum context-aware rec（Quantum Inf. Process. 2021）、hybrid classical-quantum rec（Quantum Mach. Intell. 2025）、holographic perception rec（Inf. Fusion 2026） |
| DBLP: "density matrix recommendation" | 少 | 🔴 **DASFAA 2019: Density Matrix Based Preference Evolution Networks**（待精读） |
| Semantic Scholar（首轮） | - | 429 限流，待重试 |

> **待补充**（投稿前复核）：① Semantic Scholar 重试；② Google Scholar；③ WWW 2026 原文引用链；④ **DASFAA 2019 全文精读（最高优先）**。
