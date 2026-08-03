# 论文推进与审核（PAPER_PROGRESS）

> **本文档的职责**：钉死论文的 **Motivation / Contribution / 实验路线 / 投稿定位**，并作为**每阶段给 GPT（或协作者）审核**的入口，记录其纠正与建议，防止方向漂移。
> 用法：每完成一个阶段（实验、理论、写作），① 更新本文档对应章节 → ② 用 §5 模板把最新进展贴给 GPT 审核 → ③ 把 GPT 意见和处理记入 §5 表格 → ④ 改动需在此登记。

---

## 1. Research Start（研究起点 & 投稿定位）

### 1.1 一句话起点（研究声明）
> 主流序列推荐用**点估计向量**编码用户兴趣；我们提出**动态密度状态（dynamic density-state）建模**：把用户兴趣表示为满足 $\rho\succeq0,\ \mathrm{Tr}(\rho)=1$ 约束的**偏好状态** $\rho_t$，随每次交互按**偏好惯性**演化，并用 Hilbert–Schmidt 相似度完成 next-item 预测。
> ⚠️ 定位（2026-08-03 依据 DMPEN 精读修正）：**不再声称"首次把 density matrix 用于序列推荐"**（DMPEN, DASFAA 2019 已做 density matrix + RNN 演化）。我们的定位改为：**把 density operator 从"特征表示"升级为"动态、合法、可解释的偏好状态"**。*Existing density-matrix recommendation methods mainly exploit second-order preference representations, while we study density operators as dynamically evolving preference states under sequential interaction.*

### 1.2 标题方向（2026-08-03 依 DMPEN 精读调整，避免与 DMPEN 撞车）
- **Option A（推荐）**：*Dynamic Density State Modeling for Uncertainty-Aware Sequential Recommendation*（无 quantum，WWW 风格）
- **Option B**：*From Point Preferences to Density States: Modeling Uncertainty and Evolution in Sequential Recommendation*（强调 transition）
- **Option C**：*Legality-Preserving Preference Evolution via Density Operators for Sequential Recommendation*（突出数学）
> "Quantum-inspired" 仅作来源说明，不出现在主标题。

### 1.3 目标会议（已定：WWW，或近期截稿的合适会议）
| 项 | 内容 |
|---|---|
| 目标会议 | **WWW**（The Web Conference），或近期截稿的匹配会议 |
| 篇幅 | 主会**双栏 9–10 页（含 references）**，另有短文/工业轨道可选 |
| 模板 | ACM sigconf（\acmart）LaTeX 模板 |
| 匿名性 | 双盲，正文与代码（链接）不能暴露作者 |
| 截稿 | 待查官方 CFp 填入具体日期与倒计时 |
| 定位匹配 | WWW 偏方法/系统 + 应用，与"动态偏好状态建模"匹配度高 |

> "start" 含义：**先定"我们解决什么问题 + 投哪个会 + 规模能匹配到什么程度"**，再倒推每阶段的验收标准。

---

## 2. Motivation（动机，可直接进 Intro）

三个核心 Gap（2026-08-03 依 DMPEN 精读重定义——注意：**"density matrix + sequential" 已有人做（DMPEN 2019）**，Gap 不再是"没人做 density 序列"）：

1. **Representation Gap（G1）**：主流序列推荐（SASRec/BERT4Rec）用点估计向量表示兴趣，丢失**不确定性**与**多模态偏好结构**。
2. **Evolution Gap（G2，真正创新）**：现有 density-matrix 方法（**DMPEN, DASFAA 2019**）把 density matrix 作为**二阶特征**送入 RNN（$\rho=ee^\top$，$h_t=\sigma(Wh_{t-1}+V\rho_t+b)$）——density 是**输入特征**而非**偏好状态**；主流序列模型（SASRec/BERT4Rec/GRU4Rec）的 hidden state 也无显式约束/概率解释。**没有人研究"density operator 如何作为动态偏好状态被合法演化"。**
3. **Constraint Gap（G3）**：现有演化（RNN/attention）无**结构化状态约束**——任意 $h_{t+1}=f(h_t,x_t)$ 无约束；我们的状态演化**恒保** PSD+trace（每步都是合法 preference distribution）。

**与 DMPEN 的关键区别（必须写进论文）**：DMPEN 问 "How to encode a behavior into a density matrix?"；我们问 "How does a density state evolve over time (with legality)?"——前者把 density 当表示增强，后者把 density 当**受约束的状态机**。

**定位对比表（DMPEN vs Ours vs 经典序列模型）**
| 维度 | DMPEN (2019) | 我们 (Ours) | SASRec/BERT4Rec |
|---|---|---|---|
| density matrix 用途 | **二阶特征**（输入 RNN） | **动态偏好状态**（演化对象） | 无 |
| 演化机制 | RNN hidden state | 凸组合（保 PSD+trace） | attention / 无显式演化 |
| 状态约束 | 无（hidden state 无约束） | ✅ 恒保合法性 | 无 |
| uncertainty | 弱（借二阶相关性） | 强（谱=不确定性分解） | 弱（点估计） |
| 打分 | softmax | Hilbert–Schmidt 核 | dot / softmax |

**定位提醒**：贡献主体是 *dynamic uncertainty-aware preference state modeling*；density operator 只是实现"状态合法约束 + 可解释演化"的数学工具（quantum-inspired 仅作来源说明）。

---

## 3. Contribution（贡献声明，每条绑定证据）

> 贡献按重要度排序（2026-08-03 依 DMPEN 精读修订：**C2 合法性演化 = 核心**，C1 不再声称"首次"）。

| # | 贡献（英文表述，供论文） | 对应模块 / 证据 | 回答 |
|---|---|---|---|
| **C1（★★）** | **Density states as sequential preference representations**：用受约束密度状态扩展点向量表示，带不确定性感知的二阶结构。*We introduce density states as explicit sequential preference representations, extending point-based user representations with uncertainty-aware second-order structures.*（≠ DMPEN：我们把 density 当状态而非特征） | `StateProjection`；03_theory §2；E002（含 DMPEN 对照） | RQ1 |
| **C2（★★★ 核心）** | **Legality-preserving preference transition**：保 PSD+trace 的凸组合状态演化（偏好惯性 + 自适应 $\alpha_t$），显式建模兴趣漂移。*We propose a legality-preserving preference transition mechanism based on convex state evolution, maintaining PSD and trace constraints during sequential updates.* | `StateTransition`（含 adaptive α）；03_theory §3；E003 + E004 | **RQ2（核心）** |
| **C3（★）** | **Density-state similarity (Hilbert–Schmidt)**：用 HS 核做状态相似度 + 多数据集/多 baseline 系统评估。*We develop density-state similarity learning with Hilbert–Schmidt kernel.*（不说 quantum measurement） | `match`；A-1~A-5 + 多数据集；05_experiment_plan | RQ3 |

> 写作注意：**不得再写"首次引入 density matrix 到推荐/序列"**（会被 DMPEN 击穿）；必须显式写 *Different from DMPEN that uses density matrices as second-order feature representations for RNNs, we investigate density operators as dynamic preference states with explicit legality-preserving evolution.*

---

## 4. 实验路线（Experiment Line）

### 4.1 主线（采纳 GPT 审稿建议 2026-08-02 调整）
| 阶段 | 实验 | 对比 | 目的 | 回答 |
|---|---|---|---|---|
| 1 | E001 | SASRec vector baseline（正式超参） | 确认 HR/NDCG 复现 | RQ1 对照 |
| 2 | E002 | **Representation Ablation**：vector vs state-r1 vs state-r4 vs Gaussian vs **DMPEN-style（density-as-feature + RNN）**（同参数量） | density 表示是否有必要 + 证明与 DMPEN 不同 | RQ1 |
| 3 | E003 | **Evolution Ablation**：static 最后状态 vs EMA(fixed α) vs dynamic(learnable α) | 动态演化是否优于静态（**论文核心图**） | RQ2（核心） |
| 4 | E004 | **Interest shift 模拟**：前 50 Action → 后 50 Romance，测适应步数 | 直接验证偏好漂移（比 long-tail 更能证明创新） | RQ2/H2 |

### 4.2 附实验（增强证据，非独立 RQ）
| 分析 | 内容 | 目的 |
|---|---|---|
| A-1 维度效率 | vector $d=64$ vs state $d=32,r=4$（固定参数量） | second-order 参数效率 |
| A-2 序列长度敏感性 | history 5/10/20/50 增益曲线 | "动态"优势随上下文放大 |
| A-3 兴趣漂移模拟 | 100 个 A 类 → 100 个 B 类，观察适应速度 | 直接检验 H2 |
| A-4 多样性 | coverage / ILD | RQ3 补充 |
| A-5 匹配消融 | dot vs trace（纯态下 $Tr=\langle u,i\rangle^2$） | 拆解"表示"vs"匹配" |

### 4.3 泛化（Priority 4，后置）
- 在 GRU4Rec / BERT4Rec 上复现 E002/E003 → 写成 *generalization across encoders*。

### 4.4 铁律
- 除被验证维度外**同超参**；Tr/BCE 兼容修正（§4.4 RESEARCH_LOG）落地后再跑正式实验；结果回填 `docs/02_research_log.md` §6 与本文档 §4。

---

## 5. GPT 审核机制（每阶段使用）

### 5.1 提交给 GPT 的模板（复制后替换）
```
[论文推进 · 阶段提交]
- 目标会议/定位：<填>
- 本阶段完成：<实验/理论/写作片段>
- 关键结果/内容：<贴数据或文字>
- 当前贡献声明：<C1/C2/C3>
- 我的疑问/不确定点：<明确列出>
请给出：1) 方向是否跑偏；2) 审稿人会攻击的 3 个点及应对；3) 下一步优先级。
```

### 5.2 审核反馈记录表（每阶段追加一行）
| 日期 | 阶段 | GPT 意见（要点） | 我们处理 / 采纳 | 结果 / 对论文影响 |
|---|---|---|---|---|
| 2026-08-02 | 方向收敛 | （示例）"量子只是工具，贡献主体是 uncertainty-aware 建模" | 采纳：措辞全部改为 density-state | 动机更稳（见 §2） |
| 2026-08-02 | 整稿审阅（GPT） | ① 定位从"quantum 推荐"转向"dynamic density-state"；② 贡献重排（evolution 第一）；③ 增 Constraint Gap；④ CPTP/Born 弱化；⑤ 多数据集多 baseline；⑥ interest-shift 作为核心分析；⑦ 修 Tr loss + 加 adaptive α | 全部采纳：§1/§2/§3/§4 已改；新增 04_related_work/05_experiment_plan；03_theory 弱化 CPTP/Born；RQ 对齐 | 定位更稳，防"只是换 distribution 表示"攻击 |
| 2026-08-03 | DMPEN 精读（GPT） | 发现 DMPEN(DASFAA 2019) 已做 density matrix + RNN 序列演化 → **"首次 density+sequential" 不成立**；创新点迁移到 "dynamic density state evolution + legality"；必须加 DMPEN baseline；标题避免撞车 | 全部采纳：§1/§2/§3/§4 已改；04_related_work 记入 DMPEN 详情；定位改为"density 状态机 vs 特征" | 避免被 2019 论文直接击穿 |

---

## 6. 文档管理（合并 / 删除建议）

| 文档 | 职责 | 结论 |
|---|---|---|
| `docs/02_research_log.md` | 研究追踪（RQ 状态、实验日志、问题） | ✅ **保留**（核心） |
| `docs/03_theory.md` | 论文理论支撑（正式、权威源） | ✅ **保留**，但需与 RESEARCH_LOG 新口径对齐（rank>1 谱分解、dof 分析、偏好惯性） |
| `docs/01_paper_progress.md` | 论文推进+审核驱动（本文档） | ✅ **保留**（新增） |
| `docs/06_usage_sasrec.md` | 代码使用说明 | ✅ **保留**（英文名，已决定；中文名文档已删除） |
| `docs/04_related_work.md` | 相关工作总结 + 防撞对照表（quantum rec / uncertainty rec / distributional rec） | ✅ **新增**（采纳 GPT：防止撞 idea） |
| `docs/05_experiment_plan.md` | 实验矩阵（数据集、baseline、主/消融/分析） | ✅ **新增**（采纳 GPT：所有实验矩阵集中管理） |
| `.github/skills/*` | 研究 skill | ✅ 保留（需同步 quantum-seq-rec 的"rank>1 多兴趣"措辞） |

**职责划分原则**：RESEARCH_LOG = 研究过程（是什么/发生了什么）；THEORY = 理论为什么成立；PAPER_PROGRESS = 论文怎么组织+审核入口；使用说明 = 给代码使用者。避免同一内容出现在两处。

---

## 7. 当前状态 & 下一步（截至 2026-08-02，采纳 GPT 审稿意见后）
- [x] 定位转向 Dynamic Density-State Modeling（§1.1）；贡献重排（§3）；增 Constraint Gap（§2）
- [x] 新增 `docs/04_related_work.md`（防撞）、`docs/05_experiment_plan.md`（实验矩阵）
- [x] THEORY 弱化 CPTP/Born；RQ 对齐（RQ1 表示 / RQ2 演化 / RQ3 优势时机）
- [ ] **Step 1**：RELATED_WORK 防撞检查（确认 dynamic density state 是否有人做过）→ 填对照表
- [ ] **Step 2**：冻结 Motivation + Contribution + Method overview → 形成论文第一页
- [ ] **Step 3（Priority 0）**：改代码（修正 Tr loss：temperature/logit；加 adaptive $\alpha_t$）
- [ ] **Step 4**：跑 E001 → E002 → E003 → E004（GPU）
