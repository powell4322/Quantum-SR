# 论文推进与审核（PAPER_PROGRESS）

> **本文档的职责**：钉死论文的 **Motivation / Contribution / 实验路线 / 投稿定位**，并作为**每阶段给 GPT（或协作者）审核**的入口，记录其纠正与建议，防止方向漂移。
> 用法：每完成一个阶段（实验、理论、写作），① 更新本文档对应章节 → ② 用 §5 模板把最新进展贴给 GPT 审核 → ③ 把 GPT 意见和处理记入 §5 表格 → ④ 改动需在此登记。

---

## 1. Research Start（研究起点 & 投稿定位）

### 1.1 一句话起点（研究声明 · 英文定稿）
> Sequential recommendation models user interests as hidden vectors $h_t=f(S_{\le t})$. However, user preference evolution has two characteristics: **uncertainty** and **gradual transition**. Existing methods (SASRec/BERT4Rec: point/contextual vectors; MIND: multiple but independent vectors; DMPEN: density as **feature** with RNN evolution) lack **a mathematically constrained evolving preference state**. We formulate user preference as a **dynamic density state** $\rho_t$ (PSD, trace=1) that evolves via a **legality-preserving convex transition**, and score next items by **Hilbert–Schmidt similarity**.
> ⚠️ 定位（2026-08-03 定稿）：创新主体 = **dynamic legality-preserving density state evolution**；**不做"首次 density"claim**（DMPEN 已做 density+sequential）；quantum 仅作来源说明，正文用 **operator-level** 表述。

### 1.2 标题方向（2026-08-03 二次定稿：标题不含 quantum）
- **Option 1（推荐）**：*Learning Dynamic Preference States via Legality-Preserving Density Evolution for Sequential Recommendation*
- **Option 2**：*Uncertainty-Aware Sequential Recommendation with Density State Evolution*（更安全）
> 标题**绝不出现 quantum**（WWW reviewer 对 quantum recommendation 已有 quantum CF / quantum embedding 先例，易被归为"又一个 quantum-inspired gimmick"）。

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

**英文 Motivation（可直接进 Intro）**
> Sequential recommendation models user interests as hidden vectors $h_t=f(S_{\le t})$. However, user preference evolution has two characteristics: **(1) uncertainty** (a user can hold multiple, competing interests) and **(2) gradual transition** (interests drift smoothly rather than jump). Existing methods lack a **mathematically constrained evolving preference state**:

| 方法 | 表示 | 演化 | 问题 |
|---|---|---|---|
| SASRec | 点向量 | attention | 无不确定性/结构 |
| BERT4Rec | 上下文向量 | 双向 attention | 同左 |
| MIND/ComiRec | 多个向量 | 独立、无融合 | 多向量但无序/无约束 |
| DMPEN (2019) | density 作为**特征** | RNN hidden state | density 非状态、无合法性约束 |

**三个核心 Gap（2026-08-03 定稿）**：
1. **Representation Gap（G1）**：主流序列推荐用点估计/多向量表示兴趣，丢失**不确定性**与**多模态偏好结构**。
2. **Evolution Gap（G2，真正创新）**：DMPEN 把 density matrix 当**二阶特征**送 RNN（$\rho=ee^\top$，$h_t=\sigma(Wh_{t-1}+V\rho_t+b)$）——density 是**特征**而非**偏好状态**；主流序列模型的 hidden state 无显式约束/概率解释。**无人研究"density operator 作为动态偏好状态被合法演化"**。
3. **Constraint Gap（G3）**：现有演化（RNN/attention）无**结构化状态约束**——$h_{t+1}=f(h_t,x_t)$ 无约束；我们的状态演化**恒保** PSD+trace（每步都是合法 preference distribution）。

**与 DMPEN 的关键区别（必须写进论文）**：DMPEN 问 "How to encode a behavior into a density matrix?"；我们问 "How does a density state evolve over time (with legality)?"——前者把 density 当表示增强，后者把 density 当**受约束的状态机**（density→density→density）。

**定位对比表（DMPEN vs Ours vs 经典序列模型）**
| 维度 | DMPEN (2019) | 我们 (Ours) | SASRec/BERT4Rec |
|---|---|---|---|
| density 用途 | **二阶特征**（输入 RNN） | **动态偏好状态**（演化对象） | 无 |
| 演化机制 | RNN hidden state | 凸组合（保 PSD+trace） | attention / 无显式演化 |
| 状态约束 | 无 | ✅ 恒保合法性 | 无 |
| uncertainty | 弱（借二阶相关性） | 强（谱=不确定性分解） | 弱（点估计） |
| 打分 | softmax | operator-level kernel (HS) | dot / softmax |

**定位提醒**：创新主体 = *dynamic legality-preserving density state evolution*；quantum 仅作来源说明，正文用 **operator-level** 表述。

---

## 3. Contribution（贡献声明，每条绑定证据）

> 贡献按重要度排序（2026-08-03 二次定稿：C2 唯一核心；C1 强调 state transition；C3 改为 uncertainty modeling）。

| # | 贡献（英文定稿） | 对应模块 / 证据 | 回答 |
|---|---|---|---|
| **C1（★★）** | **Preference evolution as constrained density-state transition**：把用户兴趣演化建模为满足量子态约束（PSD+trace）的概率状态转移过程（而非"首次用 density 表示"）。*We formulate user preference evolution as a constrained density state transition process.*（关键词：state transition，不是 representation） | `StateProjection` + `StateTransition`；03_theory §2/§3；E001-E003 | RQ1/RQ2 |
| **C2（★★★ 唯一核心）** | **Legality-preserving evolution**：保 PSD+trace 的凸组合状态演化（传统 hidden state 无约束：norm explosion / uninterpretable / arbitrary transition）。*We propose a legality-preserving preference transition mechanism based on convex density-state evolution.*（理论贡献） | `StateTransition`；03_theory §3；E004（约束消融） | RQ3（Theory） |
| **C3（★★）** | **Preference uncertainty modeling**：density 谱 = 多兴趣同时存在（mixture），比 vector 更能表达兴趣不确定性/竞争。*Density states capture multiple concurrent interests via their spectrum, providing interpretable preference uncertainty.*（对 WWW reviewer 最易接受；配合 entropy 分析） | 谱分解 + entropy $H(\rho)=-\mathrm{Tr}(\rho\log\rho)$；RQ4 | RQ4 |

> 写作注意：**不得写"首次 density"**；必须显式写 *Different from DMPEN that uses density matrices as second-order feature representations for RNNs, we investigate density operators as dynamic preference states with explicit legality-preserving evolution.*

## 3.5 Method 数学定义冻结（2026-08-03 定稿，先冻结再实验）
- **状态**：$\rho_t = \dfrac{L_t L_t^\top}{\mathrm{Tr}(L_t L_t^\top)}$，$L_t=\mathrm{Linear}(h_t)\in\mathbb{R}^{d\times r}$（PSD + trace=1）。
- **演化（transition）**：$\rho_{t+1}=\alpha\,\rho_t+(1-\alpha)\,\rho_{i_t}$，固定 $\alpha$（扫描 0.1–0.9）为主；learnable/adaptive 作 extension。
- **打分（score）**：$s(u,i)=\mathrm{Tr}(\rho_u\rho_i)$（Hilbert–Schmidt / operator-level kernel）。
- **损失（loss）**：**主 = BPR** $-\log\sigma(s^+-s^-)$（与 SASRec/BERT4Rec/GRU4Rec 可比）；**消融 = BCE**（配合 logit 变换）；fidelity loss 作 quantum-specific ablation。
- ⚠️ static vs dynamic 公平性：二者基于**相同** SASRec 编码器输出 $h_t$，仅差"演化"操作（见 §4.1），对比公平。

---

## 4. 实验路线（Experiment Line）

### 4.1 主实验：四阶递进逻辑（2026-08-03 二次定稿）
> 核心叙事：**Vector < Density Feature < Density State < Dynamic Evolution**——把"DMPEN 的 density 提升"与"我们的 state/evolution 提升"逐步拆开。

| 阶梯 | Baseline / 变体 | 回答 | RQ |
|---|---|---|---|
| A | **Vector**（SASRec，embedding→SASRec） | 基准 | - |
| B | **Density Feature**（模拟 DMPEN：$ee^\top$→flatten→SASRec） | density representation 是否有效 | RQ1 |
| C | **Density State static**（$\rho_T$，不演化） | state 是否有效 | RQ1 |
| D | **Density Dynamic**（$\rho_{t+1}=\alpha\rho_t+(1-\alpha)\rho_i$，最终模型） | 动态演化是否有效 | **RQ2** |

**RQ 定义（2026-08-03 二次定稿）**
- RQ1 **Representation**：density state 是否优于 vector / density feature？（A vs B vs C）
- RQ2 **Evolution**：动态演化是否优于静态？（C vs D）
- RQ3 **Theory**：PSD+trace 约束是否重要？（消融：remove PSD / remove trace / unconstrained matrix）
- RQ4 **Uncertainty**：density 为何有效？（$H(\rho)=-\mathrm{Tr}(\rho\log\rho)$ 熵分析用户兴趣变化）

**主表 baseline（6 个，不多加）**：GRU4Rec / SASRec / BERT4Rec / **DMPEN** / Ours-static / Ours-dynamic
> 暂缓：Caser / Gaussian / MIND / ComiRec（先证明 idea）。
> 公平性：static 与 dynamic 基于**相同**编码器输出，仅差演化操作（信息量相同）。

### 4.2 附实验（增强证据）
| 分析 | 内容 | 目的 |
|---|---|---|
| A-1 维度效率 | vector $d$ vs state $(d,r)$ 固定参数量 | second-order 参数效率 |
| A-2 序列长度敏感性 | history 5/10/20/50 | "动态"优势随上下文放大 |
| A-3 interest-shift 模拟 | 前 50 A 类 → 后 50 B 类，测适应步数 | 偏好漂移机制证据（附实验） |
| A-4 匹配消融 | dot vs operator-level kernel | 拆解"表示"vs"匹配" |

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
| 2026-08-03 | 定位定稿（GPT） | ① 冻结论文定位（不继续扩实验/代码）；② C1 降级、C2 唯一核心；③ 主表 baseline 精简为 6 个（GRU4Rec/SASRec/BERT4Rec/DMPEN/Ours-static/dynamic），Caser/Gaussian/MIND 暂缓；④ 实验重设计（E000 DMPEN 复现 + E001-E004）；⑤ adaptive α 不核心（fixed α 扫描 0.1–0.9）；⑥ Loss 主=logit+BCE、ablation=BPR；⑦ 03_theory 删 Kraus/depolarization/Born，保留 P1/P2/P3 | 全部采纳：§1/§2/§3/§4 重写为英文定稿；05/03 同步 | 创新叙事收紧，防"DMPEN 已做"攻击 |
| 2026-08-03 | 二次定位 review（GPT） | ① 强化"动态概率状态/uncertainty"叙事、弱化 quantum；② C1 改 state transition、C3 改 uncertainty modeling；③ **主损失改 BPR**（BCE 消融）；④ 实验改四阶递进（Vector<Density Feature<Density State<Dynamic）+ RQ1-4 + entropy 分析；⑤ 标题无 quantum；⑥ 先冻结 Method 数学定义再实验 | 全部采纳：§1.2/§3/§3.5/§4 更新；`--loss` 默认改 bpr | 叙事最稳、可比性最强 |

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
