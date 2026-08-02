# 论文推进与审核（PAPER_PROGRESS）

> **本文档的职责**：钉死论文的 **Motivation / Contribution / 实验路线 / 投稿定位**，并作为**每阶段给 GPT（或协作者）审核**的入口，记录其纠正与建议，防止方向漂移。
> 用法：每完成一个阶段（实验、理论、写作），① 更新本文档对应章节 → ② 用 §5 模板把最新进展贴给 GPT 审核 → ③ 把 GPT 意见和处理记入 §5 表格 → ④ 改动需在此登记。

---

## 1. Research Start（研究起点 & 投稿定位）

### 1.1 一句话起点（研究声明）
> 主流序列推荐用**点估计向量**编码用户兴趣；我们提出**动态密度状态（density state）** $\rho_t$，以合法性保持（PSD+trace=1）的方式随每次交互演化，并用 Hilbert–Schmidt 相似度完成 next-item 预测——首次把 density-operator 表示引入**序列**场景。

### 1.2 目标会议（已定：WWW，或近期截稿的合适会议）
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

两个核心 Gap（详见 `docs/RESEARCH_LOG.md` §1）：

1. **Sequential Gap（核心空白）**：现有 quantum-inspired 推荐（WWW 2026 等）只做**静态 CF**（静态 $\rho_u$），**从未**把 density-operator 用于建模序列兴趣的**演化与漂移**。
2. **Representation Gap**：主流序列推荐（SASRec/BERT4Rec）用点估计表示兴趣，丢失**不确定性**与**多模态偏好结构**；density-operator 提供统一的 second-order 表示。

**定位提醒（审稿风险规避）**：贡献主体是 *dynamic uncertainty-aware preference state modeling*；量子只是数学工具（density operator / convex mixing / HS kernel），用于提供合法（PSD+trace 保持）且可解释的框架。

---

## 3. Contribution（贡献声明，每条绑定证据）

| # | 贡献 | 对应模块 / 证据 | 回答 | 状态 |
|---|---|---|---|---|
| **C1** | 首次将密度状态表示引入序列推荐，显式建模用户偏好不确定性（second-order 结构） | `StateProjection`；理论 THEORY §2/§4.1；实验 E002 | RQ1 | ⬜ 待实验 |
| **C2** | 提出**合法性保持**的凸组合状态演化——偏好惯性模型（含自适应 $\alpha_t$），建模兴趣漂移 | `StateTransition`；理论 THEORY §3；实验 E003 + Exp.C | **RQ2（核心）** | ⬜ 待实验 |
| **C3** | 建立 Hilbert–Schmidt 打分，并在多个序列编码器（SASRec / GRU4Rec / BERT4Rec）与数据集（ml-1m / Beauty / 长尾）上验证**普适性** | `match`；实验 A-1~A-5 + 泛化 | RQ3 + generalization | ⬜ 待实验 |

> 写作时贡献声明务必按"问题 → 方法 → 证据"三段式，避免只列"我们做了什么"。

---

## 4. 实验路线（Experiment Line）

### 4.1 主线（决定论文成立与否）
| 实验 | 对比 | 目的 | 回答 | 状态 |
|---|---|---|---|---|
| E001 | vector（正式超参，ml-1m） | 基线复现 | RQ1 对照 | ⬜ |
| E002 | state vs vector（**同参数量**） | 状态表示是否优于点估计 | RQ1 | ⬜ |
| E003 | static vs dynamic（learnable $\alpha$） | 动态演化是否优于静态 | **RQ2（核心）** | ⬜ |

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
- 除被验证维度外**同超参**；Tr/BCE 兼容修正（§4.4 RESEARCH_LOG）落地后再跑正式实验；结果回填 `docs/RESEARCH_LOG.md` §6 与本文档 §4。

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

---

## 6. 文档管理（合并 / 删除建议）

| 文档 | 职责 | 结论 |
|---|---|---|
| `docs/RESEARCH_LOG.md` | 研究追踪（RQ 状态、实验日志、问题） | ✅ **保留**（核心） |
| `docs/THEORY.md` | 论文理论支撑（正式、权威源） | ✅ **保留**，但需与 RESEARCH_LOG 新口径对齐（rank>1 谱分解、dof 分析、偏好惯性） |
| `docs/PAPER_PROGRESS.md` | 论文推进+审核驱动（本文档） | ✅ **保留**（新增） |
| `docs/sasrec.md` | 代码使用说明 | ✅ **保留**（英文名，已决定；中文名文档已删除） |
| `.github/skills/*` | 研究 skill | ✅ 保留（需同步 quantum-seq-rec 的"rank>1 多兴趣"措辞） |

**职责划分原则**：RESEARCH_LOG = 研究过程（是什么/发生了什么）；THEORY = 理论为什么成立；PAPER_PROGRESS = 论文怎么组织+审核入口；使用说明 = 给代码使用者。避免同一内容出现在两处。

---

## 7. 当前状态 & 下一步（截至 2026-08-02）
- [x] 文档去重：保留 `docs/sasrec.md`，删除中文名文档（2026-08-02）
- [x] docs/THEORY.md 与 RESEARCH_LOG 新口径对齐（2026-08-02）
- [ ] 确定目标会议（§1.2）与截稿
- [ ] 主线：Tr 打分修正 → E001 基线复现（GPU）
- [ ] 每阶段用 §5 模板提交 GPT 审核并回填
