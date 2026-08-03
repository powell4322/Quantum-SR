# 量子启发序列推荐 · 论文理论支撑（Theory & Formulation）

> 本文档为论文提供**严格的数学表述与理论论证**，并把"量子信息概念"与"推荐算法设计"一一对应，防止审稿人质疑"只是借用名词"。
> 状态：持续更新；每个理论条目应逐步被实验（见 `docs/02_research_log.md` §6）佐证或修正。

---

## 0. 文档目的

1. 把三个研究假设（H1/H2/H3）写成精确的数学命题；
2. 给出密度矩阵构造、状态演化、测量匹配的严格定义与合法性证明；
3. 明确与已有工作（尤其 WWW 2026 静态 quantum CF）的理论边界；
4. 明确定位：density operator 作为**数学工具**（不是研究问题），控制 "quantum" 出现次数，避免审稿人认为"只是搬量子概念"。

---

## 1. 问题定义与记号

- 用户 $u$ 的历史序列 $S_u = [i_1, i_2, \dots, i_T]$，$i_t \in \mathcal{I}$（物品集合）。
- 编码器（SASRec）$f_\theta$ 输出每步隐藏表示 $h_t = f_\theta(S_{\le t}) \in \mathbb{R}^d$，即 $h_t$ 只依赖当前及之前的行为（因果性）。
- 任务：给定 $S_u$，对每个候选物品 $i \in \mathcal{I}$ 打分并排序，目标是命中 $i_{T+1}$。

### 核心记号对照表

| 经典（baseline） | 量子启发（proposed） |
|---|---|
| 兴趣向量 $h_t \in \mathbb{R}^d$ | 密度矩阵 $\rho_t \in \mathcal{H}^{d\times d}$ |
| 物品向量 $e_i \in \mathbb{R}^d$ | 物品态 $\rho_i$（与用户同构构造） |
| 相似度 $h_t^\top e_i$ | Hilbert–Schmidt 相似度 $\mathrm{Tr}(\rho_t \rho_i)$ |
| 状态更新（RNN/attention） | 状态演化 $\rho_{t+1} = F(\rho_t, \rho_{i_t})$ |

---

## 2. 状态表示：从向量到密度矩阵（H1）

### 2.1 纯态与混合态

- **纯态**（rank-1）：$\rho = |\psi\rangle\langle\psi|$，等价于一个确定方向。
- **混合态**（rank>1）：$\rho = \sum_k \lambda_k |\psi_k\rangle\langle\psi_k|$，$\lambda_k \ge 0, \sum_k\lambda_k=1$。
- ⚠️ 措辞口径：谱分解 $\sum_k\lambda_k|\psi_k\rangle\langle\psi_k|$ 提供**用户偏好不确定性的潜在分解（latent decomposition of preference uncertainty）**；rank>1 只表示秩更高，**不自动对应"多个语义兴趣"**——我们不承诺"每个特征向量=一个兴趣"。

### 2.2 密度矩阵的三条合法性公理

对任意密度矩阵 $\rho$：

1. **Hermitian**：$\rho = \rho^\dagger$；
2. **半正定（PSD）**：$\rho \succeq 0$；
3. **归一**：$\mathrm{Tr}(\rho) = 1$。

任一特征值 $\lambda_k \in [0,1]$，可视为"该兴趣分量的概率质量"——这是兴趣不确定性建模的数学基础。

### 2.3 我们的构造（保证合法性的实现）

给定 $h_t$，通过低秩 Cholesky-like 投影：

$$\rho_t = \frac{L_t L_t^\top}{\mathrm{Tr}(L_t L_t^\top)}, \qquad L_t = W h_t \in \mathbb{R}^{d \times r}$$

**性质（已用 `test_smoke.py` 验证）：**
- $L L^\top \succeq 0$ 恒成立，故 $\rho \succeq 0$；
- 除以迹后 $\mathrm{Tr}(\rho)=1$；
- $r=1$ 时为纯态（$\mathrm{rank}=1$），$r>1$ 时为混合态。

### 2.4 参数量与有效自由度（精确分析，勿再写"更多自由度"）

- 参数量：$L$ 有 $dr$ 个参数。
- 有效自由度：$\rho=LL^\top$ 对 $L$ 有**正交旋转歧义**（$L\mapsto LQ,\ Q\in O(r)$ 不改变 $\rho$），且受 trace=1 约束，故

$$\mathrm{dof}(\rho)=dr-\frac{r(r-1)}{2}-1$$

- $r=1$：$\mathrm{dof}=d-1$，恰好是单位向量所在球面 $S^{d-1}$ 的自由度 → **纯态与向量同量级**（"state 是 vector 的自然推广"的关键证据）。
- $r=d$：$\mathrm{dof}=\frac{d(d+1)}{2}-1$，即 trace 固定的对称矩阵。

> 结论表述（论文用）：*Low-rank density operators provide a structured second-order representation with controllable rank complexity.* —— 卖点是 **second-order 结构** 与 **rank 可控性**，不是"更多自由度"。
> ⚠️ 对应实验：A-1 维度效率（vector $d=64$ vs state $d=32,r=4$，固定参数量）。

---

## 3. 状态演化：偏好惯性模型（H2 / RQ2，核心创新）

### 3.1 状态演化的一般形式

我们考虑形如 $\rho_{t+1}=\mathcal{F}(\rho_t,\rho_{i_t})$ 的演化，并采用最简单、可微且**保证合法性**的形式——**凸组合**：

$$\rho_{t+1} = \alpha\,\rho_t + (1-\alpha)\,\rho_{i_t}, \qquad \alpha \in [0,1]$$

其中 $\rho_{i_t}$ 是当前交互物品诱导的状态。
> ⚠️ 表述口径（采纳 GPT 审稿建议）：**不展开 CPTP/Kraus**——我们的演化是凸组合，不是完整量子信道；避免被物理背景审稿人攻击。量子仅作为"合法性保持"的灵感来源。

### 3.2 合法性证明（保 PSD 与 trace）

- **trace**：$\mathrm{Tr}(\rho_{t+1}) = \alpha\mathrm{Tr}(\rho_t) + (1-\alpha)\mathrm{Tr}(\rho_{i_t}) = \alpha + (1-\alpha) = 1$。
- **PSD**：PSD 集合是凸锥，两个 PSD 矩阵的非负组合仍 PSD。
- 因此凸组合是"合法状态演化"的最小实现，且**标量 $\alpha$ 带来零额外参数量**（或仅 1 个可学习标量）。

### 3.3 推荐语义：偏好惯性（Preference Inertia）

凸组合 $\rho_{t+1}=\alpha\rho_t+(1-\alpha)\rho_{i_t}$ 在推荐语义上解释为**偏好惯性**：$\alpha_t$ 越大越保留旧兴趣，越小越快速吸收新兴趣。
- 与去极化信道的联系：标准去极化 $\rho'=(1-p)\rho+p\,\frac{I}{d}$ 是"向完全混合态收缩"；我们改为"向**新观测物品态**混合"，即**信息注入式演化**（用户看到新物品后兴趣被更新）。

### 3.4 与经典方法的关系（回应 "Why not GRU?"）

- **固定 $\alpha$**：展开得 $\rho_T=\alpha^{T-1}\rho_1+(1-\alpha)\sum_{k=1}^{T-1}\alpha^{T-1-k}\rho_{i_k}$，即**指数滑动平均（EMA）/ 几何加权**——旧兴趣按 $\alpha$ 指数衰减，天然具备遗忘机制；语义上等价于贝叶斯滤波的信息合并。
- **与 GRU/门控的区别（论文卖点）**：凸组合结构保证**中间状态永远是合法密度算子（PSD+trace 保持）**——这是 GRU/门控在表示层面不具备的硬约束。

| | RNN/GRU | 我们的凸组合 |
|---|---|---|
| 更新形式 | 线性变换 + 非线性（$z,r,\tilde h$） | 概率混合 |
| 状态语义 | 隐藏向量（无概率解释） | 概率分布（谱分解） |
| 合法性 | 无约束 | 恒保 PSD + trace |
| 遗忘机制 | 隐式门控 | 显式指数衰减（$\alpha$） |

### 3.5 自适应 $\alpha_t$（关键增强，C2 的一部分）

$$\alpha_t=\sigma\big(W\,[h_t;\,e_{i_t}]+b\big)$$

- 含义：不同用户（甚至不同时刻）的兴趣稳定度不同——长期稳定用户 $\alpha_t\uparrow$，探索型用户 $\alpha_t\downarrow$。
- 梯度风险：可学习/自适应 $\alpha$ 可能坍缩到 0/1（见 `docs/02_research_log.md` §4.5 开放清单）。

---

## 4. 打分：Hilbert–Schmidt 相似度（H3 / 打分模块）

### 4.1 定位：Hilbert–Schmidt 内积（作为 similarity kernel，而非强调"测量坍缩"）

当 $\rho_u,\rho_i$ 都是密度矩阵时，$\mathrm{Tr}(\rho_u\rho_i)$ 本质是 $\mathbb{R}^{d\times d}$ 上的 **Hilbert–Schmidt 内积**：

$$\langle \rho_u, \rho_i \rangle_{HS} = \mathrm{Tr}(\rho_u^\dagger \rho_i)$$

论文表述：*We adopt Hilbert–Schmidt similarity induced by density operators.*（Born-rule / 测量坍缩仅作为 discussion 的灵感来源，不做强 claim。）

### 4.2 与 dot product 的精确桥梁（重要）

当 $\rho_u,\rho_i$ 均为**纯态**（rank-1）$\rho_u=uu^\top,\ \rho_i=ii^\top$ 时：

$$\mathrm{Tr}(\rho_u\rho_i)=\mathrm{Tr}(uu^\top ii^\top)=(u\cdot i)^2=\cos^2\theta$$

即 trace 打分退化为**平方余弦**（有界 $[0,1]$），而 vector dot 无界。这：
1. 为"state ⊇ vector"提供严格桥梁（$r=1$ 时 state 恰是平方点积核）；
2. 解释了 Tr/BCE 不匹配（`docs/02_research_log.md` §4.4）：logits 被压到 $[0,1]$。

### 4.3 与 dot product 的对比

| | dot product $h^\top e$ | Hilbert–Schmidt $\mathrm{Tr}(\rho_u\rho_i)$ |
|---|---|---|
| 对象 | 向量欧氏内积 | 算子内积（含二次/交叉项） |
| 表达能力 | 一阶 | 二阶（$\rho$ 含 $h h^\top$ 类结构） |
| 有界性 | 无界 | $[0,1]$（需损失兼容修正，见 `docs/02_research_log.md` §4.4） |

> ⚠️ 注意：$\rho$ 含 $L L^\top$ 结构，$\mathrm{Tr}(\rho_u\rho_i)$ 实际等价于对"特征化后的方向"做加权内积——这是受限维度下可能优于纯 dot 的原因；**打分有效性由 A-5 匹配消融验证。**

---

## 5. 可证明/可验证的性质清单（逐步补全）

| 编号 | 命题 | 状态 |
|---|---|---|
| P1 | 构造式 $\rho = LL^\top/\mathrm{Tr}$ 恒为合法密度矩阵 | ✅ 实现已验证（`test_smoke.py`） |
| P2 | 凸组合演化保 PSD 与 trace | ✅ 数学证明 + 实现验证 |
| P3 | 低秩密度矩阵有效自由度 $\mathrm{dof}=dr-\frac{r(r-1)}{2}-1$；$r=1$ 时恰为 $d-1$（= 单位球面自由度） | ✅ 计数论证（§2.4） |
| P4 | $\mathrm{Tr}(\rho_u\rho_i)\in[0,1]$（双方均为密度矩阵）；纯态下 $=(u\cdot i)^2$ | ✅ 由 Cauchy–Schwarz + PSD 可得 |
| P5 | 动态演化对"兴趣漂移"的建模能力优于静态 $\rho_u$ | ⬜ 待实验（RQ2 / Exp.C） |
| P6 | 低维下 $\rho$ 表示优于向量表示（长尾/多样性） | ⬜ 待实验（RQ3 / A-1, A-4） |

---

## 6. 与已有工作的理论边界

| 工作 | 状态 | 演化 | 我们 vs 它 |
|---|---|---|---|
| WWW 2026 quantum CF | $\rho_u$ 静态 | 无 | 我们引入**时间维演化** $\rho_1\to\rho_T$ |
| 经典 sequence rec（SASRec/BERT4Rec） | 向量 $h_t$ | attention 隐式 | 我们显式建模**概率状态**并给出演化律 |
| Multi-interest rec（MIND/ComiRec） | 多个向量 | 无/聚类 | 我们统一为**一个密度矩阵的谱**，天然有序演化 |
| Uncertainty-aware rec | 方差/分布 | 多为静态 | 我们用密度矩阵统一"不确定性 + 兴趣结构" |

---

## 7. 论文叙事框架（可复用）

> Sequential recommendation models usually encode evolving user preferences as **deterministic latent vectors** $h_t$. We argue that user interest is inherently **uncertain and multi-faceted**, and evolves with each interaction. We propose to represent it as a **dynamic density state** $\rho_t$ (PSD, trace=1), which (i) provides a second-order representation of preference uncertainty, (ii) evolves via a **legality-preserving preference-inertia transition** (convex combination), and (iii) scores next items via **Hilbert–Schmidt similarity** $\mathrm{Tr}(\rho_t\rho_i)$.

贡献三点（与 `docs/01_paper_progress.md` §3 一致，按重要度）：
- **C1**：动态密度状态建模——把不确定偏好建模为受约束状态（second-order 结构）。
- **C2**：合法性保持的偏好惯性演化（含自适应 $\alpha_t$），显式建模兴趣漂移。
- **C3**：密度状态相似度学习（Hilbert–Schmidt 核）+ 多数据集/多 baseline 系统评估。

---

## 8. 建议引用/支撑方向（供写作时查证）

- 量子信息基础（仅作来源说明）：Nielsen & Chuang；密度矩阵 / Hilbert–Schmidt 内积。
- Quantum ML：quantum kernel / quantum representation learning（包括 WWW 2026 相关工作）。
- 序列推荐：SASRec、BERT4Rec、GRU4Rec、MIND、ComiRec、SSE-PT。
- 不确定性建模：uncertainty-aware recommendation / distributional representation。

---

## 9. 待补理论工作（Open Problems）

- [ ] ~~更一般的 CPTP/Kraus 形式~~（已按审稿建议弱化，不展开量子信道）；
- [ ] 兴趣演化的可解释性：用 $\rho_t$ 的谱分解做"推荐理由"（仅作 discussion 灵感）；
- [ ] 泛化/信息论分析：密度状态表示在有限样本下的优势界（future work）；
- [ ] 与注意力因果 mask 的严格对应：$\rho_t$ 只依赖 $S_{\le t}$ 的形式化。
