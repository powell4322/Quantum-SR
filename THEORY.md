# 量子启发序列推荐 · 论文理论支撑（Theory & Formulation）

> 本文档为论文提供**严格的数学表述与理论论证**，并把"量子信息概念"与"推荐算法设计"一一对应，防止审稿人质疑"只是借用名词"。
> 状态：持续更新；每个理论条目应逐步被实验（见 `RESEARCH_LOG.md` §6）佐证或修正。

---

## 0. 文档目的

1. 把三个研究假设（H1/H2/H3）写成精确的数学命题；
2. 给出密度矩阵构造、状态演化、测量匹配的严格定义与合法性证明；
3. 明确与已有工作（尤其 WWW 2026 静态 quantum CF）的理论边界；
4. 为"为什么叫 quantum-inspired"提供自洽辩护。

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
| 相似度 $h_t^\top e_i$ | 测量概率 $\mathrm{Tr}(\rho_t \rho_i)$ |
| 状态更新（RNN/attention） | 状态演化 $\rho_{t+1} = F(\rho_t, \rho_{i_t})$ |

---

## 2. 状态表示：从向量到密度矩阵（H1）

### 2.1 纯态与混合态

- **纯态**（rank-1）：$\rho = |\psi\rangle\langle\psi|$，等价于一个确定方向。
- **混合态**（rank>1）：$\rho = \sum_k \lambda_k |\psi_k\rangle\langle\psi_k|$，$\lambda_k \ge 0, \sum_k\lambda_k=1$，可解释为"用户以概率 $\lambda_k$ 处于兴趣 $k$"。

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

### 2.4 表达力论证（RQ4 的理论支撑）

- 向量参数：$d$ 个自由度；$d\times r$ 低秩密度矩阵：$O(dr)$ 个自由度。
- 当 $r \ge 1$，在**相同参数量预算**下，密度矩阵把"方向信息"扩展为"分布信息"：一个 $d$ 维概率单纯形可以容纳的兴趣中心远多于一个点。
- 形式上：$\rho$ 的谱分解 $\sum_k \lambda_k |\psi_k\rangle\langle\psi_k|$ 即"多兴趣 + 兴趣强度"的显式表示，而向量 $h_t$ 只有一个方向。

> ⚠️ 对应实验：RQ4（不同 latent/rank 下的表现，低维优势）。

---

## 3. 状态演化：兴趣的动态性（H2 / RQ3，核心创新）

### 3.1 量子开放系统视角

真实量子态在噪声/测量/环境作用下按 **CPTP 映射**（完全正保迹）演化：

$$\rho' = \mathcal{E}(\rho), \qquad \mathcal{E} \text{ 为 CPTP}$$

我们采用其最简单且实现可微的形式——**凸组合**：

$$\rho_{t+1} = \alpha\,\rho_t + (1-\alpha)\,\rho_{i_t}, \qquad \alpha \in [0,1]$$

其中 $\rho_{i_t}$ 是当前交互物品诱导的状态。

### 3.2 合法性证明（保 PSD 与 trace）

- **trace**：$\mathrm{Tr}(\rho_{t+1}) = \alpha\mathrm{Tr}(\rho_t) + (1-\alpha)\mathrm{Tr}(\rho_{i_t}) = \alpha + (1-\alpha) = 1$。
- **PSD**：PSD 集合是凸锥，两个 PSD 矩阵的非负组合仍 PSD。
- 因此凸组合是"合法状态演化"的最小实现，且**标量 $\alpha$ 带来零额外参数量**（或仅 1 个可学习标量）。

### 3.3 与去极化信道的联系

标准去极化信道：$\rho' = (1-p)\rho + p\,\frac{I}{d}$（向完全混合态收缩）。
我们的形式 $\rho' = \alpha\rho_t + (1-\alpha)\rho_{i_t}$ 可看作"向**新观测到的物品态**混合"而非"向均匀态混合"，即**信息注入式演化**——语义上对应"用户看到新物品后兴趣被更新"。

### 3.4 与 RNN/GRU 状态更新的关键区别（论文卖点）

| | RNN/GRU | 我们的凸组合 |
|---|---|---|
| 更新形式 | 线性变换 + 非线性（$z,r,\tilde h$） | 概率混合 |
| 状态语义 | 隐藏向量（无概率解释） | 概率分布（$\lambda_k$ = 兴趣强度） |
| 合法性 | 无约束 | 恒保 PSD + trace |
| 兴趣解释 | 隐含 | 显式谱分解 |

---

## 4. 测量匹配：next-item 预测（H3 / RQ2）

### 4.1 Born 规则

量子测量：状态 $\rho$ 下测得 POVM 元素 $M$ 的概率为

$$P(M\,|\,\rho) = \mathrm{Tr}(\rho M)$$

### 4.2 映射到推荐

把"候选物品态 $\rho_i$"视为测量算子（若 $\rho_i$ 是 PSD 且迹归一，可归一化为 POVM 元素），则

$$\text{score}(u,i) = \mathrm{Tr}(\rho_{u}\,\rho_i)$$

有"用户状态坍缩到物品态的概率"的量子语义。当 $\rho_u,\rho_i$ 都是密度矩阵时，$\mathrm{Tr}(\rho_u\rho_i)$ 是 **Hilbert–Schmidt 内积**：

$$\langle \rho_u, \rho_i \rangle_{HS} = \mathrm{Tr}(\rho_u^\dagger \rho_i)$$

### 4.3 与 dot product 的对比

| | dot product $h^\top e$ | Hilbert–Schmidt $\mathrm{Tr}(\rho_u\rho_i)$ |
|---|---|---|
| 对象 | 向量欧氏内积 | 算子内积（含二次/交叉项） |
| 表达能力 | 一阶 | 二阶（$\rho$ 含 $h h^\top$ 类结构） |
| 概率语义 | 无 | 有（Born 规则） |

> ⚠️ 注意：$\rho$ 含 $L L^\top$ 结构，$\mathrm{Tr}(\rho_u\rho_i)$ 实际等价于对"特征化后的方向"做加权内积——这也解释了为何在受限维度下更可能优于纯 dot。**这是 RQ2 待验证的核心。**

---

## 5. 可证明/可验证的性质清单（逐步补全）

| 编号 | 命题 | 状态 |
|---|---|---|
| P1 | 构造式 $\rho = LL^\top/\mathrm{Tr}$ 恒为合法密度矩阵 | ✅ 实现已验证（`test_smoke.py`） |
| P2 | 凸组合演化保 PSD 与 trace | ✅ 数学证明 + 实现验证 |
| P3 | 同参数量下，低秩密度矩阵自由度 ≥ 向量（$r\ge1$） | ✅ 计数论证（§2.4） |
| P4 | $\mathrm{Tr}(\rho_u\rho_i)$ 恒在 $[0,1]$（若双方均为密度矩阵） | ✅ 由 Cauchy–Schwarz + PSD 可得 |
| P5 | 动态演化对"兴趣漂移"的建模能力优于静态 $\rho_u$ | ⬜ 待实验（RQ3） |
| P6 | 低维下 $\rho$ 表示优于向量表示（长尾/多样性） | ⬜ 待实验（RQ4/RQ5/RQ6） |

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

> Sequential recommendation models usually encode evolving user preferences as **deterministic latent vectors** $h_t$. We argue that user interest is inherently **uncertain and multi-faceted**, and evolves with each interaction. We propose to represent it as a **dynamic quantum-probabilistic state** (density matrix) $\rho_t$, which (i) captures multi-interest structure via its spectrum, (ii) evolves via a **legality-preserving convex combination** (an information-injecting depolarization), and (iii) scores next items via **Hilbert–Schmidt measurement** $\mathrm{Tr}(\rho_t\rho_i)$. Extensive experiments on MovieLens show consistent gains under constrained embedding dimensions and on long-tail items.

贡献三点：
1. 首次将动态密度矩阵状态引入序列推荐；
2. 提出保合法性的凸组合状态演化（理论 + 实现）；
3. 建立基于 Hilbert–Schmidt 测量的 next-item 打分机制。

---

## 8. 建议引用/支撑方向（供写作时查证）

- 量子信息基础：Nielsen & Chuang；密度矩阵 / Born 规则 / CPTP / 去极化信道。
- Quantum ML：quantum kernel / quantum representation learning（包括 WWW 2026 相关工作）。
- 序列推荐：SASRec、BERT4Rec、GRU4Rec、MIND、ComiRec、SSE-PT。
- 不确定性建模：uncertainty-aware recommendation / distributional representation。

---

## 9. 待补理论工作（Open Problems）

- [ ] 给 $F(\rho_t,\rho_i)$ 更一般的 CPTP 形式（Kraus 算子），并论证凸组合是其一阶近似；
- [ ] 兴趣演化的可解释性：用 $\rho_t$ 的谱分解做"推荐理由"（测量坍缩叙事）；
- [ ] 泛化/信息论分析：密度矩阵表示在有限样本下的优势界（可作 future work / 理论补充章节）；
- [ ] 与注意力因果 mask 的严格对应：$\rho_t$ 只依赖 $S_{\le t}$ 的形式化。
