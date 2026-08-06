# 研究定位 v2（RESEARCH_POSITIONING_V2）— 唯一 Source of Truth

> **状态（2026-08-06，最终冻结）**：本文是**研究定位的唯一 source of truth**，论文与代码实现均以此为基准。
> 相对旧文档的关键修正：
> - ❌ 删除 **Quantum / Entanglement / Fidelity** 主线；
> - ❌ 不再使用 **QDM-Former** 作为核心；
> - ✅ 核心定位改为 **Dynamic Density State Transformer (DDST)**；
> - ✅ **SASRec** 作为唯一基础 backbone；
> - ✅ 其他模型均作为**递进消融**，不改变 backbone；
> - ✅ 明确理论贡献、实验目标、实现路线。

---

# Dynamic Density State Transformer for Sequential Recommendation

## 1. Research Positioning

### Final Research Direction
**Dynamic Density State Modeling for Sequential Recommendation**

核心思想：

> 将用户兴趣从 deterministic vector hidden state 转换为 uncertainty-aware density state，并设计保持合法性的动态演化机制，用于建模序列兴趣变化。

模型名称：

**DDST**（Dynamic Density State Transformer）

Backbone：

$$
\boxed{\text{SASRec}}
$$

所有实验均基于 SASRec 修改，**不替换 Transformer 架构**。

---

## 2. Research Motivation

### 2.1 Problem
当前 Sequential Recommendation 方法：

$$
h_t = f_\theta(S_{\le t}),\qquad h_t \in \mathbb{R}^d
$$

用户兴趣被压缩为**单一向量**。然而真实用户行为具有：

1. **Multi-interest**：用户同时存在多个兴趣（如 movie / sport / technology）；
2. **Preference uncertainty**：历史行为不足时，兴趣存在不确定性；
3. **Temporal evolution**：兴趣随时间变化。

### 2.2 Limitation of Existing SASRec
SASRec：$h_t=\mathrm{Transformer}(S_{\le t})$

- 优势：temporal dependency；
- 不足：
  - hidden state 没有显式概率结构；
  - vector representation 只能表达一阶关系；
  - 用户状态变化缺少显式动态约束。

---

## 3. Core Idea

### Preference Density State
将 $h_t$ 转换为 $\rho_t$，其中 $\rho_t \in \mathcal{D}_d$，满足：

$$
\rho=\rho^\top,\qquad \rho\succeq 0,\qquad \mathrm{Tr}(\rho)=1
$$

即**合法 density operator**。

---

## 4. Methodology

### 4.1 Density State Projection
输入：SASRec hidden state $h_t$；

生成：

$$
L_t = W h_t \quad \text{(reshape 到 } \mathbb{R}^{d\times r}\text{)}
$$

构造：

$$
\hat\rho_t = \frac{L_t L_t^\top}{\mathrm{Tr}(L_t L_t^\top)}
$$

- $d$：embedding dimension；$r$：low-rank。
- 作用：将 vector hidden state 映射为**二阶状态**。

### 4.2 Dynamic Density Evolution（核心贡献）
定义：

$$
\rho_t = \alpha\rho_{t-1} + (1-\alpha)\hat\rho_t
$$

初始化：

$$
\rho_0 = \frac{I}{d}
$$

- $\rho_{t-1}$：历史兴趣状态；$\hat\rho_t$：当前观测状态。
- 含义：类似 EMA（old preference + new observation），**但不同**：
  - vector EMA：$h_t=\alpha h_{t-1}+(1-\alpha)e_t$，**无法保证结构**；
  - density evolution：**始终满足 $\rho_t\in\mathcal{D}_d$**（合法性保持）。

### 4.3 Preference Matching
- 用户状态：$\rho_u=\rho_T$；
- item：$\rho_i=\mathrm{Proj}(e_i)$；
- score：

$$
s(u,i)=\mathrm{Tr}(\rho_u\rho_i)
$$

即 **Hilbert–Schmidt similarity**。

### 4.4 Training Objective
主 Loss：**BPR**

$$
\mathcal{L}=-\sum \log\sigma\big(s(u,i^+)-s(u,i^-)\big)
$$

原因：Sequential Recommendation 常用 pairwise ranking；**BCE 仅作为 ablation**。

---

## 5. Theoretical Foundation

### Proposition 1: Legality Preservation
对于任意 $\rho_{t-1},\hat\rho_t\in\mathcal{D}_d$：

$$
\rho_t = \alpha\rho_{t-1} + (1-\alpha)\hat\rho_t \in \mathcal{D}_d
$$

原因：**Density matrix 集合是凸集**。

贡献：提供 **constrained preference evolution**。

### Proposition 2: Vector Bridge
当 $r=1$：$\rho = uu^\top$，因此：

$$
\mathrm{Tr}(\rho_u\rho_i) = (u^\top i)^2
$$

说明：**density state generalizes vector similarity**。

### Proposition 3: Evolution Interpretation
展开：

$$
\rho_T = \alpha^T \frac{I}{d} + (1-\alpha)\sum_{t=1}^{T}\alpha^{T-t}\hat\rho_t
$$

说明：**旧兴趣指数衰减**。

---

## 6. Contribution

### C1. Density State Representation
> uncertainty-aware density state representation for sequential recommendation.

- Existing：vector hidden state；
- Ours：**second-order operator state**。

### C2. Legality-preserving Dynamic Evolution
> constrained state transition on density manifold.

- Existing：unconstrained hidden update；
- Ours：**PSD + trace-preserving evolution**。

### C3. Operator Similarity
> $\mathrm{Tr}(\rho_u\rho_i)$ 替代 $h_u^\top h_i$，能够捕获 **second-order interaction structure**。

---

## 7. Experimental Framework

### Backbone
所有模型：$\boxed{\text{SASRec}}$，保持：

- same Transformer layers；
- same embedding size；
- same optimizer；
- same dataset。

**只改变 representation / state mechanism**。

---

## 8. Model Comparison

| Model | Representation | Evolution | Similarity | Purpose |
|---|---|---|---|---|
| **M0** SASRec | vector ($h$) | Transformer | dot | baseline |
| **M1** Density Feature | ($ee^\top$) | none | trace | test second-order representation |
| **M2** Density State | ($\mathrm{Proj}(h)$) | none | trace | test density state |
| **M3** DDST | ($\rho_t$) | **density evolution** | trace | **main model** |
| **M4** Operator Attention | density Q/K/V | Transformer attention | trace attention | **future extension** |

---

## 9. Research Questions

### RQ1 Representation
Does density state outperform vector representation?
Experiment：$M0\ \mathrm{vs}\ M2$

### RQ2 Evolution
Does explicit density transition improve sequential modeling?
Experiment：$M2\ \mathrm{vs}\ M3$

### RQ3 Matching
Does operator similarity outperform vector dot?
Experiment：same state，$dot\ \mathrm{vs}\ \mathrm{Tr}(\rho_u\rho_i)$

### RQ4 Uncertainty
Does entropy indicate recommendation uncertainty?
Compute：$H(\rho)=-\mathrm{Tr}(\rho\log\rho)$；Analysis：high entropy users → DDS improvement?

---

## 10. Complexity Analysis

假设：$d=64,\ r=8$。

- **Projection**：$O(dr)$；
- **Density construction**：$O(d^2 r)$；
- **Trace matching**：naive $O(d^2)$；**low-rank**：

$$
\mathrm{Tr}(L_u L_u^\top L_i L_i^\top) = \lVert L_u^\top L_i\rVert_F^2,\quad O(d r^2)
$$

Therefore **scalable**.

---

## 11. Implementation Plan for Agent

- **Step 1**：保持 SASRec baseline；完成 ML-1M reproduction。
- **Step 2**：新增 `DensityProjection`（`model.py`）：hidden state → $\rho$；测试 PSD + trace=1。
- **Step 3**：实现 `DensityState`，加入 `rho_prev`：$\rho=\alpha\rho_{prev}+(1-\alpha)\rho_{new}$。
- **Step 4**：实现 Trace scorer：`score=torch.sum(rho_u*rho_i)` 替换 `score=h_u @ h_i`。
- **Step 5**：运行 `SASRec / DensityState / DynamicDensityState`，比较 **R@10 / NDCG@10**。

> 注：指标统一 **R@10（Recall@10）与 NDCG@10**（本协议下 HR@10 ≡ Recall@10）。

---

## 12. 时间成本估计（基于当前 SASRec 工程）

| 实验 | 修改量 | 预计时间 |
|---|---|---|
| SASRec baseline | 已有 | 0.5-1 天 |
| Density Projection | 新增模块 | 1-2 天 |
| Density State | 增加状态 | 2-3 天 |
| Dynamic Evolution | 核心实现 | 2-3 天 |
| Entropy analysis | 分析代码 | 1 天 |
| Full experiments ML-1M | 训练 | 2-3 天 |
| Amazon/ML20M 验证 | 训练 | 3-7 天 |

---

# Final Freeze

## 保留
- ✅ SASRec backbone
- ✅ Density representation
- ✅ Dynamic density evolution
- ✅ Trace similarity
- ✅ Entropy analysis

## 删除
- ❌ Quantum embedding
- ❌ CNOT entanglement
- ❌ Fidelity attention
- ❌ Quantum advantage claim

## 最终论文定位

> **A Dynamic Density State Transformer that models sequential preference evolution through legality-preserving operator states.**
