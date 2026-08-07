# Agent Research Plan（DDST 阶段执行计划）

> **状态（2026-08-07）**：第一轮 ml-1m 全量结果已出（见 `02_research_log.md` §6）。本计划冻结模型结构，**只做诊断 + ablation**，用实验决定最终模型，不提前引入修正项。
> 关联：`docs/09_research_positioning_v2.md`（定位 source of truth）、`docs/05_experiment_plan.md`（实验设计）。

---

## 0. 当前阶段判断

**当前实验结果不能判定 Density State 方法失败。**

第一轮实验验证的是 $\rho=\frac{LL^\top}{\mathrm{Tr}(LL^\top)},\ rank=1$，并采用 $\mathrm{score}=\mathrm{Tr}(\rho_u\rho_i)$。该配置存在**两个明确限制**：

1. **rank=1 导致密度矩阵退化为纯态**：$\rho=ll^\top$，因此 $\mathrm{Tr}(\rho_u\rho_i)=(l_u^\top l_i)^2$，本质退化为**平方 cosine similarity**。
2. **trace normalization 丢失 embedding norm 信息**：
   - 原始 SASRec：$\mathrm{score}=h_u^\top e_i$，同时包含 direction similarity 与 preference strength；
   - density：$\rho=\frac{LL^\top}{\|L\|^2}$ 只保留 direction → 用户兴趣强度丢失、score 分布压缩、BPR 优化困难。

> 结论：**Naive Density State（normalized rank-1 operator）不能直接替代 vector embedding**，而非 **Density State 无效**。

---

## 1. 研究假设（重新整理）

**Motivation**：Sequential Recommendation 用 $h_t\in\mathbb{R}^d$ 表示用户状态，但 vector 表示存在：
- **Limitation 1**：单点表示（single point），无法表达多兴趣 / 不确定性 / 兴趣混合状态；
- **Limitation 2**：缺少二阶关系，$h_i^\top h_j$ 只描述一阶相似。

因此提出将用户兴趣表示为 $\rho_t$（$\rho_t\succeq0,\ \mathrm{Tr}(\rho_t)=1$），即 **operator-valued user state**。

---

## 2. 最终模型体系（冻结，不再扩数量）

| Model | Purpose | Contribution |
|---|---|---|
| M0 SASRec | Baseline | vector sequential modeling |
| M1 Density Feature | Representation ablation | 验证二阶 feature |
| M2 Static Density State | State representation | 验证 operator state |
| **M3 Dynamic Density State Transformer** | **Main model** | 验证 state evolution |
| M4 Operator Attention | Interaction ablation | 验证 operator matching |

---

## 3. 论文核心贡献（暂定）

- **C1 Low-rank Density State Representation**：$\rho_t=\frac{L_tL_t^\top}{\mathrm{Tr}(L_tL_t^\top)}$，将兴趣从 vector space 扩展到 operator space（保留二阶统计、支持 multi-interest、低秩 $O(dr^2)$）。
- **C2 Dynamic Density Evolution**：$\rho_t=\alpha\rho_{t-1}+(1-\alpha)\rho_{v_t}$（state transition system，捕获 long-term preference + short-term behavior）。
- **C3 Operator Space Matching**：$\mathrm{Tr}(\rho_q\rho_k)$（Hilbert–Schmidt inner product），验证 operator interaction 是否优于 vector interaction。

---

## 4. 理论修正点

不再强调 ❌ quantum / ❌ entanglement / ❌ fidelity（物理解释困难、易被质疑概念包装）。
当前定位 ✅ **operator learning** / ✅ **density matrix representation** / ✅ **sequential state modeling**。

---

## 5. 下一阶段执行计划

### Phase 0：冻结代码，只做诊断（确认失败原因）

#### 5.1 Norm Analysis
统计 $\|L\|_F$（mean / std / histogram），对比 Vector $\|h\|$ vs Density $\|L\|$，验证 normalization 是否造成信息损失。

#### 5.2 Score Distribution
比较 $h_u^\top e_i$ vs $\mathrm{Tr}(\rho_u\rho_i)$ 的 mean / variance / percentile，重点判断 score 是否集中在 $[0,0.2]$。

#### 5.3 Gradient Analysis
记录最后 attention 层梯度 $\|\nabla W\|$，比较 Vector vs Density，判断是否存在梯度不足。

> 实现：`diagnose.py`（只读诊断，不改模型结构）。

### Phase 1：Rank Ablation（最高优先级）
固定 dataset=ml-1m / epoch=200 / optimizer 不变：
- Static：`state rank=1/4/8/16`
- Dynamic：`dynamic rank=4/8/16`（rank=1 已有）
记录表：`rank | NDCG | Recall | Loss`

**判定**：
- **Case A**：rank 提升 → $H1$ 成立（mixed state representation 有效）；
- **Case B**：rank 无改善 → density representation 需要重新设计。

### Phase 2：Matching Ablation
固定 `dynamic rank=8`，比较 Operator Matching $\mathrm{Tr}(\rho_u\rho_i)$ vs Vector Matching $L_u^\top L_i$，验证 $H3$（operator interaction 更好）。

### Phase 3：Confidence-aware Scoring（候选增强）
若 rank=8 有效但仍低于 SASRec，加入 $\mathrm{score}=\mathrm{Tr}(\rho_u\rho_i)\times(\|L_u\|_F\|L_i\|_F)^\gamma$（$\gamma$ 可学习）。理论解释：density 负责 preference distribution，norm 负责 preference confidence。

---

## 6. 代码修改原则

**当前禁止**：❌ 修改模型结构 / ❌ 加 gate / ❌ 改 loss / ❌ 加新模块（先验证核心假设）。

**允许**：✅ rank 参数开放（`--state_rank` 已支持） / ✅ 输出诊断指标（`diagnose.py`） / ✅ 增加实验脚本 / ✅ 保存 embedding & statistics。

---

## 7. 最终实验路线图

```
SASRec baseline
   → Density Feature
   → Static Density (rank=1/4/8/16)
   → Dynamic Density State (rank=4/8/16)
   → Operator Attention
   → Confidence-aware Enhancement
```

---

## 8. 当前论文风险判断

- **高概率成立**：✅ Dynamic state evolution 有研究价值；✅ Low-rank density representation 有理论空间；✅ Operator matching 有数学依据。
- **当前不确定**：⚠️ density 是否超过 SASRec。关键实验 = **rank=8**：若 DDST(rank=8) 接近或超过 SASRec，论文故事成立。

---

## 一句话任务总结

> 不要修模型，先证明问题来源。当前失败来自 rank=1 和 normalization，而非 density state 理论。下一步只执行诊断 + rank ablation + operator matching ablation，用实验决定最终模型，而不是提前引入修正项。
