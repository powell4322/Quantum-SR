---
name: quantum-seq-rec
description: 'Domain knowledge for the quantum-inspired sequential recommendation project (SASRec + density-matrix dynamic state). Use when designing or modifying quantum-inspired modules (StateProjection, StateTransition), choosing scoring or loss (Hilbert-Schmidt similarity vs dot product), discussing theory, reviewing RQ1-RQ3, or defending against "quantum-washing" reviewer concerns. Covers density-matrix legality (PSD, trace=1), convex-combination preference-inertia state evolution, Hilbert-Schmidt similarity scoring, and the known Tr-vs-BCEWithLogitsLoss mismatch issue.'
user-invocable: true
---

# Quantum-inspired Sequential Recommendation（领域知识）

## 核心方向（一句话）
序列推荐把用户兴趣编码为确定性向量 $h_t$；我们建模为**动态密度状态（density state）$\rho_t$**（半正定、trace=1），随每次交互按"偏好惯性"演化，并用 Hilbert–Schmidt 相似度 $\mathrm{Tr}(\rho_t\rho_i)$ 做 next-item 打分。
与 WWW 2026 *Quantum-enhanced Representation Learning and Matching Learning for Recommendation*（**静态 CF**）的差异：我们引入**时间维演化** $\rho_1\to\rho_T$。

## 密度矩阵合法性（铁律，任何实现必须满足）
1. Hermitian；2. 半正定（PSD）；3. $\mathrm{Tr}(\rho)=1$。
- **构造**：$\rho = LL^\top/\mathrm{Tr}(LL^\top)$，$L=\mathrm{Linear}(h)\in\mathbb{R}^{d\times r}$（低秩 Cholesky-like）。$r=1$ 纯态、$r>1$ 混合态；⚠️ 谱分解是"偏好不确定性的潜在分解"，**不承诺每个特征向量=一个兴趣**（RESEARCH_LOG §4.1）。实现见 `model.py::StateProjection`，合法性已由 `test_smoke.py` 校验。
- **演化**：凸组合 $\rho_{t+1}=\alpha\rho_t+(1-\alpha)\rho_{i_t}$ 保 PSD + trace（PSD 是凸锥），对应"信息注入式去极化"。实现见 `model.py::StateTransition`。

## 打分与损失（当前最重要的已知问题）
- $\rho_u,\rho_i$ 均为密度矩阵时 $\mathrm{Tr}(\rho_u\rho_i)\in[0,1]$（Hilbert–Schmidt 内积；纯态下退化为 $(u\cdot i)^2$）。
- ⚠️ **已知 bug 级问题**：`BCEWithLogitsLoss` 期望无界 logits，Tr 打分被压在 $[0,1]$ 会导致正样本损失下界 $\approx0.313$、收敛慢、指标低（`docs/02_research_log.md` §4.4）。
- **候选修正（未定，需实验对比）**：
  1. logit 变换：$\mathrm{logits}'=\log\frac{Tr}{1-Tr}$（clamp，保持 BCE 管线）；
  2. fidelity loss：正样本 $-\log Tr$、负样本 $-\log(1-Tr)$（量子 ML 惯例，语义最贴合）；
  3. 温度缩放：线性映射 logits 跨越 0。
- 设计建议：做成可切换 `--loss bce|fidelity` + `--score_transform`，把"打分/损失"本身也变成 ablation。

## 与已有工作边界
- WWW 2026 quantum CF：$\rho_u$ 静态 → 我们动态演化。
- MIND/ComiRec（multi-interest）：多向量 → 我们统一为一个密度矩阵的**谱** $\sum_k\lambda_k|\psi_k\rangle\langle\psi_k|$，有序演化。
- Uncertainty-aware rec：方差/分布 → 我们密度矩阵统一"不确定性 + 兴趣结构"。

## 详情
- 理论完整版：项目根 `docs/03_theory.md`（合法性证明、Born 规则、性质清单、论文叙事）。
- 实验与 RQ 状态：项目根 `docs/02_research_log.md`。
- 实现：`model.py`（`StateProjection` / `StateTransition` / `SASRec.variant`）。
