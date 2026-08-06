---
name: ddst-seq-rec
description: 'Domain knowledge for the DDST (Dynamic Density State Transformer) sequential recommendation project (SASRec + legality-preserving density state evolution). Use when designing or modifying density-state modules (StateProjection, StateTransition), choosing scoring/loss (Hilbert-Schmidt trace vs dot, BPR vs BCE), discussing theory (P1 legality / P2 vector bridge / P3 evolution), reviewing RQ1-RQ4, or defending "density-state is not just EMA" concerns. Unique source of truth: docs/09_research_positioning_v2.md.'
user-invocable: true
---

# DDST（Dynamic Density State Transformer）· 领域知识

## 核心方向（一句话）
序列推荐把用户兴趣编码为确定性向量 $h_t$；我们建模为**动态密度状态（density state）$\rho_t$**（半正定、trace=1），随每次交互按"合法性保持的凸组合"演化，并用 Hilbert–Schmidt 相似度 $\mathrm{Tr}(\rho_u\rho_i)$ 做 next-item 打分。

> ⚠️ 定位（2026-08-06 冻结）：**删除 Quantum / Entanglement / Fidelity 主线；不再用 QDM-Former 作为核心**。SASRec 是唯一 backbone；其他模型仅作递进消融，不换架构。**唯一 source of truth = `docs/09_research_positioning_v2.md`**。

## 核心对象：Preference Density State
$\rho_t \in \mathcal{D}_d = \{\rho:\rho=\rho^\top,\ \rho\succeq0,\ \mathrm{Tr}(\rho)=1\}$——合法 density operator，表示 uncertainty-aware preference distribution（**谱 = 不确定性/多兴趣的潜在分解**，不承诺每个特征向量=一个兴趣）。

## 密度矩阵合法性（铁律，任何实现必须满足）
1. Hermitian；2. 半正定（PSD）；3. $\mathrm{Tr}(\rho)=1$。
- **构造**：$\hat\rho = LL^\top/\mathrm{Tr}(LL^\top)$，$L=\mathrm{Linear}(h)\in\mathbb{R}^{d\times r}$（低秩 Cholesky-like）。$r=1$ 纯态、$r>1$ 混合态。实现 `model.py::StateProjection`，合法性由 `test_smoke.py` 校验。
- **演化（核心贡献 P1）**：$\rho_t=\alpha\rho_{t-1}+(1-\alpha)\hat\rho_t$，初始化 $\rho_0=I/d$；因 $\mathcal{D}_d$ 是凸集，**每步恒保 PSD+trace** → constrained preference evolution。实现 `model.py::StateTransition`。
- **向量桥接（P2）**：$r=1$ 时 $\rho=uu^\top$，$\mathrm{Tr}(\rho_u\rho_i)=(u^\top i)^2$ → density 推广 vector similarity（不写"严格包含"）。
- **演化展开（P3）**：$\rho_T=\alpha^T I/d+(1-\alpha)\sum_t\alpha^{T-t}\hat\rho_t$ → 旧兴趣指数衰减。

## 打分与损失（当前状态）
- `--matching trace`（默认）：$s(u,i)=\mathrm{Tr}(\rho_u\rho_i)$（Hilbert–Schmidt，$\in[0,1]$）→ 经 `_logit_score`（logit 变换）映射为无界 logits。
- `--matching dot`（RQ3 消融）：用一阶方向 `StateProjection.direction(h)` 做 dot。
- **主损失 = BPR**（默认）：$\mathcal{L}=-\sum\log\sigma(s^+-s^-)$；**BCE 仅作 ablation**（Tr 经 logit 变换后兼容 `BCEWithLogitsLoss`）。
- ✅ Tr/BCE 不匹配已修复（2026-08-03）：logit 变换 + 默认 BPR。
- 指标统一 **R@10 / NDCG@10**（本协议下 HR@10 ≡ Recall@10，数值等价）。

## 与已有工作边界
- WWW 2026 quantum CF：$\rho_u$ 静态 → 我们随序列时间维演化 $\rho_1\to\rho_T$。
- DMPEN (2019)：density 作**二阶特征**送 RNN → 我们作**受约束的状态**演化（density→density→density）。
- MIND/ComiRec（multi-interest）：多独立向量 → 我们统一为一个密度矩阵的谱 $\sum_k\lambda_k|\psi_k\rangle\langle\psi_k|$，有序演化。

## 模型体系（M0-M4，见 docs/05_experiment_plan.md §0）
M0 SASRec（vector/dot）→ M1 Density Feature（$ee^\top$ feature）→ M2 Density State（$\mathrm{Proj}(h)$，无演化）→ **M3 DDST（凸组合演化，main model）** → M4 Operator Attention（future extension，本期不做）。
代码映射：`vector`(M0) / `density_feature`(M1) / `state`(M2) / `dynamic`(M3) / `vector_evolve`(VE 对照)。

## RQ
- **RQ1 Representation**：M0 vs M2
- **RQ2 Evolution**：M2 vs M3（核心；**关键判据：DDS 增益 ≠ VE 的 EMA 增益**）
- **RQ3 Matching**：同状态下 dot vs $\mathrm{Tr}$（`--matching`）
- **RQ4 Uncertainty**：熵 $H(\rho)=-\mathrm{Tr}(\rho\log\rho)$ 分组（高熵用户 DDS 增益？）

## 详情
- 定位唯一 source of truth：`docs/09_research_positioning_v2.md`
- 理论完整版：`docs/09_theory_v1.md` / `docs/03_theory.md`
- 实验与 RQ 状态：`docs/02_research_log.md`
- 实验计划：`docs/05_experiment_plan.md`；服务器运行：`docs/10_server_run.md`
- 实现：`model.py`（`StateProjection` / `StateTransition` / `SASRec.variant` / `--matching` / `state_entropy`）
