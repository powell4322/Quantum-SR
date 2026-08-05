# 理论定稿 v1（THEORY_V1）：Dynamic Preference Density State Model

> **本文档是统一、严谨的理论基础**（2026-08-03 定稿）。解决 `08_math_checklist.md` 中的 F1/F2 不一致；**所有公式、命题、猜想以此为准**。修改代码/写论文前先参照本文件。
> 配套：`08_math_checklist.md`（逐条核查）、`03_theory.md`（理论支撑过程版，后续精简并指向本文）。

---

## 1. 记号与任务设定

- 物品集合 $\mathcal{I}$，item embedding $e_i\in\mathbb{R}^d$。
- 用户 $u$ 的交互序列 $S_u=(i_1,\dots,i_T)$。
- 编码器（SASRec）$f_\theta$ 输出因果表示 $h_t=f_\theta(S_{\le t})\in\mathbb{R}^d$（$h_t$ 只依赖 $S_{\le t}$）。
- **任务**：给定 $S_u$，对候选物品 $i$ 打分 $s(u,i)$ 并排序，目标是命中 $i_{T+1}$（next-item prediction）。

---

## 2. 核心对象（统一）

**Preference Density State** $\rho_t$：用户 $u$ 在时刻 $t$ 的兴趣状态，是一个**密度算子**：
$$\rho_t\in\mathcal{D}_d:=\{\rho\in\mathbb{R}^{d\times d}\mid \rho=\rho^\top,\ \rho\succeq0,\ \mathrm{Tr}(\rho)=1\}$$

- 语义：$\rho_t$ 的谱 $\sum_k\lambda_k|\psi_k\rangle\langle\psi_k|$ 表示"用户当前**可能有哪些**兴趣（$\lambda_k$=强度）+ 不确定性"。
- ⚠️ 不承诺"每个特征向量=一个语义兴趣"（仅作潜在分解）。

---

### 3. 状态构造：StateProjection（Proj / density-state constructor）

$$\mathrm{Proj}(h)=\frac{LL^\top}{\mathrm{Tr}(LL^\top)},\qquad L=\mathrm{reshape}(W h)\in\mathbb{R}^{d\times r}$$

- **维度（v1.1 修正，与代码一致）**：$h\in\mathbb{R}^d$，$W\in\mathbb{R}^{(dr)\times d}$（线性层输出 $dr$ 维），$\mathrm{reshape}(Wh)$ 成 $L\in\mathbb{R}^{d\times r}$；实现中为 $L\in\mathbb{R}^{r\times d}$、$\rho=L^\top L$，数学等价。
- $r$ = 低秩参数（$r=1$ 纯态 / $r>1$ 混合态），rank 可控。
- 数值：对 $\mathrm{Tr}(LL^\top)$ 加 $\mathrm{clamp}(\cdot,10^{-8})$ 防除零（**默认 $L\ne0$**，P1 前提）。
- ⚠️ **命名**：Proj 是"密度状态构造器"，**并非数学上的投影算子**（论文中写作 $\Phi$ 或 density-state constructor）。
- item 状态：$\rho_i=\mathrm{Proj}(e_i)$（与用户状态共享或独立 Proj，见实验配置）。
- **复杂度（v1.1）**：参数增量 $O(dr)$（Proj 层）；打分 $O(d^2)$（$\mathrm{Tr}(\rho_u\rho_i)$ 逐元素乘加）；序列演化 $O(Td^2)$。$d$ 通常 $\le100$，可接受。

**与向量表示的关系（v1.2，删除 dof 公式）**：
- 嵌入映射 $\phi(h)=hh^\top$（rank-1）：$\phi:\mathbb{R}^d\to\mathcal{D}_d$ → **vector 是 density state 的特例**（$\phi(\mathbb{R}^d)\subset\mathcal{D}_d$）。
- $r=1$ 时打分退化为平方余弦（P3）：*vector similarity is a special case of density similarity*。
- $r>1$（混合态）表示**潜在偏好方向上的分布**（谱 $\sum_k\lambda_k v_kv_k^\top$），对应 multi-interest + uncertainty（H4）。
- 表述：*density state provides a structured second-order representation with controllable rank*。

---

## 4. 状态演化：Transition（统一 F1/F2）

**观测状态**：$\hat\rho_t=\mathrm{Proj}(h_t)$（编码器当前步输出投影 = "当前观测+上下文"诱导的状态）。

**初始状态**：$\rho_0=\frac{I}{d}$（最大混合态 = 无信息先验）。

**演化**（固定 $\alpha\in[0,1]$，主）：
$$\rho_t=\alpha\,\rho_{t-1}+(1-\alpha)\,\hat\rho_t,\qquad t=1,\dots,T$$
- **$\alpha$ 设置（v1.1 讨论）**：$\alpha$ 大 → 保留旧兴趣（稳定用户）；$\alpha$ 小 → 快速吸收新兴趣。第一版建议**固定 $\alpha$ 扫描 0.1–0.9**（E003），learnable 作选项；**$\alpha$ 近 1 时首观测 $\hat\rho_1$ 对先验 $I/d$ 的贡献小（权重 $(1-\alpha)$），$\alpha$ 近 0 时近乎忽略先验**——需在实验中观察。
- learnable / adaptive $\alpha_t=\sigma(W[h_t;e_{i_t}]+b)$ 作 extension（不参与第一版）。
- **用户状态**：$\rho_u=\rho_T$（演化末状态；聚合变体 $\rho_u=\sum_t w_t\rho_t$ 为可选对照）。

---

## 5. 打分：Hilbert–Schmidt 相似度

$$s(u,i)=\mathrm{Tr}(\rho_u\,\rho_i)$$
- 本质是 $\mathbb{R}^{d\times d}$ 上的 HS 内积 $\langle A,B\rangle=\mathrm{Tr}(A^\top B)$（实对称情形，$A^\dagger=A^\top$），即 **operator-level similarity kernel**。

---

## 6. 训练目标（Loss）

- **主 = BPR**：$\mathcal{L}=-\frac{1}{|\Omega|}\sum_{(u,i,i^-)\in\Omega}\log\sigma\big(s(u,i)-s(u,i^-)\big)$（与 SASRec/BERT4Rec/GRU4Rec 可比）。
- **消融 = BCE（logit 变换）**：$z=\log\frac{s+\epsilon}{1-s+\epsilon}$，$\mathcal{L}_{\mathrm{BCE}}=\mathrm{BCEWithLogits}(z,y)$（probabilistic interpretation）。
- **消融 = Fidelity**：正 $-\log s$、负 $-\log(1-s)$（theoretical ablation）。

---

## 7. 熵与不确定性分析（RQ4）

von Neumann 熵：$H(\rho)=-\mathrm{Tr}(\rho\log\rho)$。
- **数值实现（v1.2）**：`eigvals = torch.clamp(torch.linalg.eigvalsh(rho), min=1e-8); H = -(eigvals * torch.log(eigvals)).sum(-1)`（ρ PSD → 用 `eigvalsh`；clamp 防 0 特征值导致 NaN）。
- 低熵 $\Rightarrow$ 兴趣集中；高熵 $\Rightarrow$ 兴趣多元。
- 用于：按 $H(\rho_u)$ 分低/中/高组，比较各组下 DDS 相对 V 的增益（验证 C4）。

---

## 8. 命题（Propositions，含证明）

### P1（合法性）— 核心理论
**命题**：对任意 $t\ge0$，$\rho_t\in\mathcal{D}_d$（Hermitian、PSD、$\mathrm{Tr}=1$）。
**证明**：
1. $\mathrm{Proj}(h)=LL^\top/\mathrm{Tr}(LL^\top)$：$LL^\top$ 实对称且 PSD（任意 $L$）→ 除以迹后仍 PSD、Hermitian、$\mathrm{Tr}=1$（$L\ne0$）。∎
2. $\rho_0=I/d$：PSD、Hermitian、$\mathrm{Tr}=1$。∎
3. 凸组合：PSD 集合是凸锥 → $aA+bB\succeq0\ (a,b\ge0)$；$\mathrm{Tr}(\alpha\rho_{t-1}+(1-\alpha)\hat\rho_t)=\alpha+(1-\alpha)=1$。
4. 归纳得 $\forall t$ 合法。∎

### P2（有界打分）
**命题**：对任意 $\rho,\sigma\in\mathcal{D}_d$，$0\le\mathrm{Tr}(\rho\sigma)\le1$；且 $\mathrm{Tr}(\rho\sigma)=1$ 当且仅当 $\rho=\sigma=|\psi\rangle\langle\psi|$（同一纯态）。
**证明**：
- 下界：$\mathrm{Tr}(\rho\sigma)=\mathrm{Tr}(\rho^{1/2}\sigma\rho^{1/2})\ge0$（PSD 之积的迹非负）。∎
- 上界：HS Cauchy–Schwarz $|\mathrm{Tr}(\rho\sigma)|\le\sqrt{\mathrm{Tr}(\rho^2)\mathrm{Tr}(\sigma^2)}$；且 $\mathrm{Tr}(\rho^2)=\sum_k\lambda_k^2\le(\sum_k\lambda_k)^2=1$（$\lambda_k\ge0$）。故 $\le1$。∎
- **等号条件（v1.1 补充）**：Cauchy–Schwarz 等号 $\iff\rho,\sigma$ 线性相关；$\mathrm{Tr}(\rho^2)=1\iff\rho$ 为纯态 → $\rho=\sigma=$ 同一纯态时 $=1$（完全匹配的兴趣态，可作为可解释性质）。

### P3（纯态桥接）
**命题**：$r=1$ 时 $\rho_u=uu^\top,\rho_i=ii^\top$（$\|u\|=\|i\|=1$），则 $s=\mathrm{Tr}(\rho_u\rho_i)=(u\cdot i)^2=\cos^2\theta$。
**证明**：$\mathrm{Tr}(uu^\top ii^\top)=\mathrm{Tr}\big(u(u^\top i)i^\top\big)=(u^\top i)\,\mathrm{Tr}(ui^\top)=(u^\top i)(i^\top u)=(u\cdot i)^2$。∎
- 意义：$r=1$ 时 state 恰是**平方点积核**（state ⊇ vector 的桥梁）；也解释 $s\in[0,1]$。

### P4（遗忘 / EMA）
**命题**：**固定 $\alpha$** 时，$\rho_T=\alpha^{T}\frac{I}{d}+(1-\alpha)\sum_{t=1}^{T}\alpha^{T-t}\,\hat\rho_t$。
**证明**：递推展开（归纳）：
$\rho_1=\alpha\frac{I}{d}+(1-\alpha)\hat\rho_1$；
假设 $\rho_t=\alpha^t\frac{I}{d}+(1-\alpha)\sum_{k=1}^t\alpha^{t-k}\hat\rho_k$，则 $\rho_{t+1}=\alpha\rho_t+(1-\alpha)\hat\rho_{t+1}=\alpha^{t+1}\frac{I}{d}+(1-\alpha)\sum_{k=1}^{t+1}\alpha^{t+1-k}\hat\rho_k$。∎
- **推论（遗忘机制）**：较早观测 $\hat\rho_t$ 的权重 $\propto\alpha^{T-t}$ 随 $T-t$ 指数衰减 → 旧兴趣自然遗忘。

### P5（初始态最大熵）
**命题**：在 $\mathcal{D}_d$ 中，$\rho_0=I/d$ 是熵最大的状态，$H(\rho_0)=\log d$。
**证明**：von Neumann 熵是凹函数，在均匀分布（$I/d$）取最大 $\log d$（Jensen）。∎

### 约束诱导正则化（v1.2 论证，替代"更多自由度"）
普通向量 EMA $h_t=\alpha h_{t-1}+(1-\alpha)e_t$ 在 $\mathbb{R}^d$ 中无约束；密度状态演化在 $\mathcal{D}_d$ 中，而 $\mathcal{D}_d$ 是**紧凸集**：
1. **范数有界**（状态不会爆炸）；
2. **相似度有界** $0\le\mathrm{Tr}(\rho_u\rho_i)\le1$（P2）；
3. **演化不发散**（凸组合仍在 $\mathcal{D}_d$）。
→ 表述：*legality-preserving transition acts as implicit regularization, providing a stable state space*（不声称提高性能，效果由 E004 实验检验）。

---

## 9. 假设（Hypotheses，实验可验证；v1.1：由"猜想"改为"假设"）

| # | 假设 | 合理理由 | 验证实验 |
|---|---|---|---|
| **H1** 表示优势 | 同参数量下，DS/DDS 优于 V 与 DF（RQ1） | rank-1 嵌入含向量（P3）+ 混合态表示不确定性分布（§3）+ 约束正则化（§8） | E001 |
| **H2** 演化优势 | DDS 优于 DS（RQ2） | 显式遗忘/惯性（P4）+ 合法演化（P1） | E002 |
| **H3** 约束价值 | 移除 PSD/trace 后性能下降（RQ3） | 合法性约束排除病态演化（norm explosion 等）；⚠️ 去约束后状态无界、得分尺度变，需重调损失 | E004（消融） |
| **H4** 不确定性关联 | 高熵用户上 DDS 增益更大（RQ4） | 高熵=兴趣多元=更需要概率状态表达 | entropy 分组 |

> **baseline 名称（v1.1 明确）**：**V** = SASRec dot product；**DF** = density feature（DMPEN 式，$ee^\top$→flatten→SASRec）；**DS** = 直接用 $\hat\rho_T=\mathrm{Proj}(h_T)$ 作 $\rho_u$（无演化）；**DDS** = 演化得到 $\rho_u$；**VE** = vector evolution（$h_{t+1}=\alpha h_t+(1-\alpha)e_i$）。
> 假设须由实验证实或证伪；若 H3/H4 不成立，相应贡献删除/弱化。

---

## 10. 研究基础与边界

### 10.1 理论基础
- 密度算子、谱分解、HS 内积、von Neumann 熵 → **Nielsen & Chuang**（标准引用）。
- PSD 凸锥、Cauchy–Schwarz、凸组合 → 线性代数 / 凸分析。
- EMA / 遗忘机制 → 时间序列 / 信号处理常识。

### 10.2 与已有工作的边界
| 工作 | 它的做法 | 我们的不同（数学依据） |
|---|---|---|
| **DMPEN (2019)** | density 作为**二阶特征** $\rho=e_ie_i^\top$ 送 RNN | density 作为**状态**（P1 合法性）+ 保合法演化（P4），非特征；我们问"状态如何被合法演化" |
| **WWW 2026** | quantum embedding/matching for **静态 CF** | 我们做**序列动态状态演化**（时间维） |
| **Gaussian embedding** | $z\sim\mathcal N(\mu,\Sigma)$ | 二阶算子 + **硬约束**（PSD+trace）+ 合法演化；Gaussian 无约束 |
| **MIND/ComiRec** | 多个独立向量 | 单个密度状态的**谱**统一多兴趣，且有序演化 |

### 10.3 任务/评估基础
- next-item prediction；NDCG@10 / HR@10（+ AUC 可作附表）；BPR 训练（SASRec/BERT4Rec/GRU4Rec 惯例）。

---

## 11. 与核查清单对照（F1/F2 已解决）

| 问题 | 本定稿的决定 |
|---|---|
| F1 演化注入对象 | **统一为 $\hat\rho_t=\mathrm{Proj}(h_t)$**（编码器输出投影）；03_theory 将同步，代码已一致 |
| F2 $\rho_0$ | **$\rho_0=I/d$ 纳入演化**（代码需实现：dynamic 的 prev 初始化为 $I/d$，$\rho_1=\alpha\frac{I}{d}+(1-\alpha)\hat\rho_1$） |
| C5 贝叶斯声明 | 已删除"贝叶斯等价"，仅保留 EMA/遗忘（P4 推论） |
| B5 dof | 降级为"表述性"（§3），不依赖其精确性 |

> ⚠️ **下一步**：按本定稿实现 $\rho_0$（代码小改）→ 用 `test_smoke.py` 回归 → 再跑 V/VE/DF/DS/DDS 验证 C1–C4。
