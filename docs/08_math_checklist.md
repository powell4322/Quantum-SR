# 理论数学基础核查清单（MATH_CHECKLIST）

> **用途**：列出论文所有数学命题，供逐条严格验证（查教材 / 问 GPT / 问导师）。
> 每项标注：✅ 标准结论（可引用）｜⚠️ 需核实 / 有条件成立｜❌ 可能有问题（写论文前必须解决）。
> 完整理论见 `docs/03_theory.md`。

---

## A. 密度矩阵基础（均为标准结论，可引 Nielsen & Chuang）

| # | 命题 | 结论 | 核查点 |
|---|---|---|---|
| A1 | 密度矩阵三公理：Hermitian、PSD、$\mathrm{Tr}(\rho)=1$ | ✅ | 标准定义，直接引用 |
| A2 | 谱分解 $\rho=\sum_k\lambda_k|\psi_k\rangle\langle\psi_k|$，$\lambda_k\ge0,\sum_k\lambda_k=1$ | ✅ | 标准（谱定理） |
| A3 | 纯态(rank-1) vs 混合态(rank>1) | ✅ | 标准 |
| A4 | "特征值 = 兴趣概率质量" | ⚠️ | **这是我们加的映射解释**，不是数学结论；论文要说明这是设计动机，非定理 |

## B. 构造 $\rho=LL^\top/\mathrm{Tr}(LL^\top)$（需核实）

| # | 命题 | 结论 | 核查点 |
|---|---|---|---|
| B1 | $LL^\top\succeq0$ 恒成立 → $\rho$ PSD | ✅ | 任意 $L$ 的 $LL^\top$ 都 PSD（标准） |
| B2 | 除以迹后 $\mathrm{Tr}(\rho)=1$ | ✅ | 除 $LL^\top=0$ 时未定义（代码 clamp_min(1e-8)）；**数学上需注明默认 $L\ne0$** |
| B3 | $LL^\top$ 实对称 → Hermitian | ✅ | 实数下对称=自伴 |
| B4 | 实际 $\mathrm{rank}(\rho)\le\min(d,r)$ | ⚠️ | $L$ 秩不足时 rank 更小（不是永远 rank=r） |
| B5 | $\mathrm{dof}(\rho)=dr-\frac{r(r-1)}{2}-1$ | ⚠️ | **关键核查点**：① 假设 $L$ 满秩且旋转群 $O(r)$ 作用自由；**$\mathrm{rank}(L)<\min(d,r)$ 时歧义群更大，公式可能失效**；② $r=1\to d-1$（球面）✅；③ $r=d\to\frac{d(d+1)}{2}-1$ ✅；④ **dof ≠ 表达能力**，审稿人可质疑"自由度多≠性能好"，需实验(A-1)佐证；⑤ **dof ≠ 参数量(dr)**，论文须区分 |

## C. 演化（凸组合，核心理论贡献）

| # | 命题 | 结论 | 核查点 |
|---|---|---|---|
| C1 | $\rho_{t+1}=\alpha\rho_t+(1-\alpha)\rho_{\text{inject}},\ \alpha\in[0,1]$ | ⚠️ | 🔴 **跨文档不一致（最高优先）**：`03_theory` §3 用 $\rho_{i_t}$（item 状态）；`01_paper_progress` §3.5 Method v3 用 $\rho_{\text{obs}}=\mathrm{Proj}(h_t)$（编码器输出）；**代码用 $\mathrm{Proj}(h_t)$**。**必须统一** |
| C2 | 凸组合保合法性：PSD + trace | ✅ | PSD 集合是凸锥（标准）；$\alpha\in[0,1]$（learnable α 用 sigmoid 保证；fixed α 需限定） |
| C3 | $\alpha$ 可学习时是否坍缩到 0/1 | ⚠️ | 设计风险，需实验观察（见 `02_research_log` §4.5） |
| C4 | 固定 α 展开：$\rho_T=\alpha^{T-1}\rho_1+(1-\alpha)\sum_{k=1}^{T-1}\alpha^{T-1-k}\rho_k$ | ✅ | 递推展开已手工验证（T=3 成立）；旧项权重小=遗忘 ✅；⚠️ 展开式里是各步状态 $\rho_k$ 而非 item 状态——**再次指向 C1 的不一致** |
| C5 | "语义上等价于贝叶斯滤波信息合并" | ❌ | **风险**：贝叶斯更新是乘法（似然×先验），凸组合是加法，**不是严格等价**。建议弱化为"结构上类似 EMA / 遗忘机制"，勿声称贝叶斯等价 |
| C6 | 与 GRU 区别：合法性硬约束 | ✅/⚠️ | GRU hidden state 确实无 PSD/trace 约束 ✅；但"有约束更好"须实验（E004 约束消融）支撑，不能仅凭理论 |

## D. 打分（Hilbert–Schmidt）

| # | 命题 | 结论 | 核查点 |
|---|---|---|---|
| D1 | $\mathrm{Tr}(\rho_u\rho_i)$ 是 HS 内积 $\langle A,B\rangle_{HS}=\mathrm{Tr}(A^\dagger B)$ | ✅ | HS 内积定义用 $A^\dagger$（共轭转置）；实对称 ρ 下 $A^\dagger=A$；论文写"实对称算子下的 HS 内积" |
| D2 | 纯态桥接：$\mathrm{Tr}(\rho_u\rho_i)=(u\cdot i)^2=\cos^2\theta$（$\rho_u=uu^\top,\rho_i=ii^\top,\|u\|=\|i\|=1$） | ✅ | 展开验证通过；**仅 r=1（纯态）成立**，r>1 非简单平方余弦，论文须注明 |
| D3 | $\mathrm{Tr}(\rho\sigma)\in[0,1]$（双方密度矩阵） | ✅ | 下界：$\mathrm{Tr}(\rho\sigma)=\mathrm{Tr}(\rho^{1/2}\sigma\rho^{1/2})\ge0$；上界：Cauchy–Schwarz(HS)+$\mathrm{Tr}(\rho^2)\le1$。**论文须写出这两步** |
| D4 | "Tr 等价于对特征化方向做加权内积" | ⚠️ | 解释性声称，不严格；可弱化或删 |
| D5 | 打分有效性优于 dot | ⚠️ | 待实验（匹配消融），非理论结论 |

## E. 熵 / 初始态

| # | 命题 | 结论 | 核查点 |
|---|---|---|---|
| E1 | $H(\rho)=-\mathrm{Tr}(\rho\log\rho)$（von Neumann entropy） | ✅ | 标准定义；需 ρ 可对角化（Hermitian 保证）；$0\le H\le\log d$ |
| E2 | 0 特征值时的 log | ⚠️ | $\log\rho$ 在 ρ 奇异时未定义；惯例 $0\cdot\log0\to0$，论文须注明 |
| E3 | $\rho_0=\frac{I}{d}$（最大混合态/无信息先验） | ✅ | $I/d$ 是密度矩阵（PSD、Tr=1），且为最大熵态 |
| E4 | $\rho_0$ 是否实现 | ⚠️ | 🔴 **代码未实现 $\rho_0$**（dynamic 从 $\rho_1=\mathrm{Proj}(h_1)$ 开始）；Method v3 定义了 $\rho_0$——**Method 与实现不一致**，须决定（实现 ρ0 或改 Method） |

## F. 跨文档不一致点（最高优先核查）

| # | 不一致 | 涉及 | 处理 |
|---|---|---|---|
| F1 | 演化注入对象：item 状态 vs 编码器输出 $\mathrm{Proj}(h_t)$ | 03 §3 / 01 §3.5 / 代码 | **统一**（建议以代码为准：$\rho_{\text{obs}}=\mathrm{Proj}(h_t)$，语义=当前观测+上下文） |
| F2 | $\rho_0=I/d$ 未在代码实现 | 01 §3.5 / model.py | 决定：实现 ρ0 或改 Method |
| F3 | "贝叶斯等价"声明过强 | 03 §3.4 | 弱化为"EMA / 遗忘机制"类比 |

## G. 待引用支撑（写作时查证）

- 密度矩阵 / 谱分解 / HS 内积 / von Neumann entropy：**Nielsen & Chuang**（标准）
- PSD 凸锥、Cauchy–Schwarz：线性代数 / 凸优化教材
- dof 计数（若保留 B5）：低秩流形 / 格拉斯曼流形（Grassmannian）文献，或自行严格推导

---

> ⚠️ **结论**：核心数学（PSD 构造、凸组合保合法、HS 打分、Tr∈[0,1]、熵）基本是标准结论 ✅；**真正需要重点核查的是 F1/F2（实现与文档不一致）和 C5（贝叶斯声明过强）、B5（dof 的 rank 退化条件）**。
