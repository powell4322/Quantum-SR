# 给 GPT 的讨论稿（2026-08-03）

> 用途：把项目现状 + 待讨论问题整理成可直接复制给 GPT 的稿子。含"我的倾向判断"，请 GPT 逐条给意见并指出是否合理。

---

## 一、现状总结（我们已经做了什么）

1. **定位**：从 "Quantum Recommendation" 收敛为 **Dynamic Density-State Modeling for Uncertainty-Aware Sequential Recommendation**；量子只作数学工具（density operator / convex mixing / Hilbert–Schmidt kernel）。
2. **方法框架**（基于 SASRec 改造）：
   - `StateProjection`：低秩 Cholesky 构造密度状态 $\rho = LL^\top / \mathrm{Tr}(LL^\top)$（$L=\mathrm{Linear}(h)\in\mathbb{R}^{d\times r}$），天然 PSD + trace=1；
   - `StateTransition`：凸组合演化 $\rho_{t+1}=\alpha\rho_t+(1-\alpha)\rho_{i_t}$（保合法性；fixed / learnable 标量 α；adaptive α 待实现）；
   - 打分：Hilbert–Schmidt 相似度 $\mathrm{Tr}(\rho_u\rho_i)$。
   - variant：`vector`（=原 SASRec 基线）/ `state`（密度状态，无演化）/ `dynamic`（密度状态 + 凸组合演化）。
3. **代码验证**：冒烟测试通过（三种 variant + PSD/trace 合法性校验）；5 轮 quick 实验发现 **Tr 打分与 BCEWithLogitsLoss 不匹配**（logits 被压在 [0,1]，loss 卡高）。
4. **防撞检索 + 原文核验**：
   - **DMPEN (DASFAA 2019)**：density matrix 作为**二阶特征**（$\rho=ee^\top$，不降维）送入 basic-RNN/GRU/LSTM 建模偏好演化（Amazon+Taobao）；评价 AUC/accuracy；**无 PSD/trace 合法性讨论**，论文自承 "fail to theoretically explain density-matrix representation"。
   - **WWW 2026 quantum**：quantum-enhanced representation + matching **for CF**（静态，非序列）。
   - 我们的差异立足点：**density operator 作为"动态偏好状态"（状态机）而非"特征"** + **保合法性演化** + **next-item ranking（NDCG/HR@10）** + **提供 DMPEN 缺失的理论解释**。

## 二、已定 / 暂不动

- 论文核心贡献 C2（legality-preserving preference transition）不变。
- 目标会议 WWW；主指标 NDCG@10 / HR@10。
- 文档体系 docs/00-06 各司其职。

## 三、待讨论问题（每个附我的倾向）

### Q1. Contribution 是否要改？（我的倾向：小调，不大改）
当前 C1/C2/C3：
- C1：Density states as sequential preference representations（second-order，≠ DMPEN 特征）
- C2（核心）：Legality-preserving preference transition（保 PSD+trace 凸组合 + adaptive α）
- C3：Hilbert–Schmidt similarity + 多数据集/多 baseline 评估

我的疑虑：
- C1 与 DMPEN 的区分（"状态 vs 特征"）其实落到 C2；C1 单独站不太稳。
- DMPEN 自承缺理论解释 → 是否应把**"合法性理论 / 可解释性"**显式作为一条贡献？
- 请 GPT 建议：C1/C2/C3 是否需要调整成更锋利的三条？"理论贡献"放哪里？

### Q2. Baseline 集（GRU4Rec 保留还是换 Caser？DMPEN 如何公平对比？）
- DMPEN 是 RNN-based，我们是 Transformer-based（SASRec）。跨架构对比的公平性如何处理？
- 我的倾向：**不换而是加**——GRU4Rec（RNN）保留（且与 DMPEN 同族，便于解释 DMPEN 提升来源），追加 Caser（CNN）、BERT4Rec（Transformer）。主 baseline：SASRec / BERT4Rec / GRU4Rec / Caser / DMPEN；表示对照：Gaussian embedding / MIND（multi-interest）。
- 实验分两层：① 同 encoder 内 vector vs state vs dynamic（最公平，主证据）；② 跨模型整体对比表（惯例）。
- 请 GPT 判断：baseline 集是否过多/过少？DMPEN 复现需注意什么（它是 RNN + density 特征，如何公平对齐超参）？

### Q3. Loss 是否要改？（我的倾向：必须改，做成可切换）
- 现状：BCEWithLogitsLoss + 裸 Tr（∈[0,1]），5 轮实验已证不匹配。
- 候选：① logit 变换 $\mathrm{logits}'=\log\frac{Tr}{1-Tr}$（保持 BCE，改动最小）；② fidelity loss（$-\log Tr$ / $-\log(1-Tr)$，重叠语义）；③ BPR（$-\log\sigma(Tr_{pos}-Tr_{neg})$，排序友好）；④ 温度缩放。
- 我的倾向：默认 ①logit+BCE，同时把 ②fidelity ③BPR 做成 `--loss` 消融，GPU 上一起验证。
- 请 GPT：从推荐领域惯例 + 量子重叠语义看，主损失选哪个最稳？是否需要每个都试？

### Q4. 主实验如何组织才公平有说服力？
- 我的计划：E001 SASRec 基线 → E002 Representation Ablation（vector vs state-r1/r4 vs Gaussian，同 encoder）→ E003 Evolution Ablation（static vs EMA(fixed α) vs dynamic(learnable α)）→ E004 interest-shift 模拟（核心机制证据）。
- 请 GPT：E002/E003 的设计是否足以回答 RQ1/RQ2？同参数量如何保证（vector d vs state d,r 的自由度对齐）？

### Q5. 评价指标
- DMPEN 用 AUC；我们用 NDCG@10/HR@10（next-item ranking）。跨论文对比是否需要补 AUC，或维持 ranking 惯例即可？

### Q6. adaptive α 是否必须？
- C2 里含"自适应 α"（$\alpha_t=\sigma(W[h_t;e_{i_t}]+b)$），对应不同用户兴趣稳定度。实现有梯度坍缩风险。
- 请 GPT：作为第一版，fixed/learnable α 是否够？adaptive α 是不是"加分项但非必须"？

### Q7. interest-shift 模拟是否作为核心实验？
- 构造"前 50 Action → 后 50 Romance"测适应步数，直接验证偏好漂移建模。
- 我的倾向：这是比 long-tail 更能证明"动态状态"优势的机制实验，建议作为主实验之一。

---

## 四、请你给 GPT 的问题（可直接发）
1. 我们的 Contribution C1/C2/C3 是否站得住？要不要调整成更锋利/加入"理论贡献"？
2. Baseline 集怎么定最公平有说服力（尤其 DMPEN 是 RNN、我们是 Transformer）？GRU4Rec 换 Caser 是否必要？
3. Loss 用哪种最稳（logit+BCE / fidelity / BPR / 温度）？要不要全做成消融？
4. 主实验 E001-E004 的设计能否支撑 RQ1/RQ2？
5. adaptive α 和 interest-shift 实验的必要性/优先级？
