# 相关工作与防撞对照（RELATED_WORK）

> **用途（采纳 GPT 审稿建议 2026-08-02）**：防止撞 idea——重点确认 **dynamic density state 是否已经有人做过**。这是决定论文能否投的关键核查。
> 方法：按三个方向逐条调研，填"代表工作 / 核心做法 / 是否做了 dynamic density state / 与我们差异"。**未核实的工作不要标记为"已查"。**

---

## 0. 结论占位（调研后填写）
- [ ] 是否已有工作做 **dynamic density-state evolution for sequential recommendation**？→ 结论：______
- [ ] 是否已有工作做 **legality-preserving preference transition**？→ 结论：______
- 若已有：我们的差异点 / 可引用的"我们与之不同"的表述 → ______

---

## 1. 方向 A：Quantum / Density-operator Recommendation

| 代表工作 | 核心做法 | 静态 or 动态状态 | 是否覆盖"序列演化" | 与我们差异 |
|---|---|---|---|---|
| WWW 2026 *Quantum-enhanced Representation Learning and Matching Learning for Recommendation* | 用户表示为一阶密度算子，量子表示 + 量子匹配（static CF） | **静态** $\rho_u$ | ❌ 无序列/时间维 | 我们引入 $\rho_1\to\rho_T$ 动态演化 + 保合法性转移 |
| （待调研）density-matrix 用于推荐的其他工作 | ______ | ______ | ______ | ______ |

---

## 2. 方向 B：Uncertainty-aware / Distributional Sequential Recommendation

| 代表工作 | 核心做法 | 是否动态 | 状态约束（PSD/trace） | 与我们差异 |
|---|---|---|---|---|
| Gaussian embedding 类 | $z\sim\mathcal N(\mu,\Sigma)$，KL / 内积打分 | 多为静态 | ❌ 无硬约束 | 我们用二阶算子状态 + 恒保合法演化 |
| Bayesian / distributional rec | $p(z)$ 分布表示 | 部分 | ❌ | 演化无结构化约束 |
| uncertainty-aware sequential rec | 不确定性感知的序列模型 | 有 | ❌ | 无 density 状态 / 保合法演化 |
| （待调研）______ | ______ | ______ | ______ | ______ |

> ⚠️ 该方向是**最大风险**：必须回答"我们与 Gaussian/distributional 的本质区别"——**受约束（PSD+trace）的状态 + 合法演化**，不是"把向量换成分布"。

---

## 3. 方向 C：状态演化 / State-Space / 偏好漂移建模

| 代表工作 | 核心做法 | 与我们差异 |
|---|---|---|
| 偏好漂移 / interest drift 建模 | 动态偏好建模（非量子） | 我们用 density 状态 + 保合法凸组合 |
| state-space / Bayesian filtering（Kalman 等） | 状态估计演化 | 状态是密度算子，演化是凸组合（EMA/贝叶斯合并语义） |
| 记忆 / 遗忘机制模型（EMA、门控） | 指数衰减旧兴趣 | 我们在**密度状态空间**上做 EMA，且状态恒合法 |
| （待调研）______ | ______ | ______ |

---

## 4. 调研来源建议（写作时可引）
- WWW / KDD / SIGIR / CIKM 近 3 年：quantum recommendation、uncertainty-aware recommendation、distributional embedding、interest drift。
- arXiv：quantum machine learning + recommendation；density matrix embedding。

---

## 5. 引用维护（发现新文献即追加）
```markdown
[N] {作者}. "{标题}." *{会议}*, {年份}.
> 核心工作：______
> 引用位置/原因：______
> 与我们差异：______
```
