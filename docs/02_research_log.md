# 量子启发序列推荐 · 研究追踪日志（Research Log）

> 本文档用于**持续追踪研究方向**：记录哪些 idea 被验证/舍弃、当前理论状态、实验进度与结论。
> 每次有新想法、新实验结果或代码改动，请在本文件对应章节追加记录（保留历史，不要删除旧结论）。

---

## 1. 研究定位（一句话）

> 序列推荐通常把用户兴趣编码为**点估计向量** $h_t$；我们提出**不确定性感知的动态偏好状态**（density state）$\rho_t$（半正定、trace=1 的密度算子），随每次交互按"偏好惯性"演化，并用 Hilbert–Schmidt 相似度完成 next-item 预测。

**两个核心 Gap（论文动机）**

1. **Sequential Gap（我们要填补的核心空白）**：现有 quantum-inspired 推荐（如 WWW 2026 *Quantum-enhanced Representation Learning and Matching Learning for Recommendation*）只处理**静态协同过滤**（静态 $\rho_u$），**没有时序/序列版本**——密度算子从未被用于建模序列兴趣的演化与漂移。我们把静态 $\rho_u$ 升级为随序列时间步演化的 $\rho_1,\dots,\rho_T$，直击序列推荐"兴趣漂移"本质。
2. **Representation Gap**：主流序列推荐（SASRec/BERT4Rec）用点估计表示兴趣，会**丢失不确定性**与**多模态偏好结构**。密度算子提供统一的 second-order 表示，能显式表达"不确定/多中心"的偏好。

> 定位提醒（审稿风险规避）：**量子只是数学工具，不是贡献主体**。贡献主体是 **dynamic uncertainty-aware preference state modeling**。量子（density operator / convex mixing / HS kernel）仅用于提供数学上合法（PSD+trace 保持）且可解释的表示与演化框架。

---

## 2. 方向收敛记录（已做决策，勿回退）

| 方向 | 决策 | 原因 |
|---|---|---|
| Quantum Embedding 替换 SASRec embedding | ❌ 舍弃 | 只是 embedding 空间替换，是已有 quantum CF 的增量迁移，创新不足 |
| Quantum Attention（$|\langle Q|K\rangle|^2$） | ❌ 舍弃 | 本质是换 similarity kernel，易被当作 minor modification |
| Quantum Search / Annealing 优化 Top-K | ❌ 舍弃 | 序列推荐瓶颈在 representation，不在检索 |
| 真实量子电路 | ❌ 舍弃 | 比特数/噪声/模拟器限制，与推荐问题结合弱 |
| **动态状态表示 + 状态演化 + 状态匹配** | ✅ **保留（核心）** | 命中序列推荐"动态性/时序性"本质，与静态 quantum CF 区分明显 |
| **自适应偏好惯性 $\alpha_t$** | ✅ **保留（贡献2）** | 区分不同用户的兴趣稳定度，把"动态"升级为用户级自适应演化，直接回应 "Why not GRU?" |
| **命名策略** | ✅ 采用 "Quantum-**Inspired**" | 正文强调 density-state / uncertainty，量子测量语义放 discussion，避免过度量子 claim |

---

## 3. 核心假设与研究问题（RQ 状态追踪）

### 3.1 假设（措辞已按审稿风险修正）

- **H1**：主流序列推荐的点估计表示**不能显式表达**用户兴趣的不确定性/多中心结构；density-state 能提供这种 second-order 结构 → $h_t \rightarrow \rho_t$。
  > ⚠️ 不写"向量无法表达不确定性"（distributional / Bayesian / multi-interest 已有），而是写：**点估计丢失不确定性/多模态结构，且 quantum-inspired 表示从未进入序列场景（Sequential Gap）**。
- **H2**：用户兴趣存在状态演化（interest drift）；密度状态的凸组合演化能**合法（PSD+trace 保持）且有效**地建模这种演化 → $\rho_t \rightarrow \rho_{t+1}$。
- **H3**：密度算子诱导的 Hilbert–Schmidt 匹配 $score=\mathrm{Tr}(\rho_t\rho_i)$ 能作为 next-item 打分的有效核（作为 similarity kernel，而非强调"测量坍缩"）。

### 3.2 研究问题与当前状态（收敛为 3 个核心 RQ + 次要分析）

| RQ | 问题（英文写法，供论文用） | 对应模块 | 验证方式 | 当前状态 | 结论 |
|---|---|---|---|---|---|
| **RQ1** | Can density operators better represent uncertain user preference than point embeddings? | State Projection | vector vs state（**同维度、同参数量**） | ⬜ 待验证 | — |
| **RQ2（核心）** | Does temporal state evolution improve sequential recommendation? | State Transition | static state vs dynamic state | ⬜ 待验证 | — |
| **RQ3** | Does uncertainty-aware state modeling benefit sparse & long-tail scenarios? | 全链路 | 稀疏/长尾分组 + diversity | ⬜ 待验证 | — |

**次要分析（不作为独立 RQ，作为附实验）**：

| 分析 | 内容 | 理由 |
|---|---|---|
| A-1 维度效率 | 固定参数量下比较 vector vs state（如 vector $d=64$ vs state $d=32,r=4$） | 验证 second-order 表示的参数效率（§4.1 自由度分析） |
| A-2 序列长度敏感性 | history length 5/10/20/50 下的增益曲线 | 直接支持"动态"优势随上下文增长而放大 |
| A-3 兴趣漂移模拟 | 人工构造兴趣迁移（如 100 个 A 类 → 100 个 B 类）观察状态适应速度 | 直接检验 H2 / interest drift |
| A-4 多样性 | coverage / ILD | RQ3 的补充证据 |
| A-5 匹配消融 | dot vs trace（纯态近似下 $Tr=\langle u,i\rangle^2$） | 拆解"表示"与"匹配"各自的贡献 |

> 更新方式：实验后把"当前状态"改为 ✅ 已验 / ❌ 不成立，并填"结论"一栏，附实验编号（见 §6）。

---

## 4. 理论笔记（持续更新，数学精确化）

> 本节目标：把"量子启发"包装成**可辩护的数学框架**，规避审稿人对"reparameterization / 过度量子化 / 自由度错误"的质疑。

### 4.1 密度状态构造（Density-state projection）

**定义**：$h\in\mathbb{R}^d$ 经低秩 Cholesky-like 构造投影为密度状态：

$$\rho(h)=\frac{LL^\top}{\mathrm{Tr}(LL^\top)},\qquad L=\mathrm{Linear}(h)\in\mathbb{R}^{d\times r}$$

- **合法性**：$LL^\top\succeq 0$（半正定），除以 trace 后 $\mathrm{Tr}(\rho)=1$。工程上对 trace 加 `clamp_min(1e-8)` 防除零。
- **谱的含义（措辞修正）**：~~"rank>1 ⇒ 多兴趣"~~ 改为：**谱提供用户偏好不确定性的潜在分解（latent decomposition of preference uncertainty）**。rank>1 只表示矩阵秩更高，**不自动对应"多个语义兴趣"**；我们不做"每个特征向量=一个兴趣"的强 claim。
- **参数量与自由度（精确分析，勿再写"更多自由度"）**：

  - 参数量：$L$ 有 $dr$ 个参数。
  - 有效自由度：$\rho=LL^\top$ 对 $L$ 有**正交旋转歧义**（$L\mapsto LQ,\ Q\in O(r)$ 不改变 $\rho$），且受 trace=1 约束，故

  $$\mathrm{dof}(\rho)=dr-\frac{r(r-1)}{2}-1$$

  - $r=1$：$\mathrm{dof}=d-1$，恰好是单位向量所在球面 $S^{d-1}$ 的自由度 → **纯态与向量同量级**（"state 是 vector 的自然推广"的关键证据）。
  - $r=d$：$\mathrm{dof}=\frac{d(d+1)}{2}-1$，即 trace 固定的对称矩阵。

  > 结论表述（论文用）：*Low-rank density operators provide a structured second-order representation with controllable rank complexity.* —— 卖点是 **second-order 结构**与 **rank 可控性**，不是"更多自由度"。

### 4.2 打分（Hilbert–Schmidt similarity）

$$score(u,i)=\mathrm{Tr}(\rho_u\,\rho_i)=\sum_{ij}\rho_{u,ij}\,\rho_{i,ij}$$

- **定位（措辞修正）**：不强调"Born rule / 测量坍缩"。$\mathrm{Tr}(\rho_u\rho_i)$ 本质是 $\mathbb{R}^{d\times d}$ 上的 **Hilbert–Schmidt 内积**（$\langle A,B\rangle_{HS}=\mathrm{Tr}(A^\top B)$）。论文表述：*We adopt Hilbert–Schmidt similarity induced by density operators.*
- **与 dot product 的精确关系（重要桥梁）**：当 $\rho_u,\rho_i$ 均为纯态（rank-1）$\rho_u=uu^\top,\ \rho_i=ii^\top$ 时：

  $$\mathrm{Tr}(\rho_u\rho_i)=\mathrm{Tr}(uu^\top ii^\top)=(u\cdot i)^2=\cos^2\theta$$

  即 trace 打分退化为**平方余弦**（有界 $[0,1]$），而 vector dot 无界。这一性质：
  1. 为"state ⊇ vector"提供严格桥梁（$r=1$ 时 state 恰是平方点积核）；
  2. 解释了 §4.4 的 BCE/Tr 不匹配现象（logits 被压到 $[0,1]$）。
- **计算成本**：$O(d^2)$；低秩分解下可用 $\mathrm{Tr}(AB)=\sum_{ij}A_{ij}B_{ij}$ 逐元素求和在 GPU 上高效实现。

### 4.3 动态演化：偏好惯性模型（Preference Inertia Model）

**更新规则**：

$$\rho_{t+1}=\alpha_t\,\rho_t+(1-\alpha_t)\,\rho_{i_t},\qquad 0\le\alpha_t\le 1$$

其中 $\rho_{i_t}$ 是当前交互物品诱导的密度状态。

- **为什么是凸组合（合法性 + 语义）**：
  - PSD 保持：密度状态集合是**凸锥**，凸组合仍 PSD → **合法性保持（legality-preserving）演化**；
  - trace 保持：$\mathrm{Tr}(\alpha\rho_t+(1-\alpha)\rho_{i_t})=\alpha+(1-\alpha)=1$；
  - 对应量子混合态语义，但**推荐语义**上可解释为**偏好惯性（preference inertia）**：$\alpha_t$ 越大越保留旧兴趣，越小越快速吸收新兴趣。
- **与经典方法的联系（回应 "Why not GRU?"）**：
  - 固定 $\alpha$：展开得 $\rho_T=\alpha^{T-1}\rho_1+(1-\alpha)\sum_{k=1}^{T-1}\alpha^{T-1-k}\rho_{i_k}$，即**指数滑动平均（EMA）/ 几何加权**——旧兴趣按 $\alpha$ 指数衰减，天然具备遗忘机制；
  - 语义上等价于**贝叶斯滤波**的信息合并（后验混合），而凸组合结构**保证中间状态永远是合法密度算子**——这是 GRU/门控在表示层面不具备的硬约束。
- **自适应 $\alpha_t$（关键增强，贡献2）**：

  $$\alpha_t=\sigma\big(W\,[h_t;\,e_{i_t}]+b\big)\quad\text{或}\quad \alpha_t=\sigma\big(w^\top h_t+b\big)$$

  - 含义：**不同用户（甚至不同时刻）的兴趣稳定度不同**——长期稳定用户 $\alpha_t\uparrow$，探索型用户 $\alpha_t\downarrow$；
  - 对比表：`fixed alpha`（固定标量，baseline）→ `learnable scalar alpha`（全局可学习标量，**当前代码已支持**）→ `adaptive alpha_t`（逐用户/逐步门控，**待实现**）；
  - 风险：可学习 $\alpha$ 是否坍缩到 0/1、梯度是否稳定（见 §4.5 开放清单）。

### 4.4 ⚠️ Tr 打分与 BCE 不匹配 —— 已修正（2026-08-03 实现）

**问题**：5 轮快速对比中 `state/dynamic` 的 loss 明显高于 `vector`（~1.23 vs 0.91）。原因：$\mathrm{Tr}(\rho_u\rho_i)\in[0,1]$，而 `BCEWithLogitsLoss` 期望无界 logits（正样本损失下界 ≈0.313、负样本 ≈0.693、梯度区分度差）。

**修正（2026-08-03 已实现）**：`model.py` 对 state/dynamic 的 HS 相似度做 **logit 变换**（`_logit_score`，$z=\log\frac{s+\epsilon}{1-s+\epsilon}$，$\epsilon=10^{-7}$）；`main.py` 支持 `--loss bce|bpr`（主=**bce**，消融=**bpr**：$-\log\sigma(s^+-s^-)$）。

**验证（ml-1m，1-epoch quick，CPU）**：冒烟测试通过；dynamic+bce 1-epoch loss **1.24 → 1.08**（改善）；dynamic+bpr loss ~0.43（正常收敛）。

**结论**：打分/损失已修正，可进入正式实验（GPU）。

### 4.5 理论清单（开放问题）

- [ ] 可学习/自适应 $\alpha_t$ 是否坍缩到 0/1？梯度稳定性如何？
- [ ] 低秩 $r$ 与表达能力/参数量/过拟合的权衡（对应 A-1 维度效率实验）。
- [ ] trace 打分是否需要 logit 变换才能与 BCE 兼容（§4.4 决策）。
- [ ] "Quantum" 命名合理性辩护：正文讲 density-state / quantum-inspired，标题是否保留 quantum 待定。
- [ ] 纯态近似下 $Tr=(u\cdot i)^2$ 与平方余弦的数值等价性验证（作为"state ⊇ vector"证据）。

---

## 5. 实验方案矩阵

### 5.1 方案（variant）定义（已写入代码）

| variant | 表示 | 演化 | 匹配 | 回答 RQ |
|---|---|---|---|---|
| `vector` | $h_t$ 向量 | 无 | dot product | baseline / RQ1 对照 |
| `state` | $\rho_t$ 密度矩阵 | 无（只用最后一步 $\rho_T$） | $\mathrm{Tr}$ | RQ1 |
| `dynamic` | $\rho_t$ 密度矩阵 | 凸组合演化 $\rho_{t+1}=F(\rho_t,\rho_{i_t})$（fixed / learnable 标量 $\alpha$） | $\mathrm{Tr}$ | **RQ2（核心）** |
| `dynamic-adaptive` | $\rho_t$ 密度矩阵 | 自适应 $\alpha_t=\sigma(W[h_t;e_{i_t}]+b)$（**待实现**） | $\mathrm{Tr}$ | RQ2 增强（贡献2） |

附加开关：`--state_rank`（低秩 $r$）、`--transition fixed|learnable`、`--transition_alpha`。

### 5.2 数据集（扩展：覆盖稀疏 / 长尾 / 兴趣变化）

| 数据集 | 交互数 | 用户数 | 物品数 | 定位 | 说明 |
|---|---|---|---|---|---|
| `ml-1m` | 999,611 | 6,040 | 3,416 | 快速验证 | 首选验证集，训练/验证/测试按"最后 2 个交互"划分 |
| `Beauty` | ~198k | ~22k | ~12k | **稀疏** | 交互稀疏，检验 RQ3 |
| `Amazon Sports` | ~296k | ~35k | ~18k | **长尾** | 热度分布极不均匀，检验 RQ3 长尾 |
| `Yelp`（候选） | 大 | 大 | 大 | **兴趣变化** | 交互密集、时序跨度长，利于兴趣漂移验证 |

### 5.3 评测指标（沿用仓库）

- 每个用户：真实目标 + 100 个随机负样本，模型排序，取目标物排名。
- **NDCG@10**、**HR@10**（取 10 000 用户抽样，超过 10 000 用户时）。
- RQ3 扩展：按物品热度分组（head / mid / **tail**）的 NDCG/HR、coverage / ILD。

### 5.4 Baselines（补 uncertainty 对照，否则 RQ1/H1 不成立）

| baseline | 说明 | 回答 |
|---|---|---|
| `SASRec (vector)` | 原始点估计基线（同参数量） | RQ1 对照 |
| `state`（我们） | 静态密度状态 | RQ1 |
| `dynamic`（我们） | 动态密度状态 | RQ2 |
| Gaussian embedding | $z\sim\mathcal N(\mu,\Sigma)$，用 KL/内积打分 | RQ1"不确定性"对照 |
| MIND / ComiRec | multi-interest 表示（胶囊路由 / 可组合多兴趣） | RQ1"多中心"对照 |

> 若不加入 Gaussian / multi-interest，审稿人会说"只对比了确定性向量，不公平"。

### 5.5 关键实验（支撑"dynamic / interest drift"的核心证据）

**Experiment A — Dimension efficiency（固定参数量）**

- Vector：$d=64$；State：$d=32,\,r=4$（参数量接近）。验证 state 是否参数高效（呼应 §4.1 自由度）。

**Experiment B — Sequence length sensitivity**

- history length 5 / 10 / 20 / 50 下的增益曲线；若越长 dynamic 优势越明显，强力支持"动态"贡献。

**Experiment C — Interest shift simulation（人工构造兴趣漂移）**

- 构造用户：100 个 A 类交互后突然转入 100 个 B 类，观察 state 的适应速度（对比 vector / static / dynamic）。
- 直接检验 H2（interest drift），是本方向最有力的机制实验。

### 5.6 运行命令（ml-1m，本机 CPU 用 uv 运行）

```bash
# baseline
uv run main.py --dataset ml-1m --train_dir quant_vector --variant vector --num_epochs 200 --device cpu
# 状态表示（同参数量对照：d 减半 + rank 补偿）
uv run main.py --dataset ml-1m --train_dir quant_state  --variant state  --state_rank 1 --num_epochs 200 --device cpu
# 动态演化（核心）
uv run main.py --dataset ml-1m --train_dir quant_dynamic --variant dynamic --state_rank 1 --transition learnable --num_epochs 200 --device cpu
```

批量对比请使用 `uv run run_experiments.py --dataset ml-1m --epochs 200 --device cpu`（自动跑 vector/state/dynamic 并汇总）。

> ⚠️ 本机无 NVIDIA GPU，`--device` 一律用 `cpu`；若后续有 GPU 机器，改为 `cuda` 并安装对应 CUDA 版 torch。

---

## 6. 实验结果日志（表格，实验后追加）

> ⚠️ **可读性提示**：E000 三行为 quick sanity check（5 轮、CPU、小超参），受 §4.4 的 Tr/BCE 不匹配影响，**不代表** idea 无效；其原始输出文件（`ml-1m_quant_*_quick/`、`results/exp_quick.csv`）已清理（2026-08-03）。正式结论以修正打分/损失后的 E001+ 为准。

| 实验ID | 日期 | variant | state_rank | 其它配置 | NDCG@10 | HR@10 | 相对 baseline | 结论 / 下一步 |
|---|---|---|---|---|---|---|---|---|
| E000 | 2026-08-02 | vector | - | quick: maxlen=50,hidden=32,blocks=1,epochs=5,CPU | 0.3030 | 0.5394 | 1.00x | 仅 sanity check；loss=0.91 |
| E000 | 2026-08-02 | state | 1 | quick: 同上 | 0.2414 | 0.4528 | 0.80x | loss=1.24，疑似 Tr/BCE 不匹配（见 §4.4），待修正后重测 |
| E000 | 2026-08-02 | dynamic | 1 | quick: 同上 | 0.2460 | 0.4533 | 0.81x | 同上 |
| E001 | 待补 | 复现 vector 基线（ml-1m，正式超参） | - | - | - | - | - | RQ1 对照基线 |
| E002 | 待补 | state vs vector（同参数） | 见 A-1 | - | - | - | - | RQ1 |
| E003 | 待补 | static vs dynamic | 见 A-1 | - | - | - | - | **RQ2（核心）** |

> 追加规则：每完成一个实验，新增一行；"结论"栏写清楚该实验支撑/否定了哪个 RQ；同一配置复跑则覆盖行并备注。

---

## 7. 代码修改记录

| 日期 | 文件 | 改动 | 备注 |
|---|---|---|---|
| 2026-08-02 | model.py | 新增 `StateProjection`（低秩 Cholesky-like 密度矩阵）、`StateTransition`（凸组合演化，fixed/learnable）；`SASRec` 支持 `variant` | `vector` 分支与原实现完全一致，保证基线可复现 |
| 2026-08-02 | main.py | 新增 `--variant/--state_rank/--transition/--transition_alpha` 参数 | 向后兼容（默认 vector） |
| 2026-08-02 | run_experiments.py | 批量跑多方案并汇总指标 | 见 §5.6 |
| 2026-08-02 | test_smoke.py | 冒烟测试：验证三种 variant 的 forward/predict/backward 与密度矩阵合法性 | ✅ 已通过（PSD + trace=1 校验） |
| 🔜 待做 | model.py | 实现 `dynamic-adaptive`：$\alpha_t=\sigma(W[h_t;e_{i_t}]+b)$ | 贡献2，见 §4.3 |
| ✅ 2026-08-03 | model.py / main.py / run_experiments.py | Tr 打分修正：state/dynamic 打分做 logit 变换；`--loss bce|bpr` 可切换 | 冒烟通过；1-epoch loss 1.24→1.08 |

### 运行环境

- 项目 `.venv`（Python 3.12）依赖：`torch 2.13.0+cpu`、`numpy`（用 uv 管理）。
- 统一运行前缀：`uv run main.py ...`（自动使用 `.venv`）。
- 若需 GPU：`uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu124`（按本机 CUDA 版本调整）。
- 代码改动后回归：`uv run test_smoke.py`。

---

## 8. 下一步（Next Steps，按优先级）

### Priority 1（必须先做，否则投稿无望）
1. [x] 修正 §4.4：logit 变换 + `--loss bce|bpr`（2026-08-03 已实现）
2. [ ] 复现 `vector` 基线（ml-1m，正式超参），记录 E001。
3. [ ] 跑 `state`（RQ1：vector vs state，**同参数量**对照）。
4. [ ] 跑 `dynamic`（RQ2：static vs dynamic，核心贡献点，用 learnable $\alpha$）。

### Priority 2（增强"动态"贡献）
5. [ ] 实现 `dynamic-adaptive`（$\alpha_t$ 门控），对比 fixed / learnable / adaptive。
6. [ ] Experiment B：序列长度敏感性（5/10/20/50）。
7. [ ] Experiment C：兴趣漂移模拟（A 类→B 类适应速度）。

### Priority 3（补足审稿公平性）
8. [ ] 加入 uncertainty baselines：Gaussian embedding、MIND/ComiRec（见 §5.4）。
9. [ ] 数据集扩展：Beauty（稀疏）、Amazon Sports（长尾）、Yelp（兴趣变化）。
10. [ ] Experiment A：固定参数量维度效率（vector $d=64$ vs state $d=32,r=4$）。
11. [ ] RQ3 长尾/多样性附实验（热度分组、coverage / ILD）。

### Priority 4（模型泛化：拓展到其他序列模型）
> 目标：证明 quantum 多方案框架**不限于 SASRec**——对论文的 generalization / 方法普适性是加分项。
> **当前决策（2026-08-02）**：先专注 SASRec 完成全部核心实验（RQ1/RQ2/RQ3 + Tr 修正）；拓展目标暂定为 **GRU4Rec + BERT4Rec**（与 SASRec 共 3 个核心模型）；Caser / STAMP 等视实验结论与论文相关性再定，暂不承诺。
> 当前架构已解耦：`StateProjection`（任意 $(...,C)\to(...,C,C)$）、`StateTransition`（凸组合）都是独立模块，任何输出 $(U,T,C)$ 的编码器都能接入。

12. [ ] 先完成 SASRec 的 RQ1/RQ2/RQ3 与 Tr 打分修正（主线）。
13. [ ] 拓展到 GRU4Rec（RNN，每步 $h_t$ 天然适配）、BERT4Rec（双向 Transformer）。
14. [ ] 架构：把 `SASRec._to_state_sequence` + 打分逻辑抽取为可复用 `QuantumStateHead`（基类接口：编码器只管出 $h_t$ 序列，head 负责 $\rho_t$ / 演化 / 匹配）。
15. [ ] ⚠️ 注意点：BERT4Rec 双向注意力 vs SASRec 单向因果，**演化语义要重定义**（用 [MASK] 位打分还是每步演化？）；GRU4Rec 无 position 语义，逐时间步演化最自然。
16. [ ] 结论可写成论文 "generalization across sequence encoders" 小节。

### 写作准备
12. [ ] 基于 §9 文献探讨收敛 related work 叙事：强调 Sequential Gap，量子仅作工具。
13. [ ] 更新 §4 理论清单，把"alpha 是否坍缩"等疑问转成实验。

---

> ⚠️ 重要提醒：每次实验前确认所有对比方案使用**相同超参**（除被验证维度外），并在 §6 记录完整配置，否则结论不可比。

---

## 9. 文献探讨清单（待用户提供资料，逐篇深入）

> 目的：为 motivation / 数学 / baseline 提供可引用锚点，并把我们的创新点与已有工作严格划清界限。**请用户按需提供这些文献（PDF / 链接 / 笔记），我逐篇做深度对比分析。**

### 类别 A：Quantum-inspired 表示与推荐（划清界限 + Sequential Gap 依据）
- [ ] **WWW 2026 *Quantum-enhanced Representation Learning and Matching Learning for Recommendation***（上游参照，静态 CF）——我们要探讨：其密度矩阵构造、打分、损失函数细节，用于论证"无序列版"的空白。
- [ ] Quantum Recommendation Systems（Kerenidis & Prakash, ITCS 2017）——量子采样加速推荐，与表示无关；用于在 related work 里说明"量子推荐≠我们"。
- [ ] 其他 density-matrix / quantum-inspired CF 论文——搜集以证明"现有 quantum 推荐均为静态、无时序演化"。

### 类别 B：Density matrix 在表示学习中的应用（证明 density-state 可训练且有效）
- [ ] **QSAN: A Quantum-probability based Sentiment Analysis Model（EMNLP 2020）**——低秩密度矩阵用于文本表示；探讨其构造与训练稳定性，作为"density-state 表示可行"的直接证据。
- [ ] Density Matrices with Metric Learning for Text Quantum Similarity——density + 度量学习 + HS 相似度，与我们的打分/匹配最接近，探讨其 loss 设计。

### 类别 C：Gaussian / Distributional 表示（RQ1 uncertainty baseline）
- [ ] **Word Representations via Gaussian Embedding（Vilnis & McCallum, ICLR 2015）**——高斯表示经典；探讨其打分（KL / 期望内积）与我们 HS 打分的异同。
- [ ] 分布/变分推荐相关（Bayesian CF、Variational Recommendation）——用于写"点估计→分布化"的既有工作脉络。

### 类别 D：Multi-interest 表示（RQ1 multi-center baseline）
- [ ] **MIND: Multi-Interest Network with Dynamic Routing（CIKM 2019）**——胶囊路由多兴趣；作为"多中心"对照。
- [ ] **ComiRec: Controllable Multi-Interest Framework（KDD 2020）**——可组合多兴趣；同上。

### 类别 E：序列推荐与兴趣演化（dynamic 动机与 "Why not GRU?" 回应）
- [ ] **SASRec（ICDM 2018）** / **BERT4Rec（CIKM 2019）**——基座模型；用于"点估计 baseline"定位。
- [ ] **DIEN: Deep Interest Evolution Network（AAAI 2019）**——GRU 式兴趣演化；我们需要正面回应"为何不用 GRU 而用凸组合"（关键：合法性保持 + EMA/贝叶斯滤波语义，见 §4.3）。
- [ ] 兴趣漂移 / 时序兴趣综述——为 interest drift 实验（A-3）提供背景。

### 类别 F：量子信息数学基础（保证理论表述严格）
- [ ] Nielsen & Chuang（density operator、convex combination、quantum channel 章节）——用于严谨表述"凸组合=经典混合"、"合法性保持演化"。

> 探讨产出：每篇文献确定 (1) 我们引用它支撑哪个 claim；(2) 与我们的差异点；(3) 是否引入新 baseline 或新实验。
