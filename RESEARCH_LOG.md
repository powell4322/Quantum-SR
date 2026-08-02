# 量子启发序列推荐 · 研究追踪日志（Research Log）

> 本文档用于**持续追踪研究方向**：记录哪些 idea 被验证/舍弃、当前理论状态、实验进度与结论。
> 每次有新想法、新实验结果或代码改动，请在本文件对应章节追加记录（保留历史，不要删除旧结论）。

---

## 1. 研究定位（一句话）

> 序列推荐把用户兴趣编码为**确定性向量** $h_t$；我们提出**动态量子概率状态**表示 $\rho_t$，随每次交互演化，并用"状态测量"完成 next-item 预测。

- 上游参照：WWW 2026 *Quantum-enhanced Representation Learning and Matching Learning for Recommendation*（针对 **static CF**）。
- 我们的差异化：把静态 $\rho_u$ 升级为随序列时间步演化的 $\rho_1,\dots,\rho_T$，直击序列推荐"兴趣漂移"本质。

---

## 2. 方向收敛记录（已做决策，勿回退）

| 方向 | 决策 | 原因 |
|---|---|---|
| Quantum Embedding 替换 SASRec embedding | ❌ 舍弃 | 只是 embedding 空间替换，是已有 quantum CF 的增量迁移，创新不足 |
| Quantum Attention（$|\langle Q|K\rangle|^2$） | ❌ 舍弃 | 本质是换 similarity kernel，易被当作 minor modification |
| Quantum Search / Annealing 优化 Top-K | ❌ 舍弃 | 序列推荐瓶颈在 representation，不在检索 |
| 真实量子电路 | ❌ 舍弃 | 比特数/噪声/模拟器限制，与推荐问题结合弱 |
| **动态状态表示 + 状态演化 + 状态匹配** | ✅ **保留（核心）** | 命中序列推荐"动态性/时序性"本质，与静态 quantum CF 区分明显 |

---

## 3. 核心假设与研究问题（RQ 状态追踪）

### 3.1 假设

- **H1**：确定性向量无法充分表达用户兴趣的不确定性/多中心性 → $h_t \rightarrow \rho_t$。
- **H2**：用户兴趣存在状态演化 → $\rho_t \rightarrow \rho_{t+1}$。
- **H3**：量子测量匹配 $score = Tr(\rho_t \rho_i)$ 更适合 next-item 预测。

### 3.2 研究问题与当前状态

| RQ | 问题 | 对应模块 | 验证方式 | 当前状态 | 结论 |
|---|---|---|---|---|---|
| RQ1 | 兴趣是否该建模为概率状态而非向量？ | State Projection | vector vs density | ⬜ 未开始 | — |
| RQ2 | 状态匹配是否优于 dot product？ | Prediction Layer | dot vs trace | ⬜ 未开始 | — |
| RQ3 | **动态演化是否优于静态状态？（核心创新）** | State Transition | static vs dynamic | ⬜ 未开始 | — |
| RQ4 | 量子表示是否降低 embedding 维度需求？ | Representation | 不同 latent/rank | ⬜ 未开始 | — |
| RQ5 | 是否提升 long-tail 物品推荐？ | Quality | 按 popularity 分组 | ⬜ 未开始 | — |
| RQ6 | 是否提升兴趣多样性？ | Diversity | coverage / ILD | ⬜ 未开始 | — |

> 更新方式：实验后把"当前状态"改为 ✅ 已验 / ❌ 不成立，并填"结论"一栏，附实验编号（见 §6）。

---

## 4. 理论笔记（持续更新）

### 4.1 密度矩阵合法性（必须始终满足）

$\rho$ 必须是 **半正定（PSD）+ trace=1**，否则不叫密度矩阵。工程实现上：

- **低秩构造（Cholesky-like）**：$\rho = \dfrac{LL^\top}{\mathrm{Tr}(LL^\top)}$，$L=\mathrm{Linear}(h)$ 为 $d\times r$ 低秩矩阵，天然 PSD。
  - $r=1$ → 纯态（rank-1，等价于一个方向向量，参数与向量同量级）；
  - $r>1$ → 混合态，表达多兴趣。
- **演化用凸组合**：$\rho_{t+1}=\alpha\rho_t+(1-\alpha)\rho_{i_t}$，凸组合保 PSD 且保 trace=1（量子混合态语义），**零额外参数**（$\alpha$ 可为固定标量或可学习标量）。

### 4.2 打分（测量）匹配

$$score(u,i)=\mathrm{Tr}(\rho_u\,\rho_i)$$

- 若 $\rho_u$ 与 $\rho_i$ 都是低秩 PSD，可用 $\mathrm{Tr}(AB)=\sum_{ij}A_{ij}B_{ij}$ 直接逐元素求和，成本 ~$O(d^2)$。
- 备选高效近似：item 状态用对角近似 $\rho_i\approx\mathrm{diag}(p_i)$，则 $\mathrm{Tr}(\rho_u\rho_i)=\sum_d \rho_{u,dd}\,p_{i,d}$，成本降为 $O(d)$。

### 4.3 待解决的理论问题（开放清单）

- [ ] $\alpha$ 固定 vs 可学习，哪种在长序列上更稳（可学习是否坍缩到 0/1）？
- [ ] 低秩 $r$ 与表达能力/参数量/过拟合的权衡（RQ4 依据）。
- [ ] 是否需要在打分前做"测量投影"（collapse 语义）以增强可解释性。
- [ ] "Quantum" 命名的合理性辩护：正文讲量子启发，标题是否保留 quantum 待定。

### 4.4 ⚠️ 重要发现：Tr 打分与 BCEWithLogitsLoss 不匹配（2026-08-02，需修正）

**现象**：5 轮快速对比中 `state/dynamic` 的 loss 明显高于 `vector`（~1.23 vs 0.91），指标低 ~20%。

**原因分析**：$\rho_u,\rho_i$ 均为密度矩阵时 $\mathrm{Tr}(\rho_u\rho_i)\in[0,1]$，而 `BCEWithLogitsLoss` 期望**无界 logits**。logits 被压在 $[0,1]$ 时：
- 正样本损失下界 $\approx -\log\,\mathrm{sigmoid}(1)\approx 0.313$，永远降不到 0；
- 负样本在 0 附近损失 $\approx 0.693$；
- 梯度区分度差 → 收敛慢、指标低。

**候选修正（待选定后改代码）**：
1. **logit 变换**：$\text{logits}' = \log\frac{Tr}{1-Tr}$（数值 clamp），保持 BCE 管线不变；
2. **温度缩放**：让 logits 分布跨越 0；
3. **fidelity loss**（量子 ML 惯例）：正样本 $-\log\mathrm{Tr}(\rho_u\rho_i)$，负样本 $-\log(1-\mathrm{Tr})$——语义最贴合 Born 规则。

**结论**：当前结果**不能**说明 idea 不成立；需修正打分/损失后再在 GPU 上验证。

---

## 5. 实验方案矩阵

### 5.1 方案（variant）定义（已写入代码）

| variant | 表示 | 演化 | 匹配 | 回答 RQ |
|---|---|---|---|---|
| `vector` | $h_t$ 向量 | 无 | dot product | baseline / RQ1 对照 |
| `state` | $\rho_t$ 密度矩阵 | 无（只用最后一步 $\rho_T$） | $\mathrm{Tr}$ | RQ1、RQ2 |
| `dynamic` | $\rho_t$ 密度矩阵 | 凸组合演化 $\rho_{t+1}=F(\rho_t,\rho_{i_t})$ | $\mathrm{Tr}$ | RQ3（核心） |

附加开关：`--state_rank`（低秩 $r$）、`--transition fixed|learnable`、`--transition_alpha`。

### 5.2 数据集

| 数据集 | 交互数 | 用户数 | 物品数 | 稀疏度 | 说明 |
|---|---|---|---|---|---|
| `ml-1m` | 999,611 | 6,040 | 3,416 | 高 | 首选验证集，训练/验证/测试按"最后 2 个交互"划分 |

### 5.3 评测指标（沿用仓库）

- 每个用户：真实目标 + 100 个随机负样本，模型排序，取目标物排名。
- **NDCG@10**、**HR@10**（取 10 000 用户抽样，超过 10 000 用户时）。
- 另可扩展 RQ5/6：按物品热度分组、coverage / ILD。

### 5.4 运行命令（ml-1m，示例）

```bash
# baseline
python main.py --dataset ml-1m --train_dir quant_vector --variant vector --num_epochs 200 --device cuda
# 状态表示
python main.py --dataset ml-1m --train_dir quant_state  --variant state  --state_rank 1 --num_epochs 200 --device cuda
# 动态演化
python main.py --dataset ml-1m --train_dir quant_dynamic --variant dynamic --state_rank 1 --num_epochs 200 --device cuda
```

批量对比请使用 `python run_experiments.py --dataset ml-1m --epochs 200 --device cuda`（会自动跑 vector/state/dynamic 并汇总结果）。

---

## 6. 实验结果日志（表格，实验后追加）

| 实验ID | 日期 | variant | state_rank | 其它配置 | NDCG@10 | HR@10 | 相对 baseline | 结论 / 下一步 |
|---|---|---|---|---|---|---|---|---|
| E000 | 2026-08-02 | vector | - | quick: maxlen=50,hidden=32,blocks=1,epochs=5,CPU | 0.3030 | 0.5394 | 1.00x | 仅 sanity check；loss=0.91 |
| E000 | 2026-08-02 | state | 1 | quick: 同上 | 0.2414 | 0.4528 | 0.80x | loss=1.24，疑似 Tr/BCE 不匹配（见 §4.4），待修正后重测 |
| E000 | 2026-08-02 | dynamic | 1 | quick: 同上 | 0.2460 | 0.4533 | 0.81x | 同上 |

> 追加规则：每完成一个实验，新增一行；"结论"栏写清楚该实验支撑/否定了哪个 RQ；同一配置复跑则覆盖行并备注。

---

## 7. 代码修改记录

| 日期 | 文件 | 改动 | 备注 |
|---|---|---|---|
| 2026-08-02 | model.py | 新增 `StateProjection`（低秩 Cholesky-like 密度矩阵）、`StateTransition`（凸组合演化）；`SASRec` 支持 `variant` | `vector` 分支与原实现完全一致，保证基线可复现 |
| 2026-08-02 | main.py | 新增 `--variant/--state_rank/--transition/--transition_alpha` 参数 | 向后兼容（默认 vector） |
| 2026-08-02 | run_experiments.py | 批量跑多方案并汇总指标 | 见 §5.4 |
| 2026-08-02 | test_smoke.py | 冒烟测试：验证三种 variant 的 forward/predict/backward 与密度矩阵合法性 | ✅ 已通过（PSD + trace=1 校验） |

### 运行环境

- 项目 `.venv`（Python 3.12）依赖：`torch 2.13.0+cpu`、`numpy`（用 uv 安装）。
- 终端运行前缀：`.venv\Scripts\python.exe`。
- 若需 GPU：`uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu124`（按本机 CUDA 版本调整）。
- 代码改动后回归：`.venv\Scripts\python.exe test_smoke.py`。

---

## 8. 下一步（Next Steps）

1. [ ] 复现 `vector` 基线（ml-1m），记录 E001。
2. [ ] 跑 `state`（RQ1：vector vs state，同参数对比）。
3. [ ] 跑 `dynamic`（RQ3：static vs dynamic，核心贡献点）。
4. [ ] 若 RQ1/RQ3 正向：补 RQ2（matching 消融）、RQ4（rank 扫描）、RQ5/RQ6（附表）。
5. [ ] 更新 §4 理论清单，把"alpha 可学习是否坍缩"等疑问转成实验。

---

> ⚠️ 重要提醒：每次实验前确认所有对比方案使用**相同超参**（除被验证维度外），并在 §6 记录完整配置，否则结论不可比。
