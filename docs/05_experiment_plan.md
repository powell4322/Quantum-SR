# 实验计划（EXPERIMENT_PLAN）

> **用途（2026-08-06 按 DeepSeek 最终路线重写）**：主线收敛为 **Dynamic Density State Transformer (DDST)**。
> 所有实验矩阵集中管理，保证公平对比、证据充分。与 `01_paper_progress.md` §4（主线）和 `02_research_log.md` §6（结果日志）联动。

---

## 0. 冻结路线（2026-08-06）

> 📌 **唯一 source of truth = `09_research_positioning_v2.md`（DDST）**。

**第一篇论文 = DDST**：vector hidden state → **density operator state**（不确定性/多兴趣表示）+ **legality-preserving dynamic evolution**（凸组合）+ **operator-level similarity**（Tr）。

- **不做**：QDM-Former / 量子纠缠 / CNOT / Angle Embedding / Measurement 作为主线（计算不可扩展、理论优势不可证、易被判 quantum overclaim）。
- 模型递进（**M0-M4 体系**）：

| 模型 | 表示 | 时间建模 | Matching | 作用 |
|---|---|---|---|---|
| **M0** SASRec | vector $h_t\in\mathbb{R}^d$ | Transformer | dot product | baseline |
| **M1** Density Feature (DF) | $\rho=ee^\top$（feature） | Transformer | vector projection | 验证二阶表示（DMPEN 式对照） |
| **M2** Density State (DS) | $\rho_t=\mathrm{Proj}(h_t)$ | 无显式演化 | $\mathrm{Tr}(\rho_u\rho_i)$ | 验证状态表示 |
| **M3** Dynamic Density State (DDS) | $\rho_t=\alpha\rho_{t-1}+(1-\alpha)\hat\rho_t$ | **EMA evolution（合法性保持）** | $\mathrm{Tr}(\rho_u\rho_i)$ | **核心模型** |
| **M4** Operator Attention Transformer (OAT) | density Q/K/V | Transformer attention | Tr(QK) | **第二阶段扩展（本期不做）** |

> 关键：**DS（static）与 DDS（dynamic）必须分离**，否则无法回答 "dynamic 是否有效"。
> 代码映射：`vector`(M0) / `density_feature`(M1) / `state`(M2=DS) / `dynamic`(M3=DDS)；`vector_evolve`(VE) 为"凸组合只是 EMA"的向量空间对照。

---

## 1. 数据集（本期定为三个：ml-1m / Beauty / Steam）

| 数据集 | 规模 | 定位 | 验证点 |
|---|---|---|---|
| `ml-1m` | 999,611 交互 / 6,040 用户 / 3,416 物品 | 快速验证 | 主实验、消融 |
| `Beauty` | ~198k / ~22k / ~12k | **稀疏** | RQ1/RQ4：density 在稀疏/不确定性下优势 |
| `Steam` | 大 | **长尾/游戏** | RQ4：长尾 |
- ⚠️ 数据文件不入库（gitignore），服务器/本地单独放置 `data/`。
- Yelp 等其他数据集：**本期暂缓**（先跑通上述三个再考虑）。

## 2. 模型范围（2026-08-06 定稿）

- **先用**：SASRec（= M0 vector variant）与我们自己的 M1-M3（DF/DS/DDS）→ 四阶递进 **V < DF < DS < DDS**（VE 作为 EMA 对照）。
- **暂缓**（本期不做）：GRU4Rec / BERT4Rec / DMPEN / Caser / MIND / ComiRec。DMPEN 仍保留为 Related Work 关键对照（见 `04_related_work.md`），跑通主线后再补复现。
- **核心判据**：DDS − DS 的增益 ≠ VE 的 EMA 增益（动态凸组合的合法性保持不可被普通向量 EMA 复现）。

## 3. 主实验矩阵（RQ1-RQ4；loss 主 = BPR；指标 R@10 / NDCG@10）

| RQ | 问题 | 实验 | 对比（同参数量） | 指标 |
|---|---|---|---|---|
| RQ1 | density representation 是否有效 | E001 | V vs DF vs DS | R@10 / NDCG@10 |
| RQ2 | dynamic evolution 是否有效 | E002 | DS vs DDS；E003 α 扫描 0.1–0.9 + **ρ₀ 初始化消融（I/d / 首观测 / 可学习）** | R@10 / NDCG@10 |
| RQ3 | operator similarity 是否有效 | E004 | matching：trace vs dot（`--matching`）；legality 消融（remove PSD / remove trace） | R@10 / NDCG@10 |
| RQ4 | uncertainty 是否合理 | E005 | entropy 分组（低/中/高）+ interest-shift（前 50 A → 后 50 B） | entropy / adaptation steps |

> 指标：**R@10（Recall@10）与 NDCG@10**。注意：本评估协议每用户 1 个 ground-truth + 100 负采样，**HR@10 ≡ Recall@10（数值等价）**，统一按 R@10 报告（代码已改为输出 Recall@10，见 `utils.py` 注释）。

## 4. 分析实验（附）

| 分析 | 内容 | 指标 |
|---|---|---|
| A-1 维度效率 | vector $d=64$ vs state $d=32,r=4$ | R@10 / NDCG@10 |
| A-2 序列长度 | history 5/10/20/50 | 增益曲线 |
| A-3 匹配消融 | dot vs trace（`--matching`；纯态下 $\mathrm{Tr}=(u\cdot i)^2$） | R@10 / NDCG@10 |
| A-4 长尾 | 按 popularity 分 head/mid/tail | 分组 R@10 / NDCG@10 |
| A-5 多样性 | coverage / ILD | RQ4 补充 |
| A-6 损失消融 | BPR(主) vs BCE(logit) vs fidelity | R@10 / NDCG@10 |
| A-7 ρ₀ 初始化 | Uniform I/d / First observation / Learnable（并入 E003） | R@10 / NDCG@10 |

> entropy 数值实现（RQ4）：`eigvals = clamp(torch.linalg.eigvalsh(rho), min=1e-8); H = -(eigvals*log(eigvals)).sum(-1)`（模型已内置 `SASRec.state_entropy`）。

## 5. 铁律（违反则结论作废）
- 除被验证维度外**同超参**（lr/batch/maxlen/hidden/blocks/heads/dropout/epochs）；
- **loss 主 = BPR**（默认）；BCE(logit) 作消融（Tr/BCE 修正已于 2026-08-03 落地）；
- **指标统一 R@10 / NDCG@10**；
- 结果一律回填 `docs/02_research_log.md` §6 + `docs/01_paper_progress.md` §4，并更新本表"状态"。

## 6. GPU 服务器运行（一键脚本见 `10_server_run.md`）
```bash
cd SASRec.pytorch-main
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu121  # 按 nvidia-smi 的 CUDA 版本
# 主实验（四阶递进 V / DF / DS / DDS；VE 作为 EMA 对照）
python run_experiments.py --dataset ml-1m --epochs 200 --device cuda --loss bpr --variants vector density_feature state dynamic vector_evolve --state_rank 1 --tag main
# 结果 → results/exp_main.csv → 回填 RESEARCH_LOG §6
```
