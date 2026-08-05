# 实验计划（EXPERIMENT_PLAN）

> **用途（采纳 GPT 审稿建议 2026-08-02）**：所有实验矩阵集中管理，保证公平对比、证据充分。与 `PAPER_PROGRESS.md` §4（主线）和 `RESEARCH_LOG.md` §6（结果日志）联动。

---

## 1. 数据集（WWW 标准：不能只有 ml-1m）

| 数据集 | 规模 | 定位 | 验证点 |
|---|---|---|---|
| `ml-1m` | 999,611 交互 / 6,040 用户 / 3,416 物品 | 快速验证 | 主实验、消融 |
| `Beauty` | ~198k / ~22k / ~12k | **稀疏** | RQ1/RQ4：density 在稀疏/不确定性下优势 |
| `Steam` | 大 | **长尾/游戏** | RQ4：长尾 |
| `Yelp`（候选） | 大 | **兴趣变化** | RQ2：动态演化 |
- ⚠️ 数据文件不入库（gitignore），服务器/本地单独放置 `data/`。

## 2. Baseline（2026-08-03 二次定稿：四阶递进 + 主表 6 个）

**递进逻辑（核心叙事：V < VE < DF < DS < DDS）**
| 编号 | Baseline / 变体 | 是否 density | 是否 evolution | 回答 |
|---|---|---|---|---|
| V | Vector（SASRec） | ❌ | ❌ | 基准 |
| VE | **Vector Evolution**（$h_{t+1}=\alpha h_t+(1-\alpha)e_i$） | ❌ | ✅ | **换空间对照** |
| DF | Density Feature（模拟 DMPEN：$ee^\top$→flatten→SASRec） | ✅ feature | ❌ | density 表示是否有效 |
| DS | Static Density State（$\rho_T=\mathrm{Proj}(h_T)$） | ✅ state | ❌ | state 是否有效 |
| DDS | Dynamic Density State（凸组合演化） | ✅ state | ✅ | 动态演化是否有效 |

**主表 baseline（6 个）**
| baseline | 说明 | 状态 |
|---|---|---|
| GRU4Rec | RNN 基线（与 DMPEN 同族） | ⬜ 待接入 |
| SASRec | Transformer 点估计基线（= vector variant） | ✅ 已有 |
| BERT4Rec | 双向 Transformer 基线 | ⬜ 待接入 |
| **DMPEN** | density-as-feature + RNN（关键对照） | ⬜ E000 复现 |
| Ours-static | 密度状态、无演化 | ✅ 已有（state variant） |
| Ours-dynamic | 密度状态 + 合法性演化 | ✅ 已有（dynamic variant） |

> 暂缓：Caser / Gaussian embedding / MIND / ComiRec。

## 3. 主实验矩阵（与 01_paper_progress §4.1 对应；loss 主 = BPR）

| RQ | 实验 | 对比（同参数量） | 指标 |
|---|---|---|---|
| RQ1 | E001 | V vs VE vs DF vs DS | NDCG@10 / HR@10 |
| RQ2 | E002 | DS vs DDS；E003 α 扫描 0.1–0.9 + **ρ₀ 初始化消融（I/d / 首观测 / 可学习）** | NDCG@10 / HR@10 |
| RQ3 | E004 | remove PSD / remove trace / unconstrained matrix | NDCG@10 / HR@10 |
| RQ4 | E005 | entropy 分组（低/中/高）+ interest-shift（前 50 A → 后 50 B） | entropy / adaptation steps |

## 4. 分析实验（附）

| 分析 | 内容 | 指标 |
|---|---|---|
| A-1 维度效率 | vector $d=64$ vs state $d=32,r=4$ | NDCG/HR |
| A-2 序列长度 | history 5/10/20/50 | 增益曲线 |
| A-3 匹配消融 | dot vs trace（纯态下 $Tr=(u\cdot i)^2$） | NDCG/HR |
| A-4 长尾 | 按 popularity 分 head/mid/tail | 分组 NDCG/HR |
| A-5 多样性 | coverage / ILD | RQ4 补充 |
| A-6 损失消融 | BPR(主) vs BCE(logit) vs fidelity | NDCG/HR |

## 5. 铁律（违反则结论作废）
- 除被验证维度外**同超参**（lr/batch/maxlen/hidden/blocks/heads/dropout/epochs）；
- **loss 主 = BPR**（默认）；BCE(logit) 作消融（Tr/BCE 修正已于 2026-08-03 落地）；
- 结果一律回填 `docs/02_research_log.md` §6 + `docs/01_paper_progress.md` §4，并更新本表"状态"。

## 6. GPU 服务器运行
```bash
cd SASRec.pytorch-main
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu121  # 按 nvidia-smi 的 CUDA 版本
python run_experiments.py --dataset ml-1m --epochs 200 --device cuda --loss bpr --tag main
# 结果 → results/exp_main.csv → 回填 RESEARCH_LOG §6
```
