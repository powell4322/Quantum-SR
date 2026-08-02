# 实验计划（EXPERIMENT_PLAN）

> **用途（采纳 GPT 审稿建议 2026-08-02）**：所有实验矩阵集中管理，保证公平对比、证据充分。与 `PAPER_PROGRESS.md` §4（主线）和 `RESEARCH_LOG.md` §6（结果日志）联动。

---

## 1. 数据集（WWW 标准：不能只有 ml-1m）

| 数据集 | 规模 | 定位 | 验证点 |
|---|---|---|---|
| `ml-1m` | 999,611 交互 / 6,040 用户 / 3,416 物品 | 快速验证 | 主实验、消融 |
| `Beauty` | ~198k / ~22k / ~12k | **稀疏** | RQ3：density 在稀疏下优势 |
| `Steam` | 大 | **长尾/游戏** | RQ3：长尾 |
| `Yelp`（候选） | 大 | **兴趣变化** | E004 兴趣漂移 |
- ⚠️ 数据文件不入库（gitignore），服务器/本地单独放置 `data/`。

## 2. Baseline（WWW 标准：不能只有 SASRec）

**序列 baseline（Sequential）**
| baseline | 说明 | 状态 |
|---|---|---|
| SASRec | 原始点估计基线（同参数量） | ✅ 已有（vector variant） |
| GRU4Rec | RNN 序列基线 | ⬜ 待接入（Priority 4 泛化） |
| BERT4Rec | 双向 Transformer 基线 | ⬜ 待接入（Priority 4 泛化） |

**不确定性/表示 baseline（必须，否则 RQ1 不公平）**
| baseline | 说明 | 回答 |
|---|---|---|
| Gaussian embedding | $z\sim\mathcal N(\mu,\Sigma)$，KL/内积打分 | RQ1：density vs 分布 |
| MIND / ComiRec | multi-interest 表示 | RQ1：density vs 多兴趣 |
- ⚠️ 不加这些，审稿人会说"只对比了确定性向量，不公平"。

## 3. 主实验矩阵（与 PAPER_PROGRESS §4.1 对应）

| 阶段 | 实验 | 对比（同参数量） | 指标 | 目的 |
|---|---|---|---|---|
| 1 | E001 | SASRec vector | NDCG@10 / HR@10 | 基线复现 |
| 2 | E002 | vector vs state-r1 vs state-r4 vs Gaussian | NDCG@10 / HR@10 | **Representation Ablation**（RQ1） |
| 3 | E003 | static(末状态) vs EMA(fixed α) vs dynamic(learnable α) | NDCG@10 / HR@10 | **Evolution Ablation**（RQ2 核心图） |
| 4 | E004 | 前 50 Action → 后 50 Romance | **adaptation steps** | Interest shift（RQ2/H2 最有力证据） |

## 4. 分析实验（附）

| 分析 | 内容 | 指标 |
|---|---|---|
| A-1 维度效率 | vector $d=64$ vs state $d=32,r=4$ | NDCG/HR |
| A-2 序列长度 | history 5/10/20/50 | 增益曲线 |
| A-3 匹配消融 | dot vs trace（纯态下 $Tr=(u\cdot i)^2$） | NDCG/HR |
| A-4 长尾 | 按 popularity 分 head/mid/tail | 分组 NDCG/HR |
| A-5 多样性 | coverage / ILD | RQ3 补充 |

## 5. 铁律（违反则结论作废）
- 除被验证维度外**同超参**（lr/batch/maxlen/hidden/blocks/heads/dropout/epochs）；
- **Tr/BCE 兼容修正落地后**（temperature 或 logit 变换）再跑正式实验；
- 结果一律回填 `docs/RESEARCH_LOG.md` §6 + `docs/PAPER_PROGRESS.md` §4，并更新本表"状态"。

## 6. GPU 服务器运行
```bash
cd SASRec.pytorch-main
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu121  # 按 nvidia-smi 的 CUDA 版本
python run_experiments.py --dataset ml-1m --epochs 200 --device cuda --tag main
# 结果 → results/exp_main.csv → 回填 RESEARCH_LOG §6
```
