---
name: research-experiment
description: 'Run and log SASRec multi-variant experiments (vector/state/dynamic/density_feature/vector_evolve) for the DDST (Dynamic Density State Transformer) project. Use when running experiments, comparing variants, parsing metrics (R@10/NDCG@10), filling RESEARCH_LOG section 6, preparing GPU-server runs, choosing hyperparameters, or ensuring fair same-hyperparameter comparisons. Covers run_experiments.py usage, --eval_every, --matching, metric parsing, and the strict logging protocol.'
user-invocable: true
---

# 实验协议（Research Experiment Protocol）

## 运行环境
- 本机解释器：`.venv\Scripts\python.exe`（Python 3.12，torch CPU）。
- PATH 里的 `python` 是 WindowsApps 占位符，**不可用**。
- 服务器 GPU：按 CUDA 版本装 GPU 版 torch（见"GPU 服务器"）。

## 五个方案（variant，M0-M4 映射见 docs/05_experiment_plan.md §0）
| variant | M | 表示 | 演化 | 打分 | 回答 RQ |
|---|---|---|---|---|---|
| `vector` | M0 | $h_t$ | 无 | dot | 基线 |
| `density_feature` | M1 | $ee^\top$（feature） | 无 | dot | 二阶表示 |
| `state` | M2 | $\rho_t$ | 无 | $\mathrm{Tr}$（`--matching trace`） | RQ1/RQ3 |
| `dynamic` | M3 | $\rho_t$ | 凸组合 | $\mathrm{Tr}$（`--matching trace`） | RQ2/RQ3（核心） |
| `vector_evolve` | VE | $h_t$ | 向量 EMA | dot | EMA 对照 |

- `--matching dot`：state/dynamic 用一阶方向 dot（RQ3 匹配消融）。
- 指标统一 **R@10 / NDCG@10**（本协议下 HR@10 ≡ Recall@10）。

## 跑实验
```bash
# 批量对比（默认 vector/state/dynamic，ml-1m）
.venv\Scripts\python.exe run_experiments.py --dataset ml-1m --epochs 200 --device cuda
# 快速对比（每 5 轮评估一次）
.venv\Scripts\python.exe run_experiments.py --dataset ml-1m --epochs 5 --eval_every 5 --maxlen 50 --hidden_units 32 --device cpu --tag quick
# 单方案
.venv\Scripts\python.exe main.py --dataset ml-1m --train_dir xxx --variant dynamic --num_epochs 200 --eval_every 20 --device cuda
```
结果自动汇总到 `results/exp_<tag>.csv`。`--eval_every` 控制评估频率（默认 20，保持原行为）。

## 公平对比铁律（违反则结论作废）
- 除"被验证维度"外，**所有超参必须一致**（lr/batch/maxlen/hidden/blocks/heads/dropout/epochs）。
- 同一配置复跑以覆盖更新；不同配置必须换实验行并注明。

## 记录协议（每次实验必做）
1. 把结果回填到 `docs/02_research_log.md` §6 表格（日期/variant/配置/NDCG/HR/相对基线/结论）。
2. 结论栏写清"支撑/否定哪个 RQ"；若结果与预期不符，先查 §4.4 已知问题再下结论。
3. 更新 §3.2 对应 RQ 的状态（⬜→✅/❌）。

## GPU 服务器运行步骤
1. 上传数据：`data/` 被 gitignore，需单独把 `data/ml-1m.txt` 传到服务器。
2. 拉取代码：`git clone <url>`（或同步工作区）。
3. 装依赖：`pip install torch numpy`（GPU 版，按 `nvidia-smi` 的 CUDA 版本选 wheel）。
4. 跑：`python run_experiments.py --dataset ml-1m --epochs 200 --device cuda`。
5. 回传 `results/exp_latest.csv` 或指标截图，由助手回填 `docs/02_research_log.md`。

## 回归
改 `model.py` 后必跑：`.venv\Scripts\python.exe test_smoke.py`（校验三种 variant 与密度矩阵合法性 PSD+trace=1）。

## 已知注意事项
- Tr 打分与 BCE 不匹配问题见 `docs/02_research_log.md` §4.4——已于 2026-08-03 修复（logit 变换 + 默认 BPR；Tr 经 `_logit_score` 映射）。
- 领域知识/定位见 `ddst-seq-rec` skill 与 `docs/09_research_positioning_v2.md`。
- 修正打分/损失时优先做成可切换参数，避免为每个方案维护独立代码。
