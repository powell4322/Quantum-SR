---
name: research-experiment
description: 'Run and log SASRec multi-variant experiments (vector/state/dynamic) for this quantum-inspired sequential recommendation project. Use when running experiments, comparing variants, parsing metrics, filling RESEARCH_LOG section 6, preparing GPU-server runs, choosing hyperparameters, or ensuring fair same-hyperparameter comparisons. Covers run_experiments.py usage, --eval_every, metric parsing, and the strict logging protocol.'
user-invocable: true
---

# 实验协议（Research Experiment Protocol）

## 运行环境
- 本机解释器：`.venv\Scripts\python.exe`（Python 3.12，torch CPU）。
- PATH 里的 `python` 是 WindowsApps 占位符，**不可用**。
- 服务器 GPU：按 CUDA 版本装 GPU 版 torch（见"GPU 服务器"）。

## 三个方案（variant）
| variant | 表示 | 演化 | 打分 | 回答 RQ |
|---|---|---|---|---|
| `vector` | $h_t$ | 无 | dot | 基线 |
| `state` | $\rho_t$ | 无 | $\mathrm{Tr}$ | RQ1/RQ2 |
| `dynamic` | $\rho_t$ | 凸组合 | $\mathrm{Tr}$ | RQ3（核心） |

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
1. 把结果回填到 `RESEARCH_LOG.md` §6 表格（日期/variant/配置/NDCG/HR/相对基线/结论）。
2. 结论栏写清"支撑/否定哪个 RQ"；若结果与预期不符，先查 §4.4 已知问题再下结论。
3. 更新 §3.2 对应 RQ 的状态（⬜→✅/❌）。

## GPU 服务器运行步骤
1. 上传数据：`data/` 被 gitignore，需单独把 `data/ml-1m.txt` 传到服务器。
2. 拉取代码：`git clone <url>`（或同步工作区）。
3. 装依赖：`pip install torch numpy`（GPU 版，按 `nvidia-smi` 的 CUDA 版本选 wheel）。
4. 跑：`python run_experiments.py --dataset ml-1m --epochs 200 --device cuda`。
5. 回传 `results/exp_latest.csv` 或指标截图，由助手回填 `RESEARCH_LOG.md`。

## 回归
改 `model.py` 后必跑：`.venv\Scripts\python.exe test_smoke.py`（校验三种 variant 与密度矩阵合法性 PSD+trace=1）。

## 已知注意事项
- Tr 打分与 BCE 不匹配问题见 `quantum-seq-rec` skill 与 `RESEARCH_LOG.md` §4.4——正式实验前需确认是否已采用修正方案。
- 修正打分/损失时优先做成可切换参数，避免为每个方案维护独立代码。
