# 服务器一键运行（SERVER_RUN）

> 目标：在 GPU 服务器上 **一条命令** 完成 —— 拉取仓库 → 建环境装包 → 放数据 → 冒烟；**实验按 §2 清单顺序执行**（当前：rank ablation → matching ablation → Beauty/Steam）。
> 进度：**Step 1（ml-1m 主实验）已跑完**（E001，见 `02_research_log.md` §6）。
> 配套：`05_experiment_plan.md`（实验设计）、`06_usage_sasrec.md`（参数说明）、`requirements.txt`（pip 依赖，无 uv 服务器用）。

---

## 0. 前置条件（一次性）
- 系统：Linux + GPU（`nvidia-smi` 确认 CUDA 版本，决定 torch 的 `cu1XX`）
- 已安装：`git`、**conda**（miniconda / anaconda）
- 已配置 GitHub 认证（否则 `git clone` 会要求账号）

## 1. 一键命令（复制整段，粘贴到服务器终端执行）

```bash
set -e

# ---------- 1) 拉取代码 ----------
git clone https://github.com/powell4322/Quantum-SR.git
cd Quantum-SR

# ---------- 2) 建 conda 环境 + 安装依赖（环境名 ddsr，可自定） ----------
# 若环境已存在，conda create 会报错，故先检查再创建
conda env list | grep -q '^ddsr ' || conda create -n ddsr python=3.12 -y
conda activate ddsr
pip install --upgrade pip
# ⚠️ 先按 nvidia-smi 的 CUDA 版本装 cu 版 torch（cu118 / cu121 / cu124 ...），再装其余依赖
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# ---------- 3) 放置数据（数据不入库，需自行上传） ----------
# 把三个数据集文件 ml-1m.txt / Beauty.txt / Steam.txt 放到 data/ 下（每行: user item，空格分隔，从 1 编号）
mkdir -p data
# 示例：scp data/ml-1m.txt user@server:/path/Quantum-SR/data/

# ---------- 4) 冒烟自检（可选但建议） ----------
python test_smoke.py

# ---------- 5) 跑当前待做实验：Phase 1 rank ablation（完整清单见 §2） ----------
conda activate ddsr
python run_experiments.py --dataset ml-1m --epochs 200 --device cuda --loss bpr \
    --variants state --state_rank 1 4 8 16 --tag rank
python run_experiments.py --dataset ml-1m --epochs 200 --device cuda --loss bpr \
    --variants dynamic --state_rank 4 8 16 --tag rank
```

## 2. 实验清单与先后顺序（按此执行）

> 进度：**Step 1（ml-1m 主实验）✅ 已跑完**（E001：V 0.5852 / DF 0.5745 / VE 0.5604 / DDS 0.4950 / DS 0.4622，诊断见 `02_research_log.md` §6.1）。以下为**当前待跑**，按顺序执行。

### ✅ Step 1（已完成）— 主实验 · ml-1m
- 已跑完（E001，`results/exp_main.csv`）；命令保留在 git 历史，不再重复跑。

### Step 2 — Phase 1 Rank Ablation（ml-1m，最高优先级，`agent/research_plan.md` §5 Phase 1）
验证：rank 是否把 DS / DDS 拉回 V 之上（**Case A**：rank8>rank1 → mixed state 有效；**Case B**：rank 无改善 → 需重新设计打分）。
```bash
# Static：rank=1/4/8/16
python run_experiments.py --dataset ml-1m --epochs 200 --device cuda --loss bpr \
    --variants state --state_rank 1 4 8 16 --tag rank
# Dynamic：rank=4/8/16（rank=1 已有 E001）
python run_experiments.py --dataset ml-1m --epochs 200 --device cuda --loss bpr \
    --variants dynamic --state_rank 4 8 16 --tag rank
```
> 预判（`research_log` §6.1）：瓶颈在打分尺度而非秩，rank 提升可能有限（Case B 风险）；跑完对照诊断决定是否进 Phase 3（confidence-aware scoring）。

### Step 3 — Phase 2 Matching Ablation（ml-1m，dynamic r8，dot vs Tr）
```bash
python main.py --dataset ml-1m --train_dir rq3_ds_dot  --variant state   --matching dot --device cuda --loss bpr --num_epochs 200
python main.py --dataset ml-1m --train_dir rq3_dds_dot --variant dynamic --matching dot --device cuda --loss bpr --num_epochs 200
```

### Step 4 — 主实验 · Beauty / Steam（待 rank 结论后；稀疏 / 长尾验证）
```bash
for ds in Beauty Steam; do
  python run_experiments.py --dataset $ds --epochs 200 --device cuda --loss bpr \
      --variants vector density_feature state dynamic vector_evolve --state_rank 1 --tag main
done
```

### Step 5 — E003 ρ0 初始化消融（I/d / First observation / Learnable）
> 当前代码默认 I/d；First observation / Learnable 需代码扩展后补跑（见 `02_research_log.md` §4.5 待办）。

### Step 6 — RQ4 entropy 分析（离线，用已训练 DDS 权重）
> 用 `SASRec.state_entropy` 对测试用户按 $H(\rho_u)$ 分低/中/高组，比较各组 DDS vs V 的增益。脚本在结果回填阶段补写。

---

## 4. 结果产物
- 主实验汇总 → `results/exp_main.csv`（列：`variant,ndcg10,recall10,last_loss`）
- 每个方案独立目录：`{dataset}_quant_{variant}_r{rank}_{tag}/`（含 `args.txt` / `log.txt` / 最优权重 `.pth`）
- **回填**：把 CSV 数字填入 `docs/02_research_log.md` §6 与 `docs/01_paper_progress.md` §4

## 5. 常见问题
| 现象 | 处理 |
|---|---|
| `python` 找不到 | 先 `conda activate ddsr`（或你自己的环境名） |
| torch 装不上 / CUDA 不识别 | `nvidia-smi` 查版本，改 `--index-url .../whl/cu1XX` |
| 显存不足 OOM | `--batch_size 64` 或 `--maxlen 100` 或 `--hidden_units 32` |
| 数据读不到 | 确认 `data/{dataset}.txt` 存在且格式为 `user item` |
| 想断点续训 / 只看推理 | 参考 `06_usage_sasrec.md`（`--state_dict_path` / `--inference_only`） |
