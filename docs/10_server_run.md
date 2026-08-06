# 服务器一键运行（SERVER_RUN）

> 目标：在 GPU 服务器上 **一条命令** 完成 —— 拉取仓库 → 建环境装包 → 放数据 → 冒烟；**实验按 §2 清单顺序执行**（主实验 → 消融 → 分析）。
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

# ---------- 5) 跑主实验 ml-1m（= §2 Step 1；完整顺序见 §2） ----------
python run_experiments.py --dataset ml-1m --epochs 200 --device cuda --loss bpr \
    --variants vector density_feature state dynamic vector_evolve --state_rank 1 --tag main
```

## 2. 实验清单与先后顺序（按此执行）

> 原则：**先用 ml-1m 快速验证趋势，再跑 Beauty / Steam**；**先主实验（回答 RQ1/RQ2），再消融（RQ3），最后离线分析（RQ4）**。每个 Step 产出回填 `02_research_log.md` §6。

### Step 1 — 主实验 · ml-1m（先验证四阶递进趋势）
验证目标：**V < DF < DS < DDS**（RQ1/RQ2），且 **DDS − DS 增益 ≠ VE 的 EMA 增益**。
```bash
conda activate ddsr && cd Quantum-SR
python run_experiments.py --dataset ml-1m --epochs 200 --device cuda --loss bpr \
    --variants vector density_feature state dynamic vector_evolve --state_rank 1 --tag main
```
> ⚠️ 若 ml-1m 趋势不成立：先停，检查 `02_research_log.md` §4.4/§4.5 已知问题，**不要直接跑大数据集**。

### Step 2 — 主实验 · Beauty / Steam（稀疏 / 长尾验证）
```bash
for ds in Beauty Steam; do
  python run_experiments.py --dataset $ds --epochs 200 --device cuda --loss bpr \
      --variants vector density_feature state dynamic vector_evolve --state_rank 1 --tag main
done
```

### Step 3 — RQ3 匹配消融（ml-1m，同状态下 dot vs Tr）
```bash
python main.py --dataset ml-1m --train_dir rq3_ds_dot  --variant state   --matching dot --device cuda --loss bpr --num_epochs 200
python main.py --dataset ml-1m --train_dir rq3_dds_dot --variant dynamic --matching dot --device cuda --loss bpr --num_epochs 200
```

### Step 4 — E003 ρ0 初始化消融（I/d / First observation / Learnable）
> 当前代码默认 I/d；First observation / Learnable 需代码扩展后补跑（见 `02_research_log.md` §4.5 待办）。

### Step 5 — RQ4 entropy 分析（离线，用已训练 DDS 权重）
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
