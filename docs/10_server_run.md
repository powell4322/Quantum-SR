# 服务器一键运行（SERVER_RUN）

> 目标：在 GPU 服务器上 **一条命令** 完成 —— 拉取仓库 → 建环境装包 → 放数据 → 跑完 DDST 主实验（四阶递进 **V < DF < DS < DDS** + VE 对照）。
> 配套：`05_experiment_plan.md`（实验设计）、`06_usage_sasrec.md`（参数说明）。

---

## 0. 前置条件（一次性）
- 系统：Linux + GPU（`nvidia-smi` 确认 CUDA 版本，决定 torch 的 `cu1XX`）
- 已安装：`git`、`python3`（3.10+）、`python3-venv`
- 已配置 GitHub 认证（否则 `git clone` 会要求账号）

## 1. 一键命令（复制整段，粘贴到服务器终端执行）

```bash
set -e

# ---------- 1) 拉取代码 ----------
git clone https://github.com/powell4322/Quantum-SR.git
cd Quantum-SR

# ---------- 2) 建虚拟环境 + 安装依赖 ----------
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy
# 按 nvidia-smi 的 CUDA 版本选择 cu 后缀（cu118 / cu121 / cu124 ...）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# ---------- 3) 放置数据（数据不入库，需自行上传） ----------
# 把三个数据集文件 ml-1m.txt / Beauty.txt / Steam.txt 放到 data/ 下（每行: user item，空格分隔，从 1 编号）
mkdir -p data
# 示例：scp data/ml-1m.txt user@server:/path/Quantum-SR/data/

# ---------- 4) 冒烟自检（可选但建议） ----------
python test_smoke.py

# ---------- 5) 跑主实验（ml-1m，四阶递进 + VE 对照） ----------
python run_experiments.py --dataset ml-1m --epochs 200 --device cuda --loss bpr \
    --variants vector density_feature state dynamic vector_evolve --state_rank 1 --tag main
```

## 2. 三个数据集都跑

```bash
source .venv/bin/activate
cd Quantum-SR   # 如不在仓库根目录

for ds in ml-1m Beauty Steam; do
  python run_experiments.py --dataset $ds --epochs 200 --device cuda --loss bpr \
      --variants vector density_feature state dynamic vector_evolve --state_rank 1 --tag main
done
```

## 3. 消融实验

```bash
source .venv/bin/activate

# RQ3：matching 消融 —— DDS 用 dot（一阶方向）对照 trace（二阶 HS）
python main.py --dataset ml-1m --train_dir rq3_ds_dot  --variant state   --matching dot --device cuda --loss bpr --num_epochs 200
python main.py --dataset ml-1m --train_dir rq3_dds_dot --variant dynamic --matching dot --device cuda --loss bpr --num_epochs 200

# E003：ρ0 初始化消融（I/d 为当前默认；First observation / Learnable 后续代码扩展后补跑）
```

## 4. 结果产物
- 主实验汇总 → `results/exp_main.csv`（列：`variant,ndcg10,recall10,last_loss`）
- 每个方案独立目录：`{dataset}_quant_{variant}_r{rank}_{tag}/`（含 `args.txt` / `log.txt` / 最优权重 `.pth`）
- **回填**：把 CSV 数字填入 `docs/02_research_log.md` §6 与 `docs/01_paper_progress.md` §4

## 5. 常见问题
| 现象 | 处理 |
|---|---|
| `python` 找不到 | 先 `source .venv/bin/activate` |
| torch 装不上 / CUDA 不识别 | `nvidia-smi` 查版本，改 `--index-url .../whl/cu1XX` |
| 显存不足 OOM | `--batch_size 64` 或 `--maxlen 100` 或 `--hidden_units 32` |
| 数据读不到 | 确认 `data/{dataset}.txt` 存在且格式为 `user item` |
| 想断点续训 / 只看推理 | 参考 `06_usage_sasrec.md`（`--state_dict_path` / `--inference_only`） |
