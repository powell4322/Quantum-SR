# -*- coding: utf-8 -*-
"""批量运行 SASRec 多方案对比实验（vector / state / dynamic），并汇总指标。

用法示例：
    python run_experiments.py --dataset ml-1m --epochs 200 --device cuda
    python run_experiments.py --dataset ml-1m --epochs 200 --variants vector state dynamic --state_rank 1 4
    python run_experiments.py --dataset ml-1m --epochs 100 --device cpu --tag v1

说明：
    - 每个方案使用独立的 train_dir，避免互相覆盖。
    - 指标从 main.py 的标准输出中解析（最后一个 test (NDCG@10, HR@10)）。
    - 结果同时打印到控制台并写入 results/exp_{tag}.csv。
"""
import argparse
import os
import re
import subprocess
import sys
import time

TEST_RE = re.compile(r"test \(NDCG@10: ([\d.]+), HR@10: ([\d.]+)\)")
LOSS_RE = re.compile(r"loss in epoch (\d+) iteration (\d+): ([\d.]+)")


def parse_metrics(stdout):
    """从 main.py 输出中提取 (ndcg, hr) 与最后一个 loss。"""
    tests = TEST_RE.findall(stdout)
    ndcg, hr = None, None
    if tests:
        ndcg, hr = float(tests[-1][0]), float(tests[-1][1])
    losses = LOSS_RE.findall(stdout)
    last_loss = float(losses[-1][2]) if losses else None
    return ndcg, hr, last_loss


def run_one(args, variant, rank):
    """运行单个方案，返回 (name, ndcg, hr, loss)。"""
    train_dir = "quant_{}_r{}".format(variant, rank)
    if args.tag:
        train_dir += "_{}".format(args.tag)

    cmd = [
        sys.executable, "main.py",
        "--dataset", args.dataset,
        "--train_dir", train_dir,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--maxlen", str(args.maxlen),
        "--hidden_units", str(args.hidden_units),
        "--num_blocks", str(args.num_blocks),
        "--num_heads", str(args.num_heads),
        "--dropout_rate", str(args.dropout_rate),
        "--num_epochs", str(args.epochs),
        "--device", args.device,
        "--variant", variant,
        "--state_rank", str(rank),
        "--transition", args.transition,
        "--eval_every", str(args.eval_every),
        "--loss", args.loss,
    ]
    name = "{}_r{}".format(variant, rank)

    print("\n" + "=" * 70)
    print("[EXPERIMENT] {} | cmd: {}".format(name, " ".join(cmd)))
    print("=" * 70)

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout, stderr = proc.stdout, proc.stderr
    except Exception as e:  # 命令本身失败（环境问题等）
        print("[FAILED] {}: {}".format(name, e))
        return (name, None, None, None)

    elapsed = (time.time() - t0) / 60.0
    if proc.returncode != 0:
        print("[FAILED] {} (rc={})".format(name, proc.returncode))
        if stderr:
            print(stderr[-2000:])
        return (name, None, None, None)

    ndcg, hr, loss = parse_metrics(stdout)
    print("[DONE] {} | time={:.1f}min | test NDCG@10={} HR@10={} | last_loss={}".format(
        name, elapsed, ndcg, hr, loss))
    return (name, ndcg, hr, loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--variants", nargs="+", default=["vector", "state", "dynamic"])
    parser.add_argument("--state_rank", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default=None, help="实验批次标记，用于区分不同轮次的 train_dir")
    parser.add_argument("--transition", default="fixed", choices=["fixed", "learnable"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--hidden_units", type=int, default=50)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--loss", default="bce", choices=["bce", "bpr"])
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    rows = []
    for rank in args.state_rank:
        for variant in args.variants:
            rows.append(run_one(args, variant, rank))

    # 汇总
    print("\n" + "=" * 70)
    print("SUMMARY (dataset={}, epochs={}, tag={})".format(args.dataset, args.epochs, args.tag))
    print("=" * 70)
    header = "{:<16} {:>12} {:>12} {:>12}".format("variant", "NDCG@10", "HR@10", "last_loss")
    print(header)
    print("-" * 52)
    baseline = None
    for name, ndcg, hr, loss in rows:
        if name.startswith("vector"):
            baseline = ndcg
        rel = "1.00x"
        if baseline and ndcg is not None:
            rel = "{:.2f}x".format(ndcg / baseline)
        print("{:<16} {:>12} {:>12} {:>12}   {}".format(
            name,
            "%.4f" % ndcg if ndcg is not None else "-",
            "%.4f" % hr if hr is not None else "-",
            "%.4f" % loss if loss is not None else "-",
            rel))

    # 保存 CSV
    csv_path = os.path.join("results", "exp_{}.csv".format(args.tag or "latest"))
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("variant,ndcg10,hr10,last_loss\n")
        for name, ndcg, hr, loss in rows:
            f.write("{},{},{},{}\n".format(
                name,
                "%.4f" % ndcg if ndcg is not None else "",
                "%.4f" % hr if hr is not None else "",
                "%.4f" % loss if loss is not None else ""))
    print("\nSaved to {}".format(csv_path))


if __name__ == "__main__":
    main()
