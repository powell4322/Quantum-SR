# -*- coding: utf-8 -*-
"""冒烟测试：验证 SASRec 三种 variant（vector/state/dynamic）的
forward / predict 路径正确，且密度矩阵满足 PSD + trace=1。

运行：.venv\\Scripts\\python.exe test_smoke.py
"""
import argparse

import numpy as np
import torch

from model import SASRec


def make_args(variant, rank=1, device="cpu"):
    return argparse.Namespace(
        device=device,
        hidden_units=16,
        maxlen=10,
        num_blocks=1,
        num_heads=1,
        dropout_rate=0.1,
        norm_first=False,
        variant=variant,
        state_rank=rank,
        transition="fixed",
        transition_alpha=0.9,
        matching="trace",
    )


def check_state_legality(rho, tag):
    """密度矩阵合法性：PSD 且 trace=1。"""
    assert rho.shape[-2] == rho.shape[-1], "rho 必须是方阵"
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)
    err_trace = (trace - 1.0).abs().max().item()
    assert err_trace < 1e-4, "{} trace 偏差过大: {}".format(tag, err_trace)
    # 特征值非负（对称化避免数值误差）
    sym = (rho + rho.transpose(-1, -2)) / 2
    eig = torch.linalg.eigvalsh(sym)
    min_eig = eig.min().item()
    assert min_eig > -1e-5, "{} 最小特征值异常: {}".format(tag, min_eig)
    print("    [OK] {}: trace_err={:.2e}, min_eig={:.2e}".format(tag, err_trace, min_eig))


def run_variant(variant, rank=1, matching="trace"):
    print("\n=== variant = {} (rank={}, matching={}) ===".format(variant, rank, matching))
    args = make_args(variant, rank)
    args.matching = matching
    model = SASRec(user_num=100, item_num=50, args=args)
    model.train()

    B, T, I = 4, 10, 11
    log_seqs = torch.randint(1, 51, (B, T)).numpy()
    pos_seqs = torch.randint(1, 51, (B, T)).numpy()
    neg_seqs = torch.randint(1, 51, (B, T)).numpy()
    user_ids = torch.arange(1, B + 1).numpy()

    pos_logits, neg_logits = model(user_ids, log_seqs, pos_seqs, neg_seqs)
    assert pos_logits.shape == (B, T), "pos_logits shape 错误: {}".format(pos_logits.shape)
    assert neg_logits.shape == (B, T)
    assert torch.isfinite(pos_logits).all() and torch.isfinite(neg_logits).all()
    print("    [OK] forward: pos_logits shape={}, finite={}".format(
        tuple(pos_logits.shape), bool(torch.isfinite(pos_logits).all())))

    # 密度矩阵合法性（state/dynamic）
    if variant in ("state", "dynamic"):
        log_feats = model.log2feats(log_seqs)
        rho_seq = model._to_state_sequence(log_feats)
        assert rho_seq.shape == (B, T, args.hidden_units, args.hidden_units)
        check_state_legality(rho_seq, "user rho_seq")
        # item 状态
        item_embs = model.item_emb(torch.LongTensor(np.random.randint(1, 51, (B, I))))
        rho_item = model.state_proj(item_embs)
        check_state_legality(rho_item, "item rho")

    # 低秩打分 ≡ 显式 rho 逐元素（数学等价性校验，防优化改语义）
    if variant == "state" and matching == "trace":
        L_u, n_u = model.state_proj.lowrank(log_feats)
        L_i, n_i = model.state_proj.lowrank(model.item_emb(torch.LongTensor(pos_seqs)))
        s_low = model._hs_lowrank(L_u, n_u, L_i, n_i)
        rho_u = model.state_proj(log_feats)
        rho_i = model.state_proj(model.item_emb(torch.LongTensor(pos_seqs)))
        s_tr = (rho_u * rho_i).sum(dim=(-1, -2))
        err = (s_low - s_tr).abs().max().item()
        assert err < 1e-4, "低秩打分与 Tr 不一致: {}".format(err)
        print("    [OK] low-rank == trace: max_err={:.2e}".format(err))

    # predict 路径
    item_idx = torch.randint(1, 51, (B, I)).numpy()
    logits = model.predict(user_ids, log_seqs, item_idx)
    assert logits.shape == (B, I), "logits shape 错误: {}".format(logits.shape)
    assert torch.isfinite(logits).all()
    print("    [OK] predict: logits shape={}, finite={}".format(
        tuple(logits.shape), bool(torch.isfinite(logits).all())))

    # backward 冒烟
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pos_logits.flatten(), torch.ones_like(pos_logits.flatten()))
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    print("    [OK] backward: {} grads computed, loss={:.4f}".format(sum(grads), loss.item()))


def check_lowrank_equiv_robust():
    """多 batch/rank/seed 下验证 low-rank 打分 ≡ 显式 matrix 打分（state 与 dynamic）。"""
    print("\n=== low-rank vs matrix equivalence (robust) ===")
    worst_state, worst_dyn = 0.0, 0.0
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        np.random.seed(seed)
        for rank in (1, 2, 4):
            for B in (2, 8):
                T = 6
                log_seqs = torch.randint(1, 51, (B, T)).numpy()
                pos_seqs = torch.randint(1, 51, (B, T)).numpy()

                # state：_hs_lowrank vs 显式 rho 逐元素
                model = SASRec(user_num=100, item_num=50, args=make_args("state", rank))
                log_feats = model.log2feats(log_seqs)
                L_u, n_u = model.state_proj.lowrank(log_feats)
                L_i, n_i = model.state_proj.lowrank(model.item_emb(torch.LongTensor(pos_seqs)))
                s_low = model._hs_lowrank(L_u, n_u, L_i, n_i)
                s_mat = (model.state_proj(log_feats) * model.state_proj(model.item_emb(torch.LongTensor(pos_seqs)))).sum(dim=(-1, -2))
                err_s = (s_low - s_mat).abs().max().item()
                worst_state = max(worst_state, err_s)
                assert err_s < 1e-4, "state lowrank!=matrix err={}".format(err_s)

                # dynamic：_hs_mixed（演化后 rho vs item 低秩） vs 显式 rho*rho
                model2 = SASRec(user_num=100, item_num=50, args=make_args("dynamic", rank))
                log_feats2 = model2.log2feats(log_seqs)
                rho_seq = model2._to_state_sequence(log_feats2)
                L_pos, n_pos = model2.state_proj.lowrank(model2.item_emb(torch.LongTensor(pos_seqs)))
                s_mixed = model2._hs_mixed(rho_seq, L_pos, n_pos)
                s_mat2 = (rho_seq * model2.state_proj(model2.item_emb(torch.LongTensor(pos_seqs)))).sum(dim=(-1, -2))
                err_m = (s_mixed - s_mat2).abs().max().item()
                worst_dyn = max(worst_dyn, err_m)
                assert err_m < 1e-4, "dynamic mixed!=matrix err={}".format(err_m)

                print("    [OK] seed={} rank={} B={}: state_err={:.1e} dyn_err={:.1e}".format(seed, rank, B, err_s, err_m))
    print("    [OK] low-rank == matrix (worst: state={:.1e}, dynamic={:.1e})".format(worst_state, worst_dyn))


if __name__ == "__main__":
    for v in ("vector", "state", "dynamic", "vector_evolve", "density_feature"):
        run_variant(v)
    run_variant("state", rank=4)
    # RQ3：dot matching（一阶方向 dot）在 state/dynamic 下也应通过
    for v in ("state", "dynamic"):
        run_variant(v, matching="dot")
    # 稳健等价性：不同 batch/rank/seed 下 low-rank == 显式 matrix（state 与 dynamic）
    check_lowrank_equiv_robust()
    print("\nALL SMOKE TESTS PASSED")
