# -*- coding: utf-8 -*-
"""Phase 0 诊断（DeepSeek 计划 §5）：确认第一轮 ml-1m 失败原因。

只读诊断，**不改模型结构 / 不改 loss / 不加模块**。复用 model + utils 收集：
  - Norm Analysis      : ||h_last||  vs ||L_last||_F（density 归一化是否抹平强度）
  - Score Distribution : 原始打分 h_u·e_i vs Tr(rho_u rho_i)（是否压缩到 [0, 0.2]）
  - Gradient Analysis  : 最后 attention 层梯度范数（vector vs density 是否梯度不足）

用法（本机 CPU 快速验证）：
  .venv\\Scripts\\python.exe diagnose.py --dataset ml-1m --variant vector    --maxlen 50 --steps 20
  .venv\\Scripts\\python.exe diagnose.py --dataset ml-1m --variant state     --state_rank 8 --maxlen 50 --steps 20
  .venv\\Scripts\\python.exe diagnose.py --dataset ml-1m --variant dynamic   --state_rank 8 --maxlen 50 --steps 20
服务器（GPU）：
  python diagnose.py --dataset ml-1m --variant dynamic --state_rank 8 --device cuda --steps 50
"""
import argparse

import numpy as np
import torch

from model import SASRec
from utils import data_partition, WarpSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-1m')
    p.add_argument('--variant', default='vector',
                   choices=['vector', 'state', 'dynamic', 'vector_evolve', 'density_feature'])
    p.add_argument('--state_rank', default=1, type=int)
    p.add_argument('--matching', default='trace', choices=['trace', 'dot'])
    p.add_argument('--scoring', default='trace', choices=['trace', 'covariance', 'confidence'])
    p.add_argument('--scoring_gamma', default=1.0, type=float)
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--lr', default=0.001, type=float)
    p.add_argument('--maxlen', default=200, type=int)
    p.add_argument('--hidden_units', default=50, type=int)
    p.add_argument('--num_blocks', default=2, type=int)
    p.add_argument('--num_heads', default=1, type=int)
    p.add_argument('--dropout_rate', default=0.2, type=float)
    p.add_argument('--norm_first', action='store_true', default=False)
    p.add_argument('--device', default='cpu', type=str)
    p.add_argument('--steps', default=30, type=int, help='诊断训练步数')
    p.add_argument('--loss', default='bpr', choices=['bce', 'bpr'])
    p.add_argument('--seed', default=0, type=int)
    return p.parse_args()


def report(name, arr):
    a = np.asarray(arr).ravel()
    if a.size == 0:
        return
    print('  {:<28s} mean={:8.4f} std={:8.4f}  p5={:7.4f} p50={:7.4f} p95={:7.4f}'.format(
        name, a.mean(), a.std(),
        np.percentile(a, 5), np.percentile(a, 50), np.percentile(a, 95)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    sampler = WarpSampler(user_train, usernum, itemnum,
                          batch_size=args.batch_size, maxlen=args.maxlen, n_workers=2)
    model = SASRec(usernum, itemnum, args).to(device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    bce = torch.nn.BCEWithLogitsLoss()
    model.train()

    h_norms, L_norms, s_pos, s_neg, grad_norms = [], [], [], [], []
    is_density = args.variant in ('state', 'dynamic')

    for step in range(args.steps):
        u, seq, pos, neg = sampler.next_batch()
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        pos_t = torch.LongTensor(pos).to(device)
        neg_t = torch.LongTensor(neg).to(device)

        # ---- 诊断收集（no_grad，不污染训练图） ----
        with torch.no_grad():
            log_feats = model.log2feats(seq)          # (U, T, C)
            last_h = log_feats[:, -1]                  # (U, C)
            h_norms.append(last_h.norm(dim=-1).cpu().numpy())

            if is_density:
                L_last = model.state_proj.lowrank(log_feats)[0][:, -1]   # (U, r, C)
                L_norms.append(L_last.norm(dim=(-1, -2)).cpu().numpy())
                # 统一取最后一步做打分分布诊断（与 predict 一致）
                item_pos = model.item_emb(pos_t)   # (U, T, C)
                item_neg = model.item_emb(neg_t)
                if args.matching == 'trace':
                    if args.scoring == 'trace':
                        if args.variant == 'dynamic':
                            rho_u = model._to_state_sequence(log_feats)[:, -1]        # (U, C, C)
                            L_pos, n_pos = model.state_proj.lowrank(item_pos)
                            L_neg, n_neg = model.state_proj.lowrank(item_neg)
                            sp = model._hs_mixed(rho_u, L_pos[:, -1], n_pos[:, -1])
                            sn = model._hs_mixed(rho_u, L_neg[:, -1], n_neg[:, -1])
                        else:  # state：完全低秩
                            L_user, n_user = model.state_proj.lowrank(log_feats)
                            L_pos, n_pos = model.state_proj.lowrank(item_pos)
                            L_neg, n_neg = model.state_proj.lowrank(item_neg)
                            sp = model._hs_lowrank(L_user[:, -1], n_user[:, -1], L_pos[:, -1], n_pos[:, -1])
                            sn = model._hs_lowrank(L_user[:, -1], n_user[:, -1], L_neg[:, -1], n_neg[:, -1])
                    else:  # covariance / confidence：统一 power 打分（取最后一步）
                        power = model._scoring_power
                        if args.variant == 'dynamic':
                            C_seq = model._to_state_sequence(log_feats, normalize=False)
                            C_u = C_seq[:, -1]  # (U, C, C)
                            n_u = torch.diagonal(C_u, dim1=-2, dim2=-1).sum(dim=-1)  # Tr(C_T), (U,)
                            L_pos, n_pos = model.state_proj.lowrank(item_pos)
                            L_neg, n_neg = model.state_proj.lowrank(item_neg)
                            sp = model._power_score(model._fro_mixed(C_u, L_pos[:, -1]), n_u, n_pos[:, -1], power)
                            sn = model._power_score(model._fro_mixed(C_u, L_neg[:, -1]), n_u, n_neg[:, -1], power)
                        else:  # state
                            L_user, n_user = model.state_proj.lowrank(log_feats)
                            L_pos, n_pos = model.state_proj.lowrank(item_pos)
                            L_neg, n_neg = model.state_proj.lowrank(item_neg)
                            sp = model._power_score(model._fro_lowrank(L_user[:, -1], L_pos[:, -1]), n_user[:, -1], n_pos[:, -1], power)
                            sn = model._power_score(model._fro_lowrank(L_user[:, -1], L_neg[:, -1]), n_user[:, -1], n_neg[:, -1], power)
                else:  # dot matching：一阶方向 dot（取最后一步 item）
                    u_dir = model.state_proj.direction(last_h)           # (U, C)
                    sp = (u_dir * item_pos[:, -1]).sum(-1)
                    sn = (u_dir * item_neg[:, -1]).sum(-1)
            else:  # vector / density_feature / vector_evolve：dot 打分（原始）
                sp = (log_feats * model.item_emb(pos_t)).sum(-1)
                sn = (log_feats * model.item_emb(neg_t)).sum(-1)
            s_pos.append(sp.cpu().numpy())
            s_neg.append(sn.cpu().numpy())

        # ---- 训练一步 + 梯度范数 ----
        pos_logits, neg_logits = model(u, seq, pos, neg)
        optimizer.zero_grad()
        idx = np.where(pos != 0)
        if args.loss == 'bpr':
            loss = -torch.log(torch.sigmoid(pos_logits[idx] - neg_logits[idx]) + 1e-8).mean()
        else:
            loss = bce(pos_logits[idx], torch.ones_like(pos_logits[idx])) \
                 + bce(neg_logits[idx], torch.zeros_like(neg_logits[idx]))
        loss.backward()
        gnorm = sum(p.grad.norm().item() for p in model.attention_layers[-1].parameters()
                    if p.grad is not None)
        grad_norms.append(gnorm)
        optimizer.step()

    sampler.close()

    print('\n=== DIAGNOSE: dataset={} variant={} rank={} steps={} ==='.format(
        args.dataset, args.variant, args.state_rank, args.steps))
    print('-- Norm Analysis (向量模长 vs 密度因子范数) --')
    report('||h_last||  (vector)', np.concatenate(h_norms))
    if L_norms:
        report('||L_last||_F (density)', np.concatenate(L_norms))
    print('-- Score Distribution (原始打分, 未 logit) --')
    sp = np.concatenate(s_pos)
    sn = np.concatenate(s_neg)
    report('s_pos', sp)
    report('s_neg', sn)
    report('s_pos - s_neg (BPR 边际)', sp - sn)
    alls = np.concatenate([sp, sn])
    in01 = float((np.abs(alls) <= 0.2).mean())
    print('  score 落在 [-0.2, 0.2] 比例 = {:.2%}'.format(in01))
    print('-- Gradient Analysis (最后 attention 层参数梯度范数和) --')
    report('grad_norm', np.array(grad_norms))


if __name__ == '__main__':
    main()
