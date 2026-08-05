import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class StateProjection(torch.nn.Module):
    """将向量 h 投影为密度矩阵 rho（半正定、trace=1）。

    低秩 Cholesky-like 构造: rho = L L^T / Tr(L L^T)，L 由线性投影产生。
    rank=1 时为纯态（等价于一个方向向量）；rank>1 时为混合态（可表达多兴趣）。
    """

    def __init__(self, hidden_units, rank=1):
        super(StateProjection, self).__init__()
        self.hidden_units = hidden_units
        self.rank = rank
        self.proj = torch.nn.Linear(hidden_units, hidden_units * rank, bias=False)

    def forward(self, inputs):
        # inputs: (..., C) -> L: (..., r, C)
        L = self.proj(inputs).view(*inputs.shape[:-1], self.rank, self.hidden_units)
        rho = torch.matmul(L.transpose(-1, -2), L)  # (..., C, C), PSD
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)  # (...)
        trace = trace.unsqueeze(-1).unsqueeze(-1)  # (..., 1, 1)
        rho = rho / trace.clamp_min(1e-8)  # normalize to trace=1
        return rho


class DensityFeatureInput(torch.nn.Module):
    """DMPEN 式：把 item embedding 提升为低秩密度特征（flatten L）再映射回 d 维，
    作为编码器输入（density-as-feature，非状态）。"""

    def __init__(self, hidden_units, rank=1):
        super(DensityFeatureInput, self).__init__()
        self.rank = rank
        self.proj = torch.nn.Linear(hidden_units, hidden_units * rank, bias=False)
        self.back = torch.nn.Linear(hidden_units * rank, hidden_units)

    def forward(self, emb):
        # emb: (..., d) -> L: (..., r, d) -> flatten (..., r*d) -> back to d
        L = self.proj(emb).view(*emb.shape[:-1], self.rank, emb.shape[-1])
        feat = L.reshape(*emb.shape[:-1], -1)
        return self.back(feat)


class StateTransition(torch.nn.Module):
    """动态状态演化: rho_t = alpha * rho_{t-1} + (1-alpha) * rho_{i_t}。

    凸组合保持 PSD 且 trace=1（对应量子混合态语义），alpha 为标量，参数开销为零（或仅一个标量）。
    """

    def __init__(self, hidden_units, learnable=False, init_alpha=0.9):
        super(StateTransition, self).__init__()
        self.learnable = learnable
        init_logit = np.log(init_alpha / (1.0 - init_alpha))
        if learnable:
            self.logit = torch.nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.register_buffer('logit', torch.tensor(init_logit, dtype=torch.float32))

    def forward(self, rho_prev, rho_cur):
        alpha = torch.sigmoid(self.logit)
        return alpha * rho_prev + (1.0 - alpha) * rho_cur


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

        # === variants: vector / state / dynamic / vector_evolve (VE) / density_feature (DF) ===
        self.variant = getattr(args, 'variant', 'vector')
        self.state_rank = getattr(args, 'state_rank', 1)
        self.transition_mode = getattr(args, 'transition', 'fixed')
        self.transition_alpha = getattr(args, 'transition_alpha', 0.9)

        self.state_proj = None
        self.state_transition = None
        self.df_input = None
        if self.variant in ('state', 'dynamic'):
            self.state_proj = StateProjection(args.hidden_units, rank=self.state_rank)
            if self.variant == 'dynamic':
                self.state_transition = StateTransition(
                    args.hidden_units,
                    learnable=(self.transition_mode == 'learnable'),
                    init_alpha=self.transition_alpha,
                )
        if self.variant == 'density_feature':
            # DMPEN-style: lift item embeddings to density features before the encoder
            self.df_input = DensityFeatureInput(args.hidden_units, rank=self.state_rank)

    def _to_state_sequence(self, log_feats):
        """把 (U, T, C) 的向量序列转成 (U, T, C, C) 的密度矩阵序列；
        dynamic 变体再沿时间维做凸组合演化（ρ_0 = I/d，见 docs/09_theory_v1.md）。"""
        rho_seq = self.state_proj(log_feats)  # (U, T, C, C)
        if self.variant == 'dynamic':
            T = rho_seq.shape[1]
            C = rho_seq.shape[-1]
            outputs = []
            # ρ_0 = I/d（最大混合态先验）
            prev = (torch.eye(C, device=rho_seq.device) / C).unsqueeze(0).expand(rho_seq.shape[0], -1, -1)
            for t in range(T):
                cur = rho_seq[:, t]  # (U, C, C)
                cur = self.state_transition(prev, cur)
                outputs.append(cur)
                prev = cur
            rho_seq = torch.stack(outputs, dim=1)
        return rho_seq

    def _vector_evolve(self, log_feats):
        """VE：对编码器输出 (U,T,C) 沿时间做 EMA（向量凸组合）。
        用于回应"凸组合只是 EMA"——与 DDS 的密度状态演化对比（fixed alpha）。"""
        T = log_feats.shape[1]
        alpha = float(self.transition_alpha)
        outs = [log_feats[:, 0]]
        prev = log_feats[:, 0]
        for t in range(1, T):
            cur = alpha * prev + (1.0 - alpha) * log_feats[:, t]
            outs.append(cur)
            prev = cur
        return torch.stack(outs, dim=1)

    @staticmethod
    def _logit_score(s, eps=1e-7):
        """把 [0,1] 的 Tr 相似度分数映射到无界 logits（logit 变换），
        兼容 BCEWithLogitsLoss 与 BPR（Tr 是密度矩阵 HS 内积 ∈[0,1]）。
        2026-08-03：修正 Tr/BCE 不匹配问题（见 docs/02_research_log.md §4.4）。"""
        s = torch.clamp(s, eps, 1.0 - eps)
        return torch.log(s / (1.0 - s))

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        if self.variant == 'density_feature':
            # density-as-feature 输入（模拟 DMPEN：把 item 提升为密度特征再进编码器）
            seqs = self.df_input(seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        if self.variant in ('state', 'dynamic'):
            rho_seq = self._to_state_sequence(log_feats)  # (U, T, C, C)
            rho_pos = self.state_proj(self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)))  # (U, T, C, C)
            rho_neg = self.state_proj(self.item_emb(torch.LongTensor(neg_seqs).to(self.dev)))
            pos_logits = self._logit_score((rho_seq * rho_pos).sum(dim=(-1, -2)))  # Tr → logits
            neg_logits = self._logit_score((rho_seq * rho_neg).sum(dim=(-1, -2)))
        elif self.variant == 'vector_evolve':
            evolved = self._vector_evolve(log_feats)  # (U, T, C) EMA
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
            pos_logits = (evolved * pos_embs).sum(dim=-1)
            neg_logits = (evolved * neg_embs).sum(dim=-1)
        else:  # vector / density_feature（dot 打分）
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        if self.variant in ('state', 'dynamic'):
            rho_seq = self._to_state_sequence(log_feats)  # (U, T, C, C)
            rho_user = rho_seq[:, -1, :, :].unsqueeze(1)  # (U, 1, C, C) only use last state
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
            rho_item = self.state_proj(item_embs)  # (U, I, C, C)
            logits = self._logit_score((rho_user * rho_item).sum(dim=(-1, -2)))  # Tr → logits
        elif self.variant == 'vector_evolve':
            evolved = self._vector_evolve(log_feats)  # (U, T, C) EMA
            final_feat = evolved[:, -1, :]
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        else:  # vector / density_feature（dot 打分）
            final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
