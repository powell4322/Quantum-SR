# SASRec 序列推荐模型使用说明与模型结构解析

## 1. 项目简介

这个仓库实现的是一个基于自注意力机制的序列推荐模型 SASRec（Self-Attentive Sequential Recommendation）。
它的核心思想是：用用户历史交互序列建模用户兴趣，并通过当前位置的上下文去预测下一个可能喜欢的商品或项目。

本项目包含三个核心文件：

- [main.py](main.py)：训练/评估/推理的主入口。
- [model.py](model.py)：SASRec 模型定义。
- [utils.py](utils.py)：数据划分、采样、评估函数等工具函数。

---

## 2. 代码运行方式

### 2.1 训练

在项目根目录执行下面的命令即可开始训练：

```bash
python main.py --dataset Beauty --train_dir run1 --batch_size 128 --lr 0.001 --maxlen 200 --hidden_units 50 --num_blocks 2 --num_heads 1 --dropout_rate 0.2 --num_epochs 1000 --device cuda
```

### 2.2 仅做推理

如果已经有训练好的权重，可以直接进行推理评估：

```bash
python main.py --dataset Beauty --train_dir run1 --inference_only True --state_dict_path path/to/model.pth --device cuda
```

### 2.3 常用参数说明

- `--dataset`：数据集名称，例如 `Beauty`、`ml-1m`、`Steam`、`Video`、`wikipedia`
- `--train_dir`：训练输出目录名，结果会保存在 `dataset_train_dir/` 文件夹中
- `--batch_size`：训练批次大小
- `--lr`：学习率
- `--maxlen`：最大序列长度
- `--hidden_units`：隐藏维度
- `--num_blocks`：SASRec 中堆叠的 Transformer block 数量
- `--num_heads`：多头注意力头数
- `--dropout_rate`：dropout 概率
- `--num_epochs`：训练轮数
- `--device`：训练设备，通常为 `cuda` 或 `cpu`
- `--inference_only`：是否只进行推理
- `--state_dict_path`：加载已有模型权重路径

---

## 3. 代码流程说明

### 3.1 训练流程

训练主流程在 [main.py](main.py) 中：

1. 先调用 `build_index()` 和 `data_partition()` 读取数据。
2. 使用 `WarpSampler` 生成训练批次。
3. 实例化 `SASRec` 模型。
4. 训练时，模型输入为：
   - 用户历史序列 `log_seqs`
   - 正样本 `pos_seqs`
   - 负样本 `neg_seqs`
5. 通过 `BCEWithLogitsLoss` 计算损失并更新参数。
6. 每隔一定轮数做验证和测试评估。

### 3.2 评估流程

评估逻辑主要在 [utils.py](utils.py) 中：

- `evaluate_valid()`：用于验证集评估
- `evaluate()`：用于测试集评估

它们的思路是：

- 取用户历史行为序列作为输入上下文
- 将真实目标物品作为正样本加入候选集合
- 生成若干负样本
- 让模型对所有候选项打分
- 通过排序位置计算 NDCG 和 HR 指标

---

## 4. 实际模型结构讲解

### 4.1 输入表示

SASRec 模型的输入是用户的历史行为序列。比如一个用户过去点击过的物品序列：

```text
[3, 8, 12, 19, 26]
```

模型会把序列中的每个物品映射为一个 embedding 向量，然后加上位置编码，形成可输入 Transformer 的序列表示。

在 [model.py](model.py) 中：

- `self.item_emb`：物品 embedding 层
- `self.pos_emb`：位置 embedding 层
- `self.emb_dropout`：dropout 层

也就是说，模型最终得到的输入表示是：

```text
item_embedding + positional_embedding
```

---

### 4.2 自注意力层（Self-Attention）

SASRec 的核心是多头自注意力机制。
它的作用是让模型在当前时刻看见前面所有已经发生过的物品，从而学习用户兴趣的演化过程。

在 [model.py](model.py) 中，代码通过：

```python
self.attention_layers.append(torch.nn.MultiheadAttention(...))
```

来定义多头注意力层。

因为是序列推荐，模型必须避免“看到未来信息”，因此它使用了下三角 mask：

```python
attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
```

这相当于设置了因果掩码，确保当前时间步只能关注自己和之前的行为。

---

### 4.3 残差连接与归一化

每一个 Transformer block 中，模型包含：

1. 自注意力模块
2. 残差连接
3. LayerNorm
4. 前馈网络（FeedForward）
5. 残差连接
6. LayerNorm

在代码中，结构是这样组织的：

- attention 层：`self.attention_layers`
- attention LayerNorm：`self.attention_layernorms`
- forward 层：`self.forward_layers`
- forward LayerNorm：`self.forward_layernorms`

这使得模型训练更稳定，也有助于深层网络的学习。

---

### 4.4 前馈网络（PointWiseFeedForward）

在 [model.py](model.py) 中，`PointWiseFeedForward` 是一个轻量的前馈模块，主要由两个 1x1 卷积层、ReLU 和 Dropout 组成。

它的作用是对注意力层输出做非线性变换，使模型拥有更强的表达能力。

---

### 4.5 输出层

模型最终会得到每个时间步的隐藏表示 `log_feats`。在训练阶段：

```python
pos_logits = (log_feats * pos_embs).sum(dim=-1)
neg_logits = (log_feats * neg_embs).sum(dim=-1)
```

也就是用最后一层表示去和目标物品 embedding 做点积，得到正样本和负样本的打分。

在推理阶段：

```python
final_feat = log_feats[:, -1, :]
logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
```

这表示只取序列最后一个时间位置的表示，用它去对候选物品打分，得到推荐排序。

---

## 5. 训练目标

模型使用二分类损失函数：

```python
bce_criterion = torch.nn.BCEWithLogitsLoss()
```

它的目标是：

- 提升正样本的得分
- 降低负样本的得分

这和推荐系统中常见的“排序学习”思路是一致的。

---

## 6. 关键代码对应关系

| 作用 | 代码位置 |
|---|---|
| 训练入口 | [main.py](main.py) |
| SASRec 模型定义 | [model.py](model.py) |
| 数据采样与评估 | [utils.py](utils.py) |
| 物品 embedding | `self.item_emb` |
| 位置编码 | `self.pos_emb` |
| 多头自注意力 | `self.attention_layers` |
| 前馈网络 | `PointWiseFeedForward` |
| 训练损失 | `BCEWithLogitsLoss` |

---

## 7. 适合初学者的理解方式

可以把 SASRec 理解为：

- 输入：用户过去的点击/购买序列
- 处理：用 Transformer 学习序列中的依赖关系
- 输出：预测用户下一步最可能喜欢的物品

它和传统的基于矩阵分解的方法相比，更擅长建模用户兴趣的时序演化过程。

---

## 8. 进一步建议

如果你想继续深入，可以重点关注以下几个点：

1. 观察不同 `num_blocks` 和 `num_heads` 对性能的影响
2. 调整 `maxlen`，看看序列长度是否影响效果
3. 对比 `cuda` 和 `cpu` 的训练速度
4. 结合实际推荐指标（HR、NDCG）分析结果

---

## 9. 一句话总结

SASRec 的本质是：用自注意力机制建模用户历史行为序列中的长期依赖关系，然后通过最后一个时间步的表示来预测下一个可能感兴趣的物品。
