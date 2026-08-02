# 文档地图（Docs）

> 本项目研究/论文文档的索引。**给 GPT 审核 / 推进论文**主要用前三个，`README` 帮你快速定位"哪个文档回答什么问题"。

## 一、核心研究文档（给 GPT 审核、推进论文用）

| 文档 | 用途 | 什么时候看 |
|---|---|---|
| `PAPER_PROGRESS.md` | **论文驾驶舱 + GPT 审核入口**：钉死 contribution / motivation / 实验路线 / 投稿定位（WWW），内置"给 GPT 的提交模板"和审核反馈记录表 | **每阶段完成后、准备给 GPT 审时** |
| `RESEARCH_LOG.md` | **研究追踪**：RQ 状态、实验设计、待验证问题、实验结果日志（E001+）、已知问题（如 Tr/BCE 不匹配） | 做实验、记录结果、查"在验证什么"时 |
| `THEORY.md` | **理论支撑**：密度矩阵合法性、偏好惯性演化、Hilbert–Schmidt 打分、自由度分析 | 写 Method / 理论部分、核对"为什么成立"时 |
| `RELATED_WORK.md` | **相关工作总结 + 防撞对照表**（quantum rec / uncertainty rec / distributional rec） | 投稿前防撞、写 Related Works 时 |
| `EXPERIMENT_PLAN.md` | **实验矩阵**（数据集、baseline、主/消融/分析） | 规划/跑实验、保证公平对比时 |

## 二、代码文档

| 文档 | 用途 | 什么时候看 |
|---|---|---|
| `sasrec.md` | 代码使用说明（怎么跑、参数、与理论代码对照） | 使用者 / 新成员 / 排查代码时 |

## 三、使用建议（快速定位）

- **要给 GPT review 论文** → 用 `PAPER_PROGRESS.md`（§5.1 模板提交，§5.2 回填反馈）
- **看研究进度 / 待验证什么** → `RESEARCH_LOG.md`
- **看理论是否成立 / 怎么写方法** → `THEORY.md`
- **怎么跑代码** → `sasrec.md`

## 四、分工一句话

- `RESEARCH_LOG` = 我们**做了什么、在验证什么、结果如何**（过程 + 证据库）
- `THEORY` = **为什么成立**（理论依据）
- `PAPER_PROGRESS` = **论文怎么组织 + 给 GPT 审**（驾驶舱）

> 注意：`.github/skills/` 下还有两份给 Copilot 自动加载的 skill（领域知识、实验协议），不是给人读的文档，无需进本目录。
