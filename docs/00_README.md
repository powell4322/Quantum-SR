# 文档地图（Docs）

> 编号前缀 = **建议查看顺序**；每份文档**各司其职**，尽量不重叠（实验设计集中在 05，实验结果集中在 02）。

## 文档清单（按顺序看）

| 文档 | 职责（一句话） | 什么时候看 |
|---|---|---|
| `00_README.md` | 索引 / 地图（本文件） | 进入 docs 先看 |
| `01_paper_progress.md` | **论文驾驶舱**：定位 / Motivation / Contribution / 投稿（WWW）/ GPT 审核记录 | 每阶段完成、给 GPT 审时 |
| `02_research_log.md` | **研究日志**：RQ 状态、实验**结果**（E001+）、已知问题、待验证清单 | 做实验、查进度 |
| `03_theory.md` | **理论**：密度状态合法性、偏好惯性演化、HS 打分、性质清单 | 写 Method / 理论论证 |
| `04_related_work.md` | **相关工作 / 防撞**：三方向对照 + 关键先例（🔴 DASFAA 2019 待精读） | 投稿前防撞、写 Related Works |
| `05_experiment_plan.md` | **实验计划**：数据集 / baseline / 实验设计矩阵 / 运行命令 / 铁律 | 规划、跑实验 |
| `06_usage_sasrec.md` | **代码使用说明**（怎么跑、参数） | 使用者 / 排查代码 |
| `09_research_positioning_v2.md` | **研究定位 v2（唯一 Source of Truth，DDST）** | 定位/写论文先看 |
| `09_theory_v1.md` | **理论定稿 v1**（P1-P5 命题 + 证明、合法性演化、熵） | 写 Method / 理论论证 |
| `10_server_run.md` | **服务器一键运行**（git + 装包 + 跑实验） | 上 GPU 服务器时 |

## 各司其职（避免重叠）

- 实验**设计 / 方案** → 只看 `05_experiment_plan.md`
- 实验**结果 / 日志** → 只看 `02_research_log.md`
- 论文**组织 / 贡献 / 审核** → 只看 `01_paper_progress.md`
- 理论**为什么成立** → `03_theory.md`
- 防撞 / 引用 → `04_related_work.md`
- 跑代码 → `06_usage_sasrec.md`

## ⚠️ 当前最重要的事（2026-08-06）

**研究定位唯一 source of truth = `09_research_positioning_v2.md`（DDST）。** 主线 = density operator state（M2 DS）+ legality-preserving evolution（M3 DDST）+ operator similarity（Tr）；**删除 Quantum / Entanglement / Fidelity 主线，不再用 QDM-Former 作为核心**；SASRec 为唯一 backbone；数据集 ml-1m / Beauty / Steam；指标统一 **R@10 / NDCG@10**。详见 v2 与 `05_experiment_plan.md` §0、`10_server_run.md`。

> 注意：`.github/skills/` 下另有给 Copilot 自动加载的 skill（领域知识、实验协议），非人读文档，无需进本目录。

