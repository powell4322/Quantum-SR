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
| `10_server_run.md` | **服务器一键运行**（git + 装包 + 跑实验 + 实验顺序） | 上 GPU 服务器时 |
| `agent/research_plan.md` | **Agent 执行计划**：Phase 0 诊断 + Phase 1 rank / Phase 2 matching ablation（冻结模型） | 下一阶段实验前看 |

## 各司其职（避免重叠）

- 实验**设计 / 方案** → 只看 `05_experiment_plan.md`
- 实验**结果 / 日志** → 只看 `02_research_log.md`
- 论文**组织 / 贡献 / 审核** → 只看 `01_paper_progress.md`
- 理论**为什么成立** → `03_theory.md`
- 防撞 / 引用 → `04_related_work.md`
- 跑代码 → `06_usage_sasrec.md`

## ⚠️ 当前最重要的事（2026-08-07）

**定位 source of truth = `09_research_positioning_v2.md`（DDST）**；SASRec 唯一 backbone，指标 R@10 / NDCG@10。

**当前阶段（第一轮 ml-1m 结果已出，见 `02_research_log.md` §6/§6.1）**：V(0.5852) > DF(0.5745) > VE(0.5604) > DDS(0.4950) > DS(0.4622)。**诊断结论**：失败主因 = Tr 打分被压缩到 [0,0.05]（归一化相似度高维集中），非理论/非梯度。
**下一步（冻结模型）**：按 `agent/research_plan.md` 跑 **Phase 1 rank ablation**（state/dynamic rank=4/8/16，命令见 `10_server_run.md` Step 1.5）→ Phase 2 matching ablation → 视结果决定 Phase 3 confidence-aware scoring。

> 注意：`.github/skills/` 下另有给 Copilot 自动加载的 skill（领域知识、实验协议），非人读文档，无需进本目录。

