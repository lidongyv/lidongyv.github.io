# WAM 精读总索引（按用户五问分类）

> 更新：2026-11-12 ｜ **第一波五线 + 第二波四线全部完成**（68 篇次）；**第三波：RL post-training 专项精读中** ｜ 主线文档：[wam-survey.md](../wam-survey.md)（调研总览 + §5.4「论文 × 五轴」总表）｜ 阅读版：[wam-reading.html](../wam-reading.html)

按你的五个提问分五条精读线，每条线一个独立笔记文件，随讨论逐步追加 Q&A（各文件内 `<!-- QA_LOG -->` 区）。

| # | 你的提问 | 精读笔记 | 状态 |
|---|---|---|---|
| ① | 输入表征如何表达 | [01-input-representation.md](01-input-representation.md) | ✅ 完成（10 篇；含 Fast-WAM proprio 代码对质修正） |
| ② | 怎么实机和仿真验证 | [02-sim-real-validation.md](02-sim-real-validation.md) | ✅ 完成（10 篇；XPACE/V-JEPA Policy/Cosmos 协议数值已一手核实） |
| ③ | 动作如何表示 | [03-action-representation.md](03-action-representation.md) | ✅ 完成（10 篇；L3=GR-1/Action Images/PointWorld/Motus；4 篇动作维度原文未写明已标注） |
| ④ | 决策如何表示 | [04-decision-representation.md](04-decision-representation.md) | ✅ 完成（9 篇；7 篇 L3 方法节；Meta WAM 博客未能获取已留尝试记录） |
| ⑤ | 目标如何表示 | [05-goal-representation.md](05-goal-representation.md) | ✅ 完成（9 篇 L3 方法节 + 对比表；附截断/未取到清单） |

单篇精读（按范式）：

- [fast-wam.md](fast-wam.md) — Fast-WAM（已完成首版：Abstract→Method→全表→消融→待核实项）

第二波扩展（进行中，四线并行）：

| 线 | 文件 | 内容 | 状态 |
|---|---|---|---|
| 扩展 A | [01b-input-goal-wave2.md](01b-input-goal-wave2.md) | 输入/目标二梯队 10 篇 | ✅ 完成（10/10 全 L3；全文已本地化到 notes/html+txt 可离线引用） |
| 扩展 B | [03b-action-decision-wave2.md](03b-action-decision-wave2.md) | 动作/决策二梯队 10 篇 | ✅ 完成（10/10 + 汇总表 + 6 条趋势） |
| 扩展 C | [02b-validation-wave2.md](02b-validation-wave2.md) | 验证二梯队 10 篇 | ✅ 完成（10/10 + 汇总表 + 10 条评测坑） |
| 代码对质 | [06-code-forensics.md](06-code-forensics.md) | 7 项仓库取证 | ✅ 完成（6 篇动作空间实锤 + Fast-WAM 补充；仅 GR-1 归一化/Δt 未发布） |

第三波专项（按你的新问题追加）：

| 线 | 文件 | 内容 | 状态 |
|---|---|---|---|
| RL post-training | [07-rl-posttraining.md](07-rl-posttraining.md) | 6 篇 RL 后训练论文（全 L2；对比总表+6 条共性） | ✅ 完成 |
| RL 代码取证 | [08-rl-code-forensics.md](08-rl-code-forensics.md) | VAMPO 官方仓库 RL 全链路（采样器/奖励/GRPO 目标/超参，verbatim） | ✅ 完成 |

基础概念专项（System 0/1/2(/3)）：

| 文件 | 内容 | 状态 |
|---|---|---|
| [00-concepts-system0123.md](00-concepts-system0123.md) | 四概念谱系 + 一手引文（Taniguchi 2503.06138 / GR00T N1 / Helix 博客）+ 三个混淆澄清 + 论文清单 + 未能获取清单 | ✅ 完成（一手 L3；SII 2026 talk 与公众号正文未能获取） |
| [00-concepts-marr-vs-kahneman.md](00-concepts-marr-vs-kahneman.md) | Marr 三层分析 vs Kahneman 双过程理论：原文引文、对比表、S1/S2∈L2 与 S0≈L3 的映射、分析边界 | ✅ 完成（两侧原文经转引核实，映射为分析） |

模型系列精读：

| 文件 | 内容 | 状态 |
|---|---|---|
| [09-gr00t-series.md](09-gr00t-series.md) | GR00T N1（L3 论文全文）/ N1.5（L1 官方研究页）/ GR00T 2（L1 官方研究页）：架构、数据金字塔、训练配方、关键数值、三代对比、S2 外延变化 | ✅ 完成（N1 论文全文；N1.5/GR00T 2 whitepaper 正文未取） |
| [10-distillation.md](10-distillation.md) | WAM × 蒸馏：五条路线 + **最近三个月 15 篇新工作速查**（LoopedWAM/Phoenix/HINT-WAM/WAM-Policy/APM/ReLAP/CauSyncVLA/DreamFree/W2A/WAM-OPD/AnyStep/HapticWAM/DIDO/ForeTime-VLA/DreamMimic）+ **实现细节与伪代码**（CKT/S-VAM/THAW 为 L2–L3 原文级；近三月为 L1 摘要级重构，已标注） | ✅ 完成（近三月均为摘要级，引用前需回原文核对） |
| [11-distillation-pseudocode.md](11-distillation-pseudocode.md) | 工程伪代码版：每个方法给出 角色/输入(含形状)/输出/模块表/训练伪代码/推理伪代码/超参/坑；覆盖 CKT-WAM、S-VAM、THAW-VLA、DreamFree、W2A、APM、DIDO、AnyStep、WAM-OPD 家族、LoopedWAM、GR00T 数据管线 | ✅ 完成（L1 方法形状标 `?` 并注明需回原文） |
| [12-s2-s1-s0-stack.md](12-s2-s1-s0-stack.md) | 从 S2 到 S0 的实操链路：分层异步 + 潜动作接口；各层规格/频率/训练方式、G1+SONIC+GR00T 具体实现、多速率代码骨架、训练流水线、坑、与 WAM 的关系 | ✅ 完成 |

多页面站点：

- [site/index.html](../site/index.html) —— 概念（S0-S3 与五个高频问答）、Marr × Kahneman、GR00T 系列三页 + 链接既有 WAM 总览/四概念精读页；零依赖静态站。

## 使用约定

- 来源级别：L3=全文核实 / L2=部分正文 / L1=仅摘要；所有引文为论文原文（英文 verbatim + 小节号）。
- 「原文未写（已获取范围）」= 论文在已获取范围内确实没写；「未取到」= 工具截断导致没拿到；二者都不是结论。
- 图：沙箱禁止下载，以「图编号 + verbatim caption + 在线 URL」引用，不本地保存。
- 环境限制记录：shell 出网与 pip 均被沙箱拦截，本地 PDF/抠图管线不可用；Meta WAM 官方博客直连失败，其 CoT 潜轨迹搜索细节未获一手证实，暂不作依据。

## 已知全局待核实清单（跨线共享）

1. ~~Fast-WAM 动作空间维度~~ ✅ 已解决（代码）：LIBERO 7 维 Δxyz+Δrpy+gripper、RoboTwin 14 维双臂；proprio 亦已实锤（1 token 追加文本条件序列）——见 [fast-wam.md](fast-wam.md) §4.1b
2. V-JEPA 2-AC：~~潜动作学习目标~~ 已部分解决（③线 L2）：潜动作=7 维 ΔEE@4fps×15 步、block-causal 潜表征预测（§3.1）；**仍缺**：§4 image-goal 规划的动作采样机制与真机成功率数值（48 页长文截断）
3. XPACE 68.3%/86.7%：~~此前为二手新闻稿~~ 已升级为官方项目页一手（2026-10-26 ②线中途报告）：68.3%（香蕉拣放/倒水/叠碗，20 trials/任务，matched resets）与 recovery 61.7→86.7（香蕉/倒水/可乐，8% 合成恢复）为不同任务套件；论文 §4 实验节正文仍未取到（HTML 截断于 §3.4.1）
4. Meta WAM CoT 潜轨迹搜索（博客不可达）
5. Motus Sec 9.1（VAE latent 实现细节）、Act2Goal Sec V-A（VAE 型号）
6. **扩展 C 线失败前遗留的已验证数据（防丢，重发时作为起点）**：Rolling-WAM 项目页确认 G1 真机 **85.0%** 与延迟对照 215 vs 978/548/296/285ms；WLA RoboTwin 训练协议全文=2500 clean + 25k randomized demos、100k 步、100 trials/任务，Table 1 中 WLA-0 RoboTwin2.0 Clean **92.94%**（90.02 的归属待重发线确认）；LIBERO 98.6 / RMBench 56.5 待补。
7. 代码对质唯一未解决项：GR-1 动作归一化数值与视频预测间隔 Δt（官方仓未含数据处理代码）。
