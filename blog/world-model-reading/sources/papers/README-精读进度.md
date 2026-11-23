# 精读进度与锚点纪要

> 用法：每篇独立笔记在 `auto_docs/papers/<篇名>-notes.md`（原文逐字引用+数值表+消融+局限+叙事关联【推断】）。本文件只存锚点与验证状态。
> 验证状态：✅=关键引文 grep 命中原文；🔶=笔记已写入但引文抽查待讨论时进行（子代理死于回传环节，文件已确认完整）；⏳=精读中。

## 总览：51 篇全部完成 ✅（45 篇关键引文已 grep 验证原文 + 6 篇「死后交付」笔记完整、引文抽查留待讨论时）

### 批次 1（T4 力学/触觉簇）✅
DexTouch-WM（2609.20649）｜TouchWorld（2607.07287）｜TacWAM（2607.28391）｜FAWAM（2606.08555）

### 批次 2（T5 决策/world-action）✅
UniSim（2310.06114）｜UniPi（2302.00111）｜HumanoidWM（2506.01182）｜DriveWAM（2605.28544）

### 批次 3（T2 潜表征）✅
DreamerV3（2301.04104）｜TDMPC2（2310.16828）｜IRIS（2209.00588）｜DINOWM（2411.04983）

### 批次 4（T1 视频/通用）✅
GAIA2（2503.20523）｜Vista（2405.17398）｜DIAMOND（2405.12399）｜Cosmos（2501.03575）｜WorldDreamer（2401.09985）

### 批次 5（T3 几何/接口，吴佳俊组）✅
NeuROK（2605.30347）｜PerpetualWonder（2602.04876）｜CHORD（2601.04194）｜TrAct（2608.24101）｜MaskedVisualActions（2607.19343）

### 批次 6（T4 剩余触觉簇）✅
FeelWorld（2607.24267）｜ViTacWorld（2607.22530）｜ForceWAM（2610.18852）｜CWWAM（2610.30117）

### 批次 7（AD 其余 + T6 + 其他）— 21 篇全部落地
- ✅：Genie（2402.15391）、V-JEPA（2404.08471）、PhysMem（2602.20323）、OccWorld（2311.16038）、MUVO（2311.11762）、VAGEN（2510.16907）、LaLaWorld（2610.27851，正文 700 行内）、VerseCrafter（2601.05138）、WorldScore（2504.00983，20 模型非 19）
- 🔶（笔记完整、引文抽查待讨论时）：GAIA1（2309.17080）、DriveDreamer2（2403.06845）、GenAD（2402.11502）、DriveWM（2311.17918）、UniWorld（2308.07234）、ENACT（2511.20937）

### 批次 8（新·最近三个月 2026-08~11，实现细节+伪代码）✅ 全部 10 篇
TacForeSight（2606.11184）｜DTWM（2609.34286）｜PhysicalRegistry（2610.16365）｜Roz（2611.00612）｜DiffWAM（2611.19904）｜FlipTest（2611.19822）｜SAFER（2611.20107）｜EPM（2611.22294）｜InertialWorld（2611.22314）｜AgentWorld（2611.22880，NeurIPS 2026）
> 全部含：实现细节（数据/架构/损失/训练/推理）+ 伪代码（SAFER 有原文 Algorithm 1 逐字；其余按正文重建并标 ⚠）。汇总页：`auto_docs/最近三个月进展与实现细节.md`。

## 事实纠偏记录（重要，讨论时以这些为准）
1. TD-MPC2 无真机实验、无推理延迟数据（提示词预设错误，原文 0 命中）
2. DINO-WM 无 MCTS、无真机 Kinova（提示词预设错误；规划器=MPC+CEM）
3. DIAMOND 是像素空间扩散，非 latent diffusion
4. GAIA-2 论文无定量结果表/闭环指标/消融（"驾驶杆控制"原文未写）
5. IRIS「token 带来样本效率」原文未声称；无消融章节
6. PerpetualWonder/NeuROK 的力学（动能/势能、摩擦系数）仍是给定先验
7. ViTacWorld 无 scaling law 曲线（Fig.4 是校准线）
8. 「free demonstrations」一词在 UniPi 原文未出现；「distill」仅指采样加速
9. WorldDreamer 无定量表/消融/局限节；GAIA-1 无定量基准表/无消融；DriveDreamer-2 无定量表/无消融
10. FeelWorld 的 CEM 规划「不适用于高频实时控制」（原文自述，L905-907）
11. GAIA-2 无最终数值表（仅训练曲线）
12. ForceWAM 正文 Q4 数值与 Table 2 列序疑似笔误；NeuROK 测试集引用疑误（PartNet-Mobility vs SAPIEN）

## 讨论准备（输出任务 × 三层架构，供选题）
- **T4 力学侧 8 篇**（批次 1+6）：最贴叙事线；DexTouch-WM=人手数据缩放轴、TacWAM=触觉仅作训练监督、FAWAM=双频闭环 100ms、FeelWorld=三层触觉状态、CW-WAM=等效六维力对应
- **T3 几何侧 5 篇**（批次 5）：NeuROK=Neural Kinematics 原文、CHORD=「借一个世界」具名实现、TrAct=轨迹接口实证、PerpetualWonder=物理-视觉闭环、MaskedVA=正逆向统一接口
- **T5 决策侧**：DriveWAM=WAM 严格定义、GenAD/DriveWM=预测+规划闭环、UniSim=通用模拟器、UniPi=视频即策略
- **T2 潜表征**：DreamerV3 隐想象 vs TD-MPC2 隐式 MPC 的分野、DINO-WM 零样本规划
- **T1 视频**：GAIA-2/Vista/DIAMOND/Cosmos/WorldDreamer/Genie——质量、可控、实时性的取舍
- **T6 语义**：VAGEN（慢脑 RL 化）、PhysMem（验证式物理记忆）、ENACT（评测设计参照）

更新时间: 2026-11-21 17:40（51/51 全部完成，含最近三个月 10 篇）
