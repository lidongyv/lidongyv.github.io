# 世界模型分类（二）：按「输出任务 × 能力」的功能视角

> 整理日期：2026-10-26 ｜ 本稿是 [world-model-survey.md](world-model-survey.md)（实现方式/机制视角）的姊妹篇。
> 核实纪律同前稿：ID 均经 arXiv 检索结果核实；未核实项显式标注。

## 层级声明（Marr Step 0）

- 前稿 = **L2/L3 视角**（表征与算法：token 化、DiT、RSSM、3DGS……）。
- 本稿 = **L1 视角**（计算理论）：世界模型**算的是什么输出**、**服务的下游效用是什么**、**正确性判据与误差的非对称性是什么**。
- 先对齐层级再讨论：谈「某个工作用哪种机制」请看前稿；谈「世界模型该输出什么、具备什么能力、为什么」看本稿。

---

## TL;DR

1. 按输出本体，世界模型可分 6 类输出任务：**T1 像素/视频、T2 潜表征、T3 3D/4D 结构、T4 物理量（力/触觉/接触）、T5 决策（动作/规划）、T6 语义（语言/事件）**。多数工作主输出一类、附带另一类（如 Dreamer = T2+T5）。
2. 关键修正：前稿「力学输出近空白」的判断需要更新——**2026 年下半年已形成一个「触觉/力世界模型」子领域**（TouchWorld、FeelWorld、ViTacWorld、TacWAM、FAWAM、ForceWAM、CW-WAM、DexTouch-WM 等 8+ 篇），且其中 4 篇命名直接采用 **WAM（World Action Model）**——正是吴佳俊演讲中「world action models」一词，已从口播概念变成命名共识。
3. 误差非对称性（L1 追问的答案）：快脑层的视觉预测误差是「浪费」，脊髓层的力/接触预测误差是「灾难」——现有评测（FID/FVD/成功率）只度量前者，后者尚无毫秒级在线判据。
4. 能力现状：**视觉保真、可控性、下游数据复用已成熟；物理一致性、实时交互在发展；泛化（跨本体 scaling）刚起步；毫秒级接触安全是空白**。

---

## 1. 输出任务分类（世界模型「算的是什么」）

> 每类给出：输出本体定义、正确性判据、代表工作、误差非对称性。

### T1 像素/视频输出（visual foresight）
- **算的是**：给定过去观测 + 条件（动作/语言/轨迹），生成未来的 RGB 图像序列。
- **判据**：生成质量（FID/FVD）+ 与真实演化的对齐（保真 × 预测正确）。
- **代表**：Sora、Cosmos（[2501.03575](https://arxiv.org/abs/2501.03575)）、GAIA-1（[2309.17080](https://arxiv.org/abs/2309.17080)）/GAIA-2（[2503.20523](https://arxiv.org/abs/2503.20523)）、Vista（[2405.17398](https://arxiv.org/abs/2405.17398)）、DIAMOND（[2405.12399](https://arxiv.org/abs/2405.12399)）、DriveDreamer-2（[2403.06845](https://arxiv.org/abs/2403.06845)）、Genie 系、VerseCrafter（[2601.05138](https://arxiv.org/abs/2601.05138)）。
- **误差非对称性**：预测错误 = 浪费（下游幻觉、仿真漂移），不直接造成物理伤害。

### T2 潜表征输出（latent prediction / JEPA）
- **算的是**：未来状态的潜表征（不重建像素），供规划/控制/下游任务消费。
- **判据**：下游任务收益（规划成功率、样本效率），而非视觉还原。
- **代表**：DreamerV3（[2301.04104](https://arxiv.org/abs/2301.04104)）、TD-MPC2（[2310.16828](https://arxiv.org/abs/2310.16828)）、IRIS（[2209.00588](https://arxiv.org/abs/2209.00588)）、V-JEPA（[2404.08471](https://arxiv.org/abs/2404.08471)）、DINO-WM（[2411.04983](https://arxiv.org/abs/2411.04983)）、LaLa-World（[2610.27851](https://arxiv.org/abs/2610.27851)）、GLAM（[2609.14561](https://arxiv.org/abs/2609.14561)）。
- **误差非对称性**：表征误差通过决策放大，但损失函数可端到端约束。

### T3 3D/4D 结构输出（几何/形变/轨迹）
- **算的是**：场景在 3D/4D 中的结构化状态——occupancy、3DGS、物体运动学参数、形变、自车/物体轨迹。
- **判据**：几何一致性、物理合理性、可控性（能否按要求演化）。
- **代表**：OccWorld（[2311.16038](https://arxiv.org/abs/2311.16038)）、UniWorld（[2308.07234](https://arxiv.org/abs/2308.07234)）、MUVO（[2311.11762](https://arxiv.org/abs/2311.11762)）、ManiGaussian++（[2506.19842](https://arxiv.org/abs/2506.19842)）、NeuROK（[2605.30347](https://arxiv.org/abs/2605.30347)）、CHORD（[2601.04194](https://arxiv.org/abs/2601.04194)）、PerpetualWonder（[2602.04876](https://arxiv.org/abs/2602.04876)）、World on Rails（NeurIPS 2024，[GitHub](https://github.com/dotchen/WorldOnRails)）、TrAct 的视觉轨迹接口（[2608.24101](https://arxiv.org/abs/2608.24101)）。
- **误差非对称性**：几何量误差可被下游重规划吸收；但**几何量根本不含力学量**（点云再密也补不上力/摩擦）——这是叙事线的核心论据。

### T4 物理量输出（力 / 触觉 / 接触）— 2026 新兴子领域
- **算的是**：接触力、力矩（contact wrench）、触觉信号、摩擦/滑移/稳定性等力学量，以及它们在动作作用下的演化。
- **判据**：接触预测精度 + 闭环力控/接触富集任务成功率。
- **代表**（均 2026，ID 经检索核实）：
  - DexTouch-WM（[2609.20649](https://arxiv.org/abs/2609.20649)，丁文伯组）：人手→机械手触觉动力学迁移，动作条件世界模型；
  - TouchWorld（[2607.07287](https://arxiv.org/abs/2607.07287)）：预测+反应的触觉基础模型（"Predictive and Reactive"）；
  - FeelWorld（[2607.24267](https://arxiv.org/abs/2607.24267)）：视触觉世界模型，层次化接触预测与规划；
  - ViTacWorld（[2607.22530](https://arxiv.org/abs/2607.22530)）：规模化视触觉世界模型；
  - TacWAM（[2607.28391](https://arxiv.org/abs/2607.28391)）：锚引导、力学感知触觉预测的世界动作模型；
  - FAWAM（[2606.08555](https://arxiv.org/abs/2606.08555)）：力感知世界动作模型，闭环接触富集操控；
  - ForceWAM（[2610.18852](https://arxiv.org/abs/2610.18852)）：用 demo+play 数据解耦「未来预测与响应学习」；
  - CW-WAM（[2610.30117](https://arxiv.org/abs/2610.30117)）：结构化接触力矩世界动作模型。
  - 慢速近邻：PhysMem（[2602.20323](https://arxiv.org/abs/2602.20323)，VLM 测试期摩擦/稳定性推理，非毫秒级）。
- **误差非对称性**：**预测错误 = 灾难**（滑落、捏碎、碰撞）；判据必须在线、毫秒级——这是它与 T1-T3 的本质区别。

### T5 决策输出（动作 / 规划；world-action 联合建模）
- **算的是**：与未来预测**联合**输出动作或规划——"预测是为了行动"，世界模型与策略同体。
- **判据**：任务成功率、闭环收益（而非生成质量）。
- **代表**：DreamerV3 / TD-MPC2（想象中优化策略/MPC）、GenAD（[2402.11502](https://arxiv.org/abs/2402.11502)）、Drive-WM（[2311.17918](https://arxiv.org/abs/2311.17918)）、UniPi（[2302.00111](https://arxiv.org/abs/2302.00111)，视频即策略）、TrAct（[2608.24101](https://arxiv.org/abs/2608.24101)）、DriveWAM（[2605.28544](https://arxiv.org/abs/2605.28544)）、以及触觉簇中的 TacWAM/FAWAM/ForceWAM/CW-WAM。
- **误差非对称性**：动作误差直接外溢为物理后果；WAM 化（联合建模）正是为压缩「预测→决策」的误差链。

### T6 语义输出（语言 / 事件 / 场景变化）
- **算的是**：把世界建模当作推理任务——状态描述、事件时序、场景图变化，供 VLM 慢脑决策。
- **判据**：问答/推理正确率（如 ENACT 的序列重排正确率）。
- **代表**：VAGEN（[2510.16907](https://arxiv.org/abs/2510.16907)，World Modeling Reward 训练的状态估计+转移建模）、ENACT（[2511.20937](https://arxiv.org/abs/2511.20937)，正/逆世界建模评测）、PhysMem（语言假说的验证式物理记忆）。
- **误差非对称性**：语义错误→错误规划→可通过验证器事后纠偏（「审慎的、事后的」），无法用于毫秒级反应。

---

## 2. 能力维度与现状矩阵

> 能力 = L1 的「正确性判据 + 天然约束」拆解。评级：● 成熟 ／ ○ 发展中 ／ ◌ 新兴或空白。

| 能力 | 现状 | 代表工作 | 主要瓶颈 |
|------|------|----------|----------|
| C1 视觉保真与多样性 | ● | Sora、Cosmos、Vista、GAIA-2 | 细节已好，语义/物理可信度不足 |
| C2 预测正确性（对真实演化） | ●（短时）○（长时） | GAIA-1、DriveDreamer-2 | 长时程漂移；评测仍偏 FVD |
| C3 物理与几何一致性 | ○ | PerpetualWonder、NeuROK、MUVO、VerseCrafter | 接触、穿透、守恒仍靠隐式学习 |
| C4 可控性（动作/语言/轨迹/几何接口） | ● | TrAct、Masked Visual Actions（[2607.19343](https://arxiv.org/abs/2607.19343)）、Ctrl-VI（[2510.07670](https://arxiv.org/abs/2510.07670)）、VerseCrafter | 接口与视觉空间对齐是关键（2025-26 共识） |
| C5 交互实时性与长时程记忆 | ○ | Genie 3、Matrix-Game 4.0（[2610.27999](https://arxiv.org/abs/2610.27999)，60FPS）、PerpetualWonder | 实时多靠混合架构（引擎管状态）；触觉毫秒级尚无 |
| C6 泛化（跨域/跨本体/零样本/scaling） | ◌→○ | UniSim（[2310.06114](https://arxiv.org/abs/2310.06114)）、DINO-WM、Cosmos、ViTacWorld（scaling 视触觉）、DexTouch-WM（人→机迁移） | 跨本体数据轴刚被证明可行，规模小 |
| C7 下游可复用（评估器/模拟器/合成数据源/蒸馏源） | ● | UniSim、DIAMOND（可玩引擎）、DexTouch-WM（代理环境+合成轨迹）、WorldScore（[2504.00983](https://arxiv.org/abs/2504.00983)） | 「用世界模型评测世界模型」的元评测缺失 |
| C8 接触安全与约束满足（毫秒级在线） | ◌ | TouchWorld（predictive+reactive）、FAWAM（闭环力控） | 均离线训练+推理，**在线毫秒级验证判据是空白** |

---

## 3. 代表工作 × 输出任务 × 能力 总表

> 主输出标 ●，次输出标 ○。仅收录已核实 ID 的工作（前稿清单 + 本稿触觉簇）。

| 工作 | T1 视频 | T2 潜表征 | T3 3D/4D | T4 物理量 | T5 决策 | T6 语义 | 突出能力 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|----------|
| DreamerV3 | ○ | ● | | | ● | | C2/C7，样本效率 |
| TD-MPC2 | | ● | | | ● | | C7，跨域单配置 |
| IRIS | ○ | ● | | | ○ | | C2（离散潜） |
| V-JEPA / DINO-WM | | ● | | | ○ | | C6 零样本规划 |
| LaLa-World | | ● | | | ○ | | C6/C7，低成本规划 |
| Sora / Cosmos | ● | | | | | | C1/C4/C7（Cosmos 开源） |
| Genie / Genie 2 / Genie 3 | ● | | | | | | C4/C5 交互 |
| DIAMOND | ● | | | | | | C1+C5（可玩引擎） |
| GAIA-1 / GAIA-2 | ● | | | | | | C2/C4（驾驶场景） |
| Vista / DriveDreamer-2 | ● | | | | ○ | | C1/C4 |
| GenAD / Drive-WM | ● | | | | ● | | C7（规划闭环） |
| DriveWAM | ● | | | | ● | | C6（WAM 化） |
| UniSim | ● | | | | | | C6/C7（通用模拟器） |
| OccWorld / UniWorld | | | ● | | | | C3 预训练/演化 |
| MUVO / VerseCrafter | ● | | ● | | | | C3/C4 几何控制 |
| ManiGaussian++ | | | ● | | ○ | | C3（3DGS 操控） |
| NeuROK / CHORD / PerpetualWonder | ○ | | ● | | | | C3 物理闭环（吴佳俊线） |
| World on Rails | ● | | ○ | | | | C3（自车轨分解） |
| UniPi | ● | | | | ● | | C6（视频即策略） |
| TrAct / Masked Visual Actions | ● | | ○ | | ● | | C4（视觉对齐接口） |
| VAGEN / ENACT | | | | | | ● | C6（慢脑世界建模） |
| PhysMem | | | | ○ | | ● | C3 慢速物理推理（76% vs 23%） |
| **DexTouch-WM** | ● | | | ● | ○ | | C6 人→机触觉迁移；C7 代理环境 |
| **TouchWorld / FeelWorld / ViTacWorld** | ○ | | | ● | | | C6 scaling；C8 预测+反应 |
| **TacWAM / FAWAM / ForceWAM / CW-WAM** | | | | ● | ● | | C8 闭环力控（WAM 化触觉） |

---

## 4. 对叙事线三层架构的覆盖图

| 层 | 需要什么输出/能力（L1 目标） | 现有文献覆盖 | 缺口 |
|----|------------------------------|--------------|------|
| 慢脑（组合式抽象/技能串联/前置条件） | T6 + C6 泛化 + 事后验证 | VAGEN、ENACT、PhysMem | 验证器「审慎且事后」，无法下沉到毫秒级 |
| 快脑（点云轨迹 + Neural Kinematics + 触觉反馈） | T3 + T4 融合 + C3/C4 | NeuROK/TrAct（几何侧）；2026 触觉簇（力学侧） | 两簇尚未合流：**没有「轨迹+力」联合输出的世界模型** |
| 脊髓（滑落/碰撞/失衡那毫秒） | T4 + C5（毫秒级在线）+ C8 | TouchWorld（predictive+reactive）、FAWAM（闭环） | 判据空白：离线世界模型 vs 在线毫秒级验证；「脊髓级原子技能」文献中不存在 |

**术语映射**：吴佳俊演讲中的「world action models」在 2026 年已成命名共识（DriveWAM、TacWAM、FAWAM、ForceWAM、CW-WAM）——定义即「预测与决策联合建模」，与 T5 行对应。

---

## 5. 修正后的关键判断（更新前稿）

1. **力学输出不再是空白，而是新兴子领域**：前稿「触觉/力世界模型几乎是空白」修正为——2026 年 6-10 月出现 8+ 篇触觉/力世界模型，构成独立子领域，且命名（WAM）与「world action models」口播概念互为印证。
2. **但两个论点依然成立**：① 这些工作规模远小于视频世界模型，且**力/触觉数据仍需真实采集**（DexTouch-WM 人手数据轴、ForceWAM 的 demo+play）——「力学的 free demonstrations 在互联网上不存在」未被推翻；② 主流通用世界模型（Sora/Genie/Cosmos 线）的输出本体仍止于视觉，力学是附加专家而非原生能力。
3. **误差非对称性可作为叙事线的锋利论据**：C1/C2（视觉）错误是浪费、可事后验证；C8（接触力）错误是灾难、必须在线——这个非对称正是「自动驾驶可外挂主动安全，机器人安全只能内生」的 L1 表述。

---

## 6. 核实说明

- 本稿触觉簇 ID 均来自 arXiv 检索结果直链：TouchWorld 2607.07287、ViTacWorld 2607.22530、FeelWorld 2607.24267、TacWAM 2607.28391、FAWAM 2606.08555、ForceWAM 2610.18852、CW-WAM 2610.30117、DexTouch-WM 2609.20649（本地 index.csv 复现）。
- 其余 ID 与前稿核实结果一致，完整清单见 [world-model-survey.md §6](world-model-survey.md)。
- 各触觉簇论文的具体机制（锚引导、wrench 结构化等）仅依据标题/摘要检索片段，未深读全文；如需精确到方法细节，建议补抓 PDF 精读。

更新时间: 2026-10-26 19:30

---

## 2026-11 补记：输出任务分类的两个新证据与一个新趋势

- **新趋势：T1 内出现「显式状态」子流派**（2026-08~11）：Physical Registry（可读动力学寄存器）、InertialWorld（惯性记忆）、EPM（可执行知觉记忆）——预测动力学从隐式视频 token 中被拿出来，成为显式、可读、随演化推进的状态（对应本稿「输出本体」问题的深化）。
- **评测对 T1/T2 的审判**：FlipTest 证明视频世界模型不追踪因果状态（18.3% < 25% 随机；告知结局后 73.7%）；Roz 证明潜动力学模型不编码质量/形变/摩擦/弹性（重建 MSE 减半而物理记忆分数不变）——「输出本体 = 像素/潜表征」的路线被证明**不足以承载世界状态**。
- **T4 力学实时性在逼近 100ms 级**：TacForeSight 20Hz 推理 + 「力比触觉早约 200ms」先行指标；毫秒级（脊髓层）仍无人触及。
- 详情与伪代码：[最近三个月进展与实现细节](最近三个月进展与实现细节.md)。
