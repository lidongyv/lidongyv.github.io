# 世界模型（World Model）主要参考工作与实现方式 — 整理稿

> 整理日期：2026-10-26 ｜ 用途：支撑《一只手，和一个厨房》叙事线中「世界表示 / 世界模型」章节的文献底座。
> 姊妹篇：[world-model-capability-taxonomy.md](world-model-capability-taxonomy.md) —— 按「输出任务 × 能力」的功能视角（Marr L1）分类。
> 核实纪律：所有 arXiv ID 均经 arXiv API / abs 页面逐条核对；未核实的 ID 一律不写，只标注「ID 未核实」。来源见文末。

---

## TL;DR

1. **定义**：世界模型 = 预测「环境在动作/时间作用下如何演化」的模型。当前主流实现分三条路线：**潜动力学世界模型**（Dreamer/TD-MPC 系，闭环 MBRL）、**视频生成式世界模型**（自回归 token：Genie/GAIA；扩散 DiT：Sora/DIAMOND/Vista）、**结构化 3D/4D 世界模型**（3DGS/occupancy/对象运动学：UniWorld/OccWorld/NeuROK）。
2. **2024→2026 的演化方向**：从「像素级视频预测」走向「交互式、可控、结构化」（Genie 3、Cosmos、VerseCrafter、LaLa-World），并开始正面处理物理一致性与闭环长时程（PerpetualWonder、Matrix-Game 4.0）。
3. **与本叙事线的直接接口**：
   - 吴佳俊「向视频模型借一个世界」= **SDS 式 4D 蒸馏**（其组内 CHORD / NeuROK / ZeroHSI 一脉；视频扩散模型只贡献分数当先验，不当真值）。
   - 「world action models 比 Dreamer 令人印象深刻」→ GAIA-2 / DriveWAM / Genie 3 线 vs DreamerV3 / TD-MPC2 线。
   - 「运动学 vs 力学」缺口在文献上成立：主流世界模型的输出本体止于 RGB / occupancy / 点云（几何与外观）；**力/触觉输出在 2026 下半年已形成独立子领域**（DexTouch-WM、TouchWorld、FeelWorld、ViTacWorld、TacWAM、FAWAM、ForceWAM、CW-WAM 等 8+ 篇，详见姊妹篇 §1-T4），但规模远小于视频世界模型，且力学数据仍需真实采集——「力学数据互联网上不存在」依然成立。

---

## 1. 技术路线分类总表

| # | 路线 | 核心机制 | 代表工作 | 典型用途 |
|---|------|----------|----------|----------|
| 1 | 潜动力学世界模型 | VAE 编码观测 → RSSM/Transformer 预测潜状态 → 隐空间「想象」中训练策略 | DreamerV3、TD-MPC2、IRIS、DayDreamer | MBRL、真机策略、导航 |
| 2 | 自回归 token 视频模型 | 视频 VQ tokenizer + 因果 Transformer，动作/语言/文本 token 交织预测 | Genie 系列、GAIA-1、GenAD | 交互环境、驾驶生成 |
| 3 | 扩散视频世界模型 | DiT/flow matching，动作条件化去噪生成未来帧 | Sora、DIAMOND、Vista、DriveDreamer-2、Cosmos | 视频生成、仿真数据 |
| 4 | JEPA / 潜空间预测 | 只预测潜表征（不重建像素），从预训练视觉特征上学习 | V-JEPA、DINO-WM、LaLa-World、GLAM | 表征、零样本规划、导航 |
| 5 | 结构化 3D/4D 世界模型 | occupancy / 3DGS / 对象运动学潜空间 + 可微渲染 / SDS 蒸馏 | OccWorld、UniWorld、MUVO、ManiGaussian++、NeuROK、CHORD、PerpetualWonder | 几何一致、闭环物理 |
| 6 | VLM/LLM 中心的世界建模 | 语言状态推理 + 奖励/记忆，把「状态估计+转移预测」装进 LLM | VAGEN、PhysMem、LaLa-World | Agent、慢脑侧规划 |

---

## 2. 实现方式详解

### 2.1 潜动力学世界模型（MBRL 线）
- **原理**：图像经编码器 → 低维隐状态；循环网络（Dreamer 系 RSSM）预测隐状态转移与奖励；策略在「隐空间想象」中优化，即 *imagination-based MBRL*。
- **代表**：DreamerV3（[2301.04104](https://arxiv.org/abs/2301.04104)，150+ 任务单一配置、Minecraft 钻石）、TD-MPC2（[2310.16828](https://arxiv.org/abs/2310.16828)，隐式模型预测控制）、IRIS（[2209.00588](https://arxiv.org/abs/2209.00588)，离散潜 token + Transformer）、DayDreamer（Dreamer 上真机四足/机械臂）。
- **优点**：样本效率高、真机可用、闭环自洽。**缺点**：隐状态难跨域泛化，视觉保真度低，难扩展互联网数据。
- **开源**：DreamerV3 / TD-MPC2 / IRIS 均有官方代码。

### 2.2 自回归 token 视频世界模型
- **原理**：VQ 编码器把视频帧离散化为 token，因果 Transformer 逐 token 预测；动作（潜动作/文本/连续动作）作为条件 token 交织。
- **代表**：Genie（[2402.15391](https://arxiv.org/abs/2402.15391)，2D 平台游戏、从视频学潜动作）、GAIA-1（[2309.17080](https://arxiv.org/abs/2309.17080)，视频+文本+动作统一 token）、GenAD（[2402.11502](https://arxiv.org/abs/2402.11502)）。
- **优点**：训练稳定、长序列、天然兼容语言。**缺点**：token 化损失细节，推理逐 token 慢。
- **开源**：Genie 有官方开源实现；GAIA 系列未开源。

### 2.3 扩散视频世界模型
- **原理**：DiT（Sora）或 latent flow matching，以过去帧+动作/语言/轨迹为条件去噪生成未来帧；保真度最高。
- **代表**：Sora（[OpenAI 官方博客](https://openai.com/index/video-generation-models-as-world-simulators/)，2024.02，DiT 扩展；技术报告未发 arXiv 版）、DIAMOND（[2405.12399](https://arxiv.org/abs/2405.12399)，NeurIPS 2024 Spotlight，Atari/CS:GO 扩散世界模型）、Vista（[2405.17398](https://arxiv.org/abs/2405.17398)，NeurIPS 2024，驾驶多动作条件扩散）、DriveDreamer-2（[2403.06845](https://arxiv.org/abs/2403.06845)，AAAI 2025，LLM 增强结构化条件）。
- **优点**：视觉保真、条件灵活。**缺点**：推理慢、物理一致性差、长时程漂移。
- **开源**：DIAMOND、Vista、DriveDreamer-2 开源；Sora 闭源。

### 2.4 JEPA / 潜空间预测
- **原理**：不重建像素，只预测未来潜表征（V-JEPA：自监督视频特征预测；DINO-WM/Odin：在冻结 DINOv2 特征上建世界模型）。
- **代表**：V-JEPA（[2404.08471](https://arxiv.org/abs/2404.08471)，CVPR 2024）、DINO-WM（[2411.04983](https://arxiv.org/abs/2411.04983)，零样本规划）、LaLa-World（[2610.27851](https://arxiv.org/abs/2610.27851)，2026，语言条件化潜世界模型，规划成本仅为视频生成路线 1/6 训练、13.9× 快推理）、GLAM（[2609.14561](https://arxiv.org/abs/2609.14561)，导航）。
- **优点**：训练/推理便宜、表示可迁移。**缺点**：不能直接出视觉，需下游解码。
- **开源**：V-JEPA、DINO-WM、LaLa-World 开源。

### 2.5 结构化 3D/4D 世界模型
- **原理**：世界状态用 occupancy / 3D Gaussian / 对象运动学潜空间显式表示，动力学在 3D 中演化，再用渲染/蒸馏与视觉对齐。分两类：
  - **几何驱动生成**：OccWorld（[2311.16038](https://arxiv.org/abs/2311.16038)，occupancy 演化）、UniWorld（[2308.07234](https://arxiv.org/abs/2308.07234)，世界模型预训练）、MUVO（[2311.11762](https://arxiv.org/abs/2311.11762)，3D 几何多模态）、ManiGaussian++（[2506.19842](https://arxiv.org/abs/2506.19842)，层次化 3DGS 双手操控）、VerseCrafter（[2601.05138](https://arxiv.org/abs/2601.05138)，CVPR 2026，4D 几何控制图喂视频扩散）。
  - **从视频蒸馏物理**（吴佳俊组一脉）：CHORD（[2601.04194](https://arxiv.org/abs/2601.04194)，从 2D 视频提取拉格朗日运动，生成多体 4D 动力学）、NeuROK（[2605.30347](https://arxiv.org/abs/2605.30347)，对象运动学潜空间，即演讲中的「Neural Kinematics」）、PerpetualWonder（[2602.04876](https://arxiv.org/abs/2602.04876)，物理状态↔视觉表示双向链接的闭环生成）、ZeroHSI（[2412.18600](https://arxiv.org/abs/2412.18600)，零样本人-场景交互）。
- **优点**：几何/物理一致、可控、可交互。**缺点**：依赖 3D/4D 数据，渲染与动力学解耦难。
- **开源**：OccWorld、UniWorld、ManiGaussian++ 开源。

### 2.6 VLM/LLM 中心的世界建模
- **原理**：把 POMDP 的「状态估计 + 转移建模」装进 VLM，用 RL 奖励显式世界建模推理（VAGEN），或在推理期用「先验证后应用」的记忆框架学习物理规律（PhysMem）。
- **代表**：VAGEN（[2510.16907](https://arxiv.org/abs/2510.16907)，World Modeling Reward + Bi-Level GAE）、PhysMem（[2602.20323](https://arxiv.org/abs/2602.20323)，测试期物理记忆，砖块插入 76% vs 直接检索 23%）。
- **优点**：语义泛化、免训练扩展。**缺点**：非毫秒级、接触力学仍靠语言近似。

---

## 3. 分域主要参考工作清单

### 3.1 通用 / 视频世界模型

| 工作 | 年份 | 机构 | ID / 出处 | 实现方式一句话 |
|------|------|------|-----------|----------------|
| World Models（奠基） | 2018 | NeurIPS | —（经典论文，直接搜标题） | VAE + MDN-RNN，隐空间做梦训练 |
| Genie | 2024 | DeepMind | [2402.15391](https://arxiv.org/abs/2402.15391) | 视频 token 自回归 + 潜动作，开源 |
| Genie 2 | 2024.12 | DeepMind | [官方博客](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)（arXiv ID 未核实） | 长时程交互世界模型，闭源 |
| Genie 3 | 2025.08 | DeepMind | [官方博客](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)（arXiv ID 未核实） | 语言+动作交互式世界模型，NeurIPS 2025 演示 |
| Sora | 2024.02 | OpenAI | [官方博客](https://openai.com/index/video-generation-models-as-world-simulators/)（技术报告无 arXiv 版） | DiT 视频扩散，闭源 |
| Cosmos WFMs | 2025.01 | NVIDIA | [2501.03575](https://arxiv.org/abs/2501.03575) | 自回归+扩散双路线世界基础模型，权重开源 |
| V-JEPA | 2024 | Meta | [2404.08471](https://arxiv.org/abs/2404.08471) | JEPA 潜空间预测，开源 |
| DIAMOND | 2024 | NeurIPS Spotlight | [2405.12399](https://arxiv.org/abs/2405.12399) | 扩散世界模型跑 Atari 100K（HNS 1.46）、CS:GO 可玩引擎，开源 |
| WorldDreamer | 2024 | ECCV | [2401.09985](https://arxiv.org/abs/2401.09985) | masked token 预测的通用世界模型，开源 |
| VerseCrafter | 2026 | CVPR | [2601.05138](https://arxiv.org/abs/2601.05138) | 静态点云背景+逐物体 3DGS 轨迹的 4D 几何控制 |
| Matrix-Game 4.0 | 2026 | — | [2610.27999](https://arxiv.org/abs/2610.27999) | 游戏引擎管状态 + 自回归视频渲染，480p/60FPS 实时可玩，开源 |

### 3.2 自动驾驶世界模型

| 工作 | 年份 | 机构 | ID / 出处 | 实现方式一句话 |
|------|------|------|-----------|----------------|
| GAIA-1 | 2023 | Wayve | [2309.17080](https://arxiv.org/abs/2309.17080) | 视频+文本+动作 token 自回归 |
| GAIA-2 | 2025 | Wayve | [2503.20523](https://arxiv.org/abs/2503.20523) | 多视角可控生成世界模型（[官方博客](https://wayve.ai/thinking/gaia-2/)） |
| UniSim | 2023 | Berkeley/DeepMind | [2310.06114](https://arxiv.org/abs/2310.06114) | 多数据集编排的通用交互模拟器，仿真训的策略零样本上真机 |
| DriveDreamer-2 | 2024 | AAAI 2025 | [2403.06845](https://arxiv.org/abs/2403.06845) | LLM 结构化条件 + 动作条件视频扩散 |
| GenAD | 2024 | — | [2402.11502](https://arxiv.org/abs/2402.11502) | 自回归生成式端到端驾驶 |
| Drive-WM | 2023 | — | [2311.17918](https://arxiv.org/abs/2311.17918) | 多视角视频预测 + 规划闭环 |
| Vista | 2024 | NeurIPS | [2405.17398](https://arxiv.org/abs/2405.17398) | 高保真多动作条件扩散世界模型 |
| MUVO | 2023 | Wayve | [2311.11762](https://arxiv.org/abs/2311.11762) | 3D 几何表示的多模态世界模型 |
| OccWorld | 2023 | — | [2311.16038](https://arxiv.org/abs/2311.16038) | 3D occupancy 上预测世界演化 |
| UniWorld | 2023 | — | [2308.07234](https://arxiv.org/abs/2308.07234) | 世界模型预训练（重建免标注） |
| World on Rails | 2024 | NeurIPS | [GitHub](https://github.com/dotchen/WorldOnRails)（arXiv ID 未核实） | 车辆沿「轨」前进的分解式世界模型 |
| DriveWAM | 2026 | — | [2605.28544](https://arxiv.org/abs/2605.28544) | 视频生成先验的世界-动作联合建模 |

### 3.3 机器人 / 具身世界模型

| 工作 | 年份 | 机构 | ID / 出处 | 实现方式一句话 |
|------|------|------|-----------|----------------|
| DreamerV3 | 2023 | DeepMind | [2301.04104](https://arxiv.org/abs/2301.04104) | RSSM 隐想象，150+ 任务单配置 |
| DayDreamer | 2023 | — | 项目页（RLJ 发表） | Dreamer 真机四足/机械臂 |
| TD-MPC2 | 2023 | — | [2310.16828](https://arxiv.org/abs/2310.16828) | 隐式模型预测控制，单超参跨域 |
| IRIS | 2023 | ICML | [2209.00588](https://arxiv.org/abs/2209.00588) | 离散潜 token + Transformer 世界模型 |
| UniPi | 2023 | NeurIPS | [2302.00111](https://arxiv.org/abs/2302.00111) | 文本条件视频生成 = 通用策略 |
| SuSIE | 2023 | RSS | 项目页（无 arXiv） | 图像编辑扩散实现子目标合成 |
| DINO-WM | 2024 | — | [2411.04983](https://arxiv.org/abs/2411.04983) | 冻结 DINOv2 特征上的零样本规划世界模型 |
| Odin | 2025 | NYU（LeCun 线） | ID 未核实 | DINO-WM 升级版（连续域/机器人） |
| Humanoid World Models | 2025 | NYU 等 | [2506.01182](https://arxiv.org/abs/2506.01182) | 人形机器人开源世界基础模型 |
| GR00T-N1-2B | 2025.08 | NVIDIA | 官方博客发布（arXiv ID 未核实，不附链接） | 基于 GR00T N1（[2503.14734](https://arxiv.org/abs/2503.14734)）的 2B 数据驱动世界模型 |
| ManiGaussian++ | 2025 | — | [2506.19842](https://arxiv.org/abs/2506.19842) | 主从臂层次化 3DGS 世界模型，真机 60% 成功率，开源 |
| **DexTouch-WM** | 2026 | 丁文伯组 | [2609.20649](https://arxiv.org/abs/2609.20649) | **动作条件触觉世界模型**：人手+机械手同构触觉阵列，人手 0→100h 数据提升机器人域视觉/几何/接触预测；可作策略评测代理环境与合成轨迹生成器 |
| GLAM | 2026 | 丁文伯组 | [2609.14561](https://arxiv.org/abs/2609.14561) | JEPA 式全局时空记忆潜世界模型（导航） |

### 3.4 综述与评测基准

| 工作 | 年份 | ID / 出处 | 要点 |
|------|------|-----------|------|
| World Model for Robot Learning: A Comprehensive Survey | 2026 | [2605.00080](https://arxiv.org/abs/2605.00080) | 机器人视角最全面综述（吴佳俊等参与） |
| A Comprehensive Survey on World Models for Embodied AI | 2025 | [2510.16732](https://huggingface.co/papers/2510.16732) | 具身 AI 世界模型综述 |
| The Role of World Models in Shaping Autonomous Driving | 2025 | [2502.10498](https://arxiv.org/abs/2502.10498) | AD 世界模型综述 |
| A Step Toward World Models: A Survey on Robotic Manipulation | 2025 | [2511.02097](https://arxiv.org/abs/2511.02097) | 操作方向综述 |
| Beyond Simulation: Benchmarking World Models for Planning and Causality in AD | 2025 | [2508.01922](https://huggingface.co/papers/2508.01922) | 闭环规划/因果基准（FVD 之外） |
| WorldScore | 2025 | [2504.00983](https://arxiv.org/abs/2504.00983) | 世界生成统一评测（可控性/质量/动力学，19 模型） |

---

## 4. 与本地材料的映射（《一只手，和一个厨房》叙事线）

**本地已抓取的论文（papers/ 目录）可直接引用：**

吴佳俊组（斯坦福）：
- **NeuROK**（2605.30347）= 演讲中的「Neural Kinematics」：对象运动学潜空间，从静态网格零样本生成可动性/形变 —— 「从物体本身推断可动方式」的论文载体。
- **CHORD**（2601.04194）：从 2D 视频蒸馏拉格朗日运动生成多体 4D 动力学 —— 「向视频模型借一个世界」的具名实现（distillation-based）。
- **PerpetualWonder**（2602.04876）：物理状态与视觉原语双向链接的闭环 4D 生成（「我们并没有真的生成视频……视频扩散模型只被用来计算分数」的严格版）。
- **TrAct**（2608.24101）与 **Masked Visual Actions**（2607.19343）：动作接口问题——「视觉轨迹/像素空间接口优于裸动作条件化」，即「运动学轨迹如何与视觉预测对齐」。
- **WorldScore**（2504.00983）：世界生成评测基准（对应演讲中「三个基准均评预测对不对/任务完成没完成」）。
- 另有 VAGEN（2510.16907）、ENACT（2511.20937）支撑「慢脑 VLM 世界建模」侧。

丁文伯组（清华/Xspark）：
- **DexTouch-WM**（2609.20649）：**触觉世界模型**——人手触觉数据 + 动作重定向监督同一动力学模型。对叙事线的三个直接意义：
  1. 它是「力学的 free demonstrations 从哪来」的**现有最强回答**：人手交互 100 小时是可扩展数据轴（但仍需真实采集，印证「力学数据互联网上不存在」）；
  2. 它把世界模型当「策略评测代理环境 + 合成轨迹生成器」用——即触觉侧也可以做「验证器」；
  3. 其机制（视频专家 + 轻量触觉专家、anatomy-aware 触觉 token）正好与「快脑 VTLA+TWAM」架构同构。
- GLAM（2609.14561）、ManiGaussian++（2506.19842）：组内在导航/几何侧的世界模型工作。

**叙事主张 ↔ 文献证据对照：**

| 叙事主张 | 文献坐标 | 缺口 |
|----------|----------|------|
| 「向视频模型借一个世界，只取分数不当真值」 | CHORD / NeuROK / ZeroHSI（SDS 蒸馏线） | 无 |
| 「world action models 比 Dreamer 令人印象深刻」 | Genie 3 / GAIA-2 / DriveWAM vs DreamerV3 / TD-MPC2 | 无 |
| 「运动学 vs 力学是正交缺口」 | 全部 RGB/occupancy/点云世界模型只有几何量；力/接触输出仅 2026 触觉簇（DexTouch-WM、TouchWorld、FAWAM 等）触及 | 力学预测仍无与视觉同规模的预训练 |
| 「力学的 free demonstrations 不存在」 | DexTouch-WM 仍需真实触觉采集（人手数据是新轴，非免费） | 无「Neural Dynamics」式零样本力学推断 |
| 「脊髓级原子技能」 | 文献中无对应概念（最近的是 PhysMem 的测试期物理记忆，非毫秒级） | 空白 |
| 「hold 是需连续维持的前置条件」 | PhysMem「先验证后应用」触及接触物理推理（76% vs 23%），但非实时 | 空白 |

---

## 5. 趋势与开放问题

1. **控制接口的再设计**：动作如何喂给世界模型成为 2025-2026 热点——语言（LaLa-World）、视觉轨迹（TrAct）、像素掩码（Masked Visual Actions）、4D 几何控制图（VerseCrafter）。结论趋同：**与视觉空间对齐的接口优于裸动作向量**。
2. **交互式与实时化**：Genie 3 的交互记忆、Matrix-Game 4.0 的 60FPS 实时可玩（引擎管状态、视频管外观的混合架构）。
3. **评测重心迁移**：从 FID/FVD 迁移到闭环规划收益、因果一致性（2508.01922）、可控性/动力学分解（WorldScore）。
4. **物理一致性仍是公认短板**：闭环漂移、物体穿透、接触错误（PerpetualWonder 的动机即物理-视觉解耦）。
5. **触觉/力世界模型从「空白」变为「新兴子领域」**：2026 下半年出现 8+ 篇触觉/力世界模型（含 4 篇 WAM 命名），但主流通用世界模型的观测仍止于视觉——「接触层一直没被写进任何一张图」修正为「接触层刚开始被写进图里，且只有触觉社区自己在写」。能力矩阵与误差非对称性分析见姊妹篇。

---

## 6. 核实说明

- **已核实 ID**（arXiv API/abs 页）：Genie 2402.15391、GAIA-1 2309.17080、GAIA-2 2503.20523、DIAMOND 2405.12399、DreamerV3 2301.04104、TD-MPC2 2310.16828、IRIS 2209.00588、UniPi 2302.00111、UniSim 2310.06114、V-JEPA 2404.08471、Cosmos 2501.03575、Vista 2405.17398、WorldDreamer 2401.09985、OccWorld 2311.16038、UniWorld 2308.07234、MUVO 2311.11762、Drive-WM 2311.17918、DriveDreamer-2 2403.06845、GenAD 2402.11502、DINO-WM 2411.04983、Humanoid World Models 2506.01182、ManiGaussian++ 2506.19842、VerseCrafter 2601.05138、LaLa-World 2610.27851、Matrix-Game 4.0 2610.27999、PhysMem 2602.20323、DriveWAM 2605.28544，及全部本地抓取条目（来自 papers/index.csv）。
- **ID 未核实、仅官方出处**：Genie 2/3（DeepMind 博客）、Sora（OpenAI 博客）、GR00T-N1-2B（NVIDIA 博客）、Odin、World on Rails（GitHub）、SuSIE（项目页）。
- **排除项**：搜索中出现的同名工作（如「Odin: Primitive-Level Synchronization」分布式渲染论文）已剔除，防止误引。
- 本页所有「本地」ID 均可在 `papers/index.csv` 中复现核查。

更新时间: 2026-10-26 19:30（本次：修正触觉簇判断，链接姊妹篇）

---

## 2026-11 补记：最近三个月进展

- **Genie 4**（DeepMind 官方博客，2026-11）：「A new generation of world simulation」，无 arXiv 论文（与 Genie 2/3 同口径）。
- **新晋 10 篇精读**（实现细节+伪代码见 [最近三个月进展与实现细节](最近三个月进展与实现细节.md)）：TacForeSight、DTWM、Physical Registry、Roz、DiffWAM、FlipTest、SAFER、EPM、InertialWorld、AgentWorld。
- **对本稿两条判断的更新**：①「T4 力学是新兴子领域」继续兑现（DTWM 手套触觉注入、TacForeSight 20Hz 力引导预测）；②「世界模型学的是外观而非状态」从推断升级为**有评测证据**（FlipTest 18.3%<25% 随机；Roz 物理记忆全员崩溃、重建 MSE 减半而分数不变）。
