# VLA / VA 方案主要参考论文：方法与实现方式

> 调研日期：2026-10-26 · 用途：《VLA 在 L4 的可执行落地方案》参考资料
> 核实方式：所有 arXiv 条目均经 arXiv 官方 API/检索页核对标题与年份；个别无 arXiv 论文的工作（MiniVLA、π0-FAST）以官方博客链接给出并在文末注明。已剔除 10 余个记忆错误的论文 ID。

## TL;DR

- **VLA（Vision-Language-Action）= 观测 + 语言指令 → 直接输出动作**；**VA（Vision-Action）= 弱化/去掉语言，视觉直接到动作**。你的 L4 方案语境下，两者都对应「感知不再作为上游依赖任务、直接输出驾驶行为」这一代方案，与 two-stage（模块化管线）相对。
- 三条技术主脉：
  1. **离散动作 token 的 VLM 路线**（RT-2 → OpenVLA → GR00T N1）：互联网知识迁移强、语言接地好，但推理重（7B+）；
  2. **连续动作生成路线**（ACT 回归 / Diffusion Policy 扩散 / π0 流匹配）：工程成熟、可实时、LeRobot 全家桶可直接落地，是「遥操数据 → 策略」最直接的底座；
  3. **自动驾驶端到端/VLM 路线**（UniAD/VAD → DriveVLM/LMDrive/EMMA/Senna）：动作空间是轨迹 waypoint 或控制量，与你的 two-stage 共存、遥操替换场景最贴合。
- 与你方案最对症的 6 篇（先读）：**LMDrive、Senna、DriveVLM(-Dual)、ACT、Diffusion Policy、LAPA**；外加两篇 2025 底 survey 作为目录。
- 开源实现栈已收敛：**LeRobot**（训练/推理全栈）、**OpenVLA-ft**（LoRA 微调 7B）、**openpi**（π0 系 50Hz）、**VAD/Senna/LMDrive/OpenDriveLab**（驾驶侧）。

---

## 一、概念与分类

### 1.1 VLA vs VA vs two-stage

| 术语 | 含义 | 代表工作 |
|---|---|---|
| VLA | 视觉-语言-动作：多模态观测 + 自然语言指令 → 直接输出动作（token/连续量/轨迹） | RT-2、OpenVLA、π0、GR00T N1、EMMA |
| VA | 视觉-动作：语言弱化或仅作目标条件，视觉直接映射动作 | Diffusion Policy、ACT、DP3、GNM |
| two-stage | 模块化：感知→预测→规划→控制 逐级传递，感知是上游依赖 | 你司现有方案；DriveVLM-Dual/Senna 是「VLM 高层 + 传统/E2E 低层」的混合体 |

### 1.2 动作输出范式（选型关键）

- **离散 token**：动作量化成文本 token 自回归生成（RT-2、GR00T N1、BeT/VQ-BeT）——语言对齐最好、推理较慢；
- **连续回归/分块**：直接回归 k 步动作 chunk（ACT、UniAD/VAD 的轨迹点）——最小可行、实时；
- **扩散/流匹配**：去噪/直线 ODE 生成动作（Diffusion Policy、π0）——多模态分布建模好，流匹配推理快于扩散；
- **潜空间动作**：先学视频潜动作再映射（LAPA、UniPi）——可用无标签行车视频预训练。

### 1.3 自动驾驶侧的动作空间

轨迹 **waypoint**（EMMA、UniAD、VAD、Senna 低层）vs 直接**控制量**（DriveGPT4、LMDrive 低层控制）。低速 robovan 场景两者都可用；waypoint + 已有跟踪控制器是最稳妥的组合。

---

## 二、通用 VLA 核心模型（语言条件）

| 论文 | 会议/年 | 方法核心 | 实现方式 |
|---|---|---|---|
| [RT-1](https://arxiv.org/abs/2212.06817) | RSS 2023 | 130k 真实轨迹预训练的 Robotics Transformer，图像 token + 文本指令 → 离散动作 token；先证「数据规模×模型容量」可扩展性 | 未开源权重；方法论奠基 |
| [RT-2](https://arxiv.org/abs/2307.15818) | CoRL 2023 | VLA 命名之作：把动作表达为文本 token，VLM（PaLI-X/PaLM-E）与机器人轨迹 + 网页 VQA 数据 co-fine-tune；CoT 涌现语义推理 | 未开源；概念与训练 recipe 必读 |
| [Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864) | ICRA 2024 | 21 机构/22 种机器人/527 技能数据集；RT-1/RT-2 训练出的 RT-X 跨本体正迁移 | 数据集开源（后续 OpenVLA 的训练数据来源） |
| [Octo](https://arxiv.org/abs/2405.12213) | RSS 2024 | 800k 轨迹的 27M 扩散 Transformer 通用策略；语言或目标图像条件；可几小时内微调适配新本体 | 开源（octo-models.github.io）；轻量、可微调，适合车队定制 |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | CoRL 2024 | 7B：Llama 2 + DINOv2/SigLIP 视觉融合，970k 真实演示预训练，动作离散 token；比 RT-2-X(55B) 高 16.5% 成功率、比 Diffusion Policy 高 20.4% | **开源权重+代码**；LoRA 消费级 GPU 可微调、量化无损推理；OpenVLA-ft 是微调生态 |
| [TinyVLA](https://arxiv.org/abs/2409.12514) | arXiv 2024 | 面向快速、数据高效的小型 VLA（小 VLM 骨干 + 动作头） | 开源；车载小模型路线的参考实现 |
| [MiniVLA](https://ai.stanford.edu/blog/minivla/) | 2025（斯坦福博客） | 「更小 footprint 的更好 VLA」：研究不同小 VLM 骨干 + 动作头的 Pareto 前沿 | 无 arXiv 论文（❗仅有官方博客+代码）；选骨干时直接抄它的结论 |
| [FAST](https://arxiv.org/abs/2501.09747) | arXiv 2025 | 高效动作 tokenization（FAST tokenizer）：连续动作压缩成 ~1k 离散 bin，比扩散版 π0 收敛显著更快 | 开源；π0-FAST 的动作头基础；车载 token 化效率的关键参考 |
| [π₀](https://arxiv.org/abs/2410.24164) | RSS 2025 | VLM（PaliGemma）+ 动作专家：流匹配生成 50Hz 动作 chunk；语言指令或高层 VLM 调度；多平台预训练 | **openpi 开源含权重**；「指令-动作」分流架构最值得借鉴；3B 级，车载需蒸馏/量化 |
| [π0.5](https://arxiv.org/abs/2504.16054) | arXiv 2025 | π0 增强版：开放世界泛化，few-shot 学习能力强化 | openpi 生态 |
| [π0-FAST](https://www.physicalintelligence.com/blog/pi0-fast) | 2025（PI 博客） | 自回归 + FAST tokenizer 替代扩散推理，推理加速 ~5× | 无独立论文（❗博客+openpi 代码）；「扩散→自回归蒸馏」是车载部署的现成路子 |
| [GR00T N1](https://arxiv.org/abs/2503.14734) | arXiv 2025 | NVIDIA：2B 双系统（System 1 快动作 + System 2 慢推理 VLM），开放权重 | 权重/训练数据开源（HF/NGC），Isaac GR00T 平台配套；团队自训 VLA 的流程参照 |
| [Gemini Robotics / -ER](https://arxiv.org/abs/2503.20020) | arXiv 2025 | 基于 Gemini 2.0 的 VLA（直接控制）+ ER（具身推理：3D 框、轨迹/抓取预测）；100 演示即可学新短时任务、可适配新本体 | 未开源；方法论：先训 ER 再训 VLA 的两级路线 |
| [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342) | arXiv 2025 | Motion Transfer 机制学多本体异构数据；动作与自然语言推理交错（think before acting） | 未开源；多本体数据混合 + 推理交织是 2025 主流方向 |
| [Qwen-VLA](https://arxiv.org/abs/2605.30280) | arXiv 2026 | Qwen 官方：统一操作/导航/轨迹预测于一体，DiT 动作解码器，embodiment-aware prompt；R2R/RxR 导航与轨迹预测成绩强 | 2026 最新；**唯一同时覆盖「导航 + 轨迹预测」的开源系 VLA，与车辆场景距离最近，重点跟踪** |

**Survey**：[A Survey on Vision-Language-Action Models for Embodied AI](https://arxiv.org/abs/2405.14093)（arXiv 2024，方向全景目录）。

---

## 三、VA 及动作生成方法（弱语言 / 无语言）

> 面向「遥操数据 → 动作策略」的训练方法学，是 L4 遥操替代路线的直接技术底座。

### 3.1 动作分块 + Transformer

| 论文 | 会议/年 | 方法核心 | 实现 |
|---|---|---|---|
| [ACT](https://arxiv.org/abs/2304.13705) | RSS 2023 | CVAE+Transformer 回归 k=100 步连续动作 chunk（L1+KL），多相机 RGB+关节角 | 官方 tonyzhaozh/aloha；约 80M 参数，50Hz 实时，temporal ensemble 平滑；LeRobot 已有 ACT 车辆控制实现 → 低速车入门首选 |
| [BeT](https://arxiv.org/abs/2206.11251) | NeurIPS 2022 | 动作 k-means 离散为多模态 bin 分类 + 连续残差回归 | notmahi/bet；作者在 CARLA 上评测过；多模态（直行/转弯/停车）契合驾驶 |
| [VQ-BeT](https://arxiv.org/abs/2403.03181) | ICML 2024 | Residual-VQ 动作 token 化 + GPT 式分类；推理比扩散快约 5× | jayLEE0301/vq_bet_official；适合车载实时 |

### 3.2 扩散策略

| 论文 | 会议/年 | 方法核心 | 实现 |
|---|---|---|---|
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) | RSS 2023 | 条件去噪扩散生成动作序列；receding-horizon + temporal ensemble（chunk 16 执行 8）+ FiLM 视觉条件 | diffusion-policy.cs.columbia.edu；CNN 版 <10Hz、轻量骨干可实时；业界默认基线，慢速车可直接用 |
| [DP3](https://arxiv.org/abs/2403.03954) | RSS 2024 | 稀疏点云 MLP 点编码器 + 扩散头；10 次演示/72 仿真任务、真机 85% | 3d-diffusion-policy.github.io；点云输入与 LiDAR 天然契合（iDP3/RDT 均复用其基座） |
| [iDP3](https://arxiv.org/abs/2410.10803) | IROS 2025 | DP3 改进：3D LiDAR 点云 + 本体感知，纯机载算力实机部署，>2000 rollout | humanoid-manipulation.github.io；最贴近「车载传感器 + 机载计算」 |
| [CrossFormer](https://arxiv.org/abs/2408.11812) | CoRL 2024 | 跨本体 Transformer：900K 轨迹/20 种本体（含轮式、无人机），动作空间自动对齐，扩散头 | crossformer-model.github.io；轮式平台并入多源数据的范式参考 |
| [RDT-1B](https://arxiv.org/abs/2410.07864) | ICLR 2025 | 1.2B 扩散 Transformer，物理可解释统一动作空间，1~5 演示微调 | rdt-robotics.github.io 开源；对低速车过重，统一动作空间设计可借鉴 |
| [UniPi](https://arxiv.org/abs/2302.00111) | NeurIPS 2023 | 先扩散生成未来视频，再逆动力学模型提取动作（视频内规划） | universal-policy.github.io；两步推理延迟高，仅方法储备 |
| [SeedPolicy](https://arxiv.org/abs/2603.05117) | 2026 arXiv | SEGA 门控注意力扩展观测/动作时间视野，参数比 RDT-1B 少 1–2 个数量级 | github.com/Youqiang-Gui/SeedPolicy；低成本长视野控制，值得跟踪 |

### 3.3 流匹配 / 整流流

| 论文 | 会议/年 | 方法核心 | 实现 |
|---|---|---|---|
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | ICLR 2023 | 学习直线 ODE 传输分布，reflow 使路径变直、少数步采样 | github.com/gnobitab/RectifiedFlow；少步采样=低延迟，策略流匹配的理论底座 |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | ICLR 2023 | 条件概率路径回归向量场，免仿真训练 CNF | Meta 官方 github.com/facebookresearch/flow_matching；π0 的训练目标直接来源 |
| [RFMP](https://arxiv.org/abs/2412.10855) | IEEE T-RO 2025 | SE(3) 黎曼流形上流匹配策略，比扩散/一致性策略更高效 | sites.google.com/view/rfmp；车辆航向属流形动作，几何约束思路可借鉴 |

### 3.4 潜空间动作

| 论文 | 会议/年 | 方法核心 | 实现 |
|---|---|---|---|
| [LAPA](https://arxiv.org/abs/2410.11758) | ICLR 2025 | VQ-VAE 从无标签视频学离散潜动作（帧间变化），预训练 VLA 预测潜动作，少量遥操数据微调到真机 | latentactionpretraining.github.io（UniPi 团队）；**用行车视频（无标签）预训练 → 少量遥操数据微调，最契合本项目数据形态** |

动作 token 化选型对比：BeT = k-means 单层离散（高维难扩展）；VQ-BeT = 残差 VQ 层次 token + GPT 分类（快、欠平滑）；LAPA/π0 = 潜动作/流匹配（连续、保真度高、训练要求更高）；UniPi = 视频潜空间（最重）。选择取决于部署算力与延迟预算。

### 3.5 预训练与数据

| 论文 | 会议/年 | 方法核心 | 实现 |
|---|---|---|---|
| [HPT](https://arxiv.org/abs/2409.20537) | NeurIPS 2024 | 本体特定 stem 对齐到共享 Transformer 主干，52 数据集/多本体+人类视频预训练，微调 +20% | liruiw.github.io/hpt；跨本体共享主干预训练范式可迁移到 robovan 车队 |
| [GR-1](https://arxiv.org/abs/2312.13139) | ICLR 2024 | GPT 式同时预测动作与未来帧，Ego4D 预训练 → CALVIN/真机微调 | GR1-Manipulation.github.io；视频预测辅助训练对驾驶有效 |
| [GR-2](https://arxiv.org/abs/2410.06158) | ICLR 2025 | 38M 视频片段/50B token 预训练后微调视频生成+动作预测，100+ 任务 97.7% | gr2-manipulation.github.io（字节）；模型大、车载实时难，规模化路线证明 |
| [RoboCat](https://arxiv.org/abs/2306.11706) | TMLR 2023 | 多本体多任务 VQ-GCT 决策 Transformer，自生成数据自我改进循环，100–1000 样本适配新任务 | 权重未开源；**「策略生成新数据 → 迭代训练」与「逐步替换人工遥操」闭环思想直接对应** |
| [GNM](https://arxiv.org/abs/2210.03370) | ICRA 2023 | 目标条件导航策略（topological graph + 图像目标），6 种机器人 60 小时数据跨本体泛化 | github.com/robodhruv/drive-any-robot；**唯一直接面向轮式导航/驾驶的条目，低速车导航模块可直接复用** |
| [RynnBrain 1.1](https://arxiv.org/abs/2607.17977) | 2026 arXiv | 阿里 DAMO：2B/9B/122B 家族，跨本体统一动作空间+掩码，3 类真机部署 | github.com/alibaba-damo-academy/RynnBrain；2026 具身基础模型趋势观察 |

### 3.6 开源工具栈

| 工具 | 要点 |
|---|---|
| [LeRobot](https://arxiv.org/abs/2602.22818)（ICLR 2026） | Hugging Face 端到端库：数据采集/存储/训练/异步推理全栈；内置 ACT、Diffusion Policy（CNN/Transformer）、VQ-BeT、TDMPC。推理：receding-horizon（chunk 默认 16 步每 n 步重推）+ temporal ensemble 重叠平均；ACT 加温度采样；50Hz 场景需小模型或 chunk 复用。**可直接作为低速 robovan 遥操数据训练与车端推理的统一栈** |

---

## 四、自动驾驶方向 VLA / 端到端

| 论文 | 会议/年 | 方法核心 | 实现方式 |
|---|---|---|---|
| [UniAD](https://arxiv.org/abs/2212.10156) | CVPR 2023（Award Candidate） | 「规划导向」：感知-预测-规划全栈合一网络，统一 query 接口；nuScenes SOTA | 代码/模型公开（OpenDriveLab）；e2e 先例，非语言模型 |
| [VAD](https://arxiv.org/abs/2303.12077) | ICCV 2023 | 全矢量化场景表示（agent 运动 + 地图元素作显式规划约束），去栅格化；碰撞率 -29%，VAD-Tiny 提速 9.3× | 开源 github.com/hustvl/VAD；**VAD-Tiny 的轻量化思路=车载实时参考** |
| [DriveGPT4](https://arxiv.org/abs/2310.01412) | RA-L 2024 | LLM 可解释端到端驾驶：多帧视频 + 文本问答 + 直接预测低层控制量；BDD-X 评测 | 项目页开源；「VLA 解释驾驶行为」的先例 |
| [DriveVLM / DriveVLM-Dual](https://arxiv.org/abs/2402.12289) | arXiv 2024 | VLM 做场景描述-分析-层级规划；**Dual 与传统管线混合**（VLM 慢决策 + 传统快执行），并在量产车部署验证 | 清华 MARS Lab；**与你「VLA 与 two-stage 并存、可切回」的方案逻辑直接对应** |
| [LMDrive](https://arxiv.org/abs/2312.07488) | CVPR 2024 | 语言引导的闭环端到端驾驶：多模态传感 + 自然语言指令；发布 64K 指令跟随片段 + LangAuto 基准 | 开源 github.com/opendilab/LMDrive（含数据集）；**「语言指令 + 闭环驾驶数据」训练路线的最直接参照** |
| [OmniDrive](https://arxiv.org/abs/2405.01533) | arXiv 2024 | 反事实推理对齐 VLM 与 3D 驾驶任务；Omni-L/Omni-Q 双 Agent；DriveLM Q&A 与 nuScenes 开环规划提升 | 开源（NVIDIA）；语言-规划对齐数据怎么做有完整答案 |
| [Senna](https://arxiv.org/abs/2410.22313) | arXiv 2024 | **高层规划（VLM 自然语言）与低层轨迹（E2E）解耦**；DriveX 大规模预训练 + nuScenes 微调：规划误差 -27.12%、碰撞率 -33.33% | 开源 github.com/hustvl/Senna；**架构上与「VLA 上层 + two-stage 下层」的落地形态几乎一致** |
| [EMMA](https://arxiv.org/abs/2410.23262) | arXiv 2024（Waymo） | 基于 Gemini 的端到端多模态驾驶模型：相机图像 → trajectory tokens 输出航点；语言指令可条件 | 未开源；大厂 VLA 驾驶的标杆，轨迹 token 化做法值得学 |
| [CarLLaVA](https://arxiv.org/abs/2406.10165) | arXiv 2024 | 相机-only 语言条件闭环驾驶 VLM（CARLA 平台，LLM4AD Challenge 相关工作） | OpenDriveLab 系；纯视觉低成本语言驾驶参考 |
| [GAIA-1](https://arxiv.org/abs/2309.17080) | arXiv 2023（Wayve） | 视频+文本+动作 → 生成驾驶世界的世界模型（离散 token 自回归） | 未开源；世界模型路线储备（与遥操数据仿真增强相关） |
| [Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future](https://arxiv.org/abs/2512.16760) | arXiv 2025 survey | **AD 专用 VLA survey**：End-to-End VLA vs Dual-System VLA 两范式，文本 vs 数值动作生成、显式 vs 隐式引导的分类；数据集/基准汇总 | 配套仓库 github.com/worldbench/awesome-vla-for-ad；**作为本节目录，按图索骥即可** |
| [Language2Drive](https://arxiv.org/abs/2610.16634) | 2026 arXiv | 评测驾驶 VLA 是否真正遵循指令（指令保真度） | 2025–2026 新作（标题经检索确认，摘要未逐字核对） |
| [VLDrive](https://arxiv.org/abs/2511.06256) | 2025 arXiv | 视觉增强轻量 MLLM 的高效语言引导驾驶 | 同上，跟踪项 |
| [ICR-Drive](https://arxiv.org/abs/2604.05378) | 2026 arXiv | 语言驾驶的指令反事实鲁棒性 | 同上，跟踪项 |

---

## 五、实现方式汇总（开源栈 / 模型规模 / 实时性）

### 5.1 训练与推理全栈

| 栈 | 用途 | 关键事实 |
|---|---|---|
| [LeRobot](https://github.com/huggingface/lerobot) | 数据采集→训练→异步推理 | 内置 ACT/DP/VQ-BeT；receding-horizon + temporal ensemble；**遥操数据训练的第一选择** |
| [OpenVLA / OpenVLA-ft](https://openvla.github.io/) | 7B VLA 微调 | LoRA 在消费级 GPU 可微调；量化后推理不掉点；DINOv2+SigLIP 视觉融合 |
| [openpi](https://github.com/Physical-Intelligence/openpi) | π0/π0.5/π0-FAST | 流匹配 50Hz 动作 chunk；多平台预训练权重；自回归版推理快 5× |
| [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/gr00t) | GR00T N1 | 2B 双系统开放权重 + 仿真/数据工具链 |
| [VAD](https://github.com/hustvl/VAD) / [Senna](https://github.com/hustvl/Senna) / [LMDrive](https://github.com/opendilab/LMDrive) | 驾驶侧开源 | 轨迹 waypoint 输出 + 语言条件；数据 nuScenes/nuPlan/CARLA/LangAuto/DriveX |
| [awesome-vla-for-ad](https://github.com/worldbench/awesome-vla-for-ad) | AD VLA 目录 | survey 配套论文清单 |

### 5.2 规模-实时性速查

| 档位 | 代表 | 参数 | 推理特征 |
|---|---|---|---|
| 车端可实时（小模型） | ACT / Diffusion Policy(轻骨干) / DP3 / iDP3 / VAD-Tiny / BeT / VQ-BeT | 27M–80M | 10–50Hz 量级；机载算力可跑（iDP3 已实机验证） |
| 中档（需蒸馏/量化） | TinyVLA / MiniVLA / Octo / π0-FAST | 数百 M–3B | 消费级 GPU 实时；车载需压缩 |
| 大模型（云端/离线） | OpenVLA / π0 / GR00T N1 / Gemini Robotics / EMMA | 2B–55B | 车载实时不可行；用于数据合成、标注、离线规划或蒸馏教师 |

### 5.3 数据侧

- 机器人：Open X-Embodiment（RT-X/Octo/OpenVLA 共用）；遥操采集 = ALOHA / LeRobot 采集例程。
- 驾驶：nuScenes（UniAD/VAD/DriveVLM/Senna）、nuPlan、CARLA（LMDrive/CarLLaVA）、DriveX（Senna 预训练）、LangAuto 64K 指令片段（LMDrive）、SUP-AD（DriveVLM）、BDD-X（DriveGPT4）。
- 无标签视频预训练：Ego4D（GR-1）、自车行车记录仪视频（LAPA 思路）——**与你们「用存量行车数据 + 少量遥操数据」的预算结构一致**。

---

## 六、与你 L4 方案的对应关系

对照《VLA 在 L4 的可执行落地方案》的三个阶段与切入策略：

| 你方案里的动作 | 对应的论文/实现 | 备注 |
|---|---|---|
| 用遥操数据训练 VLA，低速（8-15 kph）替换人工遥操 | ACT / Diffusion Policy + LeRobot；GNM 铺导航底子 | 80M 级模型 50Hz，最小可行闭环，成本可控 |
| 与 two-stage 并存、可切回、逐步扩大 VLA 边界 | DriveVLM-Dual（量产车验证的混合架构）；Senna（VLM 高层 + E2E 低层解耦） | 这两篇是「并存而非替换」的现成论文证据，可直接引用进方案 |
| 遥操新增模式：切 VLA / 切 two-stage / 切遥操 | DriveVLM-Dual 的 fallback 设计；RoboCat 的自改进闭环 | RoboCat 提供「策略自己产数据迭代」的运营闭环依据 |
| 存量数据低成本预训练、跟随 LLM 发展 | LAPA（无标签视频潜动作预训练）；HPT/GR-1/GR-2；FAST tokenization | 对应你方案里「pretrain 跟随 LLM 发展、成本为早期 10-15%」的判断 |
| 指令化运营（场景指令/客服指令语言控制） | LMDrive（语言指令闭环驾驶 + LangAuto 基准）；CarLLaVA；Language2Drive（指令保真度评测）；EMMA（trajectory tokens） | 若 robovan 场景需要「语言下达临时指令」，这组是训练方法 |
| 车端实时与成本收敛 | TinyVLA/MiniVLA（小 footprint）；VAD-Tiny（9.3× 提速）；π0-FAST（扩散→自回归 5×） | 引用时注意：MiniVLA、π0-FAST 无 arXiv 论文，用博客链接 |
| 感知不再作为上游依赖（debug 辅助定位） | UniAD（规划导向）、VAD（矢量化约束）、EMMA（直接航点输出） | e2e 证据链；感知显式化与否的折中看 VAD 的矢量化约束设计 |
| 长期：通用 VLA 即 WAM（世界+动作模型） | GAIA-1（驾驶世界模型）、Gemini Robotics 1.5（推理交织）、Qwen-VLA（任务统一） | 第三阶段愿景的文献支撑 |

---

## 附：核实说明与未核实项

- **核实**：本报告通用 VLA/VA 条目由调研子代理逐篇 web_fetch arXiv 核对；AD 部分条目因子代理中途失败，由主代理经 arXiv 官方 API（id_list/title 检索）重新核对，剔除 10 余个错误 ID（如 EMMA 正确 ID 为 2410.23262、VAD 为 2303.12077、Gemini Robotics 为 2503.20020、TinyVLA 为 2409.12514、FAST 为 2501.09747）。
- **❗非 arXiv 来源**：MiniVLA（斯坦福博客）、π0-FAST（PI 博客，无独立论文）；「LangDrive」未检索到可靠论文条目，已由 LMDrive / Language2Drive 替代。
- **跟踪项（摘要未逐字核对）**：Language2Drive、VLDrive、ICR-Drive、SeedPolicy、RynnBrain 1.1（仅标题/年份经检索列表确认）。
- 建议下一步：按第六节映射，把 6 篇核心（LMDrive、Senna、DriveVLM、ACT、Diffusion Policy、LAPA）做深读（模式 A 精读），补进方案第 11-12 页的文献支撑。

---
更新时间: 2026-10-26 14:45
