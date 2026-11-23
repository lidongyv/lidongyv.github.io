# 具身智能 WAM（World Action Model）调研报告

> 版本 0.2（五轴表征 + 实机/仿真验证已补全） · 生成于 2026-10-26 · 来源：arXiv 原文（逐条核实，L1/L2/L3 分级标注）+ 两篇 2026 综述官方清单
> 目录：定义 → 论文谱系 → 实现方式 → **五类表征详解（输入/目标/动作/决策）** → 实机与仿真验证 → 资源

---

## 0. TL;DR

- WAM = 联合建模**未来状态与动作**的具身基础模型，目标是联合分布 p(o′, a | o, l)；预测的未来必须进入动作路径（产生/打分/训练动作），否则不算 WAM。
- 权威入口：OpenMOSS 综述 [2605.12090](https://arxiv.org/abs/2605.12090)（Cascaded/Joint 两族）、NUS 综述 [2606.20781](https://arxiv.org/abs/2606.20781)（三族 109 篇清单 + [交互站](https://world-action-models.github.io/)）、RCL 综述 [2609.16074](https://github.com/rcl-robotics/Awesome-World-Action-Models)。
- 实现路线两条：**视频生成骨干**（Wan2.2/Cosmos-Predict DiT + 动作专家）与 **JEPA/潜空间骨干**（冻结 V-JEPA 2/DINOv3 特征 + 未来潜预测器）。
- 2026 年共识趋势：**Dream less, Act more**——训练时联合预测未来，推理时尽量少生成未来（Fast-WAM、Rift、GigaWorld-Policy、THAW-VLA 等）。

---

## 1. 定义与边界

| 范式 | 建模对象 | 说明 |
|---|---|---|
| VLA | p(a \| o, l) | 反应式映射，不显式建模干预下世界演化 |
| 世界模型 WM | p(o′ \| o, a) | what-if 模拟器，不输出动作 |
| **WAM** | **p(o′, a \| o, l)** | 未来预测参与动作条件化，是端到端策略的一部分 |

边界规则（NUS 综述原文）：预测的未来必须被用来 produce / score / train / check 动作。「带辅助 future loss 的 VLA」「只在 RL 训练中用的模拟器」「推理前丢弃的未来头」均不算 WAM。合法形态三选一：①先预测后动作（predict-then-act）②预测后果给候选动作打分（score actions）③联合生成（joint prediction）。

**术语来源**：Meta 于 2025-11 发布 WAM 智能体（V-JEPA 2 世界模型 + 潜空间 CoT 搜索，[官方博客](https://ai.meta.com/blog/wam-world-action-model/)）；学术奠基视频-WAM 为 UniPi (2023)。

---

## 2. 论文谱系（详见第一轮调研）

### 2.1 综述/教程（必读 5 篇）

| 论文 | arXiv | 价值 |
|---|---|---|
| World Action Models: The Next Frontier in Embodied AI（复旦 OpenMOSS） | [2605.12090](https://arxiv.org/abs/2605.12090) | 首个系统梳理；Cascaded/Joint taxonomy；数据生态；评测协议 |
| World Action Models: A Survey（NUS，Dream Less Act More） | [2606.20781](https://arxiv.org/abs/2606.20781) | 109 篇三族清单；组件解剖四轴 |
| WAM 机器人向综述（RCL/MBZUAI） | [2609.16074](https://github.com/rcl-robotics/Awesome-World-Action-Models) | 2×2 象限；数据金字塔；control utility |
| From World Models to WAMs: A Concise Tutorial | [2607.00836](https://arxiv.org/abs/2607.00836) | 统一比较 World Labs / JEPA / Cosmos |
| 世界模型定义与路线图（上海 AI Lab） | 2607.06401 | WAM 是跨架构功能范式 |

### 2.2 源头主线（JEPA 谱系）

| 论文 | arXiv | 要点 |
|---|---|---|
| V-JEPA | [2404.08471](https://arxiv.org/abs/2404.08471) | 纯特征预测目标，无重建/负样本 |
| V-JEPA 2 | [2506.09985](https://arxiv.org/abs/2506.09985) | >100 万小时视频自监督；V-JEPA 2-AC 用 <62h 无标注机器人视频后训练潜动作世界模型，两实验室 Franka 零样本抓放 |
| Meta WAM 发布（2025-11） | [官方博客](https://ai.meta.com/blog/wam-world-action-model/) | V-JEPA 2 + 潜空间 CoT 搜索；Atari 100K SOTA。⚠️ arXiv 论文编号未核实 |
| V-JEPA Policy | [2609.37250](https://arxiv.org/abs/2609.37250) | 冻结 V-JEPA 2.1 + 未来潜预测器 + flow-matching 动作专家（0.9B/0.6B 可训） |
| JEPA-WAM | [2609.20277](https://arxiv.org/abs/2609.20277) | 文生图生成视觉指令 → V-JEPA 2.1 goal tokens |
| DAWN（驾驶域） | [2605.11550](https://arxiv.org/abs/2605.11550) | V-JEPA 潜 + DiT 动作 predictor-denoiser 环 |
| LeWAM | [2610.12818](https://arxiv.org/abs/2610.12818) | 端到端 JEPA：下一潜状态预测 + 动作流匹配联合训练 |

### 2.3 三大实现族代表（选自 NUS 109 篇清单）

- **Render-and-Decode（58）**：UniPi [2302.00111](https://arxiv.org/abs/2302.00111)、GR-1 [2312.13139](https://arxiv.org/abs/2312.13139)、GR-2 [2410.06158](https://arxiv.org/abs/2410.06158)、WorldVLA [2506.21539](https://arxiv.org/abs/2506.21539)、RynnVLA-002 [2511.17502](https://arxiv.org/abs/2511.17502)、Motus [2512.13030](https://arxiv.org/abs/2512.13030)、VideoVLA [2512.06963](https://arxiv.org/abs/2512.06963)、DreamZero [2602.15922](https://arxiv.org/abs/2602.15922)、Veo-Act [2604.04502](https://arxiv.org/abs/2604.04502)、X-WAM [2604.26694](https://arxiv.org/abs/2604.26694)、Pelican-Unify 1.0 [2605.15153](https://arxiv.org/abs/2605.15153)、WALL-WM [2606.01955](https://arxiv.org/abs/2606.01955)、DriveWAM [2605.28544](https://arxiv.org/abs/2605.28544)
- **Latent-Only（38）**：VPP [2412.14803](https://arxiv.org/abs/2412.14803)、UVA [2503.00200](https://arxiv.org/abs/2503.00200)、UWM [2504.02792](https://arxiv.org/abs/2504.02792)、Genie Envisioner [2508.05635](https://arxiv.org/abs/2508.05635)、Cosmos Policy [2601.16163](https://arxiv.org/abs/2601.16163)、LingBot-VA [2601.21998](https://arxiv.org/abs/2601.21998)、DiT4DiT [2603.10448](https://arxiv.org/abs/2603.10448)、**Fast-WAM** [2603.16666](https://arxiv.org/abs/2603.16666)、GigaWorld-Policy [2603.17240](https://arxiv.org/abs/2603.17240)、MotuBrain [2604.27792](https://arxiv.org/abs/2604.27792)、WAV [2604.14732](https://arxiv.org/abs/2604.14732)、τ0-WM [2606.01027](https://arxiv.org/abs/2606.01027)、ST-WAM [2607.28993](https://arxiv.org/abs/2607.28993)
- **Video-Generation-Free（13）**：FLARE [2505.15659](https://arxiv.org/abs/2505.15659)、DexWM [2512.13644](https://arxiv.org/abs/2512.13644)、PointWorld [2601.03782](https://arxiv.org/abs/2601.03782)、PALM [2601.07060](https://arxiv.org/abs/2601.07060)、LDA-1B [2602.12215](https://arxiv.org/abs/2602.12215)、LaWAM [2606.15768](https://arxiv.org/abs/2606.15768)、Audio-WM [2512.08405](https://arxiv.org/abs/2512.08405)

### 2.4 产业/平台线

NVIDIA（Cosmos 3 [2606.02800](https://arxiv.org/abs/2606.02800)、GR00T N2、Cosmos Policy）、智元（Genie Envisioner、GE-Sim 2.0、GE-Act 2.0 [2609.05588](https://arxiv.org/abs/2609.05588)）、生数（Motus/MotuBrain/Vidu S2）、蚂蚁灵波（LingBot-VA）、小鹏（XPACE [2609.17372](https://arxiv.org/abs/2609.17372)）、Dyna Robotics（Dyna-2，闭源）、昆仑万维（Riemann-1.0，闭源）、WSA₁ [2607.03941](https://arxiv.org/abs/2607.03941)、WLA [2606.05979](https://arxiv.org/abs/2606.05979)、WorldDiT [2607.23909](https://arxiv.org/abs/2607.23909)、Percept-WAM [2511.19221](https://arxiv.org/abs/2511.19221)（驾驶）。

---

## 3. 实现方式总览

### 3.1 组件解剖四轴（NUS 综述）

**Backbone**（视频 DiT / 视频 U-Net / VLM / 扩散策略 / JEPA 编码器）× **Substrate**（解码像素 / 潜特征 / 点轨迹 / 流 / 可供性图 / 价值图 / 音频 / 语义 token）× **Action coupling**（联合去噪 / 未来条件 KV / 逆动力学 / 打分）× **Deployment**（测试时想象 vs action-only、异步、量化）。

### 3.2 四条典型 recipe

1. **视频生成式 Joint WAM**（主流）：Wan2.2 / Cosmos-Predict 视频 DiT + 动作 DiT 组成 Mixture-of-Transformers，共享 VAE 潜空间做 flow matching 联合去噪；动作专家以视频专家中间特征/KV 为条件。例：Motus、DiT4DiT、Fast-WAM。
2. **AR 自回归**：视觉 token + 动作 token 统一序列自回归（WorldVLA、RynnVLA-002、WLA）。
3. **JEPA/潜空间路线**：冻结 V-JEPA 2 / DINOv3 编码器 → 未来潜预测器 + flow-matching 动作专家（V-JEPA Policy 0.9B）。
4. **Cascaded 生成 + IDM**：先生成未来视频/流/点轨迹，再逆动力学反解动作（UniPi → LVP → EVA；VPP feature-tap；Im2Flow2Act 流接口）。

### 3.3 训练配方共识

- 数据金字塔：互联网/第三视角视频（时序自监督）→ 第一视角人类演示（意图先验）→ 具身轨迹（动作接地）。
- 粗到细课程：无动作视频适配 → 人→机动作迁移 → 机器人微调（XPACE L1–L4；Riemann λ 0.1→0.9）。
- 未来表征要对控制有用：EgoWAM 对比显示 DINO 特征 OOD 泛化约 4×、3D flow 域内 +20–30%。
- 失败感知/因果训练：FACT 先动作后未来 + 掩码失败 rollouts 动作；Kairos 证明仅训动作专家掉 23.2 点。

### 3.4 部署趋势

Fast-WAM（去测试时想象 4× 加速）、Rift（anticipation token 一次性写未来 K/V）、GigaWorld-Policy（视频生成因果可选 9× 加速）、Rolling-WAM（滚动去噪 215ms）、MotuBrain 11Hz、THAW-VLA 32ms。

---

## 4. 五类表征详解（核心）

> 本章由三个子代理逐篇抓取 arxiv.org/html 原文核实（2026-10-26）。来源级别：L1=摘要页；L2=HTML 正文（部分小节）；L3=全文核实。凡未核实处均标注「原文未写（已获取来源）」或「未核实」，不臆补。

### 4.1 输入表征：三条路线

**① 离散 VQ token（全模态共享词表）**

- WorldVLA [2506.21539](https://arxiv.org/abs/2506.21539)（L2）：图像用 VQ-GAN tokenizer（压缩比 16，codebook 8192；256²→256 tokens，512²→1024 tokens）；文本 BPE tokenizer；动作离散化为 7 tokens（3 相对位置+3 相对角度+1 夹爪，每维 256 bins）；总词表 65,536；序列 `[BOS]{text}[BOI]{image}…[EOI]×M [EOS][BOA]{action}…[EOA]×K`；backbone 初始化自 Chameleon；动作块生成用「屏蔽先前动作」注意力掩码。
  - 原文（Sec 3.2）："The compression ratio of the image tokenizer is 16 and the codebook size is 8192. The image tokenizer generates 256 tokens for 256×256 images and 1024 tokens for 512×512 images."；"All texts, actions, and images are discretized into tokens and are trained under the autoregressive manner."

**② 连续 VAE latent + 视频 DiT（2025–2026 主流）**

- Fast-WAM [2603.16666](https://arxiv.org/abs/2603.16666)（L3，另见[精读笔记](reading/fast-wam.md)）：多相机图拼接为单图 → Wan2.2 预训练 video VAE latent；语言=T5 编码器 cross-attention 注入所有 token；训练时三组 token（首帧 clean latent 锚点 / 未来帧 noisy latent / 动作 token），结构化掩码禁止动作分支 attend 未来帧；未来帧 4× 时间降采样（9 帧/chunk）。代码 config：`libero_uncond_2cam224`、`robotwin_uncond_3cam_384`。
- Motus [2512.13030](https://arxiv.org/abs/2512.13030)（L2，Sec 9.1 截断）：三专家 MoT——生成专家 Wan 2.2 5B（视频在 latent token 空间建模）；理解专家 Qwen3-VL-2B（文本经其进入，输入取自 VLM 末层对应 token）；动作专家同 Wan 深度；Tri-model Joint Attention 融合；视频帧率为动作 1/6（Action-Dense Video-Sparse）。潜动作：DPFlow 光流→转 RGB→DC-AE 编码为 4×512 tokens→投影为 14 维向量。
- NavWAM [2606.13494](https://arxiv.org/abs/2606.13494)（L2）：Cosmos Predict2 因果 VAE 编码观测进「9 帧 latent canvas」。
- Act2Goal [2512.23541](https://arxiv.org/abs/2512.23541)（L2，VAE 型号未写明）：当前观测与 goal image 经视频编码器 VAE 压缩为 latents 后拼接。

**③ 冻结自监督/预测特征 token（2024 早期与 2026 回归阵营）**

- GR-1 [2312.13139](https://arxiv.org/abs/2312.13139)（L2）：MAE 预训练 ViT（CLS+patch tokens 经 perceiver resampler 降量）；语言=冻结 CLIP text encoder 每 timestep 重复注入；机器人状态（6D 位姿+二值夹爪）线性层编码；Ego4D 视频预测预训练。
- V-JEPA Policy [2609.37250](https://arxiv.org/abs/2609.37250)（L2）：冻结 V-JEPA 2.1 ViT-L（0.3B），每相机两帧 context（tubelet size=2），输出时空 tubelet token（非图像、非重建），3D RoPE；条件序列=冻结 T5-XXL 文本 token + 1 个 proprio state token；predictor 0.5B 单次前向预测未来 latent；动作专家（flow matching DiT，0.1B）由 predictor 的 layer-wise context K/V 经 MoT 共享注意力条件。
- JEPA-WAM [2609.20277](https://arxiv.org/abs/2609.20277)（L2）：goal 编码用冻结 V-JEPA 2.1 ViT-B/16（80M）→ 8×8 池化 → 64 个 goal tokens。
- LaWAM [2606.15768](https://arxiv.org/abs/2606.15768)（L2）：观测经冻结视觉基础模型（DINOv3 等）编码为特征 u=f_ψ(o)，不重建像素。
- PALM [2601.07060](https://arxiv.org/abs/2601.07060)（L2）：CLIP 文本 + MAE 图像（Perceiver Resampler）+ 状态 MLP + GPT-2 风格主干 + 4 个 affordance subquery。
- GR-MG [2408.14368](https://arxiv.org/abs/2408.14368)（L2）：goal image 经 MAE tokenize 置于输入序列前端。

**趋势**（仅基于已核实材料）：2024 早期以冻结特征+像素重建为主（GR-1/GR-MG/UniPi）；2025 转向复用视频生成 VAE latent（Motus/Fast-WAM/NavWAM）；2026 出现回归冻结预测特征的阵营——V-JEPA Policy 论证「无需继承完整视频生成器，预测 latent 即可支撑 WAM」，LaWAM/JEPA-WAM 主张「像素重建冗余、latent 更紧凑抗外观扰动」。

### 4.2 目标表征：四种选项

| 选项 | 论文 | 具体做法（原文依据） |
|---|---|---|
| **纯文本指令** | WorldVLA、GR-1、Motus、Fast-WAM、UniPi | 仅语言条件，无 goal image（各论文正文未提 goal 图） |
| **文本→生成 goal image→再编码** | GR-MG、JEPA-WAM | GR-MG：InstructPix2Pix 按「当前观测+文本+进度」生成 subgoal 图，进度文本注入（如 "pick up the red block. And 60% of the instruction has been completed."，10 档离散化，T5-Base 编码）；MAE tokens 前置供全序列 attend。JEPA-WAM：冻结 DreamLite 0.39B 离线按文本生成任务完成图（每指令 bank，默认 K=1024 张），复制两帧经 V-JEPA 2.1 → 8×8 池化 → 64 goal tokens：`Z_ℓ^(k) = P_ψ(Pool_{8×8}(E_J([g_ℓ^(k), g_ℓ^(k)])))`，条件序列 `C = [C_ℓ; P_s(s); Z_ℓ]` |
| **原生 image-goal 进 latent** | NavWAM、Act2Goal | NavWAM：目标位置拍摄的图经 causal VAE 作 latent canvas 中 goal frame；Act2Goal：`z_pred = f_θ(z_t, z_g, ε)`（当前观测与 goal 的 VAE latent 拼接），MSTH 对远端帧按对数间隔稀疏采样锚定全局一致性 |
| **latent subgoal / 未来 latent** | LaWAM、V-JEPA Policy、PALM | LaWAM：`p(a, û_T, ẑ | o,l) = p_θ(ẑ|o,l)[Policy Prior] p_ω(û_T|u,ẑ)[LaWM] p_η(a|o,l,u,û_T)[Action Expert]`——潜动作解码出 latent visual subgoal û_T；V-JEPA Policy：目标=未来帧 V-JEPA latent（L1 回归）；PALM：4 类 affordance（global 对象掩码/local 接触热图/spatial 放置候选/dynamic 运动轨迹）+ 连续子任务进度 p∈[0,1] |

**已核实的利弊线索**：JEPA-WAM 指出文本-动作标注稀疏重复会诱发 shortcut（指令接地差），goal image 可消歧但其外观/布局与当前场景不符，需 JEPA 编码「去外观、保语义」；Act2Goal 主张语言粒度不足、视觉 goal 无歧义且长程需中间视觉状态；Fast-WAM 实证「推理时想象未来」非必需——训练期视频共训才是主要收益；LaWAM 主张单步 latent subgoal 最省（相对像素级 WAM 快至 24×）。

### 4.3 动作表征：四类选项

**① 连续低维控制量（最主流）**

- GR-1（L2）：**连续** ΔXYZ + Δ欧拉角（6 维）+ 二值夹爪，[ACT] token 过线性层直接回归（Smooth-L1 + BCE），无离散化；每时间步一个动作（未做 chunk）。
  - 原文 §3.2.3："Since the arm action is continuous, we use Smooth-L1 loss L_arm for training. Gripper actions are optimized using Binary Cross Entropy (BCE) loss."；§4.1："we train GR-1 to predict delta XYZ positions and delta Euler angles for arm actions and binary gripper actions."
- DiT4DiT（L2）：**连续动作轨迹 + flow matching**。Action DiT（GR00T-N1 适配）：本体状态嵌入 + 噪声动作轨迹 + 可学习 future tokens，线性头输出动作序列速度场；以视频 DiT（Cosmos-Predict2.5-2B）**中间去噪隐状态 h^{τf}**（前向钩子提取，固定时间步 τf 保证确定性）经交叉注意力条件；τa~Beta(α,β) 把训练容量偏向关键控制阶段。
  - 原文 §4.2："a forward hook mechanism intercepts intermediate hidden activations in flow timestep τf—either from a specific deep transformer block or averaged across all layers."
- τ0-WM（L2，代码实锤）：连续动作 chunk；0.5B Action DiT 与 Wan2.2-TI2V-5B 视频骨干匹配层交叉注意力耦合，联合 flow matching（λz=λa=1）；H_a/H_v 正文未写，代码=action_chunk 33 / 视频 chunk 9；动作空间见 §4.3.1。
- 注意：DiT4DiT 与 τ0-WM 的关节/EE 维度、chunk 长度在已获取部分**未写明**。

**② 离散/潜动作（桥接无标签视频）**

- LAPA [2410.11758](https://arxiv.org/abs/2410.11758)（L2）：VQ-VAE 目标学习帧间**离散潜动作** → 潜 VLA 预测 → 小规模机器人数据微调把潜动作映射为真实动作。潜动作离散码与微调动作空间：原文记潜动作码 8^4、微调后 256^7 ΔEE（ar5iv 渲染错乱，③线已按 LaTeX 语义还原并注明）。
  - 原文（摘要）："We first train an action quantization model leveraging VQ-VAE-based objective to learn discrete latent actions between image frames, then pretrain a latent VLA model to predict these latent actions… and finally finetune the VLA on small-scale robot manipulation data to map from latent to robot actions."
- V-JEPA 2-AC（L2，§3.1 核实）：潜动作空间=**7 维 ΔEE @4fps ×15 步**，block-causal 潜表征预测（<62 小时无标签 DROID 视频后训练）；**§4 image-goal 规划的动作采样机制仍未取到**（列为首要待核实项）。
- Motus（L2，代码实锤 thu-ml/Motus）：潜动作 14 维（DPFlow 光流→RGB→DC-AE 4×512 tokens→投影 14 维，rectified flow）；**真实执行动作=14 维 qpos（关节位置）**，`take_action(action_type='qpos')`，min/max 归一化到 [0,1]。
- WAV（L2）：潜动作 token z_act 与视频特征、价值嵌入交叉注意力逐块精化后解码。

**③ 像素/几何锚定动作（跨 viewpoint/embodiment 迁移）**

- Action Images [2604.06168](https://arxiv.org/abs/2604.06168)（L2）：7-DoF 动作 → 多视角 2D pixel-grounded「action images」动作视频（**3 通道多视角热力图视频 + 几何解码恢复 7-DoF**），视频骨干直接当 zero-shot policy。原文（摘要）："we translate 7-DoF robot actions into interpretable action images: multi-view action videos that are grounded in 2D pixels and explicitly track robot-arm motion."
- PointWorld [2601.03782](https://arxiv.org/abs/2601.03782)（L2）：动作=3D 点流（**H=10@0.1s，MPC-MPPI 再转回 EE 动作**）。原文（摘要）："By representing actions as 3D point flows instead of embodiment-specific action spaces (e.g., joint positions)…"

**④ 其他**：UVA（联合潜+解耦双头，推理绕过视频生成）；UWM（时间步切换见下）；DreamZero（48 步 chunk 见下）；XPACE（动作空间摘要未写）。

### 4.3.1 代码对质结论（2026-10-26 官方仓库取证，详见 [06-code-forensics.md](reading/06-code-forensics.md)）

论文正文未写明的动作空间，已全部用官方代码仓库实锤：

| 论文 | 动作空间（代码事实） | 证据文件 |
|---|---|---|
| **DiT4DiT** | **绝对关节位置**（`action_type: joint_position`）；LIBERO `action_dim: 8, action_horizon: 8`、`state_dim: 16`；G1 真机 `action_dim: 32, action_horizon: 16` | `DiT4DiT/config/libero/dit4dit_libero.yaml`、`config/real_robot/dit4dit_g1.yaml` |
| **τ0-WM** | 预训练=**相对 EEF 6d、双臂 20 通道**（xyz+6d+2 夹爪）；后训练=**绝对关节 7 维**（6+1）；`action_chunk: 33`、视频 `chunk: 9` | `sii-research/tau-0-wm` README「Action Space」、`configs/deployment/tau_pretrain_rela_eef6d.yaml`、`configs/tau_model/posttrain_taco_play_abs.yaml` |
| **UWM** | `action_len: 16`；RoboMimic 7 维（pos+quat+gripper）、DROID 10 维（3+6+1）；四模式切换=`DualTimestepEncoder`：边缘采样时把另一模态 t 固定为 timesteps[0]/[-1]，联合采样共用 t | `WEIRDLabUW/unified-world-model` `configs/model/uwm.yaml`、`models/uwm/uwm.py` |
| **DreamZero** | `action_horizon: 48`（`num_frames: 49`）；动作=**joint_position + gripper_position**（DROID `relative_action: true`）；网络维 pad 到 `max_action_dim: 32`（本体 DROID 7 / AGIBOT 22 / YAM 14） | `dreamzero0/dreamzero` `configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`、`agibot_relative.yaml`、`action_head/wan_flow_matching_action_tf.yaml` |
| **Motus** | 见上（14 维 qpos） | `thu-ml/Motus` |
| **GR-1（补充）** | 7 维相对末端位姿（6 臂 + 1 夹爪 ±1）；动作归一化 stats 与视频 Δt **官方仓未发布→未能获取** | `bytedance/GR-1` `logs/configs.json`、`models/gr1.py` |
| **Fast-WAM（补充）** | RoboTwin=14 维**原始 qpos**（`take_action(action_type="qpos")`，无 xyz+quat/rpy 拆分）；仓库位姿代码用四元数 | `yuantianyuan01/FastWAM` `configs/data/robotwin.yaml`、`deploy_policy.py` |

**⚠️ 未解决项**：仅剩 GR-1 的动作归一化数值与视频预测间隔 Δt（官方仓未含数据处理代码）。

**趋势**：从连续低维向「潜动作 / 几何-像素锚定」多元化演进，后者主要用于从无标签视频桥接与跨本体迁移。

### 4.4 决策表征：五种结构

| 结构 | 论文（来源级别） | 机制（原文依据，④线核实） |
|---|---|---|
| **直接回归/联合去噪**（多数） | GR-1、DiT4DiT、UVA、UWM、DreamZero、Action Images、XPACE 策略侧、Fast-WAM | 单次前向直出动作；Fast-WAM 明确省略外层 AR rollout（"we focus on single action chunk generation and omit the outer auto-regressive loop"） |
| **潜空间迭代推断（MPPI 风格）** | WAV（L3 至 §4.3） | 价值模块 flow matching 回归**累积折扣回报** v¹=Σγ^i·R（rule-based dense reward）；每轮采 M 组视频潜噪声 × 每组 N 个价值潜噪声，**top-K1/K2 精英更新高斯噪声分布**，K 轮后解码动作。Algorithm 1、Lemma 4.1/Prop 4.2/Cor 4.3 全文核实；Eq4/5 与 M/N/K/K1/K2 数值在 §5.3/附录未取到。原文 §4.3："Inspired by Model Predictive Path Integral (MPPI) control, we perform iterative latent-space trajectory optimization during inference…" |
| **best-of-N 评分 + 修正** | τ0-WM（L3 完整） | propose-evaluate-revise：①VAM 采 N 个候选 chunk；②**RCS 重去噪一致性分数**（Eq6-11 全文核实）排序；③max RCS<γ 时 ACVS 预测**密集任务进度轨迹**（子任务级标签+MC 传播）选最优；④**LAR** 局部动作重查询；Algorithm 1/2 全文核实；N/γ/K 数值在附录未取到 |
| **beam/树搜索** | VLP（L3）、3D-ALP（L3 全文） | VLP：VLM policy+value 双模型 **beam 爬山**（Algorithm 1 全文；A/D/B/H 数值在 App A.5 未取到）；3D-ALP：**MCTS D=2, B=4, 2s 规划预算, c=0.02**（渲染瓶颈 2400ms/帧） |
| **触发式测试时评估** | DreamAvoid（L3） | Trigger（β=5, γ=0.5, 1Hz）判定关键阶段 → SDE Proposer 采候选 → **Dream Evaluator（8 候选/40ms/20-40-40 采样，混合成功-失败-边界数据训练）**选优；σ 值在 Table 5 未取到 |
| **MPC 集成** | PointWorld（L3） | MPPI 全文：K 扰动/三次样条/指数权重/代价函数 Eq2/H=10/0.1s；K/T/β 数值在 App A.6.1 未取到 |
| **时间步切换多任务** | UWM / UVA（L1） | 控制各模态扩散时间步，同一模型切 policy/FDM/IDM/视频生成（细节摘要未写） |
| **外环规划（VLM+WM）** | WAP（L3）、Cosmos 3（L2+） | WAP：Algorithm 1 全文 + Theorem 1-3 核实；Cosmos 3：**Reasoner 具身 CoT 与 2D 轨迹规划正文未取到**（章节已定位），双塔单向衔接以 NVIDIA 官方博客引文补充（次级来源，已注明） |

**未获证实项（如实标注）**：显式搜索类已获 L3 实例（VLP beam 爬山、3D-ALP MCTS、WAP 外环规划）；但 **Meta WAM 的 CoT 潜轨迹搜索**（ai.meta.com 博客四路取文均网络失败）仍未获一手证实，不可作为依据。趋势：决策侧从单次前向转向「测试时计算」（采样-评分-修正、潜空间推断、MPC、触发式评估）。

---

## 5. 实机与仿真验证

> 来源级别同 §4 说明。关键背景：CALVIN、NAVSIM 在本批 11 篇中均未使用。

### 5.1 仿真基准清单（含协议细节）

| 基准 | 协议要点（原文依据） | 已核实结果 |
|---|---|---|
| **LIBERO** | 4 套件（Spatial/Object/Goal/Long）×10 任务×500 demos（=每任务 50）；**在基准 demos 上标准监督训练（非 zero-shot）**；Fast-WAM 口径：20k 步、2000 trials/40 任务（×50 episodes）；Cosmos Policy 口径 6000 trials | Fast-WAM 97.6（无具身预训练）；Cosmos Policy 98.5；DiT4DiT 98.6；V-JEPA Policy 97.25 —— **已近饱和** |
| **LIBERO-Plus** | 7 个受控偏移轴（viewpoint/initial state/language/lighting/texture/noise/layout）逐轴报 SR | V-JEPA Policy 79.25，DROID 视频-指令对预训练 predictor 后 91.50 |
| **RoboTwin 2.0** | 50+ 双臂任务；**clean + randomized（重度场景随机化）两档**；100 trials/任务；训练常混 2500 clean + 25k randomized demos | Fast-WAM 91.8；WSA₁ 93.3；MotuBrain 95.8(clean)/96.1(rand)（均为「平均」口径，具体定义/trials 部分未核实） |
| **RoboCasa(-GR1)** | 24 厨房任务人形操作；**demos 数量差异大（50–3000/任务）**，跨论文比较须对齐 | Cosmos Policy 67.1（24 任务×50 trials×3 seeds=3600 trials，官方项目页核实）；V-JEPA Policy 50.92；DiT4DiT 50.8 |
| **PolaRiS / Genie Sim 3.0** | DreamZero 零样本：约 500h 真机数据训练、不用其 10k h 仿真数据；Genie Sim 3.0=100 任务；指标=task progress | 2× VLA（GR00T N1.6/π0.5）——真机基准上的对比 |
| **MuJoCo 自建** | 3D-ALP：Franka Panda 5 步顺序 reach（步 4–5 目标被遮挡，需空间记忆）；MCTS D=2, B=4，2s 规划预算；30 episodes×3 seeds | 记忆步 SR 0.650±0.109 vs greedy 0.006±0.008；**无真机**（原文："Real-robot validation… is future work"） |

### 5.2 真机平台与评测惯例

- **平台**：Galaxea R1 Lite（Fast-WAM 毛巾折叠）、AgiBot G1 移动双臂 + Franka + YAM（DreamZero）、Franka ×2 实验室（V-JEPA 2-AC 零样本）、ALOHA（Cosmos Policy）、Unitree G1（DiT4DiT）、IRON-R01-1.11（XPACE）、TianJi Marvin（V-JEPA Policy）。
- **指标四类**：成功率（XPACE 每任务 20 trials）；**平均任务进度**（DreamZero，与 SR 不可混比）；**平均完成时间**（Fast-WAM 明确理由："completion time is therefore as important as success rate in assessing policy quality"——防「慢速试错成功」虚高）；延迟/控制频率（190ms、7Hz、11Hz）。
- **关键已验证数字**：Fast-WAM 真机 w.o. video co-train 掉到 **10%** 成功率、延迟 190ms vs IDM 810ms；XPACE IRON 三任务平均 **68.3%** vs DreamZero 40.0 / GR00T 6.7（**官方项目页一手核实**：任务=香蕉拣放/倒水/叠碗，20 trials/任务，matched resets，进度=里程碑均分；recovery DAgger **61.7→86.7** 为**另一任务套**香蕉/倒水/可乐 + 8% 合成恢复数据——两数字不可直接比较）；Cosmos Policy ALOHA 真机 4 任务 101 trials/方法 93.6；V-JEPA Policy 真机 **45%→80%**（2 任务×20 trials：11/20→15/20、7/20→17/20；DROID 无标签视频预训练后，②线一手核实）；V-JEPA 2-AC 零样本两实验室 Franka image-goal 抓放（**成功率数值未能核实**）。
- **扰动/鲁棒性**：多在仿真做（RoboTwin randomized、LIBERO-Plus 7 轴）；真机侧多为新环境/新物体泛化测试，**未见域随机化 sim2real 专门处理**（已核实部分原文未写）。

### 5.3 评测坑（仅基于已核实材料）

1. **全部提出方自评**：11 篇均为 arXiv 自报数字；仅 Cosmos Policy 有 ICLR 2026 同行评审版本；无第三方复现数据。
2. **基线口径不可比**：Fast-WAM 主表把「有/无 embodied pretraining」基线混排（论文靠分组比较补救）；RoboCasa 各法 demos 数 50 vs 300 vs 3000；MotuBrain「WorldArena 最强 EWMScore」为自建口径。
3. **指标口径差异**：DreamZero 用 task progress 而非 SR；LIBERO「平均」=四套件平均，跨论文同权性需查原文。
4. **XPACE 两处混淆**（数字已升至官方项目页一手，混淆仍在）：68.3% 与恢复消融 61.7→86.7 用不同任务套件（叠碗 vs 可乐）；before/after 未分离额外训练时长收益；对比 DreamZero/GR00T 时 XPACE 额外做了视频适应。
5. **披露不全/截断风险**：真机 trials 数、扰动细节普遍披露不足（WSA₁、DiT4DiT、MotuBrain、V-JEPA 2 真机部分标「未核实」；Cosmos Policy ALOHA 逐任务图未取到，trials 总数 101/方法已核实）；引用具体数值前应回 PDF 实验节逐表确认。

---

## 5.4 「论文 × 五轴」总表（五线统一核对后）

> 单元格为极简要点；细节与 verbatim 引文见 [精读索引](reading/README.md) 各线文件。来源级别：L3/L2/L1 同前。

| 论文 | ①输入 | ②验证 | ③动作 | ④决策 | ⑤目标 |
|---|---|---|---|---|---|
| Fast-WAM | 多相机拼单图→Wan2.2 VAE latent + T5 + 1 proprio token（代码实锤） | LIBERO 97.6 / RoboTwin 91.8 / 真机毛巾折叠 190ms | 7D ΔEE+夹爪（LIBERO）/14D（RoboTwin），h=32，FM | 单次前向直出（无测试时想象） | 纯文本 |
| WorldVLA | VQ-GAN 离散 token（codebook 8192） | LIBERO 81.8(512) | 7 tokens×256 bins，AR chunk K=5/10 | 并行 action 掩码 AR | 纯文本 |
| GR-1 | MAE ViT + CLIP + 6D 位姿状态 | CALVIN 94.9% | ΔXYZ+Δ欧拉+二值夹爪，horizon=1，线性回归 | 因果 AR 直出 | 纯文本 |
| Motus | Wan2.2 latent + Qwen3-VL 文本 | RoboTwin 87.8 / LIBERO 97.7 | 14 维光流潜动作，rectified flow | 联合去噪（UniDiffuser） | 纯文本 |
| V-JEPA Policy | 冻结 V-JEPA 2.1 ViT-L tubelet + T5 + 1 状态 token | LIBERO 97.25 / Plus 79.25 / 真机 45→80% | 0.1B DiT FM 动作块 | predictor K/V → 动作专家 | 未来 V-JEPA latent 回归 |
| JEPA-WAM | Fast-WAM 骨架 + V-JEPA 2.1 goal tokens | 真机三设定 87.3/74.5/80.9 | 继承 Fast-WAM | 继承 Fast-WAM 方向掩码 | 生成图 bank→64 tokens |
| NavWAM | Cosmos causal VAE 9 帧 canvas | go stanford / Diablo 真机 | action chunk+状态+价值帧联合 | policy-only 推理（无 CEM） | image-goal frame（无语言） |
| LaWAM | 冻结 DINOv3 特征 | LIBERO 98.6 / 187ms | Alternate-DiT FM | prior→LaWM→expert 三段 | latent subgoal û_T |
| PALM（VLA, CVPR26） | CLIP+MAE+MLP 状态 | LIBERO-LONG 91.8 | DiT 动作头 | GPT-2 联合解码 | affordance 4 类 + progress |
| UniPi | 首帧 + T5 | 组合任务/CLIPort | 逆动力学从生成视频反解 | 视频扩散规划（open-loop） | 文本（层级规划） |
| Act2Goal | 视频 VAE latent | Genie Envisioner 系 | WM 同构窄 DiT | flow matching | image-goal + MSTH 稀疏采样 |
| GR-MG | MAE goal tokens 前置 + CLIP | GR-1 基座 | cVAE 轨迹头 | — | 生成 subgoal + 进度文本 |
| CoT-VLA | VILA-U | LIBERO | 短动作序列 AR | 视觉 CoT 两步 | 自回归未来帧作视觉目标 |
| SWEET | FLUX Kontext 关键帧 | 40× 快于 Wan2.2 rollout | goal-conditioned DDPM 扩散策略 | 相邻帧对条件 | 图像编辑稀疏视觉计划 |
| DiT4DiT | Cosmos-Predict2.5-2B | LIBERO 98.6 / G1 真机 | 连续轨迹速度场 τa~Beta（维度未写） | 视频潜→动作去噪（τf 钩子） | 文本 |
| τ0-WM | Wan2.2-TI2V-5B + 0.5B Action DiT | RoboTwin 等 | 连续 chunk FM（H 未写） | propose-evaluate-revise（RCS/ACVS/LAR） | 文本 |
| V-JEPA 2-AC | V-JEPA 2 潜空间 | 两实验室 Franka 零样本（数值未核实） | 7D ΔEE@4fps×15 潜动作 | block-causal 潜预测 + image-goal 规划（§4 未取到） | image goal |
| Action Images | 视频骨干原生像素 | — | 3 通道热力图视频 + 几何解码 | zero-shot 直出 | — |
| PointWorld | RGB-D 点云 + DINOv3 特征 | Dex 任务 | 3D 点流 H=10 | MPC-MPPI（0.1s/rollout） | 点流目标 |
| UWM | 单 DiT | RoboMIND 等 | 动作扩散（维度未写） | 时间步 T/0 切四模式 | — |
| DreamZero | 14B 视频扩散 | PolaRiS/Genie Sim 零样本 + 真机 2× VLA | 48 步@30Hz chunk（维度未写） | 联合去噪 7Hz 闭环 | 文本 |
| XPACE | 共享因果视频 Transformer | IRON 真机 68.3→86.7 | 动作 chunk（摘要级） | 联合预测 + simulator 自改进 | 文本 |
| WAV（NeurIPS26） | Genie Envisioner 特征 | 仿真+真机 | 潜动作 token | MPPI 式潜空间迭代推断 + 价值 flow | — |
| Cosmos Policy | Cosmos-Predict2-2B | LIBERO 98.5 / ALOHA 93.6 | 动作=潜帧 | 单模型多 I/O | 文本 |
| 3D-ALP | 3D 世界模型 | MuJoCo 0.650 vs 0.006（无真机） | — | MCTS D=2 B=4 | — |
| Cosmos 3 | MoT Generator | — | policy/FDM/IDM 同 checkpoint | Reasoner CoT+2D 轨迹（正文未取到） | 文本 |

---

## 5.5 第二波扩展要点（20 篇，详见 reading/02b、03b 文件；01b 线待补）

**输入/目标（01b 线，10 篇全 L3，全文已本地化存档）**：
- 目标表征从「语言指令」向「未来视觉/未来表征」迁移：GR-2/VPP（语言为主，未来视频只是训练期信号）→ UVA（joint latent 统一）→ LingBot-VA（推理期显式生成未来观测作逆动力学条件，部分去噪 s=0.6）→ Rift（**未来 K/V cache 成为目标表征本身**，"One fixed final-clean K/V cache nearly preserves execution"）→ Pelican-Unify（语言化 CoT 压缩为 loop state z 再驱动联合生成）→ WALL-WM（事件级目标 + Staircase 潜 CoT）。
- 「目标=未来」的轻量化收敛：GE-Act 把未来视频降为 1 步 denoising 缓存；Rift 一次前向 CachePrefill——延迟从 rollout 法的 780–2270ms 降到 248–420ms（Rift Table 1/3）。
- 目标来源多元化：人类视频作目标载体（WAM-TTT 以 fast-weight 记忆吸收人视频；HOST 唯一做跨具身目标翻译：human video→robot future obs→action）；事件描述（WALL-WM）。
- 输入侧：UVA 历史 VAE latent+动作 chunk 重复 M；LingBot-VA causal VAE latent N=192/帧 + 动作 MLP token 交织（τ=4）；WALL-WM 多视角 + 事件 caption；Pelican 四模态共享嵌入器。
- **共同缺口（10 篇原文均未覆盖）**：除 Rift 的 cache 干预外，无一篇做「语言目标 vs 未来视觉目标」的贡献剥离；已记录口径冲突（LingBot 92.0 vs 92.9）与渲染伪影修正（GE 2,967h/200ms）。


- **连续动作 chunk + flow matching 已是绝对主流（10/10）**；唯一引入离散 codebook 的 LAWA 测试时也明确不投影（"do not apply nearest-neighbor projection"）。
- 新写明/实锤的动作空间：MotuBrain **10-D**（相对 EEF：位置 3+6D 旋转+夹爪 1）；OpenWAM 统一 **80-D**（构成未写）；Rolling-WAM G1 **78-D**（64 维 SONIC 潜运动+每手 7 维）；THAW-VLA RoboCasa **29-D** 双臂；Kairos 代码实锤 **RoboTwin 32 步×14 维 / LIBERO-Plus 32 步×7 维**。仍未写：GigaWorld-Policy、LAWA、DreamWAM、FACT（d_a）、X-WAM（维度数值）。
- 「动作高频、视频低频」成显式共识（MotuBrain f_va·τ、X-WAM K/H=4×、FACT H=48 动作 vs 4 未来帧、Rolling-WAM chunk 级错峰）。

**决策表征（03b 线）**：
- 重心从「是否生成未来」转向「信息流方向」：GigaWorld/FACT/LAWA 用因果掩码禁止未来视频→动作泄漏；OpenWAM 把四种 world-action 掩码模式做成受控变量（mutual / action-sees-video / video-sees-action / isolated）。
- 「不放弃未来想象、只压缩它的表征」：LAWA 潜意图序列、THAW-VLA 离线蒸馏（32ms/0.8B）、DreamWAM 四视图未来（RGB/光流/几何/语义）、Rolling-WAM 滚动去噪（215ms）。
- 「坏动作的后果」进入决策：FACT 失败 rollouts 掩动作损失 + [0,1] 进度值头做候选打分；Kairos 以 regret（条件期望物理代价）为设计准则。

**验证（02b 线，10 篇）**：MotionWAM **全文无仿真评测**（原文未写），G1 九任务 76.1 vs 43.9（+32.2pp 绝对）；WSA₁ README 口径 RoboTwin-Rand 93.1、真机 SR 80.52（与 abs 的 93.3 口径漂移，注意）；WLA 90.02=Rand ✓、98.6=LIBERO Avg ✓、56.5=RMBench ✓（Table 2 全文）；DreamZero 项目页真机 62.2/27.4/39.5/49/81%、80 rollouts/ckpt×4 机器人；XPACE 部署协议 10Hz/16 步计划（§3.2 verbatim）；Rolling-WAM RoboTwin 93.3、G1 85.0、延迟 215 vs 978/548/296/285ms。

**新增评测坑（02b 线，节选）**：π0 的 LIBERO 平均分在 Rolling-WAM 记 94.1、WLA 记 94.2（同为 94.15 的相反舍入）；"nearly doubling"=乘法、"over 30%"=加法 pp，同词不同意；WLA 正文 17.25% vs 表 17.3% 打架；效率口径互不兼容（215ms A100 稳态 vs 4.9Hz A100 chunk 率 vs 0.6s GB200 带 cache vs 40ms RTX5090）；20/30/50/100 trials 不同口径 CI 宽度不可比。

---

## 6. 资源

- 论文清单：https://github.com/world-action-models/awesome-world-action-models （109 篇 + 每周引用更新）
- OpenMOSS Awesome-WAM：https://github.com/OpenMOSS/Awesome-WAM
- RCL Awesome-WAM：https://github.com/rcl-robotics/Awesome-World-Action-Models
- 中文细读（33 篇 + 模型规格横向对比）：https://zhuyun97.github.io/embodied-ai-learning/wam/
- 开源实现：facebookresearch/vjepa2、Mondo-Robotics/DiT4DiT、LaWAM（LeRobot 原生）、DreamWAM、OpenWAM、UnifoLM（宇树）、le-wam.github.io

## 7. 来源与待核实项

- 所有 arXiv 编号均经逐条抓取核实（2026-10-26）。
- Meta WAM 原论文 arXiv 编号多次检索未命中，未编造；**其官方博客与 ai.meta.com 直连失败，CoT 潜空间搜索细节未能核实，不可作为依据**。
- 榜单数字多为提出方自评（⚠️），第三方统一评测少；§4/§5 各条目已标注 L1/L2/L3 来源级别。
- 因 HTML 长文截断而留待回查的点：Motus Sec 9.1（VAE latent 字样）、Act2Goal Sec V-A（VAE 型号）、PALM Dynamic affordance 细节句、Fast-WAM 动作空间维度与 λ 取值、τ0-WM H_a/H_v、V-JEPA 2-AC 潜动作学习目标。
- 环境限制（如实记录）：沙箱禁止 shell 出网与 pip 安装，本地 PDF 下载与抠图未完成；精读笔记采用 ar5iv/arXiv HTML 全文 + 官方代码仓 README 取证。

> 版本 0.2 · 更新时间: 2026-10-26 19:21
