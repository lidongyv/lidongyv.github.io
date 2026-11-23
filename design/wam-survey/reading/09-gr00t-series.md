# GR00T 系列精读：N1 / N1.5 / GR00T 2 —— 实现与训练

> 来源：GR00T N1 论文全文（[arXiv:2503.14734](https://arxiv.org/abs/2503.14734) HTML，L3）；GR00T N1.5 官方研究页（[research.nvidia.com/labs/gear/gr00t-n1_5](https://research.nvidia.com/labs/gear/gr00t-n1_5/)，2025-06-11，L1 官方一手）；GR00T 2 官方研究页（[research.nvidia.com/labs/gear/gr00t-2](https://research.nvidia.com/labs/gear/gr00t-2/)，2026-11，L1 官方一手）。
> 纪律：引文逐字、数值逐字；未取到的（如 N1.5 论文正文、GR00T 2 whitepaper PDF 正文）标注「未取到」。

---

## 0. 三代一句话

| 代 | 时间 | 一句话 | 关键跃迁 |
|---|---|---|---|
| **GR00T N1** | 2025-03 | 双系统 VLA：Eagle-2 VLM（S2）+ DiT flow-matching 动作头（S1），跨本体 | 数据金字塔 + 潜动作/IDM 标注无动作视频 |
| **GR00T N1.5** | 2025-06 | 冻结 VLM + 简化 adapter + 加 FLARE 未来潜表征对齐 | 语言跟随与泛化大幅提升；G1 真机 98.8% |
| **GR00T 2** | 2026-11 | 从 VLA 升级为**世界-动作模型（WAM）**：一个 DiT 联合去噪未来视频+动作 | X-Adapt 跨本体迁移、5 示教 95%、机上 124ms |

---

## 1. GR00T N1（L3，论文全文）

### 1.1 架构（原文）

> "GR00T N1 is a Vision-Language-Action (VLA) model with a **dual-system architecture**. The **vision-language module (System 2)** interprets the environment… The subsequent **diffusion transformer module (System 1)** generates fluid motor actions in real time. Both modules are **tightly coupled and jointly trained end-to-end**."

- **规模**：GR00T-N1-2B 共 **2.2B** 参数，其中 VLM **1.34B**（Eagle-2 = SmolLM2 LLM + SigLIP-2 视觉编码器，224×224 + pixel shuffle → **64 image tokens/帧**）。
- **VLM 层选择**：不用最后一层，而用**第 12 层**中间层 embedding——"using middle-layer instead of final-layer LLM embeddings resulted in both faster inference speed and higher downstream policy success rate"。
- **动作头**：DiT（交叉注意力+自注意力交替，Flamingo/VIMA 风格）；embodiment-specific MLP 编码 state/action 到共享维；**动作 chunk H=16**；flow matching 损失
  `L_fm(θ)=E_τ[‖V_θ(φ_t, A_t^τ, q_t) − (ε−A_t)‖²]`，τ~Beta((s−τ)/s; 1.5, 1)，s=0.999；推理用 Euler 积分 **K=4** 步。
- **频率/时延**：S2 在 L40 上约 **10Hz**；S1 闭环 **120Hz**；采样 16 步动作块 **63.9ms**（L40，bf16）。

### 1.2 数据（数据金字塔，原文）

> "We organize heterogeneous sources by scale: large quantities of web data and human videos lay the base of the pyramid; synthetic data … form the middle layer, and real-world data collected on the physical robot hardware complete the top."

- 底层：互联网视频 + 人类 egocentric 视频（Open X-Embodiment 中 RT-1/Bridge-v2/Language Table/DROID/MUTEX/RoboSet/Plex；AgiBot-Alpha 14 万轨迹等）。
- 中层：**潜动作**（VQ-VAE 学习帧间潜动作，encoder 当 IDM 用；把无动作视频当作一个叫 "LAPA" 的 embodiment）与 **神经轨迹**（fine-tune 图生视频模型于 88h 遥操作数据，生成 **827h** 视频，约 **10×** 扩增）。
- 顶层：真机遥操作（Fourier GR-1；VIVE Tracker + Xsens 手套，20Hz，IK retarget）。
- 仿真：DexMimicGen 生成 **78 万条**仿真轨迹 ≈ **6,500 小时**，仅用 **11 小时** 计算。

### 1.3 训练（原文）

- **预训练**：flow-matching，混合真机动作、潜动作（人类视频）、IDM 伪动作（神经轨迹）；语言组件冻结，其余微调。
- **后训练**：单 embodiment 微调；神经轨迹增强时**真机:神经轨迹 = 1:1** 采样；IDM 在低数据场景下只用低数据训练。
- **算力**：**1024 张 H100**（Quantum-2 InfiniBand，OSMO 编排）；GR00T-N1-2B 预训练约 **5 万 H100 GPU 小时**；单卡 A6000 可做 adapter+DiT 微调（batch 200）。

---

## 2. GR00T N1.5（L1，官方研究页）

### 2.1 相对 N1 的改动（原文）

> "The main differences from N1 are the following: **The VLM model is frozen during both pretraining and finetuning.** The **adapter MLP** connecting the vision encoder to the LLM is **simplified and adds layer normalization** to both visual and text token embeddings input to the LLM."

- VLM 换成 **Eagle 2.5**（2.1B），按 grounding 与物理理解调优。
- 新增 **FLARE**（Future LAtent Representation Alignment）：不做生成式未来建模，而是**对齐目标未来嵌入**（loss 系数 **0.2**，预训练与后训练都用）。
- 训练：**250K 步**，**1K H100**，全局 batch **16384**；AdamW + cosine + warmup 0.05。

### 2.2 结果（原文数值）

| 设置 | N1 | N1.5 |
|---|---|---|
| Language Table（从零训） | 52.8% | **93.2%** |
| Sim GR-1 Language（从零训） | 36.4% | **54.4%** |
| RoboCasa，30 demos/任务 | 17.4 | **47.5** |
| Sim GR-1，0-shot | 39.6 | **43.9** |
| Sim GR-1，30 demos/任务 | 43.2 | **47.4** |
| 真机 GR-1 语言跟随率 | 46.6% | **93.3%** |
| 真机 GR-1 总成功率 | 43.3% | **83.0%** |
| 新物体 0-shot | 0% | **15.0%** |
| 人视频 FLARE 后训练（新物体） | — | **55.0%** |
| DreamGen 12 个新动词 | 13.1% | **38.3%** |
| Unitree G1（1K demos） | 44.0% | **98.8%** |
| G1 新物体（5 种） | — | **84.2%** |
| VLM grounding：GR-1 IoU / RefCOCOg | — | **40.4 / 89.6**（Qwen2.5VL-3B：35.5 / 85.2） |

- 注：N1.5 的论文/whitepaper 正文未取到（官方页为 L1 一手）；上述数值均来自官方页表格。

---

## 3. GR00T 2（L1，官方研究页，2026-11）

### 3.1 范式升级：VLA → WAM（原文）

> "We introduce GR00T 2, a new generation of foundation **world-action models (WAMs)**. GR00T 2 **predicts future video and robot actions together**, learning how physical interactions unfold along with the actions that produce them."

> "**One diffusion transformer denoises future video and robot actions together.** Predicting how the scene should change lets GR00T 2 learn physics from human video; small per-robot heads turn the plan into motor commands for each body."

- 输入：最近视频 + 指令 + 机器人状态；输出：未来视频 + 动作 chunk 联合去噪；循环执行。
- **Turbo 版**：2B 参数、更小 transformer + 更高压缩视频编码器；**124ms/次预测**（G1 板载 Jetson Thor），经 SONIC 全身控制器 **50Hz** 执行。
- 注：whitepaper PDF（`gr00t2_whitepaper.pdf`）未取正文；以上为官方页逐字。

### 3.2 预训练数据（原文数值）

- **402,730 小时**人类 egocentric 工作视频（Human-1M；其中 100,000 小时以 Human-1M-100K 开放）
- **10,184 小时**手持 UMI 夹爪示教（2,897 个任务）
- **23,500 小时**机器人遥操作（YAM/Franka 平行夹爪、Sharpa/Wuji 灵巧手、Unitree G1）

### 3.3 关键能力（原文数值）

- **X-Adapt（跨本体迁移）**：human→Dexter 65%；UMI→YAM 60%（长程 4 分钟任务）；DexUMI→灵巧手（早餐/点灯）。
- **小样本适应**：50 个目标机器人未见过的任务，0-shot **40%**；5 示教（**<3 分钟**机器人数据）+ **30 分钟 LoRA**（单张 RTX 5090）→ **95%**。
- **真机 RL 自改进**：三个接触丰富任务 40–76% → **88–96%**（GRPO 自模仿）。
- **Agentic 控制**：推理模型作为 **System 2** 通过 WAM API 调用 GR00T 2，装配 97–98/100（纯 SFT 23–69，无 agent 71–84）。
- **视觉引导**：在相机画面上标注 grasp/release 点，短程指令成功率 40%→**92%**（YAM）、30%→70%（Dexter）。
- **仿真相关性**：25 任务 YAM benchmark，真机 vs 仿真 Pearson **0.92** / Spearman 0.89；加仿真数据后 OOD 成功率 6.7%→**63.3%**（GR00T 1.7 为 40.0%——GR00T 1.7 仅在此页被提及，细节未取到）。
- **仿真流程**：agentic real-to-sim（SimFoundry 重建场景，自动回放验证 + 专家反馈）。

### 3.4 与 S1/S2 概念的关系（重要）

GR00T 2 官方页**明确把推理 agent 称为 System 2**："The agent runs as **System 2 (S2)**. In an observe–reason–act–verify loop, it calls GR00T 2 as a tool…"——这是工程界对 S1/S2 用法的最新样本（且 S2 不再是 VLM 模块，而是**外部推理 agent**）。对比：N1/N1.5 的 S2 是 VLM 模块；GR00T 2 的 S2 是 agent，模型本体升级为 WAM。

---

## 4. 三代训练配方对比

| 项 | N1 | N1.5 | GR00T 2 |
|---|---|---|---|
| 范式 | VLA（双系统） | VLA（VLM 冻结） | **WAM**（视频+动作联合去噪） |
| VLM | Eagle-2（1.34B），第 12 层 | Eagle 2.5（2.1B），冻结 | 未公开（页面未列） |
| 动作头 | DiT flow matching，H=16，K=4 步 | 同 N1 + FLARE | 一个 DiT 联合去噪未来视频+动作；Turbo 2B |
| 训练规模 | 1024×H100，5 万 GPU 小时 | 1K×H100，250K 步，GBS 16384 | 未公开（页面未列） |
| 数据 | 88h 遥操作 → 827h 神经轨迹（10×）+ 78 万仿真轨迹 + 人类视频 + OXE | 同 N1 混入 DreamGen/AgiBot-Beta | 402,730h 人视频 + 10,184h UMI + 23,500h 遥操作 |
| 后训练 | 真机:神经轨迹=1:1 | 同 + FLARE 0.2 | LoRA 30 分钟（5 示教）；真机 RL（GRPO） |
| 开源 | N1-2B 权重+数据+基准 | N1.5-3B | 代码 2026-12 开放（页面："Code Available in December"） |

---

## 5. 精读要点（我的总结）

1. **N1 的核心设计是「双系统 + 数据金字塔」**：S2（VLM，10Hz）理解，S1（DiT，120Hz）出动作，端到端联合训练；数据金字塔用潜动作/IDM 把无动作视频变成可训练数据，用视频生成模型 10× 扩增真机数据。
2. **N1.5 的关键是「冻结 VLM + 简化 adapter + FLARE」**：与 N1 相比语言跟随从 46.6%→93.3%，说明双系统的主要瓶颈在**语言接地（grounding）**而非动作生成；FLARE 让「从人类视频学新物体」成为可能（55%）。
3. **GR00T 2 把 S1 从「动作头」升级为「世界-动作模型」**：一个 DiT 同时预测未来视频与动作——正是本调研主线的 WAM 范式；且 S2 从 VLM 模块变成外部推理 agent（"runs as System 2"），与 Taniguchi 的四过程框架里的 S2（缓慢审慎）在功能上对齐。
4. **工程可复用的三个数字**：N1 的 63.9ms/16 步（L40）、N1.5 的 98.8%（G1 1K demos）、GR00T 2 的 124ms（Jetson Thor）/95%（5 示教）。
5. **诚实缺口**：N1.5 与 GR00T 2 的 whitepaper 正文未取；GR00T 1.7 仅在 GR00T 2 页被提及；N1 的完整超参表（附录 D）与数据集统计表（附录 F）未取到。

<!-- QA_LOG -->
（待与用户讨论后追加）
<!-- /QA_LOG -->
