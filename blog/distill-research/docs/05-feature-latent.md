# 05 · 特征与 latent 蒸馏：蒸"教师看到了什么"，而不是"教师输出什么"

> 适用场景：教师和学生的**输出空间不一致**（7B LLM 出 latent token，车端只出轨迹；14B WAM 出未来视频，策略只出动作），此时 logit/动作对齐不可行，只能对齐中间表示。
> 本页覆盖：Orion-Lite（latent planning token）、CKT-WAM（世界模型隐状态）、THAW-VLA（世界模型特征离线缓存）、VLP（训练期教师）、S-VAM（自蒸馏 VFM 特征），以及 2026 下半年新出的 DRIFT / W2A / DreamFree / SPARK-VLA。
> **完整实现（缓存格式、LQCA、cosine 对齐、W2A 的 PCA+Q-Former）见第 12 章 §1–§4、§12。**

---

## 1. 为什么"蒸中间层"在驾驶里特别有效

Orion-Lite 的关键论断（Abstract）：

> "an efficient vision-only student model Orion-Lite can even **surpass the performance of its massive VLA teacher**"

作者的解释：LLM 在直接出轨迹的模式下，实际是一个 **"overparameterized feature extractor"**（§2.2）——它的价值在于把视觉+语言压成好的规划表征，而不是在推理期做复杂推理。把这个表征蒸出来，学生的"思考"可以在离线完成。

这条论断是整个驾驶 VLA 车端化的理论基础：**推理期的 LLM 不必存在，只要它的表征被学生继承。**

---

## 2. Orion-Lite（2604.08266）：latent planning token 的 L1 蒸馏

| 项 | 内容 |
|---|---|
| 教师 | ORION：EVA-02-L 视觉 + Vicuna-v1.5 **7B** LLM + VAE 生成式规划器 |
| 学生 | LLM 换成 **0.1B** 的 6 层 transformer decoder（input projection + learnable planning query + cross-attention + output projection，§3.3） |
| 蒸什么 | 教师 LLM 的 latent planning token `T_p ∈ R^{1×C_p}`；学生**丢弃文本 prompt**，纯视觉 |
| 损失 | `L_mimic = (1/(B·C_p)) Σ ‖T_student − T_p‖₁`（Eq.1，**L1 回归，不用 KL**） |
| 总损失 | `L_total = L_GT + L_mimic`；`L_GT = L_col + λ_bd·L_bd + λ_reg·L_reg + λ_vae·L_vae`（λ=3） |
| 冻结 | 视觉编码器 + QT-Former **严格冻结**，只训 0.1B decoder + planner |
| 结果 | Bench2Drive 闭环 **DS 80.6**（教师 77.7、MindDrive 78.0、UniDrive-WM 79.2）；推理模块 **150×**；显存 **31GB→8GB**；端到端 **806ms→267ms** |

**为什么用 L1 而不是 KL**（原文）：

> "avoiding ... suboptimal distributional metrics"

作者的对比对象是 DiMA（用 KL 蒸 VLM 特征），并指出 DiMA "limits its evaluation to open-loop metrics"。这是一个重要的方法论信号：**在驾驶里，特征回归的目标是"规划相关的几何/语义"，不是"教师的概率分布"**。

**实现伪代码**：

```python
# 教师：一次性前向，取 LLM 输出的 planning token
with torch.no_grad():
    T_p = teacher_llm(teacher_vision(imgs), prompt).planning_token  # [B, 1, C_p]

# 学生：视觉 only，无文本
T_s = student_decoder(student_vision(imgs))                        # [B, 1, C_p]
L_mimic = (T_s - T_p).abs().mean()

# GT 轨迹监督（碰撞/边界/回归/VAE 四项）
L_gt = L_col + 3*L_bd + 3*L_reg + 3*L_vae
loss = L_gt + L_mimic
```

**消融启示**（L2）：中间层 > 最后一层；纯视觉学生不需要文本也能学到语言条件下的规划（因为教师的 token 已经融合了语言）。

---

## 3. CKT-WAM（2605.06247）：跨异构 WAM 的隐状态压缩

| 项 | 内容 |
|---|---|
| 教师 | **DreamZero-14B**（WAM） |
| 学生 | **Cosmos-Policy-2B**（WAM） |
| 蒸什么 | 教师**第 20 层（共 40 层）**中间隐状态；教师只做**单次前向观测编码器**（`t*=0`，不生成未来视频/动作） |
| 压缩 | 共享 bottleneck adapter（`d_b=512`）→ **LQCA** 把 N 个 token 压成 **32+32** 个 query token（generalized / specialized 两个 bank） |
| 适配 | 常开 generalized adapter + **8 个** specialized adapter（residual bottleneck，`top-2` 路由，`λ_bal=0.01`） |
| 注入 | `C_A=[C_g;C_s]` 拼到学生**文本条件序列**尾部（加 learnable separator），cross-attention 进入学生 DiT——**不改学生去噪骨干** |
| 训练 | 教师与学生**全部冻结**，只训 CKT 模块（**187.4M = 1.17%**）；目标=学生自己的视频+动作 latent flow-matching 损失 |
| 配置 | 8×A800，bf16，AdamW（β=0.9/0.95，wd=0.01），cosine+1k warmup，LR 3e-4 |
| 结果 | LIBERO-Plus 零样本 **86.1%**（逼近全量微调）；真机 4 个长程任务平均 **83.3%** |

**为什么值得学**：它解决的是"两个 WAM 骨干不同构、动作头不对称"的问题——**不蒸输出、不蒸逐层特征，而是把教师压成一段可注入的"上下文 token"**。这是 2026 年最有工程美感的一个设计。

**伪代码**：

```python
# 教师：单次前向，t*=0，取第 20 层
with torch.no_grad():
    H_t = teacher(obs, text, t=0, return_layer=20)     # [B, N, D_t]
H_t = bottleneck(H_t)                                   # [B, N, 512]
C_g = lqca_general(H_t)                                 # [B, 32, D_s]
C_s = lqca_specialized(H_t)                             # [B, 32, D_s] × 8 experts, top-2
cond = concat([text_tokens, separator, C_g, C_s], dim=1)

# 学生：冻结 DiT，只训 CKT 模块
loss = flow_matching(student_dit(obs, cond), target_video_action_latent)
```

---

## 4. THAW-VLA（2609.24682）：世界模型特征离线缓存，部署零开销

| 项 | 内容 |
|---|---|
| 教师 | 冻结 **Cosmos 3 理解塔**（其 **Qwen3-VL-8B reasoner** 的 layer-24，每视图 mean-pool，**D_t=4096**） |
| 学生 | **QwenGR00T**：0.8B Qwen3.5-VL backbone + GR00T flow-matching action expert（θ 全可训） |
| 蒸什么 | 世界模型的内部特征（原文："What a world model knows about physical scenes lives in its internal features; generating the future is merely the objective that produced them."） |
| 怎么蒸 | 教师对训练帧**离线跑一次** → memory-mapped 缓存（key = trajectory id + base index）；训练时只加**视图级 cosine 对齐（λ=0.5 everywhere）**：`L_align = 1 − cos(p_φ(f_c^S), sg[f_c^T])`；投影 `p_φ` 训完丢弃 |
| 部署 | 蒸馏模块**全部丢弃**；"latency is identical to the undistilled student by construction"；"the deployed policy is architecturally identical to the undistilled baseline" |
| 结果 | LIBERO **97.9%**（同结构未蒸馏 +2.6）；RoboCasa-GR1 **48.2%→50.5%**（距同架构 4B 仅 4.3 点）；真机 **28/30**（参数为 4B π-style 的 1/5）；**32ms / 1.86GB**（RTX 5090）——对比 π0.5 **65ms**、DreamZero **3s/H100 + 45.9GB** |
| 训练 | LIBERO ~273k 示教步 / 2 视角；RoboCasa-GR1 24 任务 24k episodes（6.02M 步）；4×A100-80GB + ZeRO-2，有效 batch 256；教师缓存约 4 GPU·1 小时 |

**为什么这条最重要**：它是唯一一个"蒸馏对部署零成本"的严格证明——学生结构、延迟、显存与未蒸馏基线**逐位一致**，唯一变化是训练时的对齐项。

**坑**：缓存的是教师看到的状态，学生偏离后特征失效；所以 THAW 的 λ 只有 0.5，且只对齐"视图级"（粗粒度），不敢做逐 token 对齐。

---

## 5. VLP（2401.05577）：训练期教师，推理期不存在

- 机制：冻结 LLM 把 GT 感知/规划写成 prompt 编码，**对比损失**对齐 BEV 特征与规划特征；LM 冻结，只训适配层。
- 原文："VLP is only active during training, ensuring no additional parameters or computations are introduced during inference"（Fig.2 caption）。
- 价值：最省部署成本的一条路——教师的知识以"对齐目标"形式进入学生，推理期完全消失。
- 局限：对比学习的目标弱于显式回归，增益通常在 1–3 个点；适合作为辅助 loss，不适合单独扛蒸馏。

---

## 6. S-VAM（2603.16195）：自蒸馏 VFM 特征，把多步生成压成单步

- 教师 = **自身多步生成的视频**，用冻结 VFM 提特征：DPAv3（几何）+ DINOv2（语义）。
- 学生 = 轻量 decoupler（时空 Transformer），把一步去噪的噪声特征映射到 VFM 表征：`L = ||F̃ − Y||²`。
- 关键设计：用"自己多步生成的视频"而非真值未来帧——**同一条扩散轨迹**，避免轨迹错位。
- 成本：自蒸馏仅 **50k 步、单张 H100**；推理单卡 RTX 3090；CALVIN 平均序列长度 **4.16**（VPP 3.58）。
- 定位：**加速方法**（压缩推理路径），不是知识迁移；学生没有获得教师之外的新知识。

---

## 7. 2026 下半年的新面孔

### DRIFT（2611.24719，2026-11-18，L1）

- 轻量 LM（<150M）→ 规划器（全模型 255M）；蒸 **surrogate reasoning features**（latent 推理特征），推理期可把 LM 摘掉。
- 数字：Bench2Drive **DS 97.03 / SR 89.39%**（三次均值），255M 参数。
- 定位：Orion-Lite 的"更彻底版"——连 0.1B decoder 都不要，直接用 255M 规划器。

### W2A（2611.02600，2026-11-02，L2 部分正文）

- 教师：**Cosmos-Predict2** 视频 DiT（Stage I 用 LoRA 适配机器人视频）；学生：action DiT（**从零训练**）；测试时视频 DiT **整体丢弃**。
- 蒸什么（原文）："distills intermediate denoising representations from a video diffusion model into an action model during training, while entirely discarding the video model at test time"。
- 关键假设（原文）："although video latent representations are high-dimensional, a compressed low-dimensional subspace is sufficient to provide effective supervision for action generation"——所以先 **PCA top-K** 压缩，再用轻量 **Q-Former** 对齐。
- 损失（式 10–12）：`L = L_FM + λ_dist · (1/B)Σ_b w(τ_b)·(1/M)Σ_i(1 − cos(ẑ^a, z̃^v))`，`w(τ)=exp(−(τ−τ_c)²/(2σ_w²))`——**按去噪时间给权重**，只在对齐最有效的时间步上使劲。
- 数字：LIBERO **95.9%**、LIBERO-Long 93.6%；CALVIN ABC→D 平均长度 **4.30**；约 **21×** 推理加速。

### DreamFree（2611.04527，2026-11-03，L1）

- 冻结 14B 视频扩散 → 只看当前观测的预测性视觉编码器；蒸**未来视频表征**（不生成像素）。
- 数字：真实闭环 **11Hz**（14B 视频骨干）；真实+仿真成功率高于 SOTA VLA/WAM。

### SPARK-VLA（2611.10443，NeurIPS 2026，L1）

- 冻结 3D 专家 → LoRA 视觉分支；0.5B LLM backbone，约 **53M** 可训练参数；对齐注意力响应迁移 3D 理解，router 用 action-loss 梯度监督选 token。
- 定位：蒸"空间表征"而非动作；参数高效（LoRA）路线的代表。

---

## 8. 特征蒸馏的五个实现决定（按重要性排序）

1. **对齐哪一层**：中间层 > 最后一层（最后一层过拟合任务，中间层更通用）。
2. **L1 vs L2 vs cosine vs KL**：回归用 L1（对异常值鲁棒，Orion-Lite 选它）；方向对齐用 cosine（THAW）；分布对齐用 KL（DiMA，但 open-loop 有效、闭环存疑）。
3. **维度不匹配怎么办**：线性投影（最小成本）→ MLP → cross-attention query 压缩（CKT-WAM 的 LQCA）。
4. **教师特征是否离线缓存**：离线缓存省显存但引入 off-policy 偏差；在线提特征贵但准。
5. **λ 怎么定**：特征损失通常是辅助项，λ 在 0.5–1.0 之间（THAW 0.5；Orion-Lite 直接相加，λ=1）。

---

## 9. 一张对照表

| 工作 | 教师 | 学生 | 蒸什么 | 部署是否带蒸馏模块 | 关键数字 |
|---|---|---|---|---|---|
| Orion-Lite | 7B LLM | 0.1B decoder | latent planning token | 否（decoder 就是学生） | DS 80.6>77.7；150×；31→8GB |
| CKT-WAM | 14B WAM | 2B WAM | 第 20 层隐状态→64 token | 否（CKT 训完即弃） | LIBERO-Plus 86.1%；1.17% 参数 |
| THAW-VLA | Cosmos3-Nano | 0.8B VLA | layer-24 特征（4096-D） | 否（逐位一致） | 97.9%；32ms/1.86GB |
| VLP | 冻结 LLM | ADS | 对比对齐 | 否 | 零推理开销 |
| S-VAM | 自身多步视频 | 同模型 | DPAv3+DINOv2 特征 | 否 | CALVIN 4.16；50k 步单卡 |
| DRIFT | <150M LM | 255M 规划器 | latent 推理特征 | 否 | DS 97.03 |
| W2A | 视频扩散 | 动作模型 | 中间去噪表征 | 否 | LIBERO 95.9%；21× |
| DreamFree | 14B 视频扩散 | 编码器+动作头 | 未来视频表征 | 否 | 11Hz |

**共同点**：全部在部署时丢掉教师/蒸馏模块。这是 2026 年可落地蒸馏方案的必要条件。

---

更新时间: 2026-11-21
