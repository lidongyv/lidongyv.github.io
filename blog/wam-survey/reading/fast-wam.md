# Fast-WAM 精读笔记（arXiv:2603.16666）

> **论文**：Fast-WAM: Do World Action Models Need Test-time Future Imagination?
> **作者**：Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao（IIIS, Tsinghua University / Galaxea AI）
> **提交**：2026-03-17（v1）| **来源**：[ar5iv HTML 全文](https://ar5iv.labs.arxiv.org/html/2603.16666) + [arXiv HTML](https://arxiv.org/html/2603.16666v1) + [项目页](https://yuantianyuan01.github.io/FastWAM/) + [GitHub 代码](https://github.com/yuantianyuan01/FastWAM)
> **证据级别**：全部引文为论文原文（英文 verbatim）；代码依据来自官方 README/项目页。
> **环境偏差说明**：沙箱禁止 shell 出网与 pip 安装，本地 PDF 与本地抠图未完成（技能要求项）；图以在线 URL 引用。
>
> **一句话结论**：WAM 的收益主要来自**训练期视频共训（video co-training）塑造的世界表征**，而非**推理期显式未来想象**；Fast-WAM 保留前者、砍掉后者，190ms 实时推理，无具身预训练达到 RoboTwin 91.8 / LIBERO 97.6。

---

## 0. 五轴速查（本论文对你五个问题的答案）

| 轴 | Fast-WAM 的做法 | 依据 |
|---|---|---|
| **输入表征** | 多相机图像 **concatenate 成单图** → Wan2.2 预训练 **video VAE 潜 token**；语言 = Wan2.2 自带 **T5 编码器**经 **cross-attention** 注入所有 token；第一帧 clean latent 作共享视觉锚点；**proprio 经 Linear 投影为单 token 追加进文本条件序列**（论文未提，代码实锤） | 论文 §3.2/§4.1；代码 `libero_uncond_2cam224`、`robotwin_uncond_3cam_384`、`fastwam.py` |
| **目标表征** | 仅**任务指令文本**（T5 embedding）；**无 goal image** | 论文公式 p(a\|o,l) 只含 o,l；正文未提 goal image |
| **动作表征** | **动作 chunk h=32**，由 1B 参数 ActionDiT（hidden 1024，由 Wan2.2 DiT 插值初始化）以 **flow matching 去噪**生成；动作空间经代码核实：**LIBERO 7 维（Δxyz3+Δrpy3+gripper1）、RoboTwin 14 维（双臂）**；执行用 ACT 式时间集成 | 论文 §4.1；代码 `configs/data/libero_2cam.yaml`、`robotwin.yaml`、`action_ensembler.py` |
| **决策表征** | 测试时**单次前向**直接出动作（无未来视频去噪、无外层 AR rollout——论文明说为受控对比而省略）；推理 10 步去噪、CFG=1.0、logit-normal 时间调度 | 论文 §3.1/§4.1 |
| **验证** | LIBERO 四套件（500 demo/套件，20k 步，2000 试次）+ RoboTwin 2.0（>50 任务，2500 clean+25k 随机化 demo，30k 步，100 试次/任务）+ 真机毛巾折叠（Galaxea R1 Lite，60h 遥操作）；**unseen 指令评测**（代码 README 明示，与 LingBot-VA 的 seen 指令不同） | 论文 §4.2；README |

---

## 1. Abstract（原文 + 逐句指路）

> "World Action Models (WAMs) have emerged as a promising alternative to Vision-Language-Action (VLA) models for embodied control **because they explicitly model how visual observations may evolve under action**."

指路：WAM 相对 VLA 的核心卖点 = 显式建模「观测在动作下的演化」。（即联合分布里的 p(o′|o,a) 部分）

> "Most existing WAMs follow an **imagine-then-execute** paradigm, incurring substantial test-time latency from iterative video denoising, yet it remains unclear whether **explicit future imagination is actually necessary for strong action performance**."

指路：点名两类成本的纠缠——先想象（迭代视频去噪，慢）再执行；作者质疑「想象」的必要性。

> "we propose Fast-WAM, a WAM architecture that **retains video co-training during training but skips future prediction at test time**."

> "It runs in real time with **190 ms latency**, over **4× faster** than existing imagine-then-execute WAMs."

> "These results suggest that **the main value of video prediction in WAMs may lie in improving world representations during training rather than generating future observations at test time**."

指路：全文结论句——视频预测的价值在「训练期塑造表征」，不在「测试期生成未来」。

---

## 2. §1 核心论证（原文）

> "The effectiveness of WAMs may stem from two distinct sources: _(1)_ the video prediction objective during training, which may help the model acquire stronger physical priors and action-conditioned representations, and _(2)_ explicit future generation during inference, which may provide additional foresight for action prediction. **Existing WAM systems typically entangle these two factors**, making it difficult to determine which one is actually responsible for the observed gains."

指路：两个因子被现有 WAM「纠缠」在一起——这正是 Fast-WAM 用受控变体去「解缠」（disentangle）的实验设计动机。

**Figure 1（原文 caption）**："Three representative WAM paradigms. (A) Joint-modeling WAMs denoise future video and action tokens together. (B) Causal WAMs first generate future observations and then condition action prediction on the generated future representation. (C) Fast-WAM retains video co-training during training but removes explicit future generation at inference time, directly predicting actions from latent world representations in a single forward pass."
[图](https://arxiv.org/html/2603.16666v1/teaser.png)

---

## 3. §3 Method（原文 + 指路）

### 3.1 形式化（原文公式 + 句子）

想象-执行分解（式 2）：`p(a_{1:H}|o,l) = ∫ p(v_{1:T}|o,l) p(a_{1:H}|o,l,v_{1:T}) dv_{1:T}`
> "the model first predicts future observations and then conditions action generation on the imagined future."

Fast-WAM 的直出式（式 3/4）：`p_θ(a_{1:H}|o,l) = p_θ(a_{1:H}|z(o,l))`
> "The key difference from imagine-then-execute WAMs is that **z(o,l) is obtained by a single forward encoding pass**, rather than by explicitly sampling or denoising future observations v_{1:T} at inference time."

指路：注意——Fast-WAM 测试时接口与 VLA 相同（direct-policy interface），差别全在**训练目标**里多了一个视频预测项。

### 3.2 架构（原文）

> "Fast-WAM is built on top of the **video Diffusion Transformer (DiT) from Wan2.2-5B**, which serves as the world modeling backbone. We also reuse its pretrained text encoder and video VAE: **task language is encoded by the built-in T5 encoder and provided to all tokens through cross-attention, while visual observations are mapped into latent video tokens by the pretrained VAE**."

> "We organize the input tokens into **three groups**: clean latent tokens of the first observation frame, which serve as the shared visual anchor; **noisy latent tokens of future video frames, which are used only during training for video modeling**; and action tokens, which are processed by the action expert for action generation."

> "**Crucially, action tokens cannot attend to future video tokens, and the clean first-frame tokens do not attend to any other tokens.** This ensures that both video modeling and action prediction are grounded in the same visual context while **preventing future information from leaking into the action branch**."

指路（信息流设计）：动作分支永远看不到未来帧信息（防泄漏）；第一帧是唯一视觉锚点。这意味着即便在 Fast-WAM-Joint 变体里，动作与未来视频的交互也是受控的。

> "At inference time, Fast-WAM **removes the future video branch entirely**: only the clean first-frame latent tokens are retained and passed through the video backbone once to produce latent world features for the action expert."

### 3.3 训练目标（原文公式）

联合 flow matching（式 5-9）：`y_t = (1-t)y + tε`；`L_FM(y) = E[‖f_θ(y_t,t,o,l) − (ε−y)‖²]`；`L = L_act + λ L_vid`（动作项 y=a_{1:H}；视频项 y=z_{1:T} 即未来帧 VAE 潜）。

### 3.4 受控变体（消融设计的核心，原文）

> "The first variant, named **Fast-WAM-Joint**, follows the joint-generation paradigm, where future video tokens and action tokens are denoised together within a shared model… The second variant, named **Fast-WAM-IDM**, follows the video-then-action paradigm, where future video tokens are generated first… and action prediction is then conditioned on the resulting future representation."

> "We further construct a Fast-WAM variant **without video co-training**. This variant keeps the architecture and inference procedure unchanged, and **removes only the video modeling objective during training**. It therefore serves as a direct control for the role of video co-training itself."

指路：四个臂 = ①Fast-WAM（共训+不想象）②Joint（共训+联合想象）③IDM（共训+先想象）④w.o. co-train（不共训+不想象）。②③与①的差距 = 「想象」的价值；④与①的差距 = 「共训」的价值。这是全文最有说服力的实验设计。

---

## 4. §4.1 实现细节（原文 + 代码佐证）

> "The action expert shares the same architecture as the video branch but uses a reduced hidden dimension of **d_a = 1024**, resulting in a **1B action expert** and a total model size of **6B** parameters. We set the action horizon to **h = 32**. Video frames are temporally downsampled by **4×**, resulting in **9 video frames per chunk**. **Images from multiple cameras are concatenated into a single image** before being fed into the VAE."

> "we adopt a **logit-normal distribution over t** as the noise schedule… During inference, we use **10 denoising steps with classifier-free guidance (CFG) scale set to 1.0**."

> "All latency numbers are measured on a single NVIDIA **RTX 5090D V2 32GB** GPU."

> "For Fast-WAM-IDM, we follow [LingBot-VA] and apply **noise augmentation to the ground-truth video tokens with probability p = 0.5**."

**代码佐证（GitHub README）**：
- 数据 config：`libero_uncond_2cam224_1e-4`（2 相机 224²）、`robotwin_uncond_3cam_384_1e-4`（3 相机 384²）→ 输入观测的具体形状依据。
- ActionDiT 初始化：`ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`（Wan2.2 DiT 线性插值 + alpha scale + 1024 hidden）。
- 文本：`scripts/precompute_text_embeds.py`（预计算 T5 embedding 缓存）→ 语言表征实现依据。
- 动作归一化：`dataset_stats.json` + `EVALUATION.sigma_shift`（动作 scheduler shift，v1 复现用 5.0，新版默认 1.0）→ 动作按数据集统计量归一化的间接依据。
- 评测协议（重要）："**We evaluate with unseen instructions, following Motus. LingBot-VA uses seen instructions instead.** You can try EVALUATION.instruction_type=seen … which should theoretically improve performance by one or two points." → 读榜单时要对齐指令口径。
- 训练规模：LIBERO 单机 8 GPU；RoboTwin 64 GPU。

### 4.1b 动作空间与数据格式（代码依据，2026-10-26 抓取官方仓库核实）

**`configs/data/libero_2cam.yaml`（verbatim 关键行）**：
- `action: raw_shape: 7, shape: 7`；`action_output_dim: 7 # eef_pose (6) + gripper (1)`；`proprio_output_dim: 8 # eef_pose (6) + gripper (2)`
- `delta_action_dim_mask: [true, true, true, true, true, true, false] # eef poses are delta, gripper is not`
- `num_frames: 33`；`action_video_freq_ratio: 4 # 32 action, 9 video frames`；`concat_multi_camera: "horizontal"`；`norm_default_mode: min/max`
- 图像：`image` 与 `wrist_image` 两路，512²→224²

**`configs/data/robotwin.yaml`（verbatim 关键行）**：
- 3 相机（cam_high / cam_left_wrist / cam_right_wrist，480×640→240×320）；`action: raw_shape 14, shape 14`；`proprio_output_dim: 14`；`norm_default_mode: "z-score"`；`concat_multi_camera: "robotwin"`
- → 结论：LIBERO 动作 = **7 维：Δxyz(3) + Δrpy(3) + gripper(1)**（前 6 维增量、夹爪绝对，min/max 归一化）；RoboTwin 动作 = **14 维双臂**（z-score 归一化；每臂 7 维内部拆分 config 未注明）。proprio：LIBERO 8 维、RoboTwin 14 维。

**`src/fastwam/datasets/lerobot/transforms/relative_action.py`**：提供 `RelativePoseTransform`（7 维位姿 x,y,z,i,j,k,r；SE(3) 齐次矩阵做绝对↔相对变换；四元数顺序 (i,j,k,r)↔(r,i,j,k) 转换）与 `RelativeJointTransform`（相对首帧差分）→ 仓库同时支持位姿/关节两种动作空间，但默认 config 未挂载额外 transform。

**`experiments/libero/action_ensembler.py`**：ACT 式**时间集成**——按时间戳缓存多次预测、取平均执行（`averaged_action = np.mean(stacked_preds, axis=0)`）→ 闭环执行惯例的代码依据。

**`src/fastwam/models/wan22/fastwam.py`（proprio 实锤——论文方法节未提）**：
- `self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim)`（proprio 投影到与 T5 文本同维度）
- `_append_proprio_to_context`：`proprio_token = self.proprio_encoder(proprio.unsqueeze(1)) # [B, 1, D]` → `torch.cat([context, proprio_token], dim=1)` → **proprio 变成 1 个 token，追加到文本条件序列尾部**，经 cross-attention 同时喂给视频专家与动作专家
- `build_inputs`：`proprio = proprio[:, 0, :] # [B, D]`（取观测时刻状态）
→ 结论：Fast-WAM 输入 = 首帧图像 latent + T5 文本 + **1 个 proprio token**（全部进 cross-attention 条件序列）；论文 §3.2 只描述「三组 token」，proprio 属论文表述不全，而非模型无此输入。
- 新版推理提速：RTX 4090 190→110 ms；H20 470→210 ms（`compile_action_infer=true`）。
- Optional-IDM 变体（单模型双推理模式 `idm` / `first_frame`），官方权重 LIBERO 复测（40 任务 × 50 episodes）：IDM 98.55% vs First-frame 97.75%。

---

## 5. 实验（全表 + 原文结论句）

### Table 1（RoboTwin，论文原表数值）

| Method | Embodied PT. | Clean | Rand. | Average |
|---|---|---|---|---|
| π0 | ✓ | 65.92 | 58.40 | 62.2 |
| π0.5 | ✓ | 82.74 | 76.76 | 79.8 |
| Motus | ✓ | 88.66 | 87.02 | 87.8 |
| Motus from WAN2.2 | ✗ | 77.56 | 77.00 | 77.3 |
| LingBot-VA | ✓ | 92.90 | 91.50 | 92.2 |
| LingBot-VA from WAN2.2 | ✗ | 80.60 | – | 80.6 |
| **Fast-WAM** | ✗ | **91.88** | **91.78** | **91.8** |
| Fast-WAM-Joint | ✗ | 90.84 | 90.32 | 90.6 |
| Fast-WAM-IDM | ✗ | 91.16 | 91.34 | 91.3 |
| Fast-WAM w.o. video co-train | ✗ | 82.76 | 84.80 | 83.8 |

### Table 2（LIBERO，论文原表数值）

| Method | Embodied PT. | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|---|
| OpenVLA | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | ✓ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π0.5 | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| LingBot-VA | ✓ | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Motus | ✓ | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| **Fast-WAM** | ✗ | **98.2** | **100.0** | **97.0** | **95.2** | **97.6** |
| Fast-WAM-Joint | ✗ | 99.6 | 99.4 | 98.2 | 96.8 | 98.5 |
| Fast-WAM-IDM | ✗ | 98.8 | 97.8 | 97.8 | 97.6 | 98.0 |
| Fast-WAM w.o. video co-train | ✗ | 89.2 | 99.2 | 95.4 | 90.0 | 93.5 |

### §4.3.2 受控对比结论（原文）

> "On RoboTwin, Fast-WAM achieves 91.8% success rate, which is highly comparable to both Fast-WAM-Joint (90.6%) and Fast-WAM-IDM (91.3%). In contrast, **removing video co-training reduces performance to 83.8%**…"

> "A similar trend holds on LIBERO. Fast-WAM reaches 97.6% average success rate, remaining close to Fast-WAM-Joint (98.5%) and Fast-WAM-IDM (98.0%). By comparison, **Fast-WAM without video co-training drops to 93.5%, with especially visible degradation on the Spatial and Long subsets**."

> "This pattern suggests that the main benefit of WAM-style training may lie less in whether, or how, future imagination is performed at test time, and more in **the video prediction objective used to shape world-grounded representations during training**."

### §4.3.3 真机与效率（原文）

> "On this task [towel folding], pretrained π0.5 remains the strongest method… **In contrast, removing video co-training causes a dramatic degradation on both metrics: Fast-WAM without video co-training drops to only 10% success rate and also requires the longest completion time among all methods.**"

> "In terms of runtime, **Fast-WAM retains low inference latency (190 ms), whereas the imagine-then-execute variants are substantially slower, especially Fast-WAM-IDM at 810 ms.**"

注意（不脑补）：真机各方法的**具体成功率数值只在 Figure 4 散点图中**，正文未给数字，本笔记不转述未读到的数值。

---

## 5.5 why 追问（机制层面；「推断」均标注，不当作原文结论）

**Q1 为什么动作 token 被禁止 attend 未来视频 token（连 Joint 变体也如此）？**
- 原文依据（§3.2）："Crucially, action tokens cannot attend to future video tokens, and the clean first-frame tokens do not attend to any other tokens. This ensures that both video modeling and action prediction are grounded in the same visual context while **preventing future information from leaking into the action branch**."
- 原文给的直接理由：防未来信息泄漏进动作分支。
- 推断（原文未明说）：另一层动机是**训练-推理一致性**——推理时未来分支被整体移除，若训练时动作读过未来 token，测试时动作专家看到的注意力分布会漂移。

**Q2 为什么未来视频 4× 时间降采样（9 帧/chunk）、动作 horizon 却取 32？**
- 原文依据（§4.1）："We set the action horizon to h = 32. Video frames are temporally downsampled by 4×, resulting in 9 video frames per chunk."
- 代码依据（`configs/data/*.yaml`）：`num_frames: 33`（1 观测帧 + 32 动作）、`action_video_freq_ratio: 4 # 32 action, 9 video frames`——**9 帧 = 33 帧观测按 1:4 抽帧而来**（动作 32 步全保留、视频监督稀疏化）。
- 为什么取 4:1？**原文与代码注释均未写**。推断：①未来帧是 noisy latent token，数量大，降采样省训练算力；②每 3.5 步动作 1 帧，视频监督只需稀疏锚点塑造表征（与 Motus 的 Action-Dense Video-Sparse 思想一致）。均为推断。

**Q3 为什么 CFG scale 取 1.0（等于关闭引导）？**
- 原文依据（§4.1）："During inference, we use 10 denoising steps with classifier-free guidance (CFG) scale set to 1.0."
- **原文未写理由**。推断：语言+观测条件已充分，CFG>1 需双倍网络前向，与 190ms 实时目标冲突。

**Q4 为什么 w.o. video co-train 在真机掉到 10%，仿真只掉 ~8 点（RoboTwin）/ ~4 点（LIBERO）？**
- 原文依据（§4.3.3）："removing video co-training causes a dramatic degradation on both metrics… This gap is much larger than the differences among the Fast-WAM variants themselves, suggesting that video co-training is the dominant factor behind strong real-world performance."
- 原文只下结论、**未解释仿真/真机差异成因**。推断：真机仅 60h 数据（远小于 RoboTwin 27.5k demos），数据稀缺时世界建模目标的正则价值更大；毛巾折叠是长时程可变形物任务，对物理先验需求更高。均为推断。

**Q5 为什么主表放「Motus from WAN2.2」「LingBot-VA from WAN2.2」这类无具身预训练对照？**
- 原文依据（§4.3.1）："These results suggest that Fast-WAM can recover most of the benefit of prior WAM pipelines without relying on embodied pretraining."
- 目的：剥离「具身预训练」混杂变量，在同骨干（Wan2.2）下公平比较方法本身；否则「无 PT 的 Fast-WAM」对比「有 PT 的对手」无法归因。

---

## 6. 关键引文清单（verbatim 存档）

1. "WAMs… explicitly model how visual observations may evolve under action."（Abstract）
2. "the main value of video prediction in WAMs may lie in improving world representations during training rather than generating future observations at test time."（Abstract/Conclusion）
3. "action tokens cannot attend to future video tokens, and the clean first-frame tokens do not attend to any other tokens."（§3.2）
4. "At inference time, Fast-WAM removes the future video branch entirely."（§3.2）
5. "Images from multiple cameras are concatenated into a single image before being fed into the VAE."（§4.1）
6. "10 denoising steps with classifier-free guidance (CFG) scale set to 1.0."（§4.1）
7. "We evaluate with unseen instructions, following Motus. Lingbot-VA uses seen instructions instead."（GitHub README）

## 7. 全文一页总结

- **Problem**：现有 WAM 的收益来自「训练期视频预测目标」还是「推理期未来想象」？二者被纠缠，无法归因。
- **Method**：Fast-WAM = Wan2.2-5B 视频 DiT（世界骨干）+ 1B ActionDiT（MoT 共享注意力）；训练 = 动作 flow matching + 未来视频潜 flow matching 共训（λ 加权）；推理 = 只过第一帧 clean latent，直出 h=32 动作块，无未来视频去噪。三种受控变体（Joint/IDM/w.o. co-train）做因果消融。
- **Experiment**：RoboTwin 91.8（无具身预训练，超 Motus 87.8 与无 PT 的 LingBot-VA 80.6）；LIBERO 97.6；真机毛巾折叠：w.o. co-train 掉到 10%；延迟 190ms vs IDM 810ms。
- **Takeaway**：视频预测的价值在训练期表征塑造 > 测试期生成未来；去掉共训掉 7-8 点（RoboTwin），去掉测试时想象只差 ~1 点。

## 8. 待核实项（后续讨论时一起查证）

1. 「We follow [4] to refer to these models as World Action Models (WAMs)」的 [4]（=bib.bib12）对应哪篇：references 全文未取到（ar5iv 截断于 ref[3]；Semantic Scholar/OpenAlex 无可用引用表）。已缩小范围：bib12 ∈ 该句「jointly modeling future video and robot actions」引文簇（bib16/bib26/bib25/bib27/bib28/bib14/bib13/bib12），其中 bib13=Motus、bib14=LingBot-VA 已确认；bib12 待拿到参考文献表后定。
2. ~~动作空间维度（正文未写）~~ **已解决（代码依据）**：LIBERO 7 维 Δxyz+Δrpy+gripper；RoboTwin 14 维双臂；见 §4.1b。剩余未定：RoboTwin 每臂 7 维的内部拆分（xyz+quat vs xyz+rpy）config 未注明。
3. 真机 Figure 4 的具体成功率数值（正文未给）。
4. λ（视频共训权重）具体取值是否在附录（本笔记未取到附录正文）。

---

<!-- QA_LOG -->
（此处记录与用户逐条讨论的 Q&A，未开始）
<!-- /QA_LOG -->
