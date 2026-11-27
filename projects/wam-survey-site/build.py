# -*- coding: utf-8 -*-
"""Build the WAM survey multi-page site (static HTML, self-contained)."""
import html
import pathlib

OUT = pathlib.Path(__file__).resolve().parent

CSS = """
:root{--ink:#1a2430;--dim:#5f6d7b;--line:#dfe6ee;--bg:#f8fafc;--card:#fff;--acc:#0b6bcb;--acc2:#e8f2fc;--warn:#8a5a00;--warnbg:#fdf3e0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;line-height:1.7}
a{color:var(--acc)} a:hover{text-decoration:underline}
.nav{background:#0b2a4a;color:#fff;padding:10px 20px;display:flex;gap:18px;flex-wrap:wrap;align-items:center;font-size:14px}
.nav a{color:#d7e6f7;text-decoration:none}.nav a:hover{color:#fff}
.nav b{color:#fff;font-size:15px}
.wrap{max-width:1040px;margin:0 auto;padding:26px 20px 90px}
h1{font-size:26px;margin:6px 0 8px}
h2{font-size:20px;margin:26px 0 10px;padding-left:10px;border-left:4px solid var(--acc);color:#0b2a4a}
h3{font-size:16px;margin:16px 0 8px;color:#123c66}
p{margin:8px 0}
.meta{color:var(--dim);font-size:13.5px;margin-bottom:14px}
.tag{display:inline-block;font-size:12px;border-radius:20px;padding:1px 9px;margin-right:6px;border:1px solid var(--line);background:#f1f5f9;color:var(--dim)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:14px 0}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px;background:#fff}
th{background:#eef4fa;color:#123c66} th,td{border:1px solid var(--line);padding:7px 10px;vertical-align:top;text-align:left}
code{font-family:Consolas,"Cascadia Mono",monospace;font-size:13px;background:#f1f5f9;border:1px solid #e2e9f1;border-radius:4px;padding:1px 5px;word-break:break-word}
pre{background:#0f1c2b;color:#d7e6f7;border-radius:10px;padding:14px 16px;overflow-x:auto;font-size:12.8px;line-height:1.55}
pre code{background:none;border:none;color:inherit;padding:0}
blockquote{margin:10px 0;padding:12px 16px;border-left:3px solid var(--acc);background:var(--acc2);border-radius:0 10px 10px 0}
blockquote .lbl{display:block;font-size:12px;color:var(--dim);margin-bottom:4px}
.warn{background:var(--warnbg);border:1px solid #ecd9ae;border-radius:10px;padding:12px 16px;font-size:14px;color:var(--warn)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.pcard{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.pcard h3{margin:0 0 4px;font-size:15.5px}
.pcard .s{font-size:13px;color:var(--dim)}
.foot{color:var(--dim);font-size:13px;margin-top:18px}
li{margin:4px 0}
"""

NAV = """<div class="nav"><b>WAM 调研</b>
<a href="../index.html">总览</a><a href="../taxonomy.html">原理分类</a>
<a href="../cat-joint.html">联合扩散</a><a href="../cat-latent.html">潜空间</a><a href="../cat-cascaded.html">级联</a><a href="../cat-ar.html">自回归</a><a href="../cat-geo.html">几何锚定</a><a href="../cat-rl.html">RL 后训练</a>
<a href="../code.html">代码对比</a><a href="../refs.html">来源</a></div>"""

NAV_TOP = NAV.replace('../', '')

CATS = [
    ("joint", "联合扩散（Joint WAM）", "视频潜与动作在同一去噪过程中联合生成（MoT 双专家、共享注意力、流匹配）。训练期视频监督塑形表征，推理期可只解码动作。", "代表：Fast-WAM、Motus、DiT4DiT、Rolling-WAM、X-WAM"),
    ("latent", "潜空间预测（Latent / JEPA）", "不生成像素：用冻结视觉编码器（V-JEPA 2.1 / DINOv3）在潜空间预测未来，动作以未来潜为条件。", "代表：V-JEPA Policy、JEPA-WAM、LaWAM、WA-JEPA"),
    ("cascaded", "级联（Cascaded：先视频后动作）", "先生成未来视频/计划，再经 IDM 或策略反解动作；或把世界模型当数据引擎。", "代表：UniPi、LVP、DreamGen、EVA"),
    ("ar", "离散自回归（AR token）", "图像/动作/文本离散为统一词表 token，自回归生成动作或动作块。", "代表：WorldVLA、WLA、GR-1"),
    ("geo", "几何/像素锚定（Geometric / Pixel-anchored）", "动作不表示为低维向量，而表示为点流、热图视频等与视觉同构的载体。", "代表：Action Images、PointWorld"),
    ("rl", "RL 后训练（Post-training）", "SFT 之后用奖励/优势/偏好信号优化；奖励多来自世界模型、IDM、LTM 监视器或专家潜距离。", "代表：VAMPO、GigaBrain"),
]

PAPERS = [
 dict(slug="fast-wam", title="Fast-WAM: Do World Action Models Need Test-time Future Imagination?", title_cn="Fast-WAM：世界动作模型需要测试时未来想象吗？",
   arxiv="2603.16666", cat="joint", org="IIIS Tsinghua / Galaxea AI", date="2026-03",
   claim_en="we propose Fast-WAM, a WAM architecture that retains video co-training during training but skips future prediction at test time.",
   claim_cn="我们提出 Fast-WAM——一种在训练期保留视频共训、但在测试时跳过未来预测的 WAM 架构。",
   contrib_en="Decouples training-time video co-training from inference-time imagination via controlled variants; recovers most of the benefit of pretrained WAM pipelines without embodied pretraining, at 190 ms latency.",
   contrib_cn="用受控变体把「训练期视频共训」与「推理期未来想象」解耦：去掉视频共训掉 8 点，去掉测试时想象只差约 1 点；无具身预训练达到 RoboTwin 91.8 / LIBERO 97.6，延迟 190ms。",
   mech_en="At inference time, Fast-WAM removes the future video branch entirely: only the clean latent tokens of the first observation frame are retained and passed through the video backbone once to produce latent world features for the action expert.",
   mech_cn="推理时整体移除未来视频分支：只保留第一帧的干净潜 token，过一次视频骨干产生潜世界特征，供动作专家使用。",
   code="""# 官方代码（yuantianyuan01/FastWAM）
# 1) proprio 作为 1 个 token 追加到 T5 条件序列（论文未写，代码实锤）
self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim)
proprio_token = self.proprio_encoder(proprio.unsqueeze(1))      # [B,1,D]
context = torch.cat([context, proprio_token], dim=1)            # [B,L+1,D]

# 2) 结构化注意力：动作 token 只能看首帧 token，不能看未来帧
mask[video_seq_len:, :first_frame_tokens] = True                # action -> first frame
mask[video_seq_len:, video_seq_len:] = True                     # action -> action

# 3) 推理：只编码首帧，动作专家 flow matching 采样 10 步（CFG=1.0）
latents = vae.encode(first_frame)                               # [1,C,F,H,W]
latents_video[:, :, 0:1] = latents                              # 只喂首帧
for t in scheduler.steps(10):
    v = action_expert(video_kv_cache=prefill(latents), t=t)
    actions = scheduler.step(v, dt, actions)""",
   code_kind="官方代码（节选，含 config）", code_note="仓库 yuantianyuan01/FastWAM：src/fastwam/models/wan22/fastwam.py、configs/model/fastwam.yaml；act dim=7(Δxyz+Δrpy+gripper)/chunk=32/9 视频帧。",
   metrics=[("RoboTwin 2.0 平均", "91.8%（无具身预训练）", "L3"), ("LIBERO 平均", "97.6%", "L3"), ("真机 w.o. 共训", "成功率 10%", "L3"), ("延迟", "190 ms（RTX 5090D）", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2603.16666"), ("项目页", "https://yuantianyuan01.github.io/FastWAM/"), ("GitHub", "https://github.com/yuantianyuan01/FastWAM")]),

 dict(slug="motus", title="Motus: A Unified Latent Action World Model", title_cn="Motus：统一潜动作世界模型",
   arxiv="2512.13030", cat="joint", org="清华 / 生数（THU-ML）", date="2025-12",
   claim_en="Motus introduces a Mixture-of-Transformer (MoT) architecture to integrate three experts (i.e., understanding, video generation, and action) and adopts a UniDiffuser-style scheduler to enable flexible switching between different modeling modes.",
   claim_cn="Motus 用混合 Transformer（MoT）架构集成理解、视频生成、动作三个专家，并采用 UniDiffuser 式调度器，实现世界模型 / VLA / 逆动力学 / 视频生成 / 视频-动作联合预测五种模式灵活切换。",
   contrib_en="One model, five modes: optical-flow latent actions extract pixel-level delta actions for large-scale action pretraining from video; +15% over X-VLA, +45% over π0.5 in simulation.",
   contrib_cn="一个模型五种模式；用光流学习潜动作（DPFlow→DC-AE→14 维）实现大规模动作预训练；仿真较 X-VLA +15%、较 π0.5 +45%。",
   mech_en="Motus further leverages the optical flow to learn latent actions and adopts a recipe with three-phase training pipeline and six-layer data pyramid, thereby extracting pixel-level 'delta action' and enabling large-scale action pretraining.",
   mech_cn="利用光流学习潜动作，配合三阶段训练与六层数据金字塔，从像素级抽取「delta action」，实现大规模动作预训练。",
   code="""# 潜动作构造（论文 §4.2；L2）
flow   = DPFlow(video_pair)                  # 光流
rgb    = flow_to_rgb(flow)                   # 转 RGB 图
tokens = DC_AE(rgb)                          # 4 × 512 tokens
z_act  = proj(concat(tokens))                # 14 维潜动作（10% 标注数据弱监督）

# 三专家 MoT + UniDiffuser 调度器（§4.1）
h_text = Qwen3_VL(prompt)                    # 理解专家
h_vid  = Wan2_2_5B(video_latents, cond=h_text)   # 生成专家
a      = ActionExpert(h_vid, z_act)          # 动作专家
# 五种模式 = 切换各分支的噪声/条件（VLA / WM / IDM / VGM / joint）""",
   code_kind="论文伪代码（按 §4.1/§4.2 结构化）", code_note="仓库 thu-ml/Motus（README 与 config 可见）；真实执行动作为 14 维 qpos。",
   metrics=[("RoboTwin 2.0", "87.8%（有具身预训练）", "L3"), ("LIBERO", "97.7%", "L3"), ("潜动作", "14 维（光流→DC-AE）", "L2")],
   links=[("arXiv", "https://arxiv.org/abs/2512.13030"), ("GitHub", "https://github.com/thu-ml/Motus")]),

 dict(slug="dit4dit", title="DiT4DiT: Jointly Modeling Video Dynamics and Actions", title_cn="DiT4DiT：联合建模视频动态与动作",
   arxiv="2603.10448", cat="joint", org="Mondo Robotics / HKUST", date="2026-03",
   claim_en="DiT4DiT, an end-to-end Video-Action Model that couples a video Diffusion Transformer with an action Diffusion Transformer in a unified cascaded framework.",
   claim_cn="DiT4DiT 是一个端到端视频-动作模型，把视频扩散 Transformer 与动作扩散 Transformer 耦合进统一的级联框架。",
   contrib_en="Uses intermediate denoising features (not reconstructed frames) as temporally grounded action conditions; dual flow-matching with decoupled timesteps and noise scales; 98.6% LIBERO, 50.8% RoboCasa-GR1, >10x sample efficiency.",
   contrib_cn="不用重建帧，而用视频去噪中间层特征作动作条件；双流匹配（τv/τf/τa 解耦）；LIBERO 98.6%、RoboCasa-GR1 50.8%、样本效率 >10×。",
   mech_en="Instead of relying on reconstructed future frames, DiT4DiT extracts intermediate denoising features from the video generation process and uses them as temporally grounded conditions for action prediction.",
   mech_cn="不依赖重建的未来帧，而是从视频生成过程抽取中间去噪特征，作为时间接地条件供动作预测使用。",
   code="""# 双流匹配：τv 均匀采样、τf 固定、τa ~ Beta(α,β)（§4.3）
def train_step(video, actions):
    τv = uniform(0, 1)
    lat_v = add_noise(video_lat, τv)
    h_f   = video_dit(lat_v, τv)                  # 前向钩子，取固定步 τf 的中间隐状态
    h_tf  = hook_layer(h_f, layer=17)             # temporally grounded condition
    τa    = Beta(1.5, 1.0).sample()
    v_a   = action_dit(add_noise(actions, τa), τa, cond=h_tf)
    return fm_loss(v_a, actions) + fm_loss(v_video, video)""",
   code_kind="论文伪代码（按 §3/§4 结构化）", code_note="config 实锤：LIBERO action_dim=8、horizon=8；G1 真机 action_dim=32、horizon=16；action_type=joint_position。",
   metrics=[("LIBERO", "98.6%", "L3"), ("RoboCasa-GR1", "50.8%", "L3"), ("样本效率", ">10×", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2603.10448"), ("项目页", "https://dit4dit.github.io/")]),

 dict(slug="rolling-wam", title="Rolling-WAM: World Action Models with Rolling Imagination", title_cn="Rolling-WAM：滚动想象的世界动作模型",
   arxiv="2609.30247", cat="joint", org="USC / TRI 等", date="2026-09",
   claim_en="We present Rolling-WAM, a formulation that distributes joint denoising across successive replanning cycles.",
   claim_cn="Rolling-WAM 把联合去噪分摊到连续的多个重规划周期上。",
   contrib_en="Maintains a sliding window of video-action chunks at staggered noise levels; fully denoises the imminent chunk each step while partially refining farther ones; 4.5x steady-state replanning speedup.",
   contrib_cn="维护错峰噪声的视频-动作块滑动窗口：每步只完全去噪最近的块、部分精炼更远的块；稳态重规划提速 4.5×。",
   mech_en="At each step, a rolling noise schedule fully denoises the imminent action chunk for execution, while partially refining farther-future chunks.",
   mech_cn="每一步的滚动噪声调度完全去噪即将执行的动作块，同时部分精炼更远的未来块。",
   code="""# 错峰噪声 + 滑动窗口（§III；参数见 03b 笔记）
def rolling_step(window, obs, K=16, W=5, rho=5):
    for j, chunk in enumerate(window):                # j=1..W
        σ = rho * (j/W) / (1 + (rho-1) * (j/W))       # 近端低噪声、远端高噪声
        chunk.add_noise(σ)
    for _ in range(2):                                # 稳态每周期 2 步
        video, action = denoise_joint(window, obs, N=10)
    window.pop(0); window.append(new_chunk(action))
    return action[:K]                                 # 执行 16 步""",
   code_kind="论文伪代码（按 §III 结构化）", code_note="参数 K=16/W=5/N=10/ρ=5 来自 L2 笔记 03b；真机 G1 78-D（64 SONIC + 7×2）。",
   metrics=[("LIBERO", "98.1%", "L3"), ("RoboTwin 2.0", "93.3%（100 rollouts/档）", "L3"), ("真机 G1", "85.0%（3 任务×20 trials）", "L3"), ("重规划", "215 ms（4.5×）", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2609.30247"), ("项目页", "https://rolling-wam.github.io/")]),

 dict(slug="x-wam", title="X-WAM: Unified 4D World Action Modeling with Asynchronous Denoising", title_cn="X-WAM：统一 4D 世界动作建模与异步去噪",
   arxiv="2604.26694", cat="joint", org="清华 / 地瓜 等", date="2026-04",
   claim_en="We propose X-WAM, a Unified 4D World Model that unifies real-time robotic action execution and high-fidelity 4D world synthesis (video + 3D reconstruction) in a single framework.",
   claim_cn="X-WAM 是统一的 4D 世界模型，在单一框架里同时完成实时机器人动作执行与高保真 4D 世界合成（视频+3D 重建）。",
   contrib_en="Predicts multi-view RGB-D futures; replicates the final blocks of the video DiT into a depth branch; Asynchronous Noise Sampling decodes actions with fewer steps while video uses full steps; 79.2% RoboCasa / 90.7% RoboTwin.",
   contrib_cn="预测多视角 RGB-D 未来；把视频 DiT 末端若干块复制成深度分支；异步噪声采样让动作少步解码、视频满步生成；RoboCasa 79.2%、RoboTwin 90.7%。",
   mech_en="ANS applies a specialized asynchronous denoising schedule during inference, which rapidly decodes actions with fewer steps to enable efficient real-time execution, while dedicating the full sequence of steps to generate high-fidelity video.",
   mech_cn="ANS 在推理时使用专门的异步去噪调度：动作用更少步快速解码以支持实时执行，而视频仍用完整步数保高保真。",
   code="""# ANS：动作与视频不同步数去噪（§3.3/§3.4）
def infer(x0, steps_video=8, steps_action=2):
    video = randn_like(x0); action = randn_like(a0)
    for k in reversed(range(steps_video)):            # 视频走满步
        video = dit(video, t=k)
        if k % (steps_video // steps_action) == 0:    # 动作少步
            action = action_dit(action, t=k, cond=video)
    depth = depth_branch(video)                       # 复制末端块 → 深度重建
    return action, depth""",
   code_kind="论文伪代码（按 §3 结构化）", code_note="动作=双臂 EEF 位姿+夹爪「变化量」（维度附录未取到）；视频 K=32。",
   metrics=[("RoboCasa", "79.2%", "L3"), ("RoboTwin 2.0", "90.7%", "L3"), ("预训练", ">5,800 小时机器人数据", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2604.26694"), ("项目页", "https://sharinka0715.github.io/X-WAM/")]),

 dict(slug="vjepa-policy", title="V-JEPA Policy: Building Effective WAMs on Predictive Visual Latents", title_cn="V-JEPA Policy：在预测性视觉潜空间上构建有效 WAM",
   arxiv="2609.37250", cat="latent", org="清华 / 中科院等", date="2026-09",
   claim_en="we introduce V-JEPA Policy, a simple framework that builds a WAM on the latent space of a frozen V-JEPA 2.1 encoder.",
   claim_cn="我们提出 V-JEPA Policy——一个在冻结 V-JEPA 2.1 编码器的潜空间上构建 WAM 的简洁框架。",
   contrib_en="An instruction-conditioned future-latent predictor and a flow-matching action expert are jointly learned from scratch; 0.9B total / 0.6B trainable; beats video-generative WAMs under distribution shift.",
   contrib_cn="指令条件的未来潜预测器 + 流匹配动作专家从零联合训练；0.9B 总参/0.6B 可训；分布偏移下优于视频生成式 WAM。",
   mech_en="We encode the task instruction with a frozen T5 encoder and concatenate the resulting text tokens with one state token projected from q_t to form a conditioning sequence.",
   mech_cn="用冻结 T5 编码指令，把文本 token 与一个由 q_t 投影的状态 token 拼接成条件序列。",
   code="""# 冻结编码器 + 未来潜预测器 + flow-matching 动作专家（§3.1）
ctx   = VJEPA21.encode(obs)                       # 冻结 ViT-L，tubelet=2
cond  = concat(T5(instruction), proj(q_t))        # 文本 + 1 个 proprio token
Z_hat = predictor(ctx, cond, mask_queries)        # 单次前向预测未来潜
kv    = predictor.layerwise_kv                    # 逐层 K/V 作为动作条件
a     = action_expert.sample(cond=kv, steps=10)   # 0.1B DiT，流匹配""",
   code_kind="论文伪代码（按 §3 结构化）", code_note="LIBERO 97.25 / LIBERO-Plus 79.25 / RoboCasa-GR1 50.92 / 真机 45%→80%（2 任务×20 trials）。",
   metrics=[("LIBERO", "97.25%", "L3"), ("LIBERO-Plus", "79.25%（DROID 预训练后 91.50）", "L3"), ("真机", "45%→80%", "L3"), ("参数", "0.9B / 0.6B 可训", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2609.37250")]),

 dict(slug="jepa-wam", title="JEPA-WAM: Connecting Generated Visual Instructions to WAMs via JEPA Latents", title_cn="JEPA-WAM：用 JEPA 潜表征把生成式视觉指令接入 WAM",
   arxiv="2609.20277", cat="latent", org="清华 等", date="2026-08",
   claim_en="we introduce JEPA-WAM, which augments each text instruction with a bank of stochastically generated visual instructions, providing diverse visual cues for instruction following.",
   claim_cn="JEPA-WAM 为每条文本指令配一个随机生成的视觉指令库，为指令跟随提供多样的视觉线索。",
   contrib_en="Off-the-shelf text-to-image model samples a bank of goal images; a frozen V-JEPA 2.1 encoder compresses them into 64 goal tokens; cross-attends both video and action experts; +10~27 pp over π0/Fast-WAM.",
   contrib_cn="用现成文生图模型采样目标图库；冻结 V-JEPA 2.1 编码为 64 个 goal token；同时 cross-attend 视频与动作专家；真机三设定较 π0/Fast-WAM +10~27 个点。",
   mech_en="Let E_J denote a frozen V-JEPA 2.1 encoder and let P_ψ denote a learnable projector... Z_ℓ = P_ψ(Pool_{8×8}(E_J([g, g]))) ∈ R^{64×d_c}.",
   mech_cn="E_J 为冻结 V-JEPA 2.1 编码器、P_ψ 为可学习投影：把生成图复制成两帧编码、8×8 池化后得 64 个 goal token。",
   code="""# 离线建库 + 冻结 JEPA 编码 + 条件序列（§3.1/§3.2）
bank = [T2I(prompt=instruction, seed=k) for k in range(K)]   # K=1024
Z    = P_psi(pool8x8(E_J([g, g])))                            # 64 tokens
C    = concat(C_lang, P_s(state), Z)                          # 条件序列
video = video_expert(C, first_frame)                          # 方向性掩码（继承 Fast-WAM）
action = action_expert(C)                                     # flow matching""",
   code_kind="论文伪代码（按 §3 结构化）", code_note="真机三设定 87.3/74.5/80.9；K=1024 bank；JEPA 编码器 ViT-B/16 80M。",
   metrics=[("真机 ID/OOD 场景/OOD 指令", "87.3 / 74.5 / 80.9%", "L3"), ("相对 π0", "+10.0 pp 起", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2609.20277")]),

 dict(slug="lawam", title="LaWAM: Latent World Action Models", title_cn="LaWAM：潜空间世界-动作模型",
   arxiv="2606.15768", cat="latent", org="CoRL 2026", date="2026-06",
   claim_en="We obtain LaWM by revisiting latent action models (LAMs) in the latent space of frozen visual encoders, such as DINOv3.",
   claim_cn="我们在冻结视觉编码器（如 DINOv3）的潜空间里重新审视潜动作模型（LAM），得到 LaWM。",
   contrib_en="Three-stage latent world-action model: policy prior predicts a latent action, LaWM decodes a latent visual subgoal, an Alternate-DiT action expert generates actions; 98.6% LIBERO at 187 ms.",
   contrib_cn="三段式潜世界-动作模型：策略先验预测潜动作→LaWM 解码潜视觉子目标→Alternate-DiT 动作专家出动作；LIBERO 98.6%、187ms。",
   mech_en="p(a_{1:T}, û_T, ẑ | o,l) = p_θ(ẑ|o,l)[Policy Prior] p_ω(û_T|u,ẑ)[LaWM] p_η(a_{1:T}|o,l,u,û_T)[Action Expert]",
   mech_cn="联合分布分解为三段：策略先验给潜动作 ẑ、LaWM 给潜子目标 û_T、动作专家以三者条件出动作。",
   code="""# 三段式潜空间管线（§3.1）
u   = DINOv3(o)                                  # 冻结特征
z   = policy_prior(o, l)                         # 潜动作（编码器推断）
u_T = LaWM.decode(u, z)                          # 潜视觉子目标（不重建像素）
a   = AlternateDiT(o, l, u, u_T)                 # 语义流/动态流交替去噪""",
   code_kind="论文伪代码（按 §3 结构化）", code_note="LaWM 230M、总 2.3B、187ms；LIBERO 98.6 / RoboTwin 91.22。",
   metrics=[("LIBERO", "98.6%", "L3"), ("延迟", "187 ms（约 24× 快于像素级）", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2606.15768")]),

 dict(slug="wa-jepa", title="WA-JEPA: Goal-Aware Predictive Feedback Action Refinement", title_cn="WA-JEPA：目标感知的预测反馈动作精炼",
   arxiv="2611.19183", cat="latent", org="HIT / 海康 等", date="2026-11",
   claim_en="The model first generates an action proposal and predicts the resulting future scenes and proprioceptive states, then refines the proposal through two complementary feedback paths, using the task goal as a reference.",
   claim_cn="模型先生成动作提案并预测其后果（未来场景与本体状态），再以任务目标为参照、经两条互补反馈路径精炼提案。",
   contrib_en="GDRF uses goal-discrepancy residuals to guide chunk-wise refinement; FSCF preserves fine-grained predicted scene context; no candidate-action search; 10.8-32.6x faster complete-plan generation.",
   contrib_cn="GDRF 用「目标-预测」残差逐块修正；FSCF 保留细粒度预测场景上下文；免候选搜索；完整计划生成快 10.8–32.6×。",
   mech_en="Goal-discrepancy residual feedback (GDRF) uses differences between the latent representations of predicted scenes and the task goal to guide chunk-wise action refinement.",
   mech_cn="目标差异残差反馈（GDRF）用预测场景与任务目标的潜表征之差，引导逐 chunk 的动作精炼。",
   code="""# GDRF + FSCF 精炼循环（§3.3；R=4）
for r in range(R):
    Q = proposal_layer(Q, [M, G_hat, Z_pred])       # self-attn + cross-attn
    A = decoder(LN(Q)); A = range_map(A)            # 动作 chunk [H, b*d_a]
    Z_pred, P_hat = world_model(M, Z0, A)           # 预测后果（时序掩码并行）
    for h in range(H):
        U = [Q[h], mean(Z_pred[h]), G_bar - mean(Z_pred[h]), proj(P_hat[h])]
        Q[h] = Q[h] + sigmoid(gamma_r) * F_r(U)     # GDRF 残差修正
return A                                             # 只执行最终阶段""",
   code_kind="论文伪代码（按 §3 结构化）", code_note="R=4（README）；短程 TwoRoom/PushT/Cube 100/96/100%。",
   metrics=[("短程目标到达", "100% / 96% / 100%", "L3"), ("完整计划生成", "10.8–32.6× 加速", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2611.19183"), ("GitHub", "https://github.com/Dovisya/WA-JEPA")]),

 dict(slug="unipi", title="UniPi: Learning Universal Policies via Text-Guided Video Generation", title_cn="UniPi：用文本引导视频生成学习通用策略",
   arxiv="2302.00111", cat="cascaded", org="MIT / Google Brain（NeurIPS 2023）", date="2023-02",
   claim_en="we cast the sequential decision making problem as a text-conditioned video generation problem, where, given a text-encoded specification of a desired goal, a planner synthesizes a set of future frames depicting its planned actions in the future, after which control actions are extracted from the generated video.",
   claim_cn="我们把序列决策问题表述为「文本条件视频生成」问题：给定目标文本，规划器合成一组描绘未来动作的未来帧，再从生成视频中提取控制动作。",
   contrib_en="The founding video-WAM: text→video plan→inverse dynamics→open-loop execution; unifies heterogeneous state/action spaces in image space.",
   contrib_cn="视频-WAM 的奠基作：文本→生成视频计划→逆动力学提取动作→开环执行；把异构状态/动作空间统一到图像空间。",
   mech_en="We train a small model to estimate actions given input images.",
   mech_cn="训练一个小型逆动力学模型，从输入图像估计动作。",
   code="""# 视频规划 + 逆动力学（§3.1/§3.2）
video = video_diffusion(prompt=goal, first_frame=o0, tiled=o0)  # 层级规划：先粗后超分
frames = sample(video, H)                                       # 未来帧
actions = IDM(frames)                                           # 小模型回归动作序列
execute_open_loop(actions)""",
   code_kind="论文伪代码（按 §3 结构化）", code_note="T5 条件 + 首帧逐帧 tile 保持一致性；IDM 与规划器分离训练。",
   metrics=[("定位", "视频-WAM 奠基（2023）", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2302.00111"), ("项目页", "https://universal-policy.github.io/")]),

 dict(slug="lvp", title="LVP: Large Video Planner Enables Generalizable Robot Control", title_cn="LVP：大视频规划器实现可泛化机器人控制",
   arxiv="2512.15840", cat="cascaded", org="UC Berkeley / MIT 等", date="2025-12",
   claim_en="We curate an internet-scale video dataset of human activities and task demonstrations, and train, for the first time at a foundation-model scale, an open video model for generative robotics planning.",
   claim_cn="我们整理互联网规模的人类活动与任务演示视频，首次以基础模型规模训练了一个用于生成式机器人规划的开源视频模型。",
   contrib_en="Foundation-scale open video planner; zero-shot video plans for novel scenes/tasks, post-processed into executable actions; real-robot execution.",
   contrib_cn="基础模型规模的开放视频规划器；对新场景/新任务零样本生成视频计划，经后处理成可执行动作；真机可执行。",
   mech_en="The model produces zero-shot video plans for novel scenes and tasks, which we post-process to extract executable robot actions.",
   mech_cn="模型对新场景与任务零样本生成视频计划，我们再做后处理以提取可执行动作。",
   code="""# 视频计划 → 可执行动作（§3 后处理）
plan_video = video_planner(prompt, image_obs)        # 基础模型规模
frames     = decode(plan_video)
actions    = postprocess(frames)                     # 轨迹/关键点 → 机器人动作
execute(actions)""",
   code_kind="论文伪代码（按摘要/§3 结构化）", code_note="EVA 管线以 LVP 为初始化（见 EVA 页）。",
   metrics=[("规模", "互联网级视频语料、基础模型规模", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2512.15840"), ("项目页", "https://www.boyuan.space/large-video-planner/")]),

 dict(slug="dreamgen", title="DreamGen: Unlocking Generalization in Robot Learning through Video World Models", title_cn="DreamGen：用视频世界模型解锁机器人学习泛化",
   arxiv="2505.12705", cat="cascaded", org="NVIDIA GEAR", date="2025-05",
   claim_en="We introduce DreamGen, a simple yet highly effective 4-stage pipeline for training robot policies that generalize across behaviors and environments through neural trajectories - synthetic robot data generated from video world models.",
   claim_cn="DreamGen 是一个简单却高效的 4 阶段管线：用视频世界模型生成「神经轨迹」合成数据，训练可跨行为/环境泛化的机器人策略。",
   contrib_en="Fine-tune I2V world model on target robot → generate videos → pseudo-label actions (latent action model or IDM) → behavior cloning; 22 new behaviors from one pick-and-place task's teleop data.",
   contrib_cn="目标机器人上微调 I2V 世界模型→批量生成视频→用潜动作模型或 IDM 伪标注→行为克隆；仅凭单任务遥操作数据解锁 22 个新行为。",
   mech_en="we recover pseudo-action sequences using either a latent action model or an inverse-dynamics model (IDM).",
   mech_cn="使用潜动作模型或逆动力学模型（IDM）恢复伪动作序列。",
   code="""# 4 阶段数据引擎（§2）
wm = finetune_i2v(video_world_model, robot_demos)      # LoRA
videos = wm.generate(initial_frames, instructions)     # 批量生成
actions = IDM(videos)  or  LAPA(videos)                # 伪标注
policy = BC(videos + actions)                          # DP / π0 / GR00T N1""",
   code_kind="论文伪代码（按 §2 结构化）", code_note="非 RL：单向管线，无回报回环（07 线对照结论）。",
   metrics=[("新行为", "22 个（单任务遥操作数据）", "L3"), ("新环境", "28.5%（从 0 起）", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2505.12705"), ("GitHub", "https://github.com/NVIDIA/GR00T-Dreams")]),

 dict(slug="eva", title="EVA: Aligning Video World Models with Executable Robot Actions", title_cn="EVA：用可执行动作对齐视频世界模型",
   arxiv="2603.17808", cat="cascaded", org="CUHK-SZ / DexForce", date="2026-03",
   claim_en="links visual and action-level improvements through a shared alignment signal",
   claim_cn="通过共享对齐信号，把视觉层与动作层的改进联系起来。",
   contrib_en="Frozen IDM decodes generated videos to joint actions; jerk/acc Huber + velocity/acceleration limit penalties form the reward; Flow-GRPO updates only generator LoRA; RoboTwin +6.48, LIBERO +6.5~7.5.",
   contrib_cn="冻结 IDM 把生成视频解码成关节动作；用 jerk/acc Huber + 速度/加速度限幅惩罚构成奖励；Flow-GRPO 只更新生成器 LoRA；RoboTwin +6.48、LIBERO +6.5~7.5。",
   mech_en="We compute velocity v_t=(q_{t+1}−q_t)/Δt, acceleration α_t=(v_{t+1}−v_t)/Δt, and jerk j_t=(α_{t+1}−α_t)/Δt",
   mech_cn="由解码关节序列计算速度、加速度与 jerk，用于平滑度惩罚。",
   code="""# 动作空间奖励 + Flow-GRPO（§4.2/§4.3）
q = IDM(video)                                  # 冻结 IDM
v, a, j = diff(q), diff(v), diff(a)
R = -(λj*huber(j) + λα*huber(a) + λv*relu(|v|-vmax).pow(2) + λa*relu(|a|-amax).pow(2))
Adv = (R - R.mean()) / (R.std() + eps)          # 组归一化
loss = min(ratio*Adv, clip(ratio,1-ε,1+ε)*Adv) - β*KL(π||π_ref)
update(generator LoRA)""",
   code_kind="论文伪代码（按 §4 结构化）", code_note="Kinematic 通过率 70.5→91.4；LVP→SFT→RL 三级管线。",
   metrics=[("Kinematic 通过率", "70.5→91.4", "L3"), ("RoboTwin 21 任务", "45.81→52.29", "L3"), ("LIBERO", "+6.5~7.5 点", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2603.17808"), ("项目页", "https://eva-project-page.github.io/")]),

 dict(slug="worldvla", title="WorldVLA: Towards Autoregressive Action World Model", title_cn="WorldVLA：迈向自回归动作世界模型",
   arxiv="2506.21539", cat="ar", org="阿里达摩院", date="2025-06",
   claim_en="All texts, actions, and images are discretized into tokens and are trained under the autoregressive manner.",
   claim_cn="文本、动作、图像全部离散为 token，并以自回归方式训练。",
   contrib_en="One Chameleon-based LLM does action model + world model; shared 65,536 vocab (8,192 image + 256 action tokens); action-mask blocks prior actions to curb AR error accumulation.",
   contrib_cn="基于 Chameleon 的单一 LLM 兼任动作模型与世界模型；共享 65,536 词表（8,192 图像 + 256 动作 token）；「屏蔽先前动作」掩码抑制自回归误差累积。",
   mech_en="The text inputs are \"What action should the robot take to + task instruction + ?\". The overall token sequence is: [BOS]{text}[BOI]{image}...[EOI]×M [EOS][BOA]{action}...[EOA]×K",
   mech_cn="输入为固定模板文本 + M 帧图像 token，输出 K 个动作 token；动作块用并行掩码生成。",
   code="""# 序列与掩码（§3.3）
seq = [BOS] + text_tokens + repeat([BOI]+img_tokens+[EOI], M) + [BOA] + action_tokens + [EOA]
# action 分支掩码：当前动作只能看文本+图像，不能看先前动作
mask = causal_mask | block_prior_actions()
loss = CE(action_tokens) + alpha * CE(image_tokens)     # alpha=0.04""",
   code_kind="论文伪代码（按 §3.2/§3.3 结构化）", code_note="动作=7 tokens（3 相对位置+3 相对角度+1 夹爪），每维 256 bins；LIBERO 81.8(512)。",
   metrics=[("LIBERO", "81.8%（512 分辨率）", "L2"), ("词表", "65,536（含 8192 图像 + 256 动作）", "L2")],
   links=[("arXiv", "https://arxiv.org/abs/2506.21539"), ("GitHub", "https://github.com/alibaba-damo-academy/WorldVLA")]),

 dict(slug="wla", title="WLA: World-Language-Action Model", title_cn="WLA：世界-语言-动作模型",
   arxiv="2606.05979", cat="ar", org="上海 AI Lab 等", date="2026-06",
   claim_en="We propose world-language-action (WLA) models as a new class of embodied foundation models.",
   claim_cn="我们提出世界-语言-动作（WLA）模型，作为一类新的具身基础模型。",
   contrib_en="AR Transformer jointly predicts textual subtask, subgoal image and actions; world prediction affects actions implicitly via meta-queries; 2B active params, 40 ms per inference on RTX 5090.",
   contrib_cn="AR Transformer 联合预测文本子任务、子目标图像与动作；世界预测经 meta-query 隐式影响动作（推理可关闭）；2B 激活参数、RTX 5090 上 40ms/次。",
   mech_en="WLA takes textual instructions, images, and robot states as inputs to jointly predict textual subtasks, subgoal images, and robot actions",
   mech_cn="WLA 以文本指令、图像与机器人状态为输入，联合预测文本子任务、子目标图像与机器人动作。",
   code="""# 自回归：语义子任务 + 子目标图 + 动作（§3）
h = AR_backbone(o, l, state)
subtask = head_text(h)                 # 语义级意图
goal    = head_image(h)                # 子目标图（World Expert）
action  = head_action(h, meta_query)   # 动作（Action Expert）
# 推理可关闭世界预测；开启则做 test-time scaling""",
   code_kind="论文伪代码（按 §3 结构化）", code_note="RoboTwin2.0 Clean 92.94%；RMBench 56.5%；40ms（RTX 5090）。",
   metrics=[("RoboTwin2.0 Clean", "92.94%", "L3"), ("RMBench", "56.5%", "L3"), ("推理", "40 ms（2B 激活）", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2606.05979")]),

 dict(slug="gr1", title="GR-1: Unleashing Large-Scale Video Generative Pre-training", title_cn="GR-1：释放大规模视频生成预训练（用于视觉机器人操作）",
   arxiv="2312.13139", cat="ar", org="ByteDance Research", date="2023-12",
   claim_en="we train GR-1 to predict delta XYZ positions and delta Euler angles for arm actions and binary gripper actions.",
   claim_cn="我们训练 GR-1 预测机械臂动作的 ΔXYZ 位置与 Δ欧拉角，以及二值夹爪动作。",
   contrib_en="GPT-style causal transformer on video-pretrained tokens (Ego4D); predicts future frames + actions; CALVIN 94.9% over 5-task chain.",
   contrib_cn="在视频预训练 token（Ego4D）上训练 GPT 式因果 transformer；同时预测未来帧与动作；CALVIN 五任务链 94.9%。",
   mech_en="The robot state s contains the 6D pose of the robot end-effector and a binary status of the gripper",
   mech_cn="机器人状态 s 含末端执行器 6D 位姿与二值夹爪状态。",
   code="""# 因果 transformer + 线性动作头（§3.2）
h = GPT(video_tokens, lang_tokens, state_tokens)   # [OBS]/[ACT] 掩蔽
obs_pred = head_obs(h)                             # 未来帧（像素 MSE）
a = head_act(h)                                    # ΔXYZ+Δ欧拉+夹爪
loss = L_arm(SmoothL1) + L_gripper(BCE) + L_video""",
   code_kind="论文伪代码（按 §3 结构化）", code_note="动作 7 维（6 连续 + 1 二值）；无 action chunk（horizon=1）。",
   metrics=[("CALVIN", "94.9%（5 任务链）/ 平均长度 4.21", "L3"), ("10% 数据", "77.8%", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2312.13139"), ("GitHub", "https://github.com/bytedance/GR-1")]),

 dict(slug="action-images", title="Action Images: End-to-End Policy Learning via Multiview Video Generation", title_cn="Action Images：多视角视频生成的端到端策略学习",
   arxiv="2604.06168", cat="geo", org="—", date="2026-04",
   claim_en="we translate 7-DoF robot actions into interpretable action images: multi-view action videos that are grounded in 2D pixels and explicitly track robot-arm motion.",
   claim_cn="我们把 7-DoF 机器人动作转成可解释的「动作图像」：以 2D 像素接地、显式跟踪机械臂运动的多视角动作视频。",
   contrib_en="Actions encoded as RGB Gaussian heatmaps in multi-view image space; fine-tunes Wan 2.2 to co-generate observation and action videos; recovers control by multi-view ray casting.",
   contrib_cn="动作编码为多视角图像空间中的 RGB 高斯热图；微调 Wan 2.2 联合生成观测与动作视频；经多视角反投影恢复控制。",
   mech_en="Action Images encodes 7-DoF robot actions as RGB Gaussian heatmaps in multi-view image space",
   mech_cn="把 7-DoF 动作编码为多视角图像空间中的 RGB 高斯热图。",
   code="""# 动作 → 热图视频 → 反投影（§3）
heat = gaussian_heatmap(ee_pose, gripper, views)   # 3 通道多视角热图视频
vid  = wan22.generate(heat)                          # 联合生成观测+动作视频
ee   = ray_cast(vid, views)                          # 多视角反投影恢复 3D
execute(ee)""",
   code_kind="论文伪代码（按 §3 结构化）", code_note="zero-shot 策略：视频骨干直接当策略。",
   metrics=[("动作编码", "7-DoF → 多视角 RGB 热图", "L2")],
   links=[("arXiv", "https://arxiv.org/abs/2604.06168")]),

 dict(slug="pointworld", title="PointWorld: Scaling 3D World Models for In-The-Wild Manipulation", title_cn="PointWorld：面向野外操作扩展 3D 世界模型",
   arxiv="2601.03782", cat="geo", org="—", date="2026-01",
   claim_en="By representing actions as 3D point flows instead of embodiment-specific action spaces (e.g., joint positions),",
   claim_cn="通过把动作表示为 3D 点流、而非本体专属动作空间（如关节位置），……",
   contrib_en="Unifies state and action as 3D point flows; predicts per-pixel 3D displacement; 0.1 s inference, plugged into MPC/MPPI.",
   contrib_cn="把状态与动作统一为 3D 点流；预测逐像素 3D 位移；0.1s 推理，可嵌入 MPC/MPPI。",
   mech_en="PointWorld learns full-scene 3D point flow conditioned on robot point-flow actions in a single feedforward pass",
   mech_cn="PointWorld 单次前向学习以机器人点流动作为条件的全场景 3D 点流。",
   code="""# 点流预测 + MPPI（§3/§4）
pts = scene_points(rgbd)                       # DINOv3 特征
flow = point_transformer(pts, action_flow)     # 3D 位移场
for k in range(K):                             # MPPI
    cand = sample_action_flow()
    cost = task_cost(flow, cand)
best = topk(cand, cost); execute(best)         # 0.1s / rollout""",
   code_kind="论文伪代码（按摘要/§3 结构化）", code_note="H=10@0.1s；动作=3D 点流。",
   metrics=[("推理", "0.1 s/rollout", "L2")],
   links=[("arXiv", "https://arxiv.org/abs/2601.03782")]),

 dict(slug="vampo", title="VAMPO: Policy Optimization for Improving Visual Dynamics", title_cn="VAMPO：改进视觉动力学的策略优化",
   arxiv="2603.19370", cat="rl", org="浙大 等", date="2026-03",
   claim_en="we compute a verifiable relative latent-consistency reward",
   claim_cn="我们计算一种可验证的相对潜一致性奖励。",
   contrib_en="Treats multi-step denoising of the video predictor as an MDP; GRPO with expert-latent consistency reward; only the first step is stochastic (SDE); +?? on downstream success.",
   contrib_cn="把视频预测器多步去噪当 MDP；用 expert-latent 一致性奖励做 GRPO；只有首步随机（SDE）；实测下游成功率提升明显。",
   mech_en="introducing stochasticity at all denoising steps leads to severe credit assignment issues",
   mech_cn="在所有去噪步注入随机性会导致严重的信用分配问题。",
   code="""# 官方实现（OpenHelix-Team/VAMPO）
# 奖励（latent_reward_fn）：r = -(λ1*MAE + λ2*(1-cos))
mae = l1(pred_latents, true_latents).mean([2,3,4]); r = -mae + cos_sim(pred, true)
# 优势：组归一化（GRPO，无 critic）
adv = (r.reshape(bs, G, F) - mean(axis=1)) / (std(axis=1) + 1e-8)
# 仅首步 SDE + 截断代理目标
ratio = exp(logp - logp_old); loss = mean(min(-adv*ratio, -adv*clamp(ratio,.8,1.2)))""",
   code_kind="官方代码（节选）", code_note="仓库 OpenHelix-Team/VAMPO：grpo/reward_fn.py、step1_train_svd_grpo.py；G=8、lr=1e-6、I=20、clip=0.2。",
   metrics=[("优化器", "GRPO（组归一化）", "L3"), ("训练对象", "仅 VPM 去噪器", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2603.19370"), ("GitHub", "https://github.com/OpenHelix-Team/VAMPO")]),

 dict(slug="gigabrain", title="GigaBrain-0.5M*: A VLA That Learns From World Model-Based RL", title_cn="GigaBrain-0.5M*：从世界模型强化学习中学到的 VLA",
   arxiv="2602.12099", cat="rl", org="GigaBrain Team", date="2026-02",
   claim_en="we extend the KL-regularized reinforcement learning framework to our augmented state space",
   claim_cn="我们把 KL 正则化强化学习框架扩展到增强状态空间。",
   contrib_en="RAMP four-stage loop: world-model pretrain → advantage-conditioned policy finetune (KL-RL closed form as weighted NLL) → HILR rollouts → joint continual training; +~30% over RECAP.",
   contrib_cn="RAMP 四阶段闭环：世界模型预训练→优势条件化策略微调（KL-RL 闭式解改写为加权 NLL）→HILR rollout→联合持续训练；较 RECAP 约 +30%。",
   mech_en="A(s_t,a_t)=Σ γ^k r_{t+k} + γ^n v_{t+n} − v_t; advantages are discretized into a binary indicator",
   mech_cn="用 n 步 TD 计算优势，并把优势二值化为指示变量 I。",
   code="""# RAMP：奖励/优势/损失（§3.2）
r_t = 0 if success at T else (-C_fail if fail else -1)   # 稀疏 episode 奖励
A   = n_step_td(r, v)                                     # 世界模型价值
I   = 1[A > ε]                                            # 二值优势
L   = -log π(a|o,z,l) - α*log π(a|I,o,z,l)                # 加权 NLL（RL 闭式解）
mask_world_tokens(p=0.2)                                  # 随机注意力掩码""",
   code_kind="论文伪代码（按 §3.2 结构化）", code_note="4K 小时世界模型数据；>10K 小时 VLA 预训练；掩码 p=0.2。",
   metrics=[("相对 RECAP", "≈30% 提升", "L3"), ("世界模型数据", "4K 小时", "L3")],
   links=[("arXiv", "https://arxiv.org/abs/2602.12099"), ("项目页", "https://gigabrain05m.github.io")]),
]

# ---------------- render helpers ----------------

def esc(s): return html.escape(str(s))

def page(title, body, nav=NAV, depth=0):
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)} · WAM 调研</title><style>{CSS}</style></head>
<body>{nav}<div class="wrap">{body}<p class="foot">本站内容基于论文原文（verbatim 引文 + 中译标注）与官方代码；未给出处一律标注。生成于 2026-11。</p></div></body></html>"""

def paper_card(p):
    return f"""<div class="pcard"><h3><a href="papers/{p['slug']}.html">{esc(p['title_cn'])}</a></h3>
<div class="s">{esc(p['title'])}</div>
<div class="s" style="margin-top:6px">{esc(p['contrib_cn'][:70])}…</div>
<div style="margin-top:8px"><span class="tag">{esc(p['cat'])}</span><span class="tag">{p['date']}</span><span class="tag">arXiv {p['arxiv']}</span></div></div>"""

def build_index():
    cats = "".join(f'<div class="card"><h3><a href="cat-{k}.html">{esc(name)}</a></h3><p style="color:var(--dim)">{esc(desc)}</p><p class="s">{esc(rep)}</p></div>' for k, name, desc, rep in CATS)
    grid = "".join(paper_card(p) for p in PAPERS)
    body = f"""<h1>具身智能 WAM 调研：按原理分类的论文精读站</h1>
<p class="meta">世界-动作模型（World Action Model）：联合建模未来状态与动作 p(o′,a|o,l)；本站在 68 篇次精读基础上，按<b>方法原理</b>重构为 6 类，并给出每篇的<b>核心贡献（中英）、原文引文（中英）与核心代码</b>。</p>
<div class="card"><b>阅读顺序建议</b><ol><li><a href="taxonomy.html">原理分类总览</a>（先建立坐标系）</li><li>按类进入：<a href="cat-joint.html">联合扩散</a> / <a href="cat-latent.html">潜空间</a> / <a href="cat-cascaded.html">级联</a> / <a href="cat-ar.html">自回归</a> / <a href="cat-geo.html">几何锚定</a> / <a href="cat-rl.html">RL 后训练</a></li><li><a href="code.html">实战手册：代码对比</a>（同一问题的不同实现）</li><li><a href="refs.html">来源与原文索引</a></li></ol></div>
<h2>六类原理</h2>{cats}
<h2>论文列表（{len(PAPERS)} 篇）</h2><div class="grid">{grid}</div>"""
    (OUT / "index.html").write_text(page("WAM 调研", body, NAV_TOP), encoding="utf-8")

def build_taxonomy():
    rows = []
    for k, name, desc, rep in CATS:
        n = len([p for p in PAPERS if p["cat"] == k])
        rows.append(f"<tr><td><b>{esc(name)}</b></td><td>{esc(desc)}</td><td>{n}</td><td>{esc(rep)}</td></tr>")
    body = f"""<h1>方法原理分类</h1>
<p class="meta">分类依据：世界-动作模型中「未来以什么形式参与动作生成」。</p>
<table><tr><th>类别</th><th>定义</th><th>站内论文数</th><th>代表</th></tr>{''.join(rows)}</table>
<h2>为什么这样分</h2>
<div class="card"><ul>
<li><b>联合扩散</b>：未来（视频潜）与动作在同一去噪图里，耦合最紧；训练期视频监督是其收益主因（Fast-WAM 的受控实验）。</li>
<li><b>潜空间</b>：未来只以特征/潜向量存在，避免像素重建；依赖强预训练视觉编码器（V-JEPA 2.1、DINOv3）。</li>
<li><b>级联</b>：未来是「计划」而非「联合变量」，必须经 IDM/策略反解；优点是模块可替换，缺点是信息瓶颈。</li>
<li><b>自回归</b>：把动作离散成 token 与视觉同词表；优点是可直接复用 LLM，缺点是误差累积（WorldVLA 的 action mask 是针对性修补）。</li>
<li><b>几何锚定</b>：动作换一种与视觉同构的载体（点流/热图），天然跨本体；代价是解码器与坐标系工程。</li>
<li><b>RL 后训练</b>：属于训练范式轴（横切各类）；奖励多来自世界模型/IDM/LTL/专家潜距离。</li>
</ul></div>
<h2>与文献综述的对应</h2>
<p>NUS 综述按「预测什么」分 Render-and-Decode / Latent-Only / Video-Generation-Free；OpenMOSS 综述按「预测与动作的耦合方式」分 Cascaded / Joint。本站按「未来如何参与动作」合并两者并补上 AR 与几何锚定两类工程实现。</p>"""
    (OUT / "taxonomy.html").write_text(page("方法原理分类", body, NAV_TOP), encoding="utf-8")

def build_cat(k, name, desc, rep):
    ps = [p for p in PAPERS if p["cat"] == k]
    grid = "".join(paper_card(p) for p in ps)
    body = f"""<h1>{esc(name)}</h1><p class="meta">{esc(desc)}</p>
<div class="grid">{grid}</div>"""
    (OUT / f"cat-{k}.html").write_text(page(name, body, NAV_TOP), encoding="utf-8")

CODE_META = {
 "fast-wam": ("源码级", "训练期让动作专家从未来帧学表征，推理期只过首帧；动作分支看不到未来 token，避免训练-推理输入分布漂移。", "受控四臂证明：去掉视频共训掉 8 点，去掉测试时想象只差约 1 点——收益在训练期表征而非推理期生成。", "官方仓库 yuantianyuan01/FastWAM：fastwam.py、configs/model/fastwam.yaml；Wan2.2 视频 DiT + ActionDiT。", "优势：190ms 实时、无具身预训练达 RoboTwin 91.8；代价：视频共训算力大，多相机拼接损失视角结构。"),
 "motus": ("论文重建", "光流是像素级「动作痕迹」，压成 14 维潜动作后，动作预训练不再依赖动作标签。", "互联网视频没有动作标签；用光流当代理动作，可把大规模视频转成动作预训练数据。", "论文 §4.1/§4.2（L2）；仓库 thu-ml/Motus 可查实现；DPFlow→DC-AE→14 维。", "优势：一模型五模式、潜动作可跨本体；代价：光流在遮挡/透明物体上不可靠。"),
 "dit4dit": ("论文重建+config", "不重建未来帧，直接取视频 DiT 中间层特征作动作条件——特征比像素更贴近物理动态。", "重建像素有信息瓶颈；中间特征保留时空结构且省算力。", "论文 §3/§4（L2）；config 实锤 action_dim=8/horizon=8（LIBERO）。", "优势：LIBERO 98.6、样本效率 >10×；代价：τf 固定、需前向钩子，训练不稳时难调。"),
 "rolling-wam": ("论文重建", "错峰噪声把「想象未来」的成本摊到多个控制周期：近端块先干净可执行，远端块继续演化。", "每周期从头去噪整段未来太慢；滑动窗口复用上一周期结果。", "论文 §III（L2，03b 笔记）；参数 K=16/W=5/N=10/ρ=5。", "优势：4.5× 重规划提速；代价：窗口/噪声调度敏感，跨 chunk 动作不可见。"),
 "x-wam": ("论文重建", "动作与视频用不同步数去噪：动作要快（实时执行），视频要全（保真）。", "动作解码不需要视频那么高的保真度；异步采样让两者各取所需。", "论文 §3.3/§3.4（L2）；ANS 异步噪声采样。", "优势：4D 重建+动作一体；代价：训练需保证联合分布对齐。"),
 "vjepa-policy": ("论文重建", "冻结 V-JEPA 2.1 提供预测性视觉潜；动作专家以未来潜的逐层 K/V 为条件。", "不必继承完整视频生成器：预测潜足够支撑 WAM，省参数。", "论文 §3.1（L2）；ViT-L 0.3B + predictor 0.5B + action 0.1B。", "优势：0.9B 总参、分布偏移下更强；代价：依赖编码器质量，预测器与动作专家需联合训练。"),
 "jepa-wam": ("论文重建", "生成的目标图 bank 提供「任务完成的样子」，V-JEPA 编成 64 token 去外观保语义。", "纯文本指令太稀疏，模型会走视觉捷径；goal token 补语义接地。", "论文 §3.1/§3.2（L2）；K=1024 bank、ViT-B/16 80M。", "优势：真机三设定 87.3/74.5/80.9；代价：需文生图生成器与离线建库。"),
 "lawam": ("论文重建", "潜动作=「意图」，LaWM 解码成潜视觉子目标，避免生成像素未来。", "像素级未来冗余且慢；潜子目标保留语义、快约 24×。", "论文 §3.1（L2）；LaWM 230M、总 2.3B。", "优势：LIBERO 98.6/187ms；代价：潜子目标可解释性差。"),
 "wa-jepa": ("论文重建", "预测动作后果（未来场景+本体），用目标-预测残差逐块修正提案。", "候选搜索太贵；固定步数反馈精炼可在部署时直接改动作。", "论文 §3.3（L2）；代码 README 的 R=4。", "优势：免候选搜索、计划生成快 10.8–32.6×；代价：需目标图/目标潜。"),
 "unipi": ("论文重建", "把策略当视频生成：先生成「动作的未来画面」，再用逆动力学反解动作。", "可复用文生视频模型；不同动作空间统一到图像空间。", "论文 §3.1/§3.2（L3）；视频-WAM 奠基作。", "优势：跨环境/跨动作空间统一；代价：开环执行、无反馈。"),
 "lvp": ("论文重建", "基础模型规模的视频规划器，零样本生成未来视频计划。", "视频含时空动作信息，比静态图文更接近机器人行为。", "论文摘要/§3（L2）。", "优势：零样本新场景；代价：后处理成动作依赖启发式。"),
 "dreamgen": ("论文重建", "用视频世界模型生成「神经轨迹」，再用 IDM/潜动作标成动作，扩充模仿数据。", "真机数据贵；世界模型能生成大量新行为/新环境。", "论文 §2（L3）；4 阶段数据管线。", "优势：单任务遥操作解锁 22 个新行为；代价：开环数据管线、无回报回环。"),
 "eva": ("论文重建", "IDM 把生成视频解成关节动作，用平滑/限幅奖励对齐可执行性。", "视频合理≠可执行；动作空间奖励直接惩罚抖动与越限。", "论文 §4.2/§4.3（L2）。", "优势：Kinematic 70.5→91.4；代价：依赖 IDM，奖励可被 hacking。"),
 "worldvla": ("论文重建", "把图像/动作/文本都离散成 token，同一 LLM 自回归出动作与下一帧。", "统一词表可复用 LLM；世界模型分支提供未来监督。", "论文 §3.2/§3.3（L2）；Chameleon 初始化。", "优势：单一模型两用；代价：自回归误差累积（action mask 缓解）。"),
 "wla": ("论文重建", "自回归先出语义子任务/子目标图，再出动作；世界预测经 meta-query 隐式影响动作。", "显式未来太贵；用语义级计划做中间表示，兼顾解释性与速度。", "论文 §3（L2）；2B 激活、40ms。", "优势：可解释、可关世界预测；代价：子目标质量决定上限。"),
 "gr1": ("论文重建", "GPT 式因果 transformer 同时预测未来帧与动作；动作为 ΔXYZ+Δ欧拉+夹爪。", "视频预训练提供物理先验；单步动作回归简单可靠。", "论文 §3.2（L3）。", "优势：CALVIN 94.9；代价：无 chunk、单步，高频控制需外推。"),
 "action-images": ("论文重建", "把 7-DoF 动作画成多视角高斯热图视频——动作与观测在同一像素空间。", "视频生成模型天然懂像素；动作写成图像就能复用其先验。", "论文 §3（L2）。", "优势：跨视角、可解释；代价：热图解码依赖相机标定。"),
 "pointworld": ("论文重建", "状态与动作都表示为 3D 点流，预测逐像素 3D 位移。", "点流与本体无关、天然跨本体；比关节空间更适合野外。", "论文 §3（L2）；H=10@0.1s。", "优势：跨本体、可嵌 MPC；代价：需要 RGB-D 与相机位姿。"),
 "vampo": ("源码级", "对视频预测器多步去噪做 MDP；奖励=与专家未来的潜距离。", "离线可算奖励，避免真机 rollout。", "仓库 OpenHelix-Team/VAMPO：reward_fn.py、step1_train_svd_grpo.py。", "优势：离线、无 critic；代价：只改视频预测、不改动作策略。"),
 "gigabrain": ("论文重建", "世界模型价值 n 步 TD 变优势，二值化后当条件做加权模仿。", "稀疏奖励下方差大；KL-RL 闭式解转 NLL 更稳。", "论文 §3.2（L2）；无官方代码。", "优势：离线日志可用；代价：依赖世界模型价值精度。"),
}

def build_paper(p):
    metr = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td></tr>" for a, b, c in p["metrics"])
    links = " · ".join(f'<a href="{esc(u)}">{esc(t)}</a>' for t, u in p["links"])
    mech = ""
    if p.get("mech_en"):
        mech = f"""<h2>核心机制（原文 + 中译）</h2>
<blockquote><span class="lbl">原文（论文方法节，verbatim）</span>{esc(p['mech_en'])}</blockquote>
<blockquote><span class="lbl">中译（译者译）</span>{esc(p['mech_cn'])}</blockquote>"""
    level, physics, why, frm, gain = CODE_META.get(p["slug"], ("未标注", "—", "—", "—", "—"))
    body = f"""<p class="meta"><a href="../index.html">← 返回总览</a> ｜ <a href="../cat-{p['cat']}.html">类别：{esc(p['cat'])}</a></p>
<h1>{esc(p['title_cn'])}</h1>
<p class="meta">{esc(p['title'])}<br>arXiv:{p['arxiv']} ｜ {esc(p['org'])} ｜ {p['date']} ｜ <a href="../code.html">代码对比页</a> ｜ {links}</p>
<h2>核心贡献</h2>
<div class="card"><p><b>中文：</b>{esc(p['contrib_cn'])}</p><p><b>English:</b> {esc(p['contrib_en'])}</p></div>
<h2>原文断言（摘要）+ 中译</h2>
<blockquote><span class="lbl">原文（Abstract，verbatim）</span>{esc(p['claim_en'])}</blockquote>
<blockquote><span class="lbl">中译（译者译，非论文原文语言）</span>{esc(p['claim_cn'])}</blockquote>
{mech}
<h2>核心代码实现</h2>
<div class="card"><b>代码解读</b><table style="margin-top:8px">
<tr><th style="width:96px">证据级别</th><td>{esc(level)}</td></tr>
<tr><th>物理意义</th><td>{esc(physics)}</td></tr>
<tr><th>为什么这么做</th><td>{esc(why)}</td></tr>
<tr><th>方法出处</th><td>{esc(frm)}</td></tr>
<tr><th>优势与代价</th><td>{esc(gain)}</td></tr>
</table></div>
<p class="meta">代码类型：{esc(p['code_kind'])}。{esc(p['code_note'])}</p>
<pre><code>{esc(p['code'])}</code></pre>
<h2>关键数值（标注来源级别）</h2>
<table><tr><th>指标</th><th>数值</th><th>级别</th></tr>{metr}</table>
<div class="warn">来源级别：L3=全文核实（含官方代码）/ L2=部分正文 / L1=仅摘要。数值为提出方自评。</div>"""
    (OUT / "papers").mkdir(exist_ok=True)
    (OUT / "papers" / f"{p['slug']}.html").write_text(page(p['title_cn'], body, NAV), encoding="utf-8")

CODE_BLOCKS = [
 ("投机推理（训练-free 加速）", "WAMotor", "2609.25413", """# 官方代码 speculative/action.py
thr = eps * (median(|Δ|)/|Δ|)**2        # 时间步感知阈值（步长大→更严）
u = predict(a, t[0]); a = euler(a, u, Δ[0])   # 首步强制 Euler（推迟投机）
draft = a + u * cumsum(Δ[m:end])              # 一个 anchor 速度外推 w 步
v = predict_batched(draft)                     # 一次前向验证全部候选
acc = longest_prefix(mse(v, u) < thr)          # 接受最长前缀
a = euler(a, v[acc-1], Δ[m+acc])               # 边界回退一步"""),
 ("循环深度 + 弹性蒸馏", "LoopedWAM", "2611.25595", """# 12 blocks/expert = 3+6+3；core 循环 K 次
h = prelude(x)
for _ in range(K): h = core(h)          # 语义层数 3+K*6+3
y = coda(h, step_mlp(1/N))              # 弹性：步长嵌入
# 蒸馏：学生从自身状态问教师
for j in range(N):
    v = student.step(z, K_a, 1/N); zT = teacher.euler(z, τ→τ')   # 教师细步
    L_local += mse(v, stop_grad((zT - z)/Δτ))"""),
 ("频带打包统一 meta-frame", "MetaWAM", "2611.14365", """F = zeros(C,H,W)
for s in [task, ρ_t, a, ρ_tH, v]:
    F[band[s], kH=0] = pack(s)          # 固定槽，无学习参数
z_m = idct2(F)                          # 逆 DCT → 空间潜
# 读出：dct2 + unpack（解析，无解码头）
# 路由：policy 开 task+ρ_t；world 再开 a；value 再开 ρ_tH"""),
 ("需求场驱动稀疏精炼", "Elastic-WAM", "2611.14257", """D, B0 = need_field(s)                   # 总需求 + Beta 形状
if 1 - F_prev*D >= 0.5: COMMIT          # WHEN（η=0.5）
z = mode(B0); Δσ = 0.05 + z*(0.2-0.05)  # HOW FAR（δ∈[0.05,0.20]）
M = topk_groups(n_g*Δu_g, budget=96)    # WHERE（K0=96 tokens）
sparse_refine(selected=M, step=Δσ)"""),
 ("不确定性路由：何时想象", "Bind-WAM", "2611.15368", """U = router(o, l)                        # 6.65M ViT
ΔU = |U - U_prev|
mode = 'joint' if (U >= τ_U or ΔU >= τ_Δ) else 'direct'
wam(o, l, mask=M_joint if joint else M_direct)"""),
 ("滚动错峰窗口", "Rolling-WAM", "2609.30247", """σ_j = ρ(j/W)/(1+(ρ-1)(j/W))             # 近低远高
for _ in range(2): denoise_joint(window, N=10)
window = window[1:] + [new_chunk(a)]    # K=16/W=5/ρ=5"""),
 ("视频共训、推理免想象", "Fast-WAM", "2603.16666", """# 训练：L = L_act + λ*L_vid（未来潜 flow matching）
# 推理：只过首帧、动作专家单次前向
lat = vae.encode(first_frame); kv = video_dit.prefill(lat)
a = action_expert.sample(cond=kv, steps=10)"""),
 ("GRPO 训练视频去噪器", "VAMPO", "2603.19370", """r = -l1(pred, true) + cos(pred, true)   # 潜空间奖励
adv = (r - r.mean(g)) / (r.std(g)+1e-8) # 组归一化
loss = min(-adv*ratio, -adv*clip(ratio,.8,1.2))  # 截断代理
# 只首步 SDE：sde_start=0, sde_end=1"""),
]

CODE_WHY = {
 "WAMotor": ("物理：同一动作块的多步去噪里，速度在相邻步变化平滑，可用一个 anchor 速度外推多步。", "为何：省去逐步模型调用；首步强制 Euler 避开首步速度剧变。", "优势：3.24–6.62× 去噪加速、2.86–4.28× 端到端；代价：窗口/阈值敏感，需按步长缩放验收。"),
 "LoopedWAM": ("物理：把「深网络」换成「浅网络循环」——同一组权重反复用，参数少但有效深度大。", "为何：显存/参数受限下保留深度；弹性蒸馏让一个 checkpoint 覆盖多档算力。", "优势：0.58B 达 96.1%、比 Fast-WAM 少 7.4× 显存；代价：激进降深度需专门蒸馏训练。"),
 "MetaWAM": ("物理：机器人信号（本体/动作/价值）是低维平滑序列，可塞进视频潜的低频 DCT 频带。", "为何：直接拼空间通道会破坏视频先验；频域打包让 denoiser 几乎不需要修正。", "优势：延迟 −39%、多信号统一进一个 meta-frame；代价：频带布局需固定，改动不灵活。"),
 "Elastic-WAM": ("物理：并非所有时刻都需要重构整幅未来——只在「需求高」的区域/时段精炼。", "为何：密集去噪浪费算力；用需求场同时决定何时 commit、走多远、精炼哪块。", "优势：稀疏精炼大幅省算力且保精度；代价：三阶段动作接地监督，训练复杂。"),
 "Bind-WAM": ("物理：语言-物体绑定失效（指错物体/遮挡）时才需要想象未来核对。", "为何：一直想象太贵；用 6.65M 路由器预测绑定不确定性，按需切换直接/联合路径。", "优势：推理成本按需分配；代价：需离线伪标注（Qwen/SAM/DINO）与阈值调参。"),
 "Rolling-WAM": ("物理：控制是滚动的，未来不必每周期从头重算，可保留并继续精炼。", "为何：整段联合去噪延迟高；错峰噪声让近端块先干净、远端块继续演化。", "优势：稳态重规划 4.5×；代价：窗口/噪声调度敏感。"),
 "Fast-WAM": ("物理：视频共训的价值在训练期塑形世界表征，推理时机械臂看不到未来。", "为何：推理期想象未来是延迟主因；受控实验显示去掉它只掉约 1 点。", "优势：190ms、无需具身预训练；代价：视频共训算力大。"),
 "VAMPO": ("物理：同一观测下想象多条未来，越接近专家未来的越可执行。", "为何：真机奖励太贵；用专家潜距离当离线奖励，GRPO 组内比较。", "优势：完全离线、无 critic；代价：只改视频预测器。"),
}

def build_code():
    cards = ""
    for title, name, arx, code in CODE_BLOCKS:
        ph, wy, gn = CODE_WHY.get(name, ("", "", ""))
        cards += f"""<div class="card"><h3>{esc(title)}　<span class="tag">{esc(name)}</span><span class="tag">arXiv {esc(arx)}</span></h3>
<p class="meta">{esc(ph)}<br>{esc(wy)}<br>{esc(gn)}</p>
<pre><code>{esc(code)}</code></pre></div>"""
    body = f"""<h1>实战手册：代码页对比</h1>
<p class="meta">同一类问题（如何让未来参与动作？如何部署更快？）的不同实现。每块上方标注方法，代码为官方源码节选（标注）或论文结构化伪代码。</p>
<div class="card"><b>对比维度</b>
<table><tr><th>问题</th><th>方法</th><th>核心机制</th><th>代价</th></tr>
<tr><td>测试时还要不要想象未来</td><td>Fast-WAM</td><td>训练共训、推理只过首帧</td><td>去掉共训掉 8 点</td></tr>
<tr><td>去噪太慢</td><td>WAMotor / Rolling-WAM / LoopedWAM</td><td>投机验证 / 错峰窗口 / 循环深度</td><td>训练-free / 需调度 / 需蒸馏</td></tr>
<tr><td>机器人信号怎么进视频模型</td><td>MetaWAM</td><td>频带打包 meta-frame</td><td>需固定频带布局</td></tr>
<tr><td>何时该想象</td><td>Bind-WAM / Elastic-WAM</td><td>不确定性路由 / 需求场</td><td>需离线标注 / 需三阶段监督</td></tr>
<tr><td>如何用 RL 后训练</td><td>VAMPO / GigaBrain</td><td>GRPO（组优势） / KL-RL→加权 NLL</td><td>奖励设计 / 世界模型价值</td></tr>
</table></div>
{cards}"""
    (OUT / "code.html").write_text(page("代码对比", body, NAV_TOP), encoding="utf-8")

def build_refs():
    rows = "".join(f'<tr><td>{esc(p["title_cn"])}</td><td>{p["arxiv"]}</td><td><a href="papers/{p["slug"]}.html">页面</a></td><td><a href="https://arxiv.org/abs/{p["arxiv"]}">arXiv</a></td></tr>' for p in PAPERS)
    notes = """<h2>精读笔记（Markdown 源文件）</h2>
<ul><li>01 输入表征 / 02 实机仿真验证 / 03 动作表征 / 04 决策表征 / 05 目标表征（第一波，48 篇次）</li>
<li>01b/02b/03b 第二波扩展（30 篇次）</li><li>06 代码对质（动作空间 7 项）</li><li>07 RL post-training（6 篇）/ 08 VAMPO 代码取证</li>
<li>10 最新进展（2026-11 月波）/ 10 伪代码 / 11 WAMotor 代码取证 / 12 工程级伪代码</li></ul>
<p class="meta">完整 Markdown 与本地全文存档见工程仓库 <code>reading/</code> 目录。</p>"""
    body = f"""<h1>来源与原文索引</h1>
<p class="meta">所有引文均可回溯到 arXiv 原文；中译为译者译并已标注。</p>
<table><tr><th>论文</th><th>arXiv</th><th>站内</th><th>原文</th></tr>{rows}</table>
{notes}
<div class="warn">引用纪律：站点内所有「原文」块为 verbatim 摘录（摘要或方法节），「中译」为译者翻译；数值为论文自评（L1/L2/L3 标注）；未获取处不补写。</div>"""
    (OUT / "refs.html").write_text(page("来源与索引", body, NAV_TOP), encoding="utf-8")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_index(); build_taxonomy(); build_code(); build_refs()
    for k, name, desc, rep in CATS:
        build_cat(k, name, desc, rep)
    for p in PAPERS:
        build_paper(p)
    n = len(list(OUT.rglob("*.html")))
    print(f"OK: {n} html files -> {OUT}")

if __name__ == "__main__":
    main()
