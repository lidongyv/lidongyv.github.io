# WAM 最新进展扫描（2026-11 月波）：10 篇机制型代表 L2 精读

> 来源级别：L2=部分正文（arXiv abs + HTML/ar5iv 最新版）；L1=仅摘要。所有引文为论文原文（英文 verbatim + 小节号），数值逐字抄录。
> 取文链：`arxiv.org/abs/<id>` → `arxiv.org/html/<id>`（最新版）→ `ar5iv.labs.arxiv.org/html/<id>`；fetch 失败重试一次，两次失败该篇降级 L1 并标注。
> 每篇 ≤45 行，聚焦【实现细节】：架构组成、训练目标/阶段、推理流程、关键超参数、代码仓库（若有）。

---

## 本线说明与范围

- **检索口径**：2026-08-21 至今 arXiv 命中 **319 篇** WAM 相关论文（上游调研线口径）。
- **本线范围**：只做 **2026-11 月机制型代表 10 篇** 的 L2 级精读；8–10 月论文已在前两波（01–05 各线及 wave2 扩展）覆盖。
- **L2 标准**：Abstract + 正文 Method/实现相关小节（架构、训练目标与阶段、推理流程、关键超参、代码仓库）；达不到正文级的篇目标 L1 并注明原因。
- **纪律**：结论只依据论文事实或代码事实，引文 verbatim 并附小节号；找不到写「原文未写（已获取范围）」/「未取到」（标截断位置）；禁止脑补。
- **篇目**（10）：LoopedWAM 2611.25595、WAMotor 2611.25413、WA-JEPA 2611.19183、MetaWAM 2611.14365、Elastic-WAM 2611.14257、Bind-WAM 2611.15368、WATO 2611.17626、ResWAM 2611.19559、LaST0.5 2611.14048、ThinkWAM 2611.21258。

---

## 1. LoopedWAM 2611.25595 —— 权重共享循环 core + elastic self-distillation（L2）

> 来源：arXiv HTML v1（`arxiv.org/html/2611.25595v1`）；截断于 §3.5 末（"…and evaluates the teacher at its fine steps for the local targets an…"），§4 实验节未取到；ar5iv 无此文（重定向回 arxiv.org）。

- **结果（Abstract，数值逐字）**：0.58B 策略参数、无机器人预训练——"LoopedWAM achieves **96.1%** success on LIBERO, **91.9%** on RoboTwin 2.0, and **52.2%** on RoboCasa-GR1, while using **7.4×** less peak GPU memory than Fast-WAM under matched inference measurements"。弹性版 LoopedWAM-E："achieves **96.0%** success in **27.6 ms** or **95.2%** in **24.7 ms** on an H100"；"with two sampling steps and four action loops, it stays within 0.5 points of LoopedWAM on all three benchmarks"；真机 AgileX PiPER "succeeds in **71.3%** of 80 trials across four tasks"。
- **代码**：`https://github.com/anhdao69/LoopedWAM`、模型 `https://huggingface.co/anhdao69/LoopedWAM`（Abstract 页声明）。
- **架构（§3.2）**：每 expert 存 **12 blocks = 3-block prelude + 6-block shared core + 3-block coda**（Intro）；core 循环 K 次——执行 L(K)=LP+K·LC+LE 层、只存 LP+LC+LE blocks，"depth grows with K at no cost in parameters"。循环保持最简："no input injection, no loop-index embedding, and no outer residual around the core"；每执行层两 expert 共享 attention，同 dense MoT。donor 为 **Wan2.1** 视频 DiT，4 次循环恢复其 **30 层** 深度（Intro）；§3.4：prelude/coda 复制 donor 首 LP/末 LE blocks、core 复制一组连续 early-middle blocks；变窄 FF 保留等距 hidden units；action expert 用 Fast-WAM 的 resizing rule。
- **双 expert 解耦循环（§3.3）**：视频 expert 每 query 只跑一次并缓存各层 KV；动作 expert 每采样步跑 Ka 次。**late alignment**：动作 loop r 的 block b 关注视频 loop σ(r)=max(1, Kv−Ka+r) 的 block b——"so the last action loop always reads the deepest visual features"；σ 在 Ka=Kv 时退化为 identity。单 query 耗时 T(Kv,Ka,N)≈T0+L(Kv)·tv+N·L(Ka)·ta（Eq. 4）；默认 Kv=Ka=K。
- **训练目标（§3.1）**：flow matching，L_flow=E[w(τ)‖vθ(zτ,τ|x0,s,ℓ)−(ε−x)‖²]（Eq. 1），"summed over the action and future-video branches"。冻结视频 VAE 编码当前帧 x0（未来帧 x1:T **仅训练时**编码）；attention mask：干净当前帧 tokens 只互相关注、noisy 未来帧 tokens 关注当前帧+自身、action tokens 关注当前帧+自身；当前帧恒为 flow time 0。测试时丢未来帧、视频 expert 一次/query、缓存 KV。
- **推理（§3.1）**：动作 chunk 从 τ0=1 到 τN=0 用 **N 步 Euler** 积分（z_{j+1}=zj+(τj+1−τj)·vθ(zj,τj)），"without classifier-free guidance"。
- **Elastic self-distillation（§3.5）**：单 checkpoint LoopedWAM-E 覆盖 **Ka∈{1,2,4}、N∈{2,5,10}**（Intro），视频 expert 固定 Kv=K。teacher=训练好的 LoopedWAM（NT 步 Euler）冻结；student 从 teacher 复制、仅 action expert 可训，视频 expert/VAE/文本编码器/proprio 投影冻结共享（一次视频 pass 两用）；零初始化 step-condition MLP 把步长 **d=1/N** 的 embedding 加入 timestep modulation。**L_local**（Eq. 5）：student 速度对 teacher 以 student 自身状态出发的 Euler map 的逐步 MSE——"Following the teacher from the student's own states, rather than along the teacher's trajectory, exposes the student to its own errors and trains it to correct them"。另加 L_end（MSE 匹配 Hx 执行动作）、L_grip（gripper softplus margin）、L_anchor（Ka=K、d=0 时对真实动作的 flow loss）；总目标 L=L_local+λ_end·L_end+λ_grip·L_grip+λ_anchor·L_anchor（Eq. 6）。"only the last m steps are back-propagated end to end"；balanced sequence 每 |S| 次更新遍历每个 setting。
- **消融要点**：无弹性训练时降到 1 个 action loop，LIBERO 掉到 **12.3%**（Intro，§4.4 表格未取到）；"weight-shared depth preserves competitive performance, while aggressive action-depth reduction requires explicit budget-aware training"（Abstract）。
- **未取到**：§4.1 训练超参（batch/lr/步数）与 m、λ 系数、H/Hx、Da 数值——§4 全节在截断点之后。

---

## 2. WAMotor 2611.25413 —— 免训练投机推理：双非均匀去噪 + anchor velocity + time-step-aware 验证（L2）

> 来源：arXiv HTML v1（`arxiv.org/html/2611.25413v1`）；截断于 §4.3 末（"…conducts one standard Euler update before the first speculative window and uses u1"），§4.4/§4.5/§5 未取到；ar5iv 无此文（重定向回 arxiv.org）。作者：Zexun Lin, Yuan Feng, Xike Xie（USTC, Data Darkness Lab, MIRACLE Center, Suzhou Institute for Advanced Research）。

- **定位（Abstract）**：免训练（"training-free"）WAM 投机推理，即插即用——"selects reliable anchor velocities by deferring speculation and employs time-step-aware verification that scales acceptance thresholds by integration time step size"。代码 `https://github.com/Overdue-Lin/WAMotor`。
- **动机观察（§3.1/§3.2）**：①"velocity smoothness is nonuniform"——相邻速度在中间步平滑、在**首步与末步**方向/幅度剧变；②非均匀积分步长 Δtk 把相同速度残差放大为不同局部状态误差并累积到最终动作分布。故两个设计问题："When to Draft" 与 "How to Verify"。
- **问题形式化（§4.2）**：Definition 4.1 Trajectory δ-error Tolerance：‖aN^s−aN‖_A ≤ δ（Eq. 2），其中 ‖·‖_A² "denotes maximum per-token MSE over the blocks, which ensures the acceptance criterion applies to every token"。
- **理论（§4.3）**：Assumption 4.2 Lipschitz——‖vθ(x,tk,c)−vθ(y,tk,c′)‖_A ≤ Lk‖x−y‖_A + Ck‖c−c′‖_A（Eq. 3）；传播因子 Gk=Π_{j=k+1}^{N−1}(1+hj·Lj)（hj=|Δtj|）、B_cond=Σ_k Gk·hk·Ck·‖ck^s−ck‖_A；Theorem 4.3：‖aN^s−aN‖_A ≤ Σ_{k∈I} Gk·hk·‖r_{Mk,k−Mk}‖_A + B_cond（Eq. 4），"earlier residuals carry no smaller propagation factors"。
- **Stage I：Reliable Anchor Velocity Selection（§4.3）**：先执行一次标准 Euler 更新、**推迟投机**，用更新后状态上的速度 u1 作初始 anchor（避免首步速度剧变）。
- **Stage II：Time-Step-Aware Verification（§4.4，标题+Intro 描述）**：基于"the quadratic relationship between squared velocity residuals and integration time step size"——"tightening verification for larger steps and relaxing it for smaller ones while providing a trajectory error tolerance guarantee"；精确判定式未取到（截断于 §4.3）。
- **投机流程（§4.1）**：flow matching 动作生成 a_{k+1}=ak+Δtk·vθ(ak,tk,ck)，k=0,…,N−1，a_k∈R^{H×D}（H=动作 horizon、D=维度，归一化动作；上下文 ck 含 observations/language/proprioception/future representations）。anchor 步 m 用速度 um 一次性草拟 qm=min(w, N−1−m) 个候选（w=speculation window cap）：â_{m,j}=am^s+um·Σ_{i=0}^{j−1}Δt_{m+i}（Eq. 1）；随后**一次 batched 模型调用**验证所有候选 v̂_{m,j}=vθ(â_{m,j},t_{m+j},ĉ_{m,j})，残差 r_{m,j}=um−v̂_{m,j}；接受最长前缀并 commit；"For WAMs with evolving future conditioning, speculative conditioning ĉm,j is likewise constructed"。
- **集成对象（§4.5 标题 + Intro）**：Joint-WAM、Fast-WAM、Faster-WAM——"replacing their standard flow matching denoising pipelines"；附录 B.5 有 Joint-WAM 集成伪代码（未取到）。
- **结果（Abstract/Intro）**："accelerates flow matching inference by **3.24×–6.62×**, yielding overall speedups of **2.86×–4.28×** over vanilla models"；评测于 LIBERO、RoboTwin 2.0、LIBERO-Plus，"maintaining task success rates"。
- **未取到**：§4.4 验证准则公式、§5.4 超参（Initial Anchor Step K、Verification Threshold ε 的敏感性数值）、§5 各表与延迟分解、w 数值——均在截断点之后。

---

## 3. WA-JEPA 2611.19183 —— GDRF 目标差异残差反馈 + FSCF 未来场景上下文反馈（L2）

> 来源：arXiv HTML v1（`arxiv.org/html/2611.19183v1`）；§1–§3.4 完整，截断于 §4 引言段（"Ablations examine the predictive-feedback system…"），§4 数值与附录 C（loss 权重/优化配置/维度）未取到。作者：HIT + Hikvision Research Institute 等；代码 `https://github.com/Dovisya/WA-JEPA`（Abstract）。

- **定位（Abstract）**：JEPA 式潜空间世界模型 + 直接动作生成，**免候选动作搜索**——"making future predictions directly participate in action generation without candidate-action search"。
- **问题形式化（§3.1）**：决策时刻 t 输入最近 **L** 帧图像+proprio {oτ,pτ} 与目标图像 og；动作序列 **A∈R^{H×b·d_a}** = H 个时序 chunk、每 chunk b 个连续控制、维度 d_a；h 索引 chunk、r 索引内部精炼 stage——"Refinement updates the same pending plan without advancing environment time"。
- **场景表征（§3.2）**：**冻结 DINOv2** 提取观测/目标/训练未来图像特征；可学习多层 feature fusion + query pooling 汇总为 scene tokens **Zt∈R^{S×d}**；所有图像共享同一 fusion/pooling 参数（预测目标与 goal 编码同空间）；proprio 标准化后作附加 token 与每帧 scene features 融合；temporal attention 整合历史；最新帧表征与 goal 编码 G 做**双向 cross-attention** → context memory **M∈R^{L(S+1)×d}** 与 goal memory **G~∈R^{S×d}**；另保留当前场景编码 Z0 与原始目标编码 G。
- **推理循环（§3.3，Fig.1）**：proposal→prediction→goal-aware feedback 迭代精炼，固定 forward stage 数，不推进环境时间；仅最终 stage 动作序列执行。
  - **动作提案**：初始 Q(1)=learned chunk queries + temporal embeddings + （pooled Z0 与 pooled goal 的差异）条件；stage r 模块 Tr 先 chunk 间 self-attention，再对 [M, G~, Ẑ(r−1)] cross-attention；共享解码器 D + range mapping B：A(r)=B(D(LN(Q~(r))))（Eq. 1）；Ẑ(0)=Z0 沿未来时间轴复制。
  - **结果预测（Eq. 2）**：共享世界模型 (Ẑ(r), P̂(r))=Wψ(M, Z0, A(r))；Ẑ(r)∈R^{H×S×d}、P̂(r)∈R^{H×d_p}；temporal masks 限制预测 h 只读前 h 个 chunk——"predictions across future times can be computed in parallel"；特征空间直接预测、不重建图像、不再过视觉编码器。
  - **GDRF（Eq. 3）**：预测 h 的 S 个 scene token 取平均 Ẑ̄h(r)，与 pooled goal Ḡ 相减；反馈输入 Uh(r)=[Q~h(r), Ẑ̄h(r), Ḡ−Ẑ̄h(r), Ep(P̂h(r))]∈R^{4d}；**残差更新** Qh(r+1)=Q~h(r)+σ(γr)·Fr(Uh(r))——σ=sigmoid、可学习门 γr 控制修正幅度；同 stage 内各 chunk 共享 Fr 与同一目标 Ḡ。
  - **FSCF**：未平均的逐 token 预测场景经 cross-attention 供下一 stage 读取——"allowing refinement to use scene details compressed by mean pooling"。
- **训练目标（§3.4, Eq. 4）**：L_core=L_pred+L_action+L_align。L_pred：给定记录动作在每 chunk 边界预测观测结果，scene features 用 **MSE**、proprio 用 **SmoothL1**；L_action：用示范动作监督中间与最终提案；L_align：最终提案的预测场景/proprio 与示范目标对齐，并"encourages the predicted scene at the goal timestep to match the task-goal representation"。loss 权重在 App C.2.1（未取到）。
- **结果（Intro/Fig 2-3 caption）**：短程目标到达（TwoRoom/PushT/Cube）"achieves **100%**, **96%**, and **100%**, respectively"（Fig. 2 caption）；"matching or improving LeWM's success rates while accelerating complete-plan generation by **10.8–32.6×**"；长程接入 Anchored Planning（Aim Short to Reach Far）**无需重训**；全任务用 **VQ-BeT action head** 评 Push-T 与 LIBERO-Goal；成本图于 **NVIDIA A40**（Fig. 3）。
- **未取到**：§4 各表数值、App C.1/C.2（模块维度、loss 权重、优化器/训练配置）、H/b/d/S/d 具体数值——截断点之后。

---

## 4. MetaWAM 2611.14365 —— 频域打包 meta-frame + 逆 DCT + slot-aware routing（L2）

> 来源：arXiv HTML v1（`arxiv.org/html/2611.14365v1`）；§1–§4 完整、§5 截断于 LIBERO-Plus 段（"Its camera and language s…"），§5.2 真机/§5.3 消融/附录 B 未取到。作者：NUS Show Lab（Mike Zheng Shou 通讯）；代码 `https://github.com/showlab/MetaWAM`（Abstract）。

- **核心主张（Abstract）**：把机器人信号注入 VDM 潜空间是"a distribution-alignment problem"——用冻结 VDM denoiser 的相对重建误差作探针，"frequency-domain packing yields latents that the denoiser corrects far less"。
- **诊断证据（§3, Table 1）**：冻结 Cosmos-Predict2 denoiser 相对重建误差（↓ 越小越贴 prior），σ=1.0 时：Spatial Duplication **0.6459**、VAE Encoding **0.6379**、DCT Encoding **0.5922**（单信号）；多信号打包：Sep. Spatial Dup. **0.6178**、Meta-Quadrants **0.6730**、Meta-Compact **0.6234**、Unified Meta-Frame **0.5613**——"Our layout achieves the lowest error at every tested noise level"。
- **潜序列（§4.1, Eq. 1）**：**8 帧** 序列 z=(zin^m, zt^w, zt^p, zt^s, zout^m, z_{t+H}^w, z_{t+H}^p, z_{t+H}^s)——w/p/s=wrist/primary/secondary 三相机当前与未来帧；输入 meta-frame 载 one-hot 任务类型+当前 proprio ρt；输出 meta-frame 统一承载 action chunks、未来 proprio、value。自注意力随 T 平方增长，"this cuts pairwise attention by roughly **47%** with the architecture untouched"。
- **频域打包（§4.2）**：稀疏系数张量 F∈R^{C×H×W}，按**互不相交的 channel bands** 分信号：F=[F_task, F_ρt, F_a, F_ρt+H, F_v]（Eq. 2）；Pack(q) 把归一化信号值直接写入其 band 的 **kH=0 行** 固定系数槽（其余置零）——"without a temporal DCT or coefficient truncation"，无学习参数；再用**每通道正交 2D 逆 DCT** 提升为空间潜 zm=T2D^{-1}(F)；读出：F̂=T2D(ẑm)、(â, ρ̂t+H, v̂)=Unpack(F̂)（Eq. 3），"The selected coefficients directly represent the normalized signals, so no further inverse DCT is applied"。
- **输入 meta-frame（§4.3）**：复用 tokenizer 的强制性 blank slot（原为闲置），task type + 当前 proprio 恒为 conditioning，"makes input-side conditioning effectively free"。
- **Slot-aware task routing（§4.4）**：policy/世界模型/value 三任务共享同一批变量、仅因果方向不同；band 级掩码 M_slot=Φ(m_π, m_w, m_v)——policy 样本只开 task+当前 proprio band；world-model 样本额外开 action band（未来状态仍是 target）；value 样本再开未来 proprio band（仅 value 为 target）。"band-level routing is strictly finer than a frame-level mask"；**dual-granularity masking**：细掩码承载全部任务语义，另由细掩码确定性派生单通道粗掩码喂六帧视觉帧以保持与预训练 VDM 的逐字节兼容。
- **训练目标（§4.5）**：标准扩散去噪 L_diff=E[w(σ)‖Dθ(z0+σε,σ,c,M)−z0‖²]（Eq. 4，σ-prediction）；附加信号监督 L=L_diff+λ_act·L_act+λ_prop·L_prop+λ_val·L_val+λ_off·L_off（Eq. 5）——三个信号项均为 **Huber loss（δ=1）**，按 eligible samples 平均（action 取 policy 样本、未来 proprio 取 policy/world 样本、value 取全部任务）；L_off=E[mean_{C,H,W}‖(1−S)⊙F̂‖] 惩罚槽外系数，**λ_off=10^{-4}**（Eq. 6）；DCT 与槽读出可微，**无学习解码头**。
- **训练配置（§5）**：初始化 **Cosmos-Predict2-2B-Video2World**；RoboCasa（24 静态厨房任务、**50 demos/任务**）用 **AdamW lr=10^{-6}、batch 64/GPU、4 步梯度累积、4×H200**；评测 50 trials/任务×5 场景×10 trials/场景×3 seeds=**3600 trials**。
- **结果（§5.1/Abstract）**：RoboCasa **68.9%**（50 demos）胜 Cosmos Policy 67.1（同数据设定），Table 2 中 UVA 50.0/DP-VLA 57.3/UWM 60.8/π0 62.5/GR00T-N1.5 64.1/Video Policy 66.0/FLARE 66.4；LIBERO 四套件平均 **98.1%**（Table 3 Original 列）；LIBERO-Plus **83.44%** vs Cosmos Policy 82.2%（+1.24 pp，7 类扰动）；真机三任务 Franka **85.0%**；推理延迟降 **39%**。
- **未取到**：λ_act/λ_prop/λ_val 数值、Table 3 全表、§5.2 真机设置与 §5.3 消融、附录 B 训练细节——截断点之后。

---

## 5. Elastic-WAM 2611.14257 —— 时空精炼需求场 + 三阶段动作接地监督（L2）

> 来源：arXiv HTML v1（`arxiv.org/html/2611.14257v1`）；截断于 §3.3 停止监督段（"pstop∗(s)" 处），§3.3 余下/§3.4/§4/附录 A 未取到。作者：NUS + RoboScience（Lin Shao 通讯）；项目页 `https://lihaitian.com/Elastic-WAM/`（Abstract 未列代码仓库）。

- **定位（Abstract）**：对**冻结** WAM 主干做自适应世界精炼预算分配——"adaptively allocates world refinement computation across temporal denoising time and spatial future space"，学得的场在推理时"guides recurrent sparse world refinement"。
- **动机观察（§1）**：10 个 LIBERO-Plus 任务中，最优成功率的去噪步数"varies from one to four"，同任务不同状态也偏好不同深度；action-to-video 注意力图显示未来视频区域关注不均——促生选择性精炼。
- **统一场定义（§3.1）**：**NF=196** 个未来视频 token 划分为 **G=64** 个 geometry-aware groups（保持时间与相机视角边界），ng∈{1,2,4}；每 group 局部精炼曲线 qg(z|s)=ag(s)·Beta(z; mg(s), cg(s))（Eq. 1）——Beta 密度归一化使总质量=振幅 ag(s)（Eq. 2）；全局需求 Q(z|s)=Σ_g (ng/NF)·qg(z|s)。
- **参数化（§3.2）**：所有未来 token 共享去噪轨迹，故 qg(z|s)=ag(s)·B0(z|s)、B0(z|s)=Beta(z, m0(s), c0(s))（Eq. 3）；Q(z|s)=D(s)·B0(z|s)、D(s)=Σ(ng/NF)·ag(s)（Eq. 4）；空间分数中心化 Δug(s)=ug^rel(s)−Σ(nj/NF)·uj^rel(s)（Eq. 5）；ag(s)=sigmoid(b0(s)+stopgrad[Δug(s)])（Eq. 6）——b0 控总需求、Δug 控区域优先级，stop-gradient 防止两者互扰。
- **三个决策的映射（§3.2）**：
  - **WHEN**（Eq. 7）：Ck=D(sk) 作条件继续概率；ΔHk=−log(clamp(Ck,ε,1))；累积 Fk=1−Π_{j≤k}Cj；**Fk≥η 触发 World Commit，η=0.5**。
  - **HOW FAR**（Eq. 8）：p(z|sk,CONTINUE)=B0(z|sk)；取 mode zk=m0(sk)；步长 Δσk=δmin+zk·(δmax−δmin)，**δmin=0.05、δmax=min(0.20, σk−σfloor)**。
  - **WHERE**（Eq. 9）：以精确 token 预算选出区域 M_k*=argmax Σ ng·Mg·Δug(sk) s.t. Σ ng·Mg=K0；默认 **K0=96 tokens/稀疏更新**。
- **三阶段动作接地监督（§3.3）**：主干冻结，依次优化 ①时序参数 (m0,c0)、②相对空间分数 ug^rel、③全局偏移 b0。①用终端动作误差 E(s,y)=**0.6·Eseq(s,y)+0.4·EΔ(s,y)**（Eq. 10，Eseq=动作序列误差、EΔ=相邻动作差分误差），比较立即 World Commit 与不同去噪距离的继续，把误差转成 soft target 监督 B0；②停止监督比较立即 commit 与继续的期望误差：E_continue^π(s)=Σ_δ P_π(δ|s)·E_δ(s)、A_stop(s)=E_continue^π(s)−E_commit(s)（Eq. 11），p_stop* 公式未取到（截断）。
- **推理流程（§3 引言 + Fig. 2）**：先一步 **dense joint denoising** → 学得场指导的 **recurrent sparse world refinement**（只刷新选中未来 video queries，保留完整逻辑注意力上下文）→ World Commit 后用缓存的视频表征**继续 action denoising**。
- **结果（Intro）**："reducing inference latency by **32.2%** compared with Joint-WAM on RoboCasa-GR1"；"state-of-the-art control performance among methods without large-scale embodied pretraining"；真机长程茶准备任务（§4.4 未取到）。
- **未取到**：③阶段（b0）监督、§3.4 细节、§4 全部数值表、附录 A.1 base WAM 具体型号与 A.2 基准/训练细节——截断点之后；base WAM 型号在已获取范围内未写明。

---

## 6. Bind-WAM 2611.15368 —— binding uncertainty router + 选择性未来想象 + 离线伪标注管线（L2）

> 来源：arXiv HTML v1（`arxiv.org/html/2611.15368v1`）；§III 方法全节完整，截断于 Table II Bind-WAM 行（"99.8 | 99.2 |"），§IV-A/IV-C 未取到。作者：Fudan + Tongji + ROBOTOPIA（Yanwei Fu 通讯）。

- **核心问题（Abstract）**："when should a robot policy imagine the future?"——引入 **language-object binding uncertainty** 作为可解释路由信号，仅当当前观测不足以稳定生成任务相关动作时才调用 joint future-action 生成。
- **离线伪标注管线（§III-B, Fig. 3）**：每段示范**均匀采样 32 帧**；**Qwen3-VL-32B** 识别任务相关实体（被操作物体 + 关系锚点），并借助多帧 mask 叠加验证角色归属；**SAM 3.1** 按实体文本 prompt 生成实例 mask 并跨轨迹传播；**DINOv3 ViT-L/16** 特征在 mask 内池化测身份一致性。角色级标签（Eq. 2）：**IoUr<0.3 ∨ vt^r=0 ∨ st^r<0.9**（参考 mask 不可靠/不可见/DINO 余弦相似度低于 0.9）→ yt^r=1；帧级 yt=max_r yt^r（Eq. 3）——对应三类失效：reference mismatch、invisibility、identity drift。"Qwen, SAM, DINO, action signals, and trajectory context are used exclusively during offline annotation and are not required at inference time."
- **预测器（§III-C）**：Ut=fθ(ot,ℓ)∈[0,1]，**轻量 ViT 类架构、6.65M 参数**（Intro）；损失 L_bind=BCE_w(Ût,yt)+λ·(Ût−yt)²（Eq. 5，加权 BCE + Brier 项）。
- **双路径共享 WAM（§III-D）**：同一参数共享 WAM（共享 video expert、action expert、mixed transformer），仅 action-to-video 注意力掩码不同：direct 路径 M_direct(Za,Zhv)=1 仅 h=0（Eq. 6，只看当前帧）；joint 路径 M_joint(Za,Zhv)=1, h=0,…,H（Eq. 7，联合去噪未来视觉潜 + 动作 chunk）。
- **路由规则（§III-E）**：ΔUt=|Ut−Ut−1|（Eq. 8）；Ut≥τU ∨ ΔUt≥τΔ → joint，否则 direct（Eq. 9）——即"according to the predicted uncertainty and its change between consecutive replanning steps"。
- **训练目标（§III-F）**：flow matching，逐模态独立采样噪声 ϵd~N(0,I) 与噪声级 td~p_shift：ytd^d=(1−td)yd+td·ϵd、vd=ϵd−yd（Eq. 10）；共享 WAM 在 binding 相关掩码 Mg 下同时预测两速度场（Eq. 11）；L_WAM=λa·La+λv·Lv（Eq. 12），Ld=E[w(td)·MSE(v̂d, ϵd−yd)]（Eq. 13）；"pshift and w are the timestep sampling distribution and loss weighting inherited from the WAM scheduler"。
- **预测器指标（Table I）**：LIBERO——Balanced Acc **87.59%**、AUROC **0.9653**、AUPRC **0.9917**、Brier **0.0504**、Uncertain recall@0.5 **97.32%**、Certain recall@0.5 **77.86%**；RoboTwin——**96.75% / 0.9984 / 0.9999 / 0.0070 / 99.27% / 94.23%**（episode-disjoint 测试集）。
- **LIBERO 对比（Table II，部分）**：Fast-WAM† 均值 **97.1%**（9.6 s、0% imagination）、Fast-WAM-Joint† **98.4%**（15.7 s、100%）、AdaWAM **98.5%**、Bind-WAM 行 "99.8 | 99.2 |" 截断；Abstract 称"higher success rates than Fast-WAM and Fast-WAM-Joint, with a minority of future imagination and much less completion time than Fast-WAM-Joint"。
- **未取到**：τU/τΔ 阈值与 λa/λv 数值、Bind-WAM 全行、§IV-A 设置与 §IV-C 分析、真机任务细节——截断点之后。

---

## 7. WATO 2611.17626 —— 可微轨迹优化、部署时免 TO（L2）

> 来源：arXiv HTML v1（`arxiv.org/html/2611.17626v1`）；截断于 §3.3（"…dZ(Z,Z~)=H^{-1}∑_{h=1}^H‖Nz(zh"），§3.4–3.6/§5/附录 D/E/F/H 未取到。作者：Mintae Kim, Koushil Sreenath（Hybrid Robotics, BAIR, UC Berkeley）。29 页，6 图 14 表。

- **核心思想（Abstract）**：joint WAM 生成动作+未来状态，但"trajectory imitation does not ensure that these trajectories are task-effective or consistent with dynamics"——WATO 把生成的动作与状态同时作**参考**放进 dynamics-constrained 轨迹优化，并用**隐式微分**把任务损失穿过优化解回传给生成器；"using trajectory optimization only during training"。
- **问题设定（§2）**：视觉离线 goal-conditioned 控制；B 个连续动作组成 action block Ah∈U=A^B；动作轨迹 A=(A0..A_{H−1})∈U^H 覆盖 L=BH 环境步；**LeWM 风格预训练潜世界模型**：zt=Eθ(ot)、Ẑ(A)=Rθ(zt,A)=(ẑ1..ẑH)（Eq. 1）；goal g=Eθ(og)；终端目标代价 **J(Z,g)=‖zH−g‖²₂**；receding-horizon 控制。
- **生成器（§2/§3.1）**：joint WAM 生成 (Ã,Z̃)~qφ(A,Z|c)，c=(zt,g)（Eq. 2）；Z̃（生成意图）与 Ẑ(Ã)（世界模型预测后果）显式区分——"·~ denotes a generated proposal, ·^ a world model prediction, and ·⋆ an optimized trajectory"。参数化为**条件 flow matching**：线性路径 xs=(1−s)ϵ+sx、s~Unif[0,1]，训练 vφ 指向 x−ϵ。
- **WAF 模仿目标（§3.1, Eq. 3）**：L_WAF=E[MSE(vφ^A(Xs,s;c), A−ϵA) + λ_WAF·MSE(vφ^Z(Xs,s;c), Z−ϵZ)]；训练窗口以窗口末观测作 goal；"Imitation alone therefore provides neither task optimality nor per-sample dynamics-consistency."
- **被否掉的替代方案（§3.2）**：①直接 rollout L_rollout=J(Rθ(zt,Ã),g)——只有 A 收到任务梯度、Z̃ 不参与；②L_direct+cons=J+β·dZ(Rθ(zt,Ã),Z̃)——只让 Z̃ 跟随 rollout；③优化蒸馏——τ⋆ 作 detached 回归目标，"discards how τ⋆ changes with τ~"。
- **Dynamics-constrained 投影（§3.3, Eq. 4）**：可行集 Fθ(zt)={(A,Z): A∈U^H, Z=Rθ(zt,A)}；Πθ,g(τ̃)=argmin_{(A,Z)∈Fθ(zt)} J(Z,g)+λA·dA(A,Ã)+λZ·dZ(Z,Z̃)——"The task cost drives the trajectory toward the goal. The action reference preserves how the WAM proposes to move, while the state reference preserves where it proposes to go."；dA(A,Ã)=H^{-1}Σ_{h=0}^{H−1}‖Ah−Ãh‖²₂；dZ 归一化公式未取到（截断）。
- **部署（§1）**：训练期做完 TO 并摊销进生成器（"optimization amortization"）；部署时"directly generates candidate world-action trajectories and evaluates their action sequences with the world model"——轻量 **best-of-K** 选择，无迭代 TO。
- **结果（Abstract）**：7 个视觉控制环境（导航、到达、接触丰富操作、多阶段 goal-conditioned），平均成功率 **79.9% → 91.7%**；相对 CEM "on average **11.5×** faster and uses **600.5×** fewer world model evaluations at deployment"；"Removing either the action or state reference reduces performance"（§1）。
- **未取到**：隐式微分实现（§3.4/附录 D）、WATO 总目标（§3.5）、λ_WAF/λA/λZ/K 数值、§5 各表、附录 E 实现与 F.3 超参、附录 H 延迟——截断点之后。

---

## 8. ResWAM 2611.19559 —— DINOv3 视觉残差学习、0.5B 双支路 DiT（L2）

> 来源：arXiv HTML v1（`arxiv.org/html/2611.19559v1`）；截断于 §5.1 Fig. 4 图注（"…the flow-matching losses supervised by"），§5.2/§6/附录 C/D/E 未取到。作者：Michigan State / Cisco / IBM Research / U Michigan。代码 `https://github.com/OPTML-Group/ResWAM`、模型 `https://huggingface.co/collections/OPTML-Group/reswam-6af8906dc9b6f21539b30934`（首页声明）。

- **核心主张（Abstract）**："Learning What Changes, Not What Stays"——现有 WAM 监督完整未来观测，但"observations typically change little over an action chunk, so the future loss is dominated by copying static background content"；LIBERO 上当前与未来视觉特征平均余弦相似度 **0.96**（§1）。
- **视觉残差学习（§4.1）**：编码器 E 把观测映射为 N 个 patch token × D 维：Zt=E(ot)∈R^{N×D}；传统目标 Zt+H（Eq. 1）；残差目标 **Δt=Zt+H−Zt**（Eq. 2），行 δt,j=zt+H,j−zt,j——"the residual suppresses redundant static scene content while emphasizing temporal changes"。
- **残差为何更好（§4.2/§4.3）**：①秩：中心化 J=I−11^T/N 后，有效秩 reff=exp(H(p))、稳定秩 rstable=Σσi²/σ1²（Eq. 3）；LIBERO 有效秩 **15.22→22.06（+6.84）**、稳定秩 **3.08→5.48（+2.40）**；RoboTwin 2.0 **18.79→22.28（+3.49）**、**3.67→4.56（+0.89）**；②空间显著性：saliency map（逐 token 残差范数‖δt,j‖₂）+ PCA-RGB map（top-3 主成分）显示残差集中在机械臂/末端/被操作物体——"the surrounding background remains nearly inactive"。
- **架构（§5.1, Fig. 4）**：总参数量 **2.3B**，其中轻量 **0.5B 双支路 DiT**；VLM 编码当前图像 ot 与指令 c；**action 分支与 residual 分支**分别去噪动作 chunk Ât 与残差 Δ̂t，各自 cross-attend 到 VLM 输出与**冻结 DINOv3** 当前 tokens Zt，两分支间经 joint attention 交换信息；联合分布 πΘ(Ât, Δ̂t|ot, qt, c)（Eq. 4）；无像素重建、无未来解码（§1："without pixel reconstruction or future decoding"）。
- **训练/推理（§5.2 标题 + §1/Abstract）**：conditional flow matching 联合训练动作与残差分支（La 与 Lr 两 loss）；推理时同样联合去噪（具体去噪步数等细节未取到）。
- **消融关键值（§1）**：监督 Δt 比直接预测 Zt+H 在 RoboTwin 2.0 上平均成功率 **+2.9%**（同计算量，Fig. 7）。
- **结果（Abstract）**：无机器人预训练下 LIBERO-Plus **81.1%**、RoboTwin 2.0 **93.8%**、RoboDojo **9.40% success / 13.41 score**，均为同类最高平均成功率；训练算力 **2.7–4.0 TFLOPs**、推理 **1.0–1.4 TFLOPs** 全场最低；推理延迟 **92.2–99.7 ms**（§1）；真机 FR3 任务 Open/Put/Stack **80.0%**（+10.0% over π0.5）@ **92.7 ms**。
- **未取到**：§5.1 分支维度/层数细节、§5.2 训练与推理机制全文、§6 各表、附录 C 优化配置、附录 E 参数细分——截断点之后。

---

## 9. LaST0.5 2611.14048 —— interaction-centric latent expert + progress 一致性 + 14k 小时预训练（L2）

> 来源：arXiv HTML v1（`arxiv.org/html/2611.14048v1`）；Method 节完整，截断于 AWAT 末（"we report results for both LaST…"），Experiment 节与附录 A/B 未取到。作者：PKU + CUHK + Simplexity Robotics（Shanghang Zhang 等）。

- **定位（Abstract）**：现有 WAM（imagine-then-execute 与 joint modeling）都把全局场景表征直接传给动作——"global scene representations may underemphasize local robot-object interactions"；LaST0.5 在 world expert 与 action expert 之间插入 **interaction-centric latent expert**。
- **问题设定（Problem Formulation）**：模仿学习于异构语料（遥操作示范、UMI 手持夹爪、human egocentric）；输入多视角观测 It={It^k} 与指令 l；输出动作 chunk at:t+H；每臂动作 = 相对 **6-DoF EE 位姿** + embodiment 专属执行器指令（二值夹爪或多指手）。
- **联合分布分解（Model Design, Eq. 1）**：p(a,Z,V̂|ct)=p(V̂|ct)·p(Z|V̂,ct)·p(a|Z,V̂,ct)——未来视频、interaction 潜序列、动作 chunk 划分为 **M 个时间对齐段**（例：chunk 长 12 → M=3）；attention mask 限制跨 expert 注意力只到对应及前序段，显式构造 video→interaction→action 层级推理。
- **World Expert**：**Cosmos-Predict2.5-2B** 预训练 DiT；5 帧主视角历史经 causal video tokenizer 压缩为潜上下文，指令用缓存文本嵌入 cross-attention 注入；视频 **256×256**、VLM 分支图像另缩至 **224×224**；渐进去噪 **12 帧未来帧**；保留原 Cosmos 生成目标 L_world。
- **Latent Expert**：从 **Janus** 初始化；少量可学习 query tokens 单次并行前向抽取紧凑交互表征 Z——"these latent tokens are generated in a single parallel forward pass"。
- **Action Expert**：与 latent expert 同语言骨干初始化但参数独立；**flow-matching** 目标 L_act；每 timestep 两个专用噪声 token + MLP 头分别预测腕部运动与执行器指令；跨 embodiment 取最大维度、不足维零填充。
- **MoT 结构**：三 expert 共享全局 attention 流、权重独立；**blockwise-causal** 掩码（附录 A.2）；"We detach the world expert's keys and values to block backward gradient flow from the latent and action experts"——保护预训练物理先验。
- **ICLR 监督**：12 帧分 M 段、每段取末帧注入两类视觉交互线索——①**2D 曲线**=示范 3D EE 轨迹经相机参数投影到像面；②**目标 affordance 框**=Grounding DINO 按指令出候选框、与夹爪开合事件的投影 TCP 空间匹配（离线标注，推理不需要）；prompted 帧经**冻结 SigLIP** 编码为 Np 个 dv 维 patch token，可训练 MLP 聚合为 1×dv，投影归一化得 M 个潜目标 ZGT；潜损失 **L_lat=1−(1/M)Σcos(ẑi, zi^GT)**；总目标 **L=λ1·L_world+λ2·L_lat+λ3·L_act**；M 个可学习 query 训练后保留、推理直接使用。
- **AWAT**：联合训练后**冻结基座**，在世界 expert 去噪后的未来帧特征上挂轻量 MLP value 头 gφ——每段 4 帧经 Cosmos tokenizer 压为 video latent，**每段一个 progress 值**：v̂i=gφ(Pool(Hi))（Eq. 2，Hi=第 i 段世界模型 DiT 隐层 token 集）；真值 **vi^GT=ti/T**（ti=4 帧中位环境时刻、T=episode 长度）；L_value=(1/M)Σℓreg(v̂i, vi^GT)（Eq. 3）；训练时对部分片段**倒放（递减 progress）/重复帧（恒定 progress）** 做增广以暴露停滞/回退模式。
- **自适应执行**：仅当 **v̂i>v̂i−1** 时继续执行；progress 下降即截断、取新观测重新推理；若首值低于上一周期估计，则"sample multiple world-model futures in parallel"、取 value 最高者的 KV cache 供 latent/action expert 生成。
- **结果（Abstract/Intro）**：**14k+ 小时**异构机器人数据预训练（引 DROID/Open X-Embodiment 等）；LIBERO 平均 **99.5%**、RLBench 十任务 **88.0%**（超此前 SOTA）；真机 Tianji 双臂四任务，长程执行稳健、零样本泛化到未见物体/背景/位置。
- **未取到**：λ1/λ2/λ3 数值、M 具体取值、Experiment 各表、消融、附录 A.1/A.2/A.3 与附录 B 数据组成——截断点之后。

---

## 10. ThinkWAM 2611.21258 —— latent futures 动作精炼（proposal–rethink–refine）（L2）

> 来源：arXiv HTML **v3**（`arxiv.org/html/2611.21258v3`，19 Nov 2026 最新版）；截断于 §3.3 末尾（"Δmt^aff=gaff(st^aff), where"），§3.4/§3.5/§3.6/§4/附录 A 未取到。作者：Xi'an Jiaotong + Nankai + Manchester + NTU + Phantrix Research。代码 `https://github.com/HaoyunT/Thinkwam`、模型 `https://huggingface.co/Adrain11/ThinkWAM`、项目页 `https://thinkwam-project.github.io/`（v3 首页声明）。

- **核心论点（Abstract）**：初始动作 chunk 可能出错；而"a future prediction that is not conditioned on the chunk is identical for every proposal and thus carries no proposal-specific signal"——因此要执行前建立**双向反馈**。
- **proposal–rethink–refine 循环（Abstract）**："the policy first proposes a complete action chunk, the proposal corrects multi-horizon future latents through a learned residual, and the corrected latents are injected into a second pass of the same action head, which generates the executable chunk"——所有预测均在**潜空间**（不重建像素）。
- **组件（§3.2）**：输入多视角 ot、指令 ℓ、proprio qt、最近 **L 个已执行动作** At；输出 chunk At=(at..at+K−1)，并预测 horizons H 上每 horizon **Nh 个 token** 的未来视觉潜。三件套：①**冻结 V-JEPA 2.1 ViT-L** 编码器 Ev（视觉 tokens zt + 未来目标）；②**Qwen2.5-0.5B** 语言骨干 → 视觉前缀隐状态 Htv 与 language-action 特征；③**ActionDiT** 扩散 Transformer 动作头，flow matching：L_FM(c,A*)=E[‖vθ(xτ,τ;c,qt)−(A*−ϵ)‖²]（Eq. 1），xτ=(1−τ)ϵ+τA*。
- **Affordance-guided memory（§3.3）**：记忆 mt=M(zt,At)∈R^{S×dm}（Eq. 2）——S 个可学习 memory slots，每 slot cross-attend 到池化视觉 token 与投影动作历史 token 再过 FF；每决策步重算，"past information enters only through the executed actions"；temporal context transformer：Ut=T([φv(Htv); φm(mt)])（Eq. 3）。affordance 监督（predictive coding 动机）：decoder 读视觉前缀 Utv，逐 token 预测接触概率 pt,n，损失 **L_aff=BCE+λ_Dice·Dice**，目标为接触热力图按 token 网格面积平均的软标签；按确定性 ρt,n=2|pt,n−1/2| 加权汇总 st^aff=Σρ·Ut,nv/(Σρ+δ)，再经 g_aff 写回记忆残差 Δmt^aff（Eq. 4）——"we write a weighted summary of this context into memory"（§1）。
- **near-fine, far-coarse（Abstract/§1）**："allocating more tokens and using finer aggregation for near-future targets than for distant ones"；§3.4 细节未取到。
- **结果（Abstract）**：LIBERO **98.70%**、LIBERO-Plus **84.76%**（无 OOD 微调）、RoboTwin 2.0 Clean/Random **85.75%/65.85%**、RoboCasa24 **65.17%**；真机三种 OOD 条件均值 **64.4%**；两趟推理延迟 153.48 ms（LIBERO）/ 164.61 ms（RoboTwin）/ 101.2 ms（Galaxea A1Z 真机）；"Ablations show complementary gains from rethinking and affordance guidance"；诊断显示 proposal-conditioned 修正同时改变预测未来与动作轨迹。
- **未取到**：§3.3 记忆残差写入公式、§3.4 多 horizon 目标构造、**§3.5 rethink/refine 精确机制**（残差预测器/条件方式）、§3.6 训练推理流程、§4 各表、附录 A.1 参数量与 A.4 优化配置——截断点之后。

---

## 附：其余 30 篇一句话速览（L1 = arXiv 摘要）

> 说明：本清单为上游给定的一行摘要，未经本轮逐篇复核（来源级别 L1=arXiv 摘要）；上游条目清单实为 **27 条**，此处按给定清单逐条收录，未增补。

| # | 论文（arXiv ID） | 一句话（L1） |
|---|---|---|
| 1 | Mind2Act World 2611.26444 | 诊断基准（Mind/Act/Mind2Act 三视图，最强 agent 组合任务仅 12.0%） |
| 2 | SteerTrace 2611.25719 | WAM 定向攻击+内部效应追踪（4/255 扰动；恢复未来 token 移除 61-94% 效应） |
| 3 | LAQ 2611.25614 | 保未来保真度的 4-bit 量化（reader metric+read count；LIBERO-Plus 76.4/83.3，1.66× 加速） |
| 4 | CoachSI 2611.25529 | 冻结 VLM 当语言教练命名下一步→hindsight relabel 微调（长程 47.4→59.3） |
| 5 | MAWAM 2611.25105 | 多机器人联合去噪世界-动作模型（单次采样出全队动作+未来视图） |
| 6 | Z-sight 2611.24940 | 统一 VLA 与潜世界建模 |
| 7 | Phoenix 2611.24938 | 场景结构表征提升泛化 |
| 8 | HINT-WAM 2611.24789 | hindsight-informed 训练 |
| 9 | F³ 2611.24708 | 什么算 WAM 的好经验 |
| 10 | TrajWAM 2611.24192 | 轨迹条件未来学习（驾驶） |
| 11 | WAM-Policy 2611.24011 | 四足 loco-manipulation 的 on-policy 蒸馏 |
| 12 | LaST1.0 2611.23974 | 控制中心时空推理的统一潜世界-动作模型 |
| 13 | CoT-WAM 2611.23803 | VLM 与 WAM 经 plan latents 协同训练 |
| 14 | LoRN 2611.23628 | 生成式机器人策略的低秩噪声引导 |
| 15 | Making Future-Predictive Representations Actionable 2611.22994 | 让未来预测表征可执行 |
| 16 | Fly as Instructed 2611.22883 | UAV VLN 世界-动作策略精炼 |
| 17 | Distilling WAMs (Anchored Path Matching) 2611.22592 | 锚定路径匹配蒸馏双臂 |
| 18 | CURE 2611.22561 | 诊断并修复世界模型规划偏差 |
| 19 | EgoHandWAM 2611.22151 | 手部运动在 egocentric WAM 中的角色 |
| 20 | AR-WAM 2611.21851 | 自动驾驶流式 WAM |
| 21 | ReactBench-AD 2611.20011 | 驾驶 VLA/WAM 对突发危险的反应评测 |
| 22 | DiffWAM 2611.19904 | 从连续视觉差学习 |
| 23 | Cache or Denoise Less? 2611.19709 | 流匹配 WAM 的残差复用 |
| 24 | SALA 2611.17110 | 语义对齐潜动作双轴缩放（LAM vs WAM） |
| 25 | BEHAVIOR-1K 2nd place 2611.13818 | Cosmos 3 + 进度条件 + LoRA 路由（Q-score 0.613） |
| 26 | VLA Life-Cycle Survey 2611.18372 | 89 工作综述（含 WAM 对比） |
| 27 | LoongForge-Embodied 2611.25460 | 异质具身模型训练框架（1.47-4.38× 吞吐） |

---
