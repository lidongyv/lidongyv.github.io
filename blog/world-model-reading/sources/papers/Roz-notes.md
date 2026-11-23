# Roz: A Taste Test for World Model Latents — 精读笔记

> 数据源：`auto_docs/papers/txt/Roz.txt`（612 行，43,971 字符，13 页）。本文所有行号均指该文件行号。所有事实性陈述附论文原句；解读标注【推断】；论文没有的写「原文未写」。

## 1) 原文信息

- 标题：**"Roz: A Taste Test for World Model Latents"**（L1）
- 作者与机构：
  - "Kenneth Ge / Pantheon / kennethkouge@gmail.com"（L2-4）
  - "Dhruv Sheth / Pantheon / dhruvsheth@pantheon.inc"（L5-7）
  - "Kunvar Thaman / Pantheon / kunvar@pantheon.inc"（L8-10）
- 发表状态："Preprint."（L51）；"arXiv:2611.00612v1 [cs.RO] 31 Oct 2026"（L52）
- 代码："we release code at https://github.com/kenneth-ge/roz"（L66）
- 命名由来（脚注）："Inspired by The Wild Robot, and an acronym for Revealing Object Latents (Z)"（L99）
- 被评模型："a wide range of world model families, including Cosmos, Fast-W AM, Qwen3.6, V-JEPA 2, and DreamDojo"（L76-77）；"a controlled comparison of Wan2.2 against its embodied adaptation Motus"（L77-78）；以及自研模型 PantheonWM（L21、L380-385）。
- ⚠ 名称核对：任务描述中提到的 "AnonWM" 在原文中**未出现**；原文模型名写作 "PantheonWM"（L21、L380）。全文无 "AnonWM" 字样。

## 2) Abstract 全文逐字（L11-30）

> State of the art latent dynamics world models achieve impressive performance
> for robot planning. But do they encode enough information to accurately unfold
> dynamics?
>
> In this paper, we show that popular world models today achieve strong results
> on tasks that reduce to image similarity but forget key physical details like mass,
> deformation, friction, and bounciness. We achieve this by defining and justifying
> commonsense properties for what ought to be encoded in such a world model
> and introducing two new scalable evaluation methods. Our evaluation, which we
> call Roz, totals 1734 individual scenarios. We run Roz on a series of models and
> validate our eval on control tasks. On our own model, PantheonWM, we find that
> performance roughly improves with training steps, while not necessarily being
> correlated with reconstruction MSE.
>
> We argue current model failures are a symptom of evaluation strategies (probes and
> planners) that are opaque, model-specific, and high-compute. Roz requires only a
> model that can ingest still frames or video, a way to extract the latents it predicts
> from, and a distance function between them, and it can be run in one pass. We
> hope future evaluations will adopt Roz’s evaluation techniques. More broadly, we
> hope Roz can be a case study for a future where benchmarks allow rapid iteration
> and comparison across latent world models.

## 3) 问题与动机

- 机器人评测成本远高于语言建模："In comparison to language modeling, for a given model size and capability level, robotics requires an order of magnitude more compute and downstream work to evaluate."（L32-33）；原因包括 "the expensive nature of physical testing and the unreliability of simulations"（L36-37）。
- 世界模型理论上带来两个收益："a high fidelity world model could let us improve prediction and planning separately [Sutton, 1991], and an intermediate representation can help debug where errors happen [Ha and Schmidhuber, 2018]"（L39-41）。
- 现有两条评测路线及其缺陷：
  - linear probes："Probes require annotating training data [Belinkov, 2022], mostly cover high-level features [Cunningham et al., 2024], introduce confounding computations [Hewitt and Liang, 2019], and ultimately do not necessarily guarantee coherence of the underlying representation [Vafa et al., 2024]."（L46-49）
  - 下游任务（planners）："Assessing via downstream performance treats our representation as a black box, is resource intensive, and complicates debugging. Neither prioritizes encoding physical properties."（L49-50）
- 核心现象主张："popular world models today achieve strong results on tasks that reduce to image similarity but forget key physical details like mass, deformation, friction, and bounciness"（L15-17）。
- 四条贡献（L69-80）：
  1. "Rigorous and commonsense definitions for what information a latent dynamics world model should encode in its latents"（L70-71）
  2. "Two scalable evaluation techniques: contrastive and continuous nearest-neighbor, split into three task types, with varying strengths of properties they require from the distance function"（L72-73）
  3. "A probe-free, training-free, model-agnostic eval of world model latents, in a task suite totaling 10 contrastive tasks, 3 continuous tasks, and 1734 scenarios"（L74-75）
  4. "Performance numbers on a wide range of world model families, including Cosmos, FastWAM, Qwen3.6, V-JEPA 2, and DreamDojo, and the capability limitations we observe in each one; and a controlled comparison of Wan2.2 against its embodied adaptation Motus"（L76-78）；"Validation that eval performance roughly improves with training time on our own world model, PantheonWM"（L79-80）

### 3.1 潜空间应编码的定义（Section 3）

- 当前状态："The current state of objects in the world, such as the positions, velocities, rotations, colors, and shapes of all value-relevant objects."（L96-97）
- 被揭示的性质："Revealed properties that could affect future dynamics, such as mass, friction, restitution, deformation, viscosity, and buoyancy. These are typically only observed when objects interact with each other, and are revealed through history."（L105-107）
- "Importantly, many of these properties can only be revealed over time, naturally incentivizing some encoding of the trajectory history."（L108-109）
- 三条直观性质：
  - Path independence："If two paths reveal the same information (e.g. friction, mass) and have the same ending state, their representations should be roughly equivalent."（L113-114）
  - Memory："A trajectory that discovers more information about the world should naturally encode this and be distinct from one that encodes less, even if the endpoint is the same (Fig. 1)."（L115-117）
  - Temporal Consistency："Two segments encoding identical trajectories with time steps t to t+x and t+ϵ to t+x should result in the same representation."（L118-119）

## 4) 评测方法实现细节（重点）

### 4.1 形式化框架（Section 4）

- 符号：潜空间 Z∈R^n；观测空间 O、动作空间 A；轨迹前缀 τ1:t = (o1, a1, o2, …, at−1, ot)∈T；任务无动作，故 "we take A={∅}, a single null “observe” action"（L128-131）。
- "Every latent world model has a point at which observed history ends and prediction begins. We write Eθ : T → Z for the map from a prefix to the latent at that point, zt = Eθ(τ1:t), and Dθ : Z × A → Z for each prediction ... A full world model is Wθ = (Eθ, Dθ); k-step prediction is D^kθ applied to zt."（L132-135）
- 距离函数："Let d : Z × Z → R≥0 be some distance function between two latents."（L136）
- 评测对象与"天花板"论点："Roz tests whether the latent produced by Eθ could contain the dynamics information defined in Section 3. Observe that this puts a ceiling on the fidelity of the predictions produced by Dθ, which are used for planning (or “dreaming” as it is sometimes called)."（L137-139）

### 4.2 输入/接口要求（Section 5，逐条）

- 模型形态："Our evaluation takes any world model of the form Wθ : [video]→Z"（L181）
- 帧规格："in practice this means each observation ot ∈R 256×256×3 is a 256×256 pixel frame, with inter-frame prediction turned off in the video encoder to eliminate artifacts"（L181-183）
- 双目可选："We optionally adapt this to stereo vision systems."（L183）
- 潜变量提取："When models don’t output latents directly, we pick some layer and extract the hidden state."（L183-184）
- 模型特定信息（双目/本体感知等）："we designed our sim tasks such that any model-specific additional information (stereo vision, proprioception, etc.) could be easily extracted from the scene data or added post-hoc"（L185-187）
- 距离函数：双度量——"Since there is no agreed upon ideal distance metric, we test both L1 distance and cosine similarity and report numbers for both."（L188-189）；先例："L1 was used for planning in V-JEPA 2-AC [Assran et al., 2025], and cosine similarity was used for training due to its scale invariance [Schwarzer et al., 2021]"（L189-191）
- 度量假设："Like linear probes, using cosine similarity assumes continuous differences are largely represented linearly. Meanwhile, L1 assumes that similar features should cluster together in the latent space."（L191-194）
- 默认不渲染本体、不以动作条件化："we do not render any embodiments or condition actions on the world model to avoid introducing confounding factors"（L197-199）；"All scenarios are hand-designed and human validated."（L200）
- 各模型潜提取适配（Appendix B）：
  - Qwen3.6："We run a single decoder forward over the full clip’s native video tokens, generating no text, and read the mid-layer hidden states at the token positions depicting the final second—positions that, through causal attention, have integrated the entire preceding clip."（L516-518）
  - FastWAM："We hook the video expert’s middle-plus-one block, because that is where the scene has been encoded. It sees only the clip’s final frame, supplied as two stereo views side by side, with a fixed cached prompt embedding standing in for a task instruction."（L520-523）
  - Cosmos："We condition the diffusion pipeline on the clip’s last 5 frames and run a single denoising forward, hooking layer 33 of 64 on the generation (video) stream."（L524-526）
  - V-JEPA 2："We run the whole clip through V-JEPA 2’s encoder in one pass and keep the token grid for the last second, flattened in place. The clip is resampled to the checkpoint’s 64 frame slots"（L528-530）
  - DreamDojo："We encode the whole observed 13-frame clip with its video VAE, add diffusion noise at a fixed level, and run a single denoising forward with stationary latent actions. The pre-declared primary readout is the final transformer block at the final video position, pooled to an 8×8 spatial grid, at noise level 500 of 1000."（L532-535）
  - Wan2.2 / Motus："Both models receive the same whole observed 13-frame clips at 640×480 with an empty text prompt and a clean first latent, and use the same readout as DreamDojo. They share the same video VAE."（L542-543）

### 4.3 Contrastive 范式（L143-165）

- 灵感来源："Like the triangle taste test in food science [International Organization for Standardization, 2021], all our contrastive tasks test whether the model correctly distinguishes the odd one out among three scenarios, A, B, and C."（L144-146）
- 成功条件（**严格不等式**）："The model succeeds if it correctly generates latent predictions such that d(Eθ(A), Eθ(B)) < d(Eθ(A), Eθ(C)) and d(Eθ(A), Eθ(B)) < d(Eθ(B), Eθ(C)). Note that we grade with strict inequalities to enforce information-distinctiveness."（L147-150）
- 防退化设计："The primary constraint is that scenarios A and B must be distinct in some way to prevent the degenerate solution where A and B map trivially to the same latent."（L151-152）；做法是 A、B "sharing the same physical endpoint but different histories"（L154）
- 两种子类型：
  - Distinctiveness（6.1）：A/B 信息等价、差异只来自历史或非物理/非价值相关因素（如光照），C 终点相同但历史揭示不同信息（L156-160）
  - Magnitude Distinctness（6.2）："we loosened the requirement that A and B be exactly equivalent by allowing B to be some slightly perturbed version of A and making C dramatically different from either of the two"（L161-163）；"we require that large physical changes result in a greater distance than small perturbations"（L164-165）

### 4.4 Continuous 范式（L166-172）

- "This evaluation technique takes some continuous property and sweeps across it. For example, we might vary the position of a box from 0 meters to 10 meters, going in 1 m increments."（L166-168）
- 判据：每个扫描点的两个最近邻恰为相邻两步——"Our goal is then to ensure that the two nearest neighbors of each point in the sweep are the two neighboring steps."（L168-169）
- 边界不计分："we don’t grade the edges"（L171）
- 打分方式："We grade by the percentage of steps for which this property holds."（L172）
- 所需性质："we require that our distance function maps physical changes to related changes in the distance of the output representation"（L173-174）

### 4.5 任务清单（各测什么物理性质，Section 6）

**Contrastive — Distinctiveness（Section 6.1，8 个）**

| 任务 | 测什么物理性质 | 原句（行号） |
|---|---|---|
| Occlusion（L203-205） | 额外旋转历史揭示的背面颜色信息 | "we get one extra rotation of history for B. A and B both have a red backside, whereas C is blue all around."（L204-205） |
| Roll（L206-209） | 滚动摩擦信息 | "Since A and B both contain rolling friction information, this eval tests whether the model correctly encodes them as being the same, whereas C is different because it does not."（L208-209） |
| Bowl（L210-214） | 滚动摩擦（经历史揭示） | "Given that all three share the same endpoint, this eval tests whether the model encodes revealed rolling friction."（L213-214） |
| Bounce（L215-219） | 弹性/bounciness（恢复系数 restitution） | "this eval tests that the world model encodes bounciness"（L217-218）；"we sweep the ratio of restitution r1/r2 from 0.5 to 2"（L219） |
| Deform（L220-223） | 形变/刚度 | "in C, the cube deforms to a much different degree"（L221）；"We sweep across varying ratios of stiffness in C vs. A, from 0.5 to 2 excluding 1."（L222-223） |
| Collision（L224-230） | 质量 | "This eval tests whether the model notices that m1C ≠ m1A."（L229）；"We sweep the ratio f from 0.5 to 2, excluding 1, and average over k."（L229-230） |
| Path-Independence（L231-233） | 终点 vs 路径 | "This eval tests that final position matters more than path, holding all paths as free form through space."（L233） |
| Paper Airplanes（L234-237） | 真实世界飞行品质（real 任务） | "This is one of our tasks in real."（L234）；"the revealed information is how well the paper airplane flies, a property that cannot be inferred from the endpoint alone."（L236-237） |

**Contrastive — Magnitude Distinctiveness（Section 6.2，3 个）**

- Basic Position（L254-255）："A, B, and C each contain a pyramid in some positions x, x+ϵ, and y far from x respectively."
- Basic Color（L256-257）："A, B, and C each contain a cube in the same position, each with a hue h, h±0.04, and h+0.5±0.05"
- Count（L258-260）："A, B, and C each have n, n+1, and n+3 cubes respectively, sweeping n from 2 to 10."；"Holding the positions of shared objects fixed, this evaluation tests that adding two or three additional objects is more different than adding one additional object."（L259-260）

**Continuous（Section 6.3，3 个）**

- Translation（L262-263）："we place an arrangement of multicolored cubes and sweep across the camera position. We keep every object visible in every frame."
- Rotation（L264-266）："we place another arrangement of cubes and rotate the camera around the scene, sweeping the viewing angle 180 degrees."
- Velocity（L267）："We render a moving ball and sweep across different velocities, with the endpoint remaining the same and history encoding the speed."

**场景规模**：总量 "1734 individual scenarios"（L19-20、L75）。Table 1 列出的部分任务场景数：bowl (4)、path ind (30)、basic color (30)、basic position (30)、roll (30)、occlusion (30)（L243-248）。1734 的逐任务分解**原文未写**。
- 任务计数疑点：贡献条款称 "totaling 10 contrastive tasks, 3 continuous tasks"（L74-75），但 6.1 列出 8 个 + 6.2 列出 3 个 = 11 个 contrastive 任务名。若排除 real-world 任务 Paper Airplanes（"one of our tasks in real"，L234）则恰为 10。【推断：贡献中的 10 指 sim contrastive 任务，Paper Airplanes 为额外的 real 任务；原文未明说】
- DreamDojo/Wan2.2/Motus 子集："The DreamDojo, Wan2.2, and Motus runs use a 54-case subset of Roz: 48 contrastive triplets and six 16-video continuous ladders, rendered as 240 clips and 720 camera sequences."（L545-546）

### 4.6 与 probes / planners 的对比

- probes 缺点（同 §3）：需标注训练数据、多覆盖高层特征、引入混淆计算、不保证表征一致性（L46-49）。
- planners（下游评测）缺点：黑盒、资源密集、难调试、不优先物理性质（L49-50）。
- Abstract 直接对比："current model failures are a symptom of evaluation strategies (probes and planners) that are opaque, model-specific, and high-compute. Roz requires only a model that can ingest still frames or video, a way to extract the latents it predicts from, and a distance function between them, and it can be run in one pass."（L24-27）
- Roz 自身定位："probe-free, training-free, model-agnostic"（L74-75）；单次前向 "one pass"（L27）。

## 5) 伪代码（评测流程）

**原文没有 Algorithm 环境**（全文 612 行未见算法块），以下依据正文重建：

```
⚠ 依据原文重建（非原文伪代码）

输入：世界模型 Wθ=(Eθ,Dθ)                       # 行132-135
     任务场景轨迹三元组 A,B,C∈T                  # 行147
     距离函数 d∈{L1, cosine similarity}          # 行188-189
     视频帧 256×256，编码器关闭帧间预测           # 行181-183
     若模型不直接输出潜变量，取某一层隐藏状态     # 行183-184
输出：每任务 pass rate（contrastive）或最近邻正确步数百分比（continuous）

-- Contrastive（行143-150）--
for 每个场景三元组 (A,B,C):
    zA ← Eθ(A); zB ← Eθ(B); zC ← Eθ(C)          # 行147-148
    if d(zA,zB) < d(zA,zC) and d(zA,zB) < d(zB,zC):   # 严格不等式，行148-150
        pass
    else:
        fail
pass rate ← pass 数 / 场景数；机会水平 ≈ 1/3     # 行250

-- Continuous（行166-172）--
for 连续性质 p 的每个扫描值 p_i（例：位置 0→10 m，步长 1 m，行167-168）:
    检查 p_i 的两个最近邻是否恰为 p_{i-1} 与 p_{i+1}   # 行169-171
    边界点不计分："we don't grade the edges"          # 行171
得分 ← 满足条件的步数百分比                          # 行172
```

## 6) 主结果数值表

### 6.1 Table 1（非扫描 contrastive pass rate，L243-248；机会水平 ≈0.33，L250）

| 任务(N) | V-JEPA 2 cos/L1 | Qwen cos/L1 | Cosmos cos/L1 | FastWAM cos/L1 | DreamDojo† cos/L1 |
|---|---|---|---|---|---|
| bowl (4) | 1.00/1.00 | 1.00/1.00 | 0.75/0.75 | 1.00/1.00 | 1.00/1.00 |
| path ind (30) | 0.97/0.97 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |
| basic color (30) | 0.90/0.90 | 0.97/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |
| basic position (30) | 1.00/1.00 | 0.97/0.97 | 1.00/0.97 | 1.00/1.00 | 1.00/1.00 |
| roll (30) | 0.00/0.00 | 0.77/0.83 | 0.93/0.97 | 0.40/0.43 | 1.00/1.00 |
| occlusion (30) | 0.00/0.00 | 0.07/0.03 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |

（† DreamDojo：4 场景子集，报告 14B pretrain 10k 步检查点，L251-252）

### 6.2 按物理性质聚合（对应"质量/形变/摩擦/弹性"）

- **质量**（Collision 扫描 m2/m1，L312-321）：FastWAM cos/L1 全部 1.00（除 2.00 行 L1=1.00、cos=1.00，实际全 1.00；2.00 行为 1.00/1.00）；Cosmos 0.12–1.00 波动；Qwen 0.00–0.75；V-JEPA 2 全 0.00。
- **形变**（Deform 扫描 k2/k1，L302-311）：全员接近 0；FastWAM 最高 0.50/0.38（比率 0.50 行）；V-JEPA 2 与 Qwen 全 0.00。
- **摩擦**（Roll，L247）：V-JEPA 2 0.00/0.00；FastWAM 0.40/0.43；Qwen 0.77/0.83；Cosmos 0.93/0.97。（Bowl 摩擦任务全员 0.75–1.00，L243）
- **弹性/bounciness**（Bounce 扫描 r2/r1，L292-301）：全员接近 0；非零最高 0.12（Qwen 0.50 行 cos、2.00 行；FastWAM 1.50 与 2.00 行）。
- **Occlusion**（L248）：除 Qwen 0.07/0.03 外全部 0.00。

### 6.3 Table 3（continuous 最近邻，L324-328）

| 任务 | V-JEPA 2 cos/L1 | Qwen cos/L1 | Cosmos cos/L1 | FastWAM cos/L1 | DreamDojo† cos/L1 |
|---|---|---|---|---|---|
| translation nn | 0.89/0.92 | 0.71/0.71 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |
| rotation nn | 0.47/0.49 | 0.76/0.91 | 1.00/1.00 | 0.91/0.92 | 1.00/1.00 |
| velocity nn | 0.50/0.54 | 0.53/0.59 | 0.16/0.22 | 0.04/0.06 | 0.32/0.14 |

### 6.4 Count 扫描（L283-291）

- V-JEPA 2：0.25–0.88 波动；Qwen：0.75–1.00；Cosmos：0.88–1.00；FastWAM：0.50–1.00。原文结论："Despite being one of our simplest and most fundamental tasks, performance on Count shows gaps for all four models."（L342-343）

### 6.5 关键结论（原句）

- "FastWAM was the strongest performer, followed by Cosmos, which is consistent with their emphasis on building strong physics priors through pretraining."（L330-332）
- "Yet, both models fail on exactly the tasks we predict: ones where history matters and properties cannot be inferred from the endpoints alone. FastW AM performs at chance level on Deform. Our results suggest that inferring physical properties from priors alone may be insufficient."（L332-334）
- "All models we tested perform well on tasks that require image robustness in the endpoint (bowl, path independence, basic color, basic position, translation, rotation), but all struggle to remember physical properties (occlusion, bounce, deform)."（L335-337）
- "Despite taking in 64 frames of video, V-JEPA 2 consistently performs the worst."（L341-342）
- DreamDojo："Its total of 30 passes out of 42 non-null cases is matched by its own frozen video V AE (31 of 42) and nearly matched by final-frame pixels (28 of 42), suggesting that on these tasks the transformer adds little dynamics information beyond what the V AE already encodes."（L347-350）

### 6.6 训练步 vs 重建 MSE（⚠ 原文模型名 PantheonWM，非 AnonWM）

- PantheonWM 检查点梯度（Table 5，L369-385）："We rank our checkpoints by pixel reconstruction MSE which roughly correlates with training steps, with worst (20 steps), low (1k steps), middle (10k steps), and best (10k steps, different hyperparameters, lower MSE)."（L381-382）；mean rank（1=best）：worst 3.00、low 2.81、middle 2.04、best 2.15（L370）。
  - "We find that evaluation performance roughly improves with training steps. Our middle checkpoint edges out our best checkpoint, suggesting this evaluation can help discriminate dynamics sensitivity of our models that reconstruction and training steps alone cannot."（L383-385）
- DreamDojo 对照（Table 4，L354-368）：MSE×10^-3 随训练步下降（2B pretrain 6.6→5.9；2B YAM 13.3→7.1；14B pretrain 8.0→7.0；14B YAM 10.4→7.3），但 contrastive 列全部 = 30（42 中）；frozen VAE = 31；final-frame pixels = 28。
  - "Across nine released checkpoints spanning two model sizes (2B and 14B), two training stages (pretrain and YAM finetune), and 1k to 10k steps, robot-video reconstruction MSE falls by roughly half, yet the contrastive pass count is identical at every checkpoint and the velocity ladder shows no monotone trend (Table 4)."（L386-389）
- 两个梯度对照（核心论点）："PantheonWM shows that Roz can move with training when the objective rewards dynamics memory. DreamDojo shows that reconstruction loss can improve substantially with no change in what the latents encode about physical properties, and that Roz detects this gap where MSE cannot."（L390-393）
- Abstract 的表述（注意程度副词）："performance roughly improves with training steps, while **not necessarily** being correlated with reconstruction MSE"（L21-23）——原文未给出相关系数等定量证据；定量证据是 DreamDojo 的 MSE 减半而 Roz 分数不变（L386-389）。

## 7) 消融

- 距离函数一致性："we find that L1 and cosine similarity largely agree."（L277-278）
- 正/负控制成立："Basic Color and Basic Position act as positive controls where any reasonable metric should pass ... We find that this holds. Our Occlusion, Bounce, and Velocity evals act as negative controls on Cosmos and FastW AM, which do not see enough history to notice the revealed information (they see the last 5 frames and last frame respectively). We find that this holds as well, noting that both models score close to 0."（L271-277）
- DreamDojo pooling 消融（Appendix C，Table 6，L552-556）：8×8 spatial grid vs global mean pooling——"Contrastive results are nearly unchanged, with only Path-Independence dropping, but the Rotation and Velocity ladders collapse under global pooling, since averaging away spatial layout removes the information those ladders order by."（L553-556）；具体数值（L564-577）：rotation 1.00→0.14，velocity 0.32→0.11，path ind 1.00→0.50。
- Hidden-location（shell game）任务（Appendix C，L557-594）：测顺序依赖记忆；"Roz geometry stayed at 37.5% (48 of 128 relations) for every DreamDojo checkpoint and noise draw."（L588-589）；诊断用 ridge 分类器 "recovered the hidden slot at 55 to 62% on familiar move patterns, above the 50% ceiling available from initial position plus unordered move counts but below a 70% practical target; a mid-block readout reached 71 to 76%."（L589-592）；泛化："accuracy on novel combinations fell to 28 to 32%, at or below the 33% best-constant baseline. Location is thus partly readable on familiar templates, but the readout does not transfer."（L592-593）
- Wan2.2 vs Motus 对照（Appendix C，L595-606）："Both models scored exactly 37.5% on the primary Roz relation under both noise draws."（L597-598）；novel-combination recovery：Wan 22.0%/16.7%，Motus 15.4%/15.5%（L598-599）；paired 差异 −6.6 pts（95% 区间 [−9.4,−3.9]）与 −1.2 pts（[−4.0,+1.7]）（L599-601）；"We conclude that Stage 1 adaptation left Roz performance unchanged: the adapted model inherits the backbone’s representational limits rather than repairing them."（L602-603）

## 8) 局限（逐字，Section 8 = L396-410）

> "Our paper requires that each model be evaluated with a well-formed distance function. While we use controls and agreement to validate L1 and cosine similarity, we also recognize that fully validating a distance function is nontrivial and it is not yet established what these metrics ought to be. Future work could possibly use a learned distance. We also recognize that choice of distance function could be a confound, hence why we use two simple, established distance functions. Relatedly, the readout used to extract a latent (which layer and how tokens are pooled) is a per-model choice that affects continuous-task scores, as we observe for DreamDojo. We therefore recommend pre-declaring a readout before running the evaluation.
>
> Most of our scenarios are in sim, which allows finer-grained control over confounding factors and easy adaptation to new input interfaces (e.g. multi-camera systems) and embodiments. To overcome the sim-to-real gap, we design our scenarios such that differences are obvious to detect. This presents a realism tradeoff where finer-grained differences might be harder to evaluate. Future work could use these techniques to build evals in real."

附录中的附加局限（Wan vs Motus，L604-606，逐字）：

> "This is a single parent-and-child pair with differing release precisions (Wan in float32, Motus in bfloat16), so it does not isolate the training effect, and later Motus stages that add action conditioning may behave differently."

## 9) 叙事关联【推断】（≤4 条）

1.【推断】Roz 自我定位为"世界模型版的 MMLU / HellaSwag"——"we hope this benchmark helps bring about a hill climbing era for latent world model representations"（L67-68）、"benchmarks allow rapid iteration and comparison across latent world models"（L29-30）。可支撑"用轻量、免训练、单次前向的评测驱动潜世界模型快速迭代"的研究叙事。
2.【推断】"图像相似 vs 物理性质记忆"的二分（L15-17、L335-337）叠加 DreamDojo 案例（重建 MSE 减半而 Roz 分数不变，L386-393），支持一个更强论点：**像素重建损失不是动力学记忆的有效代理指标**，评估潜世界模型不能只看重建质量。
3.【推断】三条定义性质（Path independence / Memory / Temporal Consistency，L112-119）可当作潜空间设计的规范清单，任何声称"动力学完备"的潜世界模型可按此逐项自查。
4.【推断】PantheonWM 梯度表明 Roz 分数可随训练步提升（L383-385），暗示该评测有潜力被纳入训练目标或模型选型；但原文**未写**他们将 Roz 用于训练，仅用于评测。

---

（笔记完。数据源唯一为 Roz.txt，未联网、未补写外部知识。）
