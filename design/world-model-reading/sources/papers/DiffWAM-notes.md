# DiffWAM 精读笔记

> 数据源（唯一）：`C:\Users\Admin\Desktop\design\auto_docs\papers\txt\DiffWAM.txt`（1830 行，arXiv 2611.19904，21 页）。
> 规则：所有事实性陈述附论文原句（英文逐字 + 行号）；论文没有的写「原文未写」；解读标注【推断】。
> 行号格式 `(L行号)`；正文中 "DiffW AM" 为 txt 排版断词痕迹，指 "DiffWAM"。

---

## 1) 原文信息

- 标题："DiffWAM: Learning World Action Model from Consecutive Visual Difference" (L1)
- 作者："Xinheng Zhang1,2,*, Haoyu Xiong5,3*, Chao Jing6,3, Xiaohan Zhang1, Linshan Li1, Hongbang Fan5,3, and Xingyu Chen3,4B" (L2-3)；"*Equal contribution. B Corresponding author." (L88) → 通讯作者为 Xingyu Chen。
- 机构 (L4-7)：
  - "1College of Information Science and Electronic Engineering, Zhejiang University"
  - "2Polytechnic Institute, Zhejiang University"
  - "3Zhongguancun Academy" / "4Zhongguancun Institute of Artificial Intelligence"
  - "5Institute of Automation, Chinese Academy of Sciences"
  - "6School of Intelligent Robotics and Advanced Manufacturing Innovation, Fudan University"
- ID："arXiv:2611.19904v1  [cs.RO]  15 Nov 2026" (L90)；"November 2026" (L89)
- 发表状态：arXiv 预印本 v1（2026-11-15）。是否被会议/期刊接收：原文未写。
- 项目页/代码："https://zgca-hmi-lab.github.io/DiffWAM" (L8)、"https://github.com/ZGCA-HMI-Lab/DiffWAM" (L9)

## 2) Abstract 全文逐字（L92-121）

> **Abstract** (L92)
> "World Action Models (WAMs) have emerged as an impor-(L93)
> tant paradigm in robot learning, jointly predicting future (L94)
> scene states and robot actions from current observations and (L95)
> instructions. However, future states entangle task-related (L96)
> changes with large amounts of task-irrelevant content, such (L97)
> as static backgrounds, which weakens the model's ability to (L98)
> capture changes relevant to task execution. To address this, (L99)
> we propose Difference World Action Model (DiffWAM), a (L100)
> Mixture-of-Experts (MoE) framework that enhances world (L101)
> representations by jointly learning future-state prediction (L102)
> and consecutive visual difference prediction. We define con-(L103)
> secutive visual differences as differences between the latent (L104)
> representations of consecutive observations, providing an (L105)
> explicit learning signal for temporal variation. The frame-(L106)
> work comprises a World Expert for future frame prediction, (L107)
> an Action Expert for action generation, and a Diff Expert (L108)
> for predicting these latent differences along the trajectory. A (L109)
> structured unidirectional attention mask conditions the Diff, (L110)
> World, and Action Experts solely on the current-frame rep-(L111)
> resentation while blocking attention between experts. This (L112)
> design allows both the World and Diff Experts to be removed (L113)
> during inference, introducing no additional inference over-(L114)
> head. DiffWAM achieves state-of-the-art performance on (L115)
> the LIBERO and RoboTwin 2.0 benchmarks and substan-(L116)
> tially improves over FastWAM on real-world manipulation (L117)
> tasks. Specifically, it attains average success rates of 98.8%, (L118)
> 92.65%, and 88.3%, outperforming the state-of-the-art base-(L119)
> line (i.e., FastWAM) by 1.75, 1.55, and 15.0 percentage (L120)
> points, respectively." (L121)

（注：逐字保留 txt 换行；L93-94 "impor-…tant"、L100/110 等 "DiffW AM" 均为排版断词，非原文拼写错误。）

## 3) 问题与动机（简）

- 核心问题：未来状态的预测目标把「任务相关变化」与大量「任务无关内容」混在一起监督，弱化任务相关表征：
  - "However, future states entangle task-related changes with large amounts of task-irrelevant content, such as static backgrounds, which weakens the model's ability to capture changes relevant to task execution." (L96-99)
  - "The full-state prediction objective supervises these changes together with persistent content, without providing a separate learning signal for temporal variation. Thus, the representation of task-related changes relies on what the model learns implicitly from predicting complete future states." (L141-146)
- 关键洞察（principle of visual foresight）："This observation suggests a principle of visual foresight: a WAM should model not only what the future scene will look like, but also how it changes over time." (L147-149)
- 三大范式的定位（Related Work）：VLA（缺显式场景演化监督）、WAM with Future Prediction（不区分持久内容与帧间差异）、WAM with Structured Signal Prediction（依赖外部先验/估计器，可能带误差）：
  - "In contrast, DiffWAM directly models differences between consecutive frame latents alongside future states, integrating state and visual difference prediction within a WAM." (L367-369)
  - "Their modeling emphasis therefore depends on the chosen signal, and targets obtained from external estimators may contain estimation errors." (L273-275)

## 4) 实现细节（重点）

### 4.1 consecutive visual difference 的数学定义

- 联合建模目标（式 4）："pθ(f1:T ,∆f 1:T , a1:H |f 0, l),(4)" (L385)，其中 "f1:T denotes the future visual latents, a1:H the action chunk, and ∆f1:T the consecutive visual differences represented in latent space" (L386-388)。
- 定义（式 5）："∆fi =f i −f i−1, i= 1, . . . , T.(5)" (L390) —— 相邻帧 VAE 潜表示的差。
- "Let fi denote the VAE latent representation of observation oi, T the number of future visual states, and H the action chunk length." (L381-382)。T、H 的具体数值：原文未写。
- 语义边界：不限于机器人致变："Latent differences describe visual variation and may also include nuisance changes, rather than exclusively encoding changes caused by robot actions." (L403-405)

### 4.2 World / Action / Diff 三专家结构

- 总览："DiffWAM is a Mixture-of-Experts framework with a shared Observation Encoder and dedicated World, Diff, and Action Experts. The Observation Encoder and each expert contain N Transformer layers. The World and Diff Experts share parameters in their first N−M layers and use independent parameters in their last M layers." (L416-422)
- "Parameter sharing between the World and Diff Experts does not introduce cross-attention between their independent prediction pathways." (L429-431)
- 输入编码："DiffWAM inherits the pretrained text encoder and video VAE from Wan2.2-5B [49]. The task instruction l is encoded with the T5 text encoder [50]... Images from multiple camera views are concatenated before being mapped into visual latents by the pretrained VAE [51]. The clean current-frame latents f0 are processed by the Observation Encoder to provide the shared conditioning representation." (L432-439)
- 训练时视觉输入共享："During training, both the World and Diff Experts receive the same noised future-frame latents as input." (L440-441)；Diff 目标来自连续真值帧潜表示："the Diff Expert directly regresses latent differences, with targets computed from consecutive ground-truth visual latents" (L443-444)；"Actions a1:H are separately noised and embedded for the Action Expert." (L445-446)
- **World Expert**："The World Expert learns the velocity field used for future-state generation through flow matching. Its prediction pathway is used during training and is removed at inference" (L452-455)；输入流为 "Noised future-frame latents are processed by the first N−M layers shared with the Diff Expert, followed by M independent state-prediction layers." (L450-452)
- **Diff Expert**："The Diff Expert processes the same noised future-frame latents through the shared first N−M layers, followed by an independent M-layer Latent Diff Block (LDB). Conditioned on the current-observation representation and language instruction, it directly regresses inter-frame latent differences rather than generating future states through denoising. Its independent pathway does not attend to the World or Action Expert's tokens, and its outputs are not passed to these experts." (L457-465)
  - 回归目标含 i=1 项："The regression targets are differences between consecutive ground-truth frame latents, including the transition from the current observation to the first future state." (L465-468)
- **Action Expert**："The Action Expert generates the action chunk through flow matching, conditioned on the current-observation representation... It does not directly receive predicted future states or diff outputs." (L471-475)；"State and visual difference modeling improve the representation used by the action pathway through joint training." (L475-477)
- 专家间信号隔离："The joint objective does not require the Action Expert to consume predicted future states or latent differences; each expert is conditioned on the current-observation representation and language instruction." (L400-403)
- 规模（4.1）："The World and Diff Experts use the Wan2.2-5B Transformer architecture [49]... The Action Expert follows the same architecture but uses a reduced hidden dimension of da = 1024. The complete model contains approximately 7B parameters." (L665-671)

### 4.3 单向注意力掩码（结构化）

- 总规则（逐字）："a structured unidirectional attention mask controls information flow between the current-observation pathway and the experts. Current-observation tokens do not attend to future-state, diff, or action tokens. Each expert may attend to the current-observation representation and tokens within its own pathway, while attention between different expert pathways is blocked." (L480-487)
- 语言条件走 cross-attention，不参与上述掩码："These restrictions concern interactions between visual and action token groups; language conditioning is supplied separately through cross-attention." (L487-489)
- **(a) 前 N−M 层（World/Diff 共享视觉层）**："In the first N−M layers, the visual and action pathways follow a first-frame-conditioned attention scheme [15]: current-frame tokens attend to themselves, noised future-frame tokens attend to the current-frame representation and their own tokens, and action tokens attend to the current-frame representation and their own tokens. Future-frame and action tokens do not attend to each other. The World and Diff Experts share the visual processing layers at this depth." (L490-499)
  - 「first-frame-conditioned attention scheme [15]」= 继承自 FastWAM（参考文献 [15] = Fast-WAM，L1187-1189）。
- **(b) 最后 M 层（独立预测通路）**："Diff tokens attend to the current-frame representation and their own tokens, while the state and action pathways retain the attention restrictions in (a). Blocking cross-attention between these pathways prevents the Diff Expert from reading the World or Action Expert's token representations and prevents its outputs from entering their prediction pathways. This restriction does not exclude the noised future-frame input shared by the World and Diff Experts." (L500-509)
- **(c) 推理时**："At inference, only the current-observation and action pathways are retained. The Action Expert generates a1:H from the current-observation representation and language instruction without invoking future-state or diff prediction, introducing no additional overhead from the visual prediction experts." (L510-515)
- Figure 3 caption（原文）："Structured attention masks in DiffWAM. (a) Attention in the first N−M layers, where the World and Diff Experts share visual processing layers. (b) Attention in the last M layers, with independent state and diff prediction pathways. (c) Observation and action pathways retained at inference. Each pathway attends to observation representations and its own tokens, while cross-attention between expert pathways is blocked." (L360-366)

### 4.4 训练损失（§3.3, L516-577）

- 三损失构成："DiffWAM is trained with three losses: video flow matching [52] for future-state prediction, squared-error latent regression for difference prediction, and action flow matching [53] for action generation." (L517-520)
- **L_world（式 6, L525-538）**：对每个视觉预测步 i∈{1,…,T}，"we construct the interpolated sample along the probability path as fi,τ = (1−τ)ϵi +τ fi with τ∈[0,1]" (L528-529)，ϵ_i∼N(0,I)（L527）；"the World Expert is supervised to predict the corresponding velocity field" (L531-532)：
  - Lworld = E_{i,τ,ϵi,fi}[ ∥vθ(fi,τ , τ|f0, l)−(fi −ϵi)∥₂² ] (L533-537，txt 公式排版有乱码，按符号还原)
- **L_diff（式 7, L539-553）**：Diff Expert 与 World 接收同一 noised 潜表示 fi,τ（L540-542），直接输出差异预测 c∆f_i："We use squared-error supervision in latent space, as also used for latent prediction in DINO-WM [55]." (L543-545)；"the regression target ∆fi =f i −f i−1 is obtained by differencing consecutive ground-truth frame latents" (L545-547)：
  - Ldiff = E_{i,τ,ϵi,fi−1,fi}[ ∥c∆f_i −(f_i −f_{i−1})∥₂² ] (L548-550)
  - "The case i= 1 uses the difference between the first future-frame latent and the current-frame latent." (L551-553)
- **L_action（式 8, L554-572）**：a_{i,τ}=(1−τ)ϵ′_i+τ a_i（L556-557）；"the Action Expert is trained to recover the target velocity field"（L559-560）：
  - Laction = E_{i,τ,ϵ′i,ai}[ ∥uφ(ai,τ , τ|f0, l)−(ai −ϵ′i)∥₂² ] (L561-565)
  - "The Action Expert attends to the current-observation representations provided by the Observation Encoder, without directly receiving future-state or diff predictions. State and visual difference prediction influence action generation through joint training of the shared Observation Encoder." (L568-572)
- **总损失（式 9）**："The complete training objective is a weighted combination of the three terms: L=L world +λ diff Ldiff +λ actionLaction,(9) where λdiff and λaction are hyperparameters balancing the three losses." (L573-577)
  - λ_diff、λ_action 的具体数值：原文未写（全文仅此一处提及，Table 9 无该行）。
  - τ 的采样分布、优化器、warmup 等细节：原文未写。

### 4.5 推理时 World/Diff 专家移除机制

- 总规则："During inference, both the World and Diff Experts are removed, retaining only the Observation Encoder and Action Expert without additional inference overhead." (L413-415；同义 L175-177、L657-659)
- "Its prediction pathway is used during training and is removed at inference; the Observation Encoder is retained to provide the representation for action generation." (L454-456)
- 视觉预测分支影响留在共享表征里："Visual difference modeling influences the shared representation during training, while the Diff Expert is removed at inference." (L468-470)
- "At inference, the Action Expert operates with the Observation Encoder after both visual prediction experts have been removed." (L477-479)

### 4.6 训练配置：DiffWAM-S 与 DiffWAM-L（§3.4, L578-661）

- "We instantiate DiffWAM in two training configurations, DiffWAM-S for small-scale training and DiffWAM-L for large-scale training. Both configurations use the architecture and joint objective described above, with different parameter update schedules." (L579-582)
- **DiffWAM-S（小数据，冻结 LDB）**："For small-scale training, we freeze the independent M-layer LDB of the Diff Expert while jointly optimizing the Observation Encoder, World Expert, and Action Expert, including the visual layers shared by the World and Diff Experts. Freezing the LDB helps reduce overfitting to noise in small datasets while accelerating training by reducing the number of parameters to update." (L585-590)
  - 关键：diff 损失保持激活且梯度穿过冻结 LDB 回传："The diff loss remains active: its gradients propagate through the fixed LDB into the shared visual layers and Observation Encoder. Thus, difference prediction continues to shape the representation used for action generation even though the LDB parameters are frozen." (L635-640)
- **DiffWAM-L（大数据，两阶段）**："For large-scale training, we adopt a two-stage procedure. In the first stage, the Observation Encoder, World Expert, and Action Expert are frozen, and only the independent Diff Expert parameters are optimized using the diff regression loss. The visual layers shared with the frozen World Expert also remain frozen. In the second stage, all modules are jointly optimized with the full objective..." (L641-648)；"This schedule allows the Diff Expert to first adapt to difference prediction with a fixed visual representation, before using the larger dataset to jointly refine the shared representation and all three experts." (L649-652)
- 推理通路两者一致："Both configurations use the same action-generation pathway... The two configurations therefore differ in their training schedules while retaining the same inference architecture." (L653-661)

### 4.7 超参表（Table 9, L1424-1443，逐字转录）

| Setting | LIBERO | RoboTwin 2.0 | Real |
|---|---|---|---|
| Model configuration | DiffWAM-S | DiffWAM-L | DiffWAM-S |
| Training hardware | 8×A100 | 8×A100 | 8×A100 |
| Grad. accumulation | 2 | 2 | 2 |
| Weight decay | 1×10⁻² | 1×10⁻² | 1×10⁻² |
| LR schedule | Cosine | Cosine | Cosine |
| Learning rate | 1×10⁻⁴ | 1×10⁻⁵ | 1×10⁻⁴ |
| Batch size | 8 | 8 | 8 |
| Training steps | 1.3×10⁴ | 3×10⁴ | 6.3×10⁴ |
| Network depth N | 30 | 30 | 30 |
| Network depth M | 10 | 10 | 10 |
| Cameras | 2 | 3 | 2 |
| Video size | [224,448] | [384,320] | [224,576] |
| Image resize | [224,224] | [240,320] | [240,320] |
| Action dim. | 7 | 14 | 7 |
| Proprio dim. | 8 | 14 | 7 |
| Normalization | min/max | z-score | z-score |
| Validation split | 0.0 | 0.01 | 0.0 |

（Table 9 标题："Implementation configurations for each experimental setting." L1442-1443）

- 初始化：LIBERO 与真机 "DiffWAM-S is initialized from Wan2.2-5B"（L1452-1453、L1514-1515）；RoboTwin "Due to computational resource constraints, DiffWAM-L is initialized from FastWAM for this benchmark"（L1466-1468）。
- 数据集规模：LIBERO "40 tasks and 2,000 demonstrations in total"（L1447-1448）；RoboTwin 2.0 "approximately 27k demonstration trajectories in total, comprising 2,500 demonstrations from clean scenes and 25,000 demonstrations with strong scene randomization"（L1460-1463）；真机 "approximately 1.1k demonstration trajectories in total across Place Pens, Store Block, and Stack Cups... train for 63k optimization steps"（L709-711、L1510-1511）。

## 5) 伪代码

⚠ **依据原文重建（非原文伪代码）**：全文 1830 行经检索无 Algorithm 环境，以下依据 §3.2 正文（L406-515）与 Figure 2/3 caption（L311-315、L360-366）重建；每步附行号；未写细节一律不补。

### 5.1 训练伪代码（含注意力掩码构造）

```
# ⚠ 依据原文重建（非原文伪代码）
# 输入：当前帧潜表示 f0 (L438-439)；目标未来帧潜表示 f1:T (L526)；
#       目标差异 Δfi = fi − fi−1, i=1..T (L390；i=1 用 f1−f0, L551-553)；
#       动作 chunk a1:H (L556)；语言指令 l (L434)

1. 文本编码：l_emb = T5(l)                       # L434-435（经 cross-attention 注入, L488-489）
2. 观测编码：c = ObservationEncoder(f0)          # L438-439「provide the shared conditioning representation」
3. 噪声化（视觉、动作各自加噪）:
     fi,τ = (1−τ)εi + τ·fi ,  εi ~ N(0,I)       # L526-529
     ai,τ = (1−τ)ε'i + τ·ai,  ε'i ~ N(0,I)      # L555-557
4. 构造结构化单向注意力掩码（按层深度切换）:      # L480-515, Fig.3(L360-366)
   (a) 深度 1..N−M（World/Diff 共享视觉层）:      # L490-499
       - 当前帧 token：只 attend 自身             # L492
       - 未来帧 token：attend {当前帧, 自身}      # L495-496
       - 动作 token：attend {当前帧, 自身}        # L496-497
       - 未来帧 ↔ 动作 token：相互阻断            # L497-498
   (b) 深度 N−M+1..N（独立预测通路）:             # L500-509
       - Diff token：attend {当前帧, 自身}        # L501-502
       - 状态/动作通路：保持 (a) 的限制           # L502-503
       - Diff ↔ World/Action 交叉注意力：阻断     # L504-507（Diff 输出也不进入其通路, L463-465）
       # 注：World/Diff 共享同一 noised 未来帧输入，不被此阻断排除 (L508-509)
5. World Expert：共享 N−M 层 + 独立 M 层 State Block (L450-452)
     → 速度场 vθ(fi,τ, τ | c, l)，目标 (fi − εi)   # L452-453, L531-537 → L_world (式6)
6. Diff Expert：共享 N−M 层 + 独立 M 层 LDB (L457-460)
     → 直接回归 c∆f_i（非去噪），目标 (fi − fi−1)   # L460-467, L542-551 → L_diff (式7)
7. Action Expert：uφ(ai,τ, τ | c, l)，目标 (ai − ε'i)  # L471-472, L559-567 → L_action (式8)
     # Action Expert 不直接接收预测状态/diff 输出 (L474-475, L568-570)
8. 总损失：L = L_world + λ_diff·L_diff + λ_action·L_action   # L573-577（λ 数值原文未写）
9. 参数更新（按配置）:
   DiffWAM-S：冻结 M 层 LDB，其余联合优化；diff 梯度穿过冻结 LDB 回传共享层  # L585-590, L635-640
   DiffWAM-L：阶段1 仅优化独立 Diff Expert（其余冻结，仅 L_diff）;             # L641-646
              阶段2 全部模块联合优化完整目标                                   # L646-648
```

### 5.2 推理伪代码（专家移除）

```
# ⚠ 依据原文重建（非原文伪代码）
1. 移除 World Expert 与 Diff Expert（含其独立 M 层）   # L413-415, L454-456, L657-659
2. c = ObservationEncoder(f0)                          # L654-655
3. a1:H = ActionExpert(c, l)  （flow matching 去噪）   # L655-657, L471-472
4. 注意力掩码：仅剩 (c) 通路——当前帧 token + 动作 token   # L510-515, Fig.3(c) L363-366
   # 不调用未来状态或 diff 预测，无额外推理开销         # L511-514
```

## 6) 主结果数值表

### Table 1 — LIBERO（L692-705；caption L706-708："We use DiffWAM-S due to the small size of the LIBERO training dataset. † denotes our FastWAM baseline."）

| Method | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| OpenVLA [5] | 84.70 | 88.40 | 79.20 | 53.70 | 76.50 |
| OpenVLA-O [6] | 97.60 | 98.40 | 97.90 | 94.50 | 97.10 |
| MemoryVLA [56] | 98.40 | 98.40 | 96.40 | 93.40 | 96.65 |
| π0.5 [8] | 98.80 | 98.20 | 98.00 | 92.40 | 96.85 |
| GR00T-N1 [9] | 94.40 | 97.60 | 93.00 | 90.60 | 93.90 |
| Motus [13] | 96.80 | 99.80 | 96.60 | 97.60 | 97.70 |
| Lingbot-VA [14] | 98.50 | 99.60 | 97.20 | 98.50 | 98.45 |
| FastWAM† [15] | 98.20 | 99.80 | 97.80 | 92.40 | 97.05 |
| ImageWAM [57] | 97.20 | 99.20 | 98.80 | 98.40 | 98.40 |
| WLA [58] | 99.00 | 100.00 | 97.80 | 97.60 | 98.60 |
| **DiffWAM-S (Ours)** | **98.00** | **99.80** | **99.00** | **98.40** | **98.80** |

正文："DiffWAM-S achieves an average success rate of 98.8%, improving upon FastWAM by 1.75 percentage points. The improvement is most pronounced on LIBERO-Goal and LIBERO-Long, where DiffWAM-S reaches 99.0% and 98.4%, outperforming FastWAM by 1.2 and 6.0 points, respectively." (L757-763)

### Table 2 — RoboTwin 2.0（L736-749；caption L750-752："We use DiffWAM-L due to the large size of the RoboTwin 2.0 training dataset."）

| Method | Clean | Random | Avg. |
|---|---|---|---|
| π0 [7] | 65.92 | 58.80 | 62.36 |
| X-VLA [59] | 72.08 | 72.84 | 72.46 |
| π0.5 [8] | 82.74 | 76.76 | 79.75 |
| ABot-M0 [60] | 86.60 | 85.08 | 85.84 |
| QwenVLA [61] | 86.10 | 87.20 | 86.65 |
| Motus [13] | 88.66 | 87.02 | 87.84 |
| Lingbot-VA [14] | 92.90 | 91.50 | 92.20 |
| FastWAM† [15] | 91.88 | 90.32 | 91.10 |
| X-WAM [44] | 89.76 | 90.68 | 90.22 |
| WLA [58] | 92.94 | 90.02 | 91.48 |
| **DiffWAM-L (Ours)** | **93.28** | **92.02** | **92.65** |

正文："DiffWAM-L obtains an average success rate of 92.65% on RoboTwin 2.0, outperforming the reproduced FastWAM baseline by 1.55 percentage points. The gain is consistent across both evaluation conditions, increasing performance from 91.88% to 93.28% in clean scenes and from 90.32% to 92.02% under scene randomization." (L771-777)

### Table 3 — clean-to-random 泛化（OOD）（L784-793；caption L794-797："Success rates (%) for policies trained on clean demonstrations and evaluated in clean and randomized environments without further adaptation."）

| Method | Embodied PT | Clean | Random | Avg. |
|---|---|---|---|---|
| π0 [7] | Yes | 46.4 | 16.3 | 31.35 |
| X-VLA [59] | Yes | 68.0 | 20.9 | 44.45 |
| π0.5 [8] | Yes | 70.7 | 46.0 | 58.35 |
| StarVLA [62] | No | 46.5 | 3.2 | 24.85 |
| AHA-WAM [63] | No | 64.3 | 3.2 | 33.75 |
| FastWAM† [15] | No | 70.2 | 1.50 | 35.85 |
| **DiffWAM (Ours)** | **No** | **69.7** | **5.27** | **37.47** |

正文："DiffWAM achieves 5.27% success, compared with 1.50% for our FastWAM baseline, an absolute gain of 3.77 percentage points. Clean success decreases slightly from 70.2% to 69.7%, while the mean of Clean and Random improves from 35.85% to 37.47%." (L810-814)

### 真机结果（ALOHA-AgileX，3 任务 × 20 trials，DiffWAM-S）

- "DiffWAM-S achieves success rates of 90%, 85%, and 90% on the three tasks, respectively, compared with 80%, 70%, and 70% for FastWAM. Averaged across the three tasks, DiffWAM-S achieves 88.3% success versus 73.3% for FastWAM, an absolute improvement of 15.0 percentage points." (L839-844)
- 次数："These rates correspond to 18/20, 17/20, and 18/20 successful trials for DiffWAM-S, compared with 16/20, 14/20, and 14/20 for FastWAM. The average success rate increases from 73.3% (44/60) to 88.3% (53/60)" (L1613-1617)

### Table 4 — 两种配置对比（L828-836）

| Setting | FastWAM | DiffWAM-L | DiffWAM-S |
|---|---|---|---|
| LIBERO | 97.05 | 97.70 | **98.80** |
| RoboTwin-Clean | 88.47 | 87.72 | **93.13** |
| RoboTwin-Full | 91.88 | **93.28** | 92.60 |

正文："DiffWAM-S therefore outperforms DiffWAM-L by 1.10 and 5.41 percentage points in these two settings [LIBERO、RoboTwin-Clean]. With the full RoboTwin training set, DiffWAM-L instead achieves 93.28%, exceeding DiffWAM-S's 92.60% by 0.68 percentage points; both outperform FastWAM's 91.88%." (L859-864)

### 训练效率（LIBERO，Figure 6）

- "DiffWAM improves substantially faster during the early stage of training and reaches a 95% average success rate after only 6.9k steps, whereas FastWAM requires 9.6k steps. This corresponds to a 28.1% reduction in the number of optimization steps needed to reach the same performance level. DiffWAM also increases the convergence AUC from 73.1 to 79.8" (L946-953)

### Table 10 — 三随机种子复现（L1548-1559）

| Setting | Run 1 | Run 2 | Run 3 | Mean±Std. |
|---|---|---|---|---|
| RoboTwin 2.0 Full-data | 92.49 | 92.65 | 92.43 | 92.52±0.11 |
| LIBERO | 98.60 | 98.80 | 98.60 | 98.67±0.12 |
| RoboTwin 2.0 Clean-only | 37.33 | 37.40 | 37.47 | 37.40±0.07 |

### Table 12 — RoboTwin 2.0 任务级 Overall 行（L1828；全文见 L1776-1828）

- Overall（Clean/Random）：π0.5 82.74/76.76；Motus 88.66/87.02；LingBot-VA 92.90/91.50；GigaWorld-Policy 86.36/85.04；X-WAM 89.76/90.68；FastWAM 91.88/90.32；DiffWAM-L 93.28/92.02。

## 7) 消融（全部）

### Table 5 — 架构消融（LIBERO，L971-981）

| Architecture | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| World Expert（=FastWAM） | 98.20 | 99.80 | 97.80 | 92.40 | 97.05 |
| World Expert + Diff Loss | 97.80 | 100.00 | 97.40 | 95.80 | 97.75 |
| World Expert + Diff Expert（=DiffWAM-S） | 98.00 | 99.80 | 99.00 | 98.40 | 98.80 |

- 加 diff loss："adding the diff loss raises average success from 97.05% to 97.75%. The clearest gain is on LIBERO-Long, where success increases from 92.40% to 95.80% (+3.40 points)" (L963-966)
- 加 Diff Expert："Retaining the diff loss and introducing the separate Diff Expert further raises average success to 98.80% (+1.05 points over loss-only supervision)... Relative to FastWAM, the full architecture improves Long by 6.00 points and Goal by 1.20 points" (L992-999)
- 结论："the diff loss provides a measurable improvement, and the Diff Expert contributes a further gain; both are necessary to attain the full performance of DiffWAM in this ablation." (L1003-1006)
- 注意（loss-only 并非全赢）："loss-only supervision does not improve every suite: Spatial and Goal decrease slightly, while Object reaches 100.00%." (L990-991)

### Table 6 — LDB 层数 M（LIBERO，DiffWAM-S，L983-987）

| Method | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| FastWAM [15] | 98.20 | 99.80 | 97.80 | 92.40 | 97.05 |
| DiffWAM (M=5) | 99.60 | 100.00 | 97.20 | 96.00 | 98.20 |
| DiffWAM (M=10) | 98.00 | 99.80 | 99.00 | 98.40 | 98.80 |
| DiffWAM (M=15) | 98.20 | 99.80 | 97.40 | 97.00 | 98.10 |
| DiffWAM (M=20) | 97.00 | 100.00 | 98.40 | 98.00 | 98.35 |

正文："All tested LDB depths outperform the FastWAM baseline on average. Increasing M from 5 to 10 improves the average success rate from 98.20% to 98.80%... Further increasing the depth to 15 or 20 layers does not yield additional improvements. These results favor a moderate depth for the difference-prediction pathway; we therefore use M= 10 as the default configuration." (L1011-1023)

### Table 7 — 视觉专家配置（LIBERO，L1016-1018）

| Visual Experts | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| Only World Expert | 98.20 | 99.80 | 97.80 | 92.40 | 97.05 |
| Only Diff Expert | 99.20 | 100.00 | 95.80 | 89.80 | 96.20 |
| World + Diff Expert | 98.00 | 99.80 | 99.00 | 98.40 | 98.80 |

正文："World + Diff Expert improves the average success rate to 98.80%, outperforming the two single-expert configurations by 1.75 and 2.60 percentage points, respectively. The improvement is especially pronounced on LIBERO-Long, where the combined configuration reaches 98.40%, compared with 92.40% for Only World Expert and 89.80% for Only Diff Expert." (L1035-1042)
- 结论："the World Expert models future scene states, while the Diff Expert explicitly captures inter-frame visual differences. Jointly learning these capabilities enhances the representation used for action generation, particularly on long-horizon tasks." (L1043-1048)

### Table 8 — 控制「额外训练量」（RoboTwin 2.0 clean，L1061-1063）

| Method | Additional training | Clean success (%) |
|---|---|---|
| FastWAM (continued training) | 1 epoch | 88.47 |
| DiffWAM (Ours) | 1 epoch | 93.13 |

正文："we initialize FastWAM and DiffWAM from the same task-trained FastWAM checkpoint and train both for one additional epoch on clean demonstrations under matched training conditions... DiffWAM achieves 93.13% clean success, compared with 88.47% for continued FastWAM training, a gain of 4.66 percentage points. This comparison supports an improvement beyond additional training alone" (L1051-1058)；"The shared-checkpoint initialization isolates the effect of the modified training objective under a matched additional training budget." (L1082-1084)

### Table 11 — 预测质量（附录 B.5，L1596-1599）

| Metric | f1 | f2 | Δf1 | Δf2 |
|---|---|---|---|---|
| PSNR⇑ | 37.94 | 30.79 | - | - |
| SSIM⇑ | 0.9851 | 0.9701 | - | - |
| Latent Error⇓ | - | - | 0.0055 | 0.0136 |

正文："The predicted future observations preserve scene structure and reflect the progression of the manipulation trajectory. The visualized diff outputs highlight localized variation around the robot and manipulated objects... These differences describe inter-frame visual variation and do not exclusively isolate changes caused by robot actions." (L1630-1637)

## 8) 局限（逐字，L1717-1737）

> "Several limitations remain. First, the independent M-(L1717)
> layer Diff Expert pathway introduces additional computation (L1718)
> and memory during training. Freezing the LDB in DiffWAM-(L1719)
> S does not eliminate this cost, since diff predictions are still (L1720)
> computed and their gradients propagate into the shared rep-(L1721)
> resentation. Both the World and Diff Experts are removed at (L1722)
> inference, so these visual prediction pathways add no online (L1723)
> action-generation overhead. Second, the difference target (L1724)
> captures inter-frame latent variation without explicitly dis-(L1725)
> tinguishing changes caused by robot actions from camera (L1726)
> motion, illumination changes, or movement of unrelated ob-(L1727)
> jects. This ambiguity may be more significant in dynamic, (L1728)
> unstructured environments. Third, the clean-to-random result (L1729)
> establishes a benefit under the evaluated RoboTwin pertur-(L1730)
> bations, while absolute Random success remains low and (L1731)
> generalization to other shifts is not established. The experi-(L1732)
> ments cover a limited range of data scales and embodiments, (L1733)
> and DiffWAM is not pretrained on a large-scale robot dataset. (L1734)
> Broader evaluation is needed to establish how the benefits of (L1735)
> DiffWAM-S and DiffWAM-L generalize across training-data (L1736)
> regimes and deployment settings." (L1737)

另，真机部分亦有：**"Broader evaluation across objects, environments, and embodiments remains necessary to establish generality."** (L1620-1622)

未来工作方向（C.2）：更高效的 Diff Expert 设计、action-conditioned/motion-aware/object-centric 约束解耦致变来源、大规模机器人交互数据预训练（L1739-1755）。

## 9) 叙事关联【推断】

1. 【推断】DiffWAM 是对 FastWAM 的直接复现与改进（表中 "† denotes our FastWAM baseline"，L702/746/792；RoboTwin 的 DiffWAM-L 直接从 FastWAM checkpoint 初始化，L1466-1468），并显式继承了其 "first-frame-conditioned attention scheme [15]"（L491）——属于同一 WAM 研究线上「预测什么信号」的下一步。
2. 【推断】论文的叙事定位是把 WAM 的核心问题从「要不要预测未来」推进到「预测什么信号才有利于动作」：论据为 "is future-state prediction alone sufficient to learn world representations that capture task-related changes?"（L136-138）与原理句 L147-149；且其声称的优势恰在于不用外部先验（对照 WAM-s 的 "targets obtained from external estimators may contain estimation errors"，L273-275）。
3. 【推断】真机增益（+15.0，L843-844）远大于仿真增益（+1.75/+1.55，L759-760/772-774），且 FastWAM 真机仅 73.3%——暗示仿真基准接近饱和，差异信号在真实场景（更多任务无关内容与噪声）下价值更大，与 L96-99 的动机自洽。
4. 【推断】S/L 双配置形成「按数据规模选择训练策略」的设计模式（小数据冻结 LDB 防过拟合 L585-590；大数据两阶段 L641-652；Table 4 结论 "choosing DiffWAM-S when training data are limited and DiffWAM-L when more data are available"，L940-942），而其局限中承认未做大规模预训练（L1733-1734），为后续叙事留下「预训练 + diff prediction」的开放命题。

---

### 附：关键原文引用索引（速查）

- 差异定义："We define consecutive visual differences as differences between the latent representations of consecutive observations, providing an explicit learning signal for temporal variation." (L103-105)；式 (5) L390
- 掩码："A structured unidirectional attention mask conditions the Diff, World, and Action Experts solely on the current-frame representation while blocking attention between experts." (L110-112)
- 推理移除："At inference, both the World and Diff Experts are removed, retaining only the Observation Encoder and Action Expert without additional inference overhead." (L175-177 / L413-415)
- 总损失："L=L world +λ diff Ldiff +λ actionLaction,(9)"（L575；λ 数值原文未写）
- 主结果："Specifically, it attains average success rates of 98.8%, 92.65%, and 88.3%, outperforming the state-of-the-art baseline (i.e., FastWAM) by 1.75, 1.55, and 15.0 percentage points, respectively." (L118-121)
- 消融结论："the diff loss provides a measurable improvement, and the Diff Expert contributes a further gain; both are necessary to attain the full performance of DiffWAM in this ablation." (L1003-1006)
