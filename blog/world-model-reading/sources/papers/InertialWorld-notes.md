# InertialWorld 精读笔记

> 数据源：`C:\Users\Admin\Desktop\design\auto_docs\papers\txt\InertialWorld.txt`（1546 行，arXiv:2611.22314v1）
> 铁律执行：所有事实性陈述附英文原文逐字引用 + 行号；论文未写处标「原文未写」；解读标【推断】。
> 说明：本文**没有** `\begin{algorithm}` 环境，第 5 节伪代码为「⚠ 依据原文重建（非原文伪代码）」，每步附行号。

---

## 1) 原文信息

- 标题（逐字，L2）：`INERTIALVIDEOWORLDMODELS`（正文排版为 INERTIALWORLD，e.g. L14 "we introduce INERTIALWORLD, an Inertial Video World Model"）
- 作者（L3-4）：`Yuan Gao1∗, Fan Xu2∗, Yu Guo3, Yuxu Lu4, Zhengru Fang5, Hao Wu1†, Xiaomeng Huang1†`；L49：`∗Equal contribution. †Corresponding author.`
- 机构（L5-7）：`1Tsinghua University, 2University of Science and Technology of China, 3City University of Hong Kong, 4Hong Kong Polytechnic University, 5The Hong Kong University of Science and Technology`
- ID（L51）：`arXiv:2611.22314v1  [cs.CV]  17 Nov 2026`
- 发表状态（L1）：`Preprint`——具体录用会议原文未写。
- 代码（L27-28）：`Codes link: https://github. com/YuanGao-YG/InertialWorld.`
- 资助（L612）：`This work was supported by the National Natural Science Foundation of China (42125503, 42430602).`
- 页数：正文 1–10 页（L49-50 "1"，L597 "10"），参考文献 11–14 页（L641 "11"、L774 "14"），附录 15–29 页（L798 "15"、L1546 "29"）。

## 2) Abstract 全文逐字（L9-28）

> The physical world has inertia: it evolves from what already exists rather than being recreated at every instant. However, many existing video world models lack an explicit mechanism for inertial state evolution, leaving the continuity of motion and structure implicit in video prediction. During autoregressive rollouts, repeated reconstruction can accumulate errors and destabilize the predicted dynamics. To address this limitation, we introduce INERTIALWORLD, an Inertial Video World Model that makes inertia a structural prior for autoregressive prediction: existing scene evidence is carried across time rather than reconstructed at every step. Central to this formulation is Inertial Memory, which turns memory from passive predictive context into an evolving video state with two complementary views: a dense Eulerian grid preserves scene-aligned evidence, while persistent Lagrangian carriers bind local content to learned canonical coordinates. At each step, the model transports the carrier content according to action-conditioned motion, predicts the complete next video state from the transported carriers and dense grid, and uses a learned gate to combine transported and newly predicted content into the memory for the next step. Extensive experiments across multiple domains, including robot manipulation, 3D game simulation, and open-world navigation, demonstrate that INERTIALWORLD achieves competitive autoregressive prediction performance against strong video world model baselines. Codes link: https://github. com/YuanGao-YG/InertialWorld.

（与 L103-112 的贡献三点自述一致：L117-124 `• We formulate inertial video world modeling…• We introduce Inertial Memory…• We conduct extensive autoregressive prediction evaluations…`）

## 3) 问题与动机（简）

核心矛盾：物理演化是惯性的，但主流视频世界模型把它当作「每步重新条件生成」。

- L37-40：`Despite these advances, physical evolution is inertial: the world evolves from what already exists rather than being recreated at every instant. Most video world models, however, still formulate temporal evolution as repeated conditional generation, predicting persistent scene content again at every step rather than explicitly carrying it forward.`
- 累积误差机制（L40-42）：`Since each generated state conditions the next transition, small reconstruction errors can repeatedly re-enter the rollout and destabilize predicted motion and structure (Wang et al., 2017; Babaeizadeh et al., 2021).`
- 与既有 memory 工作的分野（L92-98）：`retention alone does not specify how the retained content should evolve. When memory serves as context for a generic transition, the transition remains free to synthesize the next state without explicitly advancing the stored content.`；warping 亦不够（L95-97）：`Transformation-based predictors demonstrate the value of warping observations or features (Jaderberg et al., 2015; Finn et al., 2016), yet warping does not by itself make transport the recurrent evolution rule of persistent memory across model-predicted steps.`
- 中心问题（L98-101）：`memory determines which past evidence remains available, whereas inertia additionally specifies what persists and where it moves. This distinction motivates the central question of this work: How can a video world model evolve the world it already remembers, rather than repeatedly reconstructing its persistent content?`
- 定位（L140-142）：`Rather than changing the video generator or its training objective, our work changes how the recurrent video state evolves by introducing transport-structured persistence that is compatible with different generative backbones.`

## 4) 实现细节（重点）

### 4.0 因果接口（§3.1，L163-178）

- 冻结 codec，rollout 级 Context code + 帧级 Dynamic code（L165-167）：`We use a frozen codec that represents the observed scene with a rollout-level Context code C and each predicted frame with a Dynamic code D. We encode C= Enc C(ot) and initialize the recurrent video state as St = Init(C).`
- 转移方程 Eq. 1（L169）：`( bDτ+1 ,S τ+1 ) =F θ(Sτ ,a τ ;C), boτ+1 = Dec(C, bDτ+1 ).`
- 关键语义（L170-174）：`The Context code remains fixed and anchors the observed scene throughout the rollout. The predicted Dynamic code is a complete code for the next frame, not a residual correction to a ground-truth frame.`；`Future frames define training targets D⋆τ+1 , but neither those targets nor their encoded features are fed back into the recurrence.`
- 原则（L176-178）：`We instead make persistence structural: predicted motion first acts on stored content, and prediction updates the state only after this content has been carried forward.`

### 4.1 Inertial Memory 双视图（§3.2 + App. A）

状态定义（L181 + L224）：`Its state is Sτ = (G τ ,{r τ,k }Kk=1,B τ ), where Gτ is a dense Eulerian grid and rτ,k = (A τ,k ,α τ,k ,ξ τ,k ) is a persistent Lagrangian carrier.`

- 稠密欧拉网格 = 场景对齐证据（L224-225）：`The grid remains aligned with scene coordinates and preserves evidence across the full spatial domain.`
- 持久拉格朗日载体 = 局部内容绑定规范坐标（L225-227）：`Each carrier instead moves with a local region: Aτ,k stores canonical content, ατ,k describes its support, and ξτ,k specifies where that content is placed. The soft binding mapBτ assigns evidence between the carriers and a dense fallback.`
- 无监督（L228-229）：`The carriers are learned regions rather than supervised semantic objects, so this representation does not require object annotations or a complete object decomposition.`
- 具体规模（App. A.1，L814-817）：`Its Eulerian component Gτ is a dense 16×20 grid aligned with scene coordinates. Its Lagrangian component contains K= 16 persistent regional carriers. Carrier k maintains a descriptor, a canonical feature map and support, a translation–rotation–scale coordinate, persistent velocity, visibility and existence estimates, and a bounded local deformation.`
- 完整状态 Eq. 5（L812）：`Sτ = (C,I τ ,H τ ,Q τ ),I τ = (Gτ ,{R τ,k }Kk=1,B τ ).` 其中 Hτ/Qτ 是 backbone 继承的辅助上下文，非惯性状态本体（L820-823）：`The implementation additionally keeps the four most recentpredictedDynamic states Hτ and 64 generic interaction slots Qτ . They provide predictor context inherited from the video backbone; the proposed persistent video state and its carry operation are defined byIτ . This distinction is important: the method is not equivalent to retaining a longer image history as passive attention context.`
- 初始化（App. A.2，L826-828）：`Carrier coordinates begin on a normalized 4×4 lattice with zero velocity; their canonical content is sampled from the initialized dense state. This constructs S0 = Init(C) from the single observed frame, without an object detector or calibrated geometry.`
- 绑定渲染（L829-832）：`a carrier is rendered into scene coordinates using its canonical content, support, coordinate, rigidity, and deformation. Its binding score at an Eulerian cell combines content compatibility, rendered support, visibility, and existence. A softmax over the K carriers and the background branch gives Bτ .`

### 4.2 动作条件运动传输 transport（Carry）

Eq. 2（L230-232）：`The first part of a transition predicts their action-conditioned motion and applies it to the stored state: ξτ+1 =U θ(Sτ ,a τ ), eSτ+1 = Carry(Sτ ;ξ τ →ξ τ+1 ).`

- 重渲染语义（L233-235）：`Carry re-renders each carrier's canonical content and support at its new coordinates, so local evidence moves with the carrier rather than being synthesized again at the destination.`
- 稠密网格侧（L235-237）：`For the dense grid, the binding map combines carrier-induced motion with a learned fallback flow; normalized forward splatting propagates evidence while accounting for ownership and visibility.`
- 恒等性质（L238-239）：`When the predicted motion is zero, this operation reduces to an identity carry. The coordinates are learned representational variables, not calibrated physical parameters.`
- 坐标动力学头（App. A.3，L835-837）：`The current action is combined with every carrier state, and a learned contact graph propagates action-dependent interactions between nearby carriers. The dynamics head predicts bounded changes to carrier velocity, coordinate, rigidity, existence, visibility, and deformation.`
- 速度保持 Eq. 6（L839-843）：`vτ+1,k =ηv τ,k + ∆vτ,k ,ξ τ+1,k = Π(ξτ,k +v τ+1,k ),(6) Here η= 0.90 is the fixed velocity-retention coefficient, ∆vτ,k is the bounded action- and contact-conditioned learned increment, and ξτ,k concatenates 2-D translation, rotation, and log scale. In Eq. 6, Π wraps rotation and projects translation and log scale to their valid ranges. Carrier motion is therefore accumulated through a persistent state rather than independently re-estimated from scratch.`
- 流混合 Eq. 7（L844-850）：`the binding-weighted carrier flow f Lτ,i is combined with dense fallback flow f Eτ,i according to foreground binding massm τ,i = Pk Bτ,ik…: fτ,i = (1−m τ,i)f Eτ,i +m τ,if Lτ,i.(7) Eq. 7 defines mτ,i ∈[0,1] as a Lagrangian mixture share, not a motion magnitude.`
- 所有权/可见性（L851-854）：`The resulting correspondence carries owned source features with normalized bilinear forward splatting. Source ownership and destination coverage are kept separate: moved content is vacated at its source, while a learned reveal proposal fills newly uncovered locations. This prevents a feature from being duplicated at both its old and new positions. Zero predicted motion reduces to identity carry.`

### 4.3 从传输载体 + 稠密网格预测完整下一视频状态 + 学习门控

- 动机（L240-241）：`Transport alone cannot explain newly visible content, deformation, or other changes not represented by the current memory.`
- Eq. 3（L242-245）：`A predictor reads that state and jointly produces the complete next Dynamic code, a candidate memory update, and a learned write gate… ( bD+,M new,g) =P θ( eS,a;C),M + = (1−g)⊙ fM+g⊙ Mnew.(3)`
- 三个输出的角色（L246-248）：`The first output is decoded into the complete next frame, allowing the model to generate changes that cannot be obtained by transport alone. The second output proposes new writable memory content, while the gate determines which carried values should persist and which should be replaced.`
- 默认持久（L249-251）：`Thus, persistence is the default path, but the memory can still adapt when the newly predicted state provides evidence for change.`
- 纯预测闭环（L251-253）：`Because the proposal is generated from the model prediction rather than the target Dynamic code, repeated application of Eqs. 2–3 remains strictly predicted-only.`
- 操作性定义（L252-254）：`This ordered transition—carry, predict, then update—is our operational definition of representational inertia; it does not impose a hard-coded Newtonian law.`
- 读头三来源路由（App. A.4，L857-863）：`The predictor reads the fixed Context code, carried dense state, rendered carriers, predicted-state history, interaction slots, and current action. It predicts all four lanes of bDτ+1 . For each lane and cell, the readout assigns probability to three sources: newly generated content, stationary content from the previous predicted state, and content transported from a previous predicted location. Invalid persistence routes are masked. At evaluation time, the hard route is sampled deterministically from the learned posterior using an immutable key derived from the Context code, recurrent step, lane, and cell. No route has access to a future target.`
- 完整 Eq. 8 与门语义（L866-871）：`( bDτ+1 ,M newτ+1 ,g τ+1 ) =P θ( eSτ+1 ,a τ ;C),M τ+1 = (1−g τ+1 )⊙ fMτ+1 +g τ+1 ⊙ Mnewτ+1 . (8) … gτ+1 ∈[0,1] is the element-wise write gate, and ⊙ denotes element-wise multiplication. A gate value of zero retains the carried state and a value of one replaces it with the candidate.`
- 状态回写顺序（L872-874）：`Carrier descriptors are updated from their assigned dense evidence, and canonical content is updated by sampling the new dense state back into each carrier's coordinate system. The binding map, interaction slots, and predicted-state history are updated last.`
- 固定顺序（L875-876）：`One transition therefore follows the fixed order advance carriers → carry existing content → predict the complete next video state → commit memory. This is the operational meaning of carry-before-write.`
- 门控初始化（App. B，Eq. 12，L1102-1104）：`We zero-initialize the scene-memory gate weights. For a non-vacated location at unit confidence, its keep-logit bias and keep floor give κ0 =σ(2.2) = 0.900,eκ 0 = 0.8 + 0.2κ0 = 0.980, g 0 = 1−eκ0 = 0.020,(12)`；载体内描述子门（L1107-1108）：`The carrier-descriptor keep gate also has zero weights and bias 3.0, giving σ(3.0) = 0.953.`

### 4.4 学习与 rollout（§3.3）

- 训练=推理同构（L258-262）：`We train on the same predicted states used at inference and backpropagate through the complete rollout. Discrete internal decisions remain hard on the forward path and use a straight-through estimator for optimization. The recurrence is never refreshed with a target state or detached between autoregressive steps. We progressively increase the prediction horizon from 8 to 16 and then 32 steps so that the model learns to preserve and update its own state under repeated application.`
- 目标 Eq. 4（L263-264）：`The training objective combines prediction, memory-consistency, and regularization terms: L=L pred +λ memLmem +λ regLreg.(4)`
- 系数（L270-271）：`We use λmem =λ reg = 1 in all experiments; each grouped loss is a fixed weighted sum of its constituent terms.`（完整 loss 清单与系数见原文 Table 5，L951-1034）
- 真值不注入状态（L273-274）：`Ground truth is used only to compute these losses; it never replaces Sτ in the rollout. Training therefore matches the causal, predicted-only recurrence used at inference.`

### 4.5 架构 / 超参（App. B）

- 冻结 tokenizer（L883-888）：`We train a separate tokenizer for each domain while keeping its learnable architecture, objective, and optimization recipe identical.` `Each tokenizer contains 120.386M parameters, uses 384 hidden and 256 latent channels, and maps a 256×320 internal canvas…` `Every token is represented by five finite-scalar-quantization digits with levels (7,5,5,5,5) , giving 4,375 joint states (Mentzer et al., 2024).`
- 代码接口（Table 4，L901-904）：`Context code: Two 32×40 lanes; 2,560 tokens; fixed within a rollout. / Dynamic code: Four 8×10 lanes; 320 tokens per predicted frame. / Token quantization: Five FSQ digits with levels (7,5,5,5,5).`
- 预测器 backbone（Table 4，L906-909）：`Width 1,280; 20 attention heads; 15 transition layers; three appearance-refinement layers; two Context blocks; one interaction layer; MLP ratio 4; zero dropout.`（循环特征 L910：`320 dense scene tokens, 16 carriers of width 256, 64 interaction slots, and four recent predicted Dynamic states.`）
- tokenizer 训练（L911-913）：`All tokenizers are trained from scratch for 60k optimizer steps with AdamW, learning rate 10−4, weight decay 0.02, and (β1, β2) = (0.9,0.95) .` `Training uses BF16, global batch size 32, and gradient-norm clipping at 1.0.`
- 世界模型规模（Table 6，L1117-1121）：RT-1 `818.746M, 60k`；CSGO `818.795M, 30k`；RECON `818.733M, 30k`。动作输入（L916-919）：`RT-1 and CSGO consume their raw 13-D and 51-D actions. RECON consumes normalized local-frame displacement in x and y together with wrapped relative yaw.`
- WM 优化（L1136-1140）：`All video world models use AdamW with learning rate 10−4, minimum learning rate 10−5, weight decay 0.05, and (β1, β2) = (0.9,0.95) .` `The predicted-only rollout curriculum is shared: AR8 for steps [0,4k) , AR16 for [4k,12k) , and AR32 thereafter. We backpropagate through the entire active rollout without teacher forcing or truncated BPTT.`
- 分布式（L1141-1143）：`Formal training uses seed 0, BF16, PyTorch distributed data parallel over 32 GPUs, global batch size 32, no gradient accumulation, activation checkpointing, and gradient clipping at 1.0.`
- 检查点选择（L1144-1145）：`The main results use the final raw-model checkpoint—60k for RT-1 and 30k for CSGO and RECON—rather than choosing a checkpoint on the test set. No exponential moving average is used.`
- 内存相关 loss 的 staged 引入（L926-928）：`Transport, stationary routing, contact, and local deformation are staged in during the first 12k steps so that an untrained persistence path cannot dominate the initial predictor.`

## 5) 伪代码

⚠ **依据原文重建（非原文伪代码）**——原文无 Algorithm 环境，仅编号公式 Eq. 1/2/3/6/7/8（L169、L232、L245、L839、L849、L866-867）与固定顺序句（L875-876）；每步附行号，未发明任何训练细节。

```
Input : 观察 o_t；冻结 codec 编码 Context code C = Enc_C(o_t)          [L167-168]
Init  : S_0 = Init(C)   // 载体内坐标 4×4 格点、零速度；规范内容采样自稠密状态  [L826-828]
Output: 预测帧序列 ô_{τ+1}, ...

for τ = 0, 1, 2, ... (rollout 内因果循环) do
  1. ADVANCE CARRIERS — 动作条件坐标更新:                            [L875, L230-232]
        ξ_{τ+1} = U_θ(S_τ, a_τ)                                       [Eq.2, L232]
        其中 v_{τ+1,k} = η·v_{τ,k} + Δv_{τ,k}, ξ_{τ+1,k} = Π(ξ_{τ,k}+v_{τ+1,k}), η=0.90  [Eq.6, L839-840]
  2. CARRY EXISTING CONTENT — 把已存内容搬到新坐标:                  [L875, L230-232]
        S̃_{τ+1} = Carry(S_τ; ξ_τ → ξ_{τ+1})                           [Eq.2, L232]
        每格流: f_{τ,i} = (1−m_{τ,i})·f^E_{τ,i} + m_{τ,i}·f^L_{τ,i}, m_{τ,i}=Σ_k B_{τ,ik}  [Eq.7, L849]
        归一化前向 splatting；源格内容腾空(所有权/可见性分开)，新暴露格用 learned reveal proposal  [L851-853]
        零运动 ⇒ 恒等 carry                                              [L238, L854]
  3. PREDICT COMPLETE NEXT STATE — 从 carried 状态预测完整下一帧码 + 候选记忆 + 写门:  [L875-876]
        (D̂_{τ+1}, M_new_{τ+1}, g_{τ+1}) = P_θ(S̃_{τ+1}, a_τ; C)          [Eq.8, L866-867]
        读出按三来源路由: innovation / stationary / transported(非法路径掩码)   [L858-861]
        D̂_{τ+1} 为完整下一帧码(非残差)                                  [L170-171]
  4. GATED COMMIT — 学习门控合并 carried 与新预测内容:                [L875-876]
        M_{τ+1} = (1 − g_{τ+1}) ⊙ M̃_{τ+1} + g_{τ+1} ⊙ M_new_{τ+1}      [Eq.8, L867]
        g=0 保留 carried 状态, g=1 以候选替换                            [L869-871]
  5. 解码输出:  ô_{τ+1} = Dec(C, D̂_{τ+1})                              [Eq.1, L169]
  6. 收尾更新: 载体描述子 ← 所属稠密证据; 规范内容 ← 新稠密状态按载体坐标采样;
     绑定图、交互 slot、预测历史最后更新                                  [L872-874]
  S_{τ+1} 组合: 传输后载体坐标 + 门控更新后内容                          [L248-249]
end
```

注：第 3 步读出（三来源路由）与门控是同一 predictor 输出；训练中硬路由走 straight-through、整个 rollout 反向传播、不 teacher-forcing、不 detach（L258-262、L877-880）。

## 6) 主结果数值表（三域）

评估协议共性（L283-285）：`Every method is evaluated on the same 1,024 episodes and predicts 32 future frames using its native observation interface.`；`All outputs are compared with the same 256×320 ground-truth frames.`（CSGO 256×320 70 集 L292-296；RECON 240×320 128 集 L300-303）

**Table 1（L309-329，逐字数值）**

| 域 | Model | PSNR↑ | SSIM↑ | LPIPS↓ | MSE/MAE↓ |
|---|---|---|---|---|---|
| RT-1 (MSE) | RLVR-World | 21.95±2.36 | 0.780±0.062 | 0.168±0.050 | 0.0086±0.0053 |
| | Vid2World | 21.32±2.35 | 0.724±0.072 | 0.244±0.050 | 0.0101±0.0060 |
| | WorldGym | 21.72±2.45 | 0.782±0.063 | 0.205±0.050 | 0.0094±0.0060 |
| | DriftWorld | 20.83±2.44 | 0.731±0.072 | 0.266±0.061 | 0.0116±0.0064 |
| | **INERTIALWORLD** | **24.37±2.60** | **0.839±0.054** | **0.147±0.049** | **0.0051±0.0036** |
| CSGO (MAE) | DIAMOND | 17.04±3.20 | 0.513±0.108 | 0.576±0.080 | 0.1187±0.0525 |
| | Vid2World | 16.69±3.14 | 0.487±0.108 | 0.518±0.084 | 0.1234±0.0509 |
| | **INERTIALWORLD** | **18.36±3.02** | **0.565±0.097** | **0.516±0.086** | **0.0934±0.0358** |
| RECON (MAE) | NWM | 13.03±1.46 | 0.410±0.082 | 0.507±0.051 | 0.1623±0.0359 |
| | RAE-NWM | 14.66±1.56 | 0.424±0.080 | 0.458±0.046 | 0.1285±0.0280 |
| | Vid2World | 15.59±1.74 | 0.488±0.080 | 0.416±0.054 | 0.1099±0.0262 |
| | **INERTIALWORLD** | **16.70±1.67** | **0.546±0.076** | **0.401±0.050** | **0.0895±0.0221** |

- 汇总声明（L394-398）：`Across the three domains, all 36 baseline–metric point estimates favor INERTIALWORLD. Episode-paired 95% bootstrap intervals and paired sign-flip tests with joint Holm correction support 35 of these 36 comparisons, including 11 of the 12 strongest-baseline margins; the sole unresolved case is the small CSGO LPIPS difference against Vid2World (Appendix D.3).`（Table 9，L1314：CSGO LPIPS vs Vid2World `0.0028†[−0.0065,0.0119]`）
- 逐帧稳定性（L365-369）：`INERTIALWORLDachieves the highest PSNR and SSIM and the lowest LPIPS and MSE at every predicted frame. Its advantage is retained as the prediction horizon increases, showing that the main-table gain reflects consistently more stable video evolution rather than an improvement confined to the first few transitions.`
- 域预算说明（L398-401）：`RT-1 is our primary domain and is trained for 60k steps; because training separate domain models is costly, the CSGO and RECON models use a fixed 30k-step budget and serve as secondary tests of cross-domain applicability rather than exhaustively tuned domain-specific systems.`

**运动保真（Table 2，L459-473，pHolm=1.2×10⁻⁴）**：`All 12 intervals exclude zero and survive Holm-corrected paired sign-flip tests (pHolm = 1.2×10−4).` Flow EPE margin（基线−ours，↓更好）：RLVR-World 0.076、Vid2World 0.296、WorldGym 0.515、DriftWorld 0.332；Flow Cosine margin（ours−基线）：0.118 / 0.045 / 0.203 / 0.135；Dyn. LPIPS margin：0.071 / 0.104 / 0.112 / 0.169。结论（L474-475）：`INERTIALWORLDranks first on all three diagnostics, achieving the lowest Flow EPE and dynamic-region LPIPS together with the highest Flow Cosine.`

**Real2Sim 策略评估（L532-537）**：`INERTIALWORLDhas the smallest gap for every policy and a 15.1-percentage-point overall error, compared with 20.6 for WorldGym, 29.0 for Vid2World, 31.7 for RLVR-World, and 42.1 for DriftWorld. Its pooled success rises from 13.9% for Begin to 55.6%, 58.3%, and 69.4% for the stronger policies, preserving their competence ordering rather than uniformly inflating success.`（真实参照 SIMPLER：0.0%/63.0%/79.7%/87.1%，L1234-1235）

**超训练视界 H64（Table 10，L1349-1354）**：INERTIALWORLD `22.58±2.80 / 0.795±0.095 / 0.188±0.076 / 0.0085±0.0071`；`Relative to the strongest baseline in each column, it improves PSNR by 2.36 dB and SSIM by 0.060, while reducing LPIPS by 0.0243 and MSE by 0.00534.`（L1336-1338）

## 7) 消融（全部）

协议（L560-565）：`We train four 0.8B-scale models from scratch for 30k updates under a matched protocol. The Full model combines dense Eulerian state with persistent Lagrangian carriers; Dense-only removes the carriers, canonical content, and transport while retaining the Eulerian state; Recurrent Memory replaces this structured state with a conventional GRU; and Plain AR uses no recurrent memory.`

**Table 3（L540-559，逐字；lower is better）**

| 设定 | Model | Long-lived content↓ | Post-reappearance content↓ | Late-track content↓ |
|---|---|---|---|---|
| RT-1 seed 0 | Plain AR | 0.5144±0.1275 | 0.5144±0.1538 | 0.1009±0.0558 |
| | Recurrent Memory | 0.4414±0.1289 | 0.4375±0.1525 | 0.0935±0.0565 |
| | Dense-only | 0.4247±0.1334 | 0.4183±0.1637 | 0.0896±0.0573 |
| | **Full** | **0.4145±0.1328** | **0.4036±0.1620** | **0.0876±0.0552** |
| RT-1 seed 1 | Plain AR | 0.5172±0.1272 | 0.5073±0.1569 | 0.1051±0.0579 |
| | **Full** | **0.4153±0.1354** | **0.4053±0.1644** | **0.0888±0.0561** |
| CSGO seed 0 | Plain AR | 0.6076±0.0761 | 0.5961±0.0915 | 0.4864±0.1211 |
| | **Full** | **0.5679±0.0917** | **0.5624±0.1076** | **0.4485±0.1200** |

- 排序结论（L573-574）：`On all three measures in Table 3, the Full model ranks first, followed by Dense-only, Recurrent Memory, and Plain AR.`
- 显著性（L574-576）：`The Full model also improves over Dense-only in every case; all three pointwise paired 95% percentile-bootstrap intervals exclude zero and are reported in Appendix Table 13.`（Table 13，L1516-1518：Plain AR 0.0999[0.0932,0.1067] / Recurrent 0.0269[0.0218,0.0321] / Dense-only 0.0102[0.0054,0.0149] 等，全部区间>0）
- 消融参数量（L1473-1474）：`Their active parameter counts are 824.450M (Plain AR), 819.397M (Recurrent Memory), 808.397M (Dense-only), and 818.746M (Full).`
- 载体增量贡献（L1538-1540）：`the carrier-equipped Full model reduces long-lived, post-reappearance, and late-track errors on the paired evaluation.`
- 机制诊断（Table 11，L1401-1417，H64 上 35 集）：载体同槽检索 100.0% 全程；同槽余弦 0.924→0.682（t=64）；规范内容同槽余弦 0.993→0.977；平均拉格朗日流份额 m：85.0%→76.3%→69.4%；场景写门均值 2.64%→1.76%（q0.90 仅 4.34%）。解释（L1418-1419）：`This separation between an evolving carrier descriptor and nearly invariant carried content is the behavior targeted by the persistent Lagrangian representation.`

## 8) 局限（逐字）

原文**没有**独立的 Limitations 小节（正文止于 CONCLUSION）。最接近的局限/未来工作句（L594-596，逐字）：

> `Extending the current deterministic, learned-coordinate formulation to multimodal futures and explicit three-dimensional scene representations remains an important direction.`

另有两条实验口径自白：
- 域训练预算（L398-401）：`RT-1 is our primary domain and is trained for 60k steps; because training separate domain models is costly, the CSGO and RECON models use a fixed 30k-step budget and serve as secondary tests of cross-domain applicability rather than exhaustively tuned domain-specific systems.`
- 冻结协议非严格 held-out（L288-292）：`because released methods use different native RT-1 partitions, it is not necessarily held out for every baseline. Under the released RLVR-World partition, 1,015 of the 1,024 evaluation episodes belong to its training subset. We therefore treat this as a descriptive released-checkpoint comparison, rather than a controlled estimate of held-out generalization.`

## 9) 叙事关联【推断】（≤4 条）

1. **「世界从已有状态演化而非每刻重建」→ 世界模型设计启示**：本文把自回归世界模型的正确默认路径定为「先搬运、后写入」（carry-before-write），生成器只在「传输解释不了的变化」处介入（L240-241 `Transport alone cannot explain newly visible content, deformation, or other changes…`）。【推断】这给出一个可迁移的设计判据：世界模型的状态转移应先具备「内容随运动搬迁」的守恒机制，再叠加局部生成。
2. **惯性 = 结构先验，不是物理定律**：坐标是 `learned representational variables, not calibrated physical parameters`（L239），且 `it does not impose a hard-coded Newtonian law`（L253-254）。【推断】作者刻意把「惯性」做成表征层的守恒结构（规范坐标+传输），与显式物理仿真路线划清界限——它不要求任何可辨识的物理量，却获得了运动连续性收益。
3. **门控=选择性写入**：`The small but nonzero gates permit localized innovations while retaining transported evidence as the default state-update path.`（L1428-1429）。【推断】g≈0.02 的默认小门（Eq. 12，L1102-1104）与「静态内容不重生成、交互区局部激活」的语义（L429-433）一致，可作为「状态连续性叙事」的量化佐证：默认保留 98%，仅在确有新证据处写入。
4. **与 EPM / Physical Registry 等显式状态世界模型的并列**：原文未写（Related Work 只讨论了 recurrent/spatial memory、object-centric、warping 谱系，未出现 EPM 或 Physical Registry 字样）。【推断】仅按本文自身定位，Inertial Memory 的「欧拉证据场 + 拉格朗日载体」可视为显式状态潮流中一种**坐标级（非语义对象级）**实例：持久状态是连续特征场的搬迁+门控修订，而非物体槽或物理寄存器。

---

## 附：关键公式索引

| 公式 | 内容 | 行号 |
|---|---|---|
| Eq. 1 | `( bDτ+1 ,S τ+1 ) =F θ(Sτ ,a τ ;C), boτ+1 = Dec(C, bDτ+1 )` | L169 |
| Eq. 2 | `ξτ+1 =U θ(Sτ ,a τ ), eSτ+1 = Carry(Sτ ;ξ τ →ξ τ+1 )` | L232 |
| Eq. 3 | `( bD+,M new,g) =P θ( eS,a;C),M + = (1−g)⊙ fM+g⊙ Mnew` | L245 |
| Eq. 4 | `L=L pred +λ memLmem +λ regLreg`（λmem=λreg=1） | L264, L270 |
| Eq. 5 | `Sτ = (C,I τ ,H τ ,Q τ ),I τ = (Gτ ,{R τ,k }Kk=1,B τ )` | L812 |
| Eq. 6 | `vτ+1,k =ηv τ,k + ∆vτ,k ,ξ τ+1,k = Π(ξτ,k +v τ+1,k )`（η=0.90） | L839-840 |
| Eq. 7 | `fτ,i = (1−m τ,i)f Eτ,i +m τ,if Lτ,i` | L849 |
| Eq. 8 | `( bDτ+1 ,M newτ+1 ,g τ+1 ) =P θ( eSτ+1 ,a τ ;C),M τ+1 = (1−g τ+1 )⊙ fMτ+1 +g τ+1 ⊙ Mnewτ+1` | L866-867 |
| Eq. 12 | 门初始 `κ0 =σ(2.2) = 0.900, eκ0 = 0.980, g0 = 0.020` | L1102-1104 |
| Eq. 13 | 载体身份 diag/ID/分离裕度定义 | L1362-1364 |
| Eq. 14 | 消融内容误差 `e(bz,z ⋆) = 1− bzTz⋆`（DINO 特征） | L1497 |
