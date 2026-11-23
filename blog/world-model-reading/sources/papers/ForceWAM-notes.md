# ForceWAM 精读笔记

> 数据源：`papers/txt/ForceWAM.txt`（arXiv 2610.18852，869 行）。所有原文引用逐字复制并注明行号；txt 提取丢失部分空格（如 "DECOUPLINGFUTURE"、"athattps"），引用时保持原样。本文未写处一律标注「原文未写」。

## 1. 原文信息

- **标题**（L1–3，逐字）：“FORCEWAM: DECOUPLINGFUTURE ANDRESPONSE LEARNING FROMDEMO ANDPLAYDATA FOR CONTACT-RICHMANIPULATION”（缺空格为提取伪影，即 "ForceWAM: Decoupling Future and Response Learning from Demo and Play Data for Contact-Rich Manipulation"）。
- **作者**（L4–7）：Zijun Xu、Wendi Chen、Xinkai Wang（三人并列一作，L11 “∗Equal contribution.”）、Jiachen Zou、Hongjin Chen、Hairuo Liu、Bokai Lin、Shihao Ma、Ke Ma、Zhiwei Zhang、Zhuohao Yan、Ke Wu、Haoming Song、Xueqi Li、Chuan Wen、Jieru Zhao、Wenchao Ding、Cewu Lu、Lixin Yang（通讯，L11 “†Corresponding author.”）。
- **机构**（L8–10）：“1Fudan University 2Shanghai Jiao Tong University 3Southeast University 4Shanghai Innovation Institute 5Southern University of Science and Technology 6Noematrix Ltd.”
- **ID/发表状态**（L100）：“arXiv:2610.18852v1  [cs.RO]  13 Oct 2026”——arXiv v1 预印本；是否已投/已录用会议：**原文未写**。
- **主页**（L12）：“Project page: https://force-wam.github.io/”。

## 2. Abstract 全文逐字（L79–98）

> ABSTRACT
> “Expert demonstrations show how to complete contact-rich tasks, but provide lim-
> ited coverage of how interaction forces respond to different actions. Task-agnostic
> play provides broader experience of contact dynamics without specifying task
> goals. To fully leverage these two types of data, we introduce ForceWAM, a
> force world action model that decouples knowledge of task-compatible interaction
> outcomes from knowledge of how local actions change contact. A world model
> learns from successful demonstrations to jointly predict future video and six-axis
> force/torque trajectories. Together, these futures describe the expected evolution
> of motion and contact during the task. A separate inverse dynamics model
> (IDM) learns from both task and play, mapping current observations and recorded
> visual-wrench futures to their associated robot actions. Play data thus provide
> knowledge of force responses for goal-directed execution. At deployment, the IDM
> reactively translates generated futures into actions, reusing them with fresh wrench
> feedback between world-model updates. Across real-robot force-regulated sliding,
> box flipping with and without disturbances, and ring placement, ForceWAM
> achieves 85-95% mean success, surpassing world-modeling and reactive baselines.
> Our experiments further show that adding task-agnostic play data improves success
> by 15-45 percentage points, demonstrating its value for learning contact-rich
> execution. Project page and code are available athttps://force-wam.github.io/.”

## 3. 问题与动机（为什么解耦；demo 与 play 各贡献什么）

**核心障碍：力的规律在数据中出现的「位置」与策略学习它们的「方式」错配。**（L111–112）“However, the main obstacle is a mismatch between where these regularities appear in data and how policies learn them.”

**demo（task data）的贡献与局限：** demo 展示「成功执行」，但对「不同动作下力如何响应」覆盖有限——推箱示范只展示了约 10 N 的标称接触，没有展示按压更深/后退/受扰时力如何变（L113–115）：“Expert demonstrations, which we call task data, show successful task execution. A box-pushing demonstration at roughly 10 N, for example, need not show how force changes when the robot presses farther, backs away, or encounters a disturbance.” 因此 demo 提供「任务相容的未来」（L275–276）：“Successful task data show futures compatible with the task.”，但接触响应知识不足（L80–81）“Expert demonstrations show how to complete contact-rich tasks, but provide limited coverage of how interaction forces respond to different actions.”

**play 数据的贡献：** 任务无关、不要求成功，但提供更广的接触动力学经历与「记录的 future-action 对」（L81–82；L115–118）：“Task-agnostic play provides broader experience of contact dynamics without specifying task goals.” / “By contrast, task-agnostic play data record exploratory pressing, sliding, and release, providing additional visual-wrench transitions and associated robot actions. Play transitions need not demonstrate successful behavior for a given task instruction, but they still provide recorded future-action pairs for inverse-dynamics learning.”

**为什么必须解耦：** 端到端力感知策略把两个问题耦合在同一个仅受任务成功监督的映射里（L118–120）：“End-to-end force-aware learners (Yu et al., 2025; Chen et al., 2026) still couple two questions in one task-supervised mapping: what transition advances the task, and which action realizes that transition.” 解耦之后，两类数据各自训练对应的模块（L129–130）：“We introduce ForceWAM (Figure 1), which uses visual-wrench futures to decouple knowledge of task-compatible interaction outcomes from knowledge of how local actions change contact.”

**关键数据机制（可监督≠完成任务）：**（L135–136）“Any segment that records actions and their resulting motion and contact outcomes can supervise this action prediction, even if it does not complete a task.”

## 4. 方法逐机制

**总体分解（3.1 Overview）**：把端到端策略纠缠的两个问题分开（L194–195）：“ForceWAM separates two questions that are entangled in an end-to-end policy: what should a successful interaction look and feel like?, and what robot actions can realize that interaction?” 联合分布因式分解为 ForceWM×IDM（L199–200，Eq. 1）：“p(gt,A t |o t, ℓ) = p(gt |o t, ℓ)p(At |o t,g t, ℓ) ≈ p ϕ(gt |o t, ℓ)pθ(At |o t,g t), (1)”。训练分工（L204–206）：“This conditioning lets task and play data share the same inverse-dynamics objective. We train pϕ on successful task trajectories Dtask, whereas pθ can learn from Dtask ∪ Dplay.”——注意：**play 只进 IDM，ForceWM 只用成功 demo**。

**数据定义**（L274–278）：“At time t, let It denote the wrist image, wt ∈R 6 the measured TCP force and torque, and ℓ the task instruction.” / “Play data contain the same synchronized image, wrench, TCP-pose, and gripper-state streams, but need not solve or even attempt the downstream task. We therefore use play to learn local execution responses.”

**ForceWM（3.2，未来预测，只学 task）**：预测目标为视觉+六轴力觉联合未来（L280–282）：“For a trajectory window starting at time t, we construct the future prediction target gt = (zt,W t), Wt = ( ¯wt+1, . . . ,¯wt+Hg), where zt encodes future wrist images and Wt is a normalized future wrench trajectory.” 骨干是预训练时空视频扩散模型，把潜变量排成五个语义槽（blank 前缀/当前 wrench/当前图像/未来 wrench 轨迹/未来视觉潜层），前三槽在去噪中固定、后两槽联合生成（L284–290）：“ForceWM models pϕ(gt |I t,w t, ℓ) from successful task demonstrations. Its backbone is a pretrained spatiotemporal video diffusion model. We arrange its latent input as five semantic slots: a blank prefix, the current wrench, the current image, the future wrench trajectory, and the future visual latent. The first three slots are clamped throughout denoising, while the last two are jointly generated.” 训练用 EDM 去噪损失（Eq. 2，L291–302），目标是「任务相容交互的分布」而非单条力轨迹（L304–305）：“Training on successful demonstrations targets a distribution of task-compatible interactions rather than a single force trace. Multiple visual-force futures can accomplish the task.”

**IDM（3.3，响应学习，学 task+play）**：每个训练窗口含当前图像、同一窗口内记录的 future gt、动作率 wrench 序列与动作序列；**训练时不跑 ForceWM**（L314–318）：“Each IDM training window contains the current image It, the recorded future gt constructed from later frames and wrenches in that same window, the action-rate wrench sequence, and the associated action sequence At. ForceWM is not run to create IDM training inputs. The same construction applies to task and play trajectories: although a play window need not accomplish the downstream task, it still provides one feasible action sequence associated with its realized visual-wrench future.” 使用条件扩散 IDM（Chi et al., 2023，Eq. 3，L319–329）。为降低对 ForceWM 预测误差的敏感度，对记录的未来做随机丢弃与扰动（L330–332）：“To reduce the IDM’s sensitivity to imperfect ForceWM predictions at deployment, we apply random dropout and perturbations of the recorded visual-wrench future.”

**Slow-fast 执行（3.4）**：ForceWM 慢更新、IDM 用新力反馈快修正（L334–335）：“ForceWM updates slowly, but the IDM can rapidly adjust actions with new wrench feedback.” 时间因果结构：slow encoders 编码当前观测与未来参考为固定 token，fast 循环编码器逐时刻编码测量 wrench 前缀；动作自注意力与动作→测量 wrench 注意力均用下三角掩码，且隐藏窗口偏移 ≤k 的参考 wrench token（L344–348）：“Training processes the full synchronized action and wrench window in parallel, but applies lower-triangular masks to both action self-attention and action-to-measured-wrench attention. Action token k accesses only noisy action tokens 0:k and measured wrench tokens 0:k, ensuring that its inputs are available during streaming inference. A complementary mask hides reference-wrench tokens at window offsets ≤k, leaving only later reference samples visible to that action token.” 部署时以 ForceWM 预测替换记录未来（L352–353）：“After training, the recorded future is replaced by a ForceWM prediction.” 缓存扩散噪声 ξr，用 DDIM 确定性重去噪，只解码当前动作 token（L364–369）：“Only the current action token is decoded and sent to the low-level controller. The cached base TCP pose Tr converts the relative Cartesian prediction to an absolute pose-gripper command. Within an action chunk, reusing ξr with deterministic DDIM sampling removes variation caused solely by resampling, while the growing measured-wrench feedback prefix changes the conditioning of the newly predicted action token.”

## 5. 实验与主结果

**任务（4 个真实机器人条件，L390–398）**：“Sliding drags the end effector along the centerline of a force scale while targeting a normal force of 10 N. Success requires full-path completion and satisfaction of the 5-15 N contact-force criterion. Box flipping requires sustained contact and a controlled pushing force to carry the box over its edge, and its disturbed variant (Flip-D) adds an external push during contact. Hanoi ring placement seats a held ring onto a 10 mm pole, where contact force decides whether the ring seats or jams.”

**数据（L400–405）**：“We collect 200 task demonstrations and 200 play episodes for each of the three task families. Flip and Flip-D share the same training data.” / “Play is collected without an evaluation-task instruction and consists of exploratory pressing, sliding, and making and breaking contact.”

**评估（L413–416）**：“Each evaluation run consists of 20 trials per task from matched initial states, under the same safety limits and controller interface. A trial succeeds if the task completes without a safety stop or human intervention.”

**Table 1（L423–433，逐字）**：

| Method | Slide↑ | Flip↑ | Flip-D↑ | Hanoi↑ |
|---|---|---|---|---|
| RDP (2025) | 15% | 40% | 30% | 45% |
| ImplicitRDP (2026) | 20% | 45% | 25% | 55% |
| ImplicitRDP (2026)+Play | 10% | 25% | 10% | 30% |
| ForceVLA (2025) | 15% | 55% | 30% | 60% |
| Cosmos Policy (2026) | 15% | 40% | 30% | 60% |
| Cosmos Policy (2026)+Play | 10% | 20% | 15% | 40% |
| Fast-W AM (2026) | 10% | 50% | 25% | 65% |
| FA-W AM (2026) | 20% | 60% | 40% | 50% |
| π0.5 (2025) | 10% | 40% | 20% | 60% |
| ForceW AM (Task-only IDM) | 40% | 75% | 65% | 75% |
| ForceW AM (Full-play IDM) | **85%** | **95%** | **85%** | **90%** |

**Q1 要点**（L434–436）：“ForceWAM achieves the highest mean success in all four conditions, and its Task-only IDM also outperforms every external baseline. Force-regulated sliding shows the largest gap. The Full-play IDM reaches 85% success. ImplicitRDP and FA-W AM reach 20%, and the other task-only baselines reach 10-15%.” 翻转（L447–449）：“For box flipping, ForceWAM reaches 95% success without disturbances and 85% with external pushes, compared with 60% and 40% for FA-WAM, the strongest external comparator in both conditions.” Hanoi（L456–457）：“Fast-WAM reaches 65%, and Cosmos Policy, ForceVLA, and π0.5 reach 60%, compared with 90% for ForceW AM.”

**Q2 play 扩展性（L478–480）**：“We train five IDMs with all task demonstrations and 0, 50, 100, 150, or 200 play episodes per task family, drawing the intermediate subsets independently (Figure 4). ForceWM and the number of optimizer updates are fixed across variants.” 收益（L483–485；L493–495）：“Force-regulated sliding shows the largest absolute gain, from 40% to 85% success in completing the path under the 5-15 N contact-force criterion.” / “For box flipping, play raises success from 75% to 95% without disturbances and from 65% to 85% with external pushes. … Hanoi ring placement improves from 75% to 90%”。端到端基线加 play 反而掉点（L501–504）：“By contrast, the evaluated end-to-end models, ImplicitRDP and Cosmos Policy, do not benefit from added play data: their shared policies learn directly from play actions even when those actions need not advance the task.”

**Q3 ForceWM 闭环预测**：视觉未来覆盖「接近→翻转→分离」全程，力未来同步呈现加载/维持/消退（L511–515）：“Across the rollout, the visual futures capture the progression from approach through box rotation to separation. … The force futures capture the contact loading associated with this motion: load builds as the tool engages the box, remains substantial during rotation, and subsides as the tool separates.” 预测误差（L515–516）：“Despite discrepancies in local visual details and peak wrench values, the predictions preserve the main interaction stages.” Fig.5 RMSE：Fy 2.43 N、Fz 1.05 N（L527, L532）；Fig.7 全六轴：Fx 0.99、τx 0.068、Fy 2.43、τy 0.036、Fz 1.05、τz 0.010（L833–861）。

## 6. 消融

**Q4 未来条件消融（Table 2，L551–555，逐字）**：

| Future conditioning | Slide↑ | Flip↑ | Flip-D↑ | Hanoi↑ |
|---|---|---|---|---|
| Visual+wrench | 85% | 95% | 85% | 90% |
| Visual only | 15% | 40% | 30% | 70% |
| Wrench only | 20% | 30% | 25% | 30% |
| No future conditioning | 15% | 30% | 35% | 50% |
| Visual+wrench, no augmentation | 65% | 80% | 70% | 75% |

**结论引文**（L556–562）：“Joint visual-wrench conditioning achieves the highest mean success rate in all four conditions. Relative to visual-only conditioning, it improves success by 55, 55, 20, and 70 percentage points on Flip, Flip-D, Hanoi, and Sliding, respectively. … Conversely, retaining the visual future alongside the wrench future improves success by 60-65 percentage points across conditions. Compared with no future conditioning, the joint condition improves success by 40-70 percentage points. Neither individual future modality consistently improves over no future conditioning.”【推断：该句 "55, 55, 20, 70" 与表 2 按列相减（Flip 70、Flip-D 55、Hanoi 55、Slide 20）对不上，疑为原文列序笔误；后两处 60–65、40–70 与表一致。】增强消融（L564–568）：“Finally, we ablate the future-conditioning augmentation introduced in Section 3.3 by retraining the joint visual-wrench IDM without perturbations to its recorded future inputs (final row). … Augmentation improves success by 15 percentage points on Flip, Flip-D, and Hanoi, and by 20 points on Sliding.”

**未来条件增强参数（Table 3，L806–812，逐字）**：Whole-condition dropout 0.10；Visual / wrench dropout 0.25/0.50；Scale jitter [0.85,1.15]；Wrench noise std. 0.075 in normalized coordinates；Visual-latent noise std. 0.075 times latent standard deviation；Temporal offsets {0,−1,−2,−3,−4,−5,−6,−9,−12}。

## 7. 局限 / 未来工作

独立 Limitations 章节：**原文未写**。全文仅有的相邻表述：结论（L570–575）：“We presented ForceWAM, which decouples knowledge of task-compatible interaction outcomes from knowledge of how local actions change contact. Task demonstrations train a visual-wrench future model, while task and play data train an inverse dynamics model that translates generated futures into actions with force feedback. Real-robot experiments show higher task success than the evaluated baselines and consistent gains from play. This suggests that using task and play data to learn complementary knowledge offers a promising path to scalable, stable contact-rich manipulation.” 以及预测误差的事实描述（L515–516，见 §5 Q3）。二者是否为作者承认的局限，属【推断】。

## 8. 叙事关联【推断，≤5 条】：「play 数据」对「力学的 free demonstrations」意味着什么

1. **play 免的是「成功标准+任务指令」，不是「采集成本」**：play 仍需真人采集（kinesthetic teaching / VR 遥操作，L720–722：“We collect Hanoi data via virtual-reality (VR) teleoperation using a Meta Quest 3. For box flipping and Sliding, the joint torque sensors support kinesthetic teaching during data collection, while the end-effector F/T sensor records contact forces and torques.”），且与 demo 同传感器流（L401–402：“Both sources provide synchronized wrist RGB, six-axis TCP wrench measurements, and robot actions.”）。所以对「力学的 free demonstrations」而言：**只要记录了「动作+力觉结果」的轨迹即可入训，失败/非任务轨迹也算数**（L135–136：“Any segment that records actions and their resulting motion and contact outcomes can supervise this action prediction, even if it does not complete a task.”）——这大幅降低了数据标注门槛（无需成功判定），但不是零成本数据。【推断：我们若已有带力觉记录的历史执行日志/手动示教，即可充当 IDM 的 play 数据。】
2. **play 不能替代 demo**：ForceWM 只从成功 task 数据训练，play 只进 IDM（L205–206：“We train pϕ on successful task trajectories Dtask, whereas pθ can learn from Dtask ∪ Dplay.”）。即「免费数据」只能补执行层的力响应知识，任务条件（task-compatible futures）仍要成功示范。【推断：我们的力学 pipeline 中，free data 与成功 demo 的角色必须分层，不能混用。】
3. **play 的有效性依赖「解耦」这一结构前提**：同样的 play 数据加到端到端基线（ImplicitRDP/Cosmos Policy）反而掉点（L501–504：“their shared policies learn directly from play actions even when those actions need not advance the task. The added play can therefore shift their action distributions away from successful task behavior”）。即 play 是否「有用」取决于模型能否只学其「动作-力响应」而不学其「动作分布」。【推断：free demos 的价值是结构性问题，不是数据量问题。】
4. **play 的监督信号是「记录的 future-action 对」，而非「模仿 play 动作」**（L117–118：“they still provide recorded future-action pairs for inverse-dynamics learning”；L314–318：“ForceWM is not run to create IDM training inputs.”）。部署时用 ForceWM 生成的未来替换记录未来（L352–353）。【推断：free 数据只提供「接触转移→动作」的映射监督，执行目标仍由成功数据定义。】
5. **量效关系上 play 是「越多越好」（本文范围内线性增益、无饱和迹象）**：0→200 集 play 使四个条件从 40/75/65/75% 升至 85/95/85/90%（L483–495），且 ForceWM 与优化步数固定（L481–482：“ForceWM and the number of optimizer updates are fixed across variants.”）。【推断：若目标是力控成功率，扩充「免费」力觉数据是低成本高收益路径；但 200 集以上的饱和点原文未写。】

## 9. 图注 / 表注逐字抄录

- **Figure 1**（L74–78）：“Figure 1: Motivation of ForceWAM. The illustrated end-to-end force-conditioned and force-envisioned policies use only task demonstration data: the former lack pretrained video-world-model priors, while the latter lack within-chunk force feedback. ForceW AM combines the pretrained prior with task demonstrations for task-compatible visual-wrench futures and action-labeled play for IDM execution learning, achieving the highest mean success across all four real-robot conditions.”
- **Figure 2**（L270–273）：“Figure 2: ForceWAM training and deployment. Task demonstrations train the visual-wrench future generator, while task and play data train the IDM from recorded futures and associated actions. During deployment, ForceWM supplies future conditions and the causal IDM updates actions using measured wrench feedback.”
- **Figure 3**（L388）：“Figure 3: Real-robot comparisons of ForceW AM, RDP, ForceVLA, and FA-W AM.”
- **Figure 4**（L475–477）：“Figure 4: Mean success with increasing play data. Task demos, the ForceWM model, and IDM optimizer steps are fixed.”
- **Figure 5**（L536–538）：“Figure 5: ForceWM predictions during closed-loop box flipping. (a) Recorded wrist images (top) and decoded futures (bottom). The three central predictions share one conditioning observation. (b) Recorded (gray) and predicted (blue) F y and F z (all six wrench components in Appendix F).”
- **Figure 6**（L725）：“Figure 6: Hardware setup and task objects.”
- **Figure 7**（L863–864）：“Figure 7: Full six-axis comparison for the rollout in Figure 5. Recorded (gray) and predicted (blue) forces are shown on the left and torques on the right.”
- **Table 1**（L420–421）：“Table 1: Mean real-robot success rates (the best per column in bold). Slide requires full-path completion with 5-15N contact force. “+Play”: training with additional action-labeled play data.”
- **Table 2**（L547–549）：“Table 2: Online future-conditioning ablation: mean success rates. All IDMs use task + play data, with futures from one fixed ForceWM. Rows 1-4 vary future modalities during training and deployment. The last uses both without training-time future-conditioning augmentation.”
- **Table 3**（L805）：“Table 3: Future-conditioning augmentation used for the flip-box IDM.”
