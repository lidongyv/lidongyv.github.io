# FlipTest 精读笔记

> 数据源：`C:\Users\Admin\Desktop\design\auto_docs\papers\txt\FlipTest.txt`（103,570 字符 / 1,955 行 / 35 页）
> 规则：所有事实性陈述附论文原句（英文逐字 + 行号）；论文没有的写「原文未写」；解读标注【推断】。

---

## 1) 原文信息

- 标题（L2–3）: "The Flip Test: Video World Models Fail to Learn the World State"
- 作者（L4–6）: "Mi Luo1,2,*, Lijun Yu2,†, Adams Wei Yu2, Ang Cao2, Baochen Sun2, Ming-Hsuan Yang2, Nan Hua2, Yuan Hao2, Alex Dimakis3, Ya Xu2, Ed H. Chi2, Xuezhi Wang2,† and Yinxiao Liu2,†"；机构（L6）: "1UT Austin,2Google DeepMind,3UC Berkeley"；脚注（L7）: "*Work done during an internship at Google DeepMind,†Project co-lead"
- arXiv 编号（L43）: "arXiv:2611.19822v1 [cs.CV] 15 Nov 2026"
- 项目页（L24）: "Project page: https://flip-test.github.io/"
- 文件首行日期（L1）: "2026-11-17"

## 2) Abstract 全文逐字（L8–23）

> Video world models promise to predict the future of a scene from its present. Do current models keep this promise? We probe this question counterfactually. When a small change to the present flips the future, such as a box resting a few millimeters before or past a table edge, a model that tracks the state of the world must flip its prediction too. We build FlipTest, a three-level hierarchy of increasingly strict tests over 1,680 initial frames from 10 simulated and 15 real-world tasks. At the base,visual minimal pairs, two nearly identical frames whose true futures differ, test whether predictions change across the boundary; matched controls go further and test whether they change only where the future does; and dense sweeps, the strictest level, locate exactly where they change. All 11 video world models we evaluate fall short. The best averages 18.3% pair accuracy over real and simulated scenes, below the 25% of random guessing. Models flip as often on steps that leave the future unchanged as on steps that change it, and their response stays flat across the sweeps. On the same frames, a vision-language model is selective and traces a near S-shaped curve near the boundary, and a coding agent that measures the image comes close to the true boundary. This implies that the deciding information is in the frame, and video models just miss it. We therefore urge the community to rethink world modeling as pure video prediction. Next-frame prediction must do more than draw plausible pixels; it must also identify the causal states that control future dynamics.

## 3) 问题与动机（简）

- 世界的因果性由毫米级状态决定（L26–27）: "The world is causal. Whether a cup near a table edge falls, a ball passes through a narrow gap, or a weighted beam tips is already determined in the present by a few millimeters of position."
- 单视频评测的盲点（L62–65）: "single-video evaluation has a critical blind spot: a model can hallucinate the exact same convincing rollout for two scenes whose true futures are opposite, and an isolated plausibility score will never catch the contradiction."
- 核心思路——反事实探测（L65–68）: "Our key insight is to instead probe world models counterfactually—by perturbing the present just enough to flip the future. By comparing predictions across such tipping points, we reduce evaluation to a clean, binary outcome label that generalizes across domains with a sharp physical boundary."
- 形式化（L422–426）: 设状态 s，事件 y(s)∈{0,1}，由标量/低维 margin m(s) 决定（如 support overhang, geometric clearance, net torque, collision offset，L423–424）: "y(s)=𝕀[m(s)>0]"。配对状态 straddle 边界（L430–432）: "we construct state pairs (s−,s+) straddling the boundary, m(s−)<0<m(s+)"; "their initial visual observations are nearly identical, yet their true futures are opposite: y(s−)≠y(s+)"（L432–434）。

## 4) 评测方法实现细节（重点）

### 4.1 三级层级（L440–531）

- 总帧数（L441–442）: "our FlipTest samples states along m(s) across 1,680 initial frames, organized into three difficulty levels."【推断】1,680 = 600(L1) + 600(L2) + 480(L3)。
- **Level 1: Pairs (Detection)**（L443–449）: "Each minimal pair places one state on either side of the tipping point (m(s−)<0<m(s+)). The two initial frames share the exact same scene, objects, and camera, differing only in a single outcome-controlling variable that flips the ground-truth future (e.g. stay vs. fall, pass vs. blocked)."
  - 规模（L449–452）: "comprising 300 minimal pairs (600 initial frames) across 25 tasks in six physical families (Fig. 2): statics and toppling, size and fit, friction and rolling, collision and momentum, free fall and trajectory, and fluids and materials. These are split evenly into 10 simulated tasks (15 pairs each; 300 images) and 15 real-world tasks (10 pairs each; 300 images)."
  - 仿真任务构造（L453–457）: "the 10 simulated tasks are built in MuJoCo (Todorov et al., 2012). Within each pair, the two states sit at symmetric margins ±δ around the analytic or empirically calibrated threshold. Across pairs, δ varies from knife-edge to obvious margins, while camera viewpoints and visual themes vary across pairs but remain fixed within each pair. Crucially, camera placement depends only on static scene geometry, ensuring that framing never leaks the outcome."
  - 真实任务构造（L459–463）: "Of these 300 frames, 59 come from captured footage with recorded physical outcomes, while 241 are counterfactual edits of real photographs or staged synthetic stills. Each paired frame is constructed via a minimal edit that alters only the boundary variable (e.g. sliding an object slightly further past a table edge), fixing the ground-truth label by construction while keeping the pair pixel-matched elsewhere."
- **Level 2: Controls (Selectivity)**（L464–470）: "To test selectivity, we augment each of the 10 simulated tasks with 15 control groups (600 frames total), where each group consists of four equally spaced states at margins −3δ, −δ, +δ, +3δ (Fig. 4a). Only the middle step (−δ→+δ) crosses the physical boundary; the two outer steps apply the exact same spatial perturbation and visual shift while leaving the true outcome unchanged. A model that genuinely tracks the physical state must flip on the middle step while holding steady on the outer steps (Sec. 4.2)."
- **Level 3: Sweeps (Localization)**（L526–531）: "Across four representative simulated tasks (box at a desk edge, block on an incline, seesaw, and block stack) with four scenes each, we densely sample 30 states per scene spanning both sides of the threshold, concentrated near m=0 (Fig. 5a). Across these 480 frames, the ground-truth outcome transitions as a sharp step function at m=0, allowing us to measure whether the model's transition is centered and sharp (Sec. 4.3)."

### 4.2 10 仿真 + 15 真实任务清单

- 六族（L1013–1018）: "statics and toppling (box at a desk edge, object drop, propped stack, block stack, seesaws); size and fit (pipe, wall gap, funnel slide, containment by size, bowl stacking, shape matching); friction and rolling (block on an incline, chain over an edge, loop-the-loop); collision and momentum (domino cascades, billiards, Newton's cradle); free fall and trajectory (containment by distance, ring toss, water pouring, hand sanitizer); and fluids and materials (brush stroke, dye drop)."
- 仿真任务表（Table 1, L1055–1070）: Box at desk edge（COM distance past the edge, 阈值 0, |δ| 0.008–0.045m, stay/fall）；Ball through pipe（bore radius − ball radius, 0, 0.0025–0.014m, blocked/pass）；Ball through gap（slot width − ball diameter, 0, 0.008–0.018m, stuck/pass）；Domino（first-gap ratio d/h, 1.0136, 0.10–0.50, full_fall/break）；Chain over edge（hang fraction f, 1/3, 0.03–0.13, rest/slide）；Loop-the-loop（release height h, 0.776m, 0.03–0.22m, falls/completes）；Seesaw（torque share of one side, 1/2, 0.04–0.14, left_down/right_down）；Block on incline（tilt θ, 26.57°, 1.5°–8.0°, rest/slide）；Block stack（step/block width, 1/3 or 1/2, 0.20–0.32, stable/collapse）；Funnel slide（throat gap − ball diameter, 0, 0.003–0.013m, stuck/finished）。
- 真实任务表（Table 2, L1096–1112）: Drop (desk edge) overhang stay/fall；Containment (size) fits/does_not_fit；Containment (distance) in/out；Domino full_fall/break；Billiards hit/miss；Newton's cradle 1/2/3_balls_out；Propped stack stays_standing/collapses；Brush stroke colored_stroke/no_mark；Water drop fills/misses_cup; into/misses_pool；Hand sanitizer gel_on_palm/misses_hand；Ring toss threads_on/misses_post；Dye drop colored_plume/no_visible_change；Bowl stack stacks/no_stack；Seesaw (real) left_down/right_down/balanced；Matching game fit/no_fit。
- 仿真实现细节（L1019–1037）: "We build all simulated scenes in MuJoCo and render them at 1280×720. Within a task, the 15 magnitudes |δ| are evenly spaced over the range of Tab. 1 and assigned to pairs in a fixed shuffled order. In every pair, the first member lies at −|δ| and the second at +|δ|, so each task contains both knife-edge and clear-cut cases." 相机 jitter（L1022–1024）: "we jitter the task's canonical camera in azimuth (±18◦), elevation (±8◦), and distance (×0.85–1.20) and apply the same jitter to both members of a pair." 阈值除 domino 外均解析（L1025–1026）: "All thresholds are analytic except the domino threshold, which we calibrate by bisection in simulation." 场景常数（L1026–1032）: "a 0.2m cube for the box, a bore radius of 0.06m for the pipe, a 0.12m slot for the gap, and a 0.075m throat for the funnel slide. Friction is μ=0.5 for the chain, so fc=μ/(1+μ)=1/3, and for the incline, so θc=arctanμ. The loop radius is R=0.30m with ball radius a=0.02m, so hc=2.7R−1.7a. The domino row has 10 tiles, and all gaps after the first are spaced at d/h=0.45, so only the first gap crosses the boundary." 校验（L1034–1036）: "We simulate every member, and a pair enters the benchmark only if both simulated outcomes agree with the intended analytic side. All 300 members pass this check."
- 真实实现细节（L1038–1048）: 59 帧实拍 footage + 241 帧 counterfactual edits（L1039–1041）；"the second member is an edit of the first that changes only the variable that decides the outcome... The label is therefore fixed by construction. Edits pass automated consistency checks and are regenerated on failure."（L1042–1045）；propped-stack（L1045–1047）: "we first set the stagger in MuJoCo, with a critical offset of w/N=0.024m and member offsets of 0 and 0.028m, and then restyle the renders photorealistically."；三结局任务（L1047–1048）: "Newton's cradle and the real seesaw have three outcomes, and all other real tasks are binary."

### 4.3 11 个被测模型清单（L562–570）

"Models. We evaluate 11 video world models across three categories: general video generators, including Gemini Omni Flash 1.1 (Google DeepMind, 2026d), MiniMax-H3 (MiniMax, 2026), Wan2.2-I2V-A14B (Wan et al., 2025) alongside its reasoning-tuned variant VBVR-Wan2.2 (Wang et al., 2026b), HunyuanVideo-1.5 (Wu et al., 2025), and CogVideoX1.5-5B-I2V (Yang et al., 2025); physical-AI foundation models, comprising Cosmos-Predict2.5 (Agarwal et al., 2025), Cosmos3-Nano (Agarwal et al., 2026), and Cosmos3-Super (Agarwal et al., 2026); and robotics-focused predictors, namely VPP2 (Guo et al., 2026) and LingBot-Video-MoE-30B-A3B (Ma et al., 2026). Each model generates a single video rollout per initial frame."

生成规模（L84–85）: "evaluating over 21,000 generated video rollouts." 主基准 11×600=6,600 条视频（L1116）: "Over the 11×600=6600 videos of the main benchmark"。

### 4.4 输入、判定与指标定义

- 输入（L532–536）: "every model receives only the clean initial frame and a neutral prompt stating the setup and possible outcomes, without hinting at the underlying decision rule. While VLMs predict the outcome label directly in text from the initial frame, video world models generate a video rollout whose outcome must be read out."
- 双裁判（L536–540）: "we employ two independent VLM judges, Gemini 3.6 Flash (Google DeepMind, 2026a) and Gemini 3.8 Flash (Google DeepMind, 2026b), to classify the event depicted in each generated video (or flag it as unclear). Crucially, the judges evaluate only what the generated video shows, regardless of whether that motion is physically realistic."
- 判定规则（L540–544）: "On Level 1 pairs, a rollout is scored as correct only if both judges agree on the ground-truth outcome; inter-judge disagreements and unclear rollouts are scored as incorrect, since a world model useful for planning must commit to a clear future. For Level 2 controls and Level 3 sweeps, we retain both judges' labels to estimate transition statistics (Sec. B details judge prompts, settings, and inter-annotator agreement)."
- 裁判细节（Sec. B）: 盲协议（L1074–1077）: "The judge sees only the generated video and a text prompt. It never sees the ground-truth label, the generation prompt, or the initial frame, so its reading depends on the video alone." 采样（L1087–1091）: "Two judges, Gemini 3.8 Flash and Gemini 3.6 Flash, run at temperature 0 with a JSON response schema... We sample videos at 8 frames per second, which captures sub-second events such as a ball dropping into a pipe that the default sampling of one frame per second misses."
- 指标定义（L549–561）:
  - Pair accuracy (Level 1)（L549–551）: "A minimal pair is scored as correct if and only if the model predicts the correct outcome for both frames. Under this strict paired metric, random guessing on a binary outcome yields 25%, whereas a constant predictor that ignores the visual perturbation achieves 0%."
  - Selective sensitivity (Level 2)（L552–557）: "we compare the crossing flip rate (CFR, on the −δ→+δ step) against the same-side flip rate (SFR, averaged over the −3δ→−δ and +δ→+3δ steps). We define selective sensitivity as SS=CFR−SFR, which equals zero for both constant predictors and random-flip models, and is positive only when predictions change preferentially across the true boundary. We report 95% bootstrap confidence intervals over 2,000 draws by resampling control groups and sampling one judge per frame."
  - Response curve (Level 3)（L558–561）: "we plot the psychometric curve P(event|m), the fraction of rollouts depicting the target event at each margin m. Whereas the ground-truth simulator exhibits a crisp step function at m=0, a flat, shifted, or gradual curve immediately diagnoses a missing, biased, or blurred physical boundary."

## 5) 伪代码（构造与判定流程）

原文无 Algorithm 环境（全文 1,955 行无 `Algorithm` 关键字）。以下依据 §3.1/§3.2/§3.3 与附录重建：

⚠ 依据原文重建（非原文伪代码）

```
Algorithm: FlipTest 构造与判定
输入: 25 个任务（10 仿真 + 15 真实），每任务含边界 margin m(s) 与标签 y(s)=𝕀[m(s)>0]   [L422-426]
构造:
 1. Level 1: 对每任务采样配对状态 (s−, s+)，满足 m(s−)<0<m(s+)，均接近 m=0        [L430-432, L443-445]
    - 仿真: 10 任务 × 15 对 = 300 帧；真实: 15 任务 × 10 对 = 300 帧               [L449-452]
    - 仿真对内状态取对称 margin ±δ（|δ| 在表 1 范围均匀取 15 档）                  [L453-455, L1020-1022]
    - 相机 jitter 对同对两帧完全一致；相机位置只依赖静态场景几何（不泄露结局）      [L456-457, L1022-1024]
    - 真实帧: 59 实拍 + 241 反事实编辑/摆拍合成，标签由构造确定                    [L459-463]
 2. Level 2: 每个仿真任务 15 个控制组，每组 4 状态 (−3δ, −δ, +δ, +3δ) → 600 帧    [L465-467]
    - 只有中间步 (−δ→+δ) 跨边界；外侧两步等距等视觉扰动但结局不变                 [L467-470]
 3. Level 3: 4 个仿真任务 × 4 场景 × 30 状态 = 480 帧，m=0 附近加密                [L527-531]
执行:
 4. 输入: 仅干净初始帧 + 中性 prompt（不提示决策规则）；视频模型每帧生成 1 条 rollout [L532-536, L569-570]
判定:
 5. 双裁判 Gemini 3.6/3.8 Flash（temperature 0，JSON schema，8fps 采样）读生成视频   [L537-538, L1087-1089]
    - 裁判只看视频描述，不给 GT/初始帧；unclear 计为错                              [L540-542, L1074-1077]
 6. Level 1 计分: 帧正确 ⇔ 两裁判都读 GT；对正确 ⇔ 两帧都正确                       [L540-542, L1092-1093]
指标:
 7. pair accuracy（L1）、SS = CFR − SFR + 2000 次 bootstrap 95% CI（L2）、
    P(event|m) 心理测量曲线 vs oracle 阶跃（L3）                                   [L549-561]
```

（诊断流水线——coding agent 判定流程，按 §C 五步重建，同标注 ⚠）：

```
Algorithm: Simulation agent 判定（每帧）
 1. 用代码测量图像（NumPy/SciPy/scikit-image/OpenCV/PIL）                          [L1151-1152, L1433-1434]
 2. 在搭建 MuJoCo 模型前，仅凭测量写 pre_sim.json 且之后不得修改                   [L1152-1153, L1435-1439]
 3. 判断场景可否用 MuJoCo 表示；不可（液体/墨/胶/强形变）则 method="not_simulable" 直接推理 [L1156-1157, L1440-1445]
 4. 从零 build_scene.py（不可读参数须声明假设）→ simulate.py 跑 rollout，从模拟状态读结局 [L1153-1156, L1446-1463]
 5. 报告结局（"simulated" 时须等于 sim_result.json）                               [L1155-1157, L1464-1472]
再执行校验: 离线沙箱重建场景跑仿真（15min 限制），须 ≥200 仿真步且 ≥1 动态自由度，共 761 次模拟运行全部通过 [L1172-1178]
```

## 6) 主结果数值表

### 6.1 Level 1 pair accuracy（Fig. 3, L197–258；正文确认 L574–579）

| 模型 | 平均 | 真实 | 仿真 |
|---|---|---|---|
| Gemini Omni Flash 1.1 | 18.3 | 31.3 | 5.3 |
| MiniMax-H3 | 14.3 | 25.3 | 3.3 |
| Wan2.2-I2V-A14B | 14.0 | 24.7 | 3.3 |
| VBVR-Wan2.2 | 14.0 | 22.7 | 5.3 |
| Cosmos3-Super | 13.7 | 22.0 | 5.3 |
| Cosmos3-Nano | 10.7 | 16.0 | 5.3 |
| Cosmos-Predict2.5 | 10.3 | 17.3 | 3.3 |
| VPP2 | 8.0 | 14.7 | 1.3 |
| HunyuanVideo-1.5 | 7.0 | 11.3 | 2.7 |
| LingBot-Video-MoE-30B-A3B | 6.3 | 10.0 | 2.7 |
| CogVideoX1.5-5B-I2V | 5.7 | 9.3 | 2.0 |
| chance | 25 | 25 | 25 |

> 注：Fig. 3 数字提取顺序为"平均, 真实, 仿真"【推断：由数值算术校验——如 (31.3+5.3)/2=18.3、(25.3+3.3)/2=14.3，全部 11 行一致；正文 L576–578 独立印证 31.3/25.3/24.7/9.3】。

- 正文结论（L574–579）: "On simulated tasks, no model exceeds 5.3% pair accuracy (a ceiling shared by four models), falling far below the 25% expected from random guessing. On real-world scenes, the strongest model, Gemini Omni Flash 1.1, reaches only 31.3%, followed by MiniMax-H3 (25.3%) and Wan2.2-I2V-A14B (24.7%). Only the top two models surpass chance, while CogVideoX1.5-5B-I2V drops to 9.3%. Averaged across both splits, overall accuracy spans a meager 5.7% to 18.3%."
- 失败主因 mode collapse（L580–583）: "The primary culprit is mode collapse across the boundary: models overwhelmingly generate the exact same outcome for both frames of a pair, guaranteeing that at least one rollout is wrong (Fig. 6). Pooled across all models on clearly judged pairs, predictions remain identical across 72% of real pairs and 89% of simulated pairs."
- 物理违规（L584–587）: "In roughly 14% of simulated evaluations, both judges flagged at least one video as unclear, most notably for CogVideoX1.5-5B-I2V and LingBot-Video-MoE-30B-A3B, as objects spontaneously duplicated, deformed, or launched into midair (Sec. E)."
- 每任务细分（L1343–1345）: "On simulation, no task is solved reliably, the best task mean is 13.3% (domino), and no model solves a single box pair. Real tasks spread from 52.7% (bowl stack) and 37.3% (water drop, hand sanitizer) down to 4.5% (drop, Newton's cradle)."

### 6.2 Level 2 翻转频率（Fig. 4, L406–415；正文 L625–630）

- "all 11 video models cluster tightly along the diagonal: when they do change their prediction, they flip just as often on inconsequential control steps as on decisive boundary crossings."（L625–628）
- "selective sensitivity (Fig. 4c) hovers near zero for every model, ranging from −0.07 (Cosmos-Predict2.5) to +0.07 (VBVR-Wan2.2), with every 95% confidence interval straddling zero."（L628–630）
- VLM 对照（L665–667）: "Gemini 3.6 Flash flips its prediction on 61% of crossing steps versus only 15% of same-side steps, achieving a selective sensitivity of +0.46 (95% CI [0.36, 0.56]; Fig. 4), whereas every video model hovers around zero."

### 6.3 Level 3 sweep 曲线（Fig. 5, L520–525；正文 L639–647）

- "Plotting P(event|m) against the oracle step (Fig. 5b,c) shows that all pixel generators produce flat curves, with cross-boundary probability gaps of only 0.00 to +0.06. Video predictions ignore the margin even at the sweep extremes, defaulting instead to a constant, task-specific bias in each panel (Fig. 5c)."
- "By contrast, both VLMs follow clear S-shaped transitions with gaps of +0.62 (Gemini 4 Argon) and +0.58 (Gemini 3.8 Flash), though their curves rise before m=0, over-predicting the event near the threshold."
- 结论（L646–647）: "whereas video models fail to register the boundary at all, VLMs detect it with a systematic shift."

### 6.4 诊断实验：告知结局（Fig. 8；L672–677, L762–764）

- "Relieved of deciding the outcome, Gemini Omni Flash 1.1 surges from 18.3% to 73.7% pair accuracy, with substantial gains across every model."（L675–677）
- "Even so, conditional rendering remains imperfect: the strongest model still misses over a quarter of the pairs, performing better on real scenes (86.7%) than on simulated ones (60.7%)."（L762–764）
- 其余模型（Fig. 8 数字，L806–827，按 (sim, real) 序【推断】）: MiniMax-H3 65.3/70.7；VPP2 59.3/70.0；VBVR-Wan2.2 46.0/73.3；Wan2.2-I2V-A14B 40.7/70.7；HunyuanVideo-1.5 42.7/56.0；Cosmos3-Super 28.7/57.3；LingBot-Video-MoE-30B-A3B 21.3/58.7；Cosmos-Predict2.5 27.3/45.3；Cosmos3-Nano 14.7/40.7；CogVideoX1.5-5B-I2V 2.7/19.3。

### 6.5 coding agent 近真界（Fig. 7b/c；L772–778, L833–839）

- "the direct VLM already outperforms the best video model on real pairs (87.3% vs. 31.3%; Fig. 7b) and reaches 74.2% frame accuracy on sweeps, but its errors concentrate near the tipping point: sweep accuracy drops to 38% at steps adjacent to m=0 and only exceeds 90% eleven steps away (Fig. 7c)."
- "Making perception explicit via the Measure agent eliminates nearly all of this boundary blur, boosting accuracy to 93.3% on real pairs and over 99% across the sweeps, with only three errors encountered across all sweep frames (Sec. C.1)."
- "the Simulation agent successfully builds and executes MuJoCo scenes in the vast majority of cases, solving over 91% of real pairs and 98% of sweep frames to match the Measure agent."
- "the Simulation agent's pre-simulation estimate from code measurements alone already reaches 96.0% on real frames and 97.5% on sweep frames; running the physical rollout alters the final prediction in less than 1% of runs (Sec. C.1), while requiring roughly ten minutes per frame (Sec. C)."
- 精确数字（L1299–1304）: "The Simulation agent simulated 761 of the 780 scenes and declared the other 19 not simulable (liquids, gels, and deformable objects), answering 18 of these 19 correctly by reasoning." / "The rollout overturned it in only 4 of the 761 simulated runs, with three corrections on sweep frames among the six closest to the boundary (box, incline, stack) and one error on a real bowl-stacking frame."

## 7) 消融 / 稳健性

- 真实 vs 仿真差（L604–608）: "Every model scores 7 to 26 percentage points higher on real scenes than on simulation. Real photographs often contain familiar visual priors that correlate with the outcome, such as a cup visibly overhanging a table edge or a ring clearly wider than its post. By contrast, our simulated split holds visual appearance strictly constant within each pair and perturbs only the continuous boundary variable, making it a purer and stricter test of physical state reading."
- 规模/版本/专训消融（L609–616）: "Scaling up model size, upgrading architectures, or applying domain-specific post-training yields negligible gains. The flagship physical-AI model Cosmos3-Super (13.7%) performs on par with the open general-purpose generator Wan2.2-I2V-A14B (14.0%). Within the Cosmos family, scaling from Cosmos-Predict2.5 (10.3%) and Cosmos3-Nano (10.7%) to Cosmos3-Super (13.7%) brings only marginal gains, and all three remain capped at or below 5.3% on simulation. Similarly, reasoning-tuned VBVR-Wan2.2 (14.0%) matches its base model Wan2.2-I2V-A14B (14.0%) exactly, while robotics-focused predictors VPP2 (8.0%) and LingBot-Video-MoE-30B-A3B (6.3%) lag in the bottom half of the leaderboard."
- 裁判一致性（L1116–1121）: "the two judges return the same label for 85.8% of videos (83.5% on simulated tasks, 88.2% on real tasks), with Cohen's κ=0.85 over the pooled labels. Gemini 3.8 Flash is the stricter judge and rates 16.7% of videos unclear, against 9.4% for Gemini 3.6 Flash." 与人标注（L1122–1124）: "We manually labeled 600 generated videos sampled from the main benchmark with the same outcome labels plus unclear, and Gemini 3.6 Flash returns the human label for 79% of them." 保守性（L1119–1121）: "The dual rule counts a disagreement as wrong, so it can only lower accuracy relative to either judge alone... For Gemini Omni Flash 1.1, it lowers real pair accuracy from 34.7% (Gemini 3.8 Flash alone) to 31.3%."
- 结局条件生成细节（L1544–1550）: "each conditioned prompt replaces only the outcome clause of the original prompt, and in total 51 distinct conditioned prompts cover the 600 frames."；单裁判 Gemini 3.8 Flash，"unclear videos count as wrong"。
- 逐对分解（Table 3, L1637–1656）: "Swapped pairs (both opposite outcomes, on the wrong frames) are rare, at most 8 per model and split. Models therefore seldom flip their prediction across a pair at all, and their main failure is insensitivity to the boundary."（L1555–1557）
- Agent 审计（L1158–1165）: "Each run executes in an isolated sandbox with no network except the model API... Any call to a web tool, or any network code... makes the run invalid. No run used a web tool or failed the audit."
- Measure agent 修正方向（L1294–1298）: "In 56 of the 780 runs, the final outcome differs from the first one, and 52 of these revisions turn a wrong first estimate into a correct answer, against 4 in the other direction."
- VLM 单侧误差与测量纠正（L1206–1214）: "It calls 42 resting blocks on the incline sliding and 10 staying boxes falling, and on the block stack it calls 25 collapsing towers stable. With code-based measurement, the same model makes none of these errors."；"Its sweep accuracy is 94% at the nearest step, 97% at the next, and 100% from the third step on."；提升最大处（L1212–1214）: "as for brush strokes (50%→100%), objects at the table edge (70%→100%), and the domino row (90%→100%)"；最弱处（L1214–1216）: "propped stacks (60%) and the shape-matching game (70%) remain its weakest tasks."
- 运行成本（L1220–1223, L1290–1293）: "A run takes 11.5 minutes on average for the Measure agent and 12.0 minutes for the Simulation agent, with about 24 tool calls."；"The Measure agent spends 80% of its time writing and running image-measurement code and 14% writing its answer and report"；"The Simulation agent spends 57% measuring, 17% building the MuJoCo scene, 9% running the simulation, 4% writing the pre-simulation judgement, and 9% writing its answer"。
- 可能的补救（Sec. F, L1657–1674）: "A contrastive visual schema would point the model to that condition without giving away the answer. It consists of two matched examples that differ in one causally relevant condition and lead to opposite outcomes, shown to the model alongside the target frame."；"FlipTest already supplies a ready-made library of such schemata, since every minimal pair is a contrastive schema for its task."（L1669–1670）

## 8) 局限（逐字，Sec. G, L1675–1681）

> FlipTest studies threshold dynamics of mostly rigid objects from a single initial frame. We chose this setting because it makes ground-truth outcomes unambiguous and lets us locate the physical boundary precisely. One decisive factor changes while the rest stays fixed, so any change in prediction can be attributed directly to that factor. Real scenes often involve several interacting factors that unfold over time. Current video models already miss boundaries set by a single factor, and extending FlipTest to scenes with several intersecting boundaries is a natural next step.

## 9) 叙事关联【推断】（≤4 条）

1. 【推断】FlipTest 为"视频世界模型=通用模拟器"叙事提供可量化反例锚点：从"像素拟真"到"因果状态追踪"的缺口被压缩为一个数字（最强模型 18.3% < 随机 25%，L85–87），可直接用作主张"下一代物理世界模型必须显式读状态"的引证。
2. 【推断】本文借 VDA（perception/dynamics/rendering，L655–658）三段分解做诊断，结论是"瓶颈在感知、动力学引擎几乎无用"（仿真只翻转 4/761 预测，L1303–1304）——可用以支撑"模块化/agentic 世界模拟器蓝图"（L862–864: "An agent estimates the physical state from the scene, calls a dedicated simulator or numerical solver when needed, and passes the predicted trajectory to a video model that renders the motion."）。
3. 【推断】"告知结局后 18.3%→73.7%"（L675–677）是"视频模型学到的是未来长什么样（外观统计），而不是什么决定未来（因果状态）"的直接实验证据，适合放进"端到端像素预测缺失因果"的论证链。
4. 【推断】coding agent 以 93.3%/99.4%（L777）逼近真界，支持"显式状态表征 + 代码测量"路线在物理推理上的优势叙事；但其 11.5–12 分钟/帧（L1220–1221）的高成本，与视频模型秒级生成的对比（原文未给出视频模型时延数值）构成"精度 vs 速度"的张力点，可作为后续叙事中"折中方案"的引子。

---

## 附：最关键原文引用（带行号）

1. L15–16: "All 11 video world models we evaluate fall short. The best averages 18.3% pair accuracy over real and simulated scenes, below the 25% of random guessing."
2. L580–583: "The primary culprit is mode collapse across the boundary: models overwhelmingly generate the exact same outcome for both frames of a pair... predictions remain identical across 72% of real pairs and 89% of simulated pairs."
3. L675–677: "Relieved of deciding the outcome, Gemini Omni Flash 1.1 surges from 18.3% to 73.7% pair accuracy, with substantial gains across every model."
4. L777–778 + L838: "boosting accuracy to 93.3% on real pairs and over 99% across the sweeps" / "running the physical rollout alters the final prediction in less than 1% of runs".
