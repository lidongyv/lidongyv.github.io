# ENACT 精读笔记

> 唯一数据源：`auto_docs\papers\txt\ENACT.txt`（2617 行）。所有英文引用逐字复制自该 txt 并注明行号；txt 中因 PDF 断行产生的连字符（如 "ego-/centric"）已还原，个别缺空格处按原文语义补空格。论文未写的标注「原文未写」，我的解读标注【推断】。

## 1. 原文信息

- **标题**：ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction（L1–2）
- **作者**：Qineng Wang\*¹, Wenlong Huang\*², Yu Zhou³, Hang Yin², Tianwei Bao¹, Jianwen Lyu¹, Weiyu Liu², Ruohan Zhang†², Jiajun Wu†², Li Fei-Fei†², Manling Li†¹（L3–4）
- **机构**：¹Northwestern University ²Stanford University ³UCLA（L5）；"\* Equal contribution; †Equal advising."（L6）
- **ID / 状态**：arXiv:2511.20937v1 [cs.AI] 26 Nov 2025（L77）。会议/期刊录用信息：**原文未写**。
- **官网/代码**：Website at enact-embodied-cognition.github.io（L26）；GitHub 链接在 L7（txt 提取为乱码 "/gl⌢beWebsite/githubCode"）。

## 2. Abstract 全文逐字（L9–26）

> "Embodied cognition argues that intelligence arises from sensorimotor interaction rather than passive observation. It raises an intriguing question: do modern vision-language models (VLMs), trained largely in a disembodied manner, exhibit signs of embodied cognition? We introduce ENACT, a benchmark that casts evaluation of embodied cognition as world modeling from egocentric interaction in a visual question answering (VQA) format. Framed as a partially observable Markov decision process (POMDP) whose actions are scene graph changes, ENACT comprises two complementary sequence reordering tasks: forward world modeling (reorder shuffled observations given actions) and inverse world modeling (reorder shuffled actions given observations). While conceptually simple, solving these tasks implicitly demands capabilities central to embodied cognition—affordance recognition, action–effect reasoning, embodied awareness, and interactive, long-horizon memory from partially observable egocentric input—while avoiding low-level image synthesis that could confound the evaluation. We provide a scalable pipeline that synthesizes QA pairs from robotics simulation (BEHAVIOR) and evaluate models on 8,972 QA pairs spanning long-horizon home-scale activities. Experiments reveal a performance gap between frontier VLMs and humans that widens with interaction horizon. Models consistently perform better on inverse task than forward one and exhibit anthropocentric biases, including a preference for right-handed actions and degradation when camera intrinsics or viewpoints deviate from human vision. Website at enact-embodied-cognition.github.io."

## 3. 问题与动机

**为什么用世界建模来评测具身认知**：具身认知主张智能不是被动习得、而是「enact」出来的；VLM 主要在无身（disembodied）方式下训练，却显现出交互智能迹象，缺的是度量标准：

> "Embodied cognition argues that intelligence is not passively acquired but enacted through continuous sensorimotor interaction with the world (Smith & Gasser, 2005)."（L80–81）

> "Recent advances in large foundation models such as Vision–Language Models (VLMs) (OpenAI, 2025; DeepMind, 2025), although predominantly trained in a disembodied fashion, have exhibited promising signs of interactive intelligence. Yet it remains unclear how to measure the extent to which embodied cognition emerges in these models."（L82–85）

> "Consequently, a unified objective that tightly couples egocentric perception with embodied interactions in everyday activities remains missing (Frick & Möhring, 2016; Thompson, 2005; Clark, 2006; Barsalou, 2020)."（L90–92）

**方案**：把世界建模定义为「自我中心视觉观测在动作条件下的演化」，动作=由物理仿真状态导出的场景图变化：

> "Grounded in a partially observable Markov decision process (POMDP, Åström, 1965, Figure 1), we formulate world modeling (Ha & Schmidhuber, 2018) as the evolution of egocentric visual observations of the environment conditioned on an agent's actions, which are represented as scene graph (Johnson et al., 2017) changes derived from low-level physics-based simulator state."（L95–98）

> "Specifically, we focus on two forms, forward world modeling and inverse world modeling, and they are formulated as sequence reordering for evaluating embodied cognition for a VLM."（L99–100）

**为什么用序列重排而非视频预测**：隔离低层图像合成这个混杂因素：

> "This sequence-reordering VQA view isolates long-horizon interactive visual reasoning from photorealistic video prediction and forces models to reason about how sequences of embodied actions transform the environment from egocentric observations."（L101–103）

**正/逆任务设计意图**：

> "In forward world modeling, given a visual observation and a sequence of actions, the model must reorder a shuffled sequence of future visual observations. In inverse world modeling, given an ordered sequence of visual observations, the model must reorder the corresponding shuffled action sequence."（L103–106）

> "Though seemingly a narrow lens, answering these queries implicitly involves a broad set of capabilities central to embodied cognition: affordance recognition, action–effect reasoning, and embodied awareness, together with reasoning about contact and other low-level physical consequences encoded in our predicates, all from egocentric input. Under partial observability, it demands integrating observations and actions across extended horizons, posing challenges for interactive and long-term memory."（L107–112）

> "Through the same lens, we also examine factors that inform future VLM data design, including existing biases toward human embodiment such as right-handedness and human-like egocentric viewpoints and intrinsics (e.g., FOV, aperture)."（L112–114）

## 4. 方法逐机制

**POMDP 形式化（§2.1）**：

> "Formally, the benchmark is defined over a state space S, whose elements are symbolic scene graphs derived from low-level simulator states G; an observation space O ⊂ R^{H×W×3} of robot's egocentric RGB views of the environment; and an action space A whose elements are scene-graph differences a_t = δ(s_t, s_{t−1})."（L173–176）

**场景图与场景图差动作**：

> "A symbolic scene graph is a structured data model that represents the objects in a scene as symbolic nodes (e.g., On(fridge)) and their relationships as edges (e.g., OnTop(pen, desk))."（L177–178）

> "the action connecting consecutive key frames is the visible scene-graph delta a_k := ΔVis(s_{i_{k+1}}, s_{i_k}), where ΔVis returns the subset of differences in δ(s_{i_{k+1}}, s_{i_k}) that are visible in both images."（L188–190）

> "This initial abstraction into discrete decision epochs is similar to a semi-MDP (Sutton et al., 1999). However, we treat each of these final key-frame trajectories as a self-contained POMDP instance with scene graphs S_π and observations O_π."（L185–187）

符号场景图的两点动机（L1116–1127）：①"our focus is on detecting semantic changes within a scene, a task that naturally aligns with the semantic abstraction of VLMs rather than the continuous motor trajectories"（L1119–1121）；②"the symbolic predicates we employ are grounded in practical robotics applications. This is demonstrated through our egocentric real-world experiments"（L1122–1123）。

**序列重排任务（§2.1）**：

> "For forward world modeling, given the current image o_0, the correctly ordered action sequence (a_0,... ,a_{L−2}), and a shuffled list of observation images O′ = (o′_1,... ,o′_{L−1}), the model outputs a permutation σ ∈ Sym([L−1]) that orders the images to match the actions: (o′_{σ(1)},... ,o′_{σ(L−1)}) = (o_1,... ,o_{L−1}). For inverse world modeling, given o_0, the correctly ordered observation images (o_1,... ,o_{L−1}), and a shuffled list of actions A′ = (a′_0,... ,a′_{L−2}), the model outputs a permutation τ ∈ Sym([L−1]) that orders the actions to be consistent with the observation progression: (a′_{τ(1)},... ,a′_{τ(L−1)}) = (a_0,... ,a_{L−2})."（L193–199）

**BEHAVIOR 数据管线（§2.2）**：

1. **分割帧**：BEHAVIOR 暴露布尔/关系谓词，谓词翻转或关系更新=原子状态变化；δ≠∅ 的帧进入候选池：
   > "The BEHAVIOR simulator exposes boolean and relational predicates, where flipping one predicate or updating a relation is our atomic state change. A time t enters the candidate pool if the scene-graph difference δ(s_t, s_{t−1}) is nonempty."（L205–207）
   > "we form a predicate-level change signature c_j and keep t only if its cosine similarity with the previous signature c_{j−1} is below a threshold."（L208–209）
2. **时序稳定性滤波**（附录 A.2.1）：
   > "a state change is only considered a candidate if the resulting new state persists for at least 40 frames. At our simulator's 30Hz rate, this corresponds to ≈1.3s, which is consistent with cognitive science findings that humans update attentional sub-events on the order of ∼1s"（L977–980）；"we determined a cosine similarity threshold of 0.97"（L993–994）。
3. **关键帧轨迹合成（KFTS）**：允许跳帧、把选轨迹变成「选座」组合问题：
   > "we use skipping to convert trajectory construction into a 'seat selection' combinatorics problem, choosing L seats out of M, which yields at most (M choose L) distinct candidates from a single trajectory."（L218–221）
4. **可扩展性**：
   > "By additionally leveraging a novel QA sampling scheme, our pipeline can generate up to millions of QAs from a single episode, enabling massive scaling across the thousands of trajectories common in robotics datasets."（L121–123）
5. **数据构成**：
   > "BEHAVIOR Challenge provides diverse long-horizon home-scale activities, and we select 29 of them, using one trajectory per activity to recover aligned pairs {(s_t, o_t)}."（L274–275）
   > "Across step lengths L ∈ {3,... ,10} we sample about 560 items per L for each QA type, yielding 8,972 total questions."（L278）
   > 谓词 11 类（Table 6, L1129–1134）：RightGrasping, LeftGrasping, OnTop, Inside, Under, Contains, Covered, Open, ToggledOn, Cooked, Transition。步长分布近乎均匀：3–7 步各 12.8%，8 步 12.7%，9 步 12.2%，10 步 11.1%（Figure 9, L1135–1167）。

**评测设计（§2.3）——在线验证器**：

> "Multiple valid answers can exist for a given question, so we use an online verifier that accepts any predicted permutation, σ or τ, consistent with the input constraints."（L281–282）

> "We report two metrics: Task accuracy captures exact ordering, while Pairwise accuracy grants partial credit for near-correct sequences."（L282–284）

附录 A.3.2 给出验证器语义：前向接受条件 C_i ⊆ C̃_i（"the predicted step must cover the reference's visible change"，L1191–1192），逆向 C̃_i ⊆ F_i（"the predicted action description can be a concise subset of the full reference transition at that position"，L1197–1198），"match = (exact OR semantic)"（L1193, L1199）。

## 5. 实验与主结果

**设置**：7 个闭源 VLM（OpenAI/DeepMind/Anthropic 3 家）+ 23 个开源模型（11 家）（L293–296）；"all images are resized to 512×512, and we use a unified prompt template per QA type"（L297）；温度 0（L1438）。人类基准：240 题分层抽样、3 名标注者独立标注，"Krippendorff's α = 0.83 indicates strong agreement"（L332）；附录精确值 "α = 0.8320, with a 95% bootstrap confidence interval of [0.7879, 0.8682]"（L1409–1410），"perfect agreement on 184 of the 240 selected questions, resulting in a agreement rate of 76.67%"（L1419–1420）。**原文未写**各模型调用次数/费用。

**Table 1｜Pairwise Accuracy（%），L∈{3..10}，L303–323**（表注 L324–326；附录 Table 10 与 Table 1 数值完全相同，L1591–1629）：

| 模型 | 正/逆 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5 | F | 84.62 | 75.26 | 69.96 | 64.18 | 57.48 | 52.16 | 49.45 | 46.93 |
| | I | 86.28 | 80.37 | 76.09 | 68.78 | 65.71 | 62.13 | 57.12 | 55.33 |
| GPT-5 mini | F | 87.50 | 76.25 | 70.65 | 63.41 | 58.14 | 52.38 | 46.65 | 44.11 |
| | I | 85.05 | 76.77 | 75.43 | 67.67 | 63.79 | 57.04 | 55.04 | 50.02 |
| GPT-5 nano | F | 67.83 | 50.29 | 38.61 | 30.35 | 25.97 | 21.90 | 17.59 | 16.84 |
| | I | 72.81 | 53.95 | 42.48 | 36.45 | 31.68 | 28.20 | 24.11 | 20.33 |
| Gemini 2.5 Pro | F | 86.10 | 76.42 | 69.83 | 60.80 | 53.26 | 48.12 | 40.12 | 36.98 |
| | I | 87.94 | 81.18 | 75.39 | 70.03 | 66.03 | 62.91 | 57.78 | 56.62 |
| Gemini 2.5 Flash | F | 81.64 | 67.94 | 54.17 | 43.38 | 37.43 | 32.73 | 29.88 | 28.07 |
| | I | 82.78 | 72.18 | 60.83 | 58.19 | 53.14 | 51.78 | 47.99 | 44.98 |
| Gemini 2.5 Flash-Lite | F | 64.34 | 49.07 | 38.70 | 33.87 | 27.81 | 25.44 | 23.31 | 20.31 |
| | I | 69.58 | 57.55 | 46.04 | 39.09 | 34.06 | 30.18 | 27.51 | 23.16 |
| Claude Sonnet 4 | F | 65.65 | 45.82 | 36.65 | 30.52 | 26.61 | 22.78 | 21.49 | 20.16 |
| | I | 73.25 | 56.85 | 48.87 | 43.07 | 37.00 | 32.71 | 30.50 | 28.49 |
| GLM-4.5V | F | 74.30 | 59.99 | 47.65 | 38.78 | 30.83 | 25.69 | 21.60 | 19.67 |
| | I | 80.59 | 69.28 | 57.04 | 51.53 | 46.95 | 41.68 | 37.36 | 37.93 |
| Llama-4-Mav-17B-128E-Ins | F | 72.47 | 52.09 | 43.87 | 35.30 | 29.90 | 25.89 | 22.79 | 20.49 |
| | I | 72.55 | 62.60 | 50.52 | 43.10 | 35.17 | 31.68 | 28.10 | 25.80 |
| InternVL3.5-241B-A28B | F | 75.79 | 62.25 | 50.83 | 45.85 | 37.84 | 32.88 | 27.85 | 25.24 |
| | I | 82.26 | 70.09 | 60.61 | 53.38 | 45.90 | 39.35 | 34.12 | 30.56 |
| Gemma-3-27b-it | F | 63.29 | 44.66 | 32.04 | 25.82 | 22.11 | 19.50 | 16.74 | 16.29 |
| | I | 64.95 | 48.37 | 40.04 | 33.87 | 28.53 | 23.63 | 21.74 | 19.36 |
| QVQ-72B-Preview | F | 69.14 | 52.96 | 40.83 | 36.27 | 33.16 | 30.63 | 26.30 | 24.76 |
| | I | 71.33 | 58.77 | 48.43 | 44.36 | 40.26 | 39.30 | 36.66 | 36.58 |
| Qwen2.5-VL-72B-Ins | F | 78.15 | 60.05 | 49.87 | 41.92 | 36.77 | 31.73 | 28.03 | 25.07 |
| | I | 77.80 | 65.85 | 53.30 | 48.19 | 44.07 | 37.57 | 33.76 | 36.27 |
| Qwen2.5-VL-32B-Ins | F | 67.83 | 55.46 | 44.35 | 35.75 | 27.52 | 26.42 | 22.01 | 18.07 |
| | I | 63.55 | 59.70 | 54.57 | 51.01 | 49.36 | 47.17 | 41.47 | 40.16 |
| Ovis2.5-9B | F | 58.39 | 42.51 | 34.96 | 31.08 | 24.61 | 20.78 | 18.11 | 16.96 |
| | I | 64.86 | 51.74 | 41.65 | 35.47 | 30.95 | 26.64 | 23.70 | 23.25 |
| MiniCPM-V-4.5 | F | 60.75 | 38.73 | 33.65 | 25.47 | 24.81 | 21.40 | 21.56 | 18.33 |
| | I | 69.23 | 53.08 | 47.35 | 39.55 | 34.87 | 30.63 | 27.05 | 25.71 |
| Idefics3-8B-Llama3 | F | 60.23 | 36.99 | 31.83 | 24.25 | 21.29 | 20.80 | 20.46 | 17.71 |
| | I | 47.38 | 33.86 | 27.26 | 23.48 | 19.87 | 18.50 | 17.04 | 15.16 |
| Cosmos-Reason1 | F | 56.28 | 41.86 | 34.75 | 28.40 | 26.46 | 26.49 | 25.41 | 24.88 |
| | I | 58.30 | 45.93 | 44.25 | 38.50 | 35.72 | 34.56 | 31.50 | 28.64 |
| BAGEL | F | 30.24 | 40.19 | 29.65 | 25.37 | 22.75 | 19.45 | 17.84 | 15.87 |
| | I | 56.73 | 52.85 | 40.09 | 35.44 | 29.67 | 24.39 | 28.70 | 18.91 |
| Human | F | 93.62 | 95.30 | 95.04 | 93.87 | 95.43 | 95.41 | 94.75 | 95.13 |
| | I | 92.05 | 93.56 | 94.35 | 94.25 | 95.96 | 97.74 | 96.30 | 96.29 |

**Table 9｜Task Accuracy（%），L1538–1573**（附录完整表）：

| 模型 | 正/逆 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5 | F | 80.59 | 62.72 | 47.13 | 33.62 | 20.24 | 11.58 | 7.30 | 5.00 |
| | I | 86.19 | 72.65 | 59.65 | 43.73 | 33.68 | 24.04 | 17.15 | 13.00 |
| GPT-5 mini | F | 83.39 | 62.72 | 45.22 | 31.71 | 19.02 | 9.12 | 5.29 | 2.80 |
| | I | 84.79 | 67.42 | 58.09 | 41.11 | 29.67 | 18.07 | 13.50 | 8.60 |
| GPT-5 nano | F | 58.57 | 30.66 | 9.74 | 3.83 | 1.40 | 0.00 | 0.00 | 0.00 |
| | I | 72.03 | 39.02 | 17.22 | 8.19 | 3.14 | 1.05 | 0.36 | 0.00 |
| Gemini 2.5 Pro | F | 81.99 | 62.72 | 47.30 | 29.79 | 17.80 | 10.00 | 3.28 | 3.60 |
| | I | 87.76 | 73.52 | 58.61 | 43.38 | 33.51 | 23.68 | 15.88 | 14.40 |
| Gemini 2.5 Flash | F | 75.52 | 50.52 | 25.22 | 14.29 | 6.28 | 2.98 | 1.28 | 0.20 |
| | I | 82.52 | 61.15 | 38.96 | 27.70 | 17.98 | 13.86 | 6.20 | 3.80 |
| Gemini 2.5 Flash-Lite | F | 52.97 | 27.18 | 10.09 | 3.83 | 1.40 | 0.18 | 0.18 | 0.00 |
| | I | 69.06 | 42.33 | 19.83 | 8.54 | 4.71 | 0.88 | 0.73 | 0.00 |
| Claude Sonnet 4 | F | 56.29 | 24.91 | 8.52 | 2.96 | 0.70 | 0.00 | 0.00 | 0.00 |
| | I | 72.73 | 42.16 | 24.17 | 13.59 | 6.98 | 2.63 | 1.46 | 1.00 |
| GLM-4.5V | F | 66.08 | 40.77 | 18.09 | 8.54 | 1.57 | 0.35 | 0.18 | 0.00 |
| | I | 79.55 | 57.32 | 32.52 | 20.38 | 11.69 | 5.44 | 1.64 | 0.40 |
| GLM-4.1V-Thinking | F | 57.52 | 28.40 | 11.30 | 2.26 | 0.35 | 0.18 | 0.00 | 0.00 |
| | I | 73.43 | 39.37 | 12.00 | 4.53 | 0.87 | 0.53 | 0.00 | 0.00 |
| Llama-4-Scout-17B-16E-Ins | F | 58.74 | 21.43 | 5.04 | 1.74 | 0.70 | 0.18 | 0.00 | 0.00 |
| | I | 64.34 | 34.32 | 10.26 | 2.96 | 1.75 | 0.00 | 0.18 | 0.00 |
| Llama-4-Mav-17B-128E-Ins | F | 63.99 | 32.58 | 14.78 | 4.36 | 1.57 | 0.35 | 0.00 | 0.00 |
| | I | 71.50 | 49.30 | 24.35 | 11.85 | 4.19 | 1.58 | 0.55 | 0.00 |
| InternVL3.5-241B-A28B | F | 67.83 | 43.38 | 21.22 | 12.02 | 4.71 | 1.05 | 0.36 | 0.00 |
| | I | 81.99 | 59.76 | 40.35 | 24.22 | 15.18 | 7.37 | 4.56 | 2.00 |
| InternVL3.5-14B | F | 46.33 | 14.81 | 3.48 | 1.05 | 0.00 | 0.00 | 0.00 | 0.00 |
| | I | 66.43 | 45.12 | 23.65 | 11.85 | 5.93 | 1.93 | 1.28 | 0.40 |
| InternVL3.5-8B | F | 54.72 | 25.09 | 5.39 | 1.05 | 1.22 | 0.18 | 0.00 | 0.00 |
| | I | 63.99 | 40.24 | 20.00 | 6.79 | 3.49 | 0.53 | 0.36 | 0.20 |
| InternVL3.5-4B | F | 54.55 | 22.13 | 6.43 | 2.09 | 0.52 | 0.00 | 0.00 | 0.00 |
| | I | 63.64 | 32.93 | 16.00 | 5.75 | 2.27 | 0.53 | 0.18 | 0.00 |
| Gemma-3-27b-it | F | 53.15 | 22.82 | 5.57 | 0.87 | 0.17 | 0.18 | 0.00 | 0.00 |
| | I | 63.46 | 31.88 | 14.61 | 5.05 | 1.57 | 0.35 | 0.00 | 0.60 |
| Gemma-3-12b-it | F | 51.22 | 21.78 | 6.09 | 1.05 | 0.17 | 0.00 | 0.00 | 0.00 |
| | I | 52.80 | 27.53 | 9.74 | 2.79 | 1.75 | 0.35 | 0.00 | 0.00 |
| Gemma-3-4b-it | F | 52.80 | 20.56 | 1.57 | 0.17 | 0.70 | 0.00 | 0.00 | 0.00 |
| | I | 52.45 | 18.12 | 3.83 | 1.92 | 0.17 | 0.00 | 0.00 | 0.00 |
| QVQ-72B-Preview | F | 60.84 | 29.79 | 8.17 | 2.09 | 0.70 | 0.00 | 0.00 | 0.00 |
| | I | 66.96 | 40.24 | 16.87 | 6.97 | 3.84 | 1.23 | 0.55 | 0.00 |
| Qwen2.5-VL-72B-Ins | F | 71.68 | 40.42 | 18.96 | 7.84 | 3.32 | 1.23 | 0.00 | 0.00 |
| | I | 75.87 | 53.48 | 29.74 | 17.77 | 11.52 | 4.74 | 1.46 | 0.40 |
| Qwen2.5-VL-32B-Ins | F | 51.40 | 32.75 | 10.09 | 3.48 | 0.52 | 0.00 | 0.00 | 0.00 |
| | I | 39.34 | 33.45 | 19.13 | 8.89 | 6.11 | 2.11 | 0.91 | 0.00 |
| Qwen2.5-VL-7B-Ins | F | 22.73 | 23.17 | 5.39 | 0.52 | 0.17 | 0.00 | 0.00 | 0.00 |
| | I | 70.10 | 41.11 | 16.52 | 5.23 | 1.05 | 0.00 | 0.00 | 0.00 |
| Qwen2.5-VL-3B-Ins | F | 45.98 | 13.76 | 5.91 | 0.70 | 0.17 | 0.00 | 0.00 | 0.00 |
| | I | 56.64 | 32.75 | 13.39 | 5.75 | 1.05 | 0.18 | 0.00 | 0.00 |
| Ovis2.5-9B | F | 47.55 | 23.00 | 10.61 | 2.96 | 1.05 | 0.18 | 0.00 | 0.00 |
| | I | 62.76 | 35.54 | 16.00 | 6.27 | 1.75 | 0.35 | 0.00 | 0.00 |
| Ovis2.5-2B | F | 39.69 | 17.77 | 5.91 | 0.87 | 0.52 | 0.00 | 0.00 | 0.00 |
| | I | 48.43 | 23.87 | 8.52 | 1.57 | 0.00 | 0.00 | 0.00 | 0.00 |
| MiniCPM-V-4.5 | F | 48.43 | 19.16 | 8.35 | 1.92 | 0.52 | 0.18 | 0.00 | 0.00 |
| | I | 68.01 | 37.98 | 22.09 | 9.41 | 3.66 | 1.75 | 0.18 | 0.20 |
| MiniCPM-o-2.6 | F | 26.05 | 17.07 | 5.22 | 1.57 | 0.00 | 0.00 | 0.00 | 0.00 |
| | I | 38.64 | 27.35 | 11.30 | 2.44 | 0.52 | 0.18 | 0.00 | 0.00 |
| Idefics3-8B-Llama3 | F | 48.08 | 16.20 | 2.26 | 0.52 | 0.17 | 0.00 | 0.00 | 0.00 |
| | I | 46.33 | 16.72 | 2.96 | 1.57 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cosmos-Reason1 | F | 45.45 | 21.43 | 5.04 | 0.52 | 0.17 | 0.00 | 0.00 | 0.00 |
| | I | 51.92 | 29.09 | 12.02 | 3.31 | 0.52 | 0.18 | 0.00 | 0.00 |
| BAGEL | F | 25.87 | 17.77 | 3.83 | 2.09 | 0.17 | 0.00 | 0.00 | 0.00 |
| | I | 56.29 | 35.89 | 14.61 | 7.14 | 2.97 | 0.18 | 0.18 | 0.00 |
| Human | F | 90.38 | 92.16 | 89.74 | 85.71 | 88.31 | 87.02 | 85.58 | 84.00 |
| | I | 91.78 | 90.24 | 88.70 | 88.15 | 89.53 | 92.28 | 87.73 | 85.00 |

**三个核心发现（原文结论句）**：

① 正/逆不对称：
> "Across families and step lengths, inverse consistently outperforms forward, with the margin widening as L grows."（L336–337）
> "This asymmetry suggests that models handle retrospective textual reasoning better than the prospective visual simulation required for forward planning."（L340–341）

② 随视界单调退化：
> "Accuracy decreases monotonically with L for nearly every model, proprietary and open-weight alike."（L342–343）
> "This sharp performance decay reveals that VLMs struggle to track evolving physical states."（L345–346）

③ 与人类差距随视界扩大：
> "Human performance is far better than any evaluated VLM. SOTA VLMs such as GPT-5 and Gemini-2.5 Pro are comparable to humans only at step length L = 3; their performance drops sharply as the horizon grows."（L347–349）
> "Since many models collapse at long horizons (L = 8–10, near-zero task success), we focus on the more informative Pairwise Accuracy."（L333–335）

模型间比较："GPT-5 and Gemini 2.5 Pro are the strongest overall in both forward and inverse settings. Several open-weight VLMs are competitive: InternVL3.5-241B-A28B, GLM-4.5V, and Qwen2.5-VL often close much of the gap and even surpass Claude 4 Sonnet in multiple settings. GPT-5 mini is also highly competitive, achieving the best score at several short and mid horizons (e.g., forward at L=3, 7, 8)."（L352–363）Cosmos-Reason1（具身数据训练）："For similar-sized models, Cosmos-Reason1-7B exhibits more stable and generally better performance than other VLMs when the interaction horizon exceeds 5 steps."（L365–367）

**真实世界评测（§3.2）**：3 个真实场景（kitchen/dinner table/workspace）手工标注场景图，"Applying the ENACT pipeline to these annotations yields 960 real-world QA pairs, on which we evaluate InternVL3.5-241B-A28B."（L386–388）

**Table 2｜真实世界 InternVL3.5-241B-A28B（%，L414–415）**（表注 L416–418）：

| Metric | 方向 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Task | F | 73.33 | 50.00 | 33.33 | 13.33 | 8.33 | 3.33 | 0.00 | 0.00 |
| | I | 90.00 | 85.00 | 55.00 | 38.33 | 21.67 | 13.33 | 6.67 | 3.33 |
| Pairwise | F | 80.00 | 67.78 | 61.25 | 49.00 | 38.89 | 37.14 | 26.88 | 25.74 |
| | I | 90.00 | 88.33 | 72.92 | 57.67 | 44.17 | 43.57 | 31.04 | 26.85 |

> "InternVL3.5-241B-A28B exhibits similar absolute accuracy on real-world data as in the simulator."（L404–405）
> "Overall, real-world evaluation does not reveal a significant sim-to-real gap, and the empirical trends are fully consistent with those observed in the simulator."（L407–408）

**渲染真实度（§3.2）**：基线 Ray Tracing；变体 Realistic（GPT-image-1 风格迁移）/Path Tracing/Ray Tracing Only；"For each L and each QA type (forward, inverse), we sample 50 items, yielding 300 total QAs."（L394–395）结果：
> "We find no statistically significant degradation or improvement across the spectrum. All settings have p ≥ 0.2 relative to the baseline, and observed deltas are small across both QA types and all step lengths"（L419–421）

**人类中心偏置（§3.3、§3.4）**：

- FOV：基线 Aperture 40，测试 30/60/80/Fisheye：
  > "a small change to Aperture 30 shows no significant difference from baseline (p > 0.1). Larger deviations hurt performance: Aperture 60, 80, and Fisheye are consistently and significantly worse than baseline across QA types and step lengths (p ≤ 0.01), suggesting that the model performs better with human-like intrinsics."（L455–458）
- 相机高度：基线 1.75 m，High +0.5 m、Low −0.25 m：
  > "increasing the camera height (High) significantly reduces GPT-5 mini's accuracy in the forward setting with Δ = −0.13. By contrast, the High inverse setting shows no statistically significant change, though with a performance drop Δ = −0.06. For the Low camera, both forward and inverse are not significantly different from the baseline, likely because the −0.25 m shift remains within normal human height variation."（L460–464）
- 机器人外观：White/Random/Skin Color 三变体，"the performance deltas are small (|Δ| < 0.05) and non-significant (all p > 0.10), suggesting that the model's understanding of its interaction with the world is not tied to a specific visual representation of its body."（L492–493）
- 利手：以人类约 89% 右利手为参照（L480–482）：
  > "For both forward and inverse tasks, the right hand consistently outperforms the left hand across all metrics."（L497–498）
  > "For instance, in the forward task, 9.38% of true left-hand changes were incorrectly identified as right-hand changes, whereas only 4.67% of right-hand changes were misattributed to the left."（L500–501）
  Figure 5 C.2（GPT-5 mini，L438–441）：Forward Left precision 0.4483±0.0076 / recall 0.4087±0.0072 / mixing 0.0938，Right 0.4976±0.0055 / 0.4958±0.0055 / 0.0467；Inverse Left 0.4040±0.0075 / 0.4040±0.0075 / 0.1858，Right 0.4618±0.0058 / 0.4618±0.0058 / 0.0949。Figure 18 C.2（InternVL3.5-241B-A28B，L1755–1758）：Forward Left 0.2991±0.0067 / 0.2626±0.0060 / 0.1373，Right 0.3465±0.0048 / 0.3427±0.0048 / 0.0628；Inverse Left 0.2821±0.0065 / 0.2820±0.0065 / 0.2179，Right 0.3294±0.0051 / 0.3295±0.0051 / 0.1099。

**错误分析（§3.5，GPT-5，Figure 6）**：五类错误=Entity Substitution / Polarity Inversion / Predicate Substitution / Hallucination / Omission。
> "the vast majority of errors fall into two main categories: Omission and Hallucination. For the forward task, these two error types account for a combined 81% of all failures. This figure is even higher for the inverse task, where they make up nearly 84% of errors."（L551–556）
> "In forward modeling, Hallucination is the most common error at 43.9%, followed by Omission at 37.1%. Remarkably, in the inverse task, these two errors are perfectly balanced, each accounting for exactly 41.8% of all failures."（L563–565）
> "Polarity Inversion is more common in the forward setting (12.4%) than the inverse (9.2%). Entity Substitution is also slightly more prevalent in the forward task (6.3% vs. 5.4%). Finally, Predicate Substitution remains the rarest error type, though it is more pronounced in the inverse setting (1.9%) compared to the forward task (0.3%)."（L566–571）
> "The dominance of hallucinations suggests models rely on learned textual priors rather than faithful visual grounding."（L573–574）

## 6. 消融（全部抄录）

**① Contact 消融（接触谓词触发关键帧，InternVL3.5-241B；Table 11, L1633–1634）**：

> "we add an ablation in which key frames are also gated on changes in binary contact relations between objects (e.g., touch / no-touch)."（L1579–1580）

| Metric | 方向 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Task | F | 86.67 | 43.33 | 36.67 | 20.00 | 3.45 | 3.33 | 0.00 | 0.00 |
| | I | 90.00 | 73.33 | 30.00 | 26.67 | 16.67 | 3.33 | 6.67 | 0.00 |
| Pairwise | F | 90.00 | 72.22 | 60.00 | 53.33 | 48.28 | 42.38 | 31.67 | 34.44 |
| | I | 90.00 | 82.22 | 55.83 | 57.33 | 46.11 | 32.86 | 38.75 | 27.78 |

> "the qualitative trends remain the same as in our main results. Inverse world modeling consistently outperforms forward modeling across all horizons (e.g., 86.67% vs. 90.00% task accuracy at 3 steps, and 3.45% vs. 16.67% at 7 steps), and both task and pairwise accuracies still drop substantially as the number of interaction steps increases for both directions."（L1584–1587）

**② 动作表征消融（2,304-QA 子集，InternVL3.5-241B；Table 12, L1641–1648）**：三种编码=Vanilla（自然语言）/Symbolic Predicates/Emoji-Style Encodings：

| 编码 | Metric | 方向 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla | Task | F | 68.97 | 35.17 | 27.59 | 8.97 | 6.90 | 2.07 | 0.69 | 0.00 |
| | | I | 83.45 | 60.69 | 44.14 | 24.83 | 13.79 | 7.59 | 4.14 | 0.00 |
| | Pair | F | 76.21 | 57.70 | 57.41 | 41.24 | 38.85 | 30.34 | 29.05 | 26.52 |
| | | I | 83.45 | 69.66 | 61.72 | 53.10 | 47.01 | 40.39 | 34.40 | 27.25 |
| Symbolic | Task | F | 67.59 | 40.00 | 20.00 | 9.66 | 4.14 | 0.69 | 1.38 | 0.00 |
| | | I | 79.86 | 51.03 | 40.00 | 23.45 | 8.97 | 5.52 | 2.76 | 0.73 |
| | Pair | F | 74.48 | 61.61 | 48.97 | 41.52 | 35.75 | 32.12 | 30.34 | 25.68 |
| | | I | 79.86 | 62.99 | 58.62 | 51.59 | 39.43 | 36.85 | 32.07 | 24.33 |
| Emoji | Task | F | 65.52 | 44.14 | 18.62 | 11.03 | 8.28 | 0.69 | 0.69 | 0.00 |
| | | I | 77.24 | 48.28 | 35.17 | 23.45 | 12.41 | 7.59 | 2.76 | 0.00 |
| | Pair | F | 73.10 | 64.60 | 48.62 | 39.45 | 38.05 | 27.49 | 26.72 | 23.93 |
| | | I | 77.93 | 61.38 | 59.14 | 45.38 | 40.80 | 36.06 | 27.16 | 25.30 |

> "the inverse>forward gap is preserved even with purely symbolic or emoji-style actions, suggesting that our conclusions are not driven solely by natural-language priors."（L1671–1673）

**③ 验证器宽松度（Table 13, L1653–1655）**：语义接受但谓词集为真值严格子集的比例（%）：

| 模型 | Data-Level Mismatch (%) | Pair-Level Mismatch (%) |
|---|---|---|
| GPT-5 | 1.65 | 15.31 |
| Human | 2.26 | 3.62 |
| InternVL3.5-241B | 1.49 | 13.96 |

> "At the data level, all models (and humans) exhibit very low mismatch rates (≈1–2%)... At the pair level, models show higher mismatch ratios than humans (around 14–15% vs. 3.6%)... such cases are relatively rare at the QA level and do not dominate the evaluation."（L1686–1691）

**④ 谓词噪声鲁棒性（Table 14, L1910–1914）**：2,304-QA 子集、3 个随机种子，mean±std（%）：

| Metric | 方向 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Task | F | 68.51±1.44 | 37.47±2.11 | 25.75±1.74 | 10.80±1.59 | 4.60±2.11 | 1.61±0.80 | 0.92±0.40 | 0.00±0.00 |
| | I | 83.45±0.69 | 58.39±3.40 | 43.91±1.05 | 22.76±2.07 | 13.79±0.00 | 6.67±2.22 | 4.14±2.07 | 0.73±1.26 |
| Pair | F | 75.52±1.19 | 59.77±1.82 | 54.48±2.54 | 42.25±1.30 | 37.36±1.30 | 30.90±0.59 | 28.76±0.43 | 25.98±1.01 |
| | I | 83.45±0.69 | 68.74±1.40 | 61.49±0.72 | 52.46±0.69 | 45.59±1.23 | 39.70±1.64 | 35.09±2.57 | 29.01±1.64 |

> "the qualitative trends remain unchanged: Inverse world modeling consistently outperforms forward world modeling at comparable horizons, and performance for both directions degrades sharply as the number of steps increases."（L1923–1926）

消融公共设置：主模型 GPT-5 mini + 开源对照 InternVL3.5-241B-A28B（L1698–1702）；"two-tailed unpaired Welch's t-test... consider a result to be statistically significant if p < 0.05... |Δ| < 0.05 as a small change"（L1734–1739）。

## 7. 结论与局限（原文逐字）

**Conclusions（L617–631）**：

> "In this work, we introduced ENACT, a benchmark designed to evaluate the extent to which embodied cognition emerges in VLMs trained on passive datasets. By framing evaluation as forward and inverse world modeling from egocentric interaction, ENACT assesses a model's understanding of environmental dynamics and the consequences of its actions. Grounded in a POMDP, we cast this as two sequence-reordering tasks: forward world modeling, which predicts an ordered sequence of future states from actions, and inverse world modeling, which infers an ordered sequence of actions from state changes. Our extensive experiments reveal a significant performance gap between state-of-the-art VLMs and humans, a gap that widens dramatically as the interaction horizon increases. We consistently found that models solve the inverse problem more effectively than the forward one. Furthermore, our analysis uncovered strong embodied biases within these models, including a preference for right-handed actions and a significant performance drop with non-human-like camera perspectives. An in-depth error analysis showed that reasoning failures are primarily driven by the omission and hallucination of state changes. ENACT provides a scalable and insightful tool for charting a course toward more genuinely embodied artificial intelligence."

**Limitations（L632–645）**：

> "Our work has limitations primarily related to its scope. First, while we introduce several diagnostic tasks that reveal key model biases, this set is not exhaustive. The experiments on factors like camera configuration and agent appearance serve as foundational examples, but the ENACT framework is designed to be an extensible tool. It can support future, more complex investigations into a much broader spectrum of different embodied-related settings. Second, due to the significant computational cost of evaluation on VLMs, the in-depth ablation experiments were necessarily focused on a representative subset of models and data. A broader evaluation across more architectures and larger data scales would be beneficial to generalize our findings. Furthermore, we do not explore finetuning VLM in this work, but we expect our automatic and scalable dataset can also bring benefits to improving VLM's embodied world modeling abilities. Additionally, due to the frequent physical inconsistency of generated rollouts and the difficulty of designing fair evaluation metrics, which often require costly human studies, we do not evaluate video generative models on ENACT (for unified VLMs, we evaluate BAGEL (Deng et al., 2025) and the result is shown in Table 9 and 10)."

## 8. 叙事关联【推断】（≤5 条，各附支持原句）

1. **「验证器审慎且事后」的评测视角同构**：ENACT 的判分哲学是「允许多解、只给部分分、绝不放过幻觉」——在线验证器按子集语义事后裁决，而非要求唯一答案。【推断】这与「验证器审慎且事后」的评测叙事一致：正确性由与真值转移约束的一致性决定，而非与预测格式的精确匹配。支持原句：L281–282（"we use an online verifier that accepts any predicted permutation, σ or τ, consistent with the input constraints"）；L1677–1679（"we allow partial correctness (omitting some true predicates), but never accept hallucinated predicates that conflict with the annotated transition"）；L1686–1691（QA 级 mismatch 仅 ≈1–2%，"such cases are relatively rare at the QA level and do not dominate the evaluation"）。

2. **触觉侧 Inact「考核什么」的参照**：ENACT 考核的是「自我中心观测序列 ↔ 符号状态变化序列」的因果对齐（正=动作→观测、逆=观测→动作），输入只有 RGB 图像，**不**考核触觉信号本身、不考核低层连续动力学、不考核视频生成。论文唯一涉及「接触」之处是把 touch/no-touch 作为符号谓词用于关键帧门控消融，且结论不变——【推断】这给 Inact 触觉侧的直接启示是：可把接触/触觉谓词当作与 ENACT 同构的「动作空间」元素构造正/逆重排题，ENACT 管线已被证明兼容接触谓词。支持原句：L175–176（"an observation space O ⊂ R^{H×W×3} of robot's egocentric RGB views of the environment"）；L109–110（"together with reasoning about contact and other low-level physical consequences encoded in our predicates"）；L1579–1580（"key frames are also gated on changes in binary contact relations between objects (e.g., touch / no-touch)"）；L1117–1118（"it may not capture the fine-grained details of low-level dynamics"）；L642–644（"we do not evaluate video generative models on ENACT"）。触觉通道的评测：**原文未写**。

3. **「正/逆不对称」是可迁移的预期**：模型一致地在逆任务（由果推因）优于正任务（由因推果），差距随视界扩大。【推断】若 Inact 同时考核「动作→预期感知」（前向）与「感知→动作/状态」（逆推）两类题，应预设逆推类得分更高，并以此为基线而不是缺陷。支持原句：L336–341（"inverse consistently outperforms forward, with the margin widening as L grows... models handle retrospective textual reasoning better than the prospective visual simulation required for forward planning"）。

4. **多视界是考核硬指标**：人类在所有视界保持 92.05–97.74（Pairwise，L323），VLM 单调衰减、L=8–10 时 Task 准确率近乎归零。【推断】Inact 的触觉侧考核应显式划分视界档位（短/中/长），因为「长视界退化」正是本文判定「交互记忆能力不足」的核心信号。支持原句：L333–335（"many models collapse at long horizons (L = 8–10, near-zero task success)"）；L342–346；L347–349。

5. **评测需显式控制「体态偏置」混杂**：ENACT 证明渲染真实度不是瓶颈（p ≥ 0.2，L419–421），但相机 FOV/高度、利手偏置会显著改变得分。【推断】对 Inact 的启示：考核通道自身的「非人类化」配置（如触觉传感器的空间分辨率/位姿/分布）应像本文对待 camera intrinsics 一样被当作显式消融变量测试，而非默认中性；同时应借鉴其「用符号谓词+在线验证器避免低层合成混杂」的做法。支持原句：L456–458（Aperture 60/80/Fisheye p ≤ 0.01）；L468–469（"Reliance on human-centric visual priors limits generalization to diverse robotic embodiments with non-standard optics"）；L101–103。

## 9. 主要图注/表注抄录

- **Figure 1**（L71–75）："ENACT casts embodied cognition evaluation as world modeling through egocentric interaction. Grounded in POMDP framework, ENACT considers two tasks from diverse activities and scenes: forward world modeling (ordering observations given actions) and inverse world modeling (ordering actions given observations). Evaluation shows that performance of VLMs drops as interaction horizons lengthen, performs better on inverse task, and lags behind humans."
- **Figure 2**（L146–151）："Overview of ENACT data curation pipeline. We first obtain aligned scene graphs (states) and RGB observations from a mobile manipulation dataset in a robotics simulation (BEHAVIOR). The trajectory is then segmented by identifying key-frames where an abstract state change occurs (i.e., the scene graph difference is non-empty). From this set of key-frames, we sample multiple key-frame trajectories, which are used to construct the forward and inverse world modeling VQA questions. Here N refers to the number of all sampled trajectories across all step lengths."
- **Figure 3**（L267–269）："Data sources and QA examples. ENACT is built from diverse, long-horizon activities performed by real robots (left). We provide examples for (mid) forward world modeling and (right) inverse world modeling. More QA examples and prompts are available in the Appendix A.3.3."
- **Table 1**（L324–326）："Evaluation on ENACT (Pairwise Accuracy). Dark gray indicates the best result within each category (Proprietary or Open-Weight Models), and Light gray denotes the second-best result within the category. Complete results are in Table 9 (Task Accuracy) and Table 10 (Pairwise Accuracy)."
- **Figure 4**（L357–359）："Real-World Evaluations. Key frames from the three real-world scenes used in our evaluation: kitchen, dinner table, and workspace. Together, these scenes contain diverse rigid, deformable, and articulated objects in diverse environments with varying lighting conditions."
- **Table 2**（L416–418）："Real-world evaluation of InternVL3.5-241B-A28B. Task and pairwise accuracies (%) on 960 QA pairs generated from real-world manipulation videos. We report performance across different interaction horizons for both forward and inverse world modeling."
- **Figure 5**（L443–448）："Evaluations on image realism and anthropocentric bias on human vision through ENACT. Heatmaps show two-tailed unpaired t-test results against the baseline, using Pairwise Accuracy. p < 0.05 is considered significant. Darker red means more significant. Δ is the performance change from the baseline. If significant and Δ < 0, the setting is worse than the baseline. C.2 reports the robot's performance on the left- and right-hand predicates, where Mixing is the proportion of ground truth left or right cases that are predicted as the other hand (i.e., mixing one hand into the other). ± means standard error."
- **Figure 6**（L548–549）："Error Distribution, broken down by forward and inverse tasks, evaluated on GPT-5."
- **Table 7**（L1431–1435）：认知构念映射——Forward：Action–Effect Reasoning ✓、Causal Inference ×、Affordance Recognition ✓、Embodied Awareness ✓、Temporal Abstraction ✓；Inverse：Action–Effect ×、Causal Inference ✓、Affordance Recognition ✓、Embodied Awareness ✓、Temporal Abstraction ✓。
