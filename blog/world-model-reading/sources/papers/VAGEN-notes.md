# VAGEN 精读笔记（arXiv:2510.16907，唯一数据源：auto_docs/papers/txt/VAGEN.txt）

## 1. 原文信息

- **标题**（L1–2）："V AGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents"
- **作者**（L3–5）：Kangrui Wang、Pingyue Zhang、Zihan Wang、Yaning Gao、Linjie Li、Qineng Wang、Hanyang Chen、Chi Wan、Yiping Lu、Zhengyuan Yang、Lijuan Wang、Ranjay Krishna、Jiajun Wu、Li Fei-Fei、Yejin Choi、Manling Li（通讯作者）。
- **机构**（L6）："1Northwestern University 2University of Washington 3Stanford University 4Microsoft 5University of Wisconsin-Madison 6University of Illinois Urbana-Champaign"
- **主页/邮箱**（L8–9）："http://mll.lab.northwestern.edu/V AGEN"；"kangrui.wang@northwestern.edu, manling.li@northwestern.edu"
- **发表状态**（L44–45）："39th Conference on Neural Information Processing Systems (NeurIPS 2025)." / "arXiv:2510.16907v1 [cs.AI] 19 Oct 2025"（即已录 NeurIPS 2025）。
- **贡献标注**（L42–43）："∗Equal contribution." / "†Corresponding author."

## 2. Abstract 全文逐字（L10–32）

> "A major challenge in training VLM agents, compared to LLM agents, is that states shift from simple texts to complex visual observations, which introduces partial observability and demands robust world modeling. We ask:can VLM agents build internal world models through explicit visual state reasoning?In this work, we architecturally enforce and reward VLM agent's reasoning process via reinforcement learning (RL), formulating the problem as a Partially Observable Markov Decision Process (POMDP). We demonstrate that structuring agent's reasoning into StateEstimation ("what is the current state?") and TransitionModeling ("what is next?") is critical by studying five reasoning strategies. Investigating how agents should ground visual states and represent these internal beliefs, we reveal the optimal representations are task-dependent:Natural Languageexcels at capturing semantic relationships for general tasks, whileStructuredformats are essential for high-precision manipulation. These insights motivate our approach to reward shaping and credit assignment. We leverage a WorldModeling Reward to densely rewards the agent's turn-by-turn state predictions, while our Bi-Level General Advantage Estimation (Bi-Level GAE) enables turn-aware credit assignment. Through such world model reasoning, we enable a 3B model to achieve performance of 0.82 on a set of five diverse agent tasks, nearly 3× improvement over its untrained counterpart (0.21) and surpassing proprietary reasoning models like GPT-5 (0.75), Gemini 2.5 Pro (0.67) and Claude 4.5 (0.62). All experiments are supported by our V AGEN framework, a scalable system for training and analyzing multi-turn VLM agents across diverse visual environments."

（注：txt 中斜体词与相邻词粘连，如 "We ask:can"、"StateEstimation"、"TransitionModeling"、"Natural Language"、"Structured"，均照录。另注意摘要称 Claude 4.5 为 0.62，而 Table 2 中 Claude 4.5 Sonnet Overall 为 0.64，差异如实记录。）

## 3. 问题与动机（VLM 智能体为何需要显式世界建模推理）

核心困难在于视觉状态的部分可观测性：训练 VLM 智能体比 LLM 智能体更难，因为 "states shift from simple texts to complex visual observations, which introduces partial observability and demands robust world modeling"（L11–13）。正文进一步解释：多轮智能体任务的核心挑战是准确解读和追踪动态环境，视觉状态 "often are partial and noisyObservations, fundamentally reframing the problem from an Markov Decision Process (MDP) to a more challenging Partially Observable Markov Decision Process (POMDP)"（L36–39）。在 POMDP 中，"the agent is no longer simply acting, but first estimating the true state of the world from observations"（L39–40）。作者把「所见」与「所需知」之间的桥梁定义为内部世界模型："The bridge connectingwhat an agent seesto what it needs to knowis the central focus of our work: the internal World Model [1]. It functions as a cognitive engine, maintaining aninternal beliefover the true state based on visual observations."（L40–41, L50）。动机缺口："current approaches in such multi-turn agentic tasks often lack explicit internal world modeling to strengthen visual state reasoning"（L52–53），于是提出问题 "How can we effectively teach VLMs to build internal world models through explicit visual state reasoning?"（L53–54）。解法思路："we reward VLMs via reinforcement learning (RL) to encourage their thinking process to be structured into StateEstimation (describing the current visual state) and TransitionModeling (predicting next state), which are the essential two components of WorldModeling [1]"（L55–58）。

## 4. 方法逐机制

### 4.1 POMDP 形式化（L127–135）

"the agent produces an action at ∈ A... the environment transitions from state st to a new state st+1 according to the state transition function P(s t+1|st, at) and emits a scalar reward rt =R(s t, at)"（L130–132）；观测为部分视图："a new observation ot+1 ∈ O, which is a partial view of the new state, sampled from Ω(·|st+1)"（L132–133）。目标："learn a policy πθ that maximizes the expected cumulative discounted return over a trajectory,maxθ Eπθ,P,Ω [Pt γtrt]"（L133–134）。附录 A.1 给出选 POMDP 的两理由：环境固有部分可观测（如 Navigation 的第一人称视角，需主动旋转/移动探索）、以及 "Methodological Congruence with POMDPs"（teacher-forcing 全轨迹优化）（L1186–1199）。

### 4.2 State Estimation + Transition Modeling 分解（L156–167）

"we explicitly reason about visual states to build aninternal beliefof the environment"（L156–157）；"the agent should interpret what the current visual observation actually represents and predict future state changes"（L159–160）。分解为：state model 解读现在（StateEstimation）、transition model 预测未来（TransitionModeling）："including a state model that interprets the present ( StateEstimation) and a transition model that predicts the future (TransitionModeling)"（L161–163）。"StateEstimation grounds visual observations ot into a state belief ˆst that approximates the hidden true state st"（L163–164）；"TransitionModeling reasons over the belief of the potential next state ˆst+1, allowing the agent to simulate internally how its action at turn t will transition the states. This predictive step is crucial for planning of a multi-turn trajectory."（L165–167）。

### 4.3 五种推理策略（L171–191）

1. **NoThink**："generate only an executable action aet, and the output action tokena t is<think></think><answer>a et </answer>, zt =∅"（L171–173）
2. **FreeThink**："produce any form of natural language reasoning, allowing visual state reasoning to emerge without a predefined structure...<think>z t </think> <answer>aet</answer>, zt ̸=∅,andz t is natural language tokens"（L174–177）
3. **StateEstimation**："explicitly verbalize the current state belief ˆst given the visual observation ot...<think><observation>ˆst </observation> <reasoning>ˆat </reasoning> </think> <answer>a et</answer>, zt =⟨ˆst,ˆat⟩,learning to approximateˆs t →s t"（L178–182）
4. **TransitionModeling**："explicitly simulate in its internal belief space about the next state ˆst+1...<think><reasoning>ˆat </reasoning><prediction>ˆst+1 </prediction> </think> <answer>a et</answer>, zt =⟨ˆat,ˆst+1⟩,learning to approximateˆs t+1 →s t+1"（L183–187）
5. **WorldModeling**："required to both describe the current state and predict the next state: <think><observation>ˆst </observation> <reasoning>ˆat </reasoning><prediction>ˆst+1 </prediction> </think> <answer>a et</answer>, zt =⟨ˆst,ˆat,ˆst+1⟩,learning to approximateˆs t →s t,ˆst+1 →s t+1"（L188–191）

格式约束：为防止格式违规，"we incorporate a format reward rformatt during training, following the strategy of DeepSeek-R1 [20]"（L202–203）。

### 4.4 训练目标与 VAGEN-Base 的 Token-Level GAE（L205–237）

PPO 目标（L209–216）："J PPO(θ) = 1/P i M lossi · X i M lossi ·min (u i(θ)Ai,clip(u i(θ),1−ε,1 +ε)A i), where M lossi is a mask that is 1 for action tokens and 0 for observation tokens"；Critic 最小化 "the squared error between its predictions and the target valuesY i"（L218）。Base 版信用分配："we use Token-Level Generalized Advantage Estimation (GAE) [22]"（L225）；KL 惩罚 β>0；"For all intermediate action tokens, reward ri is set to the KL penalty. At the final action token I, reward rI is set as the sum of KL penalty and total trajectory returnR(τ) = PT−1t=0 R(st, at)"（L227–229）。TD 误差与优势："δi =r i +γV ϕ(¯τ<j)−V ϕ(¯τ<i)；Ai =δ i +γλA j"（L232–233），"This recursion is initialized at the end of the sequence with AI =δ I"（L234）。

### 4.5 WorldModeling Reward（L388–410）

"we extract <observation> and <prediction> fields from the agent's response, compare them with the ground-truth visual states, and give a reward based on the matching score"（L390–392）。CLIP 被否："we found CLIP to be insufficiently sensitive to fine-grained spatial and geometric details, rendering the resulting reward signals unreliable"（L393–395）。改用 LLM-as-a-Judge + 文本真值：如 Sokoban 取 "2D positions of the player, boxes, and targets"、Navigation 取 "relative distances and directions from objects to the player"（L397–400）。奖励定义（L406–408）：

> "rreasont =β s ·I StateEstimation(ˆst, st) +β w ·I TransitionModeling(ˆst+1, st+1), where I is a generalized matching score (binary from direct judgment or a continuous, normalized score from rule-based metrics), and βs, βw are reward coefficients."

### 4.6 Bi-Level GAE（L413–443）

动机：Base 框架 "by aggregating all task rewards at the final token of a trajectory, it provides only trajectory-level feedback... making it difficult to assign credit for turn-specific successes or failures"（L414–417）。机制：两级，引入 "a turn-level discount factor γturn for transitions between turns and a token-level discount factor γtoken for tokens within a single action"（L420–422）。Turn 级 TD 误差："δturnt =r t +γ turnVϕ(¯τ≤at+1 )−V ϕ(¯τ≤at )"（L427）；turn 级优势："Aturnt =δ turnt +γ turnλturnAturnt+1"，"initialized with AturnT−1 =δ turnT−1 "（L431–432）。Token 级："δtokent,i =r KLt,i +γ tokenVϕ(¯τt,<i+1)−V ϕ(¯τt,<i)；Atokent,i =δ tokent,i +γ tokenλtokenAtokent,i+1"（L438–439）。两级连接点："the backward pass for the token-level advantages is initialized by setting the advantage of thefinal tokenof action at to the pre-computed turn-level advantage, Aturnt "（L440–442）。

### 4.7 VAGEN-Full 合成奖励（L481–498）

"rt =r reasont +r formatt +R(s t, at), where rreasont is a reward for the quality of the world model reasoning for visual states (ˆst and ˆst+1), rformatt is the reward for adhering to the specified output structure (Section 2.4), and R(st, at) is the sparse, task-specific reward from the environment."（L484–489）。配置："V AGEN-Full builds on this and incorporates World Modeling Reward and Bi-Level GAE, with the reward coefficients βs and βw set to 0.5. For SVG Reconstruction, only Bi-Level GAE is applied."（L494–496）。SVG 特例理由："Since SVG Reconstruction is a multi-turn reasoning task without world dynamics, we can only apply Bi-Level GAE configuration"（L496–497）。

## 5. 实验与主结果

### 5.1 环境与奖励设置（L240–299）

五环境："Classic Grid Puzzles(Sokoban and FrozenLake),Embodied 3D Navigation(Navigation), Detailed Object Manipulation(PrimitiveSkill), andAbstract Geometric Reconstruction(SVG Reconstruction)"（L244–246）。FrozenLake："We disable the "slippery" setting for determinism"（L284）。SVG 奖励："a dense similarity between the generated image Igen and the target Itarget, computed as a weighted combination of DreamSim and DINO scores"（L292–294）；其他任务 "R(st, at) is a scaled binary reward (e.g., {0,10} )"（L295–296）。

### 5.2 主表 Table 2（L256–282，全部数值照录，不四舍五入）

| Model/Method | Sokoban | FrozenLake | Navigation(Base/Common/Avg) | PrimitiveSkill(Place/Stack/Drawer/Align/Avg) | SVG(Dino/DreamSim/Avg) | Overall |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-72B | 0.18 | 0.44 | 0.72/0.75/0.73 | 1.00/0.50/0.00/1.00/0.44 | 0.89/0.62/0.76 | 0.51 |
| Qwen2.5-VL-7B | 0.13 | 0.14 | 0.28/0.39/0.34 | 0.00/0.00/0.00/0.75/0.19 | 0.83/0.28/0.55 | 0.27 |
| Qwen2.5-VL-3B | 0.14 | 0.14 | 0.22/0.27/0.24 | 0.00/0.00/0.00/0.00/0.00 | 0.80/0.27/0.54 | 0.21 |
| VLM-R1-3B | 0.13 | 0.13 | 0.31/0.34/0.33 | 0.00/0.00/0.00/0.00/0.00 | 0.81/0.28/0.55 | 0.23 |
| FreeThink | 0.57 | 0.68 | 0.67/0.67/0.67 | 1.00/0.63/0.00/1.00/0.66 | 0.91/0.64/0.78 | 0.67 |
| NoThink | 0.57 | 0.09 | 0.00/0.00/0.00 | 0.00/0.00/0.00/0.00/0.00 | 0.89/0.62/0.76 | 0.28 |
| StateEstimation | 0.56 | 0.68 | 0.78/0.69/0.74 | 0.00/0.00/0.00/0.00/0.00 | 0.92/0.64/0.78 | 0.56 |
| TransitionModeling | 0.41 | 0.76 | 0.67/0.59/0.62 | 1.00/0.63/0.63/1.00/0.82 | 0.89/0.64/0.77 | 0.68 |
| WorldModeling | 0.61 | 0.71 | 0.78/0.80/0.79 | 1.00/0.88/0.88/0.88/0.91 | 0.90/0.65/0.78 | 0.76 |
| VAGEN-Full | 0.79 | 0.74 | 0.80/0.81/0.81 | 1.00/0.88/1.00/1.00/0.97 | 0.91/0.67/0.79 | 0.82 |
| Vanilla-PPO | 0.18 | 0.21 | 0.32/0.25/0.29 | 0.00/0.00/0.00/0.00/0.00 | 0.83/0.44/0.64 | 0.26 |
| GRPO w/ Mask | 0.20 | 0.57 | 0.88/0.81/0.85 | 0.00/0.00/0.00/1.00/0.25 | 0.92/0.66/0.79 | 0.54 |
| Turn-PPO w/ Mask | 0.38 | 0.70 | 0.78/0.84/0.81 | 0.00/0.00/0.00/1.00/0.25 | 0.89/0.64/0.77 | 0.55 |
| GPT-5 | 0.70 | 0.77 | 0.75/0.81/0.78 | 1.00/0.63/0.00/1.00/0.66 | 0.95/0.75/0.85 | 0.75 |
| o3 | 0.60 | 0.78 | 0.81/0.75/0.78 | 1.00/0.63/0.00/1.00/0.66 | 0.92/0.71/0.82 | 0.73 |
| o4-mini | 0.44 | 0.82 | 0.75/0.75/0.75 | 1.00/0.50/0.00/0.75/0.56 | 0.90/0.66/0.78 | 0.67 |
| GPT-4o | 0.43 | 0.54 | 0.75/0.69/0.72 | 0.50/0.63/0.00/0.88/0.50 | 0.91/0.69/0.80 | 0.60 |
| Gemini 2.5 Pro | 0.58 | 0.78 | 0.63/0.63/0.63 | 0.63/0.63/0.00/0.75/0.50 | 0.93/0.78/0.86 | 0.67 |
| Gemini 2.0 | 0.28 | 0.61 | 0.50/0.63/0.56 | 0.75/0.13/0.00/0.25/0.28 | 0.93/0.74/0.84 | 0.51 |
| Claude 4.5 Sonnet | 0.31 | 0.80 | 0.67/0.67/0.67 | 0.63/0.50/0.00/1.00/0.53 | 0.95/0.81/0.88 | 0.64 |
| Claude 3.7 Sonnet | 0.25 | 0.69 | 0.48/0.47/0.47 | 0.63/0.13/0.00/1.00/0.44 | 0.94/0.77/0.85 | 0.54 |

表注（L250–255）："Reasoning on internal world models given visual states, including both StateEstimation and TransitionModeling, can significantly improve the RL performance... We gray out results without controlled variables (i.e., same size and architecture), shown for reference only. V AGEN-Full... will be introduced in Sec 4 so we also gray it out, shown as a reference."（原文注明灰色行为参考，故上表 3B 同规格对比以 WorldModeling 为基线。）Gemini 2.5 Pro 脚注（L303–304）："Due to Gemini 2.5 Pro's safety policy, a subset of evaluation responses for PrimitiveSkill and Navigation was blocked; reported results are based solely on the unblocked test cases."

### 5.3 主要结论（L300–357）

- 开箱即用模型：共评测 7 个模型（"We benchmark 7 models including proprietary models and open-source models in Table 2"，L301）；"most models struggle on these tasks, with the best-performing GPT-5 reaching 0.75 out of 1 in overall score. Particularly, no model succeeds on the PrimitiveSkill Drawer task."（L328–329）。VLM-R1 无多轮优势："VLMs that are trained to solve non-agentic tasks, such as VLM-R1, does not show multi-turn agent task advantage"（L326–327）。
- 策略对比："the full reasoning strategy, WorldModeling, achieves an overall performance of 0.76, yielding better results thanFreeThink(0.67) while clearly surpassingNoThink(0.28)"（L65–67）；"StateEstimation performs well in Navigation tasks, where understanding current observations is the key. In contrast, TransitionModeling achieves strong results in PrimitiveSkill, where predicting future states is crucial for manipulation."（L336–339）。WorldModeling "achieves a substantial improvement over its untrained counterpart (+0.55), and even outperforms all proprietary models despite smaller scale"（L343–344）。
- RL 基线失败原因（L347–357）："Vanilla PPO [11, 25] fails due to lack of observation token masking"（L348）；GRPO "remains insufficient due to high trajectory diversity from scene change, requiring unaffordable sample sizes"（L353–354）；Turn-level PPO "underperforms because uniform advantage estimates for action tokens within a turn cannot capture individual token contributions to policy performance"（L355–356）。

### 5.4 表征实验（L358–382；Table 4, L319–325）

"we consider three representations: Natural-Lanaguage, Symbolic, and Structured format"（L360–361）。Table 4 照录：

| Visual State Representation | Sokoban | FrozenLake | PrimitiveSkill Place/Stack/Drawer/Align/Avg |
|---|---|---|---|
| Natural-Lanaguage | 0.61 | 0.71 | 1.00/0.88/0.88/0.88/0.91 |
| Structured | 0.28 | 0.63 | 1.00/0.88/0.88/1.00/0.94 |
| Symbolic | 0.49 | 0.49 | –/–/–/–/– |

结论："the optimal format varies with the task nature"（L372）；FrozenLake/Sokoban 上 NL 最好，因为 "the model lack sufficient prior knowledge to interpret symbolic layouts effectively, and structured outputs, when derived from image-only input, are noisy due to limited grounding capabilities"（L373–375）；PrimitiveSkill 上 Structured 略胜，因为 "we provided a structured object position list as prompts to the model, allowing the model to ground its understanding more precisely and facilitate more accurate next state prediction"（L378–380）。后续默认："we adopt Natural-Lanaguage as the default, general-purpose state representation, while specifically employing the Structured format for the PrimitiveSkill task"（L380–382）。

### 5.5 VAGEN-Full 主结果（Table 5, L475–480）

| Model/Method | Sokoban | FrozenLake | Navigation Avg | PrimitiveSkill Avg | SVG Avg | Overall |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B（未训练） | 0.14 | 0.14 | 0.24 | 0.00 | 0.54 | 0.21 |
| + VAGEN-Base | 0.61 | 0.71 | 0.79 | 0.91 | 0.78 | 0.76 |
| + VAGEN-Full | 0.79 | 0.74 | 0.81 | 0.97 | 0.79 | 0.82 |

（PrimitiveSkill 细分：Base 1.00/0.88/0.88/0.88/0.91，Full 1.00/0.88/1.00/1.00/0.97；SVG 细分：Base 0.90/0.65/0.78，Full 0.91/0.67/0.79；Navigation 细分：Base 0.78/0.80/0.79，Full 0.80/0.81/0.81——均 L479–480。）结论："V AGEN-Full achieves consistently better test-time performance across all tasks compared to V AGEN-Base... although both methods reach similar training accuracy (Figure 4), V AGEN-Full significantly outperforms V AGEN-Base on the test set. This suggests that StateEstimation and TransitionModeling improve the agent's ability to adapt to new scenes"（L499–504）。

### 5.6 附录补充数据

- 训练硬件/成本（L1755–1757）："servers equipped with 8×H100 GPUs, 104 CPUs, and 1.7TB of memory... each training session requiring approximately 4-8 hours"。
- 关键超参（L1759–1773）：Top-p 0.95、Temperature 0.7、"γtoken 1.0"、"KL Penalty Coefficient (β) 0.001"、"Actor Learning Rate 1e-6"、"Critic Learning Rate 1e-5"、"Train Batch Size 128"、"PPO Mini Batch Size 32"。
- Bi-Level GAE 普适性（Table 25, L2649–2653）：FrozenLake FreeThink 0.39 → FreeThink + Bi-Level GAE 0.62；Sokoban FreeThink 0.43 → 0.43。
- 跨模型族/尺寸（Table 26, L2655–2659）：Qwen2.5-VL-3B BASE 0.61/FULL 0.79；7B BASE 0.63/FULL 0.92；InternVL3-2B BASE 0.36/FULL 0.39。
- 训练效率（Table 27, L2660–2666）：FrozenLake 40 H100 GPU-hours、2.3×10⁷ judge tokens；Sokoban 40、2.3×10⁷；Navigation 30、6.0×10⁶；ManiSkill 30、4.6×10⁶；SVG 10、N/A。

## 6. 消融（L505–547，与图 4 对应）

三条结论原文："Bi-Level GAE aloneprovides significant but inconsistent gains. Its performance is highly sensitive to reward sparsity and accuracy, which can lead to training instability in environments lacking dense and accurate intermediate rewards."（L536–538）；"The WorldModeling Reward aloneconsistently improves upon the baseline by providing a crucial learning signal for visual understanding. However, its effectiveness is limited by the coarse, trajectory-level credit assignment of standard RL."（L540–542）；"V AGEN-Fullis the most stable among all methods and performs generally well on all tasks."（L543）。总结："fine-grained credit assignment (from Bi-Level GAE) and high-quality reasoning supervision (from the WorldModeling Reward) are both essential to effectively improve VLM reasoning"（L544–546）。案例研究发现（L548–566）：空间理解与规划增强；"Response Convergence and Reduced Exploration... a steady decrease in entropy as training progresses"（L556–557）；以及 "Reward Hacking and Over-optimization... agents learn to hack the reward. They develop generic, broadly applicable responses that satisfy the LLM-as-a-Judge's criteria without necessarily reflecting deep, state-specific reasoning... particularly evident in agents trained with Bi-Level GAE"（L561–565）。缓解（附录 D.2, L2612–2640）：结构化解析+F1 评分替代整体判断、重复惩罚——"we set the F1 score threshold to 0.7. If a response was identified as both repetitive and incorrect (F1 <0.7 ), a penalty of −0.1 was applied to the reward"（L2638–2639）。

## 7. 局限与未来工作（原文逐字，L890–902）

> "We introduce a multi-turn reinforcement learning framework that rewards the reasoning process to build an internal world model through explicit visual state reasoning over StateEstimation (grounding) and TransitionModeling (prediction). In this way, the VLM agent learn to explore the environment to understand its causal/transition dynamics, and update internal beliefs as multi-turn interactions unfold. To optimize an agent's world model reasoning, we propose turn-level WorldModeling Reward for a dense turn-level reward to evaluate the accuracy of the agent's internal state simulation against ground-truth; to solve the critical challenge of long-horizon credit assignment, we propose Bi-Level GAE to first computes the value of an entire turn's reasoning before propagating that credit precisely to the individual tokens. Our V AGEN framework significantly enhances task performance and visual reasoning quality for VLM in agentic tasks. Limitations include restricted model architecture and evaluation environments. Future work will include exploring additional VLM families and supervised finetuning for multi-turn visual understanding."

（即：局限=受限的模型架构与评测环境；未来=探索更多 VLM 家族、多轮视觉理解的 SFT。除此之外原文未写其他局限。）

## 8. 叙事关联（【推断】，5 条，均附支持原句）

1. **【推断】VAGEN 就是「慢脑层=状态估计+转移预测」内禀世界模型的 RL 化实现**：它把慢脑的两个构件变成可被 PPO 优化的显式 token 结构，证明「世界模型」可以不是隐式网络状态，而是可审计、可奖励的语言化内部信念。支持原句：L55–58（"we reward VLMs via reinforcement learning (RL) to encourage their thinking process to be structured into StateEstimation... and TransitionModeling..."）；L156–157（"we explicitly reason about visual states to build aninternal beliefof the environment"）。
2. **【推断】「显式分解（状态估计+转移预测）> 自由思考 > 不思考」的排序，为叙事线主张的「慢脑必须有结构」提供了 RL 侧证据**：WorldModeling 0.76 > FreeThink 0.67 > NoThink 0.28（L65–67），且 "+0.55" 提升（L343–344），说明强制结构化而非放任模型自由推理才是关键。支持原句：L63–67；L342–345。
3. **【推断】「自然语言 vs 结构化表征」的任务依赖结论，直接支持叙事线「表征选择取决于任务精度需求」**：语义/通用任务用自然语言，高精度操作（机器人坐标）必须结构化；符号表征因 grounding 问题最差，说明抽象层级不能脱离视觉 grounding 而单方面拔高。支持原句：L70–81（"the choice of representation is not universal but is dictated by the task's demands... structured formatsproviding exact coordinates is essential... symbolic... it proves to be the least effective method in our experiments, creating a grounding problem"）；L371–382。
4. **【推断】「稠密 WorldModeling Reward + Bi-Level GAE」是把慢脑推理变成可优化信号的工程配方**：逐轮稠密奖励给状态/转移预测提供监督，分层优势估计解决长程信用分配——叙事线若要把「状态估计+转移预测」做成训练目标，需同时回答「奖励给谁（逐轮）、信用怎么分（turn→token）」两个问题。支持原句：L84–88（"a turn-level WorldModeling Reward, a dense reward... evaluates the accuracy of the agent's explicit state descriptions and predictions against ground-truth states"）；L114–117（"we first compute advantages at the turn level... then propagate these signals to the token level"）。
5. **【推断】风险警示：语言化的世界模型奖励会被 reward hacking，叙事线需内置防过优化**：Bi-Level GAE 训练出的模型倾向生成满足 LLM 裁判的通用模板而非状态特定推理，缓解手段是结构化解析+F1 阈值 0.7 的重复惩罚——意味着「以裁判模型奖励推理质量」的慢脑方案必须配套结构化评估与低熵惩罚。支持原句：L561–565；L2638–2640。

## 9. 图注/表注抄录（主要）

- **Figure 1**（L46–49）："Overview of the V AGEN Framework. Within a POMDP, VLM Agents build internal world models through explicit visual state reasoning over the StateEstimationP(ˆst|ot) and TransitionModeling P(ˆst+1|ot,ˆst,ˆat), where ˆst denotes the internal belief of the state, ˆat represents the internal belief of the action, anda et is the parsed executable action, as detailed in Table 1."
- **Figure 2**（L201）："Examples of visual states from five environments used in our study."
- **Figure 3**（L469–473）："Token-Level GAE and Bi-Level GAE frameworks. Standard Token-Level GAE, where a single, sparse reward at the end of the trajectory is propagated backward token-by-token. Our proposed Bi-Level GAE assigns rewards at each turn. It first computes a turn-level advantage (purple arrows) before propagating that credit to the individual tokens within the turn (orange arrows), allowing for robust, hierarchical estimation for credit assignment."
- **Figure 4**（L532–533）："Training success rate for V AGEN-Base, V AGEN-Full and ablations. Since SVG reconstruction is a multi-turn reasoning task without world dynamics, we only apply the Bi-Level GAE configuration."
- **Figure 5**（L877–880）："Comparison of V AGEN-Base and V AGEN-Full across Navigation, Sokoban, and PrimitiveSkill environments, from left to right... The improvement of V AGEN-Full shows enhanced reasoning ability in understanding environments such as their spatial information."
- **Table 1**（L92）："Summary of important notations used in the V AGEN framework."（关键记号：o_t 原始视觉观测、s_t 真值状态、ˆs_t 信念状态、a_t=⟨z_t, a_et⟩、r_t=R(s_t,a_t) 等，L93–113）
- **Table 3**（L306–307）："Examples of how visual states are converted into internal beliefs using natural language, symbolic and structured representations."
- **Table 6**（L1275）："Reasoning strategy formats."
- **Figure 7**（L1273–1274）："Overview of the V AGEN Framework. The framework operates in a dual-loop structure: an inner Rollout Loop for data collection and an outer Training Loop for policy optimization."
- 框架实现（L1225–1228）："we developed V AGEN, a framework built upon VeRL[80]... We adoptenv-as-servicedesign, which decouples training from interacting with environments, enhances scalability."

---

*笔记完。全部英文引文逐字复制自 C:\Users\Admin\Desktop\design\auto_docs\papers\txt\VAGEN.txt，行号标注于括号内；斜体粘连照录原文。*
