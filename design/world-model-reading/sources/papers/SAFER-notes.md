# SAFER 精读笔记（arXiv:2611.20107v1）

> **引用说明**：全部英文引用逐字复制自 `auto_docs/papers/txt/SAFER.txt`，行号按该文件。txt 为 PDF 提取产物，保留其原始瑕疵：粗体词丢失空格（如 "We proposeSAFER"、"0.95and0.77"）、下标带空格（如 "a t"、 "o t"）、公式处含控制字符（U+0002，显示为空）。除注明「公式转写」外，引用不做任何修正。文中标注【推断】处为我个人的解读，非论文原文。

---

## 1. 原文信息

- **标题**（L1–2）：
  > SAFER: Navigating the Exploration–Exploitation Dilemma in World-Model-Based Planning for Autonomous Driving
- **作者**（L3）：
  > Yansong Qua, Zilin Huangb, Zihao Shengb, Sikai Chenb,∗, Samuel Labia
- **机构**（L4–5）：
  > aLyles School of Civil and Construction Engineering, Purdue University, West Lafayette 47907, USA
  > bDepartment of Civil and Environmental Engineering, University of Wisconsin–Madison, Madison 53706, USA
- **通讯作者**（L48–50）：
  > ∗Corresponding author.
  > Email addresses:qu120@purdue.edu(Yansong Qu),zilin.huang@wisc.edu(Zilin Huang),zihao.sheng@wisc.edu(Zihao Sheng),sikai.chen@wisc.edu(Sikai Chen),labi@purdue.edu(Samuel Labi)
- **ID / 发表状态**（L51）：
  > arXiv:2611.20107v1  [cs.RO]  15 Nov 2026
  → arXiv 预印本 v1（2026-11-15，cs.RO）。是否被会议/期刊接收：**原文未写**。
- **项目主页**（L20–21）：
  > Project page: https://ys-qu.github.io/safer-website.
- **Keywords**（L22–23）：
  > Keywords:autonomous driving, world-model-based planning, exploration–exploitation dilemma, risk-aware planning, online adaptation, learning from failures, CARLA
- **名字来源**（L141–142）：
  > We let the world model learn instead from failures that others make, judged by an expert. We call the resulting framework SAFER, after its five stages, which Fig. 1c groups into three parts.
  五个阶段名即方法总览中的五个粗体词（L286–298）：**Stage、Annotate、Foresee、Evaluate、Refine**，取首字母 S-A-F-E-R。
- **资助**（L1624–1627）：
  > This work was supported by Purdue University's Center for Connected and Automated Transportation (CCAT) which is a part of the USDOT Region 5 University Transportation Center funded by the USDOT Award #69A3551747105. Also, general support was provided by Purdue's Next-generation Transportation Systems (NEXTRANS) Center...

---

## 2. Abstract 全文逐字（L6–23）

> 6: Abstract
> 7: An autonomous vehicle must foresee the consequences of its actions, especially in the rare, time-critical situations
> 8: where crashes happen. A world model offers this foresight, but only within the coverage of its training data, which
> 9: poses an exploration–exploitation dilemma. Within this coverage a planner can hardly improve on the data, and
> 10: beyond it the model's errors mislead it. Logs of competent driving rarely contain the failures that would widen the
> 11: coverage, and agents that learn from interaction gain them only by failing themselves, without being shown what to do
> 12: instead. We proposeSAFER , a framework that learns offline from failures that others make, judged by an expert, and
> 13: keeps learning online from its own driving. Offline, partially trained agents and scripted, recoverable perturbations
> 14: Stage failures in simulation, and an expertAnnotates every visited state with its action and value, so that much of what
> 15: the planner explores is covered before it drives. In planning, a chunked latent world modelForesees the consequences
> 16: fast enough for real-time use, and a risk-aware sampling-based plannerEvaluates candidate maneuvers by their return
> 17: minus a bounded collision penalty. Online, a loop continuouslyR efines the model wherever this driving leaves the
> 18: coverage of the offline data. In closed-loop evaluation against the compared methods, SAFER reaches the highest
> 19: route completion and reward per step on the CARLA Leaderboard routes,0.95and0.77from a privileged bird's-eye
> 20: view, and the highest episode reward in MetaDrive,53.4in regular and49.0in safety-critical traffic. Project page:
> 21: https://ys-qu.github.io/safer-website.

---

## 3. 问题与动机（简）

- **coverage 定义**（L35–36）：
  > A world model, however, can only foresee what its training data contain; we call the states and actions that these data represent itscoverage.
- **探索/利用两难**（L37–38）：
  > The dilemma is the choice between exploiting known options, whose outcomes are predictable, and exploring new ones, which may pay off in the long run but whose outcomes are uncertain
- **coverage 内难超越数据**（L39–40）：
  > A planner that exploits what the model knows, choosing among the actions that the data cover, can rely on the predictions but can hardly do better than the driving in the data.
- **coverage 外模型误差误导**（L40–44）：
  > To do better, it can explore actions that the data seldom contain, such as braking later or pulling out earlier than competent drivers do, and the states that its own decisions lead to. There the predictions are unreliable, and because the planner selects the candidate with the highest predicted value, it is drawn to exactly the candidates whose value the model overestimates
- **纠正误差所需的经验 = 失败经验，公路上不可得**（L45–47）：
  > Gathering such experience is the role of exploration in reinforcement learning (Curi et al., 2020; Sekar et al., 2020), but in driving the experience beyond competent driving is the experience of failure, which cannot be gathered on public roads.
- **现有路线的缺陷**（L128–140，节选）：
  > Logs and demonstrations contain little of it. World models trained on logs of human or expert driving... see what competent drivers do and rarely what follows from a mistake of the ego vehicle... behavior cloning on such data does not explore at all, and an expert who never makes mistakes cannot show how to recover... These failures are the learner's own. The learner has to cause them, over many trials that would be unacceptable on public roads, and each failure teaches it only the outcome, a collision or a low reward, not what should have been done instead.
- **形式化陈述**（L755–776）：Definition 1（model error）、Proposition 1（optimizer's curse：即使模型在每个固定候选上无偏，最大化后仍系统性高估所选候选）、Proposition 2（regret 由两个候选上的误差界住）：
  > Remark1.Proposition 1 is the optimizer's curse (Smith and Winkler, 2006). Even a model that is unbiased at every candidate, once maximized, overestimates the candidate it selects, by at least the expected regret of that selection. Proposition 2 shows that the regret is bounded by the errors at only two candidates, the best one and the chosen one.
  > MaximizingJthus draws ˆatowards candidates whose value the model overestimates, typically candidates that the corpus does not cover. Staging failures and annotating them with the expert (Section 3.2) reduces the error on the candidates that competent and failing drivers take, and online refinement (Section 3.6) reduces it where the planner's own choices lead.
- **育儿类比（作者动机叙事）**（L150–155）：
  > SAFER is inspired by how people resolve the same dilemma as they grow up (Fig. 2, top). A child is not left to discover the consequences of every choice alone. Parents praise what works and point to negative examples, so that the child learns what leads where without having to fail first... Later, school and work keep reshaping these beliefs in situations the parents never showed, while the core ones stay. Such learning does need failures, but failures that others make, where they can be afforded, judged by someone who knows what should have been done.

---

## 4. 实现细节（重点逐项）

### 4.0 方法总览（四部分，对应五个阶段）

（L281–299，Fig. 3 caption 对应 L388–396）
> This section describes SAFER. At every control step, the planner imagines the consequences of candidate action sequences with a learned latent world model, searches for the sequence with the best predicted return, penalized by the predicted probability that the sequence leads to a collision, and executes its first action. The five stages of SAFER map onto four parts, the four panels of Fig. 3:
> (a)StageandAnnotate: a corpus ofmixed-qualitysimulated experience, driven by a reinforcement-learning (RL) expert, by weaker checkpoints of an RL agent trained with the expert's recipe and by scripted perturbations, in which every step is labeled with the expert's action and value (Section 3.2);
> (b)Foresee: achunked latent world modelthat encodes an observation, predicts the latent states of a whole chunk of the future under a candidate action sequence in one forward pass, and reads the imagined trajectory with a physically decomposed reward, a continuation probability, an ensemble of value heads and a behavior-cloned action prior (Section 3.3), together with acollision-risk headthat predicts the probability that a collision has occurred by each future step; the risk head is trained only on the simulator's recorded outcomes and regularized so that braking harder does not raise the predicted risk (Section 3.4);
> (c)Evaluate: arisk-aware sampling-based plannerthat runs model predictive path integral control (MPPI) on the world model, seeded with the behavior-cloned prior and penalized by the predicted collision risk (Section 3.5);
> (d)Refine: anonline adaptationloop that keeps the dynamics, the reward scorers and the risk head consistent with the planner's own driving, while the planner scores against slowly moving exponential-moving-average (EMA) copies of them (Section 3.6).

与 SimDist 的关系（L300–307）：
> The design follows the decomposition of Simulation Distillation (SimDist) (Levy et al., 2026): an observation encoder, a chunked latent dynamics model conditioned on an explicit history, reward and value heads that read the imagined trajectory, and a base policy that warm-starts a sampling-based planner... SAFER therefore predicts the reward through its physical factors and the value with an ensemble, prices the predicted collision probability in the planner's score with an event-based risk head trained to be monotonic in braking, and adapts the scorers online while the planner scores with their EMA copies.

### 4.1 问题形式化与两环境实例化（Section 3.1）

- **POMDP 设定**（L308–312）：
  > We formulate driving as a partially observable Markov decision process in discrete time,t= 0,1, . . ., with steps of fixed length. At steptthe agent receives an observationo t, a partial view of the scene that describes the surroundings of the ego vehicle and its own state, and applies an actiona t = (α t, δt)made of two commands in [−1,1]: an acceleration commandα t, applied as throttleα t when it is positive and as brake−α t when it is negative
- **动力学与事件**（L397–400）：
  > The scene then evolves under unknown, stochastic dynamics, which include the motion of the other road users, and the environment returns a rewardrt, a termination flagd t ∈ {0,1}and a vectoret ∈ {0,1}ne of safety events, which marks the safety-critical outcomes of the step, such as a collision.
- **奖励是已知物理因子的闭式函数**（L404–412）：
  > We assume that the reward is a known function of quantities that the simulator records at every step, rt =f(ϕt,a t,a t−1), (1) where the reward factorsϕ t ∈R nϕ are physical quantities, such as the speed of the ego vehicle, or indicators of events, such as a contact, andfis given in closed form. The world model uses this structure by predicting the factors and evaluatingfon them (Section 3.3).
- **三种监督来源**（L413–418）：
  > The first is an expert policyπe with a value functionV e, both functions of the observation, so that they can be queried at any observation, including those of states that the expert would never reach itself; we writea et =π e(ot)for its deterministic action andv et =V e(ot)for its value at stept. The second is an offline corpusDof episodes driven by behavior policies that are allowed to fail, in which every step carries the expert's labels(a et , vet )(Section 3.2). The third is online interaction, in which the controller drives the environment itself and records(o t,a t, rt,ϕ t, dt,e t)but no expert labels, so that what it learns online does not depend on the expert.
- **滚动时域规划**（L421–429）：
  > At steptthe controller conditions on the lastCobservations and the actions between them,(o t−C+1:t ,a t−C+1:t−1 ), and searches over action chunksa t:t+H−1 = (at, . . . ,at+H−1 )∈[−1,1] H×2 ofHsteps for one that maximizes a planning objective J(a t:t+H−1 ); it executes the first action of the chunk and plans again at stept+ 1.
  > We useC=H= 16steps, i.e.1.6s of history and of horizon at the0.1s steps of both simulators.
- **CARLA 实例化**（L430–443）：
  > In CARLA (Dosovitskiy et al., 2017), steps are0.1s long ando t = (m t,s t)(Fig. 4c), wherem t, a15×192×192array with values in{0, . . . ,255}, is the privileged BEV mask stack of Roach (Zhang et al., 2021) ands t ∈R 6 is the ego state. The BEV covers a38.4 m×38.4 m window at5pixels per meter with the ego drawn40pixels from the bottom edge, and contains three static channels (drivable road, the planned route corridor, lane markings with broken markings at a lower intensity) and three dynamic channel groups (vehicles, pedestrians, and traffic-light stop lines together with stop signs) rasterized at four time offsets{0,0.5,1.0,1.5}s in the past, giving3 + 3×4 = 15channels. The dynamic groups therefore encode1.5s of motion history of every road user. The ego state iss t = (throttle,steer,brake,gear/5, vx, vy), i.e. the last applied control and the planar velocity. The safety events are a collision, a red-light violation and a stop-sign violation (ne = 3). The reward is the dense reward of Roach, whose factors are the ego speed, a desired speed that decreases towards zero as the ego approaches a vehicle, a pedestrian, a red light or a stop sign in its path, the lateral distance to the route, the heading error and the magnitude of the terminal penalty (nϕ = 5, Section 4.1);falso penalizes changes of the steering command.
- **MetaDrive 实例化**（L443–449）：
  > In MetaDrive (Li et al., 2023), on the benchmark of CAT (Zhang et al., 2023), steps are also 0.1s long ando t ∈[0,1] 101 is a vector (Fig. 4f) that holds distances along30lidar beams to other vehicles,30beams to the road edges and12beams to the lane lines,6values of the ego state,10navigation values, the lateral offset from the logged route and4values for each of the three nearest vehicles. The safety events are a contact with a vehicle, a contact with an object and leaving the road (n e = 3). CAT's reward is the progress along the route, except at the step of an arrival, an exit from the road or a contact, where it takes a fixed value; its factors are the progress and the indicators of these four events (nϕ = 5).

### 4.2 离线 Stage 与专家 Annotate（Section 3.2）

- **行为策略 = 部分训练 checkpoint**（L453–456）：
  > Besides the expert, the behavior policies areMcheckpointsπ 1, . . . , πM of a reinforcement-learning agent trained from scratch with the expert's recipe and saved at increasing numbers of training steps, so that the corpus contains the systematic mistakes of drivers at several levels of competence, such as late braking, lane departures and running red lights, together with their consequences.
- **扰动一：动作噪声（clean/corrupt 相位交替，SimDist 配方）**（L456–464）：
  > Two perturbations of the action abt proposed by a behavior policy make the driving worse in a controlled way. The first is action noise. An episode alternates between clean phases and corrupt phases of random lengths, and during a corrupt phase the executed action is at = clip(abt +ξ t,−1,1),ξ t ∼ N(0,diag(σ 2α, σ2δ )),(2) with severities(σ α, σδ)that are drawn at random and redrawn at random intervals, whilea t =a bt in a clean phase; a small share of the episodes stays clean throughout, and half of the others start in a corrupt phase, following SimDist's mixed-quality recipe.
  （Eq. (2) 公式转写：a_t = clip(a_t^b + ξ_t, −1, 1)，ξ_t ~ N(0, diag(σ_α², σ_δ²))）
- **扰动二：脚本事件（hold / late brake，可恢复）**（L465–470）：
  > The second is a scripted event. Triggered by the simulator state, it overrides the expert's longitudinal command for a few steps with the opposite decision, braking where the expert would drive on or keeping the throttle where the expert brakes for a hazard, and is followed by a recovery in which the expert's action is executed. An event that withholds braking is triggered only while the expert could still stop before the hazard, so that every staged mistake is recoverable. These events place the two situations that a planner most often gets wrong, waiting when it should go and going when it should brake, into the data together with their labeled corrections.
- **语料三分**（L471–473）：
  > The corpus is the union of three sets:D exp, driven by the expert without perturbation;D int, driven by the checkpoints with action noise; andD tgt, driven by the expert with action noise and scripted events.
- **不重置、记录后果**（L474–475）：
  > Episodes are not reset when a mistake happens; they run until the environment terminates them, so that the consequences of the mistakes, the safety events among them, are recorded.
- **每步记录 + 专家标注**（L476–478）：
  > Whichever policy drives, every step stores (ot,a t, rt,ϕ t, dt,e t,a et , vet ), (3) that is, the transition with its reward factors and safety events and, as labels, the expert's action and value at that state.
- **标签的用途（核心一句话）**（L479–480）：
  > The labels are what make mixed-quality data useful. A world model learns from the transitions what each mistake leads to, and a policy learns from the labels what should have been done at the states that the mistakes reach.
- **可迁移的最小前提**（L481–482）：
  > In this work the failures are staged in simulators; the recipe itself needs only a driver that is allowed to fail recoverably and an expert that can be queried at the visited states, so a proving ground with a safety driver would serve the same purpose.
- **与 DAgger 的关系**（L228–231）：
  > The corpus of SAFER (Section 3.2) takes DAgger one step further. The states are visited by drivers that are deliberately worse than the expert, partially trained checkpoints of an RL agent, and by perturbations of the ego's own actions that run until a consequence occurs, and the expert annotates each state with its action and value. A policy can learn recoveries from these labels, and a world model can, in addition, learn what the mistakes lead to.
- **CARLA 具体参数**（L912–918）：
  > We use CARLA (version 0.9.15; Dosovitskiy et al., 2017)... The expert is the publicly released Roach PPO agent (Zhang et al., 2021) after11.8M environment steps. Its actor provides the action labels aet , its critic the value labelsv et , and its feature extractor gives the architecture of the world-model encoder and initializes it. The intermediate drivers are theM= 13checkpoints of a Roach agent that we trained from scratch with Roach's code and hyper-parameters, saved between0.2and4M training steps, so that the bank spans a wide range of competence rather than one level of it.
- **收集范围与事件触发条件**（L919–926）：
  > The three sets of Section 3.2, summarized in Table 1, are collected in the four training towns of Roach's benchmark, Town01, Town03, Town04 and Town06 (Fig. 4a), on unbounded random routes with randomized traffic density; every checkpoint contributes the same number of transitions toDint... Aholdfires when the ego has been stopped and the reward's desired speed jumps up, as when a light turns green or a lead vehicle moves off, and holds full brake while the expert would already be starting. Alate brakefires when the expert brakes for a hazard in the ego's path and holds the previous throttle, but only if the expert could still stop before the hazard after the override.
- **终止规则与终端惩罚**（L926–929）：
  > An episode terminates at the first collision, red-light or stop-sign violation, or when the ego is blocked or leaves its lane, so every safety event ends its episode in this corpus. The reward subtracts a terminal penalty, its fifth factor, of1at any termination and1 +v t, with the ego speedv t in m/s, at a safety violation, so that a fast collision is penalized more than a slow one.
- **语料的关键统计性质**（L929–931）：
  > The corpus contains the effect that a planner must be able to foresee. Among the states in which the expert brakes for a hazard ahead, the collision rate within the next1.6s is roughly half as high when the executed action brakes as when it does not.
- **MetaDrive 语料**（L1046–1055）：
  > The offline corpus applies the recipe of Section 3.2 to the400training scenarios under CAT's training rules, in which a vehicle contact costs−1without ending the episode. The expert is a CAT agent, TD3 trained with CAT's closed-loop adversarial training and CAT's code, for106 steps, and eight of its checkpoints, saved between2.5×10 4 and7.5×10 5 steps, are the intermediate drivers... Every scenario is driven in its logged traffic and again against CAT's opponent... CAT's expert seldom brakes for hazards, so this event takes the place of the late brake of CARLA as the source of braking near hazards. The corpus holds44,800 episodes and3.2M transitions; the expert labels every state with its action, and the value labels are the noise-free expert's values, estimated by policy evaluation on the corpus with CAT's discount of0.99, which the planner also uses asγ.
- **CARLA 语料规模（Table 1，L998–1009 逐字）**：
  > Table 1: Offline corpus in CARLA. Steps are transitions at10Hz. Episodes end on the first safety violation; the intermediate policies therefore produce many short episodes. "Perturbed" counts steps whose executed action differs from the behavior policy's proposal. The lower block lists the sets that one turn of the data flywheel adds (Section 4.2); inD +tgt, scripted pedestrians step into the ego's path.
  > Set | Driver | Steps | Episodes | Collisions | Red lights | Perturbed
  > Dexp | expert | 10.00M | 1,901 | 1,669 | 133 | 0%
  > Dint | 13 intermediate checkpoints + noise | 10.05M | 52,023 | 28,629 | 10,006 | 21.6%
  > Dtgt | expert + noise + events (two runs) | 2.00M | 338 | 268 | 46 | 24.8%
  > total | | 22.06M | 54,262 | 30,566 | 10,185 |
  > added by one turn of the data flywheel
  > D+int | 13 intermediate checkpoints + noise | 4.00M | 20,931 | 11,439 | 4,000 | 21.8%
  > D+tgt | expert + noise + events, crossing pedestrians (two runs) | 2.50M | 2,640 | 2,551 | 54 | 22.0%
  > total | | 28.56M | 77,833 | 44,556 | 14,239 |

### 4.3 Chunked latent world model（Section 3.3，Foresee）

- **编码器（复用专家特征提取器）**（L484–488）：
  > An observation encoder maps every observation to a latent state, zt =E θ(ot)∈R 256.(4) It has the architecture of the expert's feature extractor, the part of the expert's network that processes the observation; it is initialized from that extractor (Section 4) and fine-tuned with the rest of the model. Initializing from the expert gives the world model, from the first step, the features that were sufficient to drive well.
- **上下文 token（每控制步一次，所有候选共享）**（L490–499）：
  > The context of the lastCobservations and the actions that led to them is turned into a set of2C context tokens, Mt =(uzi ,u ai ) C−1i=0 , uzi =z t−C+1+i +η z + PE(i),u ai =A θ(at−C+i ) +η a + PE(i), (5) whereA θ is a two-layer MLP action encoder,η z,η a are learned token-type embeddings andPE(i)is a sinusoidal position code. The action tokenu ai carries the actiona t−C+i that led toz t−C+1+i; the first one is set to zero, because the action that led to the first context observation lies outside the window. The context tokens depend on the history alone, not on the candidate, so they are computed once per control step and shared by all candidates.
  （Eq. (5) 公式转写：u^z_i = z_{t−C+1+i} + η_z + PE(i)，u^a_i = A_θ(a_{t−C+i}) + η_a + PE(i)）
- **一次性 chunk rollout（关键加速设计）**（L500–507）：
  > A candidate action chunka t:t+H−1 is encoded intoHquery tokensq k = Aθ(at+k) + PE(k),k= 0, . . . , H−1. A four-layer transformer decoderDθ (Vaswani et al., 2017) with width256 and four heads attends from the queries to the context tokens and among the queries with a causal mask, and predicts all future latents in one pass, (ˆzt+1, . . . ,ˆzt+H) =D θ(q0:H−1 ;M t),(6) where, here and below, a hat marks a quantity predicted by the model or computed from its predictions. Because of the causal mask, ˆzt+k+1 depends ona t:t+k only. Predicting the chunk jointly rather than by step-wise recursion makes the cost of scoring a candidate one decoder pass instead ofHsequential passes, which is what makes sampling-based planning affordable in real time; the causal mask preserves the semantics of an autoregressive model.
- **Scorer 结构（每 head 组前一个小 temporal transformer）**（L508–511）：
  > The scalar heads do not read the decoder states directly. Each group of heads is preceded by a small temporal transformer, a scorer that maps a sequence over theHslots to one feature vector per slot (a two-layer input MLP and one encoder layer without a causal mask), so that the prediction at slot kcan use the whole imagined trajectory.
- **Reward/value scorer 输入约定**（L511–515）：
  > The reward scorerT rθ reads, for every transition, the statebeforeit and the action taken, (f r0 , . . . ,frH−1 ) =T rθ ((z−t+k,a t+k)H−1k=0 ), z−t =z t,z −t+k = ˆzt+k (k≥1), (7) while the value scorerT vθ reads the stateafterit,(f v0 , . . . ,fvH−1 ) =T vθ (( ˆzt+k+1)H−1k=0 ).
- **分布头（symlog 101 bins）**（L515–523）：
  > The reward and value heads are distributional (Hafner et al., 2025a). Each outputs a probability vectorp∈∆ B, where∆ B is the set of probability vectors overB= 101bins whose centersc 1 <· · ·< cB are uniformly spaced insymlogspace,symlog(x) = sign(x) ln(1 +|x|), and the prediction is decoded asˆy= symexp(Pb pbcb) withsymexp(y) = sign(y)(e |y| −1); the reward bins span[−3,3]and the value bins[−10,10]insymlogunits.
- **四个每槽预测 + 因子头 + K=5 价值集成**（L524–535）：
  > Four quantities are predicted per slot, the reward and its factors from the reward features and the continuation and the values from the value features: ˆprt+k = softmax(Rθ(f rk ))∈∆ B,(8) ˆϕt+k =g ϕ(Φθ(f rk ))∈R nϕ ,(9) ˆct+k = sigmoid(Cθ(f vk )),(10) ˆpv,it+k+1 = softmax(V (i)θ (f vk )), i= 1, . . . , K.(11)
  > ˆpr is a distribution over reward bins; ˆϕpredicts the reward factors of Eq. (1) through an output mapg ϕ that is fixed for each factor. For a continuous quantity it multiplies the output by a fixed scale, after a softplus if the quantity is non-negative, and for an event indicator it applies a sigmoid, so that the prediction is a probability;ˆcis the probability that the episode continues after the transition; and{V (i)θ }Ki=1 areK= 5independently initialized value heads on the shared value trunk, in the spirit of the value ensembles of TD-MPC2 (Hansen et al., 2024).
- **因子化奖励（与直接奖励两条路径，规划时取平均）**（L535–544）：
  > The factor head allows the reward to be reconstructed from the predicted factors exactly as the simulator computes it, ˆrϕt+k =f( ˆϕt+k,a t+k,a t+k−1),(12)... Two reward estimates are thus available for every imagined step, a direct oneˆrst+k = symexp(Pb ˆprt+k,bcb) and a factorized oneˆrϕt+k. The planner uses their average (Section 3.5). The factorized path exists because the scalar reward compresses very different situations into one number, whereas the factors force the model to represent, for every candidate, the quantities from which the simulator computes the reward, such as how fast the vehicle will be going and how fast it should be going given the hazards ahead; these are the quantities on which the correct action depends.
- **BC action prior（规划器的 warm start / 消融臂）**（L546–555）：
  > A separate three-layer transformer decoderP θ attends fromHlearned queries to the same context tokensM t and outputs, for each future step, the parameters of a squashed Gaussian, πθ(at+k | Mt) :a t+k = tanh(xk), xk ∼ N(µπk ,diag((σ πk )2)),logσ πk ∈[−2,1], (13) fork= 0, . . . , H−1. Its deterministic chunkabct+k = tanh(µπk ), clipped entrywise to[−¯a,¯a]with¯a= 0.999, is the base policy that warm-starts the planner; executed on its own, it is the BC-prior arm of the ablation in Section 4.2. It reads only the context, never the candidate actions, so it costs one decoder pass per step regardless of the number of candidates.
- **训练窗口（32 步 + 终端前额外窗口）**（L556–562）：
  > The model is trained on windows ofC+H= 32consecutive steps sampled uniformly from the corpus. The firstCobservations form the context and the followingHtransitions the imagination targets, with the keyframetbeing the last context step. Windows that run into the end of an episode are padded after the terminal transition, and the padded steps are masked. When the risk head is trained on a corpus in which safety events end the episodes, windows are also started at every step before a terminal, so that each terminal event appears at every horizon position between1andH. Without this, a collision that ends an episode would appear only at the last position of a window, and the risk head would never see one at short range.
- **损失函数（Eq. 14–21，L563–627 节选）**：
  > Llat = ⟨ℓH( ˆzt+k+1,sg(E θ(ot+k+1)))⟩Ms ,(14) Lrew = ⟨CE(twohot(rt+k),ˆprt+k)⟩Mv ,(15) Lϕ = ⟨ℓϕ( ˆϕt+k,ϕ t+k)⟩Mv ,(16) Lcons = ⟨ℓH(ˆrϕt+k, rt+k)⟩Mv ,(17) Lcont = ⟨BCE(1−d t+k,ˆct+k)⟩Mv ,(18) Lval = 1K KXi=1 ⟨CE(twohot(vet+k+1),ˆpv,it+k+1)⟩Ms ,(19) Lbc = ⟨−logπ θ(aet+k | Mt)⟩Mv ,(20)
  （公式按 txt 转写；⟨x⟩M 为对 M_k=1 槽位的均值；ℓH Huber；twohot 将目标 symlog 后分配到两个 bin 中心）
- **三个值得注意的设计选择**（L609–619）：
  > Three choices in Eqs. (14)–(20) deserve comment. First, the latent target is the encoder's own output with a stop-gradient, so that the dynamics is trained to reproduce the representation rather than the representation being trained to be predictable. Second, the value heads regress the expert's valueV e evaluated at the recorded future observationso t+k+1, not a Monte-Carlo return of the behavior policy, because the corpus is driven mostly by degraded policies whose returns would measure their own mistakes, whereasV e measures how good a state is under competent driving, which is the quantity the planner needs for bootstrapping. Third, the behavior-cloning target is the expert's action at the visited state foreveryvalid step, including steps at which a degraded policy or a perturbation has taken the vehicle off the expert's trajectory. The prior therefore learns recoveries, not only expert behavior, unlike the base policy of SimDist (Levy et al., 2026), which is trained only on steps the expert executed. The consistency lossL cons ties the factor head to the scalar reward through the known reward formula, and the continuation loss provides the survival probabilities used by the planner.
- **RMS 归一化加权（免手调损失权重）**（L620–627）：
  > The losses have very different scales and would otherwise require hand-tuned weights that drift during training. Each term is therefore divided by a running estimate of its own root-mean-square, ˜Lj =L j/√νj, whereν j ← 0.99ν j + 0.01L 2j is updated at every training step and treated as a constant in the gradient, and the weights express relative importance only:
  > Lwm = ˜Llat + ˜Lrew + ˜Lϕ + 0.1 ˜Lcons + 0.1 ˜Lcont + ˜Lval + 4 ˜Lbc (+ ˜Lrisk). (21)
- **离线训练安排（CARLA）**（L932–937）：
  > Both offline stages train on the union of the three sets with uniform sampling, with no re-weighting between the expert, intermediate and targeted sets. Stage 1 trains the world model, the reward, continuation and value heads and the action prior for1.0M steps, together with the scene-factor head of Section 4.6 as an auxiliary task, whose loss is RMS-normalized like the others and added to Eq. (21) with weight0.5. Stage 2 adds the risk head, drops the scene-factor head and continues all parameters jointly for0.5M steps, with the additional windows before terminals of Section 3.3.
- **MetaDrive 训练**（L1057–1059）：
  > The world model is that of Section 3.3, with an observation encoder initialized from the first two layers of the expert's actor and CAT's reward components as reward factors; dynamics, heads, action prior and risk head are trained together in a single stage of2.5×10 5 steps.

### 4.4 Collision-risk head（Section 3.4）

- **为什么需要独立风险头（有界性动机）**（L630–637）：
  > The return in the planner's score, Eq. (28) below, already penalizes safety events, through the reward and the value bootstrap. A collision, however, enters the return only through the reward of a single imagined step, diluted by the discount and, when it ends the episode, by the survival probability, and its effect on the score depends on how well the reward head extrapolates to a rare event. A dedicated head that answers a simpler question,has a collision happened by imagined stepk, can be trained as a classifier with class re-weighting, is bounded, and yields a penalty whose scale is under direct control. Similar event heads are used by RaWMPC (Sun et al., 2026); our head differs in its cumulative target and in the monotonicity constraint below, and it is supervised only by the simulator's recorded safety events, never by any hand-written rule.
- **架构**（L638–642）：
  > The headρ θ is a three-layer MLP (LayerNorm, two hidden layers of width256with GELU and dropout0.1) applied to the concatenation of the keyframe latent and each imagined latent, ρk =ρ θ([zt; ˆzt+k+1])∈R ne , k= 0, . . . , H−1,(22) one logitρ k,j per safety-event typej= 1, . . . , ne of Section 3.1.
- **累积目标与掩码**（L643–650）：
  > The target is cumulative. With the recorded events et+k of the window and the validity maskM v of Section 3.3, yk,j = maxk′≤k(M vk′ et+k′,j), M rk,j = max(M vk , yk,j ).(23) When an event ends the episode, the steps after it are padding; the maskM rk,j keeps them supervised with target1 ("the event has happened"), so that the logit at every later slot is trained as well, whereas a channel without an event is not supervised on padding, where its outcome is unknown.
- **分类损失（正类重加权 w+ = 20）**（L650–661）：
  > The classification loss is a positively re-weighted binary cross-entropy averaged over the masked entries, Lbce = Pk,j M rk,j BCEw+(yk,j ,sigmoid(ρ k,j ))Pk,j M rk,j , w + = 20,(24) whereBCE w+ (y, p) =−w +ylogp−(1−y) log(1−p); the positive class weight compensates for the rarity of events among the transitions of a window.
- **刹车单调性正则（对抗「刹车↔危险」相关性）**（L662–682）：
  > Braking monotonicity.A classifier trained on logged data inherits its correlations. In driving data hard braking co-occurs with danger, and a head trained with Eq. (24) alone can learn that a candidate which brakes ismorelikely to be in a collision window, which is the opposite of what a planner needs. We counter this bias with a soft constraint that needs no counterfactual outcome label. For each window, a braked copy of the recorded action sequence,a−t+k = (clip(αt+k −β,−1,1), δ t+k)with a brake offsetβ∼ U[0.1,0.5]drawn once per window, is rolled out with the same context into latents ˆz−t+k+1, and the head is penalized whenever the braked rollout is predicted to be riskier:
  > Lmono = 1neH H−1Xk=0 neXj=1 relu(sigmoid(ρ−k,j )−sigmoid(ρk,j )), ρ−k =ρ θ([zt; ˆz−t+k+1]). (25)
  > The term is evaluated with dropout disabled, since two stochastic passes would differ even for identical inputs and the hinge would penalize that noise. Both latent inputs are detached, so that the term can only change the head; if it could reach the dynamics, the cheapest way to satisfy it would be to make the executed rollout look more dangerous. The complete risk objective isL risk =L bce +L mono, RMS-normalized like the other terms in Eq. (21). When the risk head is trained together with the world model, the classification term is allowed to shape the latents, so that the imagined states carry the information needed to foresee an event.
- **规划惩罚项（有界）**（L686–702）：
  > Planning penalty.For a candidateathe planner subtracts ¯ρ(a) =1H H−1Xk=0 neXj=1 ωj sigmoid(ρk,j (a)),(26) whereρ k,j (a)are the logits of Eq. (22) on the rollout ofaand the non-negative weightsω∈R ne select the events that the planner prices; whenωselects the collision channel alone,¯ρ(a)∈[0,1]is the mean over the horizon of the predicted probability that a collision has occurred. Because the target is cumulative, an event predicted early contributes at every later slot and weighs more than the same event predicted late, and a candidate that is predicted to avoid the event pays almost nothing. Channels with zero weight are trained but do not enter the score; the weights of each environment are given in Section 4.
  → 有界性来源：ρ̄ 是 sigmoid 概率的加权均值 ∈ [0, 1]，惩罚 = W·ρ̄ ≤ W，且 W=50（见 4.5）。

### 4.5 Risk-aware MPPI 规划器（Section 3.5，Evaluate）

- **搜索算法**（L705–707）：
  > At every control step the planner searches over action chunksa t:t+H−1 using the world model as its simulator. The search is the model predictive path integral (MPPI) update of Williams et al. (2017), with a shifted warm start and with the risk penalty of Eq. (26) in the score.
- **候选生成（warm start + 采样分布）**（L708–712）：
  > The context tokensM t and the action priora bc =a bct:t+H−1 of Eq. (13) are computed once per control step. The sampling distribution is a diagonal Gaussian over chunks with meanµ∈R H×2 and standard deviationσ∈R H×2>0 . Its mean is warm-started from the previous step's final mean shifted by one step, with the prior's last action filling the vacated slot (at the first step of an episode, froma bc itself), and its standard deviation is reset at every step toσ 0 = (0.25,0.08), for the longitudinal and the steering command at every horizon step.
- **候选集构造（Eq. 27：128 sampled + 6 prior proposals + 精确 prior = 135）**（L713–737）：
  > In each ofI= 2iterationsi= 0,1the candidate set is Ai = (clip(µ+σ⊙ϵ (i,n),−¯a,¯a))Nn=1 [sampled] ∪ (clip(abc +σ bc ⊙ϵ ′(n),−¯a,¯a))Nbc n=1 [prior proposals] ∪ (abc), (27) withN= 128samples,N bc = round(0.05N) = 6prior proposals withσ bc = (0.10,0.04)and the exact prior chunk as the last candidate,135candidates in total; every entry is clipped to[−¯a,¯a], as for the prior. All entries of the noiseϵ (i,n) andϵ ′(n) are drawn independently from a standard normal distribution; the sampled part is redrawn in every iteration, whereas the prior proposals are drawn once per control step and reused in both iterations. The prior proposals and the exact prior keep the behavior-cloned chunk and its neighborhood in the candidate set whatever the mean has drifted to, and including the exact prior chunk guarantees that the best candidate of every iteration scores at least as high as the behavior-cloned chunk (Lemma 1).
- **打分（Eq. 28：折现 return + 价值 bootstrap − W·ρ̄）**（L738–754）：
  > Each candidate is rolled out with Eq. (6) and scored with J(a) = H−1Xk=0 γk Sk ˆrt+k +γ H SH ¯Vt+H −W¯ρ(a), Sk = k−1Yk′=0 ˆct+k′, (28) where all predicted quantities are those of the rollout ofa,γis the discount of Section 3.1, the blended reward is ˆrt+k = 12 ˆrst+k + 12 ˆrϕt+k, the survival probabilityS k is accumulated from the continuation head (S0 = 1), the terminal value ¯Vt+H = 1K PKi=1 symexp(Pb ˆpv,it+H,b cb) is the ensemble mean of the values predicted for the last imagined state, and the risk weight isW= 50. The value bootstrap extends the effective horizon far beyond1.6s, which is necessary in driving, where the consequences of an action such as entering an intersection often arrive after the horizon. The scale ofWis set so that a collision predicted with certainty from the first imagined step,¯ρ= 1, costs about as much as the whole discounted return of a good candidate.
- **形式结果（Proposition 1/2、Lemma 1、Theorem 1、Corollary 1）**（L755–776, L797–824，节选）：
  > Definition 1(Model error and the planner's choice).At a decision state, letAbe a set of candidate chunks on which JandJ ⋆ attain their maxima. Themodel errorof a candidate ise J (a) =J(a)−J ⋆(a), the planner's choice is ˆa∈arg max a∈A J(a), and the best candidate isa ⋆ ∈arg max a∈A J ⋆(a).
  > Proposition 1(The planner overestimates its own choice)... Then E(eJ (ˆa))≥E(J ⋆(a⋆)−J ⋆(ˆa))≥0.
  > Proposition 2(Regret is bounded by the error at two candidates).For every realization of the model, 0≤J ⋆(a⋆)−J ⋆(ˆa)≤e J (ˆa)−e J (a⋆)≤ |e J (ˆa)|+|e J (a⋆)|.
  > Lemma 1(The prior is a floor)... Then for every iterationi= 0, . . . , I−1,Jbesti ≥max a∈Ft J(a)≥J(a bc).
  > Theorem 1(Consistency of the elite update).Under Assumption 1... the updated mean µof Eq.(29)and the best candidate converge to ˆaC almost surely asN bc → ∞.
  > Corollary 1(The remaining error is the model's).LetJ ⋆ attain its maximum overCata ⋆C. In the limit of Theorem 1 the planned chunk is ˆaC, and its regret is bounded by the model error at two chunks,0≤J ⋆(a⋆C)−J ⋆(ˆaC)≤ |eJ (ˆaC)|+|e J (a⋆C)|.
  > Remark2.Theorem 1 is a statement about the limit of many samples and says nothing about the rate. WithN bc = 6 proposals andN= 128samples in32dimensions the planner is far from this limit... Corollary 1 is where the two halves of the method meet. Sampling removes the error of the optimizer, and what remains is the model error at the chosen and at the best chunk, the quantity that the data (Section 3.2) and the online refinement (Section 3.6) reduce.
- **Elite 更新（Eq. 29）与执行**（L778–796）：
  > Index the candidates of an iteration asa (n),n= 1, . . . , N+Nbc + 1, and letEbe the indices of theN e = 16candidates with the highest scores, the elites. The distribution is moved towards them with exponential weights, wn = exp(κ(J(a (n))−J max))Pn′∈E exp(κ(J(a (n′))−J max)) , n∈ E, µ←Xn∈E wna(n),σ←max(√Xn∈E wn(a(n) −µ)2,σ min), (29) withJ max = max n∈E J(a (n)), which cancels in the ratio and only keeps the exponentials bounded,κ= 0.25and σmin = (0.05,0.02); the square, the root and the maximum are taken entrywise, andσis computed around the updated mean. After the last iteration the first action of the mean is executed asat, and the mean is stored for the next step's warm start. The planner executes the weighted mean rather than the single best sample, which averages out the randomness of any single sample. In the evaluation of Section 4 the planner runs at10Hz on one GPU, re-planning at every step with the full context.
- **MetaDrive 规划差异**（L1060–1066）：
  > On this benchmark the risk head predicts three events, vehicle contact, contact with an object and leaving the road, and the planner weights all three,ω= (1,1,1)in Eq. (26), since leaving the road also ends an episode;W= 50as in CARLA. Two settings differ from CARLA because of the benchmark. CAT's traffic replays the logs and does not react to the ego, so slowing down can cause a rear-end contact, and the braking-monotonicity term of Eq. (25) is not used during online adaptation. The action prior, distilled from CAT's TD3 expert, often acts at the bounds of the action range, where a plan carried over from the previous step lags behind it, so every MPPI step starts from the prior's current chunk instead of the shifted previous plan of Section 3.5.

### 4.6 在线 Refine 循环（Section 3.6，Refine）

- **对 SimDist 循环的两处修改**（L826–832）：
  > A world model trained on the corpus of Section 3.2 has learned from the states that other drivers visited, not from those that its own planner visits. SimDist addresses the analogous simulation-to-reality gap by collecting rollouts with the planner and re-fitting only the latent dynamics on them. We adopt the same loop (Algorithm 1) with two changes. First, the reward scorers and the risk head are adapted as well, because the closed-loop distribution of speeds, gaps and near-misses differs from the corpus and these heads, not only the dynamics, determine which candidate wins. Second, the planner does not score with the live scorers but with EMA copies that follow the live parameters slowly, so that the objective it optimizes does not move with every gradient step.
- **Buffer 与更新节奏**（L834–843）：
  > The planner drives in the environment to which it is to be adapted, which need not be the one in which the corpus was collected; in our experiments these are the training environments of each benchmark, so that its test routes and scenarios stay held out (Section 4). The planner's transitions are kept in an in-memory buffer of fixed size that holds only the most recent ones, and the parts of episodes still being driven are added to it whenever windows are drawn (the sizes are given in Appendix C). After every burst ofn b = 256environment steps,⌊0.125n b⌋= 32gradient updates are made (an update-to-data ratio of0.125, with fractional credit carried over). Each update batch mixes offline and online windows. Half of the windows of a batch are drawn uniformly from the offline corpus and half from the online buffer, and half of the online windows come from the episodes that the most recent burst completed or extended. Online windows use the sameC+Hlayout and terminal padding as offline ones.
- **什么被更新 / 什么被冻结**（L844–855）：
  > The encoderE θ, the action priorπ θ and the value ensemble stay frozen. The value ensemble is frozen because its targets, the expert's values, are not available online, and a value learned from the planner's own returns would bootstrap on its own mistakes. Besides the risk head, two parameter groups are trained, with the scorer losses computed on detached rollouts so that they cannot move the dynamics:
  > Londyn = ˜Llat,(30) Lonsc = ˜Lrew + ˜Lϕ + ˜Lcons + 0.1 ˜Lcont,(31) where Eq. (30) trains the dynamicsD θ (learning rate3×10 −5) and Eq. (31) trains the reward scorer and the reward, factor and continuation heads(T rθ , Rθ,Φ θ, Cθ)(learning rate10 −5). The risk head is trained withL bce +L mono on the same mixed batches, with its latent inputs detached, at learning rate10 −4. Online windows carry recorded safety events but no expert labels; Eqs. (30)–(31) and the risk losses need only rewards, reward factors, events and observations, which is precisely why these modules, and not the prior or the value heads, can be adapted without an expert in the loop.
- **EMA 打分目标（Eq. 32）**（L856–864）：
  > The planner rolls candidates out with the live dynamics but scores them with EMA copies ¯θof the scorers and of the risk head. After each burst, ¯θ←(1−τ b) ¯θ+τ b θ, τb = 1−(1−τ) nb/Nc , τ= 0.1, N c = 1000, (32) so that the target moves by the fractionτperN c collected steps regardless of the burst size (τb ≈0.027forn b = 256); the risk head usesτ b = 0.1per burst directly. The EMA copies are what is saved and what the evaluation of Section 4 uses. The reasoning is the one behind target networks in value learning. The reward and risk scorers define the planner's objective, and an objective that changes every32updates would make consecutive plans inconsistent; the slow copy keeps the objective stable while still tracking the closed-loop distribution over hundreds of bursts.
- **EMA 的语义（去更新噪声、不挡漂移）**（L878–885）：
  > Remark3.Withn b = 256,τ b = 1−0.9 0.256 ≈0.027. At this value, Proposition 4 reduces the standard deviation of the update noise in the scorers' parameters by the factor √(τb/(2−τ b))≈0.12(by0.23for the risk head,τ b = 0.1), while Proposition 3 bounds the lag behind a drifting live model by(1−τ b)∆θ/τb ≈37 ∆ θ, about37bursts of drift. The EMA does not slow a consistent drift of the live parameters, which in the steady state of Proposition 3 it follows at the same speed, only later; what it removes is the burst-to-burst noise of the updates, which is what would make consecutive plans inconsistent.
- **CARLA 在线运行**（L938–940）：
  > The online loop (Algorithm 1) drives the four training towns of the corpus under its termination rules, using the planner of Section 3.5 with the risk penalty (W= 50, pricing collisions only) and scoring against the EMA copies. It runs for10 6 environment steps,27.8h of driving.
- **MetaDrive 在线运行**（L1066–1068, L1914–1918）：
  > SAFER is evaluated after10 6 steps of online adaptation on the training scenarios. We train three SAFER systems with different seeds, offline and online, evaluate each three times, and report the mean and standard deviation over the three systems.
  （C.2：online episodes 为成对随机训练场景：logged traffic，随后 CAT opponent 针对该次驾驶生成；无刹车单调项）

### 4.7 数据与超参（Table C.1 / C.2 / Table 7 逐字）

**Table C.1（L1859–1892，逐字）**：
> Table C.1: Hyper-parameters of the evaluated CARLA system.
> Data collection
> action noise | clean phases25–100steps, corrupt phases5–30steps;σ α ∼ U[0.03,1.0],σδ ∼ U[0.005,0.30], redrawn every500–1000steps;2.5%clean episodes
> scripted events | hold30–80steps, late brake5–10steps, recovery30–60steps
> World model
> latent width / decoder | 256/ 4 layers, 4 heads, MLP ratio 4
> contextC, horizonH | 16,16steps (1.6s at10Hz)
> reward, value bins | B= 101bins, uniform insymlogspace on[−3,3]and[−10,10]
> reward factors | scaled softplus; scales6,6,1,1,6(speed, desired speed, lateral distance, heading error, terminal penalty)
> value heads | K5
> policy decoder | 3 layers;logσ π clamped to[−2,1]
> risk head | MLP512→256→256→3, dropout0.1,w + = 20,β∼ U[0.1,0.5]
> Offline training
> stage 1 (world model) | 1.0M steps, batch32windows, lr2×10 −4 →10 −4 (10k warm-up, cosine over 800k), dropout0.05(value heads0.01)
> stage 2 (joint, + risk head) | 0.5M steps, lr10 −4 (2k warm-up), risk head lr10 −3, world-model dropout0
> loss weights | Eq. (21); RMS decay0.99
> optimizer | AdamW, weight decay10 −4, gradient clip10, linear warm-up and cosine decay
> Planner
> samplesN/ prior proposalsN bc / elitesN e | 128/6/16; iterationsI= 2
> σ0,σ min,σ bc | (0.25,0.08),(0.05,0.02),(0.10,0.04)
> sampling noise | independent per step and dimension
> κ,γ | 0.25,0.985
> risk weightW, event weightsω | 50,(1,0,0)(collision only)
> Online adaptation
> environment steps, burst, UTD | 10 6,256,0.125
> buffer, batch mix | the60most recent trajectory segments, an episode being cut at every burst (at most3,840transitions);32windows, half offline and half online, half of the online from the latest burst
> learning rates (dynamics / scorers / risk) | 3×10 −5 /10 −5 /10 −4, gradient clip1, weight decay10 −5 (risk head10 −4)
> EMA | τ= 0.1per1,000steps (scorers),0.1per burst (risk head)
> frozen | encoder, action prior, value ensemble

**Table C.2（L1893–1918，逐字）**：
> Table C.2: Settings of the evaluated MetaDrive system that differ from those of the CARLA system; all other settings are those of Table C.1.
> Data collection
> scripted events | hazard brake instead of hold and late brake:5–10steps at a brake level drawn fromU[0.3,1], triggered when a road user ahead on the route is within20m and at most3s from collision, the ego drives at 2–10m/s and the expert's acceleration command is above−0.3; recovery30–60steps
> value labels | policy evaluation of the noise-free expert: its two Q-networks, initialized from its critic, fitted on all transitions of the corpus to one-step targets with the expert's action at the next state;10 5 steps, batch 1,024, lr3×10 −4, target-network rate0.005, discount0.99
> World model
> encoder | two-layer MLP,101→256→256, initialized from the first two layers of the expert's actor
> reward factors | progress along the route (linear, scale2m) and the indicators of arrival, leaving the road, vehicle contact and object contact (sigmoid)
> Offline training
> schedule | one stage of0.25M steps with the settings of stage 1 (cosine decay over 200k), in which the risk head is trained from the start at its stage-2 learning rate; no scene-factor head
> training windows | a keyframe at every step, at the start of an episode with the missing context padded; no additional windows before terminals
> Planner
> γ, event weightsω | 0.99,(1,1,1)
> warm start | none: every step starts from the prior's current chunk
> Online adaptation
> episodes | pairs on randomly drawn training scenarios: logged traffic, then CAT's opponent generated against that drive
> buffer, batch mix | the most recent finished episodes, about3,840transitions, and the episodes being driven;16windows, mixed as in CARLA
> risk-head loss | without the braking-monotonicity term

**计算效率（L1547–1557 + Table 7 L1590–1597）**：
> SAFER plans with a small model of15.5M parameters in CARLA (15.3M for the world model with its heads, 0.2M for the risk head) and14.3M in MetaDrive. Its training is also inexpensive on at most two Quadro RTX 6000 GPUs. The offline training takes66h in CARLA and7h in MetaDrive, and the online adaptation over10 6 environment steps41h and8h.
> Table 7: Inference time per driving decision (observation encoding and controller) on one Quadro RTX 6000 with a batch of one: mean±standard deviation over2,000decisions replayed from recorded closed-loop episodes, and the decision frequency it allows. All controllers use the same model of15.5M parameters. Rollouts: imagined16-step action chunks scored per decision. The agent acts at10Hz.
> Controller | Rollouts | Time per decision (ms) | Frequency (Hz)
> BC prior | 0 | 5.8±0.4 | 173
> MPPI | 270 | 23.1±3.2 | 43
> + risk term | 270 | 22.3±2.3 | 45
> + online,10 6 steps | 270 | 22.0±2.3 | 45
> The behavior-cloned prior decodes one action chunk at173Hz. The planner rolls out and scores270candidate chunks of16steps per decision, two iterations of135, and is therefore about four times slower, yet it still decides at43–45Hz, more than four times the10Hz at which the agent acts; even the slowest of the2,000decisions (42ms) would allow24Hz. The risk term and the online adaptation do not change this cost.

---

## 5. 伪代码

### 5.1 Algorithm 1（原文算法环境，逐字提取，L887–900）

> Algorithm 1Online adaptation loop.
> Require:offline corpusD, world modelθfrom Section 3.3, EMA copies ¯θ←θof the scorers and the risk head
> 1:bufferB ← ∅, credit←0
> 2:whilefewer than10 6 environment steps collecteddo
> 3:driven b = 256steps with the planner of Section 3.5 (live dynamics, EMA scorers, EMA risk head); append the transitions toB, which keeps only the most recent ones
> 4:credit←credit+ 0.125n b;U← ⌊credit⌋; credit←credit−U
> 5:forUupdatesdo
> 6:sample a batch of windows, half fromDand half fromB, half of the latter from the latest burst
> 7:updateD θ with Eq. (30), the scorers with Eq. (31), the risk head withL bce +L mono
> 8:end for
> 9: ¯θ←(1−τ b)¯θ+τ bθ ▷Eq. (32)
> 10:end while
> 11:returnlive dynamics, EMA scorers, EMA risk head, frozen encoder / prior / value ensemble

### 5.2 离线 Stage & Annotate —— ⚠ 依据原文重建（非原文伪代码）

> ⚠ 原文无此段的 Algorithm 环境；以下按 Section 3.2 + Section 4.1 正文重建，每一步附依据行号。不包含任何原文没有的细节。

```
⚠ 依据原文重建（非原文伪代码）—— 离线 Stage & Annotate

输入：专家策略 πe 与价值 Ve（可在任意观测上查询）                [L413-415]
      M 个部分训练 RL checkpoint π1..πM（CARLA：M=13，0.2M–4M 步间保存；MetaDrive：8 个，2.5×10^4–7.5×10^5 步）  [L453-456, L916-918, L1047-1049]

1. Dexp：专家无扰动驾驶；失误不重置，跑到环境终止；每步存储
   (ot, at, rt, ϕt, dt, et, aet, vet)                             [L471-472, L474-478]
2. Dint：对每个 checkpoint πm：
   2.1 episode 随机交替 clean/corrupt 相位；corrupt 相位执行
       at = clip(a^b_t + ξt, −1, 1)，ξt ∼ N(0, diag(σ²α, σ²δ))（Eq. (2)）；
       (σα, σδ) 随机抽取、每隔随机间隔重抽；clean 相位 at = a^b_t；
       少量 episode 全程 clean，其余一半从 corrupt 相位开始（SimDist 配方）  [L456-464]
       （CARLA：clean 25–100 步 / corrupt 5–30 步；σα∼U[0.03,1.0]，σδ∼U[0.005,0.30]，
        每 500–1000 步重抽；2.5% clean episodes）                [Table C.1 L1861-1862]
   2.2 驾驶直到环境终止（不重置），每步存储同上 → 失误后果（含安全事件）被记录  [L471-475]
3. Dtgt：专家 + 动作噪声 + 脚本事件，每步存储同上                  [L471-473, L476-478]
   3.1 hold：ego 已停且期望速度跃升时（绿灯/前车起步），在专家本应起步处
       保持全刹车 30–80 步，随后按专家动作恢复 30–60 步             [L922-924, L1863]
   3.2 late brake：专家为前方危险刹车时保持此前油门 5–10 步；
       仅当专家在 override 后仍能在危险前刹停时触发（可恢复）；
       随后按专家动作恢复 30–60 步                                 [L924-926, L1863]
   （MetaDrive 替代事件：hazard brake——前方路线 20m 内、≤3s 距碰撞的道路使用者
     逼近时刹车 5–10 步（刹车水平 U[0.3,1]），恢复 30–60 步）       [L1052-1054, L1895-1897]
4. 输出语料 D = Dexp ∪ Dint ∪ Dtgt
   （CARLA：22.06M 步 / 54,262 episodes / 30,566 碰撞 / 10,185 闯红灯；
     flywheel 一轮后 28.56M 步）                                  [L471-473, Table 1 L1002-1009]
5. 用途：世界模型从 transitions 学「错误导致什么」；
   BC 从标签学「到达的状态该做什么」                               [L479-480]
6. （可选，数据飞轮）分析离线语料中 coverage 最少的场景（行人相遇），
   定向补充 D+int / D+tgt（含横穿马路的脚本行人），并按原配方重训两阶段   [L1107-1114]
```

### 5.3 Risk-aware MPPI —— ⚠ 依据原文重建（非原文伪代码）

> ⚠ 原文无此段的 Algorithm 环境；以下按 Section 3.5 正文重建，每一步附依据行号。不包含任何原文没有的细节。

```
⚠ 依据原文重建（非原文伪代码）—— Risk-aware MPPI 规划（每个控制步 t，0.1s/步）[L429]

1. 计算上下文 token Mt（2C 个）；每控制步一次、所有候选共享              [L490-499]
2. 由 action prior 解码确定性 chunk abc = (tanh(µπk))，逐元素
   clip 到 [−ā, ā]，ā = 0.999                                          [L546-553]
3. 初始化采样分布（对角高斯，均值 µ ∈ R^{H×2}，标准差 σ ∈ R^{H×2}）：
   µ ← 上一步最终均值平移一步，空缺槽填 prior 的最后动作
      （episode 第一步取 abc 本身）；σ ← σ0 = (0.25, 0.08) 每步重置       [L708-712]
   （MetaDrive：无 warm start，每步从 prior 当前 chunk 开始）            [L1065-1066]
4. 构造候选集 Ai（每迭代，Eq. (27)）：
   · 128 个 sampled：clip(µ + σ ⊙ ε(i,n), −ā, ā)，ε 各元素独立标准正态；
     每迭代重抽                                                         [L713-720, L732-733]
   · 6 个 prior proposals：clip(abc + σbc ⊙ ε'(n), −ā, ā)，σbc = (0.10, 0.04)；
     每控制步抽一次、两个迭代复用                                        [L731, L734-735]
   · 第 135 个候选：精确 prior 块 abc                                    [L730-732]
5. for i = 0, 1（I = 2 次迭代）：                                       [L713]
   5.1 每个候选用 Eq. (6) 一次前向解码出全部 16 个未来 latent
       (ẑt+1, …, ẑt+H)（chunk 联合预测，非逐步递归）                     [L500-507]
   5.2 heads 读出：直接奖励 r̂s、因子化奖励 r̂ϕ（Eq. (12)）、
       生存概率 ĉ、K=5 价值集成、风险 logits ρk,j（Eq. (22)）            [L508-544, L638-642]
   5.3 打分（Eq. (28)，公式转写）：
       J(a) = Σ_{k=0}^{H−1} γ^k S_k r̂_{t+k} + γ^H S_H V̄_{t+H} − W·ρ̄(a)
       r̂_{t+k} = ½ r̂s_{t+k} + ½ r̂ϕ_{t+k}；S_k = Π_{k'=0}^{k−1} ĉ_{t+k'}（S0=1）；
       V̄_{t+H} = (1/K) Σ_i symexp(Σ_b p̂v,i_{t+H,b} cb)；
       ρ̄(a) = (1/H) Σ_{k=0}^{H−1} Σ_j ω_j sigmoid(ρ_{k,j}(a))（Eq. (26)）；
       W = 50（CARLA ω=(1,0,0)；MetaDrive ω=(1,1,1)）                    [L738-754, L686-702, L1060-1062]
   5.4 取 Ne = 16 个最高分 elites；按 Eq. (29) 指数加权更新：
       wn = exp(κ(J(a(n)) − Jmax)) / Σ_{n'∈E} exp(κ(J(a(n')) − Jmax))，κ = 0.25；
       µ ← Σ_{n∈E} wn a(n)；
       σ ← max(√(Σ_{n∈E} wn (a(n) − µ)²), σmin)，σmin = (0.05, 0.02)（逐元素）  [L778-793]
6. 执行 µ 的第一个动作作为 at；保存 µ 供下一步 warm start                  [L793-794]
   （执行加权均值 µ 而非单个最优样本，平均掉单样本随机性）                  [L793-795]
7. 下一控制步重新规划；评估中在单 GPU 上以 10Hz 运行、每步重规划            [L795-796]
```

---

## 6. 主结果数值表

### 6.1 CARLA Leaderboard（Table 3、Table 4 逐字，L1277–1311）

**Table 3（L1277–1300）**：
> Table 3: Comparison with the baselines on the CARLA Leaderboard. The end-to-end methods (top) are given with their published results. TD3+BC, IQL, TD-MPC2 and DreamerV3 (middle) are trained by us on SAFER's corpus, with the frozen encoder of the expert and without its labels, and evaluated as SAFER.
> Algorithm | Input | Driving score↑ | Route compl.↑ | Infraction penalty↑
> CILRS (Codevilla et al., 2019) | RGB | 0.05 | 0.14 | 0.55
> LBC (Chen et al., 2020) | RGB | 0.09 | 0.18 | 0.73
> TransFuser (Prakash et al., 2021) | RGB + LiDAR | 0.17 | 0.52 | 0.42
> NEAT (Chitta et al., 2021) | RGB | 0.22 | 0.42 | 0.65
> MMFN (Zhang et al., 2022) | RGB + LiDAR + HD map | 0.23 | 0.47 | 0.63
> IARL (Toromanoff et al., 2020) | RGB | 0.25 | 0.47 | 0.52
> Rails (Chen et al., 2021) | RGB | 0.31 | 0.58 | 0.56
> GRIAD (Chekroun et al., 2023) | RGB | 0.37 | 0.62 | 0.60
> Latent TransFuser (Chitta et al., 2023) | RGB | 0.45 | 0.66 | 0.72
> TransFuser (Chitta et al., 2023) | RGB + LiDAR | 0.61 | 0.87 | 0.71
> LA V (Chen and Krähenbühl, 2022) | RGB + LiDAR | 0.62 | 0.94 | 0.64
> TF++ (Jaeger et al., 2023) | RGB + LiDAR | 0.66 | 0.79 | 0.84
> TCP (Wu et al., 2022) | RGB | 0.75 | 0.86 | 0.87
> InterFuser (Shao et al., 2023) | RGB + LiDAR | 0.76 | 0.88 | 0.84
> TD-MPC2 (Hansen et al., 2024) | BEV + ego state | 0.43 | 0.70 | 0.59
> TD3+BC (Fujimoto and Gu, 2021) | BEV + ego state | 0.63 | 0.84 | 0.76
> IQL (Kostrikov et al., 2022) | BEV + ego state | 0.73 | 0.88 | 0.82
> DreamerV3 (Hafner et al., 2025a) | BEV + ego state | 0.82 | 0.94 | 0.88
> SAFER (ours) | BEV + ego state | **0.80** | **0.95** | **0.84**

**Table 4（L1301–1311）**：
> Table 4: Further metrics of the methods of Table 3 whose input is exactly that of SAFER, under the same protocol: mean±standard deviation over three seeds. Reward of Eq. (1) per step, average speed (route distance completed over the duration of the episode), share of the routes completed, share of the episodes in which the ego is blocked, and collisions and traffic-rule infractions per100km driven.
> Algorithm | Reward per step↑ | Speed (m/s)↑ | Completed (%)↑ | Blocked (%)↓ | Collisions Pedestrians | Vehicles | Red lights | Stop signs
> TD-MPC2 | −0.51±0.18 | 2.61±0.22 | 43.0±8.0 | 52.6±6.0 | 5.6±1.4 | 27.2±0.2 | 3.8±2.0 | 0.7±1.2
> TD3+BC | 0.53±0.02 | 3.08±0.09 | 68.0±3.0 | 30.3±2.6 | 8.4±2.6 | 27.9±3.3 | 4.7±0.6 | 2.6±0.8
> IQL | 0.47±0.09 | 3.20±0.05 | 75.4±2.7 | 21.5±2.7 | 4.2±2.5 | 19.3±4.3 | 3.9±2.1 | 3.9±0.9
> DreamerV3 | 0.71±0.01 | 3.26±0.05 | 88.2±2.6 | 11.8±2.6 | 1.1±0.9 | 17.1±6.0 | 2.1±1.6 | 0.0±0.0
> SAFER (ours) | **0.77±0.02** | **3.31±0.03** | **89.0±3.8** | **11.0±3.8** | 5.5±1.3 | **16.0±2.4** | **1.6±1.6** | 1.8±0.4

### 6.2 MetaDrive / CAT（Table 5 逐字，L1322–1358）

> Table 5: Closed-loop comparison on the MetaDrive/CAT benchmark (100test scenarios): mean±standard deviation over three seeds. (a) Regular: logged traffic; (b) safety-critical: CAT's opponent generated against the regular drive. Baseline results are from Sheng et al. (2026a), except those of DreamerV3, which we trained.
> (a) Regular test scenarios
> Method | Episode reward↑ | Road completion (%)↑ | Total distance↑ | Crash rate (%)↓ | Average speed↑
> SAC | 43.3±1.61 | 67.9±1.14 | 44.9±1.07 | 18.3±0.92 | 9.35±0.06
> PPO | 41.1±2.48 | 65.9±2.51 | 42.9±2.01 | 22.6±3.75 | 10.1±0.24
> TD3 | 49.3±0.83 | 72.9±0.44 | 47.9±1.08 | 18.9±1.72 | 8.14±0.80
> SAC-PID | 7.57±3.20 | 24.8±10.3 | 12.9±2.91 | 21.3±1.35 | 5.19±0.57
> TD3-Lag | 39.6±0.00 | 46.9±0.00 | 38.9±0.00 | 23.0±0.00 | 5.50±0.00
> TD3-PID | −1.00±0.91 | 6.48±0.44 | 4.55±0.37 | 22.8±1.77 | 4.17±1.08
> BC | 29.7±2.52 | 58.6±2.09 | 33.4±1.80 | 20.6±1.18 | 7.42±0.28
> GAIL | 32.5±1.99 | 57.0±1.50 | 35.1±1.82 | 23.4±1.94 | 7.84±0.74
> AIRL | 27.0±1.94 | 54.8±3.13 | 32.0±1.63 | 20.1±3.06 | 9.80±0.76
> SQIL | 20.3±0.21 | 44.7±0.78 | 25.1±0.36 | 26.9±1.30 | 6.50±0.38
> CAT | 49.1±3.89 | 73.0±3.94 | 48.5±3.39 | 15.5±1.72 | 8.45±1.23
> CLIC | 43.5±0.49 | 68.3±0.60 | 45.1±0.71 | 15.1±1.39 | 9.31±0.28
> CurricuVLM | 52.3±2.28 | 76.2±2.28 | 51.0±1.65 | 16.0±0.80 | 9.52±0.14
> DreamerV3 | 46.3±0.24 | 71.4±0.57 | 45.8±0.43 | 19.4±4.63 | 7.05±0.32
> SAFER (ours) | **53.4±1.60** | **79.7±0.79** | **51.3±1.35** | 18.6±0.57 | **10.1±0.06**
> (b) Safety-critical test scenarios
> SAC | 38.4±1.97 | 63.2±1.21 | 40.9±1.34 | 30.5±2.33 | 9.25±0.07
> PPO | 38.4±0.86 | 62.7±1.05 | 40.0±0.70 | 32.0±2.02 | 9.94±0.30
> TD3 | 42.4±1.01 | 65.2±1.40 | 42.6±1.26 | 39.7±1.04 | 8.02±0.77
> SAC-PID | 7.46±3.05 | 19.0±4.19 | 12.7±2.82 | 30.4±3.27 | 5.22±0.54
> TD3-Lag | 29.1±0.00 | 39.0±0.00 | 30.7±0.00 | 51.0±0.00 | 5.70±0.00
> TD3-PID | −0.68±1.00 | 5.50±0.90 | 4.58±0.35 | 29.4±5.83 | 4.18±1.06
> BC | 27.4±2.69 | 53.2±1.81 | 31.0±2.17 | 33.7±0.27 | 7.30±0.26
> GAIL | 29.8±1.07 | 52.3±1.42 | 32.7±1.05 | 33.7±1.78 | 7.77±0.73
> AIRL | 26.3±1.84 | 52.8±3.16 | 31.2±1.50 | 27.8±3.51 | 9.74±0.72
> SQIL | 18.9±0.56 | 41.4±1.01 | 23.5±0.52 | 38.5±1.67 | 6.54±0.34
> CAT | 42.5±3.95 | 66.6±4.37 | 43.4±3.48 | 32.1±2.08 | 8.36±1.17
> CLIC | 39.3±0.72 | 64.3±0.40 | 41.6±0.78 | 26.2±1.17 | 9.21±0.26
> CurricuVLM | 48.9±1.53 | 73.4±1.66 | 48.4±1.31 | 25.1±1.17 | 9.45±0.16
> DreamerV3 | 39.4±0.53 | 63.0±1.00 | 40.7±0.52 | 44.3±6.61 | 7.47±0.31
> SAFER (ours) | **49.0±0.54** | **74.5±0.38** | 48.2±0.45 | 31.4±1.23 | **10.0±0.05**

### 6.3 关键结果陈述（原句）

- CARLA 主结果（L1180–1185）：
  > SAFER reaches the highest route completion (0.95) and the second-highest driving score (0.80), after DreamerV3 (0.82). The strongest end-to-end baselines trade route completion for infraction penalty... SAFER completes95%of each route at an infraction penalty of0.84, that of InterFuser and TF++ and below TCP's, so its lead in driving score over them comes from completing more of each route at a comparable infraction penalty.
- 与离线 RL baseline 的对比（L1265–1269）：
  > The four methods trained on SAFER's corpus reach driving scores of0.82(DreamerV3),0.73(IQL),0.63 (TD3+BC) and0.43(TD-MPC2), against0.80for SAFER. Paired over routes and seeds, DreamerV3 and SAFER do not differ significantly in any of the three metrics... TD-MPC2 is blocked in53%of its episodes, against11%for SAFER.
- Table 4 六项第一（L1271–1272）：
  > SAFER has the best value in six of the eight measures: the highest reward per step (0.77, against at most 0.71for the baselines), the highest average speed (3.31m/s) and the largest share of completed routes (89%), and the fewest collisions with vehicles, red lights run and blocked episodes.
- MetaDrive（L1359–1367）：
  > In regular traffic SAFER has the highest episode reward, road completion and total distance in Table 5 and shares the highest average speed with PPO; its crash rate (18.6%) is lower than that of every baseline except SAC (18.3%) and the three closed-loop training methods (15.1–16.0%). In safety-critical traffic it has the highest episode reward, road completion and average speed, and its crash rate (31.4%) is close to those of SAC and CAT (30.5%and32.1%), while CurricuVLM, which builds its training curriculum from generated safety-critical scenarios, keeps the lowest crash rate (25.1%) and a slightly longer total distance (48.4against48.2).

---

## 7. 消融（全部抄录）

### 7.1 六个消融臂定义（L1097–1114 逐字）

> Five controllers share the stage-2 checkpoint and differ only in how it is used, and a sixth arm retrains it:
> •BC prior: the behavior-cloned priorπ θ of Eq. (13) executed directly, no planning;
> •MPPI: planning with the reward, continuation and value heads and no risk penalty (W= 0), i.e. SimDist's planner on our model;
> •+ risk term: the same planner with the risk penalty,W= 50;
> •+ online, dynamics only: the same planner on a model whose dynamicsD θ alone are adapted online (Eq. (30)), for10 6 steps and otherwise as in the full system, while the reward scorers and the risk head keep their offline weights, i.e. SimDist's adaptation rule on our model;
> •+ online adaptation: the same planner on the model adapted online for10 6 steps, the full SAFER system. The snapshot of the same run after5×10 5 steps is evaluated as well, for the effect of training length;
> •+ data flywheel: one turn of a loop that aims new data at the scenarios that an analysis of the offline corpus finds covered least and retrains the system. These are encounters with pedestrians. Of the21,287recorded collisions analyzed in Section 4.5, only2,896involve a pedestrian, and the risk head predicts them at the step at which they happen less often than collisions with vehicles (Fig. 6b). The corpus is therefore extended by6.5M transitions, collected like the rest of it on random routes in the training towns. Of these,4.0M come from the intermediate drivers, and2.5M from the expert with noise and late brakes, facing scripted pedestrians that step into the ego's path on straight roads and at the exits of left turns (Table 1). Both offline stages are retrained with the same recipe, and the new model is adapted online for10 6 steps with the full system's settings.

### 7.2 Table 2 全表（L1070–1095 逐字）

> Table 2: Closed-loop ablation on the76Leaderboard routes: mean±standard deviation over three seeds (76episodes per seed,228per arm).
> (a) DS: driving score; RC: route completion; IP: infraction penalty; Compl.: routes completed; Clean: routes completed without a collision;∆DS: change in driving score from the row above (for all three online rows, from + risk term), paired over route and seed, with its95%bootstrap interval over routes. The last two online rows are the snapshots of one online run after5×10 5 and10 6 environment steps; the first online row is a separate run of10 6 steps that adapts the dynamics alone. The last row is one turn of the data flywheel, in which the model is retrained on the corpus extended by the lower block of Table 1 and adapted online for10 6 steps. (b) Collisions by type, red lights run and stop signs run per100km; Blocked: episodes ended by blocking; smoothness: mean absolute step-to-step change of the executed acceleration|∆a|and steering|∆δ|commands.
> (a) Route level
> Arm | DS | RC | IP | Compl. (%) | Clean (%) | ∆DS
> BC prior | 0.647±0.021 | 0.921±0.038 | 0.701±0.026 | 85.1±4.2 | 38.6±6.5 | –
> MPPI | 0.620±0.034 | 0.965±0.009 | 0.639±0.036 | 91.7±1.5 | 38.2±4.7 | −0.028 [−0.078,+0.021]
> + risk term | 0.702±0.018 | 0.978±0.019 | 0.715±0.014 | 94.7±3.5 | 50.4±5.3 | +0.082 [+0.034,+0.131]
> + online, dynamics only | 0.719±0.036 | 0.959±0.004 | 0.750±0.037 | 90.8±1.3 | 48.7±6.0 | +0.017 [−0.028,+0.061]
> + online,5×10 5 steps | 0.740±0.036 | 0.944±0.006 | 0.788±0.040 | 88.6±0.8 | 54.4±9.0 | +0.038 [−0.013,+0.089]
> + online,10 6 steps | 0.768±0.029 | 0.967±0.008 | 0.796±0.023 | 93.4±1.3 | 57.5±6.1 | +0.066 [+0.013,+0.119]
> + data flywheel | 0.801±0.038 | 0.950±0.014 | 0.843±0.031 | 89.0±3.8 | 61.8±7.3 | +0.033 [−0.010,+0.076]
> (b) Infractions and smoothness (per100km / Smoothness)
> Arm | Vehicle | Pedestrian | Static | Red light | Stop sign | Blocked (%) | |∆a| | |∆δ|
> BC prior | 27.9±4.8 | 19.5±3.7 | 0.8±0.8 | 1.3±0.4 | 0.5±0.5 | 14.9±4.2 | 0.038±0.002 | 0.004±0.000
> MPPI | 31.7±4.4 | 20.6±5.0 | 0.3±0.4 | 5.7±0.5 | 8.0±1.2 | 7.0±2.0 | 0.244±0.002 | 0.091±0.001
> + risk term | 28.3±4.1 | 12.7±0.3 | 0.3±0.4 | 4.3±1.6 | 4.6±0.7 | 4.8±4.2 | 0.243±0.003 | 0.092±0.002
> + online, dynamics only | 22.7±4.2 | 14.7±3.4 | 0.8±0.8 | 1.0±0.5 | 1.3±0.4 | 8.3±1.5 | 0.231±0.001 | 0.074±0.001
> + online,5×10 5 steps | 20.7±4.3 | 10.8±3.7 | 0.8±0.8 | 2.1±0.5 | 1.8±1.2 | 11.0±0.8 | 0.192±0.000 | 0.069±0.000
> + online,10 6 steps | 20.6±3.5 | 8.8±2.6 | 0.0±0.0 | 1.3±0.4 | 3.1±0.8 | 6.6±1.3 | 0.194±0.002 | 0.074±0.000
> + data flywheel | 16.0±2.4 | 5.5±1.3 | 0.5±0.9 | 1.6±1.6 | 1.8±0.4 | 11.0±3.8 | 0.172±0.001 | 0.059±0.000

### 7.3 消融正文关键句（逐字）

- MPPI vs BC prior（L1119–1122）：
  > Replacing the prior by the planner without a risk term does not change the driving score significantly. The planner completes more of each route and is blocked less often (7.0%of the episodes against14.9%), but it runs more red lights (5.7against1.3per100km), so its infraction penalty falls, and its acceleration command changes six times more from one step to the next.
- + risk term（L1123–1125）：
  > Adding the risk penalty raises the driving score by0.08, and the gain comes from the infraction penalty. Collisions with pedestrians fall from20.6to12.7per100km, and the share of routes completed without a collision rises from 38%to50%.
- 在线自适应（L1127–1134）：
  > The model adapted on the planner's own driving raises the driving score by a further0.07, again through the infraction penalty, while route completion stays at the level of the offline planner (0.967against0.978). Every type of infraction becomes rarer. Per100km, vehicle collisions fall from28.3to20.6, pedestrian collisions from12.7to 8.8and red lights run from4.3to1.3, and the commands become smoother (|∆a|from0.243to0.194). The gain is concentrated where the online loop drives. The driving score rises by0.07on the50routes of the four training towns and by0.16on the ten held-out Town04 routes, too few to be conclusive, but not in the two towns without training data. Online adaptation thus extends the coverage of the towns in which the planner drives during adaptation; in the two towns where it never drives, the driving score is unchanged.
- 半程快照（L1138–1143）：
  > Halfway through the online run, after5×10 5 steps, the safety gain is largely in place... This snapshot is more cautious, though. It is blocked in11.0%of the episodes (4.8%offline,6.6% after10 6 steps) and completes less of each route, so its gain in driving score over the offline planner is smaller and not significant. In the second half of the run, route completion rises again, from0.944to0.967, while the infraction penalty holds, which gives the full system's gain.
- 只调 dynamics 不够（L1144–1150）：
  > Besides how long the loop runs, the gain depends on what it adapts. When the loop adapts the dynamics alone, as SimDist does, and leaves the reward scorers and the risk head at their offline weights, the driving score rises by only 0.02over the offline planner, which is not significant, and stays0.05below the full system's, a significant difference that comes from the infraction penalty. The adapted dynamics alone make vehicle collisions and red lights run rarer (22.7and1.0per100km, against28.3and4.3offline), but pedestrian collisions do not fall (14.7against12.7) and the planner is blocked more often (8.3%of the episodes against4.8%). The significant gain of online adaptation therefore requires adapting the reward scorers and the risk head along with the dynamics, as SAFER does (Section 3.6).
- 数据飞轮（L1151–1154）：
  > One turn of the data flywheel makes the system safer. Its infraction penalty rises by0.05, a significant gain. Pedestrian collisions fall from8.8to5.5per100km and vehicle collisions from20.6to16.0. The retrained system is also more cautious. It is blocked in11.0%of the episodes, against6.6%, and completes less of each route (0.950 against0.967), so its driving score rises by0.03, which is not significant.
- 全流程总结（L1135–1137）：
  > Over the three steps, the full system raises the driving score of the behavior-cloned prior by0.12, through both a higher route completion and a higher infraction penalty. The standard deviation of the driving score over seeds lies between0.018and0.038for every row of Table 2, below the size of each significant step.

### 7.4 附：risk head 可靠性验证与 imagination 信息量（供 4.4/9 佐证）

- **单决策分解（Fig. 5a 记录碰撞，L1158–1165）**：
  > In the first state (Fig. 5a) the ego drives at8.7m/s towards a vehicle ahead in its lane. The recorded driver kept accelerating and hit the vehicle1.6s later, although the expert's label at that moment was to brake. The planner brakes over the whole horizon. Its executed plan scoresJ= 48.5, against45.4for the behavior-cloned prior and38.2 for the recorded actions, and its predicted collision probability does not exceed0.001. For the recorded actions, by contrast, the risk head's probability rises from about0.8s ahead and reaches0.33just before the recorded impact. The same holds beyond this example. Over200recorded collisions of this kind, the executed plan brakes harder than the recorded driver in98.5%of them and carries a lower predicted risk up to the impact in98.0%(mean0.013, against 0.166for the recorded actions and0.016for the prior).
- **无碰撞场景（Fig. 5b，L1166–1172）**：
  > In the second state (Fig. 5b) the ego drives at11.6m/s through a junction with vehicles on both sides and pedestrians ahead to the right, and no collision is imminent. The risk head stays quiet. The plan's predicted collision probability never exceeds0.003, and no candidate reaches a risk penalty of0.025... The risk term rises before a real collision, changes the plan there, and stays silent where none threatens.
- **离线风险头统计（L1443–1453）**：
  > The controls never raise a false alarm. Over all4,500control decisions the largest ˆP1.6 is0.29and the99th percentile0.04(Fig. 6a)... The separation from the controls is already0.88in AUC at3.0s, when the impact is still1.4s beyond the horizon,0.97at2.0s, and at least0.998from1.1s on (Fig. 6c)... Requiring the collision to be predicted by the step at which it happens is stricter. ˆPimp exceeds0.5for56%of the events when the impact is1.6s away,54–57%between1.0and0.5s, and68%at0.1s (Fig. 6b). Vehicle collisions are timed best (73%at1.6s,90%at0.1s), pedestrian collisions at51–61%and other collisions at31–57%. Since at least 92%of these events have ˆP1.6 >0.5from1.4s before the impact on, the gap between the two readouts is a timing error, not a missed collision. The head expects the collision inside the horizon but places it later than it happens.
- **闭环风险头（L1454–1463）**：
  > In closed loop, the same test runs on the system's own collisions. In the228evaluation episodes of the final system, the SAFER of Table 3, the ego collides68times while moving (47times with a vehicle,20times with a pedestrian, once with the static layout)... No control decision reaches ˆP1.6 = 0.04, the AUC of ˆP1.6 is0.89at2.0s before the impact and0.95at0.1s, and it is at least0.95at every time to impact for the pedestrian collisions... Their median stays below0.02up to the impact, and within the last1.6s only4%of the vehicle and5%of the pedestrian collisions reach ˆP1.6 >0.5at any decision. Where the head does flag the behavior-cloned prior's chunk (ˆP1.6 >0.5, at315moments), the planner executes a plan that the head expects to be safe in98%of the cases, and92%of these moments pass without a collision in the next 3s.
- **风险区域定性（Fig. 7，L1513–1528 节选）**：
  > (a) In CARLA, at6.3m/s, a pedestrian stands at the inner curb of a left bend, about13m ahead. The risk area is confined to the inner edge of the reachable set next to the pedestrian, where the predicted collision probability reaches0.83... The prior cuts the bend towards the pedestrian and runs into the risk area (risk term9.0, higher than for96%of the candidates); the plan follows the lane outside it (0.2). Of the plan's score lead of19.8over the prior,8.8come from the risk term and10.9from the predicted return... In both decisions the plan runs through the part of the reachable set where the risk area is weakest, whereas the prior, which the planner keeps among its candidates, runs into it.
- **Imagination 携带决策信息（Table 6，L1502–1510）**：
  > Table 6: Reading the imagination of the stage-1 world model with its scene-factor head: share of recorded windows (%) in which the state read 1.6s ahead in the imagined latents matches the rule label of the recorded frame, against copying the state the head reads at the decision time...
  > Factor | All windows: Imagined / Copied | Changing windows: Imagined / Copied
  > Traffic light | 93.5 / 90.8 | 76.7 / 2.7
  > Pedestrian near the path | 98.0 / 98.3 | 56.7 / 51.3
  > Vehicle ahead | 94.8 / 95.0 | 49.3 / 30.0
  （L1540–1546：）> Where the state does change, copying is almost always wrong for traffic lights, while the imagined light is right in77%of the windows; for pedestrians and the vehicle ahead the imagination also does better than copying (57%against51%and49%against30%)... The imagination thus carries the state of traffic lights and the movement of other road users on which the planner's decisions depend.

---

## 8. 局限（原文逐字）

**原文没有专门的 Limitations 小节**（正文为 Introduction / Related work / Methodology / Experiments / Conclusion，L24、L208、L281、L901、L1599）。以下为文中自述的局限类陈述，逐字抄录：

1. **从特权 BEV 到传感器输入的迁移是「下一步」而非已完成**（Conclusion，L1620–1622）：
   > The recipe needs only a driver that is allowed to fail recoverably and an expert that can be queried at the states it visits. A proving ground with a safety driver offers both, which makes the transfer of SAFER from privileged bird's-eye-view observations in simulation to sensor-based driving its natural next step.
2. **对比不覆盖完整感知管线**（L1176–1178）：
   > SAFER observes the privileged bird's-eye view of Roach and is scored with the Leaderboard's metrics on the76Leaderboard routes of Section 4.1.1. The comparison therefore places SAFER's driving decisions among those of the baselines; it does not rank complete pipelines from sensors to control.
3. **在线增益不覆盖无训练数据城镇**（L1131–1134）：
   > The driving score rises by0.07on the50routes of the four training towns and by0.16on the ten held-out Town04 routes, too few to be conclusive, but not in the two towns without training data. Online adaptation thus extends the coverage of the towns in which the planner drives during adaptation; in the two towns where it never drives, the driving score is unchanged.
4. **风险头在闭环中置信度大幅下降**（L1457–1462）：
   > The head of the final system still separates the two, less sharply than on recorded collisions and again without a false alarm... The probabilities that the head assigns to its own plan before these collisions are much lower than those of the offline head on recorded collisions, though. Their median stays below0.02up to the impact, and within the last1.6s only4%of the vehicle and5%of the pedestrian collisions reach ˆP1.6 >0.5at any decision.
5. **风险头时点误差（timing error）**（L1451–1453）：
   > Since at least 92%of these events have ˆP1.6 >0.5from1.4s before the impact on, the gap between the two readouts is a timing error, not a missed collision. The head expects the collision inside the horizon but places it later than it happens.
6. **规划器远离渐近一致性极限**（Remark 2，L815–817）：
   > Theorem 1 is a statement about the limit of many samples and says nothing about the rate. WithN bc = 6 proposals andN= 128samples in32dimensions the planner is far from this limit.
7. **相对 baselines 的个别短板**（L1273–1276）：DreamerV3 行人碰撞更少、不闯停车标志；IQL 行人碰撞更少；TD-MPC2 停车标志更少：
   > DreamerV3 collides with pedestrians less often than SAFER (1.1against5.5per100km) and runs no stop signs, but it falls behind SAFER in the other six measures. IQL also collides with pedestrians less often than SAFER (4.2against5.5per100km) and TD-MPC2 runs fewer stop signs.
8. **安全临界交通下 crash rate 非最低**（L1361–1364）：
   > In safety-critical traffic it has the highest episode reward, road completion and average speed, and its crash rate (31.4%) is close to those of SAC and CAT (30.5%and32.1%), while CurricuVLM, which builds its training curriculum from generated safety-critical scenarios, keeps the lowest crash rate (25.1%) and a slightly longer total distance (48.4against48.2).
9. **数据飞轮一轮后更保守、DS 提升不显著**（L1152–1154）：
   > The retrained system is also more cautious. It is blocked in11.0%of the episodes, against6.6%, and completes less of each route (0.950 against0.967), so its driving score rises by0.03, which is not significant.

---

## 9. 叙事关联【推断】

> 以下 3 条均为【推断】，是把 SAFER 的机制抽出来回答父任务「力学的 free demonstrations 从哪来」这一问题；每条的类比映射均为我的解读，论文本身不涉及力学场景。

### 9.1【推断】free demonstrations 的廉价制造管线：失败演示不需要专家自己犯错误

SAFER 对「演示数据从哪来」的回答是：用**部分训练智能体 + 可恢复扰动**批量制造失败轨迹，再用**可在任意状态被查询的专家**给每个访问过的状态贴动作/价值标签——这样「演示」不是专家本人的驾驶，而是「弱司机的失败 + 专家的纠错」。支撑原句：

- L413–415（专家可在自己永远不会到达的状态上被查询——标签的来源被解耦了）：
  > The first is an expert policyπe with a value functionV e, both functions of the observation, so that they can be queried at any observation, including those of states that the expert would never reach itself
- L481–482（整条配方的唯一前提，即「可恢复地失败 + 可查询的专家」）：
  > the recipe itself needs only a driver that is allowed to fail recoverably and an expert that can be queried at the visited states, so a proving ground with a safety driver would serve the same purpose.
- L479–480（一条数据两个用途）：
  > A world model learns from the transitions what each mistake leads to, and a policy learns from the labels what should have been done at the states that the mistakes reach.
- L453–456（弱司机 = 训练中途 checkpoint，成本极低）：
  > the behavior policies areMcheckpointsπ 1, . . . , πM of a reinforcement-learning agent trained from scratch with the expert's recipe and saved at increasing numbers of training steps, so that the corpus contains the systematic mistakes of drivers at several levels of competence
- L465–470（扰动必须可恢复 + 覆盖两类最易错场景）：
  > An event that withholds braking is triggered only while the expert could still stop before the hazard, so that every staged mistake is recoverable. These events place the two situations that a planner most often gets wrong, waiting when it should go and going when it should brake, into the data together with their labeled corrections.

**类比映射（推断）**：若「力学」场景同样缺 free demonstrations，SAFER 给出的答案是三件套：① 用不同成熟度的部分训练策略 + 有界、可恢复的脚本扰动来生成偏离专家但后果可控的轨迹（演示的「量」是廉价的）；② 用一个可在任意状态查询的专家函数（策略 + 价值，如离线 Q/V 网络或规则 oracle）对每个访问状态打标（演示的「质」不依赖专家亲自到达该状态）；③ 失败跑到后果为止、不重置，使每段数据同时教「错误导致什么」和「这时该做什么」。价值标签尤其重要：价值函数可以在 off-policy 状态上查询（L413–415），这等价于把「专家会不会到那里」与「专家说那里好不好」解耦。

### 9.2【推断】在线自我精炼：专家退场后靠「记录的结果」继续扩展 coverage，专家教过的东西冻结

在线阶段不依赖专家（L417–418："no expert labels, so that what it learns online does not depend on the expert"），只更新能被记录结果监督的模块（dynamics、reward scorers、risk head），专家独家可教的模块（encoder/prior/value）冻结。支撑原句：

- L844–846：
  > The encoderE θ, the action priorπ θ and the value ensemble stay frozen. The value ensemble is frozen because its targets, the expert's values, are not available online, and a value learned from the planner's own returns would bootstrap on its own mistakes.
- L853–855（冻结/更新的判据 = 监督是否在线可得）：
  > Online windows carry recorded safety events but no expert labels; Eqs. (30)–(31) and the risk losses need only rewards, reward factors, events and observations, which is precisely why these modules, and not the prior or the value heads, can be adapted without an expert in the loop.
- L856–864（EMA 慢副本保持目标稳定）：
  > The planner rolls candidates out with the live dynamics but scores them with EMA copies ¯θof the scorers and of the risk head... an objective that changes every32updates would make consecutive plans inconsistent; the slow copy keeps the objective stable while still tracking the closed-loop distribution over hundreds of bursts.
- L883–885（EMA 的精确语义：滤掉逐 burst 噪声、不挡一致漂移）：
  > The EMA does not slow a consistent drift of the live parameters, which in the steady state of Proposition 3 it follows at the same speed, only later; what it removes is the burst-to-burst noise of the updates, which is what would make consecutive plans inconsistent.

**类比映射（推断）**：对「力学」的启示是 free demonstrations 的两段式生命周期：离线阶段用（9.1）的廉价管线把 coverage 铺满，在线阶段则完全靠 planner 自己驾驶产生的 (o,a,r,ϕ,d,e) 自监督地继续生长，且通过「冻结专家教过的 head + EMA 慢副本」防止自我强化错误与目标漂移。即：free demonstrations 不是一次性给足，而是「先批量廉价制造 + 后在线免费自举」的组合。

### 9.3【推断】碰撞惩罚的有界性设计：风险是「价格」而不是「开关」，保证探索不会被无限惩罚

SAFER 的风险项 ρ̄(a) ∈ [0,1]（sigmoid 概率沿时域的加权均值），乘固定权重 W=50，因此风险惩罚**有硬上界**（W），且「确定碰撞」的代价被刻意校准为≈一个好候选的整个折现回报；同时累积目标使「预测得早」的事件代价更高、「避开事件」的候选几乎免费。这是让规划器敢在 coverage 边缘探索的关键机制——模型误差本身无界，但风险价格有界。支撑原句：

- L634–635（有界 + 尺度可控是专用头的设计动机）：
  > A dedicated head that answers a simpler question,has a collision happened by imagined stepk, can be trained as a classifier with class re-weighting, is bounded, and yields a penalty whose scale is under direct control.
- L698–701（界的具体形式 + 早预测更重 + 避开几乎不付费）：
  > whenωselects the collision channel alone,¯ρ(a)∈[0,1]is the mean over the horizon of the predicted probability that a collision has occurred. Because the target is cumulative, an event predicted early contributes at every later slot and weighs more than the same event predicted late, and a candidate that is predicted to avoid the event pays almost nothing.
- L753–754（W 的定标逻辑）：
  > The scale ofWis set so that a collision predicted with certainty from the first imagined step,¯ρ= 1, costs about as much as the whole discounted return of a good candidate.
- L699–701 的机制对应结果句（无碰撞场景风险项静默，L1171–1172）：
  > The risk term rises before a real collision, changes the plan there, and stays silent where none threatens.

**类比映射（推断）**：对「力学」，若要在模型中给危险定价，应该学 SAFER 的「有界概率 × 固定权重」结构而非无界惩罚项：① 惩罚上限 = W 保证一个极端风险候选最多输掉一个「好 episode」的量级，回报项仍有机会胜出（保留探索空间）；② 事件预测得越早、惩罚越重，鼓励提前规避而非最后一刻救险；③ 避开风险的候选几乎不付费，因此 planner 可以选「比数据更晚刹车/更早并线」等 coverage 边缘动作而不被惩罚淹没。这三点共同构成 SAFER 对探索-利用两难中「探索侧」的支付方案。

---
