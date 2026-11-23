# UniPi 精读笔记（Learning Universal Policies via Text-Guided Video Generation）

> 数据源：C:\Users\Admin\Desktop\design\auto_docs\papers\txt\UniPi.txt（共 791 行）。所有引用均为 txt 逐字原文，行号为该文件行号。禁止联网核实，本笔记不含任何外部知识补充。

## 1) 原文信息

- 标题：`Learning Universal Policies via Text-Guided Video Generation`（L1–2）
- 作者：`Yilun Du∗ †‡, Mengjiao Yang *‡§, Bo Dai ‡¶, Hanjun Dai ‡, Ofir Nachum ‡, Joshua B. Tenenbaum †, Dale Schuurmans ‡||, Pieter Abbeel §`（L3–4）
- 机构：`MIT† Google DeepMind‡ UC Berkeley§ Georgia Tech¶ University of Alberta||`（L5）
- 项目页：`https://universal-policy.github.io/`（L6）
- 版本：`37th Conference on Neural Information Processing Systems (NeurIPS 2023).`（L44）；`arXiv:2302.00111v3 [cs.AI] 20 Nov 2023`（L45）

## 2) Abstract 全文逐字（L8–23）

> A goal of artificial intelligence is to construct an agent that can solve a wide variety
> of tasks. Recent progress in text-guided image synthesis has yielded models with
> an impressive ability to generate complex novel images, exhibiting combinatorial
> generalization across domains. Motivated by this success, we investigate whether
> such tools can be used to construct more general-purpose agents. Specifically, we
> cast the sequential decision making problem as a text-conditioned video generation
> problem, where, given a text-encoded specification of a desired goal, a planner
> synthesizes a set of future frames depicting its planned actions in the future, after
> which control actions are extracted from the generated video. By leveraging text
> as the underlying goal specification, we are able to naturally and combinatorially
> generalize to novel goals. The proposed policy-as-video formulation can further
> represent environments with different state and action spaces in a unified space of
> images, which, for example, enables learning and generalization across a variety of
> robot manipulation tasks. Finally, by leveraging pretrained language embeddings
> and widely available videos from the internet, the approach enables knowledge
> transfer through predicting highly realistic video plans for real robots2.

（注：`real robots2` 中的 `2` 为脚注标记，指 L43 的项目页链接。）

## 3) 问题与动机：为什么「视频即策略」能跨任务/跨本体泛化

**挑战在于环境多样性。** 原文：

> However, training such agents faces the inherent challenge of environmental diversity, since different
> environments operate with distinct state action spaces (e.g., the joint space and continuous controls
> in MuJoCo are fundamentally different from the image space and discrete actions in Atari). Such
> diversity hampers knowledge sharing, learning, and generalization across tasks and environments.（L31–34）

**解法是把视频和文本当作通用接口。** 该句跨 L40–41 与 L69（中间被 Figure 1 插图隔断），逐字为：

> In this work, we address the challenges in environment diversity and reward specification by lever-
> aging video (i.e., image sequences) as a universal interface for conveying action and observation（L40–41）
> behavior in different environments, and text as a universal interface for expressing task descriptions.（L69）

**MDP 抽象的三条局限**（L115–124）：

> i) The lack of a universal state interface across different control environments. In fact, since
> different environments typically have separate underlying state spaces, one would need to a
> construct a complex state representation to represent all environments, making learning difficult.（L115–117）
>
> ii) The explicit requirement of a real-valued reward function in an MDP.（L118）
>
> iii) The dynamics model in an MDP is environment and agent dependent. Specifically, the dynamics
> model T (s′|s, a) that characterizes the transition between states(s, s′) under action a is explicitly
> dependent to the environment and action space of the agent, which can be significantly different
> between different agents and tasks.（L121–124）

**为此提出 UPDP 抽象：**

> Our abstraction, termed Unified Predictive Decision Pro-
> cess (UPDP), exploits images as a universal interface across environments, texts as task specifiers to
> avoid reward design, and a task-agnostic planning module separated from environment-dependent
> control to enable knowledge sharing and generalization.（L126–130）

**跨任务/跨本体泛化的直接理由**（两点）：其一，图像空间统一了异构状态/动作空间——

> The proposed policy-as-video formulation can further
> represent environments with different state and action spaces in a unified space of
> images, which, for example, enables learning and generalization across a variety of
> robot manipulation tasks.（L18–21）

其二，规划与动作解耦使规划器与本体无关——

> Another benefit of a UPDP over an MDP is that UPDP isolates the video-based planning withρ(·|x0, c)
> from the deferred action selection using π(·|{xh}Hh=0, c). This design choice isolates planning de-
> cisions from action-specific mechanisms, allowing the planner to be environment and agent agnostic.（L147–149）

另见：`Since the space of video observations X H and task descriptions C are both naturally shared across environments and easily interpretable by humans, any video-based planner ρ(·|x0, c) can be conveniently reused, transferred and debugged.`（L144–146）

## 4) 方法逐机制

**4.1 UPDP 形式化（L131–142）。** `Formally, we define a UPDP to be a tuple G = ⟨X , C, H, ρ⟩, where X denotes the observation space of images, C denotes the space of textual task descriptions, H ∈ Nis a finite horizon length, and ρ(·|x0, c) : X × C →∆(X H ) is a conditional video generator.`（L131–133）`Intuitively, ρ synthesizes H-step image trajectories that illustrate possible paths for completing a target task c.`（L135–136）动作侧：`Given a UPDP G, we define a trajectory-task conditioned policy π(·|{xh}Hh=0, c) : X H+1 × C → ∆(AH ) to be a conditional distribution over H-step action sequences AH.`（L137–138）离线数据：`we will consider an offline RL scenario where we have access to a dataset of existing experience D = {(xi, ai)H−1i=0 , xH , c}nj=1 from which both ρ(·|x0, c) and π(·|{xh}Hh=0, c) can be estimated.`（L140–142）

**4.2 UniPi 两个组件（L196–203）：**

> (i) a diffusion model for the universal video-
> based planner ρ(·|x0, c), which synthesizes
> videos conditioned on the first frame and
> task descriptions; and (ii) a task-specific
> action generator π(·|{xh}Hh=0, c), which
> infers actions sequences from generated
> videos through inverse dynamics modeling.

**4.3 文本条件视频生成（planner）。** 与普通文生视频的区别：

> Planning through
> video generation is more challenging as it requires models to both be able to generate constrained
> videos that start at a specified image, and then complete the target task.（L210–212）

首帧条件：`Instead, we found it more effective to explicitly train a constrained video synthesis model by providing the first frame of each video as explicit conditioning context during training.`（L221–222）扩散实现与分类器无关引导：

> Denote the conditional denoiser as s(τk, k|c, x0). We leverage classifier-free guidance [18] and use
> ˆs(τk, k|c, x0) = (1 +ω)s(τk, k|c, x0) − ωs(τk, k) as the denoiser in the reverse process for sampling,
> where ω controls the strength of the text and first-frame conditioning.（L173–175）

训练设置：`To train the text-conditioned video generation model, we utilize the video diffusion algorithm in [19], where pretrained language features from T5 [22] are encoded.`（L250–251）

**4.4 轨迹一致性（Tiling，L226–233）：**

> To enforce environment consistency in conditional video synthesis, we
> provide, as additional context, the observed image when denoising each frame in the synthesized
> video. In particular, we re-purpose a temporal super-resolution video diffusion architecture, and
> provide as context the conditioned visual observation tiled across time, as the opposed to a low
> temporal-resolution video for denoising at each timestep. In this model, we directly concatenate each
> intermediate noisy frame with the conditioned observed image across sampling steps, which serves
> as a strong signal to maintain the underlying environment state across time.

**4.5 层级规划（L234–243）：** 动机是搜索空间指数爆炸——`directly generating a set of actions to reach a goal state quickly becomes intractable due to the exponential blow-up of the underlying search space.`（L235–236）机制：

> We first generate videos at a coarse level by sparsely
> sampled videos (“abstractions”) of the desired behavior along the time axis. Then we refine the videos
> to represent valid behavior in the environment by super-resolving videos across time. Meanwhile,
> coarse-to-fine super-resolution further improves consistency via interpolation between frames.（L240–243）

**4.6 动作逆模型 / 轨迹提取（L254–264）。**

> Given a set of synthesized videos, we may train a small task-specific inverse-dynamics model to
> translate frames into a set of actions as described below.（L254–255）
>
> Inverse Dynamics. We train a small model to estimate actions given input images. The training of
> the inverse dynamics is independent from the planner and can be done on a separate, smaller and
> potentially suboptimal dataset generated by a simulator.（L256–258）
>
> Action Execution. Finally, we generate an action sequence given x0 and c by synthesizing
> H image frames and applying the learned inverse-dynamics model to predict the corresponding
> H actions. Inferred actions can then be executed via closed-loop control, where we generate H
> new actions after each step of action execution (i.e., model predictive control), or via open-loop
> control, where we sequentially execute each action from the intially inferred action sequence. For
> computational efficiency, we use an open-loop controller in all our experiments in this paper.（L259–264）

逆模型规格（附录 A.2，L703–708）：`UniPi’s inverse dynamics model is trained to directly predict the 7-dimensional controls of the simulated robot arm from an image observation mean squared error.`；结构为 `a 3x3 convolutional layer, 3 layers of 3x3 convolutions with residual connection, a mean-pooling layer across all pixel locations, and an MLP layer of (128, 7) channels`；训练 `the Adam optimizer with gradient norm clipped at 1 and learning rate 1e-4 for a total of 2M steps where linear warmup is applied to the first 10k steps.`

**4.7 「任务规划 vs 动作规划」两阶段。** 注意：原文没有直接使用「task planning / action planning」这一对术语，最接近的两阶段表述是 Abstract 的 `a planner synthesizes a set of future frames depicting its planned actions in the future, after which control actions are extracted from the generated video.`（L14–16），以及 4.2 的「规划组件＋动作生成组件」与 L147–149 的「planning decisions 与 action-specific mechanisms 隔离」。规划自身的两阶段（粗→细）见 4.5。

**4.8 训练规模（附录 A.1）。** `We train each of our video diffusion models for 2M steps using batch size 2048 with learning rate 1e-4 and 10k linear warmup steps. We use 256 TPU-v4 chips for our first-frame conditioned generation model and temporal super resolution model.`（L690–692）`We use T5-XXL [22] to process input prompts which consists of 4.6 billion parameters.`（L694）仿真实验用 `10x48x64 videos (skipping every 8 frames) with 1.7B parameters and a temporal super resolution of 20x48x64 (skipping every 4 frames) with 1.7B parameters.`（L696–697）

## 5) 实验与主结果（数值全部逐字抄录）

**5.1 组合策略合成（Section 4.1）。** 任务：`In this task, a robot must manipulate blocks in an environment to satisfy language instructions, i.e., put a red block right of a cyan block .`（L276–277）动作空间为连续关节：`we predict actions in the continuous robotic joint space for both the baselines and our approach.`（L298–299）数据划分与规模：`one set of instructions (70%) that is seen during training, and another set (30%) that is only seen during testing. The precise locations of individual blocks, bowls, and plates in the environment are fully randomized in each environment iteration. We train the video model on 200k example videos of generated language instructions in the train set.`（L300–303）演示来源：`We constructed demonstrations of videos in this task by using a scripted agent.`（L304–305）

**Table 1（L282–287）：**

| Model | Seen Place | Seen Relation | Novel Place | Novel Relation |
|---|---|---|---|---|
| State + Transformer BC [24] | 19.4 ± 3.7 | 8.2 ± 2.0 | 11.9 ± 4.9 | 3.7 ± 2.1 |
| Image + Transformer BC [24] | 9.4 ± 2.2 | 11.9 ± 1.8 | 9.7 ± 4.5 | 7.3 ± 2.6 |
| Image + TT [25] | 17.4 ± 2.9 | 12.8 ± 1.8 | 13.2 ± 4.1 | 9.1 ± 2.5 |
| Diffuser [21] | 9.0 ± 1.2 | 11.2 ± 1.0 | 12.5 ± 2.4 | 9.6 ± 1.7 |
| UniPi (Ours) | 59.1 ± 2.5 | 53.2 ± 2.0 | 60.1 ± 3.9 | 46.1 ± 3.0 |

语言目标组合泛化证据（原文结论句）：`In Table 1, we find that UniPi generalizes well to both seen and novel combinations of language prompts .`（L362–363）机制解释：`Since the language description of a task is often highly correlated with control actions, text-conditioned video generation naturally focuses generation on action relevant parts of the video.`（L77–79）；`The rich combinatorial nature of language can be leveraged to synthesize novel combinatorial behaviors in the environment. This enables the proposed approach to rearrange objects to new unseen combinations of geometric relations, as shown in Section 4.1.`（L82–84）

**5.2 多环境迁移（Section 4.2）。** 设置：`We train our method using demonstrations across a set of 10 separate tasks from [28], and evaluate the ability of our approach to transfer to 3 different test tasks. Using a scripted oracle agent, we generate a set of 200k videos of language execution in the environment.`（L400–402）

**Table 3（L417–423）：**

| Model | Place Bowl | Pack Object | Pack Pair |
|---|---|---|---|
| State + Transformer BC | 9.8 ± 2.6 | 21.7 ± 3.5 | 1.3 ± 0.9 |
| Image + Transformer BC | 5.3 ± 1.9 | 5.7 ± 2.1 | 7.8 ± 2.6 |
| Image + TT | 4.9 ± 2.1 | 19.8 ± 0.4 | 2.3 ± 1.6 |
| Diffuser | 14.8 ± 2.9 | 15.9 ± 2.7 | 10.5 ± 2.4 |
| UniPi (Ours) | 51.6 ± 3.6 | 75.5 ± 3.1 | 45.7 ± 3.7 |

结论句：`The UniPi approach is able to generalize and synthesize new videos and decisions of different language tasks, and can generate videos consisting of picking different kinds of objects and different colored objects.`（L431–434）

**5.3 真实世界迁移（Section 4.3）。** 数据：`The pretraining dataset uses the same data as [19], which consists of 14 million video-text pairs, 60 million image-text pairs, and the publicly available LAION-400M image-text dataset. The robotic dataset is adopted from the Bridge dataset [29] with 7.2k video-text pairs, where we use the task IDs as texts. We partition the 7.2k video-text pairs into train (80%) and test (20%) splits.`（L439–443）评测：`We report the CLIP scores, FIDs, and FVDs (averaged across frames and computed on 32 samples) of UniPi trained on Bridge data, with and without pretraining.`（L446–447）

**Table 4（L476–478）：**

| Model (24x40) | CLIP Score ↑ | FID ↓ | FVD ↓ | Success ↑ |
|---|---|---|---|---|
| No Pretrain | 24.43 ± 0.04 | 17.75 ± 0.56 | 288.02 ± 10.45 | 72.6% |
| Pretrain | 24.54 ± 0.03 | 14.54 ± 0.57 | 264.66 ± 13.64 | 77.1% |

成功率是代理指标：`we train a success classifier that takes in the last frame of a generated video and predicts whether the task is successful or not. We find that training a model from scratch achieves 72.6% success while finetuning from a pretrained model improves performance to 77.1%.`（L483–486）泛化结论：`We find that internet-scale pretraining enables UniPi to generalize to novel task commands and scenes in the test split not seen during training, whereas UniPi trained only on task-specific robot data fails to generalize.`（L487–489）

## 6) 消融（全部抄录）

**Table 2（L352–357）**（原文表头即拼作 `Heirarchy`，逐字保留）：

| Frame Condition | Frame Consistency | Temporal Heirarchy | Place | Relation |
|---|---|---|---|---|
| No | No | No | 13.2 ± 3.2 | 12.4 ± 2.4 |
| Yes | No | No | 52.4 ± 2.9 | 34.7 ± 2.6 |
| Yes | Yes | No | 53.2 ± 3.0 | 39.4 ± 2.8 |
| Yes | Yes | Yes | 59.1 ± 2.5 | 53.2 ± 2.0 |

消融对象（原文）：`we study the effect of conditioning the video generative model on the first observation frame (frame condition), tiling the observed frame across timesteps (frame consistency) and super-resolving video generation across time (temporal hierarchy). In settings where frame consistency is not enforced, we provide a zeroed out image as context to the non-start frames in a video.`（L382–390）结论：`We found that all components of UniPi are crucial for good performance. We found that frame conditioning and consistency enabled videos to be consistent with the observed image and temporal hierarchy enabled more detailed video plans.`（L390–392）

## 7) 局限/未来工作（原文逐字，L539–547）

> Limitations. Our current approach has several limitations. First, the underlying video diffusion
> process can be slow, it can take a minute to generate highly photorealistic videos. This slowness can
> be overcome by distilling the diffusion process into a faster sampling network [ 44], which in our
> initial experimentation resulted in a 16x speed-up. UniPi may further be sped up with by faster speed
> samplers in diffusion models. Second, the environments considered in this work are generally fully
> observed. In partially observable environments, video diffusion models might make hallucination of
> objects or movements that are unfaithful or not in the physical world. Integrating video models with
> semantic knowledge about the world may help resolve this issue, and the integration of UniPi with
> LLMs would be an interesting direction of future work.

## 8) 叙事关联【推断】：「运动学轨迹由生成视频蒸馏而来（free demonstrations）」

先作诚实声明：原文中「distill」一词仅出现在采样加速语境——`This slowness can be overcome by distilling the diffusion process into a faster sampling network [44]`（L540–541），与「从视频蒸馏运动学轨迹」无关；「free demonstrations」字面亦未出现（`原文未写`）。与「视频提供免费演示、再由生成视频提取动作轨迹」这一叙事最直接对应的原文证据如下（共 5 条，均为【推断】性关联，各附原句）：

1. **互联网视频被原文明确称为"demonstrations"（带引号的演示仓库）。** `By pretraining a video generation model on a large-scale text-video dataset recovered from the internet, one can recover a vast repository of “demonstrations” that aid the construction of a text-conditioned policy in novel environments.`（L96–99）
2. **运动学动作确实是从生成的视频中提取/回归出来的。** `after which an inverse dynamics model is used to extract the underlying actions from the generated video.`（L72–73）；`A set of underlying actions are then regressed from the synthesized frames and used to construct a policy to implement the planned trajectory.`（L76–77）；Figure 2 图注 `Images are converted to actions using an inverse dynamics model.`（L190–191）
3. **动作监督可以「廉价」获得：逆模型独立于规划器，仅需仿真器生成的小数据集，不需要与视频等量的动作标注。** `The training of the inverse dynamics is independent from the planner and can be done on a separate, smaller and potentially suboptimal dataset generated by a simulator.`（L256–258）；`As only a subset of the video dataset contained action annotations, we trained the inverse-dynamics model on action annotations from 20k generated videos.`（L741–742）；`We trained the inverse dynamics on action annotation across the 200k generated videos.`（L753）
4. **跨本体（human↔robot）知识迁移以图像为统一状态与动作空间。** `We believe this paper is the first to propose images as a universal state and action space to enable broad knowledge transfer across environments, tasks, and even between humans and robots.`（L532–533）；`the approach enables knowledge transfer through predicting highly realistic video plans for real robots2.`（L22–23）
5. **预训练视频生成模型提供「演示级」先验，使未见任务也能合成正确计划（from-scratch 失败）。** `As shown in Table 4, UniPi with pretraining achieves significantly higher FID and FVD and a marginally better CLIP score than UniPi without pretraining, suggesting that pretraining on non-robot data helps with generating plans for robots.`（L448–450）；`Internet pretraining enables UniPi to synthesize videos of tasks not seen in training. In contrast, a model trained from scratch incorrectly generates plans of different tasks.`（L467–469，Figure 8 图注）

## 9) 图注/表注逐字抄录

**图注：**

- Figure 1（L65–68）：`Text-Conditional Video Generation as Universal Policies. Text-conditional video generations enables us to train general purpose policies on wide sources of data (simulated, real robots and YouTube) which may be applied to downstream multi-task settings requiring combinatorical language generalization, long-horizon planning, or internet-scale knowledge.`（原文即 `generations enables`、`combinatorical`）
- Figure 2（L189–191）：`Given an input observation and text instruction, we plan a set of images representing agent behavior. Images are converted to actions using an inverse dynamics model.`
- Figure 3（L297）：`Combinatorial Video Generation. Generated videos for unseen language goals at test time.`
- Figure 4（L312–314）：`Action Execution. Synthesized video plans and executed actions in the simulated environment. The two video plans roughly align with each other.`
- Figure 5（L350–351）：`Multitask Video Generation. Generated video plans on new test tasks in the multitask setting. UniPi is able to synthesize plan across a set of environments.`
- Figure 6（L415–416）：`High Fidelity Plan Generation. UniPi can generate high resolution video plans across different language prompts.`
- Figure 7（L377–379）：`Adaptable Planning. By guiding test-time sampling towards a an intermediate image, we can adapt our planning procedure to move a particular block.`（原文即 `towards a an`）
- Figure 8（L466–469）：`Pretraining Enables Combinatorial Generaliza-tion. Internet pretraining enables UniPi to synthesize videos of tasks not seen in training. In contrast, a model trained from scratch incorrectly generates plans of different tasks.`
- Figure 9（L472–475）：`Robustness to Background Change. UniPi learns to be robust to changes of underlying background, such as black cropping or the addition of photo-shopped objects.`
- Figure 10（L764–765）：`Combinatorial Video Generation. Additional results on UniPi’s generated videos for unseen language goals at test time.`
- Figure 11（L776–777）：`Multitask Video Generation. Additional results on UniPi’s generated video plans on different new tasks in the multitask setting.`
- Figure 12（L789–790）：`High Fidelity Plan Generation. Additional results on UniPi’s high resolution video plans across different language prompts.`

**表注：**

- Table 1（L288–289）：`Task Completion Accuracy in Combinatorial Environments. UniPi generalizes to seen and novel combinations of language prompts in Place (e.g., place X in Y) and Relation (e.g., place X to the left of Y) tasks.`
- Table 2（L358–359）：`Task Completion Accuracy Ablations. Each component of UniPi improves its performance. Perfor-mance reported on the seen place and relation tasks.`
- Table 3（L424–425）：`Task Completion Accuracy on Multitask Environment. UniPi generalizes well to new environments when trained on a set of different multi-task environments.`
- Table 4（L479–480）：`Video Generation Quality of UniPi on Real Environment. The use of existing data on the internet improves video plan predictions under all metrics considered.`
