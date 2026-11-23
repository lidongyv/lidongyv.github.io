# UniSim 精读笔记（arXiv 2310.06114）

> 唯一数据源：`papers/txt/UniSim.txt`（1102 行）。所有引文逐字复制自该 txt，行号即 txt 行号；「原文未写」= 论文中查无；【推断】= 本笔记作者的解读。

## 1. 原文信息

- 标题：`LEARNING INTERACTIVE REAL-WORLD SIMULATORS`（行 2）
- 作者与机构（行 3–5）：`Sherry Yang,1,2 Yilun Du3 Seyed Kamyar Seyed Ghasemipour2 Jonathan Tompson2 Leslie Kaelbling3 Dale Schuurmans2,4 Pieter Abbeel1`；`1UC Berkeley 2Google DeepMind 3MIT 4University of Alberta`
- 通讯邮箱（行 6）：`sherryy@{berkeley.edu, google.com}`
- arXiv ID / 版本（行 53）：`arXiv:2310.06114v3  [cs.AI]  26 Sep 2024`
- 发表状态（行 1、54）：`Published as a conference paper at ICLR 2024`（口头/海报：原文未写）
- 项目页（行 27）：`Video demos can be found at https://universal-simulator.github.io.`（行 739 另写：`Video demos can be found at anonymous-papers-submissions.github.io`）

## 2. Abstract 全文逐字（行 7–27）

> Generative models trained on internet data have revolutionized how text, image, and video content can be created. Perhaps the next milestone for generative models is to simulate realistic experience in response to actions taken by humans, robots, and other interactive agents. Applications of a real-world simulator range from controllable content creation in games and movies, to training embodied agents purely in simulation that can be directly deployed in the real world. We explore the possibility of learning a universal simulator (UniSim) of real-world interaction through generative modeling. We first make the important observation that natural datasets available for learning a real-world simulator are often rich along different dimensions (e.g., abundant objects in image data, densely sampled actions in robotics data, and diverse movements in navigation data). With careful orchestration of diverse datasets, each providing a different aspect of the overall experience, we can simulate the visual outcome of both high-level instructions such as “open the drawer” and low-level controls such as “move by∆x, ∆y” from otherwise static scenes and objects. We use the simulator to train both high-level vision-language policies and low-level reinforcement learning policies, each of which can be deployed in the real world in zero shot after training purely in simulation. We also show that other types of intelligence such as video captioning models can benefit from training with simulated experience, opening up even wider applications. Video demos can be found at https://universal-simulator.github.io.

注：行 21 原文为 `“move by∆x, ∆y”`（无空格，照抄）。

## 3. 问题与动机

**总体动机**：生成模型已能造文本/图像/视频，终极目标是模拟动作的视觉后果。逐字（行 29–35）：

> Generative models trained on internet data can now produce highly realistic text (OpenAI, 2023), image (Ramesh et al., 2022), and video (Ho et al., 2022a). Perhaps the ultimate goal of generative models is to be able to simulate the visual effects of a wide variety of actions, from how cars are driven on a street to how furniture and meals are prepared. With a real-world simulator, humans can “interact” with diverse scenes and objects, robots can learn from simulated experience without risking physical damage, and a vast amount of “real-world” like data can be simulated to train other types of machine intelligence.

**核心难题=数据壁垒**（行 36–43）：

> One roadblock to building this simulator lies in the datasets — different datasets cover different information that have to be brought together to simulate realistic experience. For instance, paired text-image data from the internet contains rich scenes and objects but little movement (Schuhmann et al., 2022; Zhai et al., 2022), video captioning and question answering data contain rich high-level descriptions but little low-level movement detail (Xu et al., 2016; Krishna et al., 2017), human activity data contains rich human action but little mechanical motion (Miech et al., 2019; Grauman et al., 2022), and robotics data contains rich robot action but are limited in quantity (Dasari et al., 2019; Mandlekar et al., 2018).

> Since different datasets are curated by different industrial or research communities for different purposes, divergence in information is natural and hard to overcome, posing difficulties to a real-world simulator that seeks to capture all visual aspects of the world. (行 43–45)

**「多数据集编排」论证**：摘要里给出图像/机器人/导航三类数据各自的贡献维度（行 15–19）：

> natural datasets available for learning a real-world simulator are often rich along different dimensions (e.g., abundant objects in image data, densely sampled actions in robotics data, and diverse movements in navigation data). With careful orchestration of diverse datasets, each providing a different aspect of the overall experience,

即【逐字归纳】：图像数据→物体丰富（abundant objects）；机器人数据→动作稠密采样（densely sampled actions）；导航数据→运动多样（diverse movements）。正文引言又把四类数据各自的「有/缺」讲全（见上引行 36–43）。解法宣言（行 46–48）：

> In this work, we propose to combine a wealth of data in a conditional video generation framework to instantiate a universal simulator (UniSim) 1. Under a unified action-in-video-out interface, the simulator enables rich interaction through fine-grained motion control of otherwise static scenes and objects.

「universal」的含义限定（行 50–51）：`Note that by “universal”, we mean the model can simulate through the unified interface of actions and videos, as opposed to being able to simulate everything. Sound, for instance, is not being simulated.`

贡献三条（行 74–82）：① `combining diverse datasets rich in along different dimensions — e.g., objects, scenes, actions, motions, language, and motor controls — in a unified action-in-video-out generative framework`；② 把该框架表述为有限历史条件下的 observation prediction model，以视频扩散模型参数化并可自回归 rollout；③ `how the simulator can enable both high-level language policies, low-level control policies, and video captioning models to generalize to the real world when trained purely in simulation, thereby bridging the sim-to-real gap.`

## 4. 方法逐机制

**模拟器定义**（行 84–86）：`We define a simulator of the real world as a model that, given some state of the world (e.g., an image frame), can take in some action as input, and produce the visual consequence of the action (in the form of a video) as output.`

### 4.1 数据构成与动作统一（Section 2.1，逐类逐字）

- **仿真执行与渲染**（行 104–110）：`While annotating actions for real-world videos is expensive, simulation engines such as Habitat (Savva et al., 2019) can render a wide variety of actions. We use datasets previously collected from these simulators, i.e., Habitat object navigation with HM3D (Ramakrishnan et al., 2021) and Language Table Data from Lynch et al. (2023) to train UniSim. We extract text descriptions as actions when available. For simulated continuous control actions, we encode them via language embeddings and concatenate the text embeddings with discretized control values.`
- **真实机器人数据**（行 111–115）：`An increasing amount of video data of real-robot executions paired with task descriptions such as the Bridge Data (Ebert et al., 2021) and data that enabled RT-1 and RT-2 (Brohan et al., 2022) are becoming increasingly available. Despite low-level control actions often being different across robots, the task descriptions can serve as high-level actions in UniSim. We further include discretize continuous controls actions when available similar to simulated robotics data.`
- **人类活动视频**（行 116–122）：`Rich human activity data such as Ego4D (Grauman et al., 2022), EPIC-KITCHENS (Damen et al., 2018), and Something-Something V2 (Goyal et al., 2017) have been curated. Different from low-level robot controls, these activities are high-level actions that humans take to interact with the world. But these actions are often provided as labels for video classification or activity recognition tasks (Goyal et al., 2017). In this case, we convert the video labels into text actions. In addition, we subsample the videos to construct chunks of observations at a frame rate that captures meaningful actions.`
- **全景扫描**（行 123–125）：`There exists a wealth of 3D scans such as Matterport3D (Chang et al., 2017). These static scans do not contain actions. We construct actions (e.g., turn left) by truncating panorama scans and utilize the information of camera poses between two images.`
- **互联网图文数据**（行 126–129）：`Paired text-image datasets such as LAION (Schuhmann et al., 2021) contain static images of a variety of objects without actions. However, the captions often contain motion information such as “a person walking”. To use image data in UniSim, we treat individual images as single-frame videos and image captions as actions.`
- **统一动作空间**（行 130–132）：`For each of these datasets, we process text tokens into continuous representations using T5 language model embeddings (Raffel et al., 2020) concatenated with low-level actions such as robot controls. This serves as the final unified action space of our simulator.`（附录 B 补充低层控制编码：行 888–889 `we encode them both as text and normalize then discretize them into 4096 bins contatenated with language embeddings (if present).`）

**训练数据明细（Table 5，行 892–906 逐字数值）**：

| 类别 | 数据集 | # Examples | Weight |
|---|---|---|---|
| Simulation | Habitat HM3D (Ramakrishnan et al., 2021) | 710 | 0.1 |
| | Language Table sim (Lynch & Sermanet, 2020) | 160k | 0.05 |
| Real Robot | Bridge Data (Ebert et al., 2021) | 2k | 0.05 |
| | RT-1 data (Brohan et al., 2022) | 70k | 0.1 |
| | Language Table real (Lynch & Sermanet, 2020) | 440k | 0.05 |
| | Miscellaneous robot videos | 133k | 0.05 |
| Human activities | Ego4D (Grauman et al., 2022) | 3.5M | 0.1 |
| | Something-Something V2 (Goyal et al., 2017) | 160k | 0.1 |
| | EPIC-KITCHENS (Damen et al., 2018) | 25k | 0.1 |
| | Miscellaneous human videos | 50k | 0.05 |
| Panorama scan | Matterport Room-to-Room scans (Anderson et al., 2018) | 3.5M | 0.1 |
| Internet text-image | LAION-400M (Schuhmann et al., 2021) | 400M | 0.05 |
| | ALIGN (Jia et al., 2021) | 400M | 0.05 |
| Internet video | Miscellaneous videos | 13M | 0.05 |

权重策略（行 889–891）：`The choice of mixture weights are either 0.1 or 0.05 without careful tuning. How data mixture weights affect simulation performance is an interesting line of future work.`

### 4.2 模型架构（observation prediction + 视频扩散）

**形式化**（行 140–147）：`We define an observation space O and an action space A... At a specific interactive step t, an agent, having observed a set of history frames ht−1 ∈ O, decides on some temporally extended action at−1 ∈ A, which can be resolved into a sequence of low-level robot commands to be executed in the real world. During the execution, the next set of video frames ot ∈ O are captured from the real world. The goal of a simulator is to predict ot from ht−1 and at−1.` 即学 `p(ot|ht−1, at−1)`。

**有限历史**（行 148–151）：`While an ideal predictive model should condition on all information of the past, i.e., (o0, a0 . . . , at−2, ot−1), through some recurrent state, we found conditioning on a finite set of frames (e.g., frames from the most recent interaction, ot−1) greatly simplifies the modeling problem.`

**自回归 rollout**（行 151–153）：`To simulate long interactions, we can sample from the observation prediction model p(ot|ht−1, at−1) autoregressively conditioned on the previously sampled observations.` 与奖励解耦（行 153–156）：`One advantage of this observation prediction model is that the simulator stays the same across all tasks and can be used in combination with any reward function, which can be separately learned.`

**扩散参数化**（行 158–165）：`We parametrize p(ot|ht−1, at−1) using diffusion models... Specifically, the reverse process learns a denoising model ϵθ(o(k)t, k|ht−1, at−1) that, conditioned on the history, generates the next observation from initial noise samples using K denoising steps. In practice, we only use previous video frames and omit previous actions as history, and concatenate previous video frames with initial noise samples o(K)t ∼ N(0, I) channelwise to serve as conditional inputs to the denoising model. To condition on an action at−1, we leverage classifier-free guidance (Ho & Salimans, 2022).`

**条件化方式（CFG 公式，行 166–168）**：

> ϵθ(o(k)t, k|ht−1, at−1) = (1 + η)ϵθ(o(k)t, k|ht−1, at−1) − ηϵθ(ot, k|ht−1), (1)
> where η controls action conditioning strength.

训练目标（行 169–171，公式中部在 txt 中乱码，两端逐字）：`we train ϵθ by minimizing LMSE = ... where ϵ ∼ N(0, I), and β(k) ∈ R are a set of K different noise levels for each k ∈ [1, K].` 采样迭代（行 172–176）：

> o(k−1)t = α(k)(o(k)t − γ(k)ϵθ(o(k)t, k|ht−1, at−1)) + ξ, ξ ∼ N(0, σ2kI), (2)
> where γ(k) is the denoising step size, α(k) is a linear decay on the current denoised sample, and σk is a time varying noise level that depends on α(k) and β(k).

**U-Net 架构**（行 177–181）：`We use the video U-Net architecture (Ho et al., 2022b) to implement UniSim by employing interleaved temporal and spatial attention and convolution layers in both the downsampling and upsampling passes. For history conditioning, we replicate the conditioning frames at all future frame indices, and concatenate the conditioning frames with the noise sample for each of the future frame to serve as input to the U-Net.`

**规模**（行 181–182）：`UniSim model has 5.6B parameters and requires 512 TPU-v3 and 20 days to train on all data.`

**分辨率结构（附录 C，行 914–922）**：`The video models in UniSim consist of one history conditioned video prediction model as the base and two additional spatial super-resolution models similar to Ho et al. (2022a). The history conditioned base model operates at temporal and spatial resolution [16, 24, 40], and the two spatial super-resolution models operate at spatial resolution [24, 40] → [48, 80] and [48, 80] → [192, 320], respectively. To condition the base video model on the history, we take 4 frames from the previous video segment and concatenate them channelwise to the noise samples inputted to the U-Net.`（行 920–922 补充：`We employ temporal attention for the forward model to allow maximum modeling flexibility but temporal convolution to the super-resolution models for efficiency reasons similar to Ho et al. (2022a).`）

**超参（Table 6，行 924–945 逐字）**：Base channels 1024；Optimizer Adam (β1 = 0.9, β2 = 0.99)；Channel multipliers 1, 2, 4；Learning rate 0.0001；Blocks per resolution 3；Batch size 256；Attention resolutions 6, 12, 24；Num attention heads 16, 16, 8；Conditioning embedding dimension 4096；Conditioning embedding MLP layers: 4；Conditioning token length 64；EMA 0.9999；Dropout 0.1；Training hardware 512 TPU-v3 chips；Training steps 1000000；Diffusion noise schedule cosine；Noise schedule log SNR range [-20, 20]；Sampling timesteps 256；Sampling log-variance interpolation γ = 0.1；Weight decay 0.0；Prediction target ϵ。

### 4.3 高/低层动作条件化

- 高层=语言指令（如 `“open the drawer”`，行 20–21）；低层=控制量（如 `“move by∆x, ∆y”`，行 21）。
- 真机数据中任务描述充当高层动作（行 114）：`the task descriptions can serve as high-level actions in UniSim`；连续控制动作离散化后拼接（行 114–115）。
- 人类活动视频的标签转换为文本动作（行 120–121）：`we convert the video labels into text actions`。
- 应用侧：VLA 策略输出低层控制 `joint movements in ∆x, ∆y`（行 376）；UniSim 可直接消费（行 379–380）：`Because UniSim can take low-level control actions as input, we can directly conduct model-based rollouts in the simulator using control actions generated by VLA policy.`

## 5. 实验与主结果

### 5.1 仿真质量（定性展示）
- Action-rich：图 3（同一初始帧→`“cut carrots”/“wash hands”/“pickup bowl”` 等，行 226–227 图注）。
- Long-horizon：图 4 为 8 步自回归交互；行 195–196：`UniSim successfully preserves objects manipulated by previous instructions (e.g., the orange and can are preserved in the drawers in Columns 4, 5, 7, 8 after being put in the drawers).`
- 多样性/随机性（行 266–271）：`UniSim can also support highly diverse and stochastic environment transitions, e.g., diverse objects being revealed after removing the towel on top (Figure 5 left), diverse object colors and locations (cups and pens in Figure 5 right), and real-world variabilities such as change in camera angles. Flexibility in diffusion models promotes simulation of highly stochastic environments that cannot be controlled by actions, so that a policy can learn to only control the controllable part (Yang et al., 2022).`
- 无多数据联合训练会失败（行 190–192）：`The model only trained on generic internet data, without action-rich manipulation data such as EPIC-KITCHENS (Damen et al., 2018), fails to simulate action-rich manipulations (Appendix F).`
- 低数据域：图 6（HM3D 仅 `only 700 training examples`，行 262–264；表 5 写 710）。

### 5.2 指标定义
- **RDG（论文明确定义，行 315–320）**：`We define the reduction in distance to goal (RDG) metric as RDG = ∥s0 − sgoal∥2 − ∥sT − sgoal∥2 / ∥s0 − sgoal∥2 , (3) where sT represents the underlying block locations after executing the policy, s0 and sgoal represents the initial and goal block locations.`
- **Succ. rate（表 3 注定义，行 345–348）**：`Percentage of successful simulated rollouts (out of 48 tasks) using the VLA policy with and without RL finetuning on Language Table (assessed qualitatively using video rollouts in the simulator).`
- **FVD/FID/Inception score/CLIP score/CIDEr**：论文未给出定义（原文未写），仅以标准指标使用（表 1、4、8、9）。
- **RL 奖励函数（附录 D.2，行 996–1002）**：`the PaLI model is finetuned to predict how many timesteps are left until the demonstration episode reaches a success state. Let us denote this function by d(o, g). The reward we use during RL training is defined as r(ot, at, ot+1, g) = −[d(ot+1, g) − d(ot, g)] · C, where C > 0 is a small constant used to stabilize training (C = 5e − 2 in this work).`

### 5.3 应用一：高层 VLM 策略（hindsight relabeling，Section 4.1）
- 数据生成（行 321–324）：`we create a total of 10k long-horizon trajectories from the simulator by doing rollouts in the simulator 3-5 times per trajectory, where each rollout corresponds to one scripted language instruction.` 监督方式（行 352–353）：`the final frame from each long-horizon rollout as a goal input and the scripted language instructions as supervision for training the VLM policy.`
- 模型：改版 PALM-E 12B，`64 TPUv3 chips for 1 day`（行 956–959）；评测设置（行 963–973）：`a subset of 3-4 blocks (randomly selected) are moved by 0.05, 0.1, or 0.2 along the x,y axes`；`x ∈ [0.15, 0.6] and y ∈ [−0.3048, 0.3048]`；每指令预测 `a video of 16 frames`，经 inverse dynamics model 恢复低层动作；`We execute 5 instructions in total`；`We 5 evaluations each with a different random seed`。
- **表 2 数值（行 331–333）**：RDG (moved) / RDG (all)：`VLM-BC 0.11 ± 0.13 / 0.07 ± 0.11`；`Simulator-Hindsight 0.34 ±0.13 / 0.34 ± 0.13`。结论（行 366–368）：`The VLM trained using long-horizon generated data performs 3-4 times better than using the original data in completing long-horizon goal-conditioned tasks.`
- **零样本真机（仿真训练策略）**（行 354–361）：`Despite the VLM policy only being trained on simulated data, it is able to produce effective high-level language actions given an initial and goal image from the real Language Table domain where the data for training the simulator was collected. The simulator can simulate video trajectories from the initial real observation, from which robot actions are recovered using an inverse dynamics model and executed on the real robot. ... We see that the simulated video trajectory is successfully translated to robot actions in the real world.`（真机成功率数值：原文未写，仅定性展示于图 7/10。）

### 5.4 应用二：低层 RL 策略（model-based RL，Section 4.2）
- 基线 VLA：PaLI 3B 经 BC 微调，输出 `joint movements in ∆x, ∆y`（行 375–379）；Stage1 超参：`learning rate 0.001, dropout rate 0.1, and batch size 128 ... for 300k gradient steps with 1k warmup steps`（行 993–994）。
- RL：REINFORCE（行 400–402）：`We then use the REINFORCE algorithm (Williams, 1992) to optimize the VLA policy, treating the rollouts from the simulator as the on-policy rollouts from the real environment and use the learned reward model to predict rewards from simulated rollouts.` Table 7：Max steps per episode 100；Number of actor processes 64；Number of image history stack 2；Learner batch size 64；Discounting factor γ 0.9。
- 仿真 sanity check（行 403–406）：`we first do a sanity check on simulating real-robot executions by applying low-level control actions (e.g., ∆x = 0.05, δy= 0.05) repeatedly for 20-30 environment steps to move the endpoint left, right, down, up, and diagonally in Figure 8 (top two rows). We see that the simulated rollouts capture both the endpoint movements and the physics of collision.`
- **表 3 数值（行 342–344）**：Succ. rate (all) / Succ. rate (pointing)：`VLA-BC 0.58 / 0.12`；`Simulator-RL 0.81 / 0.71`（48 任务）。行 408–409：`RL training significantly improves the performance of the VLA policy across a wide set of tasks, especially in tasks such as “point to blue block”.`
- **零样本真机部署（核心结果，行 409–411）**：`We then directly deploy the RL policy trained in the simulator onto the real robot in zero-shot, and observe successful task executions as shown in Figure 8 (bottom row).`（真机成功率数值：原文未写；定性结果见图 8/11/12。）
- 机制支撑（引言，行 67–69）：`This is enabled by using the simulator that is nearly visually indistinguishable from the real world, achieving one step towards bridging the sim-to-real gap in embodied learning (Rusu et al., 2017).`

### 5.5 应用三：视频 captioning（Section 4.3）
- 设置（行 434–438）：`We finetune PaLI-X (Chen et al., 2023), a VLM with 55B parameters pretrained on a broad set of image, video, and language tasks, to caption a set of videos generated by UniSim using texts from the training split of ActivityNet Captions (Krishna et al., 2017).` 数据量（行 1033–1036）：`the training split of ActivityNet Captions which consists of 30,740 text-video examples after the 50/25/25% train/val1/val2 split`；`For each of the 30,740 text, we generate 4 videos from UniSim ... As a result, we have 4X amount of the original training data (in terms the number of videos).`
- **表 4 数值（CIDEr，行 414–417）**：

| 训练数据 | Activity | MSR-VTT | VATEX | SMIT |
|---|---|---|---|---|
| No finetune | 15.2 | 21.91 | 13.31 | 9.22 |
| Activity（真数据） | 54.90 | 24.88 | 36.01 | 16.91 |
| Simulator（纯生成数据） | 46.23 | 27.63 | 40.03 | 20.58 |

- 结论（行 442–448）：`Purely finetuning on generated data drastically improves the captioning performance from no finetuning at all on ActivityNet (15.2 to 46.23), while achieving 84% performance of finetuning on true data. Furthermore, PaLI-X finetuned on generated data transfers better to other captioning tasks such as MSR-VTT (Xu et al., 2016), VATEX (Wang et al., 2019), and SMIT (Monfort et al., 2021) than PaLI-X finetuned on true data, which tends to overfit to ActivityNet. These results suggest that UniSim can serve as an effective data generator for improving broader vision-language models.`

## 6. 消融（全部抄录）

**表 1：历史条件化消融（Ego4D，行 247–250）**

| Condition | FID ↓ | FVD ↓ | IS ↑ | CLIP ↑ |
|---|---|---|---|---|
| 1 frame | 59.47 | 315.69 | 3.03 | 22.55 |
| 4 distant | 34.89 | 237 | 3.43 | 22.62 |
| 4 recent | 34.63 | 211.3 | 3.52 | 22.63 |

正文（行 283–287）：`Conditioning on 4 frames is better than conditioning on a single frame, but conditioning on history that is too far in the past (4 frames with exponentially increasing distances) can hurt performance. Increasing the number of conditioning frames beyond 4 did not further improve performance on Ego4D, but it could be helpful for applications that require memory from distant past (e.g., navigation for retrieval).`

**表 8：数据集消融（附录 E.1，行 1048–1051；1024 个 test split 样本）**

| Dataset | FVD ↓ | CLIP ↑ |
|---|---|---|
| Internet only | 219.62 | 22.27 |
| Without internet | 307.80 | 21.99 |
| Universal simulator | 211.30 | 22.63 |

正文（行 1045–1047）：`We observe that including internet data and various activity and robot data performs the best. Removing the internet data led to significantly worse FVD, highlighting the importance of using internet data in UniSim.`

**表 9：模型规模消融（附录 E.2，行 1061–1064；1024 个 test split 样本）**

| Model size | FVD ↓ | CLIP ↑ |
|---|---|---|
| 500M | 277.85 | 22.08 |
| 1.6B | 224.61 | 22.27 |
| 5.6B | 211.30 | 22.63 |

正文（行 1056–1058）：`We found that while increasing the model size improves the video modeling performance, the amount of improvement measured by FVD plateaus as the model gets bigger, which is slightly disappointing from a scaling point of view.`

**低数据域消融（Section 3.2，行 288–295）**：`During joint training of UniSim on diverse data, we found that naïvely combining datasets of highly varying size can result in low generation quality in low-data domains. While we can increase the weight of these domains in the data mixture during training, we found that attaching a domain identifier such as the name of the dataset to the actions being conditioned on improves generation quality in low-data domains, as shown in Figure 6. While such domain identifier improves in-distribution generation quality, we found domain-specific identifiers to hurt generalization to other domains, and should only be applied with the test domain is in distribution of the training domain.`

**失败对照（附录 F）**：图 13 注（行 1070–1072）与图 14 注（行 1099–1101）说明只训 SSV2、只训互联网数据、或只训 RT-1 数据时仿真失败。

## 7. 局限/未来工作（原文逐字，行 484–501）

> Our simulator has a few limitations that call for future work:
> • Hallucination. When an action is unrealistic given the scene (e.g., “wash hands” is given to a tabletop robot), we observe hallucinations (e.g., the table turns into a sink or the view turns away from the tabletop robot and a sink shows up). Ideally, we want UniSim to detect actions that are not possible to simulate as opposed to hallucinating unrealistic outcomes.
> • Limited memory. The simulator conditioned on a few frames of the recent history cannot capture long-term memory (e.g., an apple in a drawer could disappear when the drawer is opened if putting the apple in the drawer is not a part of the history for conditioning). How much history to condition on depends on the application of the (e.g., whether the simulator will be used for policy learning in a near-Markov setting or question answering that requires long-term memory).
> • Limited out-of-domain generalization. This is especially true for domains that are not represented in the training data. For instance, the simulator is mostly trained on 4 robot morphologies, and its ability to generalize to an unseen robot is limited. Further scaling up training data could help, as the training data is nowhere near all the video data available on the internet.
> • Visual simulation only. Our simulator is not suitable for environments where actions do not cause visual observation change (e.g., different forces in grasping a static cup). A true universal simulator should capture all aspects of the world beyond visual experience (e.g., sound, sensory, etc).

结论（行 480–483）：`We have shown it is possible to learn a simulator of the real world in response to various action inputs ranging from texts to robot controls. UniSim can simulate visually realistic experiences for interacting with humans and training autonomous agents. We hope UniSim will instigate broad interest in learning and applying real-world simulators to improve machine intelligence.`

## 8. 叙事关联【推断，5 条】

1.【推断】UniSim 是「通用模拟器/数据复用」方向最直接的论文级证据：它把论文自身明确的「多数据集编排」主张（不同数据各贡献一个维度）落成了可训、可评测的单一模型，且摘要把 orchestration 作为核心贡献之一。支持原句（行 15–19）：`We first make the important observation that natural datasets available for learning a real-world simulator are often rich along different dimensions (e.g., abundant objects in image data, densely sampled actions in robotics data, and diverse movements in navigation data). With careful orchestration of diverse datasets, each providing a different aspect of the overall experience...`（注：论文通篇不用「data reuse/数据复用」措辞，该词是本笔记的概括标签，原文未写。）

2.【推断】「复用」是二阶的：先复用异构数据集学出模拟器，再用模拟器产出仿真经验去训练下游模型（VLM 策略/RL/视频 captioning），等于把已有数据间接复用到新任务。支持原句（行 306–307）：`UniSim can generate large amounts of training data for VLM policies through hindsight relabeling.` 及（行 447–448）：`These results suggest that UniSim can serve as an effective data generator for improving broader vision-language models.`

3.【推断】UniSim 把「数据稀缺是 RL 落地瓶颈」作为叙事锚点，通用模拟器被定位为「无限环境访问」的替代品，这正是数据复用叙事的社会价值主张。支持原句（行 459–461）：`We focus on this exact bottleneck by learning universal simulators of the real world, enabling realistic and unlimited “environment” access for training sophisticated agents interactively.`

4.【推断】编排并非简单拼接：低数据域会退化、需要 domain identifier；移除互联网数据 FVD 显著恶化——说明「复用哪些数据、如何复用」本身就是该方法的核心技术贡献而非工程细节。支持原句（行 1045–1047）：`Removing the internet data led to significantly worse FVD, highlighting the importance of using internet data in UniSim.` 及（行 288–292）：`naïvely combining datasets of highly varying size can result in low generation quality in low-data domains... we found that attaching a domain identifier such as the name of the dataset to the actions being conditioned on improves generation quality in low-data domains.`

5.【推断】「仿真训练→零样本真机」是 UniSim 数据复用价值的最强实证闭环：摘要与 4.1/4.2 均宣称 purely in simulation 训练可零样本部署真机，若成立则等于「数据复用可以绕开真机数据采集」。支持原句（行 23–25）：`We use the simulator to train both high-level vision-language policies and low-level reinforcement learning policies, each of which can be deployed in the real world in zero shot after training purely in simulation.` 及（行 409–411）：`We then directly deploy the RL policy trained in the simulator onto the real robot in zero-shot, and observe successful task executions as shown in Figure 8 (bottom row).`（【推断】注意：真机部分无量化成功率，证据强度为定性级。）

## 9. 图注/表注逐字抄录

- **图 1**（行 56–58）：`Figure 1: A universal simulator (UniSim). The simulator of the real-world learns from broad data with diverse information including objects, scenes, human activities, motions in navigation and manipulation, panorama scans, and simulations and renderings.`
- **图 2**（行 98–103）：`Figure 2: Training and inference of UniSim. UniSim is a video diffusion model trained to predict the next (variable length) set of observation frames ( ot) given observations from the past (e.g., ot−1) and action input at−1. UniSim can handle temporally extended actions in various modalities such as motor controls (∆x1, ∆ω1, ∆x2, ...), language descriptions (“wipe table”), and actions extracted from camera motions and other sources. Each dotted arrow indicates concatenating the initial noise sample for the next video segment with the previous frame.`
- **图 3**（行 226–227）：`Figure 3: Action-rich simulations. UniSim can support manipulation actions such as “cut carrots”, “wash hands”, and “pickup bowl” from the same initial frame (top left) and other navigation actions.`
- **图 4**（行 244–246）：`Figure 4: Long-horizon simulations. UniSim sequentially simulates 8 interactions autoregressively. The simulated interactions maintain temporal consistency across long-horizon interactions, correctly preserving objects and locations (can on counter in column 2-7, orange in drawer in column 4-5).`
- **表 1**（行 251–255）：`Table 1: Ablations of history conditioning using FVD, FID, and Inception score, and CLIP score on Ego4D. Conditioning on multiple frames is better than on a single frame, and recent history has an edge over distant history.`
- **图 5**（行 280–282）：`Figure 5: Diverse and stochastic simulations. On the left, we use text to specify the object being revealed by suffixing “uncovering” with the object name. On the right, we only specify “put cup” or “put pen”, and cups and pens of different colors are sampled as a result of the stochastic sampling process during video generation.`
- **图 6**（行 261–265）：`Figure 6: Simulations of low-data domains using the Habitat object navigation using HM3D dataset (Ramakrishnan et al., 2021) with only 700 training examples. Prefixing language actions with dataset identifier leads to video samples that complete the action (top).`
- **图 7**（行 327–330）：`Figure 7: Long-horizon simulation. A VLM poliy generates high-level language actions (first row) which are executed in the simulator (middle row) similar to how they are executed in the real world (bottom row) using the Language Table robot. The VLM trained on data from the simulator complete long-horizon tasks by successfully moving three blocks (blue, green, yellow) to match their target location in the goal image.`
- **表 2**（行 334–341）：`Table 2: Evaluation of long-horizon actions. Reduction in distance to goal (RDG) defined in Equation 3 across 5 evaluation runs of VLM trained using simulated long-horizon data (bottom row) compared to VLM trained on original short-horizon data (top row). Using the simulator performs much better both in RGD of moved blocks (left) and RGD in all blocks (right).`
- **表 3**（行 345–351）：`Table 3: Evaluation of RL policy. Percentage of successful simulated rollouts (out of 48 tasks) using the VLA policy with and without RL finetuning on Language Table (assessed qualitatively using video rollouts in the simulator). Simulator-RL improves the overall performance, especially in pointing-based tasks which contain limited expert demonstrations.`
- **图 8**（行 396–399）：`Figure 8: [Top] Simulation from low-level controls . UniSim supports low-level control actions as inputs to move endpoint horizontally, vertically, and diagonally. [Bottom] Real-robot execution of an RL policy trained in simulation and zero-shot onto the real Language Table task. The RL policy can successfully complete the task of “moving blue cube to green circle”.`
- **表 4**（行 418–425）：`Table 4: VLM trained in UniSim to perform video captioning tasks. CIDEr scores for PaLI-X finetuned only on simulated data from UniSim compared to no finetuning and finetuning on true video data from ActivityNet Captions. Finetuning only on simulated data has a large advantage over no finetuning and transfers better to other tasks than finetuning on true data.`
- **图 9**（行 756–757）：`Figure 9: Additional results on long-horizon interaction with humans and robots similar to Figure 4. UniSim can generate consistent video rollouts across 3-4 high-level language actions.`
- **图 10**（行 761–763）：`Figure 10: Additional results (similar to Figure 7) on applying UniSim to train vision-language policies to complete long-horizon tasks. VLM finetuned with hindsight labeled data is able to generate long-horizon instructions that moves two or three blocks successfully to match their location in the goal image.`
- **图 11**（行 833）：`Figure 11: First real observations and last simulated observations of rolling out the RL policy trained in UniSim.`
- **图 12**（行 875–878）：`Figure 12: First real observations and last real observations of executing the RL policy trained from UniSim in the real world in zero-shot. Middle plot also shows the output of the learned reward model (steps-to-completion) during policy execution, where step 0 corresponds to the top plot (initial observation) and step 70 corresponds to the bottom plot (final observation).`
- **表 5**（行 907）：`Table 5: Dataset name, number of training examples, and mixture weights used for training UniSim.`
- **表 6**（行 946）：`Table 6: Hyperparameters for training UniSim diffusion model.`
- **表 7**（行 1027）：`Table 7: Hyperparameters for training the VLA RL policy using the ACME framework.`
- **表 8**（行 1052–1053）：`Table 8: Ablations of datasets using FVD and CLIP score on the held-out test split. Including internet data and diverse human activity and robot data in UniSim achieves the best FVD and CLIP scores.`
- **表 9**（行 1065–1066）：`Table 9: Ablations of model size using FVD and CLIP score on the held-out test split. The largest model achieves the best FVD and CLIP scores.`
- **图 13**（行 1070–1072）：`Figure 13: Failed environment simulation from the action “uncover bottle” without training on broad data as in UniSim. Top two videos are generated from only training on SSV2. Bottom two videos are generated from only training on generic internet data (without SSV2, EpicKitchen, Ego4D, and various robotics dataset).`
- **图 14**（行 1099–1101）：`Figure 14: When the text-to-video model behind UniSim is only trained on data from Brohan et al. (2022) as opposed incorporating broad data from the internet and other manipulation datasets, long-horizon interaction simulations fail half of the time (red text).`
