# WAM 精读线②：实机与仿真验证（10 篇实验协议深读）

> 更新：2026-10-26 23:00 ｜ 任务：对 10 篇 World Action Model 论文做【实机/仿真验证协议】轴深度精读。
> 取证方式：每篇先 abs 页 → arXiv HTML v1/v2 → ar5iv（同源全文）→ 官方项目页/GitHub 补截断区，并标注来源。
> **工具限制（影响全部来源级别）**：web_fetch 对长 HTML 有固定字节截断（约 ~110KB 提取文本），长文后半（实验节/附录）经常取不到；本机 shell 出网被沙箱拦截（Invoke-WebRequest 失败），无法下载 PDF 本地解析；r.jina.ai / web.archive.org / huggingface.co 均 fetch 失败；arxiv.org 与 ar5iv.labs.arxiv.org 可达，GitHub（github.com / *.github.io）可达。
> 来源级别约定：L3=实验节原文全文核实；L2=部分正文+官方项目页/官方仓库交叉核实；L1=仅摘要/二手转述。「未取到」=抓取截断导致未拿到；「原文未写」=已获取范围内确实没有。所有引文均为原文 verbatim。

---

## 1. Fast-WAM（arXiv:2603.16666）— Do WAMs Need Test-time Future Imagination?

- **元信息**：Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao（IIIS, Tsinghua / Galaxea AI）；v1 2026-03-17，v2 2026-03-23。项目页 https://yuantianyuan01.github.io/FastWAM/ ，代码 github.com/yuantianyuan01/FastWAM。
- **来源级别**：L2（正文+全表 L3，来自前序会话 fast-wam.md 已核实存档；Figure 4 散点数值与附录 RoboTwin 逐任务 Table 3 未取到）。
- **截断位置**：arXiv HTML v1 全文此前会话已取到 §4；附录（Table 3 逐任务表、λ 取值）未取到。项目页无附录表格。

### 【仿真验证】
- 基准与设定（§4.2，README 佐证）：
  - **LIBERO**：四套件（Spatial/Object/Goal/Long），"Each task suite provides a training dataset of 500 total demonstrations"（10 任务×50 demo），"20k steps"、2000 trials（40 任务×50 episodes）；评测为 **unseen instructions**——GitHub README verbatim："We evaluate with unseen instructions, following Motus. LingBot-VA uses seen instructions instead."
  - **RoboTwin 2.0**：>50 任务，2500 clean + 25k randomized demos，30k 步，100 trials/任务；多相机拼接单图（2cam224/3cam384）。
- **关键数值表（正文 Table 1/2，逐字）**：RoboTwin：π0 65.92/58.40→62.2；π0.5 82.74/76.76→79.8；Motus 88.66/87.02→87.8；LingBot-VA 92.90/91.50→92.2；**Fast-WAM 91.88/91.78→91.8**（无具身 PT）；Joint 90.6；IDM 91.3；w.o. video co-train 82.76/84.80→83.8。LIBERO：Fast-WAM 98.2/100.0/97.0/95.2→97.6；Joint 98.5；IDM 98.0；w.o. co-train 93.5。
- 受控结论（§4.3.2 verbatim）："On RoboTwin, Fast-WAM achieves 91.8% success rate, which is highly comparable to both Fast-WAM-Joint (90.6%) and Fast-WAM-IDM (91.3%). In contrast, removing video co-training reduces performance to 83.8%…"

### 【实机验证】
- 平台：Galaxea R1 Lite；任务：**毛巾折叠（towel folding）**；数据：60h teleop（README/前序笔记）。指标：成功率 + 完成时间（Figure 4 散点图）。
- 数值（正文 §4.3.3 verbatim）："In contrast, removing video co-training causes a dramatic degradation on both metrics: Fast-WAM without video co-training drops to only 10% success rate and also requires the longest completion time among all methods."；"Fast-WAM retains low inference latency (190 ms), whereas the imagine-then-execute variants are substantially slower, especially Fast-WAM-IDM at 810 ms."（RTX 5090D V2 32GB）。**Figure 4 各方法散点数值未取到**（项目页仅提供图：https://yuantianyuan01.github.io/FastWAM/static/images/method_scatter_latency.png）。
- **扰动/泛化测试**：RoboTwin Rand. 分档（randomized demos）即仿真侧泛化；真机侧未见扰动协议（原文未写，已获取范围）。

### 【why 追问】
1. 真机毛巾折叠只有 60h 遥操作、单任务、图内数据（散点图不给表），为什么不像 RoboTwin 一样按任务×trials 出表？成功率的误差条/试验次数全不披露——读者无法判断 190ms vs 810ms 与成功率的因果（IDM 慢是否因为等待时间计入任务完成时间，而非控制质量差？）。

### 【一页总结】
Fast-WAM 用受控四臂（Joint/IDM/w.o. co-train）把「训练期视频共训」与「测试期未来想象」解缠：去掉测试时想象只掉 ~1 点（RoboTwin 91.8 vs 90.6/91.3），去掉共训掉 8 点（83.8），LIBERO 同趋势（97.6 vs 98.5/98.0 vs 93.5）；真机毛巾折叠 w.o. co-train 掉到 10%。延迟 190ms（4× 快于 IDM 810ms）。

### 【待核实】
1. Figure 4 真机各方法成功率/完成时间具体数值（仅散点图）。
2. 附录 RoboTwin 逐任务 Table 3（未取到）。
3. 动作空间维度与 λ（视频共训权重）正文未写。

---

## 2. DreamZero（arXiv:2602.15922）— World Action Models are Zero-shot Policies

- **元信息**：Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, …, Yuke Zhu, Linxi "Jim" Fan, Joel Jang（NVIDIA）；2026-02-17。项目页 https://dreamzero0.github.io/ ，代码 github.com/dreamzero0/dreamzero。
- **来源级别**：L2（正文至 §4.1 已核实；§4.2、§5 主结果、附录 F/G 评测细节未取到）。
- **截断位置**：arXiv HTML 截于 §3.2.5 末（DreamZero-Flash 段后）；ar5iv 截于 §4.1 中段（"We open-source the checkpoint and inference code to run some DROID-sim evals in PolaRiS"处）。

### 【仿真验证】
- 已核实（§1 + §4 开头 verbatim）：
  - "we open-source our model weights, inference code, and code to run publicly available real-world (RoboArena) and simulation benchmarks (PolaRiS and Genie Sim 3.0)"（脚注："Genie Sim 3.0, which is a simulation benchmark comprised of 100 different tasks without being explicitly trained on the 10k hours of simulation training data"——DreamZero 只在 ~500h 真机数据上训练）。
  - §4："We validate our main hypotheses… on two robot embodiments: the AgiBot G1 mobile bimanual manipulator and the Franka single-arm robot. We pretrain separately for each embodiment…"
  - Baselines：GR00T N1.6 与 π0.5，各两种初始化（from-scratch / from-pretrained 官方权重）；"trained on identical data as DreamZero: ∼500 hours of teleoperation data we collected for AgiBot G1, and DROID for Franka. We keep the compute budget comparable… by matching total batch size and gradient steps."
- **PolaRiS / Genie Sim 3.0 / DROID-sim 的具体得分表：未取到**（§5 在截断区之后）。

### 【实机验证】
- 已核实协议骨架（§4.1 verbatim）：AgiBot G1 语料 "approximately 500 hours of teleoperation data across 22 unique environments"（7.2K episodes；图 6："each episode averages around 4.4 minutes and encompasses approximately 42 subtasks"）。
- 部署协议（§3.2.2 verbatim）："we deploy policies at an action horizon of 48 steps at 30Hz control frequency (1.6 seconds per chunk) for bimanual manipulation robots. Hence, we target inference latency below approximately 200ms…"
- 跨 embodiment：video-only 数据（YAM 机器人 20 分钟 / 人类 12 分钟）→ AgiBot G1 未见任务 "relative improvement of over 42%… with just 10–20 minutes of data"；few-shot：AgiBot G1 预训练模型 + 30 分钟 play data 适配 YAM（§1 verbatim）。
- **AgiBot G1 / YAM 真机基准的任务数、trials、成功率表：未取到**（§5.1 主表与附录 F/G 在截断区）。已核实结论句（§1）：">2× improvement in average task progress on environment and task generalization benchmarks"（vs SOTA VLA）；"outperforming state-of-the-art VLAs by 10% on average task progress"（post-training 后）。
- 效率（Table 1 完整）：Baseline 5.7s/块 → 系统级（CFG 并行 1.9×、DiT caching 5.5×）+ 实现级（compile/CUDA graphs 8.9×、kernel 9.6×、NVFP4 量化 16.6×）+ DreamZero-Flash（Beta(7,1) 解耦噪声）→ **38×、7Hz、~150ms**（GB200）。

### 【why 追问】
1. 声称 "over 2× improvement… in real-robot experiments" 但真机实验节与 trials 数在本获取范围内全部不可见——2× 的基线是 GR00T N1.6/π0.5 的哪个变体（from-scratch 还是 from-pretrained）？"average task progress" 的里程碑 rubric 定义在哪里（附录 F/G 未取到）？
2. Genie Sim 3.0（100 任务）只在脚注里给定性句，为何正表不给数值——是零样本得分太低吗？

### 【关键数值表】
Table 1（§3.2.6，逐字，H100/GB200 累积加速比）：Baseline 1×/1.1×；+CFG Parallelism 1.9×/1.8×；+DiT Caching 5.5×/5.4×；+Torch Compile+CUDA Graphs 8.9×/10.9×；+Kernel & Scheduler 9.6×/14.8×；+Quantization(NVFP4) —/16.6×；+DreamZero-Flash —/38×。

### 【一页总结】
DreamZero=14B AR 视频扩散 WAM（Wan i2v 骨干），joint 去噪视频+动作、teacher-forcing chunk 训练、KV cache 回灌真值帧防漂移；38× 加速到 7Hz。主张：异构非重复数据可学、新任务/新环境零样本、跨 embodiment 迁移（42% 相对提升，10-20 分钟视频数据；30 分钟 play 适配新本体）。**评测数字层（PolaRiS/Genie/AgiBot G1 基准表）全部在截断区，未取到。**

### 【待核实】
1. §5.1-5.2 全部表格（PolaRiS、Genie Sim 3.0、AgiBot G1 真机环境/任务泛化、DROID-sim）。
2. 附录 F（AgiBot Evaluation Details）/G（DROID Evaluation Details）的 trials 与指标定义。
3. YAM 真机任务清单与成功率。

---

## 3. XPACE（arXiv:2609.17372）— Joint World and Action Modeling from Heterogeneous Experience

- **元信息**：Jiacheng Wei, Jerry Bai, …, Yixiao Ge（XPENG Robotics）；2026-09-15。官方项目页 https://xpeng-robotics.github.io/xpace/ 。
- **来源级别**：L2（论文 §1–§3.4.1 正文 L3；§4 实验节正文未取到，但**官方项目页 Results 节逐字镜像了实验节全部数值与协议**，以下实机/仿真数值标「项目页」来源）。
- **截断位置**：arXiv HTML 截于 §3.2（skeleton conditioning 消融句）；ar5iv 截于 §3.4.1（L_III-sim 公式行）。

### 【仿真验证】
- 项目页 verbatim："Simulation and offline action prediction use 19 evaluation subsets collected separately from training. In-distribution (ID) subsets cover representative pick-and-place interactions; out-of-distribution (OOD) subsets vary scenes, objects, instructions, and behaviors."；"480 × 832 video (height × width); 97- and 193-frame horizons."
- 指标：PSNR/SSIM/OFS（光流一致）/DINO（特征相似）/Latency/Memory；离线策略：teacher-forced MSE（归一化动作空间），19 子集聚合为 "Benchmark"。
- 数值（项目页 Table 1–3 逐字）：skeleton 条件消融——token addition 最优：ID PSNR 18.07/SSIM 0.6030/DINO 0.9809；SGF：193 帧 ID PSNR 16.94→17.83（+0.89 dB）、OOD 17.74→18.75（+1.01 dB），生成快 3.00×/3.30×（50→8 ODE 步）；最终模拟器（Phase II-c+SGF）97 帧 ID PSNR 19.73。**注意：XPACE 没有用公共仿真 benchmark（无 LIBERO/RoboTwin），仿真验证=自建 ID/OOD 视频预测评测。**

### 【实机验证】
- 平台：**IRON-R01-1.11** 人形机器人。项目页 verbatim："Real robot: IRON-R01-1.11; 20 trials per task per method with matched resets. Normalized task progress averages per-trial milestone credit; success requires full completion."
- 任务清单（3 个）：T1 banana pick-and-place、T2 water pouring、T3 bowl stacking（"Bowl stacking is absent from robot demonstrations but present in human and bridge data"）。
- 成功率（项目页图 7 数据）：GR00T 20/0/0 → **6.7**；DreamZero 65/5/50 → **40.0**；XPACE 80/50/75 → **68.3**（平均）。进度：GR00T 0.40/0.33/0.35→0.36；DreamZero 0.83/0.43/0.78→0.68；XPACE 0.93/0.78/0.83→0.84。（✓ 与任务简报的 68.3 vs 40 vs 6.7 一致）
- 鲁棒性（项目页图 8）：Full-data vs robot-only，20 trials/任务：香蕉五固定位置 50→80%；block-to-box 未见干扰物 30→50%。
- Recovery DAgger（项目页 verbatim）："Fine-tuning with an 8% mixture of simulated recoveries raises mean real-robot success from **61.7% to 86.7%** and task progress from 0.81 to 0.93. Water pouring improves most, from 50% to 95% success."（注意：该 before/after 用香蕉/倒水/可乐三任务，与 68.3% 主基准的三任务构成不同——**61.7% 与 68.3% 不是同一套件**，简报中「不同套件」判断正确。）
- 人类→机器人技能迁移：qualitative 三例（关抽屉/面包转移/叠三碗），无量化成功率。

### 【why 追问】
1. 主基准（68.3%）与 DAgger 前后测（61.7→86.7%）任务集不一致（stack 换成 cola），且"20 trials/task"对 3 任务×3 方法=60 trials/方法——为什么不做同套件前后对比、不报方差？86.7% 的置信区间宽到无法排除偶然。
2. 为什么不用公共 benchmark（LIBERO/RoboTwin）做策略评测，只做视频保真度（PSNR/SSIM）——仿真与世界模型质量之间缺一条可复现的策略线。

### 【关键数值表】
见上【仿真验证】与【实机验证】两节（项目页 Table 1/2/3 + 图 7/8，逐字抄录）。

### 【一页总结】
XPACE=共享因果视频 Transformer 的 MoT（政策模式+骨架/相机条件模拟器），5000h 数据金字塔（L1 自拍视频/L2 人手动作/L3 bridge/L4 IRON 遥操）+ F 失败集仅训模拟器；Stage I 视频预训练→Stage II 粗细联合→Stage III-sim SGF 适配→Stage III-policy 8% 合成恢复+92% 原配方微调。真机：3 任务×20 trials，68.3% 超 DreamZero 40.0%、GR00T 6.7%；DAgger 后 61.7→86.7%（异套件）。

### 【待核实】
1. 论文 §4.3.1 正文表（arxiv 截断）与项目页图 7/8 的逐数字一致性（建议后续从 PDF 抽页核对）。
2. 19 个评测子集的具体构成（ID/OOD 各几个、任务名）。

---

## 4. V-JEPA 2（arXiv:2506.09985）— Self-Supervised Video Models Enable Understanding, Prediction and Planning

- **元信息**：Mido Assran, Adrien Bardes, …, Yann LeCun, Michael Rabbat, Nicolas Ballas（Meta AI）；2025-06-11，48 页 19 图。
- **来源级别**：L1（abs 页 L3；机器人实验节在 48 页长文深处，未取到；无官方项目页已取）。
- **截断位置**：未抓取全文 HTML（预算内放弃；机器人节预计位于全文 70%+ 处，固定字节截断必然取不到）。

### 【仿真验证】
- 论文的机器人部分不是仿真 benchmark，而是真机零样本部署（见下）。正文前段的视频基准（SSv2 77.3 top-1、EK-100 39.7 recall@5、PerceptionTest 84.0、TempCompass 76.9）属感知评测，不在本线轴内，仅录 abstract verbatim 备查。

### 【实机验证】
- Abstract verbatim（唯一已核实来源）："we show how self-supervised learning can be applied to robotic planning tasks by post-training a latent action-conditioned world model, V-JEPA 2-AC, using less than 62 hours of unlabeled robot videos from the Droid dataset. We deploy V-JEPA 2-AC zero-shot on Franka arms in two different labs and enable picking and placing of objects using planning with image goals. Notably, this is achieved without collecting any data from the robots in these environments, and without any task-specific training or reward."
- **任务数 / trials / 成功率数值：未取到**（实验节未达）。已知约束：<62h DROID 无标注视频；两个实验室 Franka；image-goal 规划式抓放；零样本（无环境数据、无任务微调、无 reward）。

### 【why 追问】
1. "two different labs" 的实验室与 Franka 硬件配置（相机、控制频率）是否一致？两实验室成功率是否分别报告——跨实验室协议差异若未披露，零样本结论的边界在哪？

### 【一页总结】
V-JEPA 2-AC：在 >1M 小时网视频预训练的 V-JEPA 2 上，用 <62h DROID 无标注机器人视频后训练潜动作条件世界模型，两实验室 Franka 零样本 image-goal 抓放。量化实验细节（任务/trials/SR）在本线未取到，列为头号待核实项。

### 【待核实】
1. V-JEPA 2-AC 实验节全文（任务定义、每任务 trials、两实验室各自成功率、规划 horizon/MPPI 类参数）——建议直读 PDF 第 35–45 页或 Meta 官方博客。
2. 与 V-JEPA Policy（2609.37250，见第 8 篇）的 2.1 编码器关系（该文用 V-JEPA 2.1 ViT-L）。

---

## 5. WSA₁（arXiv:2607.03941）— a 3D-Centric World-Spatial-Action Model for Generalizable Robot Control

- **元信息**：Jiahao Jiang, …, Xuanhan Wang, Jingkuan Song, Heng Tao Shen（Tongji Univ. / 上海创新研究院 / Shanghai Magic / Koala Uran）；v1 2026-07-04，v2 2026-10-13。项目页 github.com/zaleni/WSA。
- **来源级别**：L2（方法+数据配方+§4.1 开头 L3；RoboTwin 2.0 评测协议（§4.3）与真机任务表（Table 2 大部）未取到）。
- **截断位置**：ar5iv 截于 §4.1 Table 2 第一行后（AgileX PiPER 行 "RGB Block Sorting:" 处）。

### 【仿真验证】
- RoboTwin 2.0："WSA1 models achieve state-of-the-art performance among open-source models on RoboTwin2.0 (93% SR)"（§1 verbatim；abs 为 93.3%）。**评测协议（任务构成、trials/任务、指令口径）未取到**（§4.3 截断）。
- 预训练数据（Table 1 完整逐字）：InternData-A1（Sim，396M 帧，70 任务，权重 0.47）；RoboTwin（Sim，17M，50，0.07）；AgiBot-World（Real，206M，217，0.17）；RoboChallenge（Real，5M，30，0.19）；EgoDex（Human，68M，194，0.10）。总计 "6,000 hours… (including just 1,000 hours of real-robot data)"；"covers 8 different robot embodiments and 300+ robot control tasks"。
- 模型：WSA₁-B 3B（Qwen3-VL-2B 初始化）、WSA₁-L 6B（Wan2.2-5B 初始化）；三专家（2D 视觉思考/3D 世界预测/3D 逆动力学）+双向因果注意力。

### 【实机验证】
- Table 2（真机任务表，部分取到）：格式 "| 任务 | Robotic Embodiment | Trajectory Num./Time | Task Description |"；第一行：Unimanual **AgileX PiPER (7-DoF)**，30 轨迹 / 21 min，RGB Block Sorting。**其余行（任务数、embodiment 清单、trials）未取到**。
- 汇总结论（abs verbatim）："delivering an average 20% improvement over state-of-the-art models on real-world robot control tasks"（+20% average boosted performance vs SOTA RFMs）。
- 指标定义：未取到（§4.2 截断）。

### 【why 追问】
1. 真机 +20% 是「平均提升」，且表以每任务 20–30 条轨迹的小样本微调——每个任务 trials 数、success 定义、baseline 型号（哪几个 SOTA RFM）全部未在获取范围内披露；93.3% RoboTwin 是否按官方 100 trials/任务口径也未知。

### 【一页总结】
WSA₁ 用 6k h（仅 1k 真机）数据预训练三专家 3D 世界-空间-动作模型，RoboTwin 2.0 93%+、真机平均 +20%。评测协议细节（两处关键）都在截断区，Level 只能到 L2。

### 【待核实】
1. §4.3 RoboTwin 2.0 评测协议与对照方法表。
2. Table 2 全部真机任务（embodiment/trajectory 数/时长）与 §4.2 成功率表、trials 口径。

---

## 6. Cosmos Policy（arXiv:2601.16163）— Fine-Tuning Video Models for Visuomotor Control and Planning

- **元信息**：Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, …, Chelsea Finn, Jinwei Gu（NVIDIA / Stanford）；2026-01-22。项目页 research.nvidia.com/labs/dir/cosmos-policy/。
- **来源级别**：L2+（§5.1 协议与 Table 1/2 全文 L3；ALOHA 逐任务 Figure 4 数值未取到；93.6% 与 +12.5% 来自 §1 verbatim）。
- **截断位置**：ar5iv 截于 §5.2 末 ALOHA Figure 4 图片处。

### 【仿真验证】
- **LIBERO**（§5.1 verbatim）："The four primary task suites include LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long… Each task suite provides a training dataset of 500 total demonstrations (10 tasks and 50 demonstrations each)."；"we filter unsuccessful demonstrations for policy training but use the full unfiltered set for world model and value function training."；评测："averaged over 500 trials for each suite (10 tasks × 50 episodes) and three random seeds (6000 trials total)"。
- **RoboCasa**（§5.1 verbatim）："consists of 24 static kitchen manipulation tasks featuring a single Franka Emika Panda robot arm… for each task, success rate is evaluated over 50 trials across five evaluation scenes with different floor plans and styles (10 trials per scene), and the average success rate is computed across all 24 tasks over 3 random seeds (3600 trials total)."；"the RoboCasa evaluations only consist of unseen object instances, and two of the five scenes per task include styles never encountered in the training data."；训练只用 50 human demos（对比方法 >300）。
- **Table 1 LIBERO（逐字）**：Diffusion Policy 78.3/92.5/68.3/50.5→72.4；Dita 97.4/94.8/93.2/83.6→92.3；π0 96.8/98.8/95.8/85.2→94.2；UniVLA 96.5/96.8/95.6/92.0→95.2；π0.5 98.8/98.2/98.0/92.4→96.9；OpenVLA-OFT 97.6/98.4/97.9/94.5→97.1；CogVLA 98.6/98.8/96.6/95.4→97.4；**Cosmos Policy 98.1/100.0/98.2/97.6→98.5**。
- **Table 2 RoboCasa（逐字，含 demos/任务）**：GR00T-N1 300→49.6；UVA 50→50.0；DP-VLA 3000→57.3；GR00T-N1+DreamGen 300(+10000 synthetic)→57.6；GR00T-N1+DUST 300→58.5；UWM 1000→60.8；π0 300→62.5；GR00T-N1.5 300→64.1；Video Policy 300→66.0；FLARE 300→66.4；GR00T-N1.5+HAMLET 300→66.4；**Cosmos Policy 50→67.1**。

### 【实机验证】
- **ALOHA**（§5.1 verbatim）："two ViperX 300 S robot arms with three cameras: one top-down and two wrist-mounted. We reduce the controller frequency from 50 Hz to 25 Hz…"；输入 14 关节 proprio + 3 相机 + 指令；"predicting action chunks of 50 timesteps (2 seconds)"；"We deploy the full action chunk before requerying the policy."
- 任务清单（verbatim）："(1) 'put X on plate' (80 demos)… testing language following; (2) 'fold shirt' (15 demos)… long-horizon contact-rich; (3) 'put candies in bowl' (45 demos)… multimodal grasp sequences; (4) 'put candy in ziploc bag' (45 demos)… high-precision manipulation with millimeter tolerance."
- 协议："The evaluations consist of both in-distribution and out-of-distribution testing conditions, with **101 trials total per method** across all tasks. We ensure fair comparison… by using the same fixed set of initial states for each method."
- 结果：§1 verbatim "highest average success rate (**93.6%**) among state-of-the-art policies in challenging real-world bimanual manipulation tasks"；planning 增益："a 12.5 percent higher task completion rate on average in two challenging real-world manipulation tasks."（**逐任务 Figure 4 数值未取到**）。
- 规划协议（§4.3）：rollout 数据后 90% batch 训 WM/value、10% 训 policy；dual deployment（policy model + planning model）；best-of-N；WM 每动作查询 3 次、value 每未来 5 次（15 次/候选）；majority-mean 聚合；N GPU 并行。

### 【why 追问】
1. ALOHA 101 trials/方法 摊到 4 任务≈25 trials/任务，且结果藏在图里（Figure 4）不给表——93.6% 是 4 任务平均还是加权？12.5% 提升是哪两个任务的什么分布（ID/OOD）？
2. RoboCasa 只用 50 demos 而 baselines 用 300+：这是数据效率叙事，但 50 demos 里还"filter unsuccessful demonstrations"——过滤后剩多少条？训练数据量不透明时 67.1% 与 baselines 不可同口径比较。

### 【一页总结】
Cosmos Policy=单阶段把 Cosmos-Predict2-2B 的 latent 序列插入动作/未来状态/价值帧（无架构改动），parallel/AR 两种解码，rollout 数据后训练 WM+value 做 best-of-N 规划。LIBERO 98.5（6000 trials）、RoboCasa 67.1@50demos（3600 trials）、ALOHA 93.6%（101 trials/方法）。

### 【待核实】
1. ALOHA Figure 4 逐任务成功率（含 baselines 逐任务值）。
2. 规划实验的两个任务名与 12.5% 的精确任务构成（§5.3 截断）。
3. 附录 A.2/A.3 训练/评测细节（steps、checkpoint 数）。

---

## 7. DiT4DiT（arXiv:2603.10448）— Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control

- **元信息**：Teli Ma, Jia Zheng, …, Junwei Liang, Shuo Yang（Mondo Robotics / HKUST(GZ) / HKUST）；v1 2026-03-11，v2 2026-03-22。项目页 https://dit4dit.github.io/ 。
- **来源级别**：L2（方法全文 L3；§5 实验表未取到；LIBERO 98.6 / RoboCasa-GR1 50.8 / G1 任务线索来自 abstract+§1 verbatim）。
- **截断位置**：ar5iv 截于 §4.3 Algorithm 1 之后（§5 实验节整体未达）。

### 【仿真验证】
- 已核实（abstract verbatim）："reaching average success rates of **98.6% on LIBERO** and **50.8% on RoboCasa GR1** while using substantially less training data"；"DiT4DiT improves sample efficiency by over 10× and speeds up convergence by up to 7×"。
- 代理任务验证（§3 verbatim）："We validate on **24 tabletop manipulation tasks involving the GR1 humanoid robot** in the RoboCasa simulation"（Qwen3-2B vs Cosmos-Predict2.5-2B 等参对比 Grounding/FLARE-style/视频生成三代理目标；"average success rate over 24 tasks in the RoboCasa-GR1 tabletop benchmark"）。
- 主干：Cosmos-Predict2.5-2B 视频 DiT + GR00T-N1 动作 DiT；tri-timestep（τv~U、τa~Beta、τf 固定）。
- **LIBERO/RoboCasa-GR1 对照表（baselines 与各套件分数）、trials 口径：未取到**。

### 【实机验证】
- 已核实（§1 verbatim）："In real-world Unitree G1 deployments, DiT4DiT maintains clear advantages over both pre-trained (GR00T-N1.5) and parameter-matched baselines."；"relying on only a single egocentric camera, our framework extracts rich spatial reasoning capabilities, achieving the high accuracy required for precision-critical tasks such as **Arrange Flower** and **Stack Cup**."；"robust zero-shot generalization under severe distribution shifts, successfully adapting to unseen objects, category changes, and quantity variations in both simulation and physical reality."
- **G1 真机任务清单完整版、每任务 trials、成功率表：未取到**（§5 截断）。

### 【why 追问】
1. "10× sample efficiency / 7× faster convergence" 是在 24 任务 RoboCasa-GR1 的代理任务对比上测的，而主结果表（LIBERO/RoboCasa-GR1）与真机表都在截断区——数据效率声称与最终 policy 结果之间缺一张表；G1 真机"superior"到底是多少 trials 打多少分？

### 【一页总结】
DiT4DiT=视频 DiT 去噪中间隐状态（τf 固定时间步 hook 提取）作为动作 DiT 的条件，双 flow matching 联合训练。宣称 LIBERO 98.6、RoboCasa-GR1 50.8（24 任务，GR1 人形）、G1 真机 Arrange Flower/Stack Cup 高精度 + 零样本泛化。评测数字层未取到，Level L2。

### 【待核实】
1. §5 全部表格（LIBERO 四套件对照、RoboCasa-GR1 24 任务对照、G1 真机任务表与 trials）。
2. 训练步数/数据量（"substantially less training data" 的具体值）。

---

## 8. V-JEPA Policy（arXiv:2609.37250）— Building Effective WAMs on Predictive Visual Latents

- **元信息**：Yang Zhang, Jiangyuan Zhao, …, Xiu Li（Tsinghua/SJTU/Fudan/USTC/WashU/TeleAI/γ-Robotics）；v1 2026-09-29，v2 2026-10-10。官方项目页 https://breez3young.github.io/vjepa-policy/ 。
- **来源级别**：L2+（论文 §1–§5.1 正文 + 表头 L3；**官方项目页逐字给出全部关键表**（Table 3/4/5/6 与附录 F 延迟内存），以下数值两源一致）。
- **截断位置**：ar5iv 截于 §5.1 Table 1（LIBERO/LIBERO-Plus 合并表）PRTS 行。

### 【仿真验证】
- 基准（§5.1 verbatim）："(1) LIBERO… across four suites (Spatial, Object, Goal, Long-Horizon); (2) LIBERO-Plus… across seven controlled shift axes (viewpoint, initial state, language, lighting, texture, noise, layout); (3) RoboCasa-GR1… for humanoid manipulation across 24 tasks"；"(4) a TianJi Marvin dual-arm platform for real-world manipulation."
- 协议（项目页 verbatim）：LIBERO **40 tasks × 50 episodes**；LIBERO-Plus **10,030 perturbed tasks across 7 axes**（用 LIBERO 训练权重、不微调）；RoboCasa-GR1 **24 tasks × 50 episodes**。
- 训练：LIBERO 10 epochs；RoboCasa-GR1 50k optimizer steps、batch 256；From Scratch 下游更新 21,360（extended 60,000）；0.9B 参数（encoder 0.3B + predictor 0.5B + action expert 0.1B，0.6B 可训）。
- 结果（项目页 Table 5 逐字）：LIBERO 97.25（From Scratch）/98.70（Pretrained Predictor）；LIBERO-Plus 79.25/91.50；RoboCasa-GR1 50.92/55.58。（✓ 与简报 97.25/79.25/50.92 一致）
- 视觉基础对比（Table 4 逐字，LIBERO/LIBERO-Plus）：DINOv2 94.90/67.02；DINOv3 93.35/64.28；WAN2.2 VAE 92.60/46.43；InternVideo3 93.10/68.62；V-JEPA 2.1 ViT-L 97.25/79.25。编码器规模：V-JEPA2 ViT-L 74.15；2.1 ViT-L 79.25；2 ViT-G 76.08；2.1 ViT-G 78.37。
- 已获表头（论文 Table 1，部分）：OpenVLA-OFT 97.1/69.6（LIBERO/LIBERO-Plus）；π0 94.1/53.6；π0.5 96.9/80.7；VLA-JEPA 97.2/79.5；PRTS 98.4/…（后截断）。
- DROID 预测器预训练后 LIBERO-Plus 各扰动轴增益（项目页，pp）：language +29.73、texture +15.80、init +12.32、viewpoint +9.51、layout +8.85、noise +7.56、lighting +0.17。

### 【实机验证】
- 平台（项目页 verbatim）："TianJi Marvin dual-arm platform… Four RGB streams, two 7-DoF arms, and parallel-jaw grippers."
- 任务与 trials（项目页 verbatim）："Table 3 reports **20 trials per task**: Table Cleanup improves from **11/20 to 15/20**, and Saucer Racking from **7/20 to 17/20**."（即 55%→75%、35%→85%；均值 45%→80%，✓ 与 abstract "from 45% to 80%" 一致，2 任务×20 trials/任务=40 trials/条件）。
- 指标：成功率（二值）。
- 效率（项目页 verbatim）："178.17 ms action core mean latency；4.66 GiB peak allocated VRAM"（RTX 4090，batch 1，三相机；Table 6/Appendix F；不含传感器 I/O 与电机执行）。

### 【why 追问】
1. 真机只有 2 任务×20 trials，且「Table Cleanup/Saucer Racking」两个多阶段任务没有扰动测试；LIBERO-Plus 7 轴移位在真机是否复现（光照/视角鲁棒性）完全没做——sim-to-real 结论依赖 40 个 trials，为何不做更多？
2. 90%（45%→80%）的提升来自 DROID 预训练预测器，但 DROID 是机器人视频——这算"in-the-wild"还是"in-domain robot prior"？对照无预训练、同预算的结果才干净。

### 【关键数值表】
见【仿真验证】Table 5/Table 4（项目页逐字）与【实机验证】Table 3。

### 【一页总结】
V-JEPA Policy=冻结 V-JEPA 2.1 ViT-L 潜空间 + 从零训练指令条件未来潜预测器 + flow-matching 动作专家（MoT），单阶段 0.9B。LIBERO 97.25 / LIBERO-Plus 79.25 / RoboCasa-GR1 50.92；DROID 无动作预训练后 91.50/55.58、真机 45%→80%。推断 10 步 Euler（imagine-then-act），178ms。

### 【待核实】
1. 论文 Table 1 剩余行（UWM/DiT4DiT/ImageWAM/V-JEPA Policy 各套件逐值）。
2. 附录 A.1 评测协议细节（trials 是否与项目页口径一致）。

---

## 9. 3D-ALP（arXiv:2604.11302）— 3D-Anchored Lookahead Planning via World-Model-Based MCTS

- **元信息**：Bronislav Sidik, Dror Mizrahi（Huawei Technologies, Israel R&D Center）；2026-04-13，5 页 1 图 1 表。
- **来源级别**：**L3（全文完整获取，含实验节、Table 1、讨论与参考文献）**。
- **截断位置**：无（ar5iv 全文完整）。

### 【仿真验证】
- 任务（§4.1 verbatim）："A **Franka Panda arm in MuJoCo simulation** executes a **5-step sequential reach task**. Steps 1–3 visit three distinct workspace positions (all objects visible). Steps 4–5 require returning to earlier positions that are no longer observable from the current camera frame."
- 协议（§4.1 verbatim）："Both agents receive identical geometric scoring (Eq. (4)) for a controlled comparison… 3D-ALP uses lookahead depth D=2 and branching factor B=4…, a 2 s planning budget, and 10 physical actions per step. The greedy baseline samples the same number of actions per step and executes the one minimising immediate distance."；"Table 1 reports the full ablation across **30 episodes and 3 random seeds**."
- MCTS 参数（§3.4）：UCB1 c=0.02（原 √2≈1.414 校准于二元得分）；Max-MCTS（backprop max_value）；F1-F4 四个结构修复（zero-action 陷阱、树深度衰减、均值稀释、UCB1 常数失配）。
- **Table 1（逐字，mean±std）**：Greedy：Non-Mem SR 0.748±0.029，Memory SR 0.006±0.008，Δ=—，Step5 0.000；MCTS D=1：0.389±0.024，0.539±0.064，+0.533，0.622；3D-ALP D=2：0.463±0.064，**0.650±0.109**，**+0.645**，**0.822**。
- 消融归因（§4.1 verbatim）：tree search memory +0.533（82% of gain）；deeper lookahead +0.111（17%，集中于 step 5：0.622→0.822）。
- 组件验证（§4.2）：Phase 0 几何一致性 SSIM=1.000、ORB 391/391 keypoints（5 轨迹）；Phase 1 桥接角误差 0.00°（100% pass vs ≤5° 阈值）、延迟 0.5ms。
- 指标：SR 定义=每步到达目标（距离/得分口径见 Eq.3-4 hybrid scorer 与几何 oracle）。

### 【实机验证】
- **无真机**。§5 verbatim："**All experiments run in MuJoCo simulation. Real-robot validation on a Franka Panda or SO-100 arm is future work.**"
- 瓶颈披露（§5 verbatim）：InSpatio-WorldFM 渲染 ≈2400 ms/frame（单 RTX A6000），2s 预算仅容 ~5 节点/周期（< D=2,B=4 的 20 理论容量）；4-GPU 并行降到 ≈600ms；Florence-2 评分器对合成帧全部输出 ≈0.0（深度盲）。

### 【why 追问】
1. E3 用纯几何 oracle（S_exact）而非完整视觉管线——0.650 的成绩没有经过「世界模型渲染 + VLM 评分」这条实际路径；作者自己也承认渲染瓶颈让全管线跑不起来，那么「3D-ALP works」的证据其实只证明了树搜索记忆有效，视觉闭环仍是假设，为何不在更便宜的 latent 评分（他们 Phase 2 自己提出 JEPA）上先验证？
2. 30 episodes×3 seeds=90 次，且每 step 10 物理动作×5 步——episode 内的随机性来源（初始姿态分布）未定义；0.006±0.008 的 greedy 基线本身接近零，+0.645 的对比是否只是「随机 vs 非随机」？

### 【一页总结】
3D-ALP=测试时 MCTS 规划器（世界模型 InSpatio-WorldFM 做 rollout oracle + 持久 SE(3) c2w 锚点 + 混合几何-语义评分器），MuJoCo Franka 5 步顺序 reach：记忆步骤 0.650±0.109 vs greedy 0.006（Δ+0.645）；step5 0.822 vs 0.000。消融：树记忆 82%、更深前瞻 17%。无真机（future work 明示）。唯一 L3 全文核实论文。

### 【待核实】
无（全文核实）。可选：GitHub 代码中 mcts_engine.py 的 c=0.02 与 Max-MCTS 实现（作者自称 "verified in 'mcts_engine.py'"，本线未查代码）。

---

## 10. τ0-WM（arXiv:2606.01027）— A Unified Video-Action World Model for Robotic Manipulation

- **元信息**：Pengfei Zhou, Shengcong Chen, …, Jianlan Luo（Shanghai Innovation Institute / AGIBOT Finch）；v1 2026-05-31，v2 2026-08-23。项目页 https://tau0-wm.github.io ，finch.agibot.com/research/tau0-wm。
- **来源级别**：L2（方法全文+数据构成+任务图 L3；实验数值表未取到）。
- **截断位置**：ar5iv 截于 §VI-B（LAR 段落中）。

### 【仿真验证】
- 已核实（§III verbatim，数据构成）："a 27.3K-hour training corpus from three sources: **17.8K hours of real-robot teleoperation on AGIBOT-G01, ARX manipulators, and dual-arm Franka systems**; **6.5K hours of filtered open-source UMI-style demonstrations collected with Gen-DAS Grippers**; and **3.0K hours of open-source egocentric human interaction videos**."
- 架构：VAM（Wan2.2-TI2V-5B 视频 DiT + 0.5B 动作 DiT = 5.5B）+ ACVS（同骨干去动作分支，Cosmos 式动作条件 + 稠密 progress 评分）；λz=λa=1；测试时 RCS（re-denoising consistency）+ 阈值 γ 触发 ACVS 评估 + LAR（latent action rectification）。
- **标准仿真 benchmark（RoboTwin 等）：正文未见（已获取范围内原文未写）**——评测聚焦真机系统（"We evaluate τ0-WM as a robot-facing system"）。若在截断区之后有 benchmark 表，未取到。

### 【实机验证】
- 任务（Fig. 3 可见，verbatim caption）："Our evaluation tasks. (a) Storing different tools on the desk into their corresponding places in the toolbox (Toolbox). (b) Unzipping the school bag, storing objects into it, and zipping up (School Bag). (c) Connecting the hose to the faucet and securing it (Faucet). (d) Storing the badminton shuttlecocks and closing the lid (Badminton)."
- 结论句（§I verbatim）："τ0-WM achieves the **best average success rate among the evaluated baselines on four manipulation tasks**"；"heterogeneous pre-training improves both zero-shot and fine-tuned performance, while test-time computation improves single-attempt execution through both re-denoising consistency selection and simulator-assisted rectification."
- **每任务成功率数值、trials 数、baseline 名单与"三 embodiment"部署对应：未取到**（实验节截断）。数据侧三 embodiment（AGIBOT-G01 / ARX / dual-arm Franka）已核实，评测 embodiment 未写明（已获取范围）。

### 【why 追问】
1. "best average success rate… on four manipulation tasks" 不给任何数值表就下结论——4 任务×多少 trials、哪几个 baseline、哪个 embodiment？摘要级的 claim 与可复核证据之间的断层是所有 10 篇里最大的。

### 【一页总结】
τ0-WM=统一视频-动作世界模型（VAM 政策 + ACVS 动作条件模拟器，共享 Wan2.2-5B 骨干），27.3k h 异构数据（17.8k 遥操/6.5k UMI/3.0k 自拍）按模态掩码联合训练；测试时采样-去噪一致性排序-模拟器校正（LAR）。真机四长程精细任务（Toolbox/School Bag/Faucet/Badminton）"best average SR"，数值未取到。

### 【待核实】
1. 实验节（§VII 起）全部表格：四任务×baselines 成功率、trials、测试时计算消融、heterogeneous 预训练消融。
2. 评测 embodiment（G01? ARX? 双 Franka?）与任务-平台映射。

---

## 汇总表：10 篇 × 仿真基准与设定 × 真机平台与指标 × trials 披露情况

| # | 论文 | 仿真基准（任务/数据/trials） | 真机平台与任务 | 真机指标与 trials 披露 | 来源级别/截断 |
|---|---|---|---|---|---|
| 1 | Fast-WAM | LIBERO 4 套件（10 任务×50 demo，500 demos/套件，20k 步，2000 trials，unseen 指令）；RoboTwin 2.0（>50 任务，2500+25k demos，30k 步，100 trials/任务） | Galaxea R1 Lite：毛巾折叠（60h teleop） | 成功率+完成时间仅散点图（Figure 4 数值未取到）；w.o. co-train=10%；190 vs 810ms。**trials 未披露** | L2；附录 Table 3 未取到 |
| 2 | DreamZero | PolaRiS、Genie Sim 3.0（100 任务）零样本、DROID-sim（§5 数值未取到） | AgiBot G1（500h×22 环境）、Franka（DROID）、YAM 适配 | >2× task progress vs VLA；42% 相对提升（10-20min 视频）；30min play 适配。**trials/任务表未取到** | L2；截于 §4.1 |
| 3 | XPACE | 自建：19 评测子集（ID/OOD）、480×832、97/193 帧、PSNR/SSIM/OFS/DINO（无公共 benchmark） | IRON-R01-1.11：香蕉拣放/倒水/叠碗 | **20 trials/任务/方法，matched resets**；68.3 vs DreamZero 40.0 vs GR00T 6.7；进度=里程碑均分；DAgger 61.7→86.7（异套件）✓ | L2（项目页）；论文 §4 未取到 |
| 4 | V-JEPA 2 | 无（机器人实验=真机零样本） | Franka ×2 实验室：image-goal 抓放（<62h DROID，零样本） | **任务数/trials/SR 全部未取到** | L1 |
| 5 | WSA₁ | RoboTwin 2.0：93.3%（协议未取到） | AgileX PiPER 等（Table 2 部分：30 traj/21min RGB Block Sorting） | 真机 +20% 平均提升；**trials 未取到** | L2；截于 Table 2 首行 |
| 6 | Cosmos Policy | LIBERO（500 demos/套件，6000 trials）；RoboCasa（24 任务，50 trials/任务×5 场景×3 seeds=3600 trials，unseen 物体+2/5 场景 unseen 风格，50 demos） | ALOHA（ViperX 300 ×2，25Hz）：put X on plate(80)/fold shirt(15)/candies in bowl(45)/ziploc(45) demos | **101 trials/方法**（ID+OOD，固定初始状态）；93.6% 平均；+12.5% 规划增益；逐任务图未取到 | L2+；截于 Figure 4 |
| 7 | DiT4DiT | LIBERO 98.6；RoboCasa-GR1 50.8（24 任务，GR1 人形，对照表未取到）；代理任务对比 24 任务 | Unitree G1：Arrange Flower、Stack Cup（单自拍相机） | "superior" + 零样本泛化；**trials 未取到** | L2；截于 §4.3 |
| 8 | V-JEPA Policy | LIBERO 40 任务×50 episodes；LIBERO-Plus 10030 扰动任务×7 轴（不微调）；RoboCasa-GR1 24×50 | TianJi Marvin 双臂（4 RGB、2×7DoF、平行夹爪）：Table Cleanup、Saucer Racking | **20 trials/任务**：11/20→15/20、7/20→17/20（均值 45→80%）✓ | L2+（项目页全表）；截于 Table 1 |
| 9 | 3D-ALP | MuJoCo Franka 5 步 reach（D=2、B=4、2s 预算、10 动作/步、30 episodes×3 seeds） | **无真机**（"Real-robot validation… is future work"） | SR 0.650±0.109 vs greedy 0.006（step5 0.822）✓ | **L3 全文** |
| 10 | τ0-WM | 无公共 benchmark（自述"robot-facing system"；RoboTwin 未见） | 数据三 embodiment（AGIBOT-G01/ARX/双 Franka）；评测四任务：Toolbox/School Bag/Faucet/Badminton | "best average SR on 4 tasks"；**数值/trials 未取到** | L2；截于 §VI-B |

## 评测协议惯例与坑（仅基于已核实原文）

**惯例**
1. **LIBERO 是事实标准**（4/10 篇：Fast-WAM、Cosmos Policy、DiT4DiT、V-JEPA Policy），但 trials 口径必须看论文自己写：Cosmos 500 trials/套件×3 seeds=6000；V-JEPA Policy 40×50=2000；Fast-WAM 2000（20k 步/2000 trials）。三篇口径接近但表述不同。
2. **RoboTwin 2.0**：Fast-WAM（>50 任务、100 trials/任务、clean+rand 双档）与 WSA₁（93.3%）都报但协议披露程度悬殊——Fast-WAM 披露数据与 trials，WSA₁ 目前只见分数。
3. **RoboCasa 同分不同口径**：Cosmos 50 trials/任务×5 场景×3 seeds；V-JEPA Policy 24×50；RoboCasa-GR1（DiT4DiT/V-JEPA Policy）是人形 GR1 变体——跨论文比较 67.1 vs 50.8/50.92 时必须核对 embodiment 与场景协议。
4. **真机 trials 两极分化**：XPACE 20/任务/方法（3 任务）、V-JEPA Policy 20/任务（2 任务）、Cosmos 101/方法（4 任务）算披露充分的；Fast-WAM（散点图不给数）、WSA₁、DiT4DiT、τ0-WM、DreamZero 均为"结论在、表不在"。
5. **进度分（progress）作为主指标**：DreamZero（average task progress）、XPACE（milestone credit）、τ0-WM（dense task-progress）都用了连续进度分，与二值 SR 并行报告；跨论文比较 SR 时要分清。
6. **零样本/泛化协议**：V-JEPA 2（无环境数据）、DreamZero（Genie Sim 3.0 不训练 10k h 仿真数据）、DiT4DiT（unseen 物体/类别/数量）、V-JEPA Policy（LIBERO-Plus 7 轴）——"zero-shot"的定义在各篇里不同，写对比时必须附定义。

**坑**
1. **同数字不同套件**：XPACE 的 68.3%（香蕉/倒水/叠碗）与 61.7→86.7%（香蕉/倒水/可乐）不是同一任务集，不可直接相减当作"提升"。
2. **demos 过滤**：Cosmos Policy 明确"filter unsuccessful demonstrations for policy training"（LIBERO/RoboCasa 10-20% demos replay 失败）——报告的 50 demos 是训练集总数，实际用于 policy 的更少，且 WM/value 用未过滤集；跨论文比数据效率时要问这一刀。
3. **指令口径**：Fast-WAM README 明示 unseen instructions 评测（LingBot-VA 用 seen）——LIBERO 榜单比较暗含指令口径差异。
4. **延迟与成功率耦合**：Fast-WAM 真机比较里 IDM 810ms vs 190ms 而任务有时间指标（completion time），慢方法可能因超时而非控制质量被扣分；读"慢→差"结论要问计时规则。
5. **图藏表**：Fast-WAM Figure 4、Cosmos Figure 4 把关键数值只放图；本线两处均未取到数字，只能引用图 URL。
6. **评测 embodiment 不写**：τ0-WM 数据三 embodiment 明确、评测四任务的平台映射未达；WSA₁ 真机表只取到首行。凡 embodiment 未写的真机分数，先标"待核实"再引用。
7. **trials 太小**：V-JEPA Policy 真机 2 任务×20 trials、XPACE 3 任务×20 trials——成功率单任务差异 25pp 时 95% CI 约 ±20pp，小样本结论引用时务必带 trials 数。
8. **无真机也下结论**：3D-ALP 明确 sim-only（real-robot 是 future work），引用其 65% 时不得说成真机能力。

---

<!-- QA_LOG -->
（占位：此处记录与用户的逐条讨论 Q&A，尚未开始）
<!-- /QA_LOG -->
