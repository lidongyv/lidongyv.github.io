# WAM 精读线②第二波：实机与仿真验证（扩展 C，10 篇实验协议深挖）

> 更新：2026-10-27（重发轮完成）｜ 任务：把第一波 L1/L2 笔记的实验节挖到 L3——逐篇取 arXiv HTML（最新版）/ar5iv 原文，补协议、trials、数值表；官方 GitHub README/项目页作补充并标注来源级别。
> 与前作关系：[02-sim-real-validation.md](02-sim-real-validation.md)（第一波 10 篇）；本文件为重发轮全量更新版。
> **工具限制（同第一波）**：本机 shell 出网被沙箱拦截（本轮复测：Invoke-WebRequest TLS 失败、curl schannel SEC_E_NO_CREDENTIALS）；web_fetch 对长 HTML 有固定截断（截断点以「(Content truncated…)」标记）；r.jina.ai 等镜像不可用。来源级别：L3=实验节原文全文核实；L2+=正文部分+官方仓库/项目页逐字交叉；L2=部分正文；L1=仅摘要。**重发轮事实来源（各节已标注）**：WLA（ar5iv 补全 §4.2+§4.3 开头）、MotionWAM（ar5iv 取到 §6）、Rolling-WAM（Table I/II 复验+§IV-B）、GE-Act/τ0-WM（方法+结构）、DreamZero/XPACE/ABot/Kairos（截断点全部复测确认）。

---

## 1. WSA₁（arXiv:2607.03941）— a 3D-Centric World-Spatial-Action Model for Generalizable Robot Control

- **元信息**：Jiahao Jiang, Jianing Zhang, Xuanhan Wang, Duo Peng, Ruidong Chen, Zhenhan Yin, Sen Wang, Zhaoshu Yu, Pengpeng Zeng, Xiaofeng Cao, Jingkuan Song, Heng Tao Shen（Tongji Univ. / Shanghai Innovation Institute / Shanghai Magic / Koala Uran）；v1 2026-07-04，v2 2026-10-13。GitHub github.com/zaleni/WSA，项目页 zaleni.github.io/WSA1/，HF 集合 huggingface.co/collections/zaleni/wsa1；README 披露 RoboChallenge CVPR 2026 leaderboard 第 4/100+ 队（Team: MagicBot，2026-05-18）。
- **来源级别**：**L2+**（§1/§3/§4.1 协议段正文 L3——本轮重取 v2 原生 HTML 补到 §4.1 Evaluation Protocol verbatim；官方仓库 README 逐字披露全部关键表与真机 trials；**论文 §4.2 真机正文仍未取到**）。
- **截断位置**：本轮 arXiv 原生 HTML v2 截于 §4.1 Table 2 π0 行（"π0 [Black et al., 2025b] |"）；§4.2 Real Robot Experiments 与附录 B.6/B.9 仍不可达（第一波 ar5iv 同区域截断）。

### 【仿真验证】（数值=官方仓库 README，逐字抄录；协议=v2 §4.1 正文）
- **论文 §4.1 Evaluation Protocol（v2 verbatim，本轮重取新增）**："We adopt multi-task evaluation with success rate (SR) as the metric. Unless otherwise stated, models are fine-tuned on all 27,500 RoboTwin2.0 demonstrations. For scene generalization (OOD-C2R), models are trained only on clean demonstrations and tested in randomized scenes. For zero-shot cross-embodiment generalization (OOD-CE), the trained AgileX embodiment is replaced by an unseen ARX-X5 at test time. By default, WSA1-B without pre-training is adopted to conduct ablation studies for fast investigation."（v2 §4.1 "Benchmarks"：RoboTwin2.0=50 双机械臂任务 clean/randomized 双档、LIBERO=四套件；27,500=2,500 clean+25,000 randomized，与 README 口径一致）。
- **RoboTwin 2.0**（README verbatim）："Success rates averaged over 50 simulated ALOHA manipulation tasks in the **randomized setting**."（注意：仅 randomized 档；**trials/任务 README 未写**，论文 §4.3 未取到）：
  π0 58.4%｜π0.5 76.8｜ABot-M0 85.1｜Motus 87.0｜InternVLA-A1 89.6｜LingBot-VA 91.6｜Fast-WAM 91.8｜**WSA-B 92.7**｜**WSA-L 93.1**。
  ⚠️ 论文 abs（v2）为 "93.3% success rate on RoboTwin2.0"——**93.3（论文）vs 93.1（README）不一致**，见【待核实】。
- **LIBERO**（README verbatim，四套件+Spatial/Object/Goal/Long+Avg）：π0.5 98.8/98.2/98.0/92.4→96.9；OpenVLA-OFT 97.6/98.4/97.9/94.5→97.1；ABot-M0 98.8/99.8/**99.0**/96.6→**98.6**；Motus 96.8/99.8/96.6/97.6→97.7；LingBot-VA 98.5/99.6/97.2/**98.5**→98.5；Fast-WAM 98.2/**100.0**/97.0/95.2→97.6；WSA-B 98.6/99.6/97.2/94.2→97.4；WSA-L **99.4**/99.8/98.0/95.6→98.2。
- **Clean2Random 泛化（OOD-C2R）**（README verbatim）："All models are fine-tuned **only on clean demonstrations (2,500 episodes in total)**, then evaluated in both clean and randomized scenes."：π0.5 70.7/46.0→58.4（Drop 24.7）；ABot-M0 57.4/30.4→43.9（27.0）；RobotManip 73.2/62.6→67.9（10.6）；Fast-WAM 77.8/1.9→39.9（**75.9**）；InternVLA-A1 **86.6**/74.6→80.6（12.0）；**WSA-B 83.8/77.8→80.8（Drop 6.0，最小）**。
- **ARX-X5 零样本迁移（OOD-CE）**（README verbatim）："Policies are fine-tuned on ALOHA data and transferred to the **unseen ARX-X5** without target-robot policy fine-tuning. Methods are evaluated over **45 tasks with 100 episodes per task**."：π0.5 33.5/19.0→26.3；LingBot-VA 17.2/16.6→16.9；RobotManip —/42.9→42.9；Fast-WAM 25.9/22.4→24.2；InternVLA-A1 18.1/17.5→17.8；**WSA-B 47.1/47.6→47.4**。

### 【实机验证】（数值=官方仓库 README，逐字抄录）
- 平台与任务（README verbatim）："Seven tabletop tasks span the **7-DoF AgileX PiPER** and **14-DoF bimanual ARX Lift2**. Success rate (SR) measures full-task completion; completeness (C) measures the fraction of task stages completed. Both metrics are averaged equally across the seven tasks."
- 平均：π0 33.73/43.24；π0.5 55.45/63.33；InternVLA-A1 38.45/50.37；**WSA-B 77.30/82.27；WSA-L 80.52/85.21**（SR%/C%）。
- **逐任务表（SR/C%，README verbatim）**："Each cell gives **SR / C (%)**. All policies use **30 trials per task**, except Sweep Trash (34) and Unzip Pencil Bag (20)."：
  | 任务 | 机器人 | π0 | π0.5 | InternVLA-A1 | WSA-B | WSA-L |
  |---|---|---|---|---|---|---|
  | RGB Block Sorting | PiPER | 70.00/70.00 | 70.00/70.00 | 50.00/50.00 | 80.00/80.00 | 83.33/83.33 |
  | Object Organization | PiPER | 50.00/50.00 | 76.67/76.67 | 36.67/36.67 | 80.00/80.00 | 76.67/76.67 |
  | Trash Cleaning | Lift2 | 36.67/53.33 | 76.67/86.67 | 43.33/58.33 | 80.00/90.00 | 86.67/91.67 |
  | Pen Holder Placement | Lift2 | 6.67/28.33 | 70.00/76.67 | 33.33/38.33 | 86.67/90.00 | 76.67/76.67 |
  | Toy Box Organization | Lift2 | 13.33/18.33 | 13.33/26.67 | 36.67/48.33 | 60.00/71.67 | 70.00/85.00 |
  | Sweep Trash | Lift2 | 29.41/40.19 | 26.47/44.12 | 44.12/78.41 | 79.41/89.21 | 85.29/93.12 |
  | Unzip Pencil Bag | Lift2 | 30.00/42.50 | 55.00/62.50 | 25.00/42.50 | 75.00/75.00 | 85.00/90.00 |
- 部署（README verbatim）："The released checkpoints were evaluated on **NVIDIA GeForce RTX 4090 GPUs**."；真机推理为 GPU policy server + robot-side client 分离（evaluation/Real_Piper_Example、Real_Lift2_Example）；训练用 "8x NVIDIA H200 GPUs"。
- 扰动/泛化：真机无扰动协议（已获取范围内原文未写）；仿真侧由 Clean2Random 与 ARX-X5 零样本两个 OOD 套件承担泛化验证。

### 【why 追问】
1. **93.3 vs 93.1**：论文 abs「93.3%」与官方 README「WSA-L 93.1%（仅 randomized 档）」对不上——93.3 是 clean+randomized 平均还是 README 数字未随 v2 更新？若论文主表口径不同，榜首声明（"state-of-the-art among open-source models"）的口径必须注明。
2. 真机 7 任务的 SR/C 两指标平均权重相等，但 completeness 的「任务阶段」rubric（每个任务分几个阶段、谁标注）README 与已取正文均未写；阶段划得粗，C 就会虚高——80.52% 的 WSA-L 对比 33.73% 的 π0，中间隔着 rubric 定义。

### 【关键数值表】
见上【仿真验证】四个表与【实机验证】逐任务表（全部官方 README 逐字抄录）。预训练数据：v2 Table 1（本轮逐字）三行聚合=Simulation Data 424M 帧/70 任务/权重 0.54、Egocentric Human Data 89M/194/0.10、Real Robot Data 99M/36/0.36（Σ612M 帧≈6,000h，8 具身、300+ 任务）；第一波 v1 五源细表=InternData-A1 396M/0.47、RoboTwin 17M/0.07、AgiBot-World 206M/0.17、RoboChallenge 5M/0.19、EgoDex 68M/0.10（两版聚合粒度不同）。

### 【一页总结】
WSA₁ 以 6k h（仅 1k 真机）预训练 3B/6B 双规格三专家模型。README 口径下：RoboTwin 2.0 randomized 50 任务 WSA-L 93.1（论文称 93.3）；LIBERO 98.2；Clean2Random 掉点仅 6.0（Fast-WAM 75.9）；ALOHA→ARX-X5 零样本 47.4（45 任务×100 episodes）。真机 7 任务×30 trials（两例外 34/20）WSA-L 80.52% SR / 85.21% C，超 π0.5（55.45）25pp、超 π0（33.73）47pp。

### 【待核实】
1. 93.3 的口径：v2 §1 verbatim "WSA1 achieves a 93.3% success rate on RoboTwin2.0 and improves over state-of-the-art approaches by an average of 20% on real-world robot control tasks." vs README WSA-L 93.1（仅 randomized 档）——93.3 是 Clean/Rand 平均还是 Clean 值？Table 2 截断（截于 π0 行）无法确认。
2. 论文 §4.2 真机正文表与 README 逐值一致性（v2 编号；本轮重取仍截于 Table 2，未达 §4.2）。
3. completeness 的阶段 rubric 定义（v2 附录 B.6 Real-Robot Tasks and Per-Task Results / B.9 Long-Horizon Real-Robot Execution，均不可达）。

---

## 2. WLA（arXiv:2606.05979）— World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis

- **元信息**：Yi Yang, Zhihong Liu, Siqi Kou, Yiyang Chen, Yanzhe Hu, Jianbo Zhou, Boyuan Zhao, Zhijie Wei, Xiao Xia, Xueqi Li, Pengfei Liu, Zhijie Deng（SJTU / SII / HUST / SCUT / ECUST / SHU / NJUPT）；v1 2026-06-04（本线取到版本；GitHub README 标注 CoRL 2026）。代码 github.com/SJTU-DENG-Lab/WLA。
- **来源级别**：**L3 仿真全段 + §4.3 开头**（本轮重取核实：§4.2 全段含 LIBERO/RMBench 协议与 Table 1/2 逐字；§4.3 Task Setup 首段 + Fig. 3 caption 已取；**§4.3 后段（三设定成功计数/完成时间/延迟数值）、§4.4、附录 A–D 未取到**）。
- **截断位置**：本轮 arXiv 原生 HTML v1 截于 §4.2 RoboTwin 段末（"Furthermore, the World Expert ablation confirms that future-state prediction effectively refines action generation." 处）；ar5iv 截于 §4.3 Task Setup "(1) standard setup, (2)" 处——比上一轮（§4.2 中段）前进了 LIBERO/RMBench 两段 + §4.3 开头。

### 【仿真验证】
- **RoboTwin 2.0 协议（§4.2 verbatim）**："RoboTwin 2.0 is a challenging bimanual manipulation benchmark comprising 50 tasks requiring coordinated dual-arm control. Following prior multi-task training protocols, we train WLA-0 on a mixed demonstration dataset with **2,500 clean-scene trajectories and 25,000 strongly randomized trajectories for 100k steps with a global batch size of 256**. Language-based subtask prediction is disabled due to the benchmark's short task horizons. Discarding the World Expert at inference leaves ∼2B active parameters... **Table 1 reports the average success rates over 100 trials per task**."
- **Table 1（§4.2，逐字）**：
  | Method | Active Params | Embodied PT | RoboTwin Clean/Rand. | LIBERO S/O/G/L→Avg |
  |---|---|---|---|---|---|
  | π0 | 3B | ✓ | 65.92/58.40 | 96.8/98.8/95.8/85.2→94.2 |
  | π0.5 | 3B | ✓ | 82.74/76.76 | 98.8/98.2/98.0/92.4→96.9 |
  | Motus | 8B | ✓ | 88.66/87.02 | 96.8/99.8/96.6/97.6→97.7 |
  | Lingbot-VA | 5B | ✓ | 92.90/91.50 | 98.5/99.6/97.2/98.5→98.5 |
  | Fast-WAM | 6B | ✗ | 91.88/91.78 | 98.2/100.0/97.0/95.2→97.6 |
  | **WLA-0** | **2B** | ✗ | **92.94**/90.02 | 99.0/100.0/97.8/97.6→**98.6** |
  | −ℒwm | 2B | ✗ | 90.98/89.34 | 98.4/99.6/97.0/96.4→97.9 |
  | ++TTS | 2B | ✗ | –/– | 99.2/100.0/98.4/97.8→98.9 |
- 消融（§4.2 verbatim）："WLA-0 achieves a 92.94% success rate in clean environments, remains highly robust under domain randomization, and matches or outperforms prior pipelines while using significantly fewer parameters and no embodied pretraining."；−ℒwm（无 World Expert loss）Clean 90.98/Rand 89.34 → 世界模型损失带来 Clean +1.96pp；++TTS 使 LIBERO Avg 98.6→98.9（+0.3pp）。
- **LIBERO（§4.2 verbatim，本轮补全）**："We train WLA-0 on all four LIBERO suites, i.e., Spatial, Object, Goal, and Long, **each containing 10 tasks with 50 demonstrations per task**. A single model is trained across all suites for **100k steps with a batch size of 256** and evaluated over **50 trials per task**. In practice, we observe strong performance after only 30k training steps, highlighting the training efficiency of WLA-0."；结论："WLA-0 achieves **98.6% average success**, outperforming all WAM and VLA baselines."；TTS："With test-time scaling using **6 candidates and an imagination horizon of 2**, the average success rate further improves to 98.9%."
- **RMBench（§4.2 verbatim + Table 2 逐字，本轮补全）**："RMBench is a long-horizon, memory-dependent benchmark for bimanual manipulation. We evaluate WLA-0 on its **M(n) subset**, whose four tasks require repeated exploration, trial-and-error recovery, long-term memory, and inference of the currently executable subtask from interaction history... Following the RMBench protocol, we adopt a **single-task training setup, training one model per task for 30k steps with a global batch size of 448**." Table 2：
  | Method | Battery Try | Blocks Ranking Try | Cover Blocks | Press Button | Average |
  |---|---|---|---|---|---|
  | π0.5 | 16% | 6% | 0% | 0% | 5.5% |
  | X-VLA | 26% | 1% | 2% | 0% | 7.3% |
  | Mem-0 | 28% | 18% | 68% | 0% | 28.5% |
  | Fast-WAM | 16% | 37% | 0% | 0% | 13.3% |
  | **WLA-0** | **45%** | **23%** | **84%** | **74%** | **56.5%** |
  | −ℒlang | 38% | 12% | 18% | 1% | 17.3% |
  ⚠️ 正文句 "removes the subtask prediction loss reduces the average success rate from 56.5% to **17.25%**" 与 Table 2 的 17.3% 不一致（17.25 vs 17.3）。结论句："WLA-0 achieves the best average success rate of **56.5%**, substantially outperforming Fast-WAM and nearly doubling Mem-0."——§1 的 "nearly doubling the performance of the best baseline" 即 Mem-0 28.5%（56.5/28.5≈1.98×）。
- **归属确认（本轮落实，回应上轮遗留）**：90.02 = RoboTwin 2.0 **Rand.** 档（Table 1 WLA-0 行 Clean 92.94 / Rand. 90.02，表头列序已逐字复验）；98.6 = **LIBERO 四套件 Avg**；56.5 = **RMBench M(n) 四任务 Avg**。
- 实现（§4.1 verbatim）：WLA-0 3.4B 总参（RynnBrain-2B 2.1B 骨干 + SANA-600M 900M 含 VAE World Expert + 390M flow-matching Action Expert，各 28 层）；64 meta-queries；action chunk=8（LIBERO）/32（其他）；α=0.1、β=0.005；DeepSpeed+AdamW（wd 1e-8，grad clip 1.0，cosine LR 5e-5→5e-6，1000 warm-up）。

### 【实机验证】
- **§4.3 Task Setup（verbatim，本轮补全）**："We designed four long-horizon tasks to evaluate the real-world performance of WLA-0 (Fig. 3 (a)). **Unscrew Cap and Pack Object** evaluate fine-grained manipulation, while **Stack Cup and Dispose Trash** assess inference efficiency. In Dispose Trash, the robot must grasp scattered paper balls and place them into a trash bin rotating at varying speeds, requiring accurate trajectory prediction and low-latency control. For each task, we collect **60 demonstrations using the AgilexRobotics Piper dual-arm platform**. Each task is evaluated under three settings: (1) standard setup, (2)…"（此处截断）。
- **Fig. 3 caption（verbatim）**："(a) Illustrations of the four long-horizon tasks. (b) Examples of OOD Object and OOD Scenario tasks. (c) Success count comparison. Bar colors, from light to dark, denote the standard setup, OOD object, and OOD scenario, respectively. (d) Inference efficiency comparison. The left y-axis shows task completion time (s), and the right y-axis shows inference latency (ms)."——三设定=standard / OOD object / OOD scenario；各设定成功**计数**（非成功率）、完成时间(s)、延迟(ms) 数值均在图 3(c)/(d) 条形图内，文字版在截断区**未取到**。
- 结论句（§1 verbatim，沿用）："Real-world evaluations demonstrate robustness in dynamic and out-of-distribution (OOD) settings; **on Stack Cup, WLA-0 halves the completion time of the baseline WAM**, highlighting its suitability for latency-sensitive control."（基线 WAM 型号与绝对数值未取到；附录 C 不可达）。
- 延迟（abs + §3.2 verbatim）："achieves **40 ms per inference** on an NVIDIA RTX 5090"（efficient mode，World Expert 关闭；加速技术细节在附录 A，未取到）。
- 测试时扩展（§3.2 verbatim）：TTS 模式采样 K 个候选 action chunk（变种子）→ World Expert 预测各候选未来帧 → 价值模型打分 → 执行最高分 chunk；"The value model is trained using rollouts from the fine-tuned WLA-0... assigns the discounted value label as v_t = y·γ^{T−t}"。

### 【why 追问】
1. 92.94% 的 RoboTwin Clean 排名声明建立在**无具身预训练 + 2B 激活**上，但 WLA-0 总参 3.4B 且训练中 World Expert 始终存在——"2B 激活"是推理口径；RoboTwin 100 trials/任务的口径与 Fast-WAM（同一 2500+25k 数据配方）可比，但 Fast-WAM 的数字（91.88）与第一波笔记（91.8）一致——WLA 领先 1.06pp，误差条未报，100 trials/任务下约 ±4pp 的 95% CI 内。
2. ~~RMBench 声明断层~~ ✅ 已补：最佳基线=Mem-0 28.5%，56.5/28.5≈1.98×。但 Table 2 里 WLA-0 的 Blocks Ranking Try（23%）明显弱于 Fast-WAM（37%）——"best average" 掩盖单任务落后；且四任务分数跨度 45/23/84/74，平均 56.5% 的方差极大，trials/任务未在已取段写明。
3. 真机报的是成功**计数**（"Success count comparison"）不是成功率——"count" 的分母（trials）未在已取段写明；Stack Cup 完成时间(s) 与延迟(ms) 数值只在 Fig. 3(d) 条形图内；"halves the completion time" 的基线 WAM 型号仍未知。

### 【关键数值表】
Table 1 全文（上，本轮逐字复验与上轮一致）；**Table 2 RMBench 全文（见【仿真验证】RMBench 条目）**；真机数值表（§4.3 后段）未取到。已确认 90.02/98.6/56.5 三数归属。

### 【一页总结】
WLA 用 AR VLM 骨干（RynnBrain-2B）+ 文本 subtask 意图 + meta-query 物理动力学（World Expert SANA-600M 监督、推理可关），2B 激活/40ms（RTX 5090）。**仿真协议全核实**：RoboTwin2.0 Clean 92.94/Rand 90.02（100 trials/任务）；LIBERO 98.6（四套件×10 任务×50 demo、100k 步、50 trials/任务；TTS 6 候选/想象视野 2→98.9）；RMBench 56.5（单任务 30k 步×448 批；四任务 45/23/84/74；近 2× Mem-0 28.5）。真机：AgilexRobotics Piper 双臂、4 长程任务（Unscrew Cap/Pack Object/Stack Cup/Dispose Trash）×60 demos、三设定（standard/OOD object/OOD scenario），成功计数与完成时间/延迟数值仅存 Fig. 3 条形图（文字未取到）。

### 【待核实】
1. §4.3 后段文字：三设定逐任务成功计数、Stack Cup 完成时间(s)、延迟(ms)、基线 WAM 型号（附录 C 同）。
2. §4.4 Learning New Tasks from Videos 的量化结果（README 训练侧有 45 seen + 5 unseen 同/跨具身共训设定）。
3. 附录 A 加速技术（40ms 测量条件：批大小/量化/视觉编码是否计入）与附录 B 额外表格。

---

## 3. MotionWAM（arXiv:2606.09215）— Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation

- **元信息**：Jia Zheng, Teli Ma, Yudong Fan, Zifan Wang, Shuo Yang, Junwei Liang（Mondo Robotics / HKUST(GZ) / HKUST）；v1 2026-06-08（10 页，CC BY 4.0；本轮取到接近全文）。论文未给项目页/GitHub 链接。
- **来源级别**：**L3**（§1–§6 全文 + Table 2/3 逐字；**附录 A–F 未取到**）。
- **截断位置**：本轮 ar5iv 截于 References 前段（§6 Limitations 之后）；arXiv 原生 HTML v1 截于 §4.2 中段（"even π0.5 stays under 20%" 处）。

### 【仿真验证】
- **论文没有仿真评测**：§4 开头 verbatim "We design our real-robot evaluation around three questions: (Q1) does coupling a video world model into the policy yield a tangible advantage over strong VLA baselines on whole-body humanoid loco-manipulation? (Q2) is each of the three training stages necessary, and what does each contribute? (Q3) can WAM runs fast enough for closed-loop humanoid control compared to VLA baselines?"——全部实验为真机 Unitree G1；§6 自述 "our evaluation does not include a controlled novel-object generalization study—training and test object sets share visual similarity, and we do not report success on strictly out-of-distribution objects."（原文未写仿真/OOD 评测，全文级）。
- 数据侧（§3.3 verbatim）："Stage 1: egocentric video pretraining. We assemble roughly **2,136 hours** of egocentric human video and humanoid-robot video... Stage 2: cross-embodiment action post-training... heterogeneous mixture of Unitree G1 humanoid data spanning different end-effectors and action-annotation formats... Stage 3: whole-body fine-tuning... collected on Unitree G1 (**200 episodes per task** across the nine real-world tasks)"。

### 【实机验证】
- 平台（§4.1 verbatim）："We conduct all real-world experiments on a **Unitree G1 humanoid with dual ALOHA2 grippers**, observing the scene through a head-mounted **Intel RealSense D435i RGB camera**. Whole-body teleoperation demonstrations are collected via a **PICO VR three-point tracking setup retargeted to the robot through SMPL**, and at deployment the policy outputs are tracked by the **SONIC** whole-body controller. MotionWAM and all baselines run as WebSocket policy servers on a **single NVIDIA RTX 4090** workstation and are queried in closed loop by the on-board controller."
- 任务与 trials（§4.1 verbatim）："we evaluate on a suite of **nine** real-world whole-body loco-manipulation tasks that jointly stress the five core capabilities highlighted in Fig. 1: _waist control_, _height regulation_, _squatting locomotion_, _task-driven foot interaction_, and _body–hand coordination_... Each method is tested for **20 trials per task**"；正文出现的任务名：Kick Soccer / Load Cart / Retrieve Item / Wipe Board / Do Laundry / Lift Basket / Toss Garbage（+2 个未在已取段出现，清单在附录 A.1）。
- 基线（§4.1 verbatim）："Diffusion Policy and ACT as representative non-VLA visuomotor policies; π0.5 and GR00T-N1.7 as state-of-the-art generalist VLAs; and Qwen3DiT, a custom parameter-matched ablation that pairs the Qwen3-VL 2B backbone with our Motion DiT"；"All baselines consume the same egocentric RGB observations, language goals, and proprioceptive states as MotionWAM, and emit unified motion latents in the same action space."
- 主结果（§4.2 verbatim）："MotionWAM wins on every task and lifts the overall success rate from **43.9% (the strongest baseline, GR00T-N1.7) to 76.1%, an over 32% absolute gain**."——"over 30%"=**绝对差 32.2pp**（上轮疑问已解）。逐任务差距（verbatim）："Kick Soccer (+40%), Load Cart (+40%), Retrieve Item (+40%), Wipe Board (+45%), and Do Laundry (+30%)"；"the VLM-only baseline (Qwen3DiT) collapses to near-zero success on every locomotion-heavy task and even π0.5 stays under 20% overall"。**逐任务数值在 Fig. 4 条形图内，文字版未在已取段出现**。
- 消融（§4.3 Table 2 逐字，20 trials/任务、5 任务子集、Stage 3 全开）：
  | Variant | Stage 1 | Stage 2 | Lift Basket | Retrieve Item | Load Cart | Toss Garbage | Kick Soccer | Avg. |
  |---|---|---|---|---|---|---|---|---|
  | w/o Stage 2 | ✓ | – | 65 | 45 | 30 | 30 | 40 | 42.0 |
  | w/o Stage 1 | – | ✓ | 70 | 75 | 60 | 35 | 55 | 59.0 |
  | Full | ✓ | ✓ | 80 | 90 | 75 | 45 | 60 | 70.0 |
  结论（verbatim）："removing Stage 1 and Stage 2 leads to **11% and 28% absolute performance drop** respectively."
- 频率（§4.4 Table 3 逐字，单卡 NVIDIA A100）：GR00T-N1.7 1.6B→**6.5 Hz**；Qwen3DiT 2.3B→**9.0 Hz**；Cosmos Policy 2.0B→**0.7 Hz**；MotionWAM 2.5B→**4.9 Hz**。"The reported frequency is the _chunk-wise_ rate at which the policy emits a whole action chunk."；"against Cosmos Policy—another world-model-based policy with a comparable parameter count—MotionWAM is **seven times faster** (4.9 Hz vs. 0.7 Hz)"。
- 动作表示（§3.1 verbatim）：SONIC 共享 latent "bottlenecked by Finite Scalar Quantization with **22 tokens of 32 levels each**, so the SONIC token kt∈{−1,−15/16,…,1}⁶⁴ is a **64-dimensional discretized vector**"；连续部分=左右夹爪/灵巧手指令，直驱手部。
- 局限（§6 verbatim）："our evaluation does not include a controlled novel-object generalization study—training and test object sets share visual similarity, and we do not report success on strictly out-of-distribution objects."；单目 FOV 丢失/头部视角漂移即失稳；三阶段配方仅在 G1 验证。

### 【why 追问】
1. "over 30%" 已解=绝对差 32.2pp（43.9→76.1，最强基线 GR00T-N1.7）；但 **9 任务×20 trials=180 trials/方法**，单任务 20 trials 下 Kick Soccer +40% 等逐任务差距的 95% CI 约 ±20pp，Fig. 4 不给误差条。
2. 76.1%（9 任务）与 Table 2 消融 Full Avg 70.0（5 任务子集）不是同一任务集，不能互相推演；且 Diffusion Policy/ACT 的分数未在正文出现（附录 D 只有训练配置）。
3. 无仿真评测、无 OOD 物体测试（§6 自认）——"泛化"主张全部由 9 个已知任务×20 trials 承载，与 Rolling-WAM 真机（3 任务×20 trials）同量级；两文都未给真机误差条。

### 【一页总结】
MotionWAM=双 DiT（Cosmos-Predict2.5-2B 视频 DiT 单次前向取 τf≈1 中间去噪特征 + Motion DiT）→ 统一 motion latent（SONIC 22 token×32 级 64 维 + 连续手部通道），三阶段训练（2,136h 第一人称视频预训练→跨具身 G1 数据→200 eps/任务×9 任务遥操作微调）。真机 Unitree G1（ALOHA2 双夹爪、D435i、SONIC 跟踪、RTX 4090 policy server）：9 任务×20 trials，**76.1% vs 最强 VLA 基线 GR00T-N1.7 43.9%（+32.2pp 绝对）**；π0.5 <20%、Qwen3DiT 近 0；消融 Stage1 −11pp、Stage2 −28pp；4.9Hz@A100（Cosmos Policy 0.7Hz 的 7×）。**无仿真评测、无 OOD 物体实验（§6 自认）。**

### 【待核实】
1. Fig. 4 九任务逐任务数值（条形图）与 Diffusion Policy/ACT 的分数（正文未写，附录 D 仅训练配置）。
2. 附录 A.1 九任务语言提示全文与剩余 2 个任务名；附录 E 各阶段数据构成（Stage 2 数据集名单）。
3. 附录 B 遥操作硬件细节（VR→SMPL→G1 流程）。

---

## 4. ABot-M0.5（arXiv:2607.00678）— Unified Mobility-and-Manipulation World Action Model

- **元信息**：AMAP CV Lab（Ronghan Chen, Yandan Yang, Zuojin Tang, Dongjie Huo, Tong Lin, Haoning Wu, Haoyun Liu, Yuzhi Chen, Lulu Zheng, Botai Yuan, Tianlun Li, Mingxin Wang, Dekang Qi, Bin Hu, Wei Mei, Yuze Xuan, Haolong Yang, Yanqing Zhu, Mu Xu, Zhiheng Ma, Xinyuan Chang；v1 2026-07-01，v2 2026-07-06，CC BY-NC-SA 4.0）。代码 github.com/amap-cvlab/ABot-Manipulation。
- **来源级别**：**L2**（§1–§3.3 中段正文 L3；**§3.3 后段–§4、§5 Experiments 全部（5.1 设定/5.2 移动操作主结果/5.3 RoboTwin+LIBERO/LIBERO-Plus/5.4 消融/5.5 真机）与附录 8 未取到**）。
- **截断位置**：本轮 ar5iv 与原生 HTML v2 均截于 §3.3 Structured Joint Attention 段（"the model must still suppo..." 处）——§5 实验节在全文 60%+ 处，不可达。

### 【仿真验证】
- abs verbatim（沿用）："Experiments on challenging mobile and fine-grained manipulation benchmarks demonstrate that ABot-M0.5 achieves state-of-the-art performance in both long-horizon task success and fine-grained control accuracy."——**数值层未取到**：§5.2（mobile manipulation 主结果，RoboCasa365 归属待正文证实）、§5.3（RoboTwin 2.0 94.1 / LIBERO / LIBERO-Plus 83.4 设定）均在截断区。
- 方法骨架（§2.2/§3.1 verbatim，本轮补）：三瓶颈="Temporal Granularity Mismatch / Action Structure Mismatch / Rollout Condition Mismatch"；解法="Video Latent zt+1 → Frame-level Motion Intents (Bridging Space) → Robot Action at"；D-MoT（§3.3 verbatim）："we explicitly decouple the action space at into two distinct subspaces: manipulation a_manip and mobility a_move... strict channel-to-subtower assignment, where each sub-tower is equipped with its own dedicated feed-forward network (FFN) and prediction head"；Dream Forcing="exposes inverse dynamics learning to self-dreamed videos, directly aligning the conditioning context of training with that of inference"；骨干="built upon the Wan2.2 video diffusion backbone"（3D VAE + UMT5 文本编码）。
- 训练（§4 TOC 层级）：Pretraining Data → World Model Pretraining → Latent Action Model Pretraining → Progressive SFT（Stage I 联合世界模型+逆动力学；Stage II Dream Forcing）→ Efficient Structured Attention & Latent Augmentation——正文细节未取到。

### 【实机验证】
- Fig. 1 caption（verbatim，本轮补）："Right: ABot-M0.5 achieves state-of-the-art performance on both simulation benchmarks (radar chart) and real-world mobile manipulation tasks (bar charts). Bottom: Real-world execution sequences (e.g., **Arrange Flower, Find Toaster**) show successful long-horizon mobile manipulation, where blue and green colors distinguish the navigation and manipulation phases."——真机任务名见 Arrange Flower / Find Toaster；**§5.5 Real-World Experiments（Experimental Setup + Experimental Results）不可达**，平台与 trials 未取到。

### 【why 追问】
1. 待取文后补。

### 【一页总结】
（取文后补）

### 【待核实】
1. §5.1–5.3：mobile 主结果 benchmark 名与协议、RoboTwin 2.0 94.1 口径（Clean/Rand？trials/任务）、LIBERO/LIBERO-Plus 83.4 设定。
2. §5.4 消融数值（latent action / D-MoT / Dream Forcing / pretraining+SFT）。
3. §5.5 真机平台、任务清单（Arrange Flower/Find Toaster 等）、trials。

---

## 5. GE-Act 2.0（arXiv:2609.05588）— Pretraining and Scaling a World-Action Model for Robotic Manipulation

- **元信息**：Renhang Liu, Wenzhi Zhao, …, Sanping Zhou, Maoqing Yao, AgiBot Research Team（AgiBot / 上交等）；v1 2026-09-04，v2 2026-10-09。项目页 ge-act-v2.github.io（JS 渲染页，web_fetch 只取到空壳）。
- **来源级别**：**L2**（§1–§3.3.3 正文 L3 + Table 1 表征探针表逐字（本轮 ar5iv）；**§4 数据构成、§5 全部评测（5.1 协议/5.2 缩放/5.3 指令遵循/5.4 仿真 OOD）、附录 A–C 未取到**；数值层仍以 abs 为准）。
- **截断位置**：本轮 ar5iv 截于 §3.3.3 Toward Unified Understanding and Generation（VLM 门控公式后）；原生 HTML v2 截于 §3.2.2 Instruction grounding 段——§5 与附录不可达。项目页（JS 渲染）上轮已确认为空壳。

### 【仿真验证】
- abs verbatim（已取）："We evaluate pretrained checkpoints directly, **without per-task fine-tuning, on 100 tasks across 20 manipulation skill groups** with **held-out scenes, backgrounds, lighting, and object instances**. Scaling co-training data from 300 to 30,000 hours raises success from **17.1% to 44.1% on G1-OP** and from **13.4% to 31.1% on G2-90D**; despite comprising less than 2% of the co-training data, G2-90D improves by 17.7 points, suggesting cross-embodiment transfer. Gains span **19/20 and 18/20 skill groups**, and skill-specific coverage strongly correlates with zero-shot out-of-distribution (OOD) success (**Pearson r=0.80; Spearman rho=0.85**). Under the same protocol, the model grounds object, color, shape, and position references in **at least 90% of trials** and follows explicit instructions even when they conflict with an already-committed behavior or a conventional scene association."
- **评测哲学（§2.5 verbatim，本轮补）**："We therefore report no-fine-tuning evaluation as the primary measure of pretrained capability and state which visual factors are excluded and which forms of task adaptation are absent (Section 5.1). Simulation and world-model benchmarks (LIBERO, RLBench, WorldArena) provide **horizontal reference points rather than direct evidence** for the capability acquired by our manipulation-video pretraining."——仿真 benchmark 被明确定位为横向参照而非直接证据。
- **仿真评测存在但未取到**：§5.4 "OOD Performance on Simulation Benchmarks" + 附录 A "OOD Generalization in Simulation"（A.1 RoboTwin Clean-to-Random Generalization、A.2 GenieSim-Instruction、A.3 LIBERO-Plus Robustness across Perturbation Axes）——结构与节名已确认（TOC），数值不可达。
- KASO（abs verbatim）："The components are then jointly trained with knowledge-aligned selective optimization (KASO), which reduces mismatched supervision by selecting only predicted futures judged behaviorally compatible with the recorded action."
- **表征探针（§3.2.2 Table 1 逐字，本轮补）**——CoAE 验证线（可控探针而非基准）："Action recovery... We freeze each encoder and train the same two-layer inverse-dynamics probe... on a set of **400 samples** from a visually cluttered shelf task... We report action MAE on **100 held-out samples**"；指令落地探针="**5,000 GenieSim-Instruction episodes**... distributed uniformly across all ten benchmark tasks and instruction categories"。
  | Representation | Spatial downsampling | Channels | Pixel decoder | Action MAE ↓ | Caption-match error (%) ↓ |
  |---|---|---|---|---|---|
  | CoAE | 64×64 | 512 | ✓ | 0.01673 | 2.05 |
  | DC-AE | 64×64 | 128 | ✓ | 0.02573 | 2.71 |
  | DINOv3 | 16×16 | 1024 | ✗ | 0.01273 | 2.75 |
  | V-JEPA 2.1 | 16×16 | 1024 | ✗ | 0.01487 | 4.15 |
  | SigLIP | 32×32 | 2048 | ✗ | 0.03138 | 3.28 |
- 各 skill group 逐值表、trials/任务、G1-OP/G2-90D 平台定义（§5.1/附录 C.1）：**未取到**。

### 【实机验证】
- 主评测=真机零样本 OOD（abs verbatim，沿用）："We evaluate pretrained checkpoints directly, **without per-task fine-tuning, on 100 tasks across 20 manipulation skill groups** with **held-out scenes, backgrounds, lighting conditions, and object instances**."；Fig. 1 caption（verbatim，本轮补）："Scaling co-training data from 300 to 30,000 hours improves zero-shot OOD success across a **100-task real-robot suite**. Each tile in the right panel is one task, coloured by skill group and shaded by success rate"。
- 协议哲学（§1 verbatim，本轮补）："We instead evaluate GE-Act 2.0 directly under out-of-distribution (OOD) conditions, without task- or embodiment-specific SFT, and **keep the protocol fixed across scaling checkpoints**."
- 缩放数值（abs，沿用）：300→30,000h：17.1%→44.1%（G1-OP）、13.4%→31.1%（G2-90D）；G2-90D +17.7 点（占比 <2%）；"The gains span **19/20 and 18/20** skill groups, respectively"；"skill-specific coverage strongly correlates with zero-shot OOD success (**Pearson r=0.80; Spearman ρ=0.85**)"；"the model grounds object, color, shape, and position references in **at least 90% of trials**"。
- **trials/任务、100 任务清单、G1-OP/G2-90D 具体平台、104ms 部署数值：未取到**（§5.1/§B.6 Deployment/附录 C 不可达）。

### 【why 追问】
1. ~~公共参照问题~~ 部分解决：§5.4+附录 A 确有 RoboTwin C2R / LIBERO-Plus 扰动轴仿真 OOD 结果，但数值不可达；且 §2.5 已明言仿真 benchmark 只是 "horizontal reference points rather than direct evidence"——主证据=自建 100 任务真机套件（held-out 场景/物体/光照/背景）。trials/任务未取到，44.1% 的方差不可估。
2. Pearson r=0.80/Spearman ρ=0.85 的样本单位（20 个 skill group 还是 100 个任务？）未取到——若 n=20，相关显著性很弱；"at least 90% of trials" 的维度分解（object/color/shape/position 各自百分比）在附录 C.3，不可达。

### 【一页总结】
GE-Act 2.0=从零预训练的 WAM（CoAE 64× 压缩控制导向自编码器 + SVP 单步视觉规划器 + 独立预训练 IDM；KASO 只对行为兼容的预测未来做共训）。验证哲学：**无 SFT 零样本 OOD 真机 100 任务（20 skill groups、held-out 场景/背景/光照/物体）为主证据**，sim benchmark（RoboTwin C2R/LIBERO-Plus，§5.4/附录 A）定位为 "horizontal reference points"。300→30k h 缩放：G1-OP 17.1→44.1%、G2-90D 13.4→31.1%；r=0.80/ρ=0.85；指令指代 ≥90% trials。**trials/任务、逐 skill group 表、平台定义、104ms 未取到。**

### 【待核实】
1. §5.1/§C.1：100 任务的构成与 trials；G1-OP/G2-90D 具体平台定义；OOD 协议细节（held-out 各维度如何交叉）。
2. §5.4/附录 A：RoboTwin C2R 与 LIBERO-Plus 扰动轴数值；§5.2 逐 skill group 缩放表；KASO 消融（§3.4.5 Controlled Real-Robot Ablation）。
3. §B.6 Deployment：104ms 部署数值与条件。

---

## 6. τ0-WM（arXiv:2606.01027）— A Unified Video-Action World Model for Robotic Manipulation

- **元信息**：Pengfei Zhou, Shengcong Chen, …, Jianlan Luo（Shanghai Innovation Institute / AGIBOT Finch）；v1 2026-05-31，v2 2026-08-23。项目页 tau0-wm.github.io，代码 github.com/sii-research/tau-0-wm，HF sii-research/tau-0-wm。
- **来源级别**：**L2+**（§I–§VI-B 方法全文 L3（本轮 ar5iv 复验至 §VI-B 末）；§III 数据构成 L3 沿用第一波；**§VII 数值表仍未取到——两源再试确认**；官方项目页定性 + GitHub README 动作空间/部署接口）。
- **截断位置**：本轮 ar5iv 截于 §VI-B LAR 段 "we perform a second policy query conditioned on the selected future rollout. Specifically, the rollout latent" 处（§VII Experimental Evaluation 之前）；arXiv 原生 HTML v2 截于 §VI Alg. 1 第 4 行——§VII-A Main Results / §VII-B Ablation / 附录 A（Training and Deployment Details）不可达。README 本轮重试成功（无数值表）。

### 【仿真验证】
- 已核实（第一波 §III verbatim，数据构成）："a 27.3K-hour training corpus from three sources: **17.8K hours of real-robot teleoperation on AGIBOT-G01, ARX manipulators, and dual-arm Franka systems**; **6.5K hours of filtered open-source UMI-style demonstrations collected with Gen-DAS Grippers**; and **3.0K hours of open-source egocentric human interaction videos**."（项目页同口径："5B world model… approximately 27,300 hours"）
- 标准仿真 benchmark（RoboTwin/LIBERO）：已获取范围内原文未写（第一波结论不变）。

### 【实机验证】
- 任务（Fig. 3 caption verbatim，本轮补全）："(a) Storing different tools on the desk into their corresponding places in the toolbox (Toolbox). (b) Unzipping the school bag, storing objects into it, and zipping up (School Bag). (c) Connecting the hose to the faucet and securing it (Faucet). (d) Storing the badminton shuttlecocks and closing the lid (Badminton)."——四长程任务；Fig. 3 位于 §VI-B 尾，紧随其后的 §VII 主表在截断区。
- 动作空间（官方仓库 README verbatim，本轮重试成功）："*state*... the **absolute** poses of two end-of-effectors, including 14 channels (xyz and quaternion with order *xyzw*)... *gripper state*... 2 channels, ranging from 0 to 120 (0 for opening and 120 for close). *action*... the **absolute** poses of end-of-effectors with shape {T, 16}"（左 eef 7ch+左夹爪 1+右 eef 7ch+右夹爪 1）；预训练口径："In the pretraining stage, τ0-WM is optimized to predict the **relative** pose of end-effectors, including 20 channels (xyz and 6d-rotation for each arm)."
- 项目页 verbatim："On the **four tasks unseen in the pretraining data**, τ0-WM obtains the **best average success rate** and is strongest on most precision-sensitive tasks. In particular, Faucet remains difficult for all methods, indicating that the task is far from saturated; nevertheless, τ0-WM is more robust under these strict alignment constraints."
- **每任务成功率、trials、baseline 名单、评测 embodiment：未取到**（§VII 数值表在截断区；项目页仅图）。数据侧三 embodiment（AGIBOT-G01/ARX/双 Franka）已核实；评测 embodiment 与任务映射仍未写明（已获取范围）。
- 测试时协议（第一波 §VI verbatim 骨架）：RCS（re-denoising consistency）排序 → 阈值 γ 触发 ACVS 评估 → LAR 修正。

### 【why 追问】
1. 第一波问题仍成立：四任务 "best average SR" 无任何数值就下结论；本轮项目页补充了 "Faucet remains difficult for all methods"——即优势主要来自三任务、最难的 Faucet 差距最小，这恰恰需要逐任务数值才能评估。trials 数（Faucet 精度敏感任务，若无 20+ trials 结论不可靠）完全未披露。
2. "four tasks unseen in the pretraining data"——unseen 的定义（指令未见？物体未见？场景未见？）在已获取范围未写明。

### 【关键数值表】
未取到（主结果表在 §VII，截断区；项目页以图片呈现；README 无表）。两源再试确认截断（ar5iv §VI-B 末 / 原生 HTML v2 §VI Alg.1）。

### 【一页总结】
τ0-WM=统一视频-动作世界模型（VAM 政策 + ACVS 动作条件模拟器，共享 Wan2.2-5B 骨干，27.3k h 异构数据），测试时 propose-evaluate-revise。真机四长程任务（Toolbox/School Bag/Faucet/Badminton）"best average SR，Faucet 均未饱和"。**数值层（trials/逐任务 SR/baselines）仍未取到，是本线最薄的一篇。**

### 【待核实】
1. §VII 全部表格（四任务×baselines、trials、TTS 消融、heterogeneous 预训练消融）——本轮 ar5iv/原生 HTML v2 均再试，截断点确认（§VI-B 末 / §VI Alg.1），唯一可行路径=PDF 抽页。
2. 评测 embodiment 与任务-平台映射（README 动作空间为双臂 16ch 绝对位姿，未指定四任务对应平台）。
3. ~~GitHub README 重试~~ ✅ 已成功：无数值表，但有动作空间/部署接口（已录上）。

---

## 7. DreamZero（arXiv:2602.15922）— World Action Models are Zero-shot Policies

- **元信息**：Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, …, Yuke Zhu, Linxi "Jim" Fan, Joel Jang（NVIDIA）；2026-02-17。项目页 dreamzero0.github.io（上轮已取一手数值），代码 github.com/dreamzero0/dreamzero，Eval Gallery dreamzero0.github.io/evals_gallery（100+ 零样本任务）。
- **来源级别**：**L2+**（§1–§4.1 正文 L3 + Table 1 逐字（本轮 ar5iv 复验一致）；**官方项目页逐字给出真机协议数值**；§5 主表与 PolaRiS/Genie Sim 3.0/DROID-sim 得分仍未取到——两源再试确认不可达）。
- **截断位置**：本轮 ar5iv 重取仍截于 §4.1 末（"We open-source the checkpoint and inference code to run some DROID-sim evals in PolaRiS ([Jain et al. 2025])" 处）；§5 Experimental Results 与附录 F/G 不可达（arXiv 原生 HTML 上轮截于 §3.2.5 末，ar5iv 更远但到不了 §5）。官方仓库 README：PolaRiS 与 Genie 3.0 sim 支持均标 **"Coming Soon"**（评测代码尚未开源）。

### 【仿真验证】
- 已核实（第一波 §1+§4 开头 verbatim）："we open-source our model weights, inference code, and code to run publicly available real-world (RoboArena) and simulation benchmarks (**PolaRiS and Genie Sim 3.0**)”；Genie Sim 3.0=100 任务、DreamZero 未在其 10k h 仿真数据上训练（脚注）；两 embodiment（AgiBot G1 + Franka）；baselines=GR00T N1.6 与 π0.5（各 from-scratch/from-pretrained 两初始化）、"trained on identical data as DreamZero: ∼500 hours of teleoperation data we collected for AgiBot G1, and DROID for Franka"、"matching total batch size and gradient steps"。
- **PolaRiS / Genie Sim 3.0 / DROID-sim 具体得分表：未取到**（§5 截断区；项目页未列这三项）。

### 【实机验证】（数值=官方项目页，verbatim 抄录）
- **AgiBot 预训练评测**（项目页 verbatim）："We evaluate pretrained models out-of-the-box on tasks from the pretraining distribution, but in _zero-shot environments with unseen objects_. DreamZero (also trained from scratch) achieves **62.2% average task progress**—over 2× higher than the best pretrained VLA baseline (**27.4%**). VLAs trained from scratch achieve near-zero performance... For tasks entirely absent from training—such as untying shoelaces and shaking hands—DreamZero reaches **39.5% task progress** while VLAs again struggle... **We evaluate 80 rollouts per checkpoint across 4 robots, each in different environments with different objects.**"
- 任务构成（项目页 verbatim）：AgiBot "10 seen tasks + 10 unseen tasks"（seen：Pick & Place Fruit、Take Fruit out of Bag、Wipe the Mess、Pick & Place Fork/Spoon、Pen in Cup、Cup on Coaster、Stack Bowls/Cups、Folding Shirts、Folding Shorts、Stacking Clothes；unseen：Untie Shoe/Gift、Take Hat Off/On from Mannequin、Draw a Circle、Take out the Straw、Stack the Cubes、Paint with the Brush、Iron the Clothes、Shake Hands with the Human、Fold the Map、Pulling Cart）。
- **DROID 评测**（项目页 verbatim）："We evaluate on **20 seen tasks and 20 tasks with _unseen verbs_** (actions absent from DROID). DreamZero outperforms pretrained baselines, achieving **49% task progress on unseen verbs compared to 25–32% for state-of-the-art VLAs**."
- **Post-training**（项目页 verbatim）："We post-train on three downstream tasks: _shirt folding_, _fruit packing_, and _table bussing_... enables stronger post-training results across three tasks, indicating that environment generalization is retained after post-training."（Table Bussing：10 consecutive evaluations，Task Progress 81%，单次 session 未剪辑录像）。
- **新 embodiment 适配**（项目页 verbatim）："With only **30 minutes** of play data (**55 trajectories**), DreamZero adapts to the YAM robot and generalizes zero-shot to novel objects like pumpkins, teddy bears, and paper bags... what previously demanded hundreds of hours of demonstrations, we accomplish in 30 minutes (no other YAM data was used)."
- 延迟（项目页 verbatim）："Through model, system, and implementation optimizations, DreamZero achieves real-time inference at **150ms per action chunk**—enabling **7Hz closed-loop control**. Combined with asynchronous inference and action chunk smoothing..."；DreamZero-Flash 在 1 步扩散下保持性能。
- 论文 §3.2.1/§3.2.5（verbatim，本轮补）："A naive implementation of DreamZero on a single GPU requires approximately **5.7 seconds per action chunk** due to three bottlenecks: (1) iterative denoising across **16 diffusion steps**..."；"we reduce the diffusion steps from four to one, cutting inference from **∼350ms to ∼150ms** with minimal performance loss (Table 3)."
- 官方仓库 README（2026-02-27 News，verbatim）："DreamZero is **#1 on both MolmoSpaces and RoboArena**! DreamZero-DROID is trained *from scratch* using only the DROID dataset — no pretraining on large-scale robot data, unlike competing VLAs."；推理实测（README verbatim）："After warming up, inference takes **~0.6s on GB200 and ~3s on H100**."（DiT caching 开；RoboArena 为真机榜单，分数页未在本线范围取到）。
- 交互式提示：qualitative 15 任务清单（Push Elevator Button/Pop the Balloon/…）+ evals_gallery 100+ 任务。
- 指标口径：**"average task progress" 为连续进度分**（非二值 SR；milestone rubric 在附录 F/G，未取到）。

### 【why 追问】
1. 62.2% vs 27.4% 的 ">2×" 基线是 "best pretrained VLA"（GR00T N1.6 与 π0.5 二选一，具体哪个未写）；且 80 rollouts/checkpoint×4 robots 摊到 20 任务=每任务每机器人仅 4 rollouts——单任务方差极大，62.2% 的置信区间未给；"task progress" 的里程碑 rubric（附录 F）决定分数尺度，未取到。
2. PolaRiS/Genie Sim 3.0 只有脚注定性句（"code to run"），项目页也不列得分——零样本 sim 基准结果若存在为何两处都不给数值？
3. 项目页 "unseen verbs" 49% vs "25–32%" 是区间表述，具体基线名称与数值未给（§5 截断）。

### 【关键数值表】
真机数值=项目页 verbatim（62.2/27.4/39.5/49/81%）；Table 1 效率表沿用第一波（H100/GB200：Baseline 1×/1.1× → +CFG 1.9×/1.8× → +DiT Caching 5.5×/5.4× → +Torch Compile+CUDA Graphs 8.9×/10.9× → +Kernel & Scheduler 9.6×/14.8× → +Quantization(NVFP4) —/16.6× → +DreamZero-Flash —/38×）。

### 【一页总结】
DreamZero=14B AR 视频扩散 WAM，38× 加速至 7Hz/150ms。真机（项目页口径）：AgiBot 10 seen+10 unseen 零样本，62.2% vs 最佳 VLA 27.4%（80 rollouts/checkpoint×4 robots）；unseen 任务 39.5%；DROID unseen verbs 49% vs 25–32%；post-train 3 任务、table bussing 81%；YAM 30min/55 轨迹新 embodiment。**PolaRiS/Genie Sim 3.0 的 sim 数字与 §5 表仍未取到。**

### 【待核实】
1. §5 全部表格（PolaRiS、Genie Sim 3.0、AgiBot G1 环境/任务泛化、DROID-sim）——本轮 ar5iv 再试确认不可达（截于 §4.1 末）；README 显示 PolaRiS/Genie sim 评测代码仍 "Coming Soon"。
2. 附录 F/G 的 trials 与 milestone rubric 定义（"average task progress" 的刻度）。
3. YAM 适配任务的量化成功率（项目页 qualitative）；RoboArena 榜单具体分数（可另取 robo-arena.github.io/leaderboard）。

---

## 8. XPACE（arXiv:2609.17372）— Joint World and Action Modeling from Heterogeneous Experience

- **元信息**：Jiacheng Wei, Jerry Bai, …, Yixiao Ge（XPENG Robotics）；2026-09-15。官方项目页 xpeng-robotics.github.io/xpace/。
- **来源级别**：**L2+**（论文 §1–§3.4.1 正文 L3 + **§3.2 真机部署段 verbatim（本轮新增）** + 官方项目页逐字镜像实验节全部数值与协议；§4 正文仍未取到——本轮按任务要求重试两源，截断点不变）。
- **截断位置**：本轮重试：arXiv 原生 HTML v1 截于 §3.2 skeleton conditioning 段（"The skeleton patch embedding is zero-initialized..." 处）；ar5iv 截于 §3.4.1 L_III-sim 公式行——与第一波完全一致，§4 Experiments 不可达（数值源仍为项目页）。

### 【仿真验证】（项目页来源，verbatim，沿用第一波核实）
- "Simulation and offline action prediction use **19 evaluation subsets** collected separately from training. In-distribution (ID) subsets cover representative pick-and-place interactions; out-of-distribution (OOD) subsets vary scenes, objects, instructions, and behaviors."；"480 × 832 video (height × width); 97- and 193-frame horizons."
- 指标：PSNR/SSIM/OFS/DINO/Latency/Memory；离线策略 teacher-forced MSE（19 子集聚合为 "Benchmark"）。**无公共仿真 benchmark（无 LIBERO/RoboTwin）**。
- 数值（项目页 Table 1–3）：token addition 最优 ID PSNR 18.07/SSIM 0.6030/DINO 0.9809；SGF 193 帧 ID PSNR 16.94→17.83（+0.89dB）、OOD 17.74→18.75（+1.01dB），快 3.00×/3.30×（50→8 ODE 步）；Phase II-c+SGF 97 帧 ID PSNR 19.73。

### 【实机验证】（项目页来源，verbatim，沿用第一波核实）
- **真机部署协议（§3.2 "Real-robot deployment" verbatim，本轮新增）**："We deploy XPACE for closed-loop control of the IRON robot, using streaming visual encoding and asynchronous action execution."；"Streaming VAE encoding processes incoming observations incrementally, producing one latent frame per four new video frames while the robot continues executing queued commands. After 4 video frames, the policy uses the accumulated visual latents, current robot state, and task instruction to generate a new 16-step action plan."；"Overlapping-horizon stitching connects successive plans without pausing execution for inference. Predicted actions are converted to robot control units and interpolated for command publication at approximately **10 Hz**."
- 平台："Real robot: **IRON-R01-1.11**; **20 trials per task per method with matched resets**. Normalized task progress averages per-trial milestone credit; success requires full completion."
- 三任务：banana pick-and-place / water pouring / bowl stacking（"Bowl stacking is absent from robot demonstrations but present in human and bridge data"）。
- 成功率（图 7 数据）：GR00T 20/0/0→**6.7**；DreamZero 65/5/50→**40.0**；XPACE 80/50/75→**68.3**。进度：GR00T 0.36；DreamZero 0.68；XPACE 0.84。
- 鲁棒性（图 8）：Full-data vs robot-only，20 trials/任务：香蕉五固定位置 50→80%；block-to-box 未见干扰物 30→50%。
- Recovery DAgger verbatim："Fine-tuning with an 8% mixture of simulated recoveries raises mean real-robot success from **61.7% to 86.7%** and task progress from 0.81 to 0.93. Water pouring improves most, from 50% to 95% success."（⚠️ 与 68.3% 主基准非同一套件：61.7→86.7 用香蕉/倒水/可乐）。
- 人类→机器人迁移：qualitative 三例（关抽屉/面包转移/叠三碗），无量化成功率。

### 【why 追问】
1. 沿用第一波：主基准（68.3%）与 DAgger（61.7→86.7%）任务集不同（stack↔cola），不能作同口径前后对比；20 trials/任务下 25pp 差异的 CI 约 ±20pp。
2. 沿用第一波：不用公共 benchmark 做策略线，PSNR/SSIM 与世界模型质量之间缺可复现的策略评测线。

### 【关键数值表】
见【仿真验证】/【实机验证】两节（项目页 Table 1/2/3 + 图 7/8，第一波已逐字抄录；本轮无新增）。

### 【一页总结】
XPACE=共享因果视频 Transformer 的 MoT（政策模式+骨架/相机条件模拟器），5000h 数据金字塔 + 失败集 F；Stage I→III-sim SGF→III-policy 8% 合成恢复。真机 IRON-R01-1.11：3 任务×20 trials，68.3% 超 DreamZero 40.0%、GR00T 6.7%；DAgger 后 61.7→86.7%（异套件）。§4 论文正文仍未取到（与第一波结论一致）。

### 【待核实】
1. 论文 §4.3.1 正文表与项目页图 7/8 逐数字一致性——本轮原生 HTML 与 ar5iv 再试均确认截断于 §3.2/§3.4.1，唯一可行路径=PDF 抽页核对。
2. 19 个评测子集构成（ID/OOD 各几、任务名）。

---

## 9. Rolling-WAM（arXiv:2609.30247）— World Action Models with Rolling Imagination

- **元信息**：Yinghua Zhou¹,² (USC/Brown), Junjie Ye¹, Yiqi Zhao¹, Hao Dong¹, Celina Shiyu Wang¹, Ruohai Ge¹ (USC), Tingyi Yang¹,³ (USC/Fudan), Basile Van Hoorick⁴ (TRI), Gaurav Sukhatme¹ (USC), Vitor Guizilini⁴ (TRI), Yue Wang¹ (USC)；v1 2026-09-24，v2 2026-10-05（10 页 7 图 5 表，CC BY 4.0）。项目页 rolling-wam.github.io，代码 github.com/zyinghua/Rolling-WAM。
- **来源级别**：**L2+**（§I–§IV-B 正文 L3 + Table I/II 完整——本轮 ar5iv 重取逐字复验一致；**Table III（论文版延迟）/IV（真机逐任务）/V（消融）与 §IV-C 后段/§IV-D 后段/§IV-E/F 未取到**；G1 平均分 85.0 与延迟条形图数值来自官方项目页）。
- **截断位置**：本轮 arXiv 原生 HTML v2 截于 Table I LingBot-VA 行（"98.5 | 99.6 | 97.2 | <"）；ar5iv 截于 Table II 后 "Baselines. We compare with both VLA and WAM policies, including π0 [30]" 处——两源截断点与上轮一致，Table III–V 不可达。GitHub README 无数值表（"Code and checkpoints are being prepared and will be released soon."）。

### 【仿真验证】
- **LIBERO（§IV-A1 verbatim）**："We evaluate the Spatial, Object, Goal, and Long suites of LIBERO, each containing 10 tasks and 500 demonstrations. Observations include external and wrist RGB views. We train one policy across all four suites for **10 epochs with an effective batch size of 128**, and evaluate **each task with 50 rollouts**."
- **RoboTwin 2.0（§IV-A2 verbatim）**："We evaluate **50 bimanual manipulation tasks in the Clean and Randomized settings** of RoboTwin 2.0. Observations include RGB images from the head and two wrist cameras. We train one policy across 50 tasks using **2,500 clean demonstrations and 25,000 demonstrations from randomized scenes, for 5 epochs with an effective batch size of 1024**. **Each task is evaluated with 100 rollouts per setting**."
- **实现细节（§IV-B verbatim，本轮补全）**："We initialize the video expert from Wan2.2-TI2V-5B and reuse its pretrained text encoder and VAE. The action expert has 30 layers and hidden dimension da=1024 (approximately 1B parameters). Its backbone is initialized by interpolating the video expert's weights. We jointly train both experts and the proprioceptive encoder, while freezing the VAE and text encoder. All training uses benchmark demonstrations without additional embodied pretraining."；训练/采样："We use AdamW with learning rate 10⁻⁴, weight decay 10⁻², 5% linear warmup followed by cosine decay, and BF16 mixed precision. Video and action losses are averaged separately, with λv=λa=1. Initialization and rolling modes are sampled with probabilities 0.2 and 0.8... shifted noise schedule σ(τ)=ρτ/[1+(ρ−1)τ] with ρ=5... we use N=10 denoising steps per chunk, a window of W=5 chunks, and K=16 actions per chunk. Each steady-state replanning cycle therefore uses N/W=2 denoising steps. The classifier-free guidance scale is 1."
- **Table I（LIBERO，逐字，本轮复验）**：π0 96.8/98.8/95.8/85.2→94.1；π0.5 98.8/98.2/98.0/92.4→96.9；Motus 96.8/99.8/96.6/97.6→97.7；LingBot-VA 98.5/99.6/97.2/98.5→98.5；Fast-WAM 98.2/100.0/97.0/95.2→97.6；Joint-WAM 99.6/99.4/98.2/96.8→98.5；**Rolling-WAM 98.2/98.0/98.2/97.8→98.1**（无具身预训练，P.T.=✗）。（⚠️ 同 LIBERO 组件下 π0 平均本表记 94.1、WLA Table 1 记 94.2——(96.8+98.8+95.8+85.2)/4=94.15，两文舍入方向不同，见文末评测坑。）
- **Table II（RoboTwin 2.0，逐字）**：π0 65.9/58.4→62.2；π0.5 82.7/76.8→79.8；Motus 88.7/87.0→87.8；LingBot-VA 92.9/91.5→92.2；Fast-WAM 91.9/91.8→91.8；Joint-WAM 90.8/90.3→90.6；**Rolling-WAM 93.5/93.0→93.3**（caption："Rolling-WAM achieves the highest success rate in both settings."）。
- 结论句（§I verbatim）："It achieves a **98.1% average success rate on LIBERO and 93.3% on RoboTwin**, remaining competitive with or exceeding state-of-the-art WAMs."

### 【实机验证】
- **协议（§IV-A3 verbatim）**："We evaluate **three tasks on a Unitree G1 humanoid**: _Doll Placement_ (placing a plush dog in a box), _Plate Stacking_ (stacking three plates), and _Bead Pouring_ (transferring beads from a bottle into a glass). The policy receives a **320×224 egocentric RGB image and a 43-dimensional robot state**. It predicts **78-dimensional actions comprising a 64-dimensional SONIC motion latent and 7-dimensional commands for each hand**. We train one policy across all three tasks using **50 demonstrations per task, recorded at 10 Hz**. Training runs for **7,500 steps with an effective batch size of 192**. We execute actions at 10 Hz and **evaluate 20 trials per task**."
- **平均成功率**（来源：上一轮代理核实+官方项目页，verbatim）：项目页指标卡 "**85.0%** real-world G1 average success"（论文 Table IV 逐任务值未取到）。
- **延迟**（来源：上一轮代理核实+官方项目页，verbatim 条形图数值）："**215ms** steady-state replanning… Single NVIDIA A100 · Controlled RoboTwin 2.0 setup. visual encoding + denoising; warm-up and initialization excluded. No torch.compile, TensorRT, or custom CUDA kernels"；对照：Joint-WAM **978**、Fast-WAM **548**、π0.5 **296**、GR00T N1.7 **285** ms；4.5× 即 978/215≈4.55（§I verbatim："delivers a 4.5× steady-state replanning speedup over standard joint WAMs"）。
- 扰动/泛化：RoboTwin Rand. 档承担仿真侧泛化；真机未见扰动协议（已获取范围原文未写）。

### 【why 追问】
1. 4.5×（978→215ms）是 **A100 单卡 + RoboTwin 受控设定**下的稳态 replanning 数字，且**排除 warm-up/初始化**（初始化模式要 N=10 全步）；真机 G1 的延迟未报——85.0% 的真机成绩用了几步 denoising、是否同样滚动模式，§IV-E 未取到。且 215ms 含 visual encoding，但 Joint-WAM 978ms 是"标准 joint WAM"实现（同仓库复现）——与 Fast-WAM 548ms（本线第一波真机 190ms@RTX 5090D 是另一设定）不可跨文比较。
2. 93.3% RoboTwin 是 Clean 93.5/Rand 93.0 的平均——与 Fast-WAM 的 91.8、WLA 的 92.94（Clean）、WSA 的 93.1（仅 Rand）口径各异（Clean 档 vs Rand 档 vs 平均），跨论文排名必须拆开两档看。
3. 真机 3 任务×20 trials=60 trials/方法，Plate Stacking/Bead Pouring 是精度敏感任务，85.0 的 CI 约 ±9pp；且 baselines（Joint-WAM/Fast-WAM）真机数值在 Table IV（未取到），"competitive" 无从核对。

### 【关键数值表】
Table I / Table II（逐字，上，本轮 ar5iv 逐字复验一致）；延迟条形图数值（项目页）；Table III/IV/V 未取到（两源截断已再确认）。

### 【一页总结】
Rolling-WAM 把 joint video-action 去噪切成 W=5 个 chunk 的滑动窗（N=10 步/chunk、K=16 动作/chunk、稳态 2 步/chunk），跨 replan 周期分摊去噪；MoT（Wan2.2-TI2V-5B 视频专家 + 1B 动作专家 30 层）。LIBERO 98.1（无具身 PT）、RoboTwin Clean/Rand 93.5/93.0→93.3（均超 LingBot-VA 92.2）；稳态 replan 215ms=Joint-WAM 978ms 的 4.5× 快（A100，排除 warm-up）；真机 G1 三任务×20 trials 平均 85.0%（项目页）。

### 【待核实】
1. Table III 论文版延迟表（设定/GPU 是否与项目页 A100 一致）；Table IV 真机逐任务表（含 baselines）；Table V 消融（窗口大小/训练噪声调度/cross-chunk 注意力）——本轮原生 HTML 与 ar5iv 均已再试，截断点不变。
2. §IV-E/F 正文（真机部署协议、rolling 模式在真机的实际步数）。

---

## 10. Kairos（arXiv:2606.16533）— A Native World Model Stack for Physical AI（v3 标题；v1/v2 标题含 "Regret-Aware Native World-Action Model Stack"）

- **元信息**：Kairos Team（v1 2026-06-15，v2 2026-06-16，v3 2026-08-10 报告页眉日期；arXiv v3 提交日 2026-07-03）。代码 github.com/kairos-agi/kairos-sensenova，HF huggingface.co/kairos-agi，ModelScope modelscope.cn/collections/kairos-team/kairos30。与扩展 B 线重叠（动作/决策轴），本线侧重验证协议。
- **来源级别**：**L2**（§1–§2.3 方法正文 L3（本轮 ar5iv）；**§3–§6 全部评测与附录理论部分未取到**；数值层仅 §1 定性句）。
- **截断位置**：本轮 ar5iv 截于 §2.3 Theoretical Analysis "Problem Setup" 段（全文约 1/4 处）——§6 Evaluation Results 不可达。

### 【仿真验证】
- §1 verbatim（本轮补）："In general world-modeling and generative benchmarks as **WorldModelBench** and **DreamGen Bench**, Kairos matches or surpasses significantly larger counterparts... More importantly, when evaluated on rigorous embodied control and World-Action Model benchmarks, Kairos exhibits superior manipulation and generalization capabilities. **It achieves new SOTA milestones on trajectory-driven task suites including LIBERO-plus and RoboTwin 2.0**."——§6 评测结构（TOC 确认）：6.1 Embodied World Model Benchmarks（含消融）/ 6.2 World Action Model Benchmarks（含消融；LIBERO-Plus、RoboTwin 2.0 应在此）/ 6.3 General World Model Benchmarks / 6.4 Long-horizon Generation。
- LIBERO-Plus 89.0→90.8 的设定与对照、各表数值与 trials：**未取到**（§6 不可达；89.0→90.8 仍是上轮遗留的未证实数值）。

### 【实机验证】
- **目录中无真机实验节**：§6 四个小节均为基准评测（6.1 具身世界模型 / 6.2 世界动作模型 / 6.3 通用世界模型 / 6.4 长程生成），未见 real-robot/hardware 小节——已获取范围（含 TOC 全文）未见真机评测安排；若真机结果嵌入 §6.2 则不可达（按「未取到」计）。真机部署主张目前只由 §5.3 Inference Efficiency（timestep distillation / hardware-aware 优化 / 效率对比）支撑，数值同样不可达。

### 【why 追问】
1. 待取文后补。

### 【一页总结】
Kairos（v3）=Native World Model Stack：Qwen 系 VLM 理解 + 视频 VAE + LinearDiT（SWA/DSWA/GLA(GatedDeltaNet) 混合线性时序注意力，理论保证误差积累有界）；World Prediction=Video DiT + 1/5 规模 Action DiT（action-only 推理模式）。评测=4 类基准（§6.1–6.4：具身世界模型 / 世界动作模型 LIBERO-Plus+RoboTwin 2.0 / 通用世界模型 / 长程生成）+ §5.3 效率评测；**目录无真机实验节**。数值层未取到（截于 §2.3）。

### 【待核实】
1. §6.2 表格：LIBERO-Plus（7 轴移位协议？）与 RoboTwin 2.0 数值、trials、基线；89.0→90.8 归属。
2. §6.1/6.3 基准名单与数值；§6.4 长程生成评测设定；§5.3.3 效率对比数值（含消费级硬件）。
3. 是否任何真机评测（TOC 未见；若存在应嵌入 §6.2，需 PDF 核对）。

---

<!-- ================= 汇总表（重发轮终态，2026-10-27） ================= -->

## 汇总表：10 篇 × 来源级别 × 验证协议终态

| # | 论文 | 来源级别 | 仿真验证（协议/数值） | 实机验证（协议/数值） | 本轮核实的关键数值 | 仍缺失（截断确认） |
|---|---|---|---|---|---|---|
| 1 | WSA₁ | L2+（README 数值逐字 + v2 §4.1 协议） | RoboTwin 50 任务、27,500 demos；OOD-C2R/C2E 定义 verbatim | 7 任务×30 trials（PiPER/Lift2，两例外 34/20） | README：WSA-L RoboTwin-Rand 93.1、LIBERO 98.2、真机 SR 80.52/C 85.21 | 论文 §4.2 表（截于 Table 2 π0 行）；93.3 口径 |
| 2 | WLA | **L3 仿真全段 + §4.3 开头** | RoboTwin 92.94/90.02（100 trials/任务）；LIBERO 98.6（50 trials，TTS 6 候选→98.9）；RMBench 56.5 | Piper 双臂 4 任务×60 demos、三设定（数值在图 3） | **90.02=Rand ✓ / 98.6=LIBERO Avg ✓ / 56.5=RMBench Avg ✓（Table 2 全文）** | §4.3 后段成功计数/完成时间/延迟（图 3 内） |
| 3 | MotionWAM | **L3（§1–§6 全文）** | **无仿真评测（原文未写）** | G1 9 任务×20 trials；**76.1 vs 43.9（+32.2pp 绝对）**；消融/频率表全文 | 消融 70.0/59.0/42.0；4.9Hz@A100；200 eps/任务 | Fig.4 逐任务值（条形图）；DP/ACT 分数 |
| 4 | ABot-M0.5 | L2（§1–§3.3 中段） | 基准名单未证实（§5 不可达） | 真机任务名见 Fig.1 caption（Arrange Flower/Find Toaster） | 三瓶颈+D-MoT+Dream Forcing 骨架 verbatim | §5 全部数值（RoboTwin 94.1/LIBERO-Plus 83.4 仍二手） |
| 5 | GE-Act 2.0 | L2（§1–§3.3.3 + Table 1） | §5.4+附录 A（RoboTwin C2R/LIBERO-Plus 扰动轴）存在、数值不可达；§2.5：sim 只是 "horizontal reference points" | 100 任务×20 skill groups 零样本 OOD；44.1/31.1（abs） | Table 1 探针逐字（CoAE 0.01673/2.05） | §5.1 trials、G1-OP/G2-90D 定义、104ms |
| 6 | τ0-WM | L2+（§I–§VI-B 全文） | 无公共 sim benchmark（已获取范围原文未写） | 四任务（Toolbox/School Bag/Faucet/Badminton）caption verbatim；数值在 §VII | 数据 17.8k+6.5k+3.0k h；动作 16ch 绝对位姿（README） | §VII 主表（两源再试确认截断） |
| 7 | DreamZero | L2+（§1–§4.1 + 项目页真机数值） | PolaRiS/Genie Sim 3.0（§5 不可达；README 标 Coming Soon） | 项目页：62.2/27.4/39.5/49/81%、80 rollouts/ckpt×4 robots、150ms/7Hz | Table 1 效率 38× 逐字 | §5 全部表、附录 F/G rubric |
| 8 | XPACE | L2+（§1–§3.4.1 + §3.2 部署段 + 项目页数值） | 自建 19 子集 ID/OOD 视频预测（无公共基准） | IRON-R01 3 任务×20 trials：**68.3 vs 40.0/6.7**；DAgger 61.7→86.7（异套件） | §3.2 部署协议 verbatim（10Hz/16 步计划） | §4 正文（两源再试确认截断） |
| 9 | Rolling-WAM | L2+（§I–§IV-B + Table I/II 复验 + 项目页） | LIBERO 98.1；RoboTwin 93.5/93.0→93.3（100 rollouts/档） | G1 3 任务×20 trials：**85.0**；延迟 **215 vs 978/548/296/285ms**（上一轮代理核实+项目页） | Table I/II 逐字复验一致；§IV-B 实现细节全文 | Table III/IV/V（两源再试确认截断） |
| 10 | Kairos | L2（§1–§2.3） | 4 类基准（§6.1–6.4；LIBERO-Plus/RoboTwin 在 6.2）数值不可达 | **目录无真机节**（TOC 证据） | 架构/理论骨架 verbatim | §6 全部数值；89.0→90.8 归属 |

## 新增评测坑（仅基于原文，重发轮）

1. **同一基线数字不同舍入（跨论文）**：π0 的 LIBERO 四件套 96.8/98.8/95.8/85.2，Rolling-WAM Table I 记平均 **94.1**、WLA Table 1 记 **94.2**——(96.8+98.8+95.8+85.2)/4=94.15，两文舍入方向相反；跨论文对比平均分必须回溯组件分数。
2. **相对表述口径不一**：WLA "nearly doubling the best baseline"=Mem-0 28.5%→56.5%（≈1.98× 乘法）；MotionWAM "over 30%"=43.9%→76.1% 的**绝对** +32.2pp（加法）——同词不同意，引用前必须回查原文两种算法。
3. **正文数字与表格数字打架**：WLA 正文 "reduces the average success rate from 56.5% to **17.25%**" vs Table 2 的 **17.3%**——正文与表不能互证，须两处都录。
4. **"Success count" ≠ 成功率**：WLA 真机 Fig. 3(c) 是成功计数而非成功率，分母（trials）未在已取段披露；转述时不能把计数当率。
5. **主结果藏于条形图**：MotionWAM Fig. 4（9 任务逐值）、WLA Fig. 3(c)/(d) 等——图内数字无法逐字抄录，笔记必须标「图内数值，未抄录」而不是省略说明。
6. **零样本 vs 微调口径混排风险**：GE-Act 2.0 主表全部无 SFT 零样本（§2.5 明言 sim benchmark 只作横向参照），与 WSA/WLA/Rolling 的 2500+25k 微调协议不可同表比较。
7. **「未报告」本身是结论**：MotionWAM 全文无仿真评测、无 OOD 物体测试（§6 自认）；Kairos 目录无真机节；τ0-WM 已获取范围无公共 sim benchmark——写「原文未写（已获取范围）」而非「未取到」。
8. **官方页与论文版本漂移**：WSA abs "93.3%" vs README "WSA-L 93.1%（仅 randomized）"；DreamZero README 的 PolaRiS/Genie sim 支持标 "Coming Soon" 而论文声称已评测——引用官方页时须标其日期/版本。
9. **效率口径互不兼容**：Rolling-WAM 215ms=A100 稳态 replanning（排除 warm-up/初始化、含视觉编码）；MotionWAM 4.9Hz=A100 chunk 发射率；DreamZero 0.6s=GB200 带 DiT-cache（README 实测）；WLA 40ms=RTX 5090 efficient mode——跨论文延迟对比必须逐项拆 GPU/模式/warm-up。
10. **trials 口径差异 ≈2× CI 宽度**：20 trials/任务（Rolling G1、MotionWAM、XPACE）、30（WSA 真机）、50（LIBERO 惯例）、100（RoboTwin 惯例）、100 episodes/任务（WSA ARX-X5）、80 rollouts/ckpt×4 robots（DreamZero 项目页）——混排不同 trials 的分数时置信区间不可比。

<!-- QA_LOG -->
（占位：此处记录与用户的逐条讨论 Q&A，尚未开始）
<!-- /QA_LOG -->
