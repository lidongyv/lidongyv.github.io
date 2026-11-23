# 02 · VLA 蒸馏：驾驶与操作两条线的实现与数字

> VLA = Vision-Language-Action。蒸馏它有两种完全不同的动机：
> **驾驶**——把 7B 级 LLM 的车端推理省掉（延迟/显存是硬约束）；
> **操作**——把 flow/diffusion 动作头的 NFE 压下来（控制频率是硬约束）。
> 两条线的论文几乎不互引，但机制高度重合。本页按"蒸什么"组织。
> **实现级细节（张量形状、完整伪代码、超参）统一放在第 12 章《实现手册》。**

---

## 1. 全景：VLA 蒸馏的 8 种做法

| # | 做法 | 代表 | 教师 | 学生 | 部署带走什么 |
|---|---|---|---|---|---|
| 1 | latent planning token 回归 | **Orion-Lite** (2604.08266) | 7B LLM | 0.1B decoder | 只有学生 |
| 2 | latent 推理特征 + 可摘 LM | **DRIFT** (2611.24719) | <150M LM | 255M 规划器 | 只有规划器 |
| 3 | 特权/未来观测蒸馏 | **FPD-Drive** (2610.13232)、**OPTED** (2609.20756) | 特权教师 | 纯视觉规划器 | 只有学生 |
| 4 | 多教师特征蒸馏 | **ConDenseDrive** (2610.16908) | 3 个 VFM | 单一视觉编码器 | 只有编码器 |
| 5 | 同体渐进 + token 级纠正 | **Reflect-VLA** (2610.13453) | 自身渐进教师 | 单次前向 VLA | 只有学生 |
| 6 | 动作头步数蒸馏 | **SnapFlow** (2604.05656)、**π0-FAST**、**FastOPD** (2610.02832) | 10-NFE / flow 教师 | 1-NFE | 只有学生 |
| 7 | 多专家策略合并（OPD） | **VLA-MOPD** (2611.12286)、**REPAIR** (2611.02777) | 多任务/RL 专家 | 单一 VLA | 只有学生 |
| 8 | 剪枝 + 离线隐层蒸馏 | **2609.19579** | OpenVLA-OFT 7B | 宽度剪枝 63–72% | 剪枝后的学生 |

---

## 2. 驾驶线：省掉推理期的 LLM

### 2.1 Orion-Lite（2604.08266，L3）——最硬的证据

| 项 | 教师 ORION | 学生 Orion-Lite |
|---|---|---|
| 结构 | EVA-02-L + Vicuna-v1.5 **7B** + VAE 规划器 | EVA-02-L（冻结）+ **0.1B** 6 层 decoder + planner |
| 输入 | 图像 + 文本 prompt | **纯视觉**（丢弃文本） |
| 蒸什么 | — | 教师 LLM 的 latent planning token `T_p` |
| 损失 | — | `L = L_GT + L_mimic`，`L_mimic = ‖T_s − T_p‖₁` |
| Bench2Drive 闭环 DS | **77.7** | **80.6** |
| 能力分 | 54.7 | 60.5 |
| 推理模块延迟 | 基线 | **150×** |
| 端到端延迟 | — | **806ms → 267ms** |
| 显存 | **31GB** | **8GB** |

**要点**：
- 冻结视觉编码器 + QT-Former，只训 0.1B decoder；
- 用 L1 而不是 KL（作者明确对比 DiMA 的 KL 路线）；
- 闭环反超，作者归因于"LLM 在直接模式下是过参数化的特征提取器"。

### 2.2 DRIFT（2611.24719，L1）——把 LM 做成"可摘除的推理支架"

- 轻量 LM（<150M）→ 规划器（全模型 **255M**）；蒸 **surrogate reasoning features**（latent 推理特征），推理期可把 LM 摘掉。
- 数字：Bench2Drive **DS 97.03 / SR 89.39%**（三次均值）。
- 解读：比 Orion-Lite 更彻底——连 0.1B decoder 都不保留，直接把推理压进 255M 规划器。**但 97.03 与 Orion-Lite 的 80.6 不可比**（前者大概率是不同评测设置/更大数据），只说明"小模型可以做到很高"。

### 2.3 FPD-Drive（2610.13232，L1）——特权教师

- 教师：可见未来观测的特权教师；学生：纯视觉规划器；蒸未来多目标轨迹监督。
- 数字：NAVSIM v1 **94.9 PDMS**；v2 navhard **55.9 EPDMS**；Pro 95.5/58.1；HUGSIM HD-Score 45.1、RC 55.3。
- 定位：典型的 Learning-by-Cheating 变体，部署不带教师。

### 2.4 OPTED（2609.20756，L1）——闭环 on-policy 经验蒸馏

- 教师：向量化输入（HD 图 + 框）的 RL 特权教师；学生：相机（TransFuser/VaVAM）。
- 做法：AlpaSim + 3DGS 在线采样，闭环蒸馏。
- 数字：DS 提升 **1.6× 与 9.5×**；同等闭环性能少约 **3 个数量级**仿真交互。
- 价值：证明"闭环经验"比"离线轨迹"值钱得多——3 个数量级是这条线最夸张的样本效率数字。

### 2.5 Reflect-VLA（2610.13453，L1）——同体自蒸馏

- 教师：同体渐进教师（Reflective Guidance 作为训练期特权）→ 单次前向 VLA。
- 做法：**token 级纠正分布蒸馏**；风险轨迹蒸教师、可靠轨迹自模仿（双路径）。
- 数字：训练时间较 RL 减少 **90%+**；NAVSIM v1/v2、Bench2Drive（摘要未给分值）。
- 价值：把"自我反思/纠错"变成训练期信号，推理期不需要反思循环。

### 2.6 ConDenseDrive（2610.16908，L1）——多教师特征蒸馏

- 教师：3 个视觉基础模型（视觉/语义/几何）→ 单一 E2E 视觉编码器；配 register token 压缩。
- 数字：**2.0×** 加速；NAVSIM v1 **94.5 PDMS**、v2 **90.8 EPDMS**；HUGSIM 37.4 HD-Score。

### 2.7 数据蒸馏（驾驶，见 07 章）

AutoVLA（Qwen2.5-VL-72B → 3B，45.6k nuPlan + 7.2k Waymo CoT）、Senna（GPT-4o）、DriveGPT4（ChatGPT）、OmniDrive（GPT-4）、LMDrive（指令改写）。

---

## 3. 操作线：压动作头的 NFE

### 3.1 π0-FAST（PI 官方，无 arXiv，L2）

- 把 π0 的 flow matching 动作专家蒸馏成 **FAST tokenizer 的自回归专家**，推理约 **5×**（官方博客 + openpi 代码）。
- 注意：没有同行评议论文；引用时标 ❗。

### 3.2 SnapFlow（2604.05656，L3）

- 教师=π0.5（10-NFE），学生=同网络 1-NFE；短切目标 + zero-init destination-time projection。
- 数字：LIBERO **98.75% vs 教师 97.75%**；π0 单步版超教师 **+10.5pp**；延迟 **−3.3×**。
- 详见 04 章。

### 3.3 FastOPD（2610.02832，L1）

- 教师：π0.5 / LingBot-VLA / MolmoAct2；学生：轻量 VLA。
- 蒸什么：教师速度场（flow map 单状态监督 + self-consistency），**on-policy** 在学生状态上查教师。
- 数字：LIBERO 2 步保留 **84%**，延迟 **−78.1%**；RoboTwin2.0 单步 **+15.9pp**。
- 详见 06 章。

### 3.4 VLA-MOPD（2611.12286，L1）

- 多任务 flow-VLA 专家 → 单一共享 flow VLA；在**学生观测 + 每个 ODE 中间状态**做 masked local matching。
- 数字：LIBERO mean success-once **75.4%→98.9%**；RoboTwin 2.0 **73.0%→89.3%**；Piper 真机两轮 **70.0%→90.0%**。
- 价值：多专家合并第一次做到 ODE 轨迹级对齐。

### 3.5 REPAIR（2611.02777，L1）

- 轻量 RL 专家 + critic → 单一 flow VLA；critic 加权多专家动作候选成监督分布。
- 数字：六个真机任务各 30 trials **全部 100%**；末端 jerk 均低于专家辅助执行。

### 3.6 TAD（2610.33650，L1）——黑盒教师的上限

- 教师只给轨迹库（joint-position 特征），不给动作分数、不给在线查询。
- 判别器比较教师/学生 rollout 特征边缘分布 → reward → on-policy 学习。
- 数字：**26.2%→76.6%**（+50.4 点）；但白盒 OPD 是 **81.6%**，且 TAD 花 **5× 交互**、**12.5× batch-passes**。

### 3.7 TILDE（2610.25376，L1）

- 预训练 EFM → 未做动作预训练的策略；on-policy 教师监督 + 教师恢复轨迹；AutoDAgger 用 reward model 触发干预。
- 数字：三个真机任务最高 **4.1×** 于 behavior cloning；恢复轨迹采集比 human-in-the-loop 快 **>2×**。

### 3.8 剪枝 + 离线隐层蒸馏（2609.19579，L1）——最实用的基线

- 教师：OpenVLA-OFT **7B** → 学生：**宽度剪枝 63%** 的同构模型（CogACT 也测）。
- 做法：**离线缓存教师 hidden state**，直接对齐同形状残差流；无 projector、无 online rollout。
- 数字：
  - 63% 剪枝使 LIBERO-Long **93.2%→0.8%**；蒸馏后回到教师 **3.5 点内**，约 **8 GPU-hours**；
  - 72% 压缩学生真机 **77.5%**（vs 监督恢复 59.5%）；
  - 板载 **2.23×** 快、显存少 **62%**。
- 价值：**如果你只想低成本恢复一个被剪坏的 VLA，这是最划算的做法**——纯离线、8 GPU 小时。

### 3.9 SPARK-VLA（2611.10443，NeurIPS 2026，L1）

- 冻结 3D 专家 → LoRA 视觉分支；0.5B LLM backbone，约 **53M** 可训练参数；对齐注意力响应、router 用 action-loss 梯度监督选 token。
- 定位：蒸"空间表征"而非动作。

---

## 4. 横向总表（VLA）

| 工作 | 场景 | 教师 | 学生 | 蒸什么 | 教师效果 | 学生效果 | 速度/成本 |
|---|---|---|---|---|---|---|---|
| Orion-Lite | 驾驶 | 7B LLM | 0.1B decoder | latent token | DS 77.7 | **DS 80.6** | 150× 模块 / 806→267ms / 31→8GB |
| DRIFT | 驾驶 | <150M LM | 255M 规划器 | latent 推理特征 | — | DS 97.03 / SR 89.39 | 255M |
| FPD-Drive | 驾驶 | 特权教师 | 视觉规划器 | 未来轨迹 | — | 94.9 PDMS (v1) | — |
| OPTED | 驾驶 | 向量化 RL 教师 | 相机学生 | 闭环经验 | — | DS 1.6–9.5× | 交互 −3 数量级 |
| ConDenseDrive | 驾驶 | 3 VFM | 视觉编码器 | 特征 | — | 94.5 PDMS | 2.0× |
| Reflect-VLA | 驾驶 | 同体渐进 | 单次 VLA | token 级纠正 | — | 未给分值 | 训练 −90% |
| SnapFlow | 操作 | π0.5 10-NFE | 1-NFE | 端点 | 97.75% | **98.75%** | −3.3× |
| FastOPD | 操作 | π0.5 等 | 轻量 | 速度场 | — | 2 步保留 84% | −78.1% |
| VLA-MOPD | 操作 | 多专家 | 单一 VLA | ODE 轨迹 | — | 75.4→98.9% | — |
| REPAIR | 操作 | RL 专家 | flow VLA | critic 加权动作 | — | 真机 100% | — |
| TAD | 操作 | 轨迹库 | OpenVLA-OFT 7B | 判别器 reward | — | 26.2→76.6% | 5× 交互 |
| TILDE | 操作 | EFM | 策略 | 恢复轨迹 | — | 4.1× BC | 采集 −2× |
| 2609.19579 | 操作 | 7B | 剪枝 63–72% | 离线 hidden | 93.2% | 教师 −3.5 点内（真机 77.5%） | 8 GPU-h；2.23× |
| SPARK-VLA | 操作 | 3D 专家 | 0.5B + LoRA | 空间表征 | — | 未给 | 53M 可训 |

> 表内数字来自不同 benchmark，**不可直接横比**；作用是看"量级"和"代价结构"。所有 L1 条目仅摘要级。

---

## 5. 三条实现范式（可直接抄）

### 范式 A：latent token 蒸馏（驾驶）

```python
# 教师：一次性前向，取 planning token
with torch.no_grad():
    T_t = teacher(images, text).planning_token      # [B, 1, C]

# 学生：纯视觉
T_s = student_decoder(student_vision(images))       # [B, 1, C]

loss = L_traj_gt + (T_s - T_t).abs().mean()         # L1，λ=1
# 冻结视觉塔；只训 decoder + planner
```

### 范式 B：动作头步数蒸馏（操作）

```python
# 教师：10 步 Euler 到端点（或直接用教师多步输出）
with torch.no_grad():
    a_t = teacher_sample(obs, steps=10)

# 学生：1 步
a_s = student_sample(obs, steps=1)

loss = flow_matching_loss(student, a_t) + λ * shortcut_loss   # SnapFlow 式
```

### 范式 C：剪枝 + 离线隐层（最便宜）

```python
# 离线：教师对所有训练帧提 hidden state，存 memmap
teacher_hidden = cache[(traj_id, frame_idx)]         # [D]

# 训练：学生（剪枝后的同构模型）对齐同一层
h_s = student(images, return_hidden=True)
loss = L_task + λ * (h_s - teacher_hidden).pow(2).mean()
```

---

## 6. 选型

| 约束 | 选 |
|---|---|
| 驾驶、要省推理 LLM | Orion-Lite（0.1B）或 DRIFT（255M） |
| 驾驶、有特权教师/仿真 | OPTED（闭环经验，样本效率最高） |
| 操作、flow 动作头太慢 | SnapFlow（无教师、一次训练）或 FastOPD（可到 2 步） |
| 多个任务专家要合并 | VLA-MOPD（ODE 轨迹级） |
| 真机、有 RL 专家 | REPAIR（critic 加权） |
| 没预算、只想恢复剪坏的模型 | 2609.19579 的离线隐层（8 GPU-h） |

---

更新时间: 2026-11-21
