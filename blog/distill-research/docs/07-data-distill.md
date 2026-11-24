# 07 · 数据蒸馏：用大模型造监督信号（教师只当标注器）

> 定义：学生**从不拟合教师的分布**，只吃教师生成的标注/数据。教师是"标注器"，不是"被模仿的对象"。
> 这一类的上限由**数据覆盖度**决定，不由教师容量决定——这是它与 KD 最本质的区别。
> **生成→过滤→混合的实现代码见本章 §7。**

---

## 1. 什么时候该用数据蒸馏

| 场景 | 为什么 |
|---|---|
| 遥操数据没有语言/CoT 标注 | 用大 VLM 生成指令、理由、场景描述 |
| 没有动作标签，只有视频 | 用潜动作模型（LAPA/VQ-VAE）造伪动作 |
| 长尾任务缺数据 | 用视频生成模型造轨迹（DreamGen/Veo-Act） |
| 动作空间/机器人形态不一致 | 用 IDM 把"别人的动作"映射到自己的 embodiment |

**判断标准**：如果你把教师换成一个规则系统/人工标注员，方法是否照样成立？成立 → 数据蒸馏。

---

## 2. CoT / 标注蒸馏（驾驶为主）

| 工作 | 教师 → 学生 | 生成什么 | 原文依据 |
|---|---|---|---|
| **AutoVLA** (2506.13757) | Qwen2.5-VL-**72B** → 3B | 45.6k(nuPlan) + 7.2k(Waymo) CoT 驾驶推理 | "This pipeline enables automatic generation of high-accuracy reasoning annotations and supports knowledge distillation from a large capable model to a more compact target model"（§3.2）；GT 动作作 hint 抑制幻觉 |
| **DriveGPT4** (2310.01412) | ChatGPT（+YOLOv8 特权信息） | 40K 驾驶 QA（动作/理由/控制） | "ChatGPT is leveraged as a teacher to generate more conversations about the ego vehicle"（§3） |
| **Senna** (2410.22313) | GPT-4o | planning-oriented QA（红绿灯/VRU/意图/场景） | §III-A：QA 监督 VLM，meta-action 由 GT 轨迹规则化生成 |
| **OmniDrive** (2405.01533) | GPT-4（+规则 checklist） | 反事实 3D 驾驶 QA | §2：模拟/专家轨迹入 prompt |
| **LMDrive** (2312.07488) | ChatGPT | 指令改写变体（8 种） | "we utilized ChatGPT API to generate eight different variants"（§3） |
| **SmolVLA** (2506.01844) | Qwen2.5-VL-3B-Instruct | 社区数据集任务描述重写 | "we used an off-the-shelf VLM (Qwen2.5-VL-3B-Instruct) to auto-generate concise task descriptions"（§3.2）——**只是标注清洗，不算蒸馏训练** |

**实现模板**（以 AutoVLA 为例）：

```python
# 1) 用教师模型生成 CoT，输入 = 前视图 + 自车状态 + GT 动作 hint
prompt = build_prompt(front_img, ego_state, gt_action_hint)
cot = teacher.generate(prompt, temperature=0.7, max_tokens=512)

# 2) 过滤：动作合法性 + 格式 + 与 GT 轨迹的一致性
if not check_action_consistency(cot, gt_traj):
    cot = teacher.regenerate(prompt, temperature=0.3)

# 3) 学生 SFT：输入不含 hint，只含图像+状态+指令
loss = CE(student(front_img, ego_state, instruction), cot)
```

**关键 trick**：
- **GT 动作作 hint**：不给 hint 教师会编造动作；给 hint 后 CoT 与动作一致（AutoVLA §3.2）。
- **温度调度**：先高温采样多样性，再低温重生成保证质量。
- **一致性过滤**：CoT 里描述的动作必须和 GT 轨迹对得上，否则丢。

---

## 3. 视频 → 伪动作（无动作标签时）

### LAPA（2410.11758）

- **做法**：VQ-VAE 学帧间**离散潜动作**（8⁴ 码本），潜 VLA 预测潜动作，小数据微调映射到真实动作。
- **价值**：把"无动作视频"变成可训练标签；256⁷ ΔEE 微调（渲染还原）。
- **坑**：潜动作没有物理含义，跨 embodiment 迁移需要重新学映射。

### GR00T N1（2503.14734）

- **神经轨迹**：在 88h 遥操作上微调图生视频模型 → **827h** 生成视频（≈10×）；与真机动作**1:1** 混合做 flow-matching 目标。
- **潜动作**：VQ-VAE 帧间潜动作，encoder 当 IDM；IDM 伪动作混进训练。
- **规模**：预训练 5 万 H100 小时；DexMimicGen 仿真 78 万条 ≈ 6500h。
- **启示**：数据蒸馏在工业级落地时，**比例**（真机:合成 = 1:1）比方法本身更关键。

### DreamGen（GR00T N1.5 官方页，L1）

- 生成机器人轨迹做新任务数据；12 个新动词：N1.5 **38.3%** vs N1 13.1%。

### Veo-Act / NovaFlow / RIGVid（均 L1，清单级）

| 工作 | 生成什么 | 怎么变成动作 |
|---|---|---|
| Veo-Act (2604.04502) | Veo-3 生成未来图像序列 | IDM 在 random play 上训练，生成计划→IDM→VLA 执行 |
| NovaFlow (2510.08568) | 视频 → 3D object flow | 流 → IK/规划/抓取 |
| RIGVid (2507.00990) | 现成视频扩散生成演示 | VLM 过滤 + 6D 跟踪 → 机器人轨迹 |

---

## 4. 经验蒸馏：教师"陪着跑"，只留成功片段

这一类介于"数据蒸馏"与"on-policy"之间，单独拆出来看更清楚（详见 06 章）：

- **H2E**（2611.16246）：LLM planner + 冻结 VLA 组队跑任务，只保留到达目标的 session，再单独训 VLA；测试期零 LLM token，快 3.7×，成功率 +15.6%。
- **Verify-Act-Distill**（2610.14885）：verifier 选候选，成功 episode 的观测+聚合动作蒸进 LoRA。
- **Beyond Policy Patches**（2611.02777）：把 RL 专家蒸进 VLA。

---

## 5. 数据蒸馏的四个工程要点

1. **教师看到的输入必须和部署时一致**（除了故意的 hint）。否则学生学到"只有教师能用的线索"。
2. **合成数据的比例要调**：GR00T N1 用 1:1；比例过高会"学飞"（生成视频的物理不真实被学生当真）。
3. **过滤比生成重要**：一致性检查、物理可行性检查、格式检查，三道过滤后能用的可能只有 40–60%。
4. **别用合成数据做评测**：教师生成的分布和真实评测分布不同，会高估。

---

## 6. 与 KD 的边界（一句话）

> 数据蒸馏改的是**训练集**，KD 改的是**损失函数**。前者换教师不用改学生代码，后者换教师要对齐接口。

## 7. 实现代码（逐方法）

> 数据蒸馏的代码都长一个样：**生成 → 过滤 → 混入训练集**。区别只在"生成什么"和"怎么过滤"。

### 7.1 AutoVLA（2506.13757）——CoT 标注

**物理意义（解释）**：大模型写「驾驶理由」，GT 动作当 hint 抑幻觉；学生学 CoT + 动作。

```python
# 输入: 前视图, ego_state, gt_traj；输出: (cot, action) 样本
prompt = build_prompt(front_img, ego_state, gt_action_hint)   # GT 作 hint 抑幻觉
cot = teacher.generate(prompt, temperature=0.7, max_tokens=512)
if not check_action_consistency(cot, gt_traj):                # 一致性过滤
    cot = teacher.regenerate(prompt, temperature=0.3)
student_sft(student(front_img, ego_state, instruction), cot)
```

### 7.2 GR00T N1（2503.14734）——神经轨迹 + 潜动作

**物理意义（解释）**：视频生成模型造 827h 神经轨迹 + VQ-VAE 潜动作；与真机 1:1 混合。

```python
# 1) 在 88h 遥操上微调图生视频模型，生成 827h 视频
video_model = finetune_video_gen(teleop_88h)
fake_videos = video_model.generate(prompts, n=...)              # ≈10× 数据

# 2) VQ-VAE 帧间潜动作（encoder 当 IDM）
latent_action = vqvael.encode(frame_t, frame_t1)

# 3) 与真机动作 1:1 混合，做 flow-matching
batch = mix(real_batch, fake_batch, ratio=1.0)
flow_matching(student, batch.obs, batch.latent_action)
```

### 7.3 LAPA（2410.11758）——无动作视频 → 潜动作

**物理意义（解释）**：无动作视频学离散潜动作，再小数据映射到真实动作；把「看视频」变成「有标签」。

```python
# 输入: 无动作视频帧对；输出: 离散潜动作
z = vqvael.encode(frame_t, frame_t1).argmax(-1)                 # [B, 8^4]
latent_vla.train(video_frames, z)                                # 预训练
# 小数据微调：潜动作 → 真实 ΔEE
map_head.fit(latent_vla.features, delta_ee)                      # 256^7
```

### 7.4 DreamGen / Veo-Act / RIGVid（L1）

**物理意义（解释）**：生成演示 → IDM/IK/跟踪 → 机器人轨迹；用生成模型补长尾任务。

```python
# 生成演示 → IDM/IK/跟踪 → 机器人轨迹
video = veo3.generate(instruction, first_frame=obs0)
actions = idm(video)                       # 逆动力学模型
# 或 NovaFlow: video → 3D object flow → IK
# 或 RIGVid: video → VLM 过滤 → 6D 跟踪 → 轨迹
dataset.add(actions); train_bc(student, dataset)
```

### 7.5 过滤与混合（工程要点）

**物理意义（解释）**：三道过滤（格式/物理/动作一致性）+ 真机:合成 1:1；数据蒸馏的上限由覆盖度决定。

```python
def keep(sample):
    return (check_format(sample) and
            check_physics(sample) and
            check_action_consistency(sample, sample.gt))
dataset = [s for s in generated if keep(s)]        # 通常只留 40–60%
loader  = mix(real_data, dataset, ratio=1.0)       # GR00T 经验值
```

---

更新时间: 2026-11-21
