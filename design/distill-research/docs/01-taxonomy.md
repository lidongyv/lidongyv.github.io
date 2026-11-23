# 01 · 蒸馏分类框架：先分清"蒸什么"和"教师何时存在"

> 本文是整站的坐标系。后面每一篇论文都放到这两个轴上定位，避免把「数据蒸馏」「推理加速」「on-policy RL」混为一谈。
> 纪律：分类是**基于机制的归纳**，不是任何论文的官方分类；每个数字都标来源级别（L3 全文 / L2 正文部分 / L1 摘要或官方页）。

---

## 1. 一句话结论

VLA/WAM 的"蒸馏大模型"实际包含 **6 种被转移的对象** × **5 种教师存在方式**，共 30 个格子；过去一年真正有论文落地的只有 **14 个格子**，其中最近三个月的新增几乎全部集中在 **on-policy（学生自己跑、教师来打分）** 和 **步数蒸馏（多步 ODE → 1 步）** 两条线上。

---

## 2. 轴一：蒸什么（6 类转移对象）

| 代号 | 转移对象 | 典型信号 | 学生需要什么 | 代表工作 |
|---|---|---|---|---|
| **S1 输出** | 教师的动作/logits/速度场/端点 | KL、交叉熵、MSE、flow-matching loss | 同构或可对齐的输出头 | 经典 KD、SnapFlow、VLA-SSD |
| **S2 表示** | 中间隐层、attention、latent planning token、世界模型特征 | L1/L2 回归、cosine、CKA、KL on features | 有可对齐的层 + 投影器 | Orion-Lite、THAW-VLA、CKT-WAM、FitNets |
| **S3 行为** | 轨迹、动作 chunk、成功 episode、on-policy 状态-动作对 | 行为克隆 / DAgger / 加权回归 | 能复现教师行为的策略 | H2E、TAD、Beyond Policy Patches |
| **S4 数据** | CoT/QA 标注、生成视频、伪动作、潜动作 | 监督学习（数据即知识） | 只要数据，架构不限 | AutoVLA、GR00T N1、LAPA、DreamGen |
| **S5 计算结构** | 步数、层数、token 数、搜索预算 | 一致性/短路径/蒸馏损失 | 精简后的同一模型 | π0-FAST、SnapFlow、DMD 系、token 剪枝 |
| **S6 训练信号** | verifier / reward / critic / 特权信息 | 加权、重排、RL 目标 | 训练期可查询教师 | Verify-Act-Distill、VLP、RL 专家蒸馏 |

**关键区分**：S5 里的"自蒸馏"经常**没有独立教师**（教师=模型自己的多步输出或 EMA），这在工程上意味着不需要额外显存，但对齐目标更弱、更容易塌缩。

---

## 3. 轴二：教师何时存在（5 种）

| 代号 | 教师生命周期 | 学生看到的状态分布 | 成本 | 风险 |
|---|---|---|---|---|
| **T1 离线 teacher-forced** | 只在造数据/提特征时 | 教师轨迹的分布（off-policy） | 一次性前向 | 复合误差、exposure bias |
| **T2 on-policy** | 训练中持续可查 | 学生自己跑出来的状态 | 每步一次教师前向 | 教师查询成本高、需并行 |
| **T3 训练期教师** | 训练时在，推理时消失 | 混合 | 训练期显存 | 部署零开销，但容量上限受学生限制 |
| **T4 自蒸馏** | 教师=自身（多步/EMA/早停） | 自身分布 | 低（同一模型） | 目标可能退化、模式塌缩 |
| **T5 无教师（数据合成）** | 只有生成模型，无对齐损失 | 生成数据分布 | 生成成本 | 分布偏移、幻觉进入数据 |

> 判断标准（本站采用）：**有教师权重/特征/输出被学生显式拟合 → 蒸馏**；只有同一模型训练-推理路径不同、或调度优化 → **加速**，单列辨析（见 03 章 E 节）。

---

## 4. 交叉矩阵：哪些格子真的有工作

| | T1 离线 | T2 on-policy | T3 训练期 | T4 自蒸馏 | T5 数据合成 |
|---|---|---|---|---|---|
| **S1 输出** | 经典 KD、π0-FAST | FastOPD、VLA-MOPD、WAM-Policy | — | SnapFlow、VLA-SSD | — |
| **S2 表示** | Orion-Lite、THAW-VLA、CKT-WAM、DiMA | 少数（需要在线上提特征） | VLP、feature cache | S-VAM | — |
| **S3 行为** | TAD、H2E | DAgger 系、Reflect-VLA | — | — | — |
| **S4 数据** | AutoVLA、Senna、DriveGPT4 | — | — | — | GR00T N1、LAPA、DreamGen |
| **S5 结构** | π0-FAST、token 剪枝 | — | 训练共训、推理删分支 | DMD/consistency/shortcut | — |
| **S6 信号** | Verify-Act-Distill | RL 专家 → VLA（Beyond Policy Patches） | VLP | — | — |

**读法**：行是"蒸什么"，列是"教师何时在"。同一篇论文可能占多个格子（例如 SnapFlow 既是 S1 也是 S5，且是 T4）。

---

## 5. 速度账：蒸馏到底省在哪一段

VLA 推理延迟不是一块铁板，拆开看才知道蒸馏该打哪里：

| 阶段 | 典型占比（7B 级 VLA） | 蒸馏手段 | 可省 |
|---|---|---|---|
| 视觉编码 | 10–20% | 换小 backbone / token 剪枝 / 特征缓存 | 中 |
| LLM prefill | 30–50% | 隐层蒸馏换小 LLM（Orion-Lite 150× 仅指 reasoning 模块） | 大 |
| 自回归 decode | 10–30% | FAST tokenizer、并行解码、层跳过 | 中 |
| 动作头（flow/diffusion） | 20–50%（NFE=10 时） | 步数蒸馏 10→1（SnapFlow、DMD） | 大 |
| 测试时计算（Best-of-N/verifier） | 0–N× | 把 verifier 增益蒸进权重（Verify-Act-Distill） | 大 |

> 注意：论文里的"快 150×"通常只指**被替换的那个模块**，不是端到端。端到端要看 806ms→267ms 这种全链路数字（Orion-Lite）。

---

## 6. 三条最容易踩的误判

1. **把"小模型直接训练"叫蒸馏**：SmolVLA、TinyVLA 是从零/微调小模型，没有教师拟合（SmolVLA §2 原文把 TinyVLA 描述为 "trained a lightweight sub-1B model from scratch"）。
2. **把"训练时用未来信息、推理删掉"叫蒸馏**：Fast-WAM、Rift、GIGAWorld-Policy 是训练-推理解耦或 KV 预填，没有教师被拟合。
3. **把"用大模型造数据"当成 KD**：数据蒸馏里学生从没对齐过教师分布，教师只是标注器；这类方法的上限由**数据覆盖度**决定，不由教师容量决定。

---

更新时间: 2026-11-21
