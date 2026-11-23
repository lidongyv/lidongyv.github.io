# 基本概念精读：System 0 / 1 / 2（/3）

> 目的：把「System 0/1/2(/3)」这个常被混用的概念族拆清楚，标出每个概念的一手出处与原文。
> 铁律：引文为原文 verbatim（带小节/页面定位）；未取到的一律标「未能获取/未核实」，不脑补。
> 更新：2026-11-19

---

## 0. 一句话结论

- **S1/S2 的概念源头是心理学双过程理论（Kahneman 2011）**；具身领域**没有**给 S1/S2 新定义，工程界（GR00T N1、Figure Helix、Gemini Robotics）的「双系统」只是把 S1/S2 落到「慢 VLM + 快动作模块」两套网络上。
- **S0 与 S3 是 Taniguchi et al. (2025) 在 2503.06138 中新增的**：S0 = 前认知的具身过程（形态计算/被动动力学，超快），S3 = 集体智能与符号涌现（超慢）。**S0/S3 在 GR00T N1、Helix 等工程文章里均不出现**。
- **Figure 的 Helix 文章只讲 S1/S2**（原文见 §2.3），没有 S0/S3。
- 「Physical AI + 四概念」确有结合的讲法：Taniguchi 的 IEEE SII 2026 全会报告标题为 "…A System 0/1/2/3 Perspective on Hierarchical Physical AI"（标题级证据；内容未能获取）。中文行业另有「Physical AI 三层脑」类文章（标题级，内容未能获取）。

---

## 1. 概念谱系（谁定义了什么）

| 概念 | 一手出处 | 原文定位 | 定义要点 | 来源级别 |
|---|---|---|---|---|
| **S1 / S2**（双过程理论） | Daniel Kahneman, *Thinking, Fast and Slow*, 2011 | 全书（概念源头） | S1：快、直觉、自动；S2：慢、审慎、分析 | 概念来源（未逐字引书；2503.06138 §3.1 称沿用 "the conventional definitions"） |
| **S0 / S1 / S2 / S3**（四过程框架） | Taniguchi et al., *System 0/1/2/3: Quad-process theory…*, [arXiv:2503.06138](https://arxiv.org/abs/2503.06138)（Artificial Life 31(4):465） | Abstract、§1、§3.1–3.2 | 在 S1/S2 之下加 **S0=前认知具身过程**；之上加 **S3=集体智能/符号涌现** | L3（全文核实） |
| **工程「双系统」S2/S1** | NVIDIA **GR00T N1**, [arXiv:2503.14734](https://arxiv.org/abs/2503.14734)（2025-03-18） | Abstract | S2=视觉语言模块（慢推理）；S1=扩散 Transformer 动作模块（实时动作）；紧耦合、端到端联合训练 | L3（摘要 verbatim） |
| **工程「双系统」S2/S1** | Figure **Helix 博客**（2025-02-20，[figure.ai/news/helix](https://www.figure.ai/news/helix)） | "Helix: A 'System 1, System 2' VLA…" 一节 | S2=7–9 Hz 板载 VLM；S1=200 Hz 反应式 visuomotor 策略；S2 产 latent 向量条件 S1 | L1（厂商博客，全文核实） |
| **Physical AI × 四概念** | Taniguchi, *Collective Predictive Coding and World Models in LLMs: A System 0/1/2/3 Perspective on Hierarchical Physical AI*（IEEE SII 2026 Plenary Talk） | 标题级（speakerdeck 直连失败） | 标题明确把四过程框架用于「分层 Physical AI」 | ⚠️ 内容未能获取 |
| 中文行业「三层脑」 | 《Physical AI 的三层脑：慢脑想，快脑做，脊髓兜底》（微信公众号） | 标题级（公众号触发验证） | 「慢脑/快脑/脊髓」疑与 S2/S1/S0 一一对应，但**未获一手正文** | ⚠️ 未核实，仅提示 |

---

## 2. 一手引文（精读核心）

### 2.1 Taniguchi 2503.06138 —— S0/S1/S2/S3 的学术定义

**Abstract**：
> "Expanding upon **System 1 (fast, intuitive thinking)** and **System 2 (slow, deliberative thinking)**, we incorporate **System 0, which represents pre-cognitive embodied processes**, and System 3, which encompasses collective intelligence and symbol emergence."

**§1 Introduction**：
> "physical interactions that do not require neural processing exhibit super-fast dynamics. Recent AI studies have often overlooked this layer, **which we refer to as System 0 in this paper**."

**§3.1（S0 的正式定义）**：
> "**System 0 represents the foundational layer of cognition** in our proposed framework, encompassing the **pre-cognitive, embodied processes that occur at the fastest (i.e., super-fast) timescale**. … This system is characterized by **direct physical interactions between an agent's body and its environment and operates without explicit computation**. System 0 includes phenomena such as **morphological computation and passive dynamics**, exemplified by the self-stabilizing properties of bipedal walking or the inherent grasping capabilities of a soft robotic hand."

**§3.1（S1/S2 沿用常规定义）**：
> "For Systems 1 and 2, we **follow the conventional definitions** [69]."

**§3.1（S3 定义）**：
> "**System 3 represents the highest level** … encompassing **collective intelligence and symbol emergence** … This system operates on the longest timescale, facilitating the development and evolution of shared symbolic systems, such as language, cultural norms, and societal institutions."

**§2.1.1（S1/S2 在 AI 里的对应）**：
> "**Symbolic AI is considered to correspond to System 2 and machine learning to System 1.**"

**精读要点**：
1. 时间尺度是 S0/1/2/3 的划分主轴：S0 超快（物理交互，无需神经计算）→ S1 快（直觉/自动）→ S2 慢（审慎推理）→ S3 超慢（社会/符号演化）。
2. S0 不是「世界模型」，而是**身体本身的物理计算**（形态计算、被动动力学）。论文用「无脑也能下坡行走的被动动态步行者」「软体手的固有抓取能力」作例证 —— 这是最容易与行业话语里的「世界模型层」混淆的地方。
3. S1/S2 无新定义：论文明确沿用常规定义，并把符号 AI 对应 S2、机器学习对应 S1。
4. 理论底座：Bergson 多时间尺度哲学 + Friston 自由能原理 + 集体预测编码（CPC，S3）。

### 2.2 GR00T N1 2503.14734 —— 工程 dual-system 的 S1/S2

**Abstract（verbatim）**：
> "GR00T N1 is a Vision-Language-Action (VLA) model with a **dual-system architecture**. The **vision-language module (System 2)** interprets the environment through vision and language instructions. The subsequent **diffusion transformer module (System 1)** generates fluid motor actions in real time. Both modules are tightly coupled and jointly trained end-to-end."

**精读要点**：S2=视觉语言模块（语义理解/指令），S1=扩散 Transformer 动作头（实时动作）；两者「紧耦合 + 端到端联合训练」——这与 Helix 的设计取舍（见下）不同，是「双系统」内部实现差异的关键点。

### 2.3 Figure Helix 博客 —— 只讲 S1/S2，未提 S0/S3

**原文（"A 'System 1, System 2' VLA" 一节）**：
> "Helix is a first-of-its-kind 'System 1, System 2' VLA model… **System 2 (S2): An onboard internet-pretrained VLM operating at 7-9 Hz** for scene understanding and language comprehension… **System 1 (S1): A fast reactive visuomotor policy** that translates the latent semantic representations produced by S2 into precise continuous robot actions at **200 Hz**."

> "S2 distills all semantic task-relevant information into **a single continuous latent vector**, passed to S1 to condition its low-level actions."

> "Gradients are backpropagated from S1 into S2 via the latent communication vector… allowing joint optimization of both components."

**精读要点**：
1. Helix 的 S1/S2 完全对应工程双系统：慢 VLM（7–9Hz）+ 快 visuomotor（200Hz），接口=单连续 latent 向量。
2. 训练细节值得学：给 S1/S2 输入加**时间偏移**（匹配部署时的推理延迟差），部署时 S2 异步跑、S1 保持 200Hz 实时环——「训练镜像部署」的工程范式。
3. **全文无 System 0 / System 3** —— 用户问题的直接答案：Figure 的文章不讲解这四个概念，只讲解 S1/S2 两个。

### 2.4 「Physical AI」侧的结合

- Taniguchi IEEE SII 2026 全会报告（标题）："Collective Predictive Coding and World Models in LLMs: **A System 0/1/2/3 Perspective on Hierarchical Physical AI**" —— 说明「Physical AI」与四概念确有结合讲法，但内容未能获取（speakerdeck 直连失败）。
- 中文《Physical AI 的三层脑：慢脑想，快脑做，脊髓兜底》——标题暗示「慢脑/快脑/脊髓」三层映射（≈S2/S1/S0），但公众号触发验证，**内容未核实，不作为依据**。

---

## 3. 三个常见混淆（引用前必读）

1. **「双系统」≠ 四过程框架**：工程文章（GR00T N1/Helix/Gemini）只取 S1/S2 并重定义为「两套网络」；S0/S3 只在 2503.06138 里存在。两者不是同一理论。
2. **S0 ≠ 世界模型**：2503.06138 的 S0 是形态计算/被动动力学（无计算）；行业话语里的「世界模型层」属于 S2 或独立世界模型，不要混。
3. **S1/S2 在工程里还分三种解耦**（频率/语义层级/梯度，见[具身星图专题](https://zhuyun97.github.io/embodied-ai-learning/vla/papers/dual-system-architecture)，二手整理）：GR00T N1/Helix 是频率解耦；π0.5 是语义层级（语言子任务）；π0.6 的 Knowledge Insulation 是梯度解耦——「双系统」标签被滥用。

---

## 4. 基本概念论文/文章清单（精读顺序建议）

1. **Kahneman 2011**, *Thinking, Fast and Slow* —— S1/S2 概念源头（读第 1 章即可）。
2. **[Taniguchi et al. 2025, arXiv:2503.06138](https://arxiv.org/abs/2503.06138)** —— 四过程框架唯一一手出处；精读 §1、§2.2（形态计算）、§3.1–3.2（四层定义与 CPC）、§4（Bergson 时间尺度）。
3. **[GR00T N1, arXiv:2503.14734](https://arxiv.org/abs/2503.14734)** —— 工程 dual-system 定义（Abstract + §2 架构）。
4. **[Figure Helix 博客](https://www.figure.ai/news/helix)**（2025-02-20）—— S1/S2 频率、latent 接口、时间偏移训练、异步部署。
5. **π0.5, [arXiv:2504.16054](https://arxiv.org/abs/2504.16054)** —— 语义层级解耦实例（自然语言子任务为中介）。
6. π0.6（Knowledge Insulation，梯度解耦）—— 论文 ID 待核实，暂列。
7. **Gemini Robotics**（云-端双系统）—— 论文 ID 待核实（此前仅从二手页获得频率数字），暂列。
8. 二手汇总：[具身星图·双系统/分层架构专题](https://zhuyun97.github.io/embodied-ai-learning/vla/papers/dual-system-architecture)（三种解耦辨析 + 跨模型频率表，数字均为厂商自评）。

---

## 5. 未能获取清单（诚实记录）

1. Taniguchi IEEE SII 2026 全会报告正文（speakerdeck 直连失败两次）。
2. 微信公众号《Physical AI 的三层脑》正文（触发环境验证）。
3. π0.6 / Gemini Robotics 的 arXiv ID（未核实，暂不引用具体数字）。
4. Kahneman 原书逐字引文（未取原文；以 2503.06138 的「conventional definitions」引用为准）。

<!-- QA_LOG -->
（待与用户逐条讨论后追加）
<!-- /QA_LOG -->
