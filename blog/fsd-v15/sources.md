# Sources / 来源清单

**As of:** 2026-11-23  
**Article version:** v1.2  
**Rule:** every number in the article should be traceable to one of the rows below, or explicitly labeled `INFERRED` / `HYPOTHESIS`.

## Evidence levels

| Label | Meaning |
|---|---|
| `PUBLIC` | Tesla release notes, public Musk/Ashok statements, CVPR/ICCV talks, official safety data. |
| `REPORTED` | Credible media or technical posts, not independently verified by this note. |
| `INFERRED` | Derived from parameters, latency, compute, or architecture constraints; may be overturned. |
| `HYPOTHESIS` | Industry discussion or this note's speculation; not to be cited as fact. |

## Main sources

| # | Source | Date | What it supports | Level |
|---|---|---|---|---|
| 1 | Sohu / 环球网, “特斯拉开始推送FSD v15：重点修复标志识别，新增 Reality View” — https://www.sohu.com/a/1104750786_99900743 | 2026-11-22 | V15 rollout, 25× critical intervention rate, Reality View, HW4, feature list | `REPORTED` |
| 2 | ZOL, “特斯拉FSD V15发布Reality View生成式实景导航” — https://ai.zol.com.cn/1276/12762706.html | 2026-11-23 | Reality View is generative, no game engine, same end-to-end model, half a TRIP accelerator, lower compute than old visualization | `REPORTED` |
| 3 | OFweek / Vehicle, “特斯拉 FSD V15 拆解：视觉编码器是什么，Reality View 的画面从哪来” — https://www.ofweek.com/auto/2026-11/ART-70109-8140-30712906.html | 2026-11-23 | UVE, internet-video pretraining, 10B media claim, NVS, 3DGS, 220 ms reconstruction, half-TRIP math | `REPORTED` / `PUBLIC` mix |
| 4 | OFweek / Vehicle, “特斯拉 CVPR 2026 演讲全文和详解：把自动驾驶，做成「所有机器人的基础模型」” — https://www.ofweek.com/auto/2026-06/ART-70109-8110-30689565.html | 2026-06-05 | 2B input tokens → 2 output tokens, intermediate outputs (occupancy, 3DGS, language), World Sim NN ↔ Policy NN closed loop, shared encoder across robots | `PUBLIC` talk reconstruction |
| 5 | 人人都是产品经理 / 杰克说AI, “从特斯拉ICCV分享看未来：端到端自动驾驶的下半场将如何演绎？” — https://www.woshipm.com/ai/6309531.html | 2025-12-22 | ICCV 2025 talk, 20B/2B token framing, 3DGS, language reasoning, evaluation | `REPORTED` |
| 6 | Weibo technical analysis, “特斯拉FSD V15技术细节分析 工程思路受关注” — https://weibo.com/2/detail/5368224994558491 | 2026-11-22 | UVE 18 Hz, 3B action + 1B render fork, conditional mean-flow, 8-frame chunk / 4 UVE cycles, ~36 fps target (latency not used) | `INFERRED` / community |
| 7 | Weibo, “马斯克回复提及特斯拉FSD V15模型参数相关信息” — https://weibo.com/2/detail/5368185698390766 | 2026-11-22 | Musk reply / screenshot showing 3B + 1B; 4B question | `REPORTED` (screenshot) |
| 8 | TechWeb, “重构自动驾驶底层架构背后：特斯拉FSD v15带来了什么？” — https://www.techweb.com.cn/it/2026-11-23/2980783.shtml | 2026-11-23 | 1B → 10B claim, architecture rebuild, shared vision encoder, HW4 | `REPORTED` |
| 9 | 网通社, “特斯拉FSD v14.3.5推送：统一智能召唤、FSD与Robotaxi模型” — https://m.news18a.com/news/storys_277919.html | 2026-07-21 | V14.3.5 unified model; Robotaxi/L4 model distilled to HW4; MLIR runtime | `REPORTED` |
| 10 | Sohu Auto, “科技先声｜FSD V15刷屏：全网看屏幕，特斯拉在赌什么？” — https://www.sohu.com/a/1105158056_99900743 | 2026-11-23 | 25× interpretation, Robotaxi link, China comparison, Reality View latency caveat | `REPORTED` |
| 11 | Sina/Weibo relay of Musk's posts, “关于马斯克最新公布的特斯拉 AI 世界模型…” — https://www.sina.cn/news/detail/5368140141953489.html | 2026-11-22 | UVE full name (Universal Vision Encoder), 18 Hz, 3B+1B, 8 frames / 36 fps, half TRIP, 3 TRIPs per SoC B | `REPORTED` (second-hand) |

## The conflicts this note does not hide

### 1) 4B vs 10B

- Musk's reply (screenshot, source 7) is read as **3B action + 1B renderer = 4B**.
- Sources 3 and 8 report **~10B parameters**, described as a 10× increase from ~1B.
- This note treats **4B as the on-car inference graph**. It does **not** treat 10B as the teacher: the scope is undefined, and it conflicts with Musk's 3B+1B. Grok is a different order of model, far larger than 10B. One open hypothesis is that **UVE itself is part of Grok**, possibly after multi-stage distillation; public information does not confirm it.

### 2) 25× critical intervention rate vs “25× safer”

- Source 1 and source 10 describe the official claim as **critical intervention rate** improvement versus V14.
- It is **not** a 25% improvement, and **not** a direct 25× reduction in crash rate.
- The V14 baseline and the exact definition are not public. Third-party fleet data is needed.

### 3) UVE full name and 18 Hz provenance

- The full name **Universal Vision Encoder** and the 18 Hz / 3B / 1B / 8-frame / 36 fps / half-TRIP details come from second-hand relays of Musk's posts (source 11), not from Tesla's release notes.
- The release notes say only “vision encoder.” Treat the acronym expansion and the timing numbers as `REPORTED`, not `PUBLIC`.

## What is public vs inferred

**Public / official**
- V15 rollout, HW4-only, feature list, 25× claim, Reality View, novel view synthesis, 3DGS, same end-to-end model, half a TRIP accelerator, shared encoder across FSD/Robotaxi/Optimus, internet-video pretraining.
- CVPR 2026: 2B input tokens → 2 output tokens; intermediate outputs; World Sim NN ↔ Policy NN; shared foundation model across robots.
- ICCV 2025: 3DGS, 220 ms reconstruction, language reasoning, evaluation challenges.

**Reported (second-hand)**
- UVE full name (Universal Vision Encoder), 18 Hz, 3B + 1B, 8 frames / 36 fps, half TRIP: relayed from Musk's posts.
- 10B parameters: a media number with unclear scope; it conflicts with 3B+1B and is **not** treated as the teacher.
- V14.3.5 unified model + Robotaxi→HW4 distillation.
- Chinese industry narrative around 蔚小理 / Huawei / Xiaomi.

**Inferred / hypothesis**
- V15 training pipeline: general VLM/world-model pretrain → VLA/WAM adaptation → distillation deployment.
- SFT may remain, may be absorbed, or may be only an annealing interval; public information is insufficient.
- Compact token shift from instance-centric to scene-centric.
- Task framing: three task sets — 4D dense world (4D video prediction basic, with NVS/3DGS/depth as sub-tasks), 4D sparse world (AD perception), planner (WAM S01, with multi-modal trajectory prediction and agent prediction).
- 4D NVS data recipe along the trajectory; pose as language.
- External spatial/temporal super-resolution.
- “2亿/月” as an order-of-magnitude reference.
- VLA + WAM convergence.

## Update log

| Date | Change |
|---|---|
| 2026-11-23 | v1.0: first public version; 14 sections + appendix, 11 figures, ZH/EN, print CSS, WeChat editions. |
| 2026-11-23 | v1.1: retitled “管中窥豹 FSD V15”; corrected UVE to Universal Vision Encoder; clarified 18 Hz / 10B / Grok provenance; changed task taxonomy to three sets; merged distillation/deployment; moved the 11-point table to Appendix A; removed overconfident framing. |
| 2026-11-23 | v1.2: removed the Evidence key card and community latency estimates; rewrote second-person/meta prose for readers; expanded UVE to Universal Vision Encoder; added inline source links for media numbers; renamed the stage “蒸馏部署”; restructured the three task sets and WAM S01 sub-tasks; added a dedicated VLM → VLA/WAM → 蒸馏部署 subsection. |
