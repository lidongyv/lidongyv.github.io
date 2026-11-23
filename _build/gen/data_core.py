# -*- coding: utf-8 -*-
"""Single source of truth for the site content (zh + en).

All facts come from the 4 resume PDFs (2026-06-30) and verified public sources.
Nothing here should be invented: if a number is not in the resume it is not here.
"""


def L(zh, en):
    return {"zh": zh, "en": en}


PROFILE = {
    "name": L("于立冬", "Lidong Yu"),
    "name_en": "Lidong Yu",
    "handle": "lidongyv",
    "title": L("自动驾驶算法负责人", "Head of Autonomous Driving Algorithms"),
    "org": L("新石器慧通（北京）科技有限公司", "Neolix Technology (Beijing) Co., Ltd."),
    "headline": L(
        "把 <em>VLA</em> 与 <em>World&nbsp;Action&nbsp;Model</em> 从论文推进到量产车上",
        "Taking <em>VLA</em> and <em>World&nbsp;Action&nbsp;Models</em> from papers to production vehicles",
    ),
    "lede": L(
        "自动驾驶算法负责人，带 38 人算法团队、约 8,700 张 GPU。"
        "十年从立体匹配、单目深度、跟踪起步，经蔚来、元戎、地平线、NVIDIA 到新石器，"
        "完整走完 <strong>感知模块 → 两阶段端到端 → 单阶段 VLA / 世界模型</strong> 这条技术路径，"
        "并亲手把每一代方案送上车。",
        "Head of autonomous-driving algorithms, leading 38 engineers and ~8,700 GPUs. "
        "I started from stereo matching, monocular depth and tracking, then went through NIO, DeepRoute, "
        "Horizon Robotics, NVIDIA and now Neolix — walking the full path from "
        "<strong>modular perception → two-stage end-to-end → one-stage VLA / world models</strong>, "
        "and shipping every generation onto real vehicles.",
    ),
    "email": "yvlidong@gmail.com",
    "phone": "+86 151-6523-1608",
    "location": L("北京，中国", "Beijing, China"),
    "scholar": "https://scholar.google.com/citations?hl=en&user=cYLbx3AAAAAJ",
    "github": "https://github.com/lidongyv",
    "site": "https://lidongyv.github.io/",
    "status": L("正在寻找 VLA / 世界模型落地方向的机会", "Open to roles on VLA / world-model deployment"),
    "status_short": L("求职中", "Open to work"),
    "focus_tags": [
        L("VLA", "VLA"),
        L("World Action Model", "World Action Model"),
        L("端到端自动驾驶", "End-to-end AD"),
        L("数据闭环", "Data closed loop"),
        L("感知 → 规控", "Perception → planning"),
    ],
}

STATS = [
    {"value": "8", "label": L("同行评审论文（ICCV/CVPR/NeurIPS/AAAI）", "Peer-reviewed papers (ICCV/CVPR/NeurIPS/AAAI)")},
    {"value": "5", "label": L("发明专利", "Invention patents")},
    {"value": "6", "label": L("段量产级算法经历", "Production algorithm roles")},
    {"value": "38 / 8.7k", "label": L("团队规模（人） / 算力（GPU）", "Team size (people) / compute (GPUs)")},
]

# --- What I work on: the VLA / WAM thesis -----------------------------------
FOCUS = [
    {
        "kicker": L("01 · 建模", "01 · Modeling"),
        "title": L("单阶段：感知后置，规划直接吃图像流", "One-stage: perception moves downstream, planning reads the image stream"),
        "body": L(
            "把 planning 从「基于感知结果」改成「基于图像流」，感知元素以自回归 token 形式进入策略，"
            "去掉 perception tokenizer 并把它降级为 debugger 任务。多模态回归 → 自回归 → 扩散，"
            "再叠加 RL post-training 与 Cosmos 世界模型闭环训练。",
            "Planning moved from consuming perception outputs to consuming the raw image stream: "
            "perception elements enter the policy as autoregressive tokens, the perception tokenizer is removed "
            "and demoted to a debugger task. Regression → AR → diffusion, plus RL post-training and "
            "closed-loop training against a Cosmos world model.",
        ),
        "tags": [L("One-stage E2E", "One-stage E2E"), L("AR + Diffusion", "AR + Diffusion"), L("RL post-training", "RL post-training")],
    },
    {
        "kicker": L("02 · 数据", "02 · Data"),
        "title": L("VLA 的瓶颈不在模型，在数据闭环", "The bottleneck of VLA is not the model — it is the data loop"),
        "body": L(
            "0→1 搭建数据闭环并用自动化标注把交付量从 1k 推到 30k 交叉口/月，"
            "把 issue 解决周期从 D+10 压到 D+3，把数据冗余从 80% 降到 5%。"
            "VLM 自动 triage、agent 驱动评测与仿真，是我现在做 VLA 落地的日常。",
            "Built the data closed loop from 0 to 1, then pushed automated labelling throughput from "
            "1k to 30k intersections per month, cut issue turnaround from D+10 to D+3 and redundancy from 80% to 5%. "
            "VLM-based auto triage plus agent-driven evaluation and simulation is the daily work of shipping VLA.",
        ),
        "tags": [L("4D GT", "4D GT"), L("Auto mining", "Auto mining"), L("VLM triage", "VLM triage")],
    },
    {
        "kicker": L("03 · 评测", "03 · Evaluation"),
        "title": L("世界模型要能当「考场」，不只是「渲染器」", "A world model must be an exam hall, not just a renderer"),
        "body": L(
            "我把 render 的目标从「单场景重建」改成「泛化的生成」，DiT 与 3D-GS 只作为渲染手段；"
            "用 Cosmos / VGGT / Qwen 蒸馏做 backbone 与 CoT inception，让世界模型同时支撑闭环评测、"
            "失败 case 复现与 conditional inception（语言 / layout / 轨迹）。",
            "I reframed rendering from single-scene reconstruction to generalized generation, with DiT and 3D-GS "
            "as rendering tools only; backbones and CoT inception were distilled from Cosmos / VGGT / Qwen so the "
            "world model simultaneously supports closed-loop evaluation, failure-case replay and conditional "
            "inception (language / layout / trajectory).",
        ),
        "tags": [L("Cosmos", "Cosmos"), L("Closed-loop eval", "Closed-loop eval"), L("Conditional inception", "Conditional inception")],
    },
]

# --- Career ------------------------------------------------------------------
CAREER = [
    {
        "company": L("新石器慧通（北京）科技有限公司", "Neolix Technology (Beijing) Co., Ltd."),
        "company_short": "Neolix",
        "role": L("AD 算法负责人 · AD 二部", "Head of AD Algorithms · AD Division II"),
        "when": L("2025.08 — 至今", "2025.08 — present"),
        "scope": L("算法团队 38 人 · 约 8,700 张 GPU", "38-person algorithm org · ~8,700 GPUs"),
        "summary": L(
            "负责感知、端到端与世界模型的整体算法架构：用一个 group query-based 模型统一单任务与多任务，"
            "把感知任务重构成适配 E2E 的形态，并用两阶段预训练 + 世界模型把 VLA 推到可量产。",
            "Own the algorithm architecture across perception, end-to-end driving and world models: one group "
            "query-based model unifies single-task and multi-task training, perception is refactored to fit E2E, "
            "and a two-stage pretrain plus world model takes VLA to production readiness.",
        ),
        "bullets": [
            L("用 group query-based 模型统一单任务 / 多任务训练，从 multi-head 设计改为数据友好的 single-head；"
              "在 token 级别（BEV token / image token）而非 task-loss 级别做任务权重重配平。",
              "Unified single-task and multi-task training behind one group query-based model: multi-head → "
              "data-friendly single-head, rebalancing task weights at token level (BEV / image tokens) instead of task-loss level."),
            L("一个 query-based 模型覆盖 object detection、lane detection、occupancy network、E2E 红绿灯检测与导航模型；"
              "lane 与红绿灯升级为 navi-based static E2E，为上游决策提供强先验。",
              "A single query-based model covers object detection, lane detection, occupancy network, E2E traffic-light "
              "detection and the navigation model; lane and traffic-light detection became a navi-based static E2E model that "
              "gives upstream decisions a strong prior."),
            L("两阶段预训练：从 Cosmos / VGGT 做 LLM 蒸馏（CoT、3D 几何估计、3D OD、lane、failsafe、场景识别），"
              "并做 perception model inception；支持 frozen head / frozen backbone 的可控迭代与随时加数据。",
              "Two-stage pretraining: LLM distillation from Cosmos / VGGT (CoT, 3D geometry, 3D OD, lane, failsafe, "
              "scene recognition) plus perception-model inception; controllable iteration with frozen head / frozen "
              "backbone and data added at any time."),
            L("基于自回归的轨迹预测：多任务 FT + CoT debugger + 多模态扩散 + RL post-training；"
              "通用世界模型用 D4RT 统一并行 AR 与 3D insight，条件注入支持语言 / layout / 轨迹。",
              "AR-based trajectory prediction with multi-task fine-tuning, a CoT debugger, multi-modal diffusion and "
              "RL post-training; the general world model uses D4RT for unified parallel AR plus 3D insight, with "
              "conditional injection from language / layout / trajectory."),
            L("AI-native 与 RaaS 转型：VLM 自动 triage、agent skills 驱动评测与仿真、仿真器 auto checker、"
              "agent 驱动的前后端开发与客服。",
              "AI-native and RaaS transformation: VLM auto triage, agent skills driving evaluation and simulation, "
              "simulator auto checker, and agent-driven front-end / back-end development and customer service."),
        ],
        "tags": ["VLA", "World Model", "Group Query", "Occupancy", "RL post-training", "Cosmos", "VGGT"],
    },
    {
        "company": L("NVIDIA", "NVIDIA"),
        "company_short": "NVIDIA",
        "role": L("Principal Engineer · 自动驾驶部门 Free IC", "Principal Engineer · Free IC, Autonomous Vehicle Department"),
        "when": "2024.07 — 2025.08",
        "scope": L("端到端与世界模型方向", "End-to-end driving and world models"),
        "summary": L(
            "从感知模型演进到两阶段端到端，再升级到单阶段 Alpamayo；负责 backbone 预训练 / 微调 / post-training 框架，"
            "以及 Cosmos 世界模型的预训练与 render 研究。",
            "Evolved perception models into a two-stage end-to-end stack and then into one-stage Alpamayo; owned the "
            "backbone pretrain / fine-tune / post-training framework and Cosmos world-model pretraining plus render research.",
        ),
        "bullets": [
            L("感知模型 → 两阶段 E2E：lane detection 从 segment-based（所见即所得，优化超分、几何精度、BEV 多尺度融合、"
              "动态多分辨率 DCN）演进到 query-based（静态 E2E，引入 navi info 与过路口、split/merge、换道等上游决策），"
              "再到 token-based。",
              "Perception → two-stage E2E: lane detection went from segment-based (what-you-see-is-what-you-get, with "
              "better super-resolution, geometric precision, BEV multi-scale fusion and dynamic multi-resolution DCN) to "
              "query-based (static E2E with navi info, intersections, split/merge and lane changes) and then to token-based."),
            L("token-based lane detection 打通两阶段 → 单阶段 Alpamayo：以 AR 方式 tokenize 所有感知元素，"
              "支撑上游 conditional SA 与下游 debugger 可视化。",
              "Token-based lane detection unlocked the jump from two-stage to one-stage Alpamayo: all perception elements "
              "are tokenized autoregressively, supporting upstream conditional SA and downstream debugger visualization."),
            L("PNC → 单阶段：planning 模块模型化，单模态 → 多模态回归，回归 → AR → 扩散；"
              "backbone 蒸馏自 Cosmos / Qwen / Llama，做 CoT、3D 几何与 AD 多任务 inception。",
              "PNC → one-stage: planning was model-ized, single-mode → multi-mode regression, regression → AR → diffusion; "
              "backbones distilled from Cosmos / Qwen / Llama with CoT, 3D geometry and AD multi-task inception."),
            L("世界模型：Cosmos 预训练 → render 研究；从 VGGT 与 Qwen 蒸馏；"
              "E2E 数据闭环用离线感知模型自动标注、用 VLM 自动挖掘。",
              "World model: Cosmos pretraining → render research, distilled from VGGT and Qwen; the E2E data closed loop "
              "uses offline perception models for auto-labelling and VLM for auto mining."),
        ],
        "tags": ["E2E", "Alpamayo", "Cosmos", "Distillation", "Diffusion planning", "Tokenization"],
    },
    {
        "company": L("北京地平线机器人技术研发有限公司", "Horizon Robotics"),
        "company_short": "Horizon",
        "role": L("静态感知算法负责人 · 「土星五号」项目", "Head of Static Perception Algorithms · \"Saturn V\" Project"),
        "when": "2023.06 — 2024.07",
        "scope": L("15 人模型团队 + 5 人后处理团队 · 约 1,200 张 GPU", "15-person modeling team + 5-person post-processing team · ~1,200 GPUs"),
        "summary": L(
            "负责轻地图 PNC 方案与众包建图下的道路结构认知：把车道线检测从 segment-based 换到 query-based，"
            "同时重建整条数据闭环。",
            "Owned the light-map PNC solution and the road-structure understanding module under crowdsourced mapping: "
            "replaced segment-based lane detection with query-based and rebuilt the whole data loop around it.",
        ),
        "bullets": [
            L("路沿采用「所见即所得」策略，模型只输出车道线与路沿、不输出中心线；针对路口遮挡等场景重新设计数据清洗与标注规则。",
              "Kept a what-you-see-is-what-you-get strategy for road edges — the model emits lane lines and edges only, no "
              "centerlines — and redesigned data cleaning and labelling rules for occluded intersections."),
            L("以 query-based 方案替换 segment-based：去掉后处理、人力前移，计算效率持平而 <strong>F1 提升 15%</strong>；"
              "路口两侧横向及远处弱可见区域 recall 显著提升，且相比 entry-point 策略可以直接构建车道。",
              "Replaced segment-based with query-based: post-processing removed, effort shifted upstream, compute cost flat "
              "while <strong>F1 improved 15%</strong>; recall on lateral intersection sides and distant weakly-visible "
              "regions improved markedly, and unlike the entry-point strategy it can build lanes directly."),
            L("离散要素（arrow / crosswalk / junction）用 query-based 替换两阶段检测器：分数持平，"
              "<strong>计算效率提升 30% 以上</strong>；query 分组把单任务与多任务的指标差距拉平，支持同源数据的单模型多任务训练。",
              "Discrete elements (arrows, crosswalks, junctions) moved from a two-stage detector to query-based: score flat, "
              "<strong>compute efficiency up >30%</strong>; query grouping closed the single-task / multi-task metric gap and "
              "enabled one-model multi-task training on same-source data."),
            L("数据闭环：构建道路结构全要素、多任务、同源时序数据集；参与自动化采集系统、路测链路、车云协同、"
              "道路采集覆盖等大闭环系统的 0→1 架构与实现。",
              "Data closed loop: built full-element, multi-task, same-source temporal datasets for road structure, and "
              "co-architected from 0 to 1 the automated collection, road-test link, vehicle-cloud collaboration and "
              "road-coverage systems."),
            L("联调自动化标注工具（4D GT），交付量 <strong>1k → 30k 交叉口/月</strong>；"
              "自动挖掘把数据冗余从 <strong>80% 降到 5%</strong>；重构闭环后 issue 解决周期 <strong>D+10 → D+3</strong>。",
              "Co-developed the automated labelling tool (4D GT), lifting delivery from <strong>1k to 30k "
              "intersections/month</strong>; auto mining cut redundancy from <strong>80% to 5%</strong>; re-architecting the "
              "loop moved issue turnaround from <strong>D+10 to D+3</strong>."),
            L("GT 生成：搭建 HD map → local BEV 的 GT 生成系统（200 万 image、1,200 万 lane instance）；"
              "segment-based 车道线检测 AP <strong>13% → 60%</strong>、EPE <strong>1.5 m → 0.18 m</strong>。",
              "GT generation: an HD-map → local-BEV GT system (2M images, 12M lane instances); segment-based lane detection "
              "went from AP <strong>13% to 60%</strong> and EPE from <strong>1.5 m to 0.18 m</strong>."),
            L("拓扑学习三条路线并行：基于 language model 的自回归关键点生成；基于 MaskFormer 的拓扑关键点分割；"
              "以及把 query-based 点链生成「逆」成 segment 算法的 reverse transformer。",
              "Three parallel topology-learning routes: autoregressive keypoint generation with a language model; "
              "MaskFormer-based topology keypoint segmentation; and a \"reverse transformer\" that turns query-based "
              "point-chain generation back into a segment algorithm."),
        ],
        "tags": ["Lane detection", "Query-based", "Mapless", "4D GT", "Data loop", "Topology"],
    },
    {
        "company": L("深圳元戎启行科技有限公司", "DeepRoute.ai"),
        "company_short": "DeepRoute",
        "role": L("静态感知模型负责人", "Head of Static Perception Models"),
        "when": "2022.04 — 2023.06",
        "scope": L("核心团队 4 人 · 600 张 GPU", "Core team of 4 · 600 GPUs"),
        "summary": L(
            "负责 MapFree 下的车道线检测系统：2023 年交付国内第一套 Mapless 无图方案与问界（AITO）POC。",
            "Owned the lane-detection system under MapFree and delivered China's first Mapless solution in 2023 "
            "together with the AITO (Wenjie) POC.",
        ),
        "bullets": [
            L("Lane segment 方案支持城区点到点与高速自动巡航，<strong>匝道 / 路口通过率 >85%</strong>。",
              "The lane-segment solution supported urban point-to-point and highway auto-cruise with a "
              "<strong>>85% pass rate at ramps and intersections</strong>."),
            L("模型演进：lane segment → query-based → AR-based lane detection。",
              "Model evolution: lane segment → query-based → AR-based lane detection."),
            L("基建：200M+ 图像数据挖掘平台，约 1,200 万 frame 自动挖掘；多任务训练 auto-balance 框架。",
              "Infrastructure: a 200M+ image data-mining platform with ~12M frames auto-mined, plus an auto-balance "
              "framework for multi-task training."),
        ],
        "tags": ["Mapless", "Lane segment", "Data mining", "Auto-balance"],
    },
    {
        "company": L("上海蔚来汽车有限公司", "NIO"),
        "company_short": "NIO",
        "role": L("主任工程师 · 自动驾驶算法", "Principal Engineer · Autonomous Driving Algorithms"),
        "when": "2021.03 — 2022.04",
        "scope": L("ET7 / ES8 量产项目", "ET7 / ES8 production programs"),
        "summary": L(
            "国内第一代 AD 自研算法：ET7、ES8 的 Mono3D 检测跟踪模型、数据闭环与模型评测框架。",
            "China's first-generation in-house AD algorithms: the Mono3D detection-tracking model for ET7 and ES8, "
            "plus the data closed loop and model-evaluation framework.",
        ),
        "bullets": [
            L("数据驱动的 Mono3D 检测跟踪模型；object detection / lane detection 的数据闭环与模型评测框架。",
              "A data-driven Mono3D detection-tracking model, with data closed loops and evaluation frameworks for "
              "object detection and lane detection."),
            L("7 相机 + 1 激光雷达的规则化关联后处理；设计可学习的 cost metric 与匹配方案替代规则与二分图匹配；"
              "ReID 数据标注与伪标签生成。",
              "Rule-based association post-processing for 7 cameras + 1 LiDAR, then a learnable cost metric and matching "
              "scheme replacing the rules and bipartite matching; ReID labelling and pseudo-label generation."),
            L("结果：在 20% 3D 误差率下 <strong>0.03 ID switch、86% recall</strong>，整体超过 EQ4；"
              "150 m 外远距离小目标关联分裂率 <strong>0.1%</strong>；全图稠密深度 <strong>6.8% rel error</strong>、"
              "地表 <strong>1.3% rel error</strong>。",
              "Results: at 20% 3D error rate, <strong>0.03 ID switch and 86% recall</strong>, overall beyond EQ4; "
              "<strong>0.1%</strong> split rate for distant small objects beyond 150 m; dense depth "
              "<strong>6.8% relative error</strong> and ground <strong>1.3%</strong>."),
            L("通过量产 HOMO 测试，LKA / LKW 功能测试通过；支撑 ET7 SUD 与 NAD 路测与量产部署。",
              "Passed production HOMO testing and LKA / LKW feature tests; supported ET7 SUD and NAD road testing and "
              "mass-production deployment."),
        ],
        "tags": ["Mono3D", "Multi-object tracking", "ReID", "Quantization", "Mass production"],
    },
    {
        "company": L("驭势科技（北京）有限公司", "UISEE Technology (Beijing)"),
        "company_short": "UISEE",
        "role": L("高级研究员 · 南京 AI 研究院", "Senior Researcher · Nanjing AI Research Institute"),
        "when": "2019.05 — 2020.04",
        "scope": L("2019 年国内首家 Shadow Mode 数据闭环 + IL-based E2E 上车", "2019: first in China to deploy a Shadow-Mode data loop + IL-based E2E on-vehicle"),
        "summary": L(
            "2019 年国内第一家把 Shadow Mode 数据闭环系统 + 基于模仿学习的端到端上车；"
            "做面向 AD 的模仿学习、强化学习、数据生成与仿真，以及 3D 目标检测与跟踪联合。",
            "In 2019, the first team in China to put a Shadow-Mode data closed loop plus imitation-learning-based "
            "end-to-end driving on a real vehicle; worked on imitation learning, RL, data generation and simulation "
            "for AD, together with joint 3D detection and tracking.",
        ),
        "bullets": [
            L("模仿学习：用 world model、GAIL、InfoGAIL 作为基线，并加入预测模块使结果更稳定；"
              "在仿真平台上取得 970 的驾驶得分，超过多数强化学习方法。",
              "Imitation learning: world models, GAIL and InfoGAIL as baselines plus a prediction module for stability; "
              "reached a driving score of 970 in simulation, outperforming most RL methods."),
            L("动作预测模块成功上真车，在封闭道路完成相对导航。",
              "The action-prediction module was deployed on a real car and completed relative navigation on closed roads."),
            L("单目标跟踪：用特征解耦把前景信息从特征中分离再做相关运算，OTB100 精度 0.389、VOT 0.549，"
              "长视频更稳定、对相机抖动更鲁棒。",
              "Single-object tracking: disentangled foreground features before correlation, reaching 0.389 accuracy on "
              "OTB100 and 0.549 on VOT, more stable on long videos and more robust to camera vibration."),
            L("多目标跟踪与视频目标检测：用形变卷积预测帧间形变并连接检测结果；"
              "KITTI 上车辆 AP 0.628（基线 0.560）、Waymo 上 0.663（基线 0.626）。",
              "Multi-object tracking and video object detection: deformable convolution predicting inter-frame deformation "
              "to link detections; AP 0.628 on KITTI vehicles (baseline 0.560) and 0.663 on Waymo (baseline 0.626)."),
        ],
        "tags": ["Imitation learning", "GAIL", "Shadow mode", "SOT", "MOT", "Deformable conv"],
    },
]

EDUCATION = [
    {
        "school": L("北京理工大学", "Beijing Institute of Technology"),
        "badge": "985",
        "degree": L("计算机科学与技术 · 硕士", "M.S. in Computer Science & Technology"),
        "dept": L("计算机学院", "School of Computer Science"),
        "when": "2016.08 — 2019.06",
        "where": L("北京", "Beijing"),
        "note": L("导师：贾云得教授、吴宇威教授", "Advisors: Prof. Yunde Jia, Prof. Yuwei Wu"),
    },
    {
        "school": L("中国海洋大学", "Ocean University of China"),
        "badge": "985",
        "degree": L("计算机科学与技术 · 本科", "B.S. in Computer Science & Technology"),
        "dept": L("计算机学院", "School of Computer Science"),
        "when": "2012.08 — 2016.06",
        "where": L("青岛", "Qingdao"),
        "note": "",
    },
]
