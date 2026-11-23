# -*- coding: utf-8 -*-
"""Work experience, taken from the *detailed* resume (3-page version).

Each role carries `sections`: labelled groups of bullets, in the order the
detailed resume uses. Nothing from that resume is dropped — the summary line, the
perfected groupings and every bullet are kept.
"""

from .data_core import L

CAREER = [
    {
        "company": L("新石器慧通（北京）科技有限公司", "Neolix Technology (Beijing) Co., Ltd."),
        "company_short": "Neolix",
        "role": L("AD 算法负责人 · AD 二部", "Head of AD Algorithms · AD Division II"),
        "when": L("2025.08 — 2026.11", "2025.08 — 2026.11"),
        "scope": L("算法团队 38 人 · 约 8,700 张 GPU", "38-person algorithm org · ~8,700 GPUs"),
        "summary": L(
            "负责 AD 二部的算法：感知、端到端与世界模型，以及支撑它们的 infra。",
            "Owned AD Division II algorithms: perception, end-to-end driving and world models, plus the infrastructure "
            "behind them.",
        ),
        "sections": [
            {
                "label": L("Perception — 重构单任务 / 多任务 task formulation",
                           "Perception — refactored single-task / multi-task formulation"),
                "items": [
                    L("用 group query-based model 统一 single-task / multi-task 训练，从 multi-head 设计转为数据友好的 single-head 多任务设计。",
                      "Used a group query-based model to unify single-task and multi-task training, moving from a "
                      "multi-head design to a data-friendly single-head multi-task design."),
                    L("一个 query-based model 覆盖 object detection、lane detection、occupancy network、E2E 红绿灯检测与导航模型。",
                      "One query-based model for object detection, lane detection, occupancy network, E2E traffic-light "
                      "detection and the navigation model."),
                    L("single head 可自设 data balance 而不影响其他任务；按 BEV token / image token 在 token 级别（而非 task-loss 级别）rebalance task weight。",
                      "A single head allows self-configured data balance without affecting other tasks; task weights are "
                      "rebalanced by BEV token / image token at token level rather than at task-loss level."),
                    L("重构感知任务以适配 E2E：Lidar 前融合、object detection、MOT 升级为 object prediction，future tokenizer 支持 pre/post-training。",
                      "Refactored perception tasks to fit E2E: LiDAR early fusion, object detection, MOT upgraded to "
                      "object prediction, and a future tokenizer supporting pre- and post-training."),
                    L("lane detection、红绿灯检测升级为 navi-based static E2E model，降低网络学习难度，为前置决策提供 strong prior。",
                      "Upgraded lane and traffic-light detection to a navi-based static E2E model, reducing learning "
                      "difficulty and providing a strong prior for upstream decisions."),
                ],
            },
            {
                "label": L("E2E 与世界模型（WM）", "E2E and world model"),
                "items": [
                    L("两阶段 pretrain —— 从 Cosmos / VGGT 进行 LLM 蒸馏（CoT 能力、3D 几何估计、3D OD / lane 任务、failsafe 与场景识别）；"
                      "perception model inception（提供 AD task-force 知识，服务 RL 与 vis debugger）。",
                      "Two-stage pretrain — LLM distillation from Cosmos / VGGT (CoT, 3-D geometry estimation, 3D OD, "
                      "lane tasks, failsafe and scene recognition); perception model inception (AD task-force knowledge "
                      "for RL, plus a visual debugger)."),
                    L("可控的迭代训练发版 —— frozen head 预训练 backbone；frozen backbone 微调 perception head；支持随时加数据。",
                      "Controllable iterative training and release — pretrain the backbone with a frozen head; fine-tune "
                      "the perception head with a frozen backbone; add data at any time."),
                    L("基于自回归（AR）的轨迹预测 —— pre/post perception/perceiver 的多任务 FT；CoT debugger；多模态 diffusion；RL post-training。",
                      "Autoregressive trajectory prediction — multi-task fine-tuning with pre/post perception/perceiver; "
                      "CoT debugger; multi-modal diffusion; RL post-training."),
                    L("通用世界模型 —— pretrain（backbone LLM 蒸馏，Cosmos → CoT 的 language inception；D4RT 提供统一并行 AR 与 3D insight）；"
                      "render（面向 generalized generation 而非单场景重建，DiT 与 3D-GS 仅用于 render）；conditional inception（language / layout / trajectory）。",
                      "Generalized world model — pretrain (LLM distillation for the backbone; Cosmos → CoT language "
                      "inception; D4RT for unified parallel AR plus 3D insight); render (generalized generation rather "
                      "than single-scene reconstruction, DiT and 3D-GS only as rendering tools); conditional inception "
                      "(language / layout / trajectory)."),
                ],
            },
            {
                "label": L("Infra — AI native 与 RaaS 转型", "Infra — AI-native and RaaS transformation"),
                "items": [
                    L("VLM 自动 triage；agent skills 驱动评测与仿真；个人 AI agent 提效会议；Vibe coding 驱动的自动研究。",
                      "VLM auto triage; evaluation and simulation driven by agent skills; personal AI agents for "
                      "meetings; vibe-coding-driven automatic research."),
                    L("E2E 与 WM 的数据闭环：离线感知模型自动标注；VLM 自动挖掘。",
                      "Data closed loop for E2E and WM: auto-labelling with offline perception models; auto mining with "
                      "a VLM."),
                    L("agent 驱动的前后端开发与客服；仿真器 auto checker。",
                      "Agent-driven front-end / back-end development and customer service; auto checker for the "
                      "simulator."),
                ],
            },
        ],
    },
    {
        "company": L("NVIDIA", "NVIDIA"),
        "company_short": "NVIDIA",
        "role": L("Principal Engineer · 自动驾驶部门 Free IC",
                  "Principal Engineer · Free IC, Autonomous Vehicle Department"),
        "when": L("2024.07 — 2025.08", "2024.07 — 2025.08"),
        "scope": L("端到端与世界模型方向", "End-to-end driving and world models"),
        "summary": L(
            "从感知模型演进到两阶段端到端，再升级到单阶段 Alpamayo；负责 backbone 预训练 / 微调 / post-training 框架，"
            "以及 Cosmos 世界模型的预训练与 render 研究。",
            "Evolved perception models into a two-stage end-to-end stack and then into one-stage Alpamayo; owned the "
            "backbone pretrain / fine-tune / post-training framework and Cosmos world-model pretraining plus render "
            "research.",
        ),
        "sections": [
            {
                "label": L("Perception Model → 两阶段 E2E", "Perception model → two-stage E2E"),
                "items": [
                    L("segment-based lane detection（所见即所得）：优化超分、提高几何精度、BEV 多尺度融合、动态多分辨率 DCN。",
                      "Segment-based lane detection (what-you-see-is-what-you-get): optimised super-resolution, better "
                      "geometric precision, BEV multi-scale fusion, dynamic multi-resolution DCN."),
                    L("query-based lane detection（静态 E2E）：摆脱「所见即所得」formulation，增强脑补能力，输入 navi info，"
                      "新增过路口、split/merge、lane change 等 navi model 前置决策。",
                      "Query-based lane detection (static E2E): moved beyond the WYSIWYG formulation, added reasoning, "
                      "took navi info as input, and covered intersection crossing, split/merge and lane changes as "
                      "upstream navi decisions."),
                    L("token-based lane detection：协助两阶段 E2E 升级至 one-stage Alpamayo，以 AR 方式 tokenize 所有 perception 元素，"
                      "支撑上游 conditional SA 与下游 debugger 可视化。",
                      "Token-based lane detection: enabled the two-stage → one-stage Alpamayo upgrade by tokenizing all "
                      "perception elements autoregressively, supporting upstream conditional SA and downstream debugger "
                      "visualisation."),
                ],
            },
            {
                "label": L("PNC → 单阶段 Alpamayo", "PNC → one-stage Alpamayo"),
                "items": [
                    L("规划模块模型化；单模态 → 多模态回归；回归 → AR → diffusion 形态。",
                      "Model-ised the planning module; single-mode → multi-mode regression; regression → AR → "
                      "diffusion."),
                    L("为 one-stage 升级把感知后置：Cosmos / Qwen / Llama backbone 蒸馏；CoT、3D 几何与 AD 多任务 inception 训练；"
                      "去掉 perception tokenizer（改作 debugger 任务）；规划从基于感知结果改为基于图像流。",
                      "Moved perception downstream for the one-stage upgrade: Cosmos / Qwen / Llama backbone "
                      "distillation; CoT, 3-D geometry and AD multi-task inception; removed the perception tokenizer "
                      "(repurposed as a debugger task); planning changed from perception-based to image-stream-based."),
                    L("规划 FT 与 post-training：AR、diffusion、多模态轨迹预测；RL post-training；Cosmos-WM 闭环训练。",
                      "Planning fine-tuning and post-training: AR, diffusion and multi-modal trajectory prediction; RL "
                      "post-training; closed-loop training with Cosmos-WM."),
                ],
            },
            {
                "label": L("世界模型", "World model"),
                "items": [
                    L("Cosmos pretrain → render 研究；从 VGGT 与 Qwen 蒸馏。",
                      "Cosmos pretrain → render research; distillation from VGGT and Qwen."),
                ],
            },
            {
                "label": L("Modeling 与 Infra", "Modeling and infra"),
                "items": [
                    L("Roadnet（16 人，~1,200 卡，3 个月）HD → Mapless，segment → query → token-based。",
                      "Roadnet (16 people, ~1,200 GPUs, 3 months): HD → Mapless, segment → query → token-based."),
                    L("Planner（13 人，~1,500 卡，3 个月）经典 PNC → model-based，two-stage E2E with AR。",
                      "Planner (13 people, ~1,500 GPUs, 3 months): classical PNC → model-based, two-stage E2E with AR."),
                    L("E2E（13 人，~3,500 卡，6 个月）Alpamayo 量产 ORIN 交付奔驰 SOP，one-stage E2E with Cosmos CoT 与 Alpamayo WM-RL。",
                      "E2E (13 people, ~3,500 GPUs, 6 months): Alpamayo mass-production ORIN delivery for the Mercedes-Benz "
                      "SOP, one-stage E2E with Cosmos CoT and Alpamayo WM-RL."),
                    L("Infra：感知持续训练（数据自动配平 + 挖掘框架）、蒸馏驱动的多任务训练优化；E2E-Alpamayo 的 backbone 预训练 / FT / post-training 框架。",
                      "Infra: perception continuous training (auto data balancing plus a mining framework), "
                      "distillation-driven multi-task training optimisation; the E2E-Alpamayo backbone pretrain / "
                      "fine-tune / post-training framework."),
                ],
            },
        ],
    },
    {
        "company": L("北京地平线机器人技术研发有限公司", "Horizon Robotics"),
        "company_short": "Horizon",
        "role": L("静态感知算法负责人 · 「土星五号」项目", "Head of Static Perception Algorithms · \"Saturn V\" Project"),
        "when": L("2023.06 — 2024.07", "2023.06 — 2024.07"),
        "scope": L("15 人模型团队 + 5 人后处理团队 · 约 1,200 张 GPU",
                   "15-person modeling team + 5-person post-processing team · ~1,200 GPUs"),
        "summary": L(
            "轻地图 PNC 方案与众包建图下的道路认知模块：把车道线检测从 segment-based 换到 query-based，并重建整条数据闭环。",
            "Light-map PNC solution and the road-understanding module under crowdsourced mapping: replaced "
            "segment-based lane detection with query-based and rebuilt the data loop around it.",
        ),
        "sections": [
            {
                "label": L("算法研发 —— 连续要素（车道线）", "R&D — continuous elements (lane lines)"),
                "items": [
                    L("路沿用「所见即所得」策略：模型只输出车道线与路沿，不输出中心线；针对路口遮挡等场景重新设计数据清洗与标注规则。",
                      "Kept a WYSIWYG strategy for road edges: the model outputs lane lines and edges only, no centre "
                      "lines; redesigned data cleaning and labelling rules for occluded intersections."),
                    L("用 query-based 替换 segment-based：去掉后处理、人力前移，计算开销相当而 F1 提升 15%。",
                      "Replaced segment-based with query-based: post-processing removed, effort shifted upstream, "
                      "comparable compute cost, F1 improved 15%."),
                    L("路口两侧与远处弱可见区域 recall 大幅提升；相比 entry-point 策略可直接构建车道。",
                      "Large recall gains on lateral intersection sides and distant weakly-visible regions; unlike the "
                      "entry-point strategy it can build lanes directly."),
                    L("query-based 时序策略把跟踪放进网络；PV refinement 降低对 BEV 特征的强依赖，引入新的多模态融合方式；query 分组支持 case 级精调。",
                      "A query-based temporal strategy moved tracking into the network; PV refinement reduced the strong "
                      "dependence on BEV features and introduced a new multi-modal fusion; query grouping enabled "
                      "case-level fine-tuning."),
                ],
            },
            {
                "label": L("算法研发 —— 离散要素（arrow / crosswalk / junction）",
                         "R&D — discrete elements (arrow, crosswalk, junction)"),
                "items": [
                    L("用 query-based 替换两阶段检测器，分数相当而计算效率提升 30% 以上。",
                      "Replaced the two-stage detector with a query-based solution at comparable score and more than "
                      "30% higher compute efficiency."),
                    L("query 分组拉平单任务与多任务的指标差距，支持同源数据的单模型多任务训练。",
                      "Query grouping closed the single-task / multi-task metric gap and enabled one-model multi-task "
                      "training on same-source data."),
                ],
            },
            {
                "label": L("数据闭环", "Data closed loop"),
                "items": [
                    L("构建道路结构全要素、多任务、同源时序数据集；针对 query-based 方案优化采集、标注规则与配平。",
                      "Built full-element, multi-task, same-source temporal datasets for road structure; optimised "
                      "collection, labelling rules and balancing for the query-based approach."),
                    L("参与自动化标注工具（4D GT）联调；参与自动化采集、路测链路、车云协同、道路采集覆盖等大闭环系统的 0→1 架构与实现。",
                      "Co-developed the automated labelling tool (4D GT); helped architect and implement, from 0 to 1, "
                      "the large closed-loop systems (automated collection, road-test link, vehicle-cloud collaboration, "
                      "road coverage)."),
                ],
            },
            {
                "label": L("GT 生成与拓扑学习", "GT generation and topology learning"),
                "items": [
                    L("GT 生成：搭建 HD map → local BEV 的 GT 生成系统（200 万 image、1,200 万 lane instance）。",
                      "GT generation: an HD-map → local-BEV GT system (2M images, 12M lane instances)."),
                    L("面向投影 GT 的模型：优化 segment-based 车道线检测（AP 13% → 60%，EPE 1.5 m → 0.18 m）；标注规则定义与数据清洗；"
                      "direction-head 连接 + 超分提升精度。",
                      "Projection-GT-friendly model: optimised segment-based lane detection (AP 13% → 60%, EPE 1.5 m → "
                      "0.18 m); labelling-rule definition and data cleaning; direction-head linking with "
                      "super-resolution for higher precision."),
                    L("query-based 车道线检测：车道–车道线–点链的多级回归；数据拓扑定义、算法、评测与可视化工具全栈开发。",
                      "Query-based lane detection: multi-level regression over lane–lane-line–point chains; full-stack "
                      "development of data-topology definition, algorithm, evaluation and visualisation tools."),
                    L("基于 language model 的拓扑学习：自回归关键点生成；数据拓扑定义与算法开发。",
                      "Language-model-based topology learning: autoregressive keypoint generation; data-topology "
                      "definition and algorithm development."),
                    L("基于 segment 的拓扑学习：MaskFormer 拓扑关键点分割；把 query-based 点链生成「逆」成 segment 算法的 reverse transformer。",
                      "Segment-based topology learning: MaskFormer topology-keypoint segmentation; a reverse transformer "
                      "that turns query-based point-chain generation back into a segment algorithm."),
                    L("数据驱动的模型自动迭代与评测。",
                      "Data-driven automatic model iteration and evaluation."),
                ],
            },
        ],
    },
    {
        "company": L("深圳元戎启行科技有限公司", "DeepRoute.ai"),
        "company_short": "DeepRoute",
        "role": L("静态感知模型负责人", "Head of Static Perception Models"),
        "when": L("2022.04 — 2023.06", "2022.04 — 2023.06"),
        "scope": L("核心团队 4 人 · 600 张 GPU", "Core team of 4 · 600 GPUs"),
        "summary": L(
            "MapFree 下的自车道线检测：一套 data-driven 的车道线检测系统。",
            "Lane detection under MapFree: a data-driven lane-detection system.",
        ),
        "sections": [
            {
                "label": L("MapFree 下的车道线检测", "Lane detection under MapFree"),
                "items": [
                    L("负责 static perception 的 model + post-process + data infra；lane detection 由 segment-based → query-based → group-wise multi-task；"
                      "static E2E + navi model。",
                      "Owned static-perception model, post-processing and data infra; lane detection went segment-based "
                      "→ query-based → group-wise multi-task; static E2E plus the navi model."),
                    L("Infra（开发 7 人 + 外包 5 人 + 下游 20 人）：0–1 搭建数据闭环；重构架构（issue 解决效率 D+10 → D+3）；"
                      "自动化标注工具（4D GT），1k → 3w 路口/月；data auto mining，冗余 80% → 5%。",
                      "Infra (7 devs + 5 contractors + 20 downstream): built the data closed loop 0 → 1; re-architected "
                      "it (issue turnaround D+10 → D+3); automated labelling tool (4D GT), 1k → 30k intersections per "
                      "month; auto data mining, redundancy 80% → 5%."),
                    L("2023 年完成国内第一套 Mapless 无图方案，交付问界 POC；Lane segment 方案支持城区点到点与高速自动巡航，"
                      "匝道/路口通过率 >85%；lane segment → query-based → AR-based lane detection。",
                      "Delivered China's first Mapless solution in 2023 and the AITO (Wenjie) POC; the lane-segment "
                      "solution supported urban point-to-point and highway auto-cruise with a >85% pass rate at ramps "
                      "and intersections; lane segment → query-based → AR-based lane detection."),
                    L("Infra：>2 亿 image 数据挖掘平台，约 1,200 万 frame 自动挖掘；多任务训练 auto-balance 框架。",
                      "Infra: a 200M+ image data-mining platform with ~12M frames auto-mined; an auto-balance framework "
                      "for multi-task training."),
                ],
            },
        ],
    },
    {
        "company": L("上海蔚来汽车有限公司", "NIO"),
        "company_short": "NIO",
        "role": L("主任工程师 · 自动驾驶算法", "Principal Engineer · Autonomous Driving Algorithms"),
        "when": L("2021.03 — 2022.04", "2021.03 — 2022.04"),
        "scope": L("ET7 / ES8 量产项目", "ET7 / ES8 production programs"),
        "summary": L(
            "3D 目标检测与车道线的量产交付：关联后处理、单目深度与 ground model。",
            "Production delivery for 3D object detection and lane lines: association post-processing, monocular depth "
            "and the ground model.",
        ),
        "sections": [
            {
                "label": L("职责", "Responsibilities"),
                "items": [
                    L("7 相机 + 1 激光雷达的规则化关联后处理。",
                      "Rule-based association post-processing for 7-camera + 1-LiDAR tracking association."),
                    L("基于学习的关联：设计可学习的 cost metric 与匹配方案，替换规则与二分图匹配。",
                      "Learning-based association: a learnable cost metric and matching scheme replacing rules and "
                      "bipartite-graph matching."),
                    L("ReID 数据标注与伪标签生成。", "ReID labelling and pseudo-label generation."),
                    L("后处理车端部署与测试；ET7 NAD & SUD 功能交付。",
                      "On-vehicle deployment and testing of post-processing; ET7 NAD & SUD feature delivery."),
                    L("单目深度估计：数据标注、算法开发、量化部署与评测。",
                      "Monocular depth estimation: labelling, algorithm development, quantised deployment and "
                      "evaluation."),
                    L("Ground model：数据标注、算法开发、量化部署、路测回测。",
                      "Ground model: labelling, algorithm development, quantised deployment, road-test regression."),
                    L("3D 车道线方案支持与后处理研发。",
                      "3D lane-line solution support and post-processing R&D."),
                ],
            },
            {
                "label": L("结果", "Results"),
                "items": [
                    L("20% 3D 误差率下关联：0.03 ID-switch、86% recall，整体超过 EQ4。",
                      "Association at a 20% 3D error rate: 0.03 ID-switch, 86% recall — overall beyond EQ4."),
                    L("150 m 外远距离小目标分裂率 0.1%；拥堵与泊车场景关联稳定。",
                      "0.1% split rate for distant small objects beyond 150 m; stable association in congestion and "
                      "parking scenarios."),
                    L("全图 dense depth 6.8% rel. error；地表 1.3% rel. error。",
                      "Full-image dense depth at 6.8% relative error; ground at 1.3%."),
                    L("通过量产 HOMO 测试；LKA、LKW 功能测试通过。",
                      "Passed production HOMO testing; LKA and LKW feature tests passed."),
                    L("支撑 ET7 SUD & NAD 路测与量产部署。",
                      "Supported ET7 SUD & NAD road testing and mass-production deployment."),
                ],
            },
        ],
    },
    {
        "company": L("驭势科技（北京）有限公司", "UISEE Technology (Beijing)"),
        "company_short": "UISEE",
        "role": L("高级研究员 · 南京 AI 研究院", "Senior Researcher · Nanjing AI Research Institute"),
        "when": L("2019.05 — 2020.04", "2019.05 — 2020.04"),
        "scope": L("2019 年国内首家 Shadow Mode 数据闭环 + IL-based E2E 上车",
                   "2019: first in China to deploy a Shadow-Mode data loop + IL-based E2E on-vehicle"),
        "summary": L(
            "面向自动驾驶的模仿学习、强化学习、数据生成与仿真，以及 3D 目标检测与跟踪联合。",
            "Imitation learning, RL, data generation and simulation for AD, plus joint 3D detection and tracking.",
        ),
        "sections": [
            {
                "label": L("研究与交付", "Research and delivery"),
                "items": [
                    L("2019 年国内第一家实现 Shadow Mode 数据闭环系统 + IL-based E2E 上车。",
                      "In 2019, the first team in China to deploy a Shadow-Mode data closed loop plus IL-based E2E "
                      "on-vehicle."),
                    L("面向 AD 的 imitation learning、RL、数据生成与仿真。",
                      "Imitation learning, RL, data generation and simulation for AD."),
                    L("3D 目标检测与跟踪联合。", "Joint 3D object detection and tracking."),
                    L("模仿学习：以 world model、GAIL、InfoGAIL 为基线并加入动作预测模块，仿真平台取得 970 的驾驶得分；"
                      "动作预测模块上真车，在封闭道路完成相对导航。",
                      "Imitation learning: world models, GAIL and InfoGAIL as baselines plus an action-prediction "
                      "module, reaching a driving score of 970 in simulation; the module then went onto a real car for "
                      "closed-road relative navigation."),
                    L("单目标跟踪：用特征解耦把前景信息从特征中分离再做相关运算，OTB100 精度 0.389、VOT 0.549。",
                      "Single-object tracking: disentangled foreground features before correlation — 0.389 on OTB100, "
                      "0.549 on VOT."),
                    L("多目标跟踪与视频目标检测：用形变卷积预测帧间形变再连接检测；KITTI 车辆 AP 0.628（基线 0.560）、"
                      "Waymo 0.663（基线 0.626）。",
                      "Multi-object tracking and video object detection: deformable convolution predicting inter-frame "
                      "deformation to link detections; KITTI vehicles AP 0.628 (baseline 0.560), Waymo 0.663 (0.626)."),
                ],
            },
        ],
    },
]
