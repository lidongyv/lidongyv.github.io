# -*- coding: utf-8 -*-
"""Publications, patents and project catalogues."""

from .data_core import L

# ---------------------------------------------------------------------------
# Peer-reviewed publications (all links verified — see _rebuild_work/papers.md)
# `me` marks the author position that belongs to Lidong Yu.
# ---------------------------------------------------------------------------
PAPERS = [
    {
        "id": "illu25",
        "title": "3D Visual Illusion Depth Estimation",
        "title_zh": "3D 视错觉深度估计",
        "venue": "NeurIPS",
        "year": "2025",
        "badges": [],
        "authors": [
            ("Chengtang Yao", 0), ("Zhidan Liu", 0), ("Jiaxi Zeng", 0),
            ("Lidong Yu", 1), ("Yuwei Wu", 0), ("Yunde Jia", 0),
        ],
        "paper": "https://arxiv.org/abs/2505.13061",
        "paper_label": "arXiv:2505.13061",
        "code": "https://github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation",
        "project": "",
        "problem": L(
            "3D 视错觉（把平面伪装成立体的绘画、屏幕、镜面）能骗过人类视觉，它是否同样骗过机器的深度估计？"
            "单目、双目、多视图方法在幻视场景下会错到什么程度，能否用视觉语言模型的常识来纠偏？",
            "3D visual illusions — paintings, screens and mirrors that disguise a flat surface as a solid object — fool human "
            "vision. Do they fool machine depth estimation too? How badly do monocular, stereo and multi-view methods fail on "
            "illusion scenes, and can the common sense of a vision-language model correct them?",
        ),
        "method": L(
            "构建约 3k 场景、20 万张图像的 3D-Visual-Illusion 数据集（生成数据 + RealSense L515 实采双目数据，"
            "SAM 加人工标注幻视区域掩码）；提出 VLM 常识引导的单目—双目融合框架，"
            "由视觉语言模型判断何处该信单目、何处该信视差，自适应融合两路深度。",
            "Built a 3D-Visual-Illusion dataset of ~3k scenes and 200k images (synthetic plus real stereo captured with an "
            "Intel RealSense L515, illusion-region masks from SAM plus human annotation); proposed a VLM-guided "
            "monocular–stereo fusion framework in which the vision-language model decides where to trust the monocular prior "
            "and where to trust disparity, fusing the two depth sources adaptively.",
        ),
        "result": L(
            "SOTA 单目模型在真实幻视区域大幅失真（DepthPro+align bad3 34.98%、AbsRel 0.09；Marigold bad3 59.67%），"
            "双目模型同样被镜面 / 透明幻视欺骗（RAFT-Stereo bad3 13.20%、EPE 1.62）。"
            "该融合方法在幻视区域 EPE 1.77 / bad5 3.60，同时在中规场景 Middlebury zero-shot 上取得最低 EPE 1.50"
            "（RAFT-Stereo 1.92、Selective-RAFT 2.34、StereoAnything 2.89）。",
            "SOTA monocular models break down inside real illusion regions (DepthPro+align bad3 34.98%, AbsRel 0.09; "
            "Marigold bad3 59.67%), and stereo models are equally fooled by mirrors and transparency (RAFT-Stereo bad3 "
            "13.20%, EPE 1.62). The fusion method reaches EPE 1.77 / bad5 3.60 inside illusion regions while taking the "
            "lowest zero-shot EPE 1.50 on standard Middlebury (RAFT-Stereo 1.92, Selective-RAFT 2.34, StereoAnything 2.89).",
        ),
        "why": L(
            "数据集 + 评测 + 方法 + 开源权重一次交付，并明确指出基础模型在安全相关场景的系统性失效——"
            "这是自动驾驶感知负责人最该有的判断力。",
            "One delivery containing dataset, benchmark, method and open weights — with an explicit statement of where "
            "foundation models systematically fail in safety-relevant scenes. That is exactly the judgement an AD "
            "perception lead needs.",
        ),
        "tags": ["Depth estimation", "VLM", "Dataset", "Illusion"],
    },
    {
        "id": "div25",
        "title": "Diving into the Fusion of Monocular Priors for Generalized Stereo Matching",
        "title_zh": "深入单目先验融合：面向泛化的立体匹配",
        "venue": "ICCV",
        "year": "2025",
        "badges": ["oral"],
        "authors": [
            ("Chengtang Yao", 0), ("Lidong Yu", 1), ("Zhidan Liu", 0),
            ("Jiaxi Zeng", 0), ("Yuwei Wu", 0), ("Yunde Jia", 0),
        ],
        "paper": "https://arxiv.org/abs/2505.14414",
        "paper_label": "arXiv:2505.14414",
        "alt_paper": "https://openaccess.thecvf.com/content/ICCV2025/html/Yao_Diving_into_the_Fusion_of_Monocular_Priors_for_Generalized_Stereo_ICCV_2025_paper.html",
        "alt_paper_label": "CVF",
        "code": "https://github.com/YaoChengTang/Diving-into-the-Fusion-of-Monocular-Priors-for-Generalized-Stereo-Matching",
        "project": "",
        "problem": L(
            "把视觉基础模型（DepthAnything V2 等）的无偏单目先验融进立体匹配，能改善遮挡与非朗伯面等病态区域，"
            "但直接融合有三个硬问题：相对深度与绝对视差不对齐；迭代更新中对视差过自信导致局部最优；"
            "前几轮噪声视差会误导融合。",
            "Fusing the unbiased monocular priors of visual foundation models (DepthAnything V2 and friends) into stereo "
            "matching helps in occlusions and non-Lambertian surfaces, but naive fusion hits three hard problems: relative "
            "depth does not align with absolute disparity; iterative updates become over-confident and fall into local "
            "optima; and noisy early-iteration disparities mislead the fusion.",
        ),
        "method": L(
            "用二值局部排序图（binary local ordering map）把深度图转成二值相对表示，统一相对与绝对深度；"
            "同一排序图用来重加权初始视差更新，抑制局部最优与早期噪声；"
            "把单目深度到视差的最终融合建模为配准问题，用逐像素线性回归模块做全局自适应对齐。",
            "A binary local ordering map converts the depth map into a binary relative representation that unifies relative "
            "and absolute depth; the same map re-weights the initial disparity update to suppress local optima and early "
            "noise; and the final monocular-depth-to-disparity fusion is formulated as a registration problem solved by a "
            "per-pixel linear regression module for globally adaptive alignment.",
        ),
        "result": L(
            "从 SceneFlow 泛化到 Middlebury / ETH3D 误差降低一半以上，Booster 上 Bad-x.0 提升约 10 个点，"
            "且不使用额外双目数据或数据增广；效率几乎不降。DrivingStereo 雨天 EPE 1.29"
            "（RAFT-Stereo 1.80、IGEV 2.32、Selective-IGEV 2.22）。",
            "Generalizing from SceneFlow to Middlebury / ETH3D cuts error by more than half and improves Booster Bad-x.0 by "
            "about 10 points without extra stereo data or augmentation, at essentially unchanged runtime. DrivingStereo "
            "rainy EPE is 1.29 versus 1.80 (RAFT-Stereo), 2.32 (IGEV) and 2.22 (Selective-IGEV).",
        ),
        "why": L(
            "ICCV 2025 Oral，也是「基础模型先验如何真正落地到几何任务」的范本：训练、评测、可视化全链路可复现，"
            "对做端到端感知的团队可以直接迁移。",
            "An ICCV 2025 Oral and a template for actually landing foundation-model priors in a geometric task: the "
            "training, evaluation and visualization pipeline is fully reproducible and directly transferable to "
            "end-to-end perception teams.",
        ),
        "tags": ["Stereo matching", "Foundation models", "Generalization"],
    },
    {
        "id": "lane23",
        "title": "Sparse Point Guided 3D Lane Detection",
        "title_zh": "稀疏点引导的 3D 车道检测",
        "venue": "ICCV",
        "year": "2023",
        "badges": [],
        "authors": [("Chengtang Yao", 0), ("Lidong Yu", 1), ("Yuwei Wu", 0), ("Yunde Jia", 0)],
        "paper": "https://openaccess.thecvf.com/content/ICCV2023/html/Yao_Sparse_Point_Guided_3D_Lane_Detection_ICCV_2023_paper.html",
        "paper_label": "CVF Open Access",
        "code": "https://github.com/YaoChengTang/Sparse-Point-Guided-3D-Lane-Detection",
        "project": "",
        "problem": L(
            "3D 车道检测通常要建立前视图与 BEV 的稠密对应，但车道只占极小比例，绝大部分对应是冗余背景——"
            "稠密高分辨率对应因此成为算力瓶颈。",
            "3D lane detection usually builds a dense front-view ↔ BEV correspondence, but lanes occupy a tiny fraction of "
            "the image and most of that correspondence is redundant background — so dense high-resolution correspondence "
            "becomes the compute bottleneck.",
        ),
        "method": L(
            "由粗到细两阶段：粗阶段在极低分辨率下建立「稠密但便宜」的前视图—BEV 对应得到粗车道；"
            "细阶段沿粗车道采样稀疏点，从高分辨率前视图特征图上取局部特征，在 BEV 空间由低到高分层次精化 3D 车道。",
            "A coarse-to-fine two-stage design: the coarse stage builds a dense-but-cheap front-view ↔ BEV correspondence at "
            "very low resolution to get coarse lanes; the fine stage samples sparse points along those lanes, pulls local "
            "features from the high-resolution front-view feature map and refines the 3D lanes hierarchically in BEV space.",
        ),
        "result": L(
            "OpenLane 验证集 F1 52.3%（PersFormer 50.5%）；分场景统计整体 F1 53.7，比当时 SOTA 高 3.2 分，"
            "困难场景最高 +6.2 分；同时显存几乎减半、推理速度约 2 倍。",
            "F1 52.3% on the OpenLane validation split (PersFormer 50.5%); per-scene analysis gives overall F1 53.7, "
            "3.2 points above the SOTA at the time and up to +6.2 on hard scenes — with roughly half the memory and about "
            "2× the inference speed.",
        ),
        "why": L(
            "「用稀疏性换算力」的典型工业级设计，直接对应量产车上车道与道路结构感知的算力预算约束。",
            "A textbook industrial design that trades sparsity for compute, directly matching the compute budget "
            "constraints of production lane and road-structure perception.",
        ),
        "tags": ["3D lane detection", "Efficiency", "BEV"],
    },
    {
        "id": "pcv23",
        "title": "Parameterized Cost Volume for Stereo Matching",
        "title_zh": "参数化代价体立体匹配",
        "venue": "ICCV",
        "year": "2023",
        "badges": [],
        "authors": [
            ("Jiaxi Zeng", 0), ("Chengtang Yao", 0), ("Lidong Yu", 1),
            ("Yuwei Wu", 0), ("Yunde Jia", 0),
        ],
        "paper": "https://openaccess.thecvf.com/content/ICCV2023/html/Zeng_Parameterized_Cost_Volume_for_Stereo_Matching_ICCV_2023_paper.html",
        "paper_label": "CVF Open Access",
        "code": "https://github.com/JiaxiZeng/Parameterized-Cost-Volume-for-Stereo-Matching",
        "project": "",
        "problem": L(
            "大视差范围让立体匹配的计算量爆炸；动态代价体只关注局部视差空间、缺乏全局视野，"
            "需要很多次迭代才能逼近真值。",
            "Large disparity ranges explode the cost of stereo matching; dynamic cost volumes only look at a local "
            "disparity neighbourhood and lack global view, so they need many iterations to approach the truth.",
        ),
        "method": L(
            "发现动态代价体近似于「固定小方差的单高斯」，于是用多高斯分布参数化整个视差空间："
            "每个像素的视差分布由权重、均值、方差表示，均值与方差用于采样视差候选并计算代价，权重与均值用于输出视差；"
            "参数通过 JS 散度优化，实现为前馈模块中的梯度下降更新。",
            "Observing that a dynamic cost volume approximates a single fixed-small-variance Gaussian, the work "
            "parameterizes the whole disparity space with a mixture of Gaussians: each pixel's disparity distribution is "
            "given by weights, means and variances; means and variances sample disparity candidates and compute cost, "
            "weights and means produce the output. Parameters are optimized through JS divergence, implemented as gradient "
            "descent inside a feed-forward module.",
        ),
        "result": L(
            "SceneFlow EPE 0.62、单帧 62 ms（RAFT-Stereo 0.65 / 254 ms，约 4 倍加速）；KITTI 2015 上比 RAFT-Stereo 快 7 倍、"
            "比 CREStereo 只差 0.24% D1-all 却快 7–8 倍；Middlebury 上比 RAFT-Stereo 快近 15 倍；"
            "Booster Bad2 27.41% 对 RAFT-Stereo 38.66%。",
            "SceneFlow EPE 0.62 at 62 ms per frame (RAFT-Stereo 0.65 / 254 ms — about 4× faster); on KITTI 2015 it is 7× "
            "faster than RAFT-Stereo and only 0.24% D1-all behind CREStereo while being 7–8× faster; on Middlebury nearly "
            "15× faster than RAFT-Stereo; Booster Bad2 27.41% vs 38.66%.",
        ),
        "why": L(
            "把精度—速度帕累托前沿整体往前推了一截，是实时双目深度可以直接替换的模块化方案。",
            "Pushes the accuracy–speed Pareto frontier forward as a whole, and is a drop-in module for real-time stereo depth.",
        ),
        "tags": ["Stereo matching", "Cost volume", "Real-time"],
    },
    {
        "id": "fog22",
        "title": "FoggyStereo: Stereo Matching With Fog Volume Representation",
        "title_zh": "FoggyStereo：用雾体积表示做立体匹配",
        "venue": "CVPR",
        "year": "2022",
        "badges": [],
        "authors": [("Chengtang Yao", 0), ("Lidong Yu", 1)],
        "paper": "https://openaccess.thecvf.com/content/CVPR2022/html/Yao_FoggyStereo_Stereo_Matching_With_Fog_Volume_Representation_CVPR_2022_paper.html",
        "paper_label": "CVF Open Access",
        "code": "https://github.com/YaoChengTang/FoggyStereo-Stereo-Matching-with-Fog-Volume-Representation",
        "project": "https://yaochengtang.github.io/FoggyStereo-Stereo-Matching-with-Fog-Volume-Representation/",
        "problem": L(
            "雾天散射会模糊图像、让匹配产生歧义；已有方法把雾当噪声先去掉，但雾本身其实携带深度线索。",
            "Fog scattering blurs images and makes matching ambiguous. Prior work removes fog as if it were noise — but fog "
            "itself carries depth cues.",
        ),
        "method": L(
            "从渲染视角「反向」利用大气散射模型：对每个视差候选对应的深度，把该深度范围内的雾去掉重新渲染；"
            "渲染图越清晰说明该深度越接近真值。把不同深度渲染出的图沿通道堆叠成雾体积，与代价体融合，纠正雾导致的匹配歧义。",
            "Uses the atmospheric scattering model in reverse, from the rendering viewpoint: for each disparity candidate's "
            "depth, remove the fog within that depth range and re-render. The clearer the re-rendered image, the closer that "
            "depth is to the truth. Images rendered at different depths are stacked along channels into a fog volume and "
            "fused with the cost volume to disambiguate fog-corrupted matching.",
        ),
        "result": L(
            "雾天场景较当时 SOTA 提升 10%~30%，清晰场景性能基本持平；KITTI 2012 合成雾上比 DeepPruner 提升 30%、"
            "比 PSMNet 提升约 10%；SceneFlow 雾天上相比 DeepPruner 精度提升近 3 倍。",
            "Improves 10–30% over the SOTA of the time on foggy scenes while holding performance on clear ones; on KITTI "
            "2012 synthetic fog it is 30% better than DeepPruner and ~10% better than PSMNet, and on SceneFlow fog nearly "
            "3× more accurate than DeepPruner.",
        ),
        "why": L(
            "只用一对双目图像、不需要额外雾天监督，就把「恶劣天气」从减分项变成信息源——量产感知里典型的低成本鲁棒性思路。",
            "With a single stereo pair and no extra foggy supervision it turns bad weather from a penalty into a source of "
            "information — the classic low-cost robustness play in production perception.",
        ),
        "tags": ["Stereo matching", "Adverse weather", "Robustness"],
    },
    {
        "id": "aaai18",
        "title": "Deep Stereo Matching With Explicit Cost Aggregation Sub-Architecture",
        "title_zh": "显式代价聚合子架构的深度立体匹配",
        "venue": "AAAI",
        "year": "2018",
        "badges": ["first", "spotlight"],
        "authors": [("Lidong Yu", 1), ("Yucheng Wang", 0), ("Yuwei Wu", 0), ("Yunde Jia", 0)],
        "paper": "https://arxiv.org/abs/1801.04065",
        "paper_label": "arXiv:1801.04065",
        "alt_paper": "https://aaai.org/papers/12267-deep-stereo-matching-with-explicit-cost-aggregation-sub-architecture/",
        "alt_paper_label": "AAAI",
        "code": "https://github.com/lidongyv/RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture",
        "project": "/project1.html",
        "problem": L(
            "深度立体匹配的研究多集中在特征提取与相似度度量，而同样关键的代价聚合长期被忽视"
            "（多靠固定窗口或隐式网络）。",
            "Research on deep stereo matching concentrated on feature extraction and similarity measurement, while the "
            "equally critical cost aggregation step was left to fixed windows or implicit networks.",
        ),
        "method": L(
            "把代价聚合重构成「聚合提案的生成 + 选择」的学习问题，用双流子架构实现：一条流生成候选聚合结果，另一条流做选择；"
            "选择依据来自轻量卷积网络提取的低层结构信息。这样给代价聚合引入全局视野，"
            "纠正匹配代价计算阶段局部视野造成的错误。",
            "Reframes cost aggregation as a learning problem of generating and then selecting aggregation proposals, "
            "implemented as a two-stream sub-architecture: one stream generates candidate aggregations, the other selects "
            "among them using low-level structural cues from a lightweight CNN. This gives cost aggregation a global view "
            "and corrects errors caused by the local receptive field of cost computation.",
        ),
        "result": L(
            "SceneFlow 上 error>1px 8.93%（GC-Net 11.3%）、error>3px 5.62%（7.2%）、MAE 1.75 px（2.21 px）；"
            "KITTI 2015 D1-all 2.79%（GC-Net 2.87%、PSMNet 2.38%）。",
            "On SceneFlow: error>1px 8.93% (GC-Net 11.3%), error>3px 5.62% (7.2%), MAE 1.75 px (2.21 px); KITTI 2015 "
            "D1-all 2.79% (GC-Net 2.87%, PSMNet 2.38%).",
        ),
        "why": L(
            "第一作者论文，AAAI spotlight，是把代价聚合显式学习化的早期代表工作之一——"
            "可以看出我从读研阶段就在啃立体匹配里最不「性感」却最影响精度的环节。",
            "First-author work, AAAI spotlight, and one of the early papers to make cost aggregation explicitly learned — "
            "evidence that since my master's I have been working on the least glamorous but most accuracy-critical part of "
            "stereo matching.",
        ),
        "tags": ["Stereo matching", "First author", "Cost aggregation"],
    },
    {
        "id": "ca20",
        "title": "Content-Aware Inter-Scale Cost Aggregation for Stereo Matching",
        "title_zh": "内容感知的跨尺度代价聚合立体匹配",
        "venue": "arXiv",
        "year": "2020",
        "badges": ["preprint"],
        "authors": [
            ("Chengtang Yao", 0), ("Yunde Jia", 0), ("Huijun Di", 0),
            ("Yuwei Wu", 0), ("Lidong Yu", 1),
        ],
        "paper": "https://arxiv.org/abs/2006.03209",
        "paper_label": "arXiv:2006.03209",
        "code": "",
        "project": "",
        "problem": L(
            "多尺度代价聚合为了拿上下文把代价体下采样，上采样回细尺度时必然丢细节；"
            "如何在上采样过程中既恢复细节又聚合跨尺度信息？",
            "Multi-scale cost aggregation downsamples the cost volume to gain context, then inevitably loses detail when "
            "upsampling back. How can upsampling both restore detail and aggregate cross-scale information?",
        ),
        "method": L(
            "学习内容感知的动态滤波权重，对细尺度代价体上每个位置自适应地从粗代价体聚合信息；"
            "并提出分解策略——先由两尺度特征图学 2D 相似度，再据此构造 3D 滤波权重，"
            "把 3D 空间—视差联合聚合拆成「1D 视差 + 2D 空间」两步，大幅降低 3D 卷积算力。",
            "Learns content-aware dynamic filter weights so every position in the fine-scale cost volume adaptively "
            "aggregates from the coarse one; a decomposition strategy first learns a 2D similarity from the two feature "
            "maps and then constructs the 3D filter weights from it, splitting joint 3D space–disparity aggregation into "
            "1D disparity plus 2D space and cutting 3D-convolution cost sharply.",
        ),
        "result": L(
            "SceneFlow EPE 0.57，为当时最优（约为 PSMNet 基线的两倍好）；KITTI 2015 上接近 SOTA。"
            "消融显示把 PSMNet 的双线性 / 反卷积上采样换成该方法，EPE 分别改善 0.48 / 0.33。",
            "SceneFlow EPE 0.57, state of the art at the time and roughly twice as good as the PSMNet baseline; close to "
            "SOTA on KITTI 2015. Ablations replacing PSMNet's bilinear / deconvolutional upsampling with this method "
            "improve EPE by 0.48 / 0.33 respectively.",
        ),
        "why": L(
            "把 3D 代价聚合拆成「1D 视差 + 2D 空间」的思路，对后续高效代价聚合设计有直接参考价值。",
            "Splitting 3D cost aggregation into 1D disparity plus 2D space remains a directly useful idea for efficient "
            "cost-aggregation design.",
        ),
        "tags": ["Stereo matching", "Multi-scale", "Dynamic filter"],
    },
    {
        "id": "drl23",
        "title": "DRL-Based Trajectory Tracking for Motion-Related Modules in Autonomous Driving",
        "title_zh": "基于深度强化学习的轨迹跟踪：面向自动驾驶运动模块",
        "venue": "arXiv",
        "year": "2023",
        "badges": ["technical report"],
        "authors": [("Yinda Xu", 0), ("Lidong Yu", 1)],
        "paper": "https://arxiv.org/abs/2308.15991",
        "paper_label": "arXiv:2308.15991",
        "code": "https://github.com/MARMOTatZJU/drl-based-trajectory-tracking",
        "project": "",
        "problem": L(
            "轨迹跟踪是规划、控制等运动相关模块的基础例程；传统方法对场景与动力学模型做强假设，"
            "真实场景一变就不鲁棒。",
            "Trajectory tracking is a basic routine for planning and control modules; classical methods make strong "
            "assumptions about scenes and dynamics models and lose robustness as soon as reality deviates.",
        ),
        "method": L(
            "用深度强化学习直接学跟踪策略，model-free、data-driven，靠深度网络的表示能力加 RL 的探索获得鲁棒性与精度；"
            "策略可与多种动力学模型组合（自行车模型等），并可与 iLQR 后优化串联。",
            "Learns the tracking policy directly with deep reinforcement learning — model-free and data-driven, gaining "
            "robustness and accuracy from representation power plus RL exploration. The policy composes with several "
            "dynamics models (bicycle model and others) and can be chained with iLQR post-optimization.",
        ),
        "result": L(
            "初始速度 5 m/s、自行车模型、优化迭代 0 步时平均跟踪误差 0.0981，对比纯跟踪 + iLQR 的 0.2555（相对降低 61.6%）；"
            "初始速度越高（20 / 30 m/s）增益越明显；对噪声鲁棒、可适配多种动力学模型。",
            "At 5 m/s with a bicycle model and zero optimization iterations, mean tracking error is 0.0981 versus 0.2555 "
            "for pure pursuit + iLQR — a 61.6% relative reduction; the gain grows at higher initial speeds (20 / 30 m/s). "
            "Robust to noise and adaptable across dynamics models.",
        ),
        "why": L(
            "少数从感知跨到规控、并把「RL 策略 + 经典优化器」工程化组合讲清楚的公开工作，对做端到端自动驾驶的团队是很好的参考样本。",
            "One of the few public works that crosses from perception into planning and control and explains how to "
            "combine an RL policy with a classical optimizer in engineering terms.",
        ),
        "tags": ["Reinforcement learning", "Trajectory tracking", "Planning"],
    },
]

# ---------------------------------------------------------------------------
# Unpublished manuscripts / submission archives (from the legacy project pages).
# Honesty matters more than volume at a job search: these are NOT publications.
# ---------------------------------------------------------------------------
MANUSCRIPTS = [
    {
        "title": "Explicit Context Mapping for Stereo Matching",
        "title_zh": "显式上下文映射立体匹配",
        "status": L("未发表手稿（CVPR'19 / ICCV'19 投稿稿，稿本匿名）", "Unpublished manuscript (CVPR'19 / ICCV'19 submissions, anonymized)"),
        "page": "/project2.html",
        "pdfs": [("cvpr19.pdf", "CVPR'19 投稿版", "CVPR'19 submission"), ("iccv2019.pdf", "ICCV'19 投稿版", "ICCV'19 submission")],
        "code": "https://github.com/lidongyv/Explicit-Context-Mapping-for-Stereo-Matching",
        "summary": L(
            "把低分辨率到高分辨率的视差上映射写成贝叶斯推断，用 HR 与 LR 特征向量的相似度构造显式上下文映射矩阵，"
            "代替双线性插值与反卷积做上采样。",
            "Formulates low-to-high-resolution disparity up-mapping as Bayesian inference and builds an explicit context "
            "mapping matrix from HR/LR feature similarity, replacing bilinear interpolation and deconvolution.",
        ),
    },
    {
        "title": "Learning-based Region Support for Stereo Matching",
        "title_zh": "可学习的区域支撑立体匹配",
        "status": L("未发表手稿（CVPR'18 / ECCV'19 投稿稿，稿本匿名）", "Unpublished manuscript (CVPR'18 / ECCV'19 submissions, anonymized)"),
        "page": "/project3.html",
        "pdfs": [("interpreatble cvpr18.pdf", "CVPR'18 投稿版", "CVPR'18 submission"), ("interpretable eccv19.pdf", "ECCV'19 投稿版", "ECCV'19 submission")],
        "code": "https://github.com/lidongyv/Interpretable-stereo-matching-with-region-support",
        "summary": L(
            "用单目深度推断网络生成区域支撑（深度相近像素构成的区域），作为代价聚合与视差精化的显式引导，"
            "让遮挡与弱纹理区域的错误可被解释。",
            "Uses a monocular depth network to produce region support (regions of similar depth) as explicit guidance for "
            "cost aggregation and disparity refinement, making errors in occluded and textureless regions explainable.",
        ),
    },
    {
        "title": "Region-Support Depth Inference From a Single Image",
        "title_zh": "基于区域支撑的单图像深度推断",
        "status": L("未发表手稿（NeurIPS / CVPR'19 投稿稿，稿本匿名）", "Unpublished manuscript (NeurIPS / CVPR'19 submissions, anonymized)"),
        "page": "/project4.html",
        "pdfs": [("nips.pdf", "NeurIPS 投稿版", "NeurIPS submission"), ("nips-cvpr19.pdf", "CVPR'19 投稿版", "CVPR'19 submission")],
        "code": "https://github.com/lidongyv/Monocular-depth-esitimation-with-region-support-cvpr",
        "summary": L(
            "用区域支撑作为单目深度推断的引导：生成模块用金字塔单元从 RGB 确定区域支撑，"
            "精化模块把区域支撑与 RGB 拼接成推理引导，做由粗到细的迭代深度精化。",
            "Region support guides monocular depth inference: a generation module fixes the region support from RGB with "
            "pyramid units, and a refinement module concatenates support with RGB into inference guidance for "
            "coarse-to-fine iterative depth refinement.",
        ),
    },
    {
        "title": "Deep Stereo Matching With Regularization of Pixel-Set Knowledge",
        "title_zh": "像素集知识正则化的深度立体匹配",
        "status": L("未发表手稿（TPAMI 格式，单作者）", "Unpublished manuscript (TPAMI format, sole author)"),
        "page": "/project3.html",
        "pdfs": [("interpretable tpami.pdf", "TPAMI 手稿", "TPAMI manuscript")],
        "code": "",
        "summary": L(
            "提出像素集知识作为正则化：把像素组织成判别性像素集与代表性像素集等先验，"
            "只对代表性像素做匹配、其余像素由集合内共享的代价与视差约束，兼顾细节与冗余计算。",
            "Proposes pixel-set knowledge as regularization: pixels are organized into discriminative and representative "
            "pixel sets, only representative pixels are matched and the rest inherit shared cost and disparity constraints — "
            "keeping detail while cutting redundant computation.",
        ),
    },
]

# ---------------------------------------------------------------------------
# Patents
# ---------------------------------------------------------------------------
PATENTS = [
    {"title": L("三维重建方法、装置、系统和存储介质", "3D reconstruction method, apparatus, system, and storage medium"), "no": "CN201980002779.1"},
    {"title": L("图像预测及车辆行为规划方法、装置、系统和存储介质", "Image prediction and vehicle behavior planning method, apparatus, system, and storage medium"), "no": "CN202010196263.X"},
    {"title": L("多目标跟踪方法、装置、系统和存储介质", "Multi-object tracking method, apparatus, system, and storage medium"), "no": "CN201911334843.4"},
    {"title": L("单目标跟踪方法、装置、系统和存储介质", "Single-object tracking method, apparatus, system, and storage medium"), "no": "CN201911194289.4"},
    {"title": L("生成对抗网络的训练方法和系统、电子设备和存储介质", "Training method and system for generative adversarial networks, electronic device, and storage medium"), "no": "CN201980002600.2"},
]

# ---------------------------------------------------------------------------
# Research / engineering project pages that already exist on the site.
# These keep the legacy URLs alive and are the "deep dive" layer.
# ---------------------------------------------------------------------------
PROJECT_PAGES = [
    {
        "id": "project9",
        "href": "/project9.html",
        "kind": L("已发表 · CVPR 2022", "Published · CVPR 2022"),
        "kind_cls": "tag--venue",
        "title": L("雾体积表示的立体匹配", "Stereo matching with a fog volume representation"),
        "desc": L(
            "把雾从「噪声」变成「信息源」：逆向大气散射、按候选深度去雾重渲染，把渲染图堆成雾体积与代价体融合。"
            "雾天场景较当时 SOTA 提升 10%–30%，清晰场景持平。",
            "Turns fog from noise into a source of information: reverse the atmospheric scattering, re-render with the "
            "fog removed at each candidate depth, and stack the renders into a fog volume fused with the cost volume. "
            "10–30% better than the SOTA of the time on foggy scenes, comparable on clear ones.",
        ),
        "img": "/assets/img/project9/fig1.png",
        "year": "2022",
        "tags": ["Stereo matching", "Fog volume", "CVPR"],
    },
    {
        "id": "project1",
        "href": "/project1.html",
        "kind": L("已发表 · AAAI 2018", "Published · AAAI 2018"),
        "kind_cls": "tag--venue",
        "title": L("显式代价聚合的深度立体匹配", "Deep stereo matching with explicit cost aggregation"),
        "desc": L(
            "把代价聚合重构成「生成候选 + 选择」的双流子架构，用低层结构信息做全局视野的聚合决策。"
            "成果为 AAAI 2018 spotlight 论文（第一作者）。",
            "Reframes cost aggregation as a generate-and-select two-stream sub-architecture, using low-level structural "
            "cues for a globally-aware aggregation decision. Published as an AAAI 2018 spotlight (first author).",
        ),
        "img": "/assets/img/project1/overview.png",
        "year": "2017 — 2018",
        "tags": ["Stereo matching", "Cost aggregation", "AAAI"],
    },
    {
        "id": "project5",
        "href": "/project5.html",
        "kind": L("工程报告 · 驭势科技", "Engineering report · UISEE"),
        "kind_cls": "",
        "title": L("模仿学习做相对导航", "Imitation learning for relative navigation"),
        "desc": L(
            "用 world model、GAIL、InfoGAIL 做基线并加入动作预测模块，在仿真平台取得 970 驾驶得分；"
            "动作预测模块上真车，在封闭道路完成相对导航。",
            "World models, GAIL and InfoGAIL as baselines plus an action-prediction module, reaching a driving score of 970 "
            "in simulation; the module then went onto a real car for closed-road relative navigation.",
        ),
        "img": "/assets/img/project5/new/4.png",
        "year": "2019 — 2020",
        "tags": ["Imitation learning", "GAIL", "Shadow mode"],
    },
    {
        "id": "project8",
        "href": "/project8.html",
        "kind": L("工程报告 · 驭势科技", "Engineering report · UISEE"),
        "kind_cls": "",
        "title": L("视频目标检测与时序聚合", "Video object detection with temporal aggregation"),
        "desc": L(
            "以 RepPoint 为基线，用形变卷积预测帧间形变并把相邻帧信息聚合到当前帧；"
            "KITTI 车辆 AP 0.628（基线 0.560）、Waymo 0.663（基线 0.626）。",
            "RepPoint as baseline with deformable convolution predicting inter-frame deformation and aggregating "
            "neighbouring frames into the current one; KITTI vehicle AP 0.628 (baseline 0.560), Waymo 0.663 (baseline 0.626).",
        ),
        "img": "/assets/img/project8/10.png",
        "year": "2020",
        "tags": ["Video object detection", "Deformable conv", "RepPoints"],
    },
    {
        "id": "project6",
        "href": "/project6.html",
        "kind": L("工程报告 · 驭势科技", "Engineering report · UISEE"),
        "kind_cls": "",
        "title": L("特征解耦的单目标跟踪", "Single-object tracking via feature disentanglement"),
        "desc": L(
            "把前景信息从特征中解耦出来再做相关运算，提升判别力与长时稳定性；"
            "OTB100 精度 0.389、VOT 0.549，长视频与相机抖动下更稳。",
            "Disentangles foreground information before correlation to raise discriminability and long-term stability; "
            "0.389 on OTB100 and 0.549 on VOT, steadier on long videos and under camera vibration.",
        ),
        "img": "/assets/img/project6/6.png",
        "year": "2020",
        "tags": ["Single-object tracking", "Feature disentanglement", "GAN"],
    },
    {
        "id": "project7",
        "href": "/project7.html",
        "kind": L("工程报告 · 驭势科技", "Engineering report · UISEE"),
        "kind_cls": "",
        "title": L("形变预测的多目标跟踪", "Multi-object tracking with deformation prediction"),
        "desc": L(
            "用形变卷积学习两帧间的相对形变，把 T-1 帧检测结果 warp 到 T 帧再与当前帧检测合并；"
            "KITTI 上车辆从 0.419 提升到 0.752、行人从 0.356 提升到 0.817。",
            "Deformable convolution learns the relative deformation between two frames; detections at T-1 are warped to T "
            "and merged with current detections. KITTI vehicles 0.419 → 0.752, pedestrians 0.356 → 0.817.",
        ),
        "img": "/assets/img/project7/0.png",
        "year": "2019 — 2020",
        "tags": ["Multi-object tracking", "Deformable conv", "KITTI"],
    },
    {
        "id": "project2",
        "href": "/project2.html",
        "kind": L("投稿存档 · 未发表", "Submission archive · unpublished"),
        "kind_cls": "tag--mono",
        "title": L("显式上下文映射立体匹配", "Explicit context mapping for stereo matching"),
        "desc": L(
            "把 HR↔LR 视差映射写成贝叶斯推断，用特征相似度构造显式映射矩阵替代双线性 / 反卷积上采样。"
            "两版投稿稿存档于此，未见录用记录。",
            "Formulates HR↔LR disparity mapping as Bayesian inference and replaces bilinear / deconvolutional upsampling "
            "with an explicit similarity-based mapping matrix. Two submission drafts are archived here; no acceptance record.",
        ),
        "img": "/assets/img/project2/illustration2.png",
        "year": "2019",
        "tags": ["Stereo matching", "Up-sampling", "Manuscript"],
    },
    {
        "id": "project3",
        "href": "/project3.html",
        "kind": L("投稿存档 · 未发表", "Submission archive · unpublished"),
        "kind_cls": "tag--mono",
        "title": L("区域支撑的可解释立体匹配", "Interpretable stereo matching with region support"),
        "desc": L(
            "用区域支撑（深度相近像素构成的区域）引导代价聚合与视差精化，让遮挡 / 弱纹理区域的错误可解释。"
            "含 CVPR'18、ECCV'19 投稿稿与 TPAMI 格式扩展手稿。",
            "Region support — regions of similar depth — guides cost aggregation and disparity refinement so errors in "
            "occluded and textureless regions become explainable. Includes CVPR'18 and ECCV'19 drafts plus a TPAMI-format "
            "extension manuscript.",
        ),
        "img": "/assets/img/project3/overview.png",
        "year": "2018 — 2020",
        "tags": ["Stereo matching", "Region support", "Interpretability"],
    },
    {
        "id": "project4",
        "href": "/project4.html",
        "kind": L("投稿存档 · 未发表", "Submission archive · unpublished"),
        "kind_cls": "tag--mono",
        "title": L("区域支撑的单目深度推断", "Region-support monocular depth inference"),
        "desc": L(
            "用区域支撑作为单目深度推断的引导，由粗到细迭代精化；NYU 上相对基线有明显提升。"
            "含 NeurIPS 与 CVPR'19 投稿稿。",
            "Region support guides monocular depth inference with coarse-to-fine iterative refinement, improving clearly "
            "over the baseline on NYU. Includes NeurIPS and CVPR'19 submission drafts.",
        ),
        "img": "/assets/img/project4/overview_cvpr.png",
        "year": "2018 — 2019",
        "tags": ["Monocular depth", "Region support", "Manuscript"],
    },
]

# ---------------------------------------------------------------------------
# FoggyStereo has its own (non-legacy) project page: the repo carried a card in
# _includes/projects.html plus a figure, but no project9.html was ever created.
# ---------------------------------------------------------------------------
FOGGY = {
    "title": "FoggyStereo: Stereo Matching With Fog Volume Representation",
    "title_zh": "FoggyStereo：用雾体积表示做立体匹配",
    "venue": "CVPR 2022",
    "authors": "Chengtang Yao, Lidong Yu",
    "figure": "/assets/img/project9/fig1.webp",
    "figure_cap": L(
        "雾体积（fog volume）的构造：对每个视差候选对应的深度反向渲染去雾，再沿通道堆叠。",
        "Building the fog volume: reverse-render each disparity candidate's depth with the fog removed, "
        "then stack the results along the channel axis.",
    ),
    "problem": L(
        "雾天场景的立体匹配很难：散射效应让图像变模糊，匹配因此产生歧义。以往的做法是把雾当成噪声、"
        "在匹配前先丢掉。我们的出发点相反——雾里其实藏着深度线索。",
        "Stereo matching in fog is hard: scattering blurs the image and makes matching ambiguous. Prior work treats "
        "fog as noise and discards it before matching. We take the opposite view — the fog itself carries depth cues.",
    ),
    "method": L(
        "从渲染视角挖这些线索：逆转大气散射过程，把选定深度范围内的雾去掉重新渲染。渲染图的质量反映所选深度的"
        "正确性——越接近真实深度，渲染图越清晰。对这些渲染图做通道堆叠得到<strong>雾体积</strong>，"
        "再与代价体融合，纠正雾导致的匹配歧义。",
        "We mine those cues from the rendering viewpoint: reverse the atmospheric scattering process and re-render with "
        "the fog inside a selected depth range removed. The quality of the rendered image reflects how correct the "
        "selected depth is — the closer to the true depth, the clearer the render. Stacking these renders along the "
        "channel axis gives a <strong>fog volume</strong>, which is fused with the cost volume to rectify the ambiguity "
        "that fog introduces.",
    ),
    "results": L(
        "雾天场景较当时 SOTA 提升 <strong>10%–30%</strong>，清晰场景性能基本持平。"
        "只用一对双目图像、不需要额外的雾天监督。",
        "Improves <strong>10–30%</strong> over the SOTA of the time on foggy scenes while staying comparable on clear "
        "ones — using a single stereo pair and no extra foggy supervision.",
    ),
    "links": [
        ("https://openaccess.thecvf.com/content/CVPR2022/html/"
         "Yao_FoggyStereo_Stereo_Matching_With_Fog_Volume_Representation_CVPR_2022_paper.html",
         L("CVF 论文页", "CVF Open Access"), "doc"),
        ("https://yaochengtang.github.io/FoggyStereo-Stereo-Matching-with-Fog-Volume-Representation/",
         L("项目主页", "Project page"), "link"),
        ("https://github.com/YaoChengTang/FoggyStereo-Stereo-Matching-with-Fog-Volume-Representation",
         L("代码", "Code"), "github"),
        ("/publications.html", L("在论文列表中查看", "See in publications"), "arrow"),
    ],
    "tags": ["Stereo matching", "Fog volume", "Atmospheric scattering", "Robustness"],
}

# ---------------------------------------------------------------------------
# The engineering projects behind the resume (used on /projects.html)
# ---------------------------------------------------------------------------
PROJECTS_INTRO = L(
    "下面这些是带具体指标、可以直接被追问细节的项目。每一条我都写清了问题、我的动作和结果；"
    "论文与开源代码另有专页。",
    "These are the projects with concrete numbers that survive follow-up questioning. For each one I state the "
    "problem, what I actually did and the result. Papers and open-source code live on their own pages.",
)
