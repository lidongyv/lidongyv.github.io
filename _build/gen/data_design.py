# -*- coding: utf-8 -*-
"""Projects that are published as browsable subsites under /design/.

Source directories are configured in ``_build/sources.local.json`` (gitignored) —
this file carries only the public metadata. Dates are not written by hand: the
build derives them from the newest source file mtime.

``root``   which source root to read from (see sources.local.json)
``images`` when "webp", screenshots are re-encoded and the html is rewritten
"""

DESIGN = [
    # ---------------------------------------------------------------- design/
    {
        "slug": "wam-survey",
        "src": "wam-survey-project",
        "title": "世界模型（WAM）综述",
        "title_en": "World-Action Model survey",
        "summary": "把 2024—2026 年的世界模型、驾驶世界模型与 WAM 的工作整理成一条叙事线，并给出能力分类与评测口径。",
        "summary_en": "A single narrative line through the 2024-2026 literature on world models, driving world models "
                      "and world-action models, with a capability taxonomy and an evaluation lens.",
        "tags": ["世界模型", "WAM", "综述"],
        "tags_en": ["World models", "WAM", "Survey"],
    },
    {
        "slug": "gs-reading",
        "src": "gs-reading-site",
        "title": "3D Gaussian Splatting 阅读站",
        "title_en": "3D Gaussian Splatting reading site",
        "summary": "CVPR / ICCV / ECCV / NeurIPS 上 3DGS 相关工作的检索、归类与精读笔记。",
        "summary_en": "Retrieval, clustering and close-reading notes for 3DGS work at CVPR / ICCV / ECCV / NeurIPS.",
        "tags": ["3DGS", "论文阅读"],
        "tags_en": ["3DGS", "Paper reading"],
    },
    {
        "slug": "world-model-reading",
        "src": "world-model-reading",
        "title": "世界模型精读笔记",
        "title_en": "World-model close reading",
        "summary": "按主题串联的世界模型论文精读：每篇记问题、方法、关键数字和它在这条线上的位置。",
        "summary_en": "Thematic close readings of world-model papers: problem, method, key numbers and where each paper "
                      "sits on the line.",
        "tags": ["世界模型", "论文阅读"],
        "tags_en": ["World models", "Paper reading"],
    },
    {
        "slug": "embodied-wam",
        "src": "embodied-wam",
        "title": "具身智能与 WAM",
        "title_en": "Embodied AI and WAM",
        "summary": "机器人侧的 VLA / 世界模型工作梳理，以及与自动驾驶落地差异的对照。",
        "summary_en": "VLA and world-model work on the robotics side, set against the differences in landing on "
                      "autonomous driving.",
        "tags": ["具身智能", "VLA", "WAM"],
        "tags_en": ["Embodied AI", "VLA", "WAM"],
    },
    {
        "slug": "distill-research",
        "src": "distill-research",
        "title": "蒸馏与压缩研究",
        "title_en": "Distillation and compression research",
        "summary": "面向车端的模型蒸馏、剪枝与推理加速调研，含具体方案对比。",
        "summary_en": "Distillation, pruning and inference acceleration for in-vehicle deployment, with a scheme-by-scheme "
                      "comparison.",
        "tags": ["蒸馏", "部署"],
        "tags_en": ["Distillation", "Deployment"],
    },
    {
        "slug": "fsd-v15",
        "src": "fsd-v15",
        "skip": ["src"],
        "title": "FSD V15 拆解",
        "title_en": "Inside FSD V15",
        "summary": "把 FSD V15 的公开信息拆成可讨论的技术组件，并给出自己的判断。",
        "summary_en": "Breaking the public information on FSD V15 into discussable technical components, with my own read.",
        "tags": ["FSD", "端到端", "拆解"],
        "tags_en": ["FSD", "End-to-end", "Teardown"],
    },
    {
        "slug": "vla-rl",
        "src": "vla-rl-调研工程",
        "title": "VLA 与强化学习调研",
        "title_en": "VLA and reinforcement learning",
        "summary": "VLA 策略的 RL 后训练路线调研：reward 设计、偏好优化与闭环训练。",
        "summary_en": "A survey of RL post-training routes for VLA policies: reward design, preference optimisation and "
                      "closed-loop training.",
        "tags": ["VLA", "RL", "后训练"],
        "tags_en": ["VLA", "RL", "Post-training"],
    },
    # ---------------------------------------------------------------- demos/
    {
        "slug": "cvpr-awards",
        "root": "demos",
        "src": "paper_reading__cvpr_awards",
        "images": "webp",
        "title": "CV 顶会获奖与 Finalist 论文库",
        "title_en": "Award and finalist papers from the CV conferences",
        "summary": "CVPR / ICCV / ECCV 历年获奖与 finalist 论文的整理：每篇一页，配主图与摘要。",
        "summary_en": "Award-winning and finalist papers from CVPR / ICCV / ECCV, one page each with the main figure and "
                      "abstract.",
        "tags": ["论文阅读", "CV 顶会"],
        "tags_en": ["Paper reading", "CV conferences"],
    },
    {
        "slug": "cosmos-reading",
        "root": "demos",
        "src": "paper_reading__world_model",
        "images": "webp",
        "title": "Cosmos 世界模型教学文档",
        "title_en": "Cosmos world model, explained",
        "summary": "NVIDIA Cosmos 世界模型的原理梳理：扩散基础、条件生成、多视角与实现细节。",
        "summary_en": "How NVIDIA's Cosmos world model works: diffusion basics, conditional generation, multi-view and "
                      "implementation details.",
        "tags": ["世界模型", "Cosmos"],
        "tags_en": ["World models", "Cosmos"],
    },
    {
        "slug": "deepseek-v4-reading",
        "root": "demos",
        "src": "deepseek-v4-reading",
        "images": "webp",
        "title": "DeepSeek-V4 技术报告精读",
        "title_en": "DeepSeek-V4 technical report, read closely",
        "summary": "DeepSeek-V4 技术报告的逐节精读与要点整理。",
        "summary_en": "A section-by-section reading of the DeepSeek-V4 technical report.",
        "tags": ["论文阅读", "LLM"],
        "tags_en": ["Paper reading", "LLM"],
    },
    {
        "slug": "mamba-intro",
        "root": "demos",
        "src": "mamba-intro",
        "images": "webp",
        "title": "Mamba 入门：从 SSM 到选择性扫描",
        "title_en": "Mamba: from state-space models to selective scan",
        "summary": "从状态空间模型讲到 Mamba 的选择性扫描，配公式与示意图。",
        "summary_en": "From state-space models to Mamba's selective scan, with the equations and diagrams.",
        "tags": ["论文阅读", "SSM"],
        "tags_en": ["Paper reading", "SSM"],
    },
    {
        "slug": "vggt-reading",
        "root": "demos",
        "src": "paper_reading__vggt",
        "images": "webp",
        "title": "VGGT 论文分析与后续工作",
        "title_en": "VGGT: paper analysis and follow-ups",
        "summary": "VGGT 的方法拆解，以及后续在几何基础模型方向上的相关工作梳理。",
        "summary_en": "A breakdown of VGGT's method plus the follow-up work on geometry foundation models.",
        "tags": ["论文阅读", "3D"],
        "tags_en": ["Paper reading", "3D"],
    },
]