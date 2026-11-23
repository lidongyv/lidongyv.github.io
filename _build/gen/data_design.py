# -*- coding: utf-8 -*-
"""Publishable research projects that live outside this repository.

Source: C:\\Users\\Admin\\Desktop\\design — eight self-authored research sites plus a
pile of caches, scraped conference pages, a paper-PDF corpus (1.14 GB) and two
saved third-party articles. Only the self-authored text/markup is published; see
_build/design.py for the filter and the exclusions.
"""

DESIGN = [
    {
        "slug": "wam-survey",
        "src": "wam-survey-project",
        "title": "世界模型（WAM）综述",
        "title_en": "World-Action Model survey",
        "summary": "把 2024—2026 年的世界模型、驾驶世界模型与 WAM 的工作整理成一条叙事线，并给出能力分类与评测口径。",
        "summary_en": "A single narrative line through the 2024–2026 literature on world models, driving world models "
                      "and world-action models, with a capability taxonomy and an evaluation lens.",
        "date": "2026-11-10",
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
        "date": "2026-11-02",
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
        "date": "2026-10-26",
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
        "date": "2026-10-18",
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
        "date": "2026-10-08",
        "tags": ["蒸馏", "部署"],
        "tags_en": ["Distillation", "Deployment"],
    },
    {
        "slug": "fsd-v15",
        "src": "fsd-v15",
        "title": "FSD V15 拆解",
        "title_en": "Inside FSD V15",
        "summary": "把 FSD V15 的公开信息拆成可讨论的技术组件，并给出自己的判断。",
        "summary_en": "Breaking the public information on FSD V15 into discussable technical components, with my own read.",
        "skip": ["src"],
        "date": "2026-09-28",
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
        "date": "2026-09-15",
        "tags": ["VLA", "RL", "后训练"],
        "tags_en": ["VLA", "RL", "Post-training"],
    },
]
