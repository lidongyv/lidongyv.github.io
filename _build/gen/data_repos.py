# -*- coding: utf-8 -*-
"""Self-created GitHub repositories, grouped by type.

Source: GitHub REST API snapshot 2026-11-23 (see _rebuild_work/github-own-repos.md).
134 public repos in total: 39 created by me, 95 forks. Only the 39 own repos appear
as projects; forks are listed separately and explicitly labelled.
"""

from .data_core import L

# kind values: research | tools | media | finance | archive
REPOS = [
    # ---------------- research ----------------
    {
        "name": "SOT-Projects", "kind": "research", "lang": "Python", "pushed": "2020-09",
        "featured": True,
        "desc": L(
            "单目标跟踪 + 特征解耦，基于 PySOT 改写；配套技术报告与项目页 project6。",
            "Single-object tracking with feature disentanglement, adapted from PySOT; ships with a technical report "
            "and the project6 page.",
        ),
    },
    {
        "name": "Reppoint-Tracking", "kind": "research", "lang": "Jupyter Notebook", "pushed": "2020-09",
        "featured": True,
        "desc": L(
            "把 mmdet 检测器与 RepPoints 表征接成完整的多目标跟踪训练 / 评测框架，体量最大的工程级仓库。",
            "Wires the mmdet detector and RepPoints representation into a complete multi-object-tracking training and "
            "evaluation framework — my largest engineering repo.",
        ),
    },
    {
        "name": "RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture",
        "kind": "research", "lang": "TensorFlow", "pushed": "2020-05",
        "featured": True,
        "desc": L(
            "AAAI 2018 论文的参考实现：显式代价聚合子网络 RAN，可插拔进现有立体匹配网络。",
            "Reference implementation of the AAAI 2018 paper: RAN, an explicit cost-aggregation sub-network that plugs "
            "into existing stereo matchers.",
        ),
    },
    {
        "name": "Monocular-depth-esitimation-with-region-support-cvpr",
        "kind": "research", "lang": "Python", "pushed": "2020-09",
        "featured": True,
        "desc": L(
            "单目深度估计主线成果：区域支撑 + 由粗到细的深度回归网络，region-support 系列的主仓库。",
            "The main line of my monocular depth work: region support plus a coarse-to-fine depth regression network — "
            "the primary repo of the region-support series.",
        ),
    },
    {
        "name": "Explicit-Context-Mapping-for-Stereo-Matching",
        "kind": "research", "lang": "Python", "pushed": "2020-09",
        "featured": True,
        "desc": L(
            "立体匹配中显式建模上下文映射，处理弱纹理与重复纹理区域。",
            "Explicit context mapping for stereo matching, aimed at weak-texture and repetitive-texture regions.",
        ),
    },
    {
        "name": "Interpretable-stereo-matching-with-region-support",
        "kind": "research", "lang": "Python", "pushed": "2020-09",
        "featured": True,
        "desc": L(
            "以像素集合（pixel-set）为单位的可解释立体匹配，强调中间结果可视化。",
            "Interpretable stereo matching at pixel-set granularity, with an emphasis on visualising intermediate results.",
        ),
    },
    {
        "name": "world-pytorch", "kind": "research", "lang": "PyTorch", "pushed": "2019-06",
        "featured": True,
        "desc": L(
            "World Models（Ha & Schmidhuber）的 PyTorch 复现：VAE + RNN 世界模型 + controller。",
            "PyTorch reproduction of World Models (Ha & Schmidhuber): VAE, RNN world model and controller.",
        ),
    },
    {
        "name": "gym-car", "kind": "research", "lang": "Python", "pushed": "2019-08",
        "featured": True,
        "desc": L(
            "自建仿真环境里联调「感知 + 控制器」闭环的试验代码，是从感知走向导航控制的雏形。",
            "Experimental code that closes a perception + controller loop inside a self-built simulator — the seed of "
            "my move from perception into navigation and control.",
        ),
    },
    {
        "name": "3dndiv", "kind": "research", "lang": "Python", "pushed": "2019-11",
        "featured": False,
        "desc": L(
            "用 NDIV loss 学习 3D 表征的实验代码，含数据与权重，是所有自有仓库里体量最大的一个。",
            "Experimental code learning 3D representations with an NDIV loss, including data and weights — the bulkiest "
            "of all my own repos.",
        ),
    },
    {
        "name": "GCNet", "kind": "research", "lang": "Python", "pushed": "2017-11",
        "featured": False,
        "desc": L(
            "复现 GC-Net（几何 + 上下文立体深度估计），证明能读论文落到代码。",
            "A GC-Net reproduction (geometry + context stereo depth), showing I can turn a paper into working code.",
        ),
    },
    {
        "name": "Monocular-depth-esitimation-with-region-support-nips",
        "kind": "research", "lang": "Python", "pushed": "2018-08",
        "featured": False,
        "desc": L(
            "同一思路的早期投稿原型，只保留核心实验代码，是 cvpr 仓库的前身。",
            "Early prototype of the same idea, keeping only the core experiment code — the predecessor of the cvpr repo.",
        ),
    },
    # ---------------- tools ----------------
    {
        "name": "tensorflow_trick", "kind": "tools", "lang": "Python", "pushed": "2017-12",
        "featured": True,
        "desc": L("TensorFlow 使用技巧与踩坑片段集合，是我自有仓库里 star 最高的一个。",
                  "A collection of TensorFlow tips and pitfalls — the most-starred of my own repos."),
    },
    {
        "name": "matlab", "kind": "tools", "lang": "MATLAB", "pushed": "2016-08",
        "featured": False,
        "desc": L("MATLAB 小技巧脚本合集，日常工具箱性质。", "Small MATLAB utility scripts — a day-to-day toolbox."),
    },
    {
        "name": "matlab-rename", "kind": "tools", "lang": "MATLAB", "pushed": "2016-08",
        "featured": False,
        "desc": L("MATLAB 批量「读文件 → 重命名 → 另存」脚本。", "MATLAB batch script for read → rename → save-as."),
    },
    {
        "name": "Python-XML2Dict", "kind": "tools", "lang": "Python", "pushed": "2015-03",
        "featured": False,
        "desc": L("把 XML 转成 Python dict 的小工具。", "A small utility that turns XML into a Python dict."),
    },
    # ---------------- media / knowledge base ----------------
    {
        "name": "Thomas-S-Huang-Papers", "kind": "media", "lang": "—", "pushed": "2020-04",
        "featured": True,
        "desc": L(
            "整理黄煦涛（Thomas S. Huang）31 篇最高引论文的资料库，含 collection / paperreading / proceedings 主题。",
            "A curated library of Thomas S. Huang's 31 most-cited papers, tagged collection / paperreading / proceedings.",
        ),
    },
    # ---------------- archive ----------------
    {"name": "DM-HomeWork-HorseIllness", "kind": "archive", "lang": "Python", "pushed": "2017-04",
     "desc": L("数据挖掘课作业（马病分类）。", "Data-mining coursework (horse-illness classification).")},
    {"name": "DM-SearchForImages", "kind": "archive", "lang": "—", "pushed": "2017-04",
     "desc": L("数据挖掘课期末 tutorial（图像检索）。", "Data-mining final tutorial (image retrieval).")},
    {"name": "cg3DTrans-Final", "kind": "archive", "lang": "C++", "pushed": "2015-08",
     "desc": L("计算机图形学课程大作业：C++ 实现的 3D 变换与渲染。", "Computer-graphics coursework: 3D transforms and rendering in C++.")},
    {"name": "opengl", "kind": "archive", "lang": "C", "pushed": "2015-08",
     "desc": L("OpenGL 模型显示试验。", "OpenGL model-rendering experiments.")},
    {"name": "tooth-control-in-unity", "kind": "archive", "lang": "ASP", "pushed": "2015-08",
     "desc": L("用 Unity 驱动 Maya 牙齿模型的本科项目。", "Undergraduate project driving a Maya tooth model from Unity.")},
    {"name": "LibrarySystem", "kind": "archive", "lang": "C++", "pushed": "2015-08",
     "desc": L("C++ 图书管理系统课程设计。", "Library-management system coursework in C++.")},
    {"name": "TEST_MVC", "kind": "archive", "lang": "Java", "pushed": "2015-08",
     "desc": L("Java MVC 模式练习（书店示例）。", "Java MVC pattern practice (bookstore example).")},
    {"name": "Xinge_back7", "kind": "archive", "lang": "Java", "pushed": "2015-08",
     "desc": L("Java 后端练习项目。", "Java back-end practice project.")},
    {"name": "Robot", "kind": "archive", "lang": "C++", "pushed": "2015-08",
     "desc": L("竞赛机器人控制程序的旧版本。", "Older version of a competition robot control program.")},
    {"name": "six", "kind": "archive", "lang": "C++", "pushed": "2015-03",
     "desc": L("六子棋对弈程序。", "A Connect-six game program.")},
    {"name": "ShipDemo", "kind": "archive", "lang": "HTML", "pushed": "2015-03",
     "desc": L("HTML 船舶演示页。", "An HTML ship demo page.")},
    {"name": "my_rabbitmq_test", "kind": "archive", "lang": "Java", "pushed": "2015-03",
     "desc": L("照官方示例做的 RabbitMQ 练手。", "RabbitMQ practice following the official examples.")},
    {"name": "project", "kind": "archive", "lang": "—", "pushed": "2015-08",
     "desc": L("GIS 课程项目（空仓库）。", "GIS coursework (empty repo).")},
    {"name": "Projects", "kind": "archive", "lang": "—", "pushed": "2016-02",
     "desc": L("VS2013 工程占位（空仓库）。", "VS2013 project placeholder (empty repo).")},
    {"name": "Visual-Studio-2013", "kind": "archive", "lang": "—", "pushed": "2016-02",
     "desc": L("IDE 相关占位（空仓库）。", "IDE placeholder (empty repo).")},
    {"name": "eclipse", "kind": "archive", "lang": "—", "pushed": "2016-02",
     "desc": L("IDE 相关占位（空仓库）。", "IDE placeholder (empty repo).")},
    {"name": "python_sample", "kind": "archive", "lang": "—", "pushed": "2016-02",
     "desc": L("OpenCV3 试验（空仓库）。", "OpenCV3 experiments (empty repo).")},
    {"name": "matlab_sample", "kind": "archive", "lang": "—", "pushed": "2016-02",
     "desc": L("MATLAB 试验（空仓库）。", "MATLAB experiments (empty repo).")},
    {"name": "setup", "kind": "archive", "lang": "—", "pushed": "2015-02",
     "desc": L("建仓测试。", "Repository setup test.")},
    {"name": "test3", "kind": "archive", "lang": "—", "pushed": "2015-02",
     "desc": L("建仓测试。", "Repository setup test.")},
    {"name": "test2", "kind": "archive", "lang": "—", "pushed": "2015-02",
     "desc": L("建仓测试（空仓库）。", "Repository setup test (empty repo).")},
    {"name": "test", "kind": "archive", "lang": "—", "pushed": "2015-02",
     "desc": L("建仓测试（空仓库）。", "Repository setup test (empty repo).")},
    {"name": "3d", "kind": "archive", "lang": "—", "pushed": "2026-01",
     "desc": L("3D 实验占位（空仓库）。", "3D experiment placeholder (empty repo).")},
]

REPO_GROUPS = [
    {"kind": "research", "label": L("研究 · Research", "Research"), "note": L(
        "立体匹配、单目深度、跟踪与世界模型的可运行实现，多数配有报告或项目页。",
        "Runnable implementations across stereo matching, monocular depth, tracking and world models — most with a "
        "report or project page.")},
    {"kind": "tools", "label": L("工具 · Tools", "Tools"), "note": L(
        "日常小工具与速查笔记。", "Everyday utilities and quick-reference notes.")},
    {"kind": "media", "label": L("知识与媒体 · Knowledge & media", "Knowledge & media"), "note": L(
        "论文资料库一类的内容型仓库。", "Content-style repos such as curated paper libraries.")},
    {"kind": "archive", "label": L("归档 · Archive", "Archive"), "note": L(
        "2015—2017 年的课程作业与占位仓库，保留只为历史完整，不作为作品展示。",
        "Coursework and placeholders from 2015–2017, kept only for completeness — not presented as work.")},
]

# ---------------------------------------------------------------------------
# Finance / quant: honest framing. I have no self-created public quant repo;
# what exists are upstream projects I studied and extended locally.
# ---------------------------------------------------------------------------
FINANCE = {
    "title": L("金融与量化：方向探索", "Finance & quant: exploratory track"),
    "lead": L(
        "我对量化交易与「用生成模型做因子挖掘」这条线有持续的个人兴趣，并系统读过下列上游开源项目的源码。"
        "需要说明的是：这些仓库的作者不是我，我是在本地做学习、改造与实验；"
        "我目前没有可公开的自建量化仓库。",
        "I keep a serious personal interest in quantitative trading and in using generative models for factor mining, and I "
        "have read the source of the upstream projects below in depth. To be precise: I did not author these repos — I "
        "studied, extended and experimented with them locally. I do not currently have a public, self-authored quant repo.",
    ),
    "items": [
        {
            "name": "vnpy", "url": "https://github.com/lidongyv/vnpy",
            "upstream": "https://github.com/vnpy/vnpy",
            "desc": L("Python 开源量化交易平台框架。我关注它的网关抽象与事件引擎设计，"
                     "以及如何把实盘接入与回测统一到同一套事件模型。",
                     "An open-source Python quant trading platform. I studied its gateway abstraction and event engine, and "
                     "how live trading and backtesting can share one event model."),
        },
        {
            "name": "AlphaGPT", "url": "https://github.com/lidongyv/AlphaGPT",
            "upstream": "https://github.com/FinStep-AI/AlphaGPT",
            "desc": L("用符号回归在 A 股与加密市场做高效因子挖掘。我关注的是它把「搜索」与「表示」结合的方式，"
                     "这与我在自动驾驶里做自动数据挖掘是同一类问题。",
                     "Symbolic regression for efficient factor mining on A-shares and crypto. What interests me is how it "
                     "couples search with representation — structurally the same problem as automated data mining in AD."),
        },
    ],
    "note": L(
        "如果你有私有的量化项目可以公开，这一区块留好了位置：在 _build/gen/data_repos.py 的 FINANCE 里加一条即可。",
        "If you have private quant work you can open-source, the slot is ready: add one entry to FINANCE in "
        "_build/gen/data_repos.py.",
    ),
}

FORKS = {
    "total": 95,
    "title": L("另外还有 95 个 fork", "Plus 95 forks"),
    "body": L(
        "这些是别人的仓库，我只做了本地学习与改造，因此不列为我的作品。完整列表在 GitHub 个人页。",
        "These belong to other people; I only studied and modified them locally, so they are not listed as my work. "
        "The full list lives on my GitHub profile.",
    ),
}
