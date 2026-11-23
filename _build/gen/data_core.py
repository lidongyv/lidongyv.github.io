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
    "title": L("自动驾驶 · 世界模型", "Autonomous driving · world models"),
    "org": L("新石器慧通（北京）科技有限公司", "Neolix Technology (Beijing) Co., Ltd."),
    "headline": L(
        "VLA · 世界模型 · 端到端自动驾驶",
        "VLA · World models · End-to-end driving",
    ),
    "lede": L(
        "2019 年起做自动驾驶算法，先后在驭势科技（南京 AI 研究院）、蔚来、元戎启行、地平线、"
        "NVIDIA 和新石器慧通工作，做过感知、跟踪、数据闭环与端到端。"
        "关注 VLA 与世界模型的落地。",
        "Autonomous-driving algorithms since 2019 — UISEE (Nanjing AI Research Institute), NIO, "
        "DeepRoute.ai, Horizon Robotics, NVIDIA and Neolix, across perception, tracking, data loops and "
        "end-to-end driving. My current interest is landing VLA and world models.",
    ),
    "email": "yvlidong@gmail.com",
    "phone": "+86 151-6523-1608",
    "location": L("北京，中国", "Beijing, China"),
    "scholar": "https://scholar.google.com/citations?hl=en&user=cYLbx3AAAAAJ",
    "github": "https://github.com/lidongyv",
    "site": "https://lidongyv.github.io/",
    "status": L("开放全职与顾问机会", "Open to full-time and advisory roles"),
    "status_short": L("开放机会", "Open to roles"),
    "focus_tags": [
        L("VLA", "VLA"),
        L("World Action Model", "World Action Model"),
        L("端到端自动驾驶", "End-to-end AD"),
        L("数据闭环", "Data closed loop"),
        L("感知 → 规控", "Perception → planning"),
    ],
}

# --- Research areas (keywords only) ----------------------------------------
FOCUS_AREAS = [
    L("VLA / 视觉-语言-动作模型", "VLA / vision-language-action models"),
    L("世界模型与 World Action Model", "World models and World Action Models"),
    L("端到端自动驾驶", "End-to-end autonomous driving"),
    L("数据闭环与自动标注", "Data closed loops and auto-labelling"),
    L("感知：检测、跟踪、车道线", "Perception: detection, tracking, lanes"),
    L("强化学习后训练", "Reinforcement-learning post-training"),
]

# --- Career ------------------------------------------------------------------
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
