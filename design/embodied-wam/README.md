# 具身智能概念与 WAM 研究工程（embodied-wam）

> 一个自包含的调研/精读工程：从通用概念（System 0/1/2(/3)、Marr 三层）到 WAM（世界-动作模型）与 GR00T 系列的实现细节。
> 原则：一切以论文原文或官方代码为准；英文引文附中文译文；未获取处如实标注。

## 快速开始

双击打开 **[site/index.html](site/index.html)**（纯静态、零依赖）。

阅读顺序建议：概念页 → Marr × Kahneman 页 → WAM 总览 → GR00T 系列页。

## 目录结构

```
embodied-wam/
├── README.md                    # 本文件：工程说明
├── wam-survey.md                # WAM 调研主文档（定义/谱系/五轴表征/论文×五轴总表/第二波要点）
├── wam-reading.html             # WAM 阅读版（自包含 HTML，含五轴详解与总表）
├── site/                        # 概念站（本次对话的五个问答 + 对比 + GR00T）
│   ├── index.html               # 站点入口
│   ├── style.css
│   └── pages/
│       ├── concepts.html        # System 0/1/2(/3)：定义、例子、五个高频问答
│       ├── marr.html            # Marr 三层 × Kahneman 双过程：对比 + 可比性评估
│       └── groot.html           # GR00T N1 / N1.5 / GR00T 2 精读
└── notes/                       # 精读笔记（Markdown）
    ├── README.md                # 精读总索引（五问分线 + 波次 + 待核实清单）
    ├── 00-concepts-system0123.md        # 四概念谱系与原文
    ├── 00-concepts-marr-vs-kahneman.md  # Marr × Kahneman 详细对比
    ├── 01-input-representation.md       # ① 输入表征 10 篇
    ├── 02-sim-real-validation.md        # ② 实机/仿真验证 10 篇
    ├── 03-action-representation.md      # ③ 动作表征 10 篇
    ├── 04-decision-representation.md    # ④ 决策表征 9 篇
    ├── 05-goal-representation.md        # ⑤ 目标表征 9 篇
    ├── 01b-input-goal-wave2.md          # 第二波：输入/目标 10 篇
    ├── 02b-validation-wave2.md          # 第二波：验证 10 篇
    ├── 03b-action-decision-wave2.md     # 第二波：动作/决策 10 篇
    ├── 06-code-forensics.md             # 官方仓库代码对质（动作空间）
    ├── 07-rl-posttraining.md            # RL 后训练 6 篇
    ├── 08-rl-code-forensics.md          # VAMPO RL 代码链路
    ├── 09-gr00t-series.md               # GR00T N1/N1.5/GR00T 2
    ├── fast-wam.md                      # Fast-WAM 单篇全精读
    ├── system0123-reading.html          # 四概念英中对照精读页
    └── html/ txt/ pdfs/ fast-wam-figs/  # 10 篇论文的本地全文存档与图（离线可查）
```

## 内容边界与来源

- **WAM 主线**：OpenMOSS 综述（[2605.12090](https://arxiv.org/abs/2605.12090)）+ NUS 综述（[2606.20781](https://arxiv.org/abs/2606.20781)）+ 逐篇原文核实（L1/L2/L3 分级标注在各笔记中）。
- **System 0/1/2(/3)**：Taniguchi et al. [2503.06138](https://arxiv.org/abs/2503.06138)（S0/S3 唯一学术一手出处）；S1/S2 概念源头 Kahneman 2011；工程用例 GR00T N1（[2503.14734](https://arxiv.org/abs/2503.14734)）与 Figure Helix 博客。
- **Marr 三层**：Marr《Vision》1982（经摘录页逐字复刻）；Kahneman 引文经 arXiv:2410.08328 转引。
- **GR00T 系列**：N1 论文全文（L3）；N1.5 与 GR00T 2 官方研究页（L1 一手）。

## 已知缺口（作者如实记录）

1. Meta WAM 博客不可达 → CoT 潜轨迹搜索未获一手证实（不引用）。
2. Taniguchi IEEE SII 2026 讲稿正文、中文「Physical AI 三层脑」公众号正文未取到。
3. N1.5 / GR00T 2 的 whitepaper 正文未取到（仅有官方页）。
4. GR00T 1.7 仅在 GR00T 2 页被提及，细节未取。
5. Kahneman / Marr 原书逐字页码未核（以转引为准）。

## 版本

- 生成：2026-11-20
- 精读规模：两波 + 专项共 70+ 篇次；代码对质 7 项；概念专项 3 份。
