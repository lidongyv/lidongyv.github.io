# WAM 调研与精读工程

> 具身智能 **WAM（World Action Model，世界-动作模型）** 的调研与论文精读合集。
> 生成于 2026-10/11 ｜ 全部内容依据**论文原文（verbatim 引文 + 小节号）或官方代码取证**，未取到处一律标注，零脑补。

## 快速开始

**双击 `index.html`** 即可在浏览器中阅读全部内容（单文件、自包含、无需网络）。

## 目录结构

```
wam-survey-project/
├── index.html                  ← 阅读入口：14 个文件的全部内容（含目录锚点导航）
├── README.md                   ← 本说明
├── build_html.py               ← 重新生成 index.html 的脚本（内容有更新时运行）
└── reading/                    ← 全部精读笔记（Markdown 源文件）
    ├── README.md               ← 精读总索引 + 全局待核实清单
    ├── 01-input-representation.md         ① 输入表征 10 篇
    ├── 02-sim-real-validation.md          ② 实机/仿真验证 10 篇
    ├── 03-action-representation.md        ③ 动作表征 10 篇
    ├── 04-decision-representation.md      ④ 决策表征 9 篇
    ├── 05-goal-representation.md          ⑤ 目标表征 9 篇
    ├── 01b-input-goal-wave2.md            第二波A 输入/目标 10 篇
    ├── 02b-validation-wave2.md            第二波C 验证 10 篇
    ├── 03b-action-decision-wave2.md       第二波B 动作/决策 10 篇
    ├── 06-code-forensics.md               代码对质：动作空间 7 项取证
    ├── 07-rl-posttraining.md              RL post-training 精读 6 篇
    ├── 08-rl-code-forensics.md            RL 代码取证：VAMPO 全链路
    ├── 10-recent-progress.md               最新进展扫描（2026-11 月波：10 篇精读 + 30 篇速览）
    ├── 10-pseudocode.md                    伪代码提取（10 个机制，含依据分级）
    ├── fast-wam.md                         Fast-WAM 单篇全精读
    ├── html/                  ← 10 篇论文全文存档（离线逐句引用用）
    └── txt/                   ← 同上的纯文本版
```

## 内容地图（先读什么）

1. **快速建框架**：index.html 目录第 1 节「调研主文档」的 TL;DR → 定义与边界 → 论文×五轴总表。
2. **按五个问题深入**：精读线 ①输入 ②验证 ③动作 ④决策 ⑤目标，每节有 verbatim 引文与截断位置标注。
3. **实现与实操**：代码对质（动作空间取证）、RL post-training（6 篇精读 + VAMPO 代码全链路）。
4. **回查证据**：所有引文带论文小节号，可直接定位原文；10 篇 A 线论文全文在 `reading/html/`、`reading/txt/` 可离线查阅。

## 证据纪律（本工程铁律）

- 来源分级：**L3**=全文核实 / **L2**=部分正文 / **L1**=仅摘要，逐条标注。
- 「原文未写（已获取范围）」= 论文确实没写；「未取到」= 工具截断没拿到——两者都不是结论。
- 已知缺口（Meta WAM 博客不可达、部分附录数值截断、GR-1 归一化未发布等）集中在 `reading/README.md` 的待核实清单。

## 更新方式

1. 修改/新增 `reading/` 下的 Markdown 笔记；
2. 运行：`python build_html.py`（用 DSH 自带的 Python 即可）；
3. `index.html` 自动重新生成。
