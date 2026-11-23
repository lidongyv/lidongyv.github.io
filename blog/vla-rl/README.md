# VLA / VA 与 RL post-training 调研工程

面向《VLA 在 L4 的可执行落地方案》的技术调研独立工程包：驾驶 VLA 的 L/A/V 拆解、RL post-training 方法与实现蓝图，全部结论附论文原文引用（章节号）或开源代码依据。

## 阅读入口

双击打开 **`index.html`**（无需联网、无需构建，任意现代浏览器可用）。

| 文件 | 内容 |
|---|---|
| `index.html` | 首页：目录导航 + 核心结论速览 + 核实纪律 |
| `01-vla-va-参考论文.html` | VLA/VA 参考论文总览（约 50 篇，含方法/实现/与你方案的对应表） |
| `02-驾驶VLA-LAV拆解.html` | 15 篇驾驶 VLA 逐篇 L/A/V 拆解 + RL post-training + 实现蓝图（含 GRPO forward/backward/loss 逐行实现） |
| `03-讨论问答.html` | 六轮讨论问答速览 + 证据留档清单 |
| `style.css` | 站点样式 |

## 目录结构

```
vla-rl-调研工程/
├── index.html / 01-*.html / 02-*.html / 03-*.html / style.css   # 阅读站（生成物）
├── docs/                # 站点源文档（Markdown，事实的单一来源）
│   ├── vla-va-参考论文.md
│   └── 驾驶VLA-LAV拆解.md
├── evidence/            # 论文全文留证（逐字引用与截断说明）
│   ├── vlp_2401.05577.txt          be_vdriver_2503.03074.txt
│   ├── genad_2403.09630.txt        drivevlaw0_2510.12796.txt
│   ├── vl_drive_full.txt           icr_drive_full.txt
│   └── senna_main.txt
└── build_site.py        # 一键重建脚本（见下）
```

## 如何修改与重建

1. 编辑 `docs/` 下的 Markdown（或 `build_site.py` 里的问答内容）；
2. 在本文件夹内运行：

```
python build_site.py
```

即可重新生成全部 HTML 页面（页面 TOC、导航、样式自动刷新）。

## 核实纪律（本项目的一贯约定）

- 所有论文结论附**逐字原文引用 + 章节号**，代码结论附**函数名/仓库文件**（如 verl `core_algos.py` 的 `compute_grpo_outcome_advantage`、`kl_penalty`）；
- arXiv 链接均经官方 API/检索页核对（调研中曾剔除 10 余个记忆错误的论文 ID，如 EMMA 正确 ID 为 2410.23262）；
- 不存在的论文明确标注（如 **EMMA-2 不存在**、LangDrive 未找到可靠条目）；
- 论文未写之处标「原文未写」；基于事实的工程拼装建议标「[合成]」。

## 关键链接（调研使用的外部资源）

- 论文原文：各页表格内 arXiv 链接
- 开源实现：`verl`（GRPO 训练）、`ucla-mobility/AutoVLA`、`autonomousvision/navsim`（PDMS 奖励）、`huggingface/lerobot`、`Physical-Intelligence/openpi`

---
生成时间：2026-11-20
