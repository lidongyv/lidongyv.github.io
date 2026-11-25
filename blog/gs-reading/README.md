# 3D Gaussian Splatting 讨论与调研阅读站

一次完整讨论的整理成果，独立、自包含的静态 HTML 工程。**双击 `index.html` 即可离线阅读**（论文清单页的搜索/筛选不需要网络）。

## 内容

- **工作原理**（pages/basics.html）：输入是什么、学的是什么、为什么“没有神经网络”、单场景优化与泛化、渲染管线四步、颜色与球谐系数（按 Marr 三层组织）。
- **范式演进**（pages/paradigm.html）：per-scene 优化 vs 前馈泛化 3DGS、DiT 术语校准、2024 锚点与 2025–2026 版图、五条趋势。
- **per-scene 优化点**（pages/per-scene.html）：208 篇论文的七大着力方向 + 底层四个优化旋钮 + 为什么 per-scene 仍是最大盘。
- **最近三个月**（pages/latest-3months.html）：2026-08→11 arXiv 最新 170 篇精选一览（前馈重建/密度控制/压缩部署/动态/语义五线程）+ 四篇深读（PaveSplat / CrossPressor / InstantHand / Cross3DGS）的实现细节、公式与伪代码；原文留存 raw/reading/。
- **方法核心实现**（pages/methods.html）：按方法原理分四类（表征修正 / 融合路由 / 监督正则 / 系统执行），每篇方法页含中英核心说明 + 核心代码实现（示意实现，按论文公式整理）：m-pavesplat / m-crosspressor / m-instanthand / m-cross3dgs。
- **调研总报告**（pages/survey-master.html）：2025–2026 五会议 870 篇统计、10 类 × 7 届分布、八条趋势、代表论文索引、覆盖审计。
- **论文全清单**（survey-list.html）：870 篇逐条（标题 + 已验证链接 + 会议 + 类别 + 一句话核心），支持按会议/类别筛选与全文搜索。

## 目录结构

```
gs-reading-site/
├── index.html                站点入口
├── survey-list.html          870 篇可搜索清单（生成物）
├── assets/                   style.css / app.js
├── data/papers.js            870 条结构化数据（生成物）
├── pages/                    五个内容页
├── reports/                  四份调研源报告（md，含排除名单与覆盖自检）
├── raw/                      审计原始数据与深读原文
│   ├── cvpr/                 CVPR 2025/2026 页面抓取与脚本
│   ├── iccv-eccv/            ICCV 2025 / ECCV 2026 页面抓取与脚本
│   └── reading/              最近三个月四篇深读的原文 HTML/文本与笔记
└── build_site.py             构建脚本：reports/*.md → data/papers.js + survey-list.html
```

## 数据来源与审计

- 数据快照：2026-11-09。CVF OpenAccess（CVPR/ICCV 全量页 + 逐天页交叉）、eccv.ecva.net（ECCV 2026，CVF 未上线）、OpenReview 官方 API（NeurIPS 2025）、ojs.aaai.org（AAAI 2025/2026）。
- 合计 870 篇 = CVPR 343 + ICCV/ECCV 313 + NeurIPS/AAAI 214；标题含 gaussian/splat 后人工判定，排除误报 84 篇（名单在各报告附录）。
- 已知缺口：NeurIPS 2026 未公开；ECCV 2026 官方列表标注 preliminary；NeurIPS 2025 未含 Workshops；NeurIPS/AAAI 按关键词枚举存在纯缩写漏检风险；TranSplat(AAAI-25) arXiv 链接待补。详见 reports/ 各报告头部。

## 重新构建

```
python build_site.py    # 从 reports/*.md 重新生成 data/papers.js 与 survey-list.html
```

预期输出三行解析计数（343 / 313 / 214）与 `TOTAL: 870 (expect 870)`。

更新时间: 2026-11-20
