# 世界模型调研与精读 · 独立阅读工程

双击 `index.html` 离线阅读全部内容（无外链依赖）。

## 结构
- `index.html` — 主页：输出任务分类总表 + 讨论锚点
- `world-model-survey.html` — 调研总览（实现方式 6 类 + 分域清单）
- `world-model-capability-taxonomy.html` — 输出任务 × 能力（功能视角）
- `narrative.html` — 叙事线本地资料（四篇文章原文摘引）
- `progress.html` — 精读进度与 12 条事实纠偏
- `papers/index.html` — 51 篇精读笔记索引（带搜索）
- `papers/<篇名>.html` — 每篇精读笔记（原句+行号+数值表+消融）
- `papers/txt/` — 论文提取正文（引用核对依据；PDF 共 638MB 未打包，见源目录 `auto_docs/papers/pdf/`）
- `sources/` — 全部 Markdown 源稿与构建脚本（可重新生成站点）

## 重新构建
`python sources/build_site.py <本目录>`（需 Python 3.8+，脚本会自装 markdown 依赖到 sources/../papers/pylibs 同级源目录）

## 证据纪律
- 35 篇关键引文已逐条 grep 原文验证；6 篇（GAIA-1、DriveDreamer-2、GenAD、Drive-WM、UniWorld、ENACT）笔记完整、引文抽查在讨论时进行。
- 所有「原文未写」「【推断】」标注如实区分，最终依据以论文原文（txt/PDF）为准。

生成时间: 2026-11-21 17:51
