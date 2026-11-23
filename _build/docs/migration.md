# 本次重构说明（旧站 → 新站）

> 时间：2026-11-23。原则：**不删内容**，只重新归并、分类与重排。

## 1. 旧站是什么

Particle Jekyll 主题（2019 年的第三方模板）：

- `index.html` 只有一行 front matter，正文来自 `_includes/projects.html`（12.8 KB 的硬编码 HTML）；
- 8 个项目详情页是 `project1.html … project8.html`，每个只有 `layout: projectN`，内容在 `_layouts/projectN.html`；
- 前端依赖 gulp + SASS 预编译到 `assets/css/main.css`，外加 CDN 上的 `particles.js` 和 `sweet-scroll`；
- 有一个 `google-analytics.html`，条件是 `site.google-analytics`，但 `_config.yml` 里从未设过 → **从未生效**；
- 文案停留在 2020 年：`_config.yml` 里写着「我正在找全职工程师和 PhD 位置」。

## 2. 新站是什么

纯静态站点，**零运行时依赖**：

- 生成器：`_build/`（Python 标准库，无第三方包）；
- 输出：仓库根目录的预渲染 HTML，GitHub Pages 直接托管，**不需要 Jekyll、不需要 Actions**；
- `.nojekyll` 已加入 → GitHub Pages 不再跑 Jekyll 构建，同时也就不再消耗「每小时 10 次构建」配额；
- 样式：`assets/site/site.css`（手写，28 KB，含深色模式与打印样式）；
- 脚本：`assets/site/site.js`（6.7 KB，主题切换、移动端菜单、仓库筛选、复制邮箱、阅读进度、返回顶部）；
- 无外部字体、无 CDN、无追踪脚本。

## 3. URL 对照表

| 旧 URL | 新 URL | 处理方式 |
|---|---|---|
| `/` | `/` | 全新首页 |
| `/project1.html` … `/project8.html` | **同 URL** | 转成独立静态页，内容原样保留，加一条「返回新主页」的顶栏 |
| `/assets/pdf/*.pdf` | 同路径 | 未改动 |
| `/assets/img/**` | 同路径 | 未改动 |
| — | `/experience.html` | 新增：六段工作经历，含职责范围与指标 |
| — | `/publications.html` | 新增：8 篇论文逐篇详解 + 未发表手稿存档 |
| — | `/projects.html` | 新增：8 个项目卡片 + 量产交付 |
| — | `/opensource.html` | 新增：39 个自建仓库按类型分类 |
| — | `/blog/` | 新增：博客与研究笔记 |
| — | `/about.html` | 新增：简介、联系方式、简历下载 |
| — | `/en/**` | 新增：以上全部页面的英文版 |
| — | `/resume/*.pdf` | 新增：4 份简历 PDF |
| — | `/sitemap.xml`、`/robots.txt`、`/blog/feed.xml` | 新增 |

`project1.html … project8.html` 之所以必须保持原 URL：**四份简历 PDF 里的「Selected Projects」直接链到这四个页面**
（`project1`、`project5`、`project6`、`project8`），换 URL 会让简历里的链接全部失效。

## 4. 保留了但不再使用的东西

以下文件**一个都没有删除**，只是不再被新站引用：

| 文件 / 目录 | 状态 | 说明 |
|---|---|---|
| `_config.yml` | 保留 | 旧主题配置，已被 `_build/gen/data_core.py` 取代 |
| `_layouts/`、`_includes/` | 保留 | 旧 Jekyll 模板；`_build/legacy.py` 读取它们生成新的静态页 |
| `src/` | 保留 | 旧 SASS/JS 源码 |
| `assets/css/main.css`、`main.min.css` | **仍在使用** | 8 个旧 project 页继续用它们，保证视觉不变 |
| `assets/js/`、`assets/fonts/` | 仍在使用 | 同上 |
| `gulpfile.js`、`package.json`、`yarn.lock` | 保留 | 旧构建链，已不需要 |
| `index.md`、`particle.jpg` | 保留 | 旧主题素材 |
| `LICENSE.txt`、`README.md` | 保留（README 已重写） | |

## 5. 顺手修掉的 3 个旧站 bug

转换旧页面时发现并修正了 3 个**已经存在多年**的失效链接：

1. `project3.html` → `/assets/pdf/interpreatble tpami.pdf`（拼写错误，实际文件是 `interpretable tpami.pdf`）
2. `project4.html` → `/assets/pdf/nips-cvprpdf`（文件名被截断，实际是 `nips-cvpr19.pdf`）
3. `project5.html` → `/assets/img/project5/new/14.png`（该图片从未提交，目录里只有 0–13）

## 6. 内容层面的更正

旧站把这些**匿名投稿稿**当作已发表工作展示（页面标题写 "Explicit Context Mapping for Stereo Matching" 等，并挂了「Paper v1 / Paper v2」）：

- `project2`：CVPR'19 / ICCV'19 投稿稿
- `project3`：CVPR'18 / ECCV'19 投稿稿 + TPAMI 格式手稿
- `project4`：NeurIPS / CVPR'19 投稿稿

核对结果：**这些稿件的正文是匿名评审版，查不到任何录用记录、没有 DOI，OpenAlex / Crossref 无对应条目。**
新站在 `/publications.html` 里把它们单独列为「未发表手稿与投稿存档」并明确标注，**不混进发表论文列表**。
这一条对正在求职的场景尤其重要——把投稿写成已发表，是面试里最容易被抓住的硬伤。

`project1` 是真实的 AAAI 2018 论文（第一作者、spotlight），正常保留在论文列表第一位。

## 7. 回滚方式

三处快照，任选其一：

```powershell
# 1) 本次改动前的完整工作树
Expand-Archive _backup\worktree-20261123-*.zip -DestinationPath C:\restore

# 2) 完整 git 历史（含所有分支与 tag）
git clone _backup\git-history-20261123-*.bundle C:\restore\repo

# 3) git tag（在仓库内）
git checkout pre-redesign-20261123-*
```

`_backup/` 已写入 `.gitignore`，不会推送、不占 Pages 配额。
