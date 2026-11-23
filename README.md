# lidongyv.github.io

于立冬 / Lidong Yu 的个人主页与博客 —— **纯静态站点**，面向 VLA（Vision-Language-Action）与世界模型（World Action Model）落地方向的求职展示。

线上地址：<https://lidongyv.github.io/>

---

## 这是什么技术栈

**没有技术栈。** 输出的是预渲染好的 HTML + 一个手写 CSS + 一个小 JS，GitHub Pages 直接托管：

- 不依赖 Jekyll（仓库根目录有 `.nojekyll`），不消耗「每小时 10 次构建」配额；
- 发布由 `.github/workflows/pages.yml` 完成：仓库的 Pages 配置是 `build_type: workflow`，
  没有 workflow 就不会发布（2022-10-29 之后站点停止更新就是这个原因，现已修好）；
- 无外部字体、无 CDN、无追踪脚本，首屏没有任何阻塞资源；
- 深色模式自动跟随系统，也可手动切换并记忆；
- 中英双语，右上角切换。

## 目录结构

```
.                          ← GitHub Pages 从这里发布
├─ index.html              首页（中文）
├─ experience.html         工作经历
├─ publications.html       论文（8 篇详解 + 未发表手稿存档）
├─ projects.html           项目（8 个项目页入口 + 量产交付）
├─ opensource.html         开源（39 个自建仓库，按类型分类）
├─ about.html              关于 / 联系方式 / 简历下载
├─ blog/                   博客：index.html、每篇一个 HTML、feed.xml
├─ design/                 研究站：7 个独立成站的研究笔记与综述（见下文）
├─ en/                     以上全部页面的英文版
├─ 404.html
├─ sitemap.xml  robots.txt
├─ project1.html … project8.html   ← 旧版项目页（保持原 URL，见下文）
├─ resume/                 4 份简历 PDF
├─ assets/
│  ├─ site/                新站样式、脚本、favicon
│  ├─ og/                  社交分享图（1200×630）
│  ├─ img/  pdf/ css/ js/  fonts/   ← 旧站素材，仍在被使用
└─ _build/                 ★ 站点生成器与全部内容源
```

**要改内容，只改 `_build/`，不要手改生成出来的 HTML**——下次构建会覆盖。

## 构建 / 预览 / 部署

```powershell
# 生成站点（Python 3.9+，只用标准库，无需 pip install）
python _build\build.py

# 本地预览（默认 8000 端口）
python _build\preview.py 8000
# 打开 http://127.0.0.1:8000/

# 结构自检：标签闭合、内部链接、资源存在、模板残留
python _build\check.py

# 若某些文件因权限无法就地写入，构建会给出 staged 清单，执行：
python _build\deploy.py

# 发布
git add -A
git commit -m "site: update"
git push
```

推送后 `.github/workflows/pages.yml` 会自动发布，约 1 分钟生效。
进度看 Actions 页面的「Deploy site to GitHub Pages」。**不需要**在仓库设置里选主题。

## 内容在哪里改

| 想改什么 | 改哪个文件 |
|---|---|
| 姓名、邮箱、电话、一句话定位、求职状态 | `_build/gen/data_core.py` → `PROFILE` |
| 首页那 4 个数字（论文 / 专利 / 经历 / 团队规模） | `_build/gen/data_core.py` → `STATS` |
| 首页三张「在做什么」卡片 | `_build/gen/data_core.py` → `FOCUS` |
| 工作经历、教育、指标 | `_build/gen/data_core.py` → `CAREER` / `EDUCATION` |
| 论文（标题、作者、链接、问题/方法/结果） | `_build/gen/data_works.py` → `PAPERS` |
| 未发表手稿存档 | `_build/gen/data_works.py` → `MANUSCRIPTS` |
| 专利 | `_build/gen/data_works.py` → `PATENTS` |
| 项目卡片 | `_build/gen/data_works.py` → `PROJECT_PAGES` |
| GitHub 仓库与分类 | `_build/gen/data_repos.py` → `REPOS` / `REPO_GROUPS` |
| 金融/量化板块 | `_build/gen/data_repos.py` → `FINANCE` |
| 导航栏、页脚、UI 文案 | `_build/gen/layout.py` → `NAV` / `UI` |
| 配色、排版、间距 | `_build/assets/site.css`（CSS 变量集中在文件开头） |
| 新增博客文章 | 往 `_build/posts/` 丢一个 `.md`，然后重新构建 |

### 新增一篇博客

在 `_build/posts/` 新建 `YYYY-MM-DD-slug.md`，开头写 front matter：

```markdown
---
title: 中文标题
title_en: English title
date: 2026-12-01
tags: VLA, 世界模型
summary: 一句话中文摘要。
summary_en: One-line English summary.
---

正文用 Markdown（支持标题、列表、表格、代码块、引用、链接、图片）。
```

然后 `python _build\build.py`，会自动出现在中英文博客列表、RSS 和 sitemap 里。
文件名日期只用于排序，URL 用的是 `slug`（即去掉日期的文件名）。

### 新增一篇论文

在 `_build/gen/data_works.py` 的 `PAPERS` 列表最前面加一条（字段照抄相邻条目）。
`badges` 可选值：`oral`、`spotlight`、`first`（第一作者）、`preprint`、`technical report`。
`authors` 里用 `("Lidong Yu", 1)` 标记自己，`1` 会渲染成高亮下划线。

## 关于「诚实标注」

`/publications.html` 分两块：

1. **8 篇同行评审论文** —— 每条都有 arXiv / CVF / AAAI 官方链接，全部核验过。
2. **未发表手稿与投稿存档** —— `project2/3/4` 上的那几份稿子。经核验它们是**匿名评审稿**，
   查不到录用记录、没有 DOI、OpenAlex 与 Crossref 均无条目。
   因此明确标为未发表，**不混进论文列表**。求职场景下这一点很重要。

如果你后来真的中了一篇，把它从 `MANUSCRIPTS` 移到 `PAPERS` 即可。

## 旧版项目页为什么还在

`project1.html … project8.html` 是旧版 Jekyll 的产物。**四份简历 PDF 都直接链到这几个 URL**，所以它们必须继续可用。

- 内容**原样保留**（`_build/legacy.py` 从 `_layouts/project*.html` 读取后转成独立静态页）；
- 继续加载旧的 `assets/css/main.css`，视觉不变；
- 顶部加了一条「← 返回新主页 / Home」的细栏；
- 顺手修了 3 个旧站遗留的死链（详见 `_build/docs/migration.md`）。

## 研究站（`/design/`）

`design/` 下是 12 个独立成站的研究笔记 / 综述，由 `_build/design.py` 从两个源目录复制，
清单在 `_build/gen/data_design.py`。

**源目录写在 `_build/sources.local.json`（已 gitignore），不进仓库**——里面是本机路径，不公开。

| 源 | 说明 | 条目 |
|---|---|---|
| `design` | 个人研究站点 | `wam-survey`、`gs-reading`、`world-model-reading`、`embodied-wam`、`distill-research`、`fsd-v15`、`vla-rl` |
| `demos` | 论文阅读 demo | `cvpr-awards`、`cosmos-reading`、`deepseek-v4-reading`、`mamba-intro`、`vggt-reading` |

`demos` 组标了 `"images": "webp"`，截图会被重新编码（最长边 1500、q80）并把 HTML 里的引用改成 `.webp`。

它们同时出现在博客列表里（带「站」标记），并各自有 `/design/<slug>/` 入口。日期不是手写的，
取源目录里最新的文件修改时间。

**只发布自建正文**，以下内容被排除：

| 排除项 | 原因 |
|---|---|
| `*.pdf` | 论文 PDF（两个源合计 1.7 GB），既超出 Pages 容量，也不是我们该托管的 |
| `raw/` | 抓取的 CVPR / ICCV 会议页面 |
| 保存的 arXiv 论文页 | 第三方全文，且依赖 arxiv 自己的 `/static/` 资源 |
| 公众号文章副本 | 别人写的，不是你的博客 |
| `__pycache__`、`*.pyc`、`*.bin`、`*.whl`、`*.tex`、`*.mp4`、会话转储 | 构建垃圾 |
| 单文件 > 6 MB | 体积保护 |
| 与公司相关的 demo | 见下 |

### 脱敏

`demos` 里 56 个目录多数是公司项目（运营数据、OKR、issue 管理、团队信息、量产方案等），
**只挑出与公司完全无关的 5 个论文阅读项目**。构建时还会做一次文本脱敏：

- 替换表在 `sources.local.json` 的 `scrub` 里（同样是 gitignore 的），代码里不出现公司名；
- 每次构建会打印替换次数，当前为 **0**（这 5 个项目本身不含公司标识）；
- 发布前可用这条命令复核：搜索 `design/` 下所有 html/md/json/txt，应当 0 命中。

如果源目录新增了项目：在 `_build/gen/data_design.py` 里加一条（slug / root / src / 标题 /
摘要 / 标签），然后 `python _build/build.py`。构建会先清空 `design/` 再复制，删掉条目对应站点即消失。

## 容量与成本

- 发布内容约 **135 MB**（`design/` 113.8 MB + 其余 20.8 MB），GitHub Pages 上限 **1 GB**（约 13%）；
- 12 个论文 / 报告 PDF（共 133 MB）已迁到 Release
  [`papers-2026-11`](https://github.com/lidongyv/lidongyv.github.io/releases/tag/papers-2026-11)，
  页面链接直接指向 Release 下载地址，不再占用 Pages 配额，也不会再吃站点流量；
- 166 张旧项目截图已转成 WebP（53.4 MB → 12.3 MB）。原图由 tag `images-png-originals` 固定，
  恢复命令：`git checkout images-png-originals -- assets/img`；
- 仓库已改为 public，所以 **GitHub Pages 用 Free 就够了**，Pro 年费可以停掉；
- 详细测算见 [`_build/docs/github-pages-capacity.md`](_build/docs/github-pages-capacity.md)。

> 注意：**如果以后把仓库改回 private**，私有仓库的 Release 资产不能公网下载，这些链接会全部 404。
> 届时把 `_build/gen/releases.py` 的 `ENABLED` 改回 `False`，从 git 历史恢复 `assets/pdf/`，重新构建推送。

## 备份与回滚

改动前做了三份快照（`_backup/`，已 gitignore，不会推送）：

| 快照 | 内容 |
|---|---|
| `_backup\git-history-*.bundle` | 完整 git 历史（所有分支 + tag） |
| `_backup\worktree-*.zip` | 改动前的完整工作树 |
| `git tag pre-redesign-*` | 仓库内轻量快照，`git checkout` 即回滚 |

## 文档索引

| 文档 | 内容 |
|---|---|
| `_build/docs/migration.md` | 本次重构做了什么、URL 对照表、保留清单、回滚方式 |
| `_build/docs/github-pages-capacity.md` | GitHub Pages 容量/流量/付费的完整调研与实测 |
| `_build/docs/research/papers.md` | 8 篇论文 + 4 份手稿的逐条核验记录（含链接、页码、判定依据） |
| `_build/docs/research/github-own-repos.md` | 134 个公开仓库的盘点：39 个自建 + 95 个 fork，分类依据 |

## 许可

站点内容（文字、简历、图片）版权归于立冬，未经许可请勿转载。
`_build/` 下的生成器代码可自由参考。
