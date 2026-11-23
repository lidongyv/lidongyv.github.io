# GitHub Pages 能挂多少内容：容量预算与实测

> 调研时间：2026-11-23。依据：GitHub 官方文档 [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)
> 与 [About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 一句话答案

**公开发布出去的内容上限是 1 GB，不是「无限」。** 你现在的站点大约 193 MB，占了上限的 **19%**，还有约 **5 倍**的余量；
真正最先撞到的不是容量，而是**每月 100 GB 的流量软上限**——因为站里有 132 MB 的论文与报告 PDF。

## 官方硬指标（逐条对应你的问题）

| 指标 | 上限 | 说明 | 站点现状 |
|---|---|---|---|
| 发布站点总大小 | **1 GB** | 「Published GitHub Pages sites may be no larger than 1 GB」 | 约 **193 MB**（19%） |
| 源仓库推荐大小 | **1 GB**（recommended） | 超过后 Pages 构建可能失败 | 约 193 MB（不含本地备份） |
| 每月流量 | **100 GB**（soft） | 软限制，超了会收到 GitHub 的邮件提醒，可能限速 | 见下方测算 |
| 构建次数 | **10 次/小时**（soft） | 只针对 Jekyll 自动构建；用自定义 Actions 工作流不受此限 | 本站已关闭 Jekyll，**不触发构建配额** |
| 单用户站点数 | 1 个 | 一个账号只能有一个 `*.github.io` | 1 个 |
| 单次推送体积 | **2 GB** | 超过直接拒绝 push | 单次 push 增量很小 |
| 单个文件 | **100 MiB 硬上限** | 超过 100 MiB 会被拒绝；超过 50 MiB 会警告 | 最大 `sot.pdf` 61.9 MB、`mot.pdf` 55.5 MB，**都已超过 50 MiB 警告线，但未触 100 MiB 上限** |

## 流量测算：真正的瓶颈在哪

100 GB/月 ÷ 单次访问量 = 每月可承受的访问次数。

| 访问场景 | 单次流量 | 100 GB 可支撑 |
|---|---|---|
| 只看主页（图 + CSS + JS） | ≈ 0.3 MB | 约 33 万次 |
| 看完主页 + 论文页 + 开源页 | ≈ 0.5 MB | 约 20 万次 |
| 点开一次 `project8` 项目页（60 张截图） | ≈ 12 MB | 约 8300 次 |
| 下载一次 `sot.pdf`（62 MB） | 62 MB | **约 1600 次** |
| 下载一次 `mot.pdf`（56 MB） | 56 MB | 约 1800 次 |

结论：**日常浏览完全不用担心；两个大 PDF 才是唯一风险点。**
如果哪天这两个 PDF 被转发到群里，几千次下载就能吃掉一个月的额度。

## PDF 迁移到 Releases：已做完上传，但被「私有仓库」卡住

实测过程（2026-11-23）：

1. 12 个 `assets/pdf/*.pdf`（共 133 MB）已**全部上传成功**到 Release
   [`papers-2026-11`](https://github.com/lidongyv/lidongyv.github.io/releases/tag/papers-2026-11)，
   API 返回 12 个资产、`state=uploaded`。
2. 但 `releases/download/<tag>/<file>` 这个**公网下载地址返回 404**，未登录 404，
   带 token 也 404。原因是：**这个仓库是私有的，GitHub 不会把私有仓库的 Release 资产
   对外公开提供**（网页端点无法用 Bearer 头认证）。
3. 结论：**只要仓库还是 private，PDF 就只能留在仓库里**，否则线上论文页会出现死链。
   因此已把 PDF 恢复到 `assets/pdf/`，链接全部指向本地。

### 如果你想让这次迁移生效（顺带省掉年费）

只需一步：**把仓库改成 public**（Settings → General → 最下面 Danger Zone → Change visibility）。
之后执行：

```powershell
# 1. 打开开关：_build/gen/releases.py 里 ENABLED = False -> True
python _build\build.py
git rm -r assets/pdf
git add -A
git commit -m "site: serve archived PDFs from the release"
git push
```

收益：发布体积从 **193 MB 降到约 60 MB**，100 GB/月的流量风险基本清零。
上传脚本是 `_build/tools/upload_release.py`（已完成，重复运行会跳过已存在的资产）。

## 其余两条优化

2. **把 `project6/7/8` 的截图压成 WebP**（可选）
   这 3 个页面共 121 张 PNG、约 40 MB。压到 WebP q82 大约能到 4 MB，
   单页访问流量降到 1/10。旧页面按原样保留是为了「不删内容」，但体积可以优化。

3. **不要在仓库里放大视频 / 大模型权重**
   单文件 100 MiB 是硬墙，2 GB 是单次推送的硬墙。真要放就走 Releases 或外部对象存储。

## 与你「每年付费」的关系（重要更正）

实测发现：**这个仓库当前是 private**，且 Pages 的 `build_type` 是 `workflow`。这有两个直接后果。

1. **私有仓库发 Pages 需要付费套餐。** 官方文档写的是：Pages「available in public repositories with GitHub Free…
   and in public and private repositories with **GitHub Pro / Team / Enterprise Cloud**」。
   所以只要你保持仓库私有，这笔 Pro 年费是**必须的**，不是可选项。
   反过来说——**把仓库改成 public，这笔钱立刻就能省掉**，因为个人站点在公开仓库上用 Free 就能发。
2. **`build_type: workflow` 意味着必须有一个 Actions 工作流才会真正发布。** 之前仓库里没有任何 workflow，
   所以最后一次成功发布停在 **2022-10-29**（那次部署的正是最后一条 `cvpr2022` 提交 `6fb050b`）。
   之后四年间的所有改动都没上线。现在 `.github/workflows/pages.yml` 已补上。
   现在 `.github/workflows/pages.yml` 已经补上，推 master 即自动发布。

### 1 GB / 100 GB 这两个额度与付费无关

Free 和 Pro 完全一样，加钱买不到更高配额。所以判断很简单：

- 只是个人主页 + 博客 → **把仓库改成 public，年费可以省**；
- 需要私有仓库也能发布（或用私有仓库放草稿）→ 保留付费。

## 当前站点体积构成（实测）

| 目录 | 大小 | 占比 | 备注 |
|---|---:|---:|---|
| `assets/pdf` | 132.6 MB | 69% | 12 个 PDF，其中 `sot.pdf` 61.9 MB、`mot.pdf` 55.5 MB |
| `assets/img` | 51.9 MB | 27% | 8 个旧项目页的截图 |
| `resume` | 2.0 MB | 1% | 4 份简历 PDF |
| `assets/site` + `assets/og` | 0.4 MB | 0.2% | 新版样式、脚本、分享图 |
| 全部 HTML | 1.2 MB | 0.6% | 27 个页面 |
| `_build`（源码） | 0.4 MB | 0.2% | 生成器与内容源 |
| **合计（发布内容）** | **约 193 MB** | **19% of 1 GB** | |

不含：本地备份 `_backup/`（约 183 MB，已在 `.gitignore` 里，不会推送）、`.git` 历史（385 MB，只在本地/远端仓库，不计入发布站点）。
