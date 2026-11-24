# projects/ — 之后的工程放这里

这个目录是**仓库内的原始工程目录**，也是博客内容的唯一真相源（canonical）。

```
projects/<slug>/          ← 你写的东西放这里（可以随便改，git 会记录）
        ↓  publish.cmd
blog/<slug>/              ← 构建产物（导航条、脱敏、webp 都在这一步注入）
```

## 为什么要有这个目录

之前的 40 个站点来自仓库外的两个目录（`_build/sources.local.json` 里的 `design` / `demos`），
仓库里只有复制品，所以「改了源文件」和「线上内容」之间必须手动跑一次发布。

新工程直接放这里，就不用再维护「外面一份、里面一份」：

- 原始工程进 git，历史可追溯；
- 想改内容直接改 `projects/<slug>/`，不用回头找外面对不对；
- 旧项目（外面那批）保持原样不动，继续由 `sources.local.json` 指向。

`blog/` 下的文件**永远不要手改**，下次发布会覆盖。

## 加一个新工程

1. 把工程目录拷进来：`projects/<slug>/`（根目录最好有 `index.html`；没有就用条目里的 `"index"` 指认入口页）。
2. 在 `_build/gen/data_design.py` 的 `DESIGN` 里加一条：

```python
{"slug": "<slug>", "root": "repo", "src": "<slug>", "images": "webp",
 "title": "中文标题", "title_en": "English title",
 "summary": "一两句摘要。", "summary_en": "One or two sentences.",
 "tags": ["具身"], "tags_en": ["Embodied"]},
```

3. 跑 `publish.cmd`（或 `python _build\publish.py -m "..."`）。

## 体积约定

单文件上限 6 MB；图片会在发布时转 WebP（最长边 1500、q80），所以原始截图可以放心放大 PNG。
PDF / zip / 视频不会进 `blog/`，需要给下载就放 Release。
