# FSD V15 / Technical Map

A bilingual (Chinese / English), single-page technical teardown of Tesla FSD V15: the public facts, the engineering inferences, and the open hypotheses — with 11 original diagrams and a static WeChat edition.

**Live entry:** `index.html`

---

## 1. What is in this folder

| File | Purpose |
|---|---|
| `index.html` | The main bilingual, interactive article. Single file + `assets/` images. |
| `assets/fig-01.png` … `fig-09.png` | The 11 original diagrams (1600×1000, 2× PNG). |
| `wechat-zh.html` | Static, inline-styled Chinese edition for the WeChat editor. |
| `wechat-en.html` | Static, inline-styled English edition for the WeChat editor. |
| `sources.md` | Source list, evidence levels, and notes on conflicting numbers. |
| `src/` | Editable source parts for the main page (`head.html`, `zh.html`, `en.html`, `foot.html`). |
| `build_page.py` | Rebuilds `index.html` from `src/`. |
| `export_wechat.py` | Rebuilds the two WeChat editions from `index.html`. |
| `build_figures.py` | Rebuilds the 9 diagram HTML files in `figures/` (screenshot with Chrome to refresh PNGs). |
| `figures/` | Fixed-size HTML sources for the diagrams. |

> The article is self-contained apart from `assets/`. Host the whole folder, not just `index.html`.

---

## 2. Host it

### GitHub Pages
1. Create a repository, upload the whole folder.
2. Settings → Pages → Deploy from branch → `main` / root.
3. Open `https://<user>.github.io/<repo>/`.

### Netlify / Vercel / Cloudflare Pages
- Drag the folder into Netlify Drop, or set the project root to this folder.
- No build command is required; it is static HTML.

### Any web server
- Upload `index.html`, `assets/`, and the two `wechat-*.html` files.
- Keep the relative paths (`assets/fig-01.png`, etc.) intact.

### Language links
- Chinese: `index.html` or `index.html?lang=zh`
- English: `index.html?lang=en`
- The toggle also stores the choice in `localStorage` and updates the URL.

---

## 3. Print / PDF

- Click **打印 / PDF** in the top bar, or press `Ctrl/Cmd + P`.
- The print stylesheet hides navigation, TOC, rail, lightbox, and buttons, and prints the currently selected language.
- For best results: A4, default margins, “Background graphics” on.
- Figures are intentionally dense; use **⤢ Zoom** / click any figure for the full-resolution view.

---

## 4. WeChat edition

1. Open `wechat-zh.html` (or `wechat-en.html`) in a browser.
2. Click **全选正文 / Select article** at the top.
3. Paste into the WeChat official-account editor.
4. Images: if they do not upload automatically on paste, upload `assets/fig-*.png` to the WeChat asset library and replace them.
5. The toolbar at the top is outside the article and is **not** copied when you use the button.

**Why a separate file?** WeChat strips `<style>`, external CSS, JS, and many layout properties. The WeChat editions inline all styles, convert tables into stacked cards for mobile readability, and keep the diagrams as PNGs. The interactive page is for the web; the static page is for the editor.

---

## 5. Edit the article

The main page is assembled from four parts:

```
src/head.html   → <head>, CSS, top bar, hero, TOC, opening <div class="lang-zh">
src/zh.html     → Chinese article sections (00–13 + Appendix A)
src/en.html     → English article sections (00–13 + Appendix A)
src/foot.html   → rail, footer, lightbox, JavaScript, closing tags
```

After editing:

```bash
python build_page.py      # rebuild index.html
python export_wechat.py   # rebuild wechat-zh.html / wechat-en.html
```

Each language has 14 numbered sections plus Appendix A (15 total) with IDs `zh-*` / `en-*` and a shared `data-sec`. If you add a section, update:
- the TOC in `src/head.html`,
- the mobile nav (generated from the TOC at runtime),
- both `zh.html` and `en.html`,
- the section count in the hero if needed.

### Diagrams

`build_figures.py` writes fixed-size HTML files to `figures/`. To refresh PNGs, screenshot each file with headless Chrome at `1600×1000`, 2× scale. Example on Windows:

```powershell
chrome --headless=new --no-sandbox --disable-gpu --hide-scrollbars `
  --force-device-scale-factor=2 --window-size=1600,1000 `
  --screenshot="assets/fig-01.png" "file:///.../figures/fig-01.html"
```

The figures use only HTML/CSS (no external libraries), so they are reproducible and easy to restyle.

---

## 6. Evidence and known limitations

- This is **not** an official Tesla document. It combines public sources, the operator's engineering notes, and the author's inference.
- Evidence labels: `PUBLIC` / `REPORTED` / `INFERRED` / `HYPOTHESIS`. See `sources.md`.
- **4B vs 10B:** 3B + 1B = 4B is the on-vehicle graph (Musk post, screenshot); the reported “10B” is likely a training/cloud teacher. Tesla has not clarified. The article treats this as an open scope question, not a contradiction.
- **25×:** official framing is “critical intervention rate” improvement versus V14, not “25× safer” and not “25%”. The baseline is not public.
- **Grok + FSD:** public evidence supports a shared general-pretrain stack / internet-video encoder; there is no public evidence that Grok weights directly co-train with FSD. The article separates the readings.
- **V15 pipeline:** the “general pretrain → distill → deploy” diagram is an inference, not an official pipeline.
- **Latency / 18 Hz / 8-frame chunks:** community teardown and arithmetic, not Tesla disclosures.
- Figures are dense by design. The web page supports click-to-zoom; the WeChat edition relies on tap-to-zoom.

If you have first-hand V15 details, the most useful additions are: UVE pretraining data mix, 4D NVS conditioning, compact-token structure, and teacher/student distillation configuration.

---

## 7. Citation

```
FSD V15 Technical Map, 2026-11-23, v1.0.
```

License: text and original diagrams © 2026. Public source excerpts remain the property of their respective owners. If you republish, please link back to the original sources listed in `sources.md`.
