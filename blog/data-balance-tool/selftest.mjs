// selftest.mjs — 无依赖 mock-DOM 自测：加载 index.html 内联脚本并验证核心交互链路。
// 运行：node demos/data-balance-tool/selftest.mjs
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");

// 提取唯一的内联 <script>（本页无外部 src script）
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) throw new Error("未能在 index.html 中找到内联 <script>");
const scriptBody = m[1];

let failures = 0;
const assert = (cond, msg) => {
  if (cond) { console.log("  ✓ " + msg); }
  else { console.log("  ✗ " + msg); failures++; }
};

/* ---------- 轻量 mock DOM ---------- */
const mockCtx = new Proxy({}, {
  get: (t, p) => (p in t ? t[p] : () => {}),
  set: (t, p, v) => { t[p] = v; return true; },
});

function mkEl() {
  const el = {
    _children: [], _listeners: {}, _html: "",
    style: {}, dataset: {}, value: "", textContent: "", files: [],
    onclick: null, onchange: null, oninput: null, onload: null, onerror: null,
    classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    addEventListener(t, f) { (this._listeners[t] = this._listeners[t] || []).push(f); },
    removeEventListener() {},
    appendChild(c) { this._children.push(c); return c; },
    remove() {},
    click() { if (typeof this.onclick === "function") this.onclick(); },
    focus() {}, setAttribute() {}, getAttribute() { return null; },
    querySelector() { return mkEl(); },
    querySelectorAll() { return []; },
    getContext() { return mockCtx; },
    getBoundingClientRect() { return { width: 900, height: 700, left: 0, top: 0, right: 900, bottom: 700 }; },
  };
  Object.defineProperty(el, "innerHTML", {
    get() { return this._html; },
    set(v) { this._html = v; this._children = []; },  // 设 innerHTML 视为清空子节点
  });
  Object.defineProperty(el, "children", { get() { return this._children; } });
  return el;
}

const elCache = new Map();
const store = {};
let alertCount = 0;

const documentMock = {
  body: mkEl(),
  getElementById(id) { if (!elCache.has(id)) elCache.set(id, mkEl()); return elCache.get(id); },
  createElement() { return mkEl(); },
  addEventListener() {}, removeEventListener() {},
  querySelectorAll() { return []; },
};
const windowMock = { addEventListener() {}, removeEventListener() {}, devicePixelRatio: 1 };

const sandbox = {
  document: documentMock,
  window: windowMock,
  console,
  alert: () => { alertCount++; },
  confirm: () => true,
  localStorage: {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
  },
  Blob: function Blob(parts, opts) { this.parts = parts; this.opts = opts; },
  URL: { createObjectURL: () => "blob:test", revokeObjectURL: () => {} },
  FileReader: class { readAsText() {} },
  setTimeout: (fn) => fn && fn(),
};

/* ---------- 执行真实脚本 ---------- */
console.log("[1] 加载并执行 index.html 内联脚本");
let loadErr = null;
try {
  vm.createContext(sandbox);
  vm.runInContext(scriptBody, sandbox, { filename: "index.html#inline" });
} catch (e) { loadErr = e; }
assert(loadErr === null, "脚本无语法/运行时错误地完成 init()" + (loadErr ? `  → ${loadErr}` : ""));

const $ = (id) => documentMock.getElementById(id);

/* ---------- 交互链路 ---------- */
console.log("[2] 初始渲染");
assert($("palette").children.length === 11, "画笔面板渲染出 11 个类别");

console.log("[3] 载入示例 + 帧级标签");
$("btn-sample").onclick();
assert($("obj-count").textContent === "14 个", "示例载入后物体清单计数为 14 个");
assert($("quota-body").children.length === 11, "配额表渲染出 11 行类别");
const autoHtml = $("auto-tags")._html || "";
assert((autoHtml.match(/✓/g) || []).length === 5, "示例 5 个自动标签(大车/行人/骑行/障碍/红绿灯)全部命中");
assert($("custom-tags").children.length === 2, "自定义标签渲染 2 个(路口/施工区)");
assert($("custom-tags").children.filter(c => c.className.split(/\s+/).includes("on")).length === 1, "示例勾选 1 个自定义标签(路口)");

console.log("[4] 清空物体");
$("btn-clear").onclick();
assert($("obj-count").textContent === "0 个", "清空后计数为 0 个");

console.log("[5] 选画笔 + 画布点击添加物体");
$("palette").children[0].onclick();           // 选中第一个类别（轿车）
const cv = $("bev");
const downs = cv._listeners.mousedown || [];
assert(downs.length >= 1, "canvas 绑定了 mousedown 监听");
downs[0]({ clientX: 450, clientY: 350, target: { tagName: "CANVAS" } });  // 屏幕中心≈world(15,0)
assert($("obj-count").textContent === "1 个", "画布点击后新增 1 个物体");

console.log("[6] 导出当前场景（不应抛错）");
let exportErr = null;
try { $("btn-export").onclick(); } catch (e) { exportErr = e; }
assert(exportErr === null, "导出当前场景未抛错" + (exportErr ? `  → ${exportErr}` : ""));

console.log("[7] 保存场景到本地");
$("btn-save-scene").onclick();
let saved = null;
try { saved = JSON.parse(store["dbt_scenes_v1"]); } catch {}
assert(Array.isArray(saved) && saved.length === 1, "场景已写入 localStorage，且为含 1 条的数组");
assert(saved && saved[0].objects.length === 1, "保存的场景包含 1 个物体");
assert(saved && saved[0].range && saved[0].range.x_min === -50 && saved[0].range.x_max === 80, "场景 range 纵向 [-50,80] 正确");
assert(saved && Array.isArray(saved[0].tags) && saved[0].tags.includes("路口"), "保存的场景含自定义 tag「路口」");
assert(saved && Array.isArray(saved[0].auto_tags), "保存的场景含 auto_tags 数组");

console.log("[8] 添加任意自定义 tag");
$("tag-input").value = "逆光";
$("btn-add-tag").onclick();
assert($("custom-tags").children.length === 3, "添加「逆光」后自定义标签变为 3 个");

/* ---------- 坐标变换互逆（算法层）---------- */
console.log("[9] 坐标变换 w2s/s2w 互逆性");
{
  const RANGE = { x_min:-50, x_max:80, y_min:-50, y_max:50 };
  const W = 900, H = 700, pad = 30;
  const spanX = RANGE.x_max - RANGE.x_min, spanY = RANGE.y_max - RANGE.y_min;
  const scale = Math.min((W - 2*pad)/spanY, (H - 2*pad)/spanX);
  const drawW = spanY*scale, drawH = spanX*scale;
  const left = (W-drawW)/2, top = (H-drawH)/2;
  const originX = left + RANGE.y_max*scale, originY = top + RANGE.x_max*scale;
  const w2s = (x,y)=>({sx:originX-y*scale, sy:originY-x*scale});
  const s2w = (sx,sy)=>({x:(originY-sy)/scale, y:(originX-sx)/scale});
  let maxErr = 0;
  for (const [x,y] of [[0,0],[80,50],[-50,-50],[15,0],[30,-20],[60,12]]) {
    const s = w2s(x,y); const w = s2w(s.sx,s.sy);
    maxErr = Math.max(maxErr, Math.abs(w.x-x), Math.abs(w.y-y));
  }
  assert(maxErr < 1e-9, `w2s→s2w 往返误差<1e-9（实测 ${maxErr.toExponential(2)}）`);
}

console.log("\n" + (failures === 0 ? "全部通过 ✅" : `${failures} 项失败 ❌`));
process.exit(failures === 0 ? 0 : 1);
