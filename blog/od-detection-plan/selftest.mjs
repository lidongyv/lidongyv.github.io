import { readFileSync } from "node:fs";

const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");

// extract the last <script> ... </script> block (the render logic)
const m = html.match(/<script>([\s\S]*?)<\/script>\s*<\/body>/);
if (!m) { console.error("FAIL: could not find inline script"); process.exit(1); }
const code = m[1];

// minimal DOM stub
const store = {};
function mkEl(id) {
  return {
    _id: id, _html: "", _href: "",
    set innerHTML(v) { this._html = v; },
    get innerHTML() { return this._html; },
    set href(v) { this._href = v; },
    get href() { return this._href; },
    classList: { add() {} },
  };
}
const ids = ["stats", "heroCta", "feishuCta", "sparseDocs", "temporalDocs", "parkingDocs", "allDocs"];
ids.forEach(i => (store[i] = mkEl(i)));

globalThis.document = {
  getElementById: (id) => store[id] || mkEl(id),
  querySelectorAll: () => [],
};
globalThis.IntersectionObserver = class { constructor() {} observe() {} };

try {
  eval(code);
} catch (e) {
  console.error("FAIL: runtime error ->", e.message);
  process.exit(1);
}

// assertions
let ok = true;
function check(id, mustContain) {
  const h = store[id]._html || store[id]._href || "";
  const pass = h.includes(mustContain);
  console.log(`${pass ? "PASS" : "FAIL"}  #${id} contains "${mustContain}"`);
  if (!pass) ok = false;
}
check("stats", "CIPV 60m");
check("sparseDocs", "GroupedSparse4DHead");
check("temporalDocs", "Stream time fusion");
check("parkingDocs", "角点 CA");
check("allDocs", "感知模型整体方案 Review");
check("allDocs", "角点 CA");
check("feishuCta", "K9apdwRGqoj1kIx38h9cF0Cgnlf");
check("heroCta", "K9apdwRGqoj1kIx38h9cF0Cgnlf");

// structural checks on the HTML itself
const opens = (html.match(/<section/g) || []).length;
console.log(`INFO  <section> count = ${opens}`);
if (opens < 7) { console.log("FAIL  expected >=7 sections (incl. parking)"); ok = false; }
if (!html.includes('id="parking"')) { console.log("FAIL  missing #parking section"); ok = false; }

console.log(ok ? "\nALL CHECKS PASSED" : "\nSOME CHECKS FAILED");
process.exit(ok ? 0 : 1);
