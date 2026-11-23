import { webcrypto as crypto } from "node:crypto";
import fs from "node:fs";

const [file, pw, out] = process.argv.slice(2);
const html = fs.readFileSync(file, "utf8");
const m = html.match(/<script id="wm-payload" type="application\/json">([\s\S]*?)<\/script>/);
if (!m) { console.error("no payload"); process.exit(2); }
const p = JSON.parse(m[1]);
const dec = (s) => Uint8Array.from(Buffer.from(s, "base64"));
const enc = new TextEncoder();
const baseKey = await crypto.subtle.importKey("raw", enc.encode(pw), { name: "PBKDF2" }, false, ["deriveKey"]);
const key = await crypto.subtle.deriveKey(
  { name: "PBKDF2", salt: dec(p.salt), iterations: p.iter, hash: "SHA-256" },
  baseKey, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
try {
  const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv: dec(p.iv) }, key, dec(p.ct));
  fs.writeFileSync(out, Buffer.from(pt));
  console.log("ok bytes=" + pt.byteLength);
} catch (e) {
  console.error("decrypt failed: " + e);
  process.exit(3);
}