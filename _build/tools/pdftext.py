"""Minimal pure-python PDF text extractor for Skia/Chrome-printed PDFs.

Handles: classic xref PDFs (no object streams), FlateDecode content streams,
Type0/CIDFontType2 fonts with ToUnicode CMaps, Type3 fonts with ToUnicode.
"""
import re
import sys
import zlib
import io


OBJ_RE = re.compile(rb'(\d+)\s+(\d+)\s+obj\b')


def iter_objects(data):
    """Yield (objnum, dict_bytes, stream_bytes_or_None)."""
    pos = 0
    n = len(data)
    while True:
        m = OBJ_RE.search(data, pos)
        if not m:
            return
        objnum = int(m.group(1))
        start = m.end()
        end = data.find(b'endobj', start)
        if end < 0:
            return
        body = data[start:end]
        stream = None
        si = body.find(b'stream')
        if si >= 0:
            dic = body[:si]
            p = si + 6
            if body[p:p + 2] == b'\r\n':
                p += 2
            elif body[p:p + 1] in (b'\n', b'\r'):
                p += 1
            e = body.rfind(b'endstream')
            if e < 0:
                e = len(body)
            stream = body[p:e]
            if stream.endswith(b'\r\n'):
                stream = stream[:-2]
            elif stream.endswith(b'\n') or stream.endswith(b'\r'):
                stream = stream[:-1]
        else:
            dic = body
        yield objnum, dic, stream
        pos = end + 6


def inflate(raw, dic):
    if raw is None:
        return b''
    if b'FlateDecode' not in dic:
        return raw
    try:
        return zlib.decompress(raw)
    except Exception:
        try:
            return zlib.decompressobj().decompress(raw)
        except Exception:
            return b''


# ---------------- CMap ----------------

def parse_cmap(text):
    """Parse a ToUnicode CMap -> {code_bytes_len: {code: unicode_str}}"""
    out = {}
    for m in re.finditer(rb'begincmap(.*?)endcmap', text, re.S):
        cm = m.group(1)
    # codespacerange
    space = []
    for m in re.finditer(rb'begincodespacerange(.*?)endcodespacerange', text, re.S):
        for a, b in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', m.group(1)):
            space.append((len(a) // 2, int(a, 16), int(b, 16)))

    def dec(hexs):
        bs = bytes.fromhex(hexs.decode())
        if len(bs) % 2 == 0 and len(bs) >= 2:
            try:
                return bs.decode('utf-16-be')
            except Exception:
                return bs.decode('latin-1')
        return bs.decode('latin-1')

    for m in re.finditer(rb'beginbfchar(.*?)endbfchar', text, re.S):
        for src, dst in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', m.group(1)):
            n = len(src) // 2
            out.setdefault(n, {})[int(src, 16)] = dec(dst)
    for m in re.finditer(rb'beginbfrange(.*?)endbfrange', text, re.S):
        blk = m.group(1)
        for lo, hi, dst in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
            n = len(lo) // 2
            a, b = int(lo, 16), int(hi, 16)
            base = int(dst, 16)
            for i in range(b - a + 1):
                try:
                    out.setdefault(n, {})[a + i] = chr(base + i)
                except Exception:
                    pass
        # array form: <lo> <hi> [ <d1> <d2> ... ]
        for lo, hi, arr in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[(.*?)\]', blk, re.S):
            n = len(lo) // 2
            a, b = int(lo, 16), int(hi, 16)
            items = re.findall(rb'<([0-9A-Fa-f]+)>', arr)
            for i, it in enumerate(items):
                if a + i > b:
                    break
                out.setdefault(n, {})[a + i] = dec(it)
    return out, space


# ---------------- content stream text ----------------

def unescape_pdf_string(s):
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i:i + 1]
        if c == b'\\':
            nxt = s[i + 1:i + 2]
            if nxt in (b'n',):
                out += b'\n'; i += 2; continue
            if nxt == b'r':
                out += b'\r'; i += 2; continue
            if nxt == b't':
                out += b'\t'; i += 2; continue
            if nxt == b'b':
                out += b'\b'; i += 2; continue
            if nxt == b'f':
                out += b'\f'; i += 2; continue
            if nxt in (b'(', b')', b'\\'):
                out += nxt; i += 2; continue
            m = re.match(rb'[0-7]{1,3}', s[i + 1:i + 4])
            if m:
                out.append(int(m.group(0), 8) & 0xFF)
                i += 1 + len(m.group(0)); continue
            i += 2; continue
        out += c; i += 1
    return bytes(out)


TOKEN_RE = re.compile(
    rb'\((?:\\.|[^\\()])*\)'      # literal string
    rb'|<[0-9A-Fa-f\s]*>'          # hex string
    rb'|\[(?:[^\[\]\\]|\\.)*\]'    # array
    rb'|/[^\s/\[\]()<>{}]+'        # name
    rb'|-?\d*\.?\d+'               # number
    rb'|[A-Za-z\'"*]+',            # operator
    re.S)


def decode_str_bytes(raw, cmap, space):
    """Turn raw byte string into text using cmap; choose code length."""
    if not cmap:
        return raw.decode('latin-1', 'replace')
    lens = sorted(cmap.keys())
    mapped = {}
    for L in lens:
        mapped.update(cmap[L])
    # prefer 2-byte if cmap has 2-byte entries and bytes decode cleanly
    use_len = None
    if space:
        use_len = space[0][0]
    if use_len is None:
        if 2 in cmap and 1 not in cmap:
            use_len = 2
        elif 1 in cmap and 2 not in cmap:
            use_len = 1
        else:
            use_len = 2 if len(raw) % 2 == 0 else 1
    out = []
    i = 0
    while i < len(raw):
        chunk = raw[i:i + use_len]
        if len(chunk) < use_len:
            break
        code = int.from_bytes(chunk, 'big')
        ch = mapped.get(code)
        if ch is None:
            if use_len == 2:
                # try 1-byte fallback
                c1 = raw[i]
                ch = mapped.get(c1)
                if ch is not None:
                    out.append(ch); i += 1; continue
            ch = ''
        out.append(ch)
        i += use_len
    return ''.join(out)


def extract_stream_text(content, fontmap):
    """content: bytes of content stream. fontmap: name -> cmap dict."""
    tokens = [m.group(0) for m in TOKEN_RE.finditer(content)]
    cur_font = None
    out = []
    stack = []
    last_y = None
    i = 0
    n = len(tokens)

    def cmap_for(name):
        entry = fontmap.get(name)
        if not entry:
            return None, None
        return entry

    while i < n:
        t = tokens[i]
        if t.startswith(b'/'):
            stack.append(t)
        elif re.fullmatch(rb'-?\d*\.?\d+', t):
            stack.append(float(t))
        elif t.startswith(b'('):
            s = unescape_pdf_string(t[1:-1])
            cm, sp = cmap_for(cur_font)
            out.append(decode_str_bytes(s, cm, sp))
        elif t.startswith(b'<') and not t.startswith(b'<<'):
            hexs = re.sub(rb'\s', b'', t[1:-1])
            if len(hexs) % 2:
                hexs += b'0'
            try:
                s = bytes.fromhex(hexs.decode())
            except Exception:
                s = b''
            cm, sp = cmap_for(cur_font)
            out.append(decode_str_bytes(s, cm, sp))
        elif t.startswith(b'['):
            # TJ array: strings and numbers; negative numbers imply kerning (space)
            inner = t[1:-1]
            for sub in TOKEN_RE.finditer(inner):
                tk = sub.group(0)
                if tk.startswith(b'('):
                    s = unescape_pdf_string(tk[1:-1])
                    cm, sp = cmap_for(cur_font)
                    out.append(decode_str_bytes(s, cm, sp))
                elif tk.startswith(b'<'):
                    hexs = re.sub(rb'\s', b'', tk[1:-1])
                    if len(hexs) % 2:
                        hexs += b'0'
                    try:
                        s = bytes.fromhex(hexs.decode())
                    except Exception:
                        s = b''
                    cm, sp = cmap_for(cur_font)
                    out.append(decode_str_bytes(s, cm, sp))
                elif re.fullmatch(rb'-?\d*\.?\d+', tk):
                    try:
                        v = float(tk)
                    except Exception:
                        continue
                    if v < -80:
                        out.append(' ')
        elif t == b'Tf':
            # /Name size Tf
            for back in reversed(stack):
                if isinstance(back, bytes) and back.startswith(b'/'):
                    cur_font = back[1:].decode('latin-1')
                    break
        elif t in (b'Td', b'TD'):
            nums = [x for x in stack[-2:] if isinstance(x, float)]
            if len(nums) == 2:
                ty = nums[1]
                if abs(ty) > 0.5:
                    out.append('\n')
                elif nums[0] > 0:
                    out.append(' ')
        elif t == b'T*':
            out.append('\n')
        elif t == b'Tm':
            out.append('\n')
        elif t in (b"Tj", b"'", b'"'):
            out.append('')
        elif t == b'ET':
            out.append('')
        if t in (b'Tf', b'Td', b'TD', b'Tm', b'T*', b'TJ', b'Tj', b"'", b'"', b'cm', b're', b'Do', b'gs', b'q', b'Q') and t not in (b'Tf',):
            stack = []
        elif t in (b'Tf',):
            stack = []
        i += 1
    return ''.join(out)


def main(path):
    data = open(path, 'rb').read()
    objs = {}
    for num, dic, stream in iter_objects(data):
        objs[num] = (dic, stream)

    def get_ref(dic, key):
        m = re.search(key + rb'\s+(\d+)\s+0\s+R', dic)
        return int(m.group(1)) if m else None

    # build font cmaps per object
    font_cmaps = {}
    for num, (dic, stream) in objs.items():
        if b'/Type' in dic and b'/Font' in dic:
            tu = get_ref(dic, b'/ToUnicode')
            base = get_ref(dic, b'/DescendantFonts')
            cmap = {}
            space = []
            if tu and tu in objs:
                d2, s2 = objs[tu]
                raw = s2 if s2 is not None else b''
                if b'FlateDecode' in d2:
                    raw = inflate(raw, d2)
                cmap, space = parse_cmap(raw)
            font_cmaps[num] = (cmap, space)

    # pages
    pages = []
    for num, (dic, stream) in objs.items():
        if re.search(rb'/Type\s*/Page[^s]', dic):
            pages.append((num, dic))

    def page_order():
        # try /Pages /Kids order
        for num, (dic, stream) in objs.items():
            if re.search(rb'/Type\s*/Pages', dic):
                kids = re.findall(rb'(\d+)\s+0\s+R', dic)
                return [int(k) for k in kids]
        return []

    order = page_order()
    pagemap = dict(pages)
    ordered = [(k, pagemap[k]) for k in order if k in pagemap] or pages

    results = []
    for num, dic in ordered:
        # fonts in resources
        fontmap = {}
        res = dic
        rm = get_ref(dic, b'/Resources')
        if rm and rm in objs:
            res = objs[rm][0]
        fm = get_ref(res, b'/Font')
        fdict = res
        if fm and fm in objs:
            fdict = objs[fm][0]
        for name, ref in re.findall(rb'/([A-Za-z0-9_.+-]+)\s+(\d+)\s+0\s+R', fdict):
            fontmap[name.decode()] = font_cmaps.get(int(ref), ({}, []))
        # contents
        contents = b''
        cm = re.search(rb'/Contents\s+(\d+)\s+0\s+R', dic)
        if cm:
            cnum = int(cm.group(1))
            if cnum in objs:
                d2, s2 = objs[cnum]
                contents = inflate(s2, d2)
        elif re.search(rb'/Contents\s*\[', dic):
            m = re.search(rb'/Contents\s*\[(.*?)\]', dic, re.S)
            for ref in re.findall(rb'(\d+)\s+0\s+R', m.group(1)):
                cnum = int(ref)
                if cnum in objs:
                    d2, s2 = objs[cnum]
                    contents += inflate(s2, d2) + b'\n'
        txt = extract_stream_text(contents, fontmap)
        results.append(txt)
    return results


if __name__ == '__main__':
    p = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    pages = main(p)
    text = '\n\n===== PAGE BREAK =====\n\n'.join(pages)
    if out:
        io.open(out, 'w', encoding='utf-8').write(text)
        print('WROTE', out, len(text), 'chars', len(pages), 'pages')
    else:
        print(text)
