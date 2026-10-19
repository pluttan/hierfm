#!/usr/bin/env python3
"""Reader of the ATS fact_region tables with the standard library only.

ATS publishes .xls (BIFF8 inside an OLE2 container) until 2025 and .xlsx from 2025.
`rows(path)` returns the first sheet as a list of rows; each row is a list of
the cells A..C (str, float or None). No xlrd or openpyxl is needed.

    python3 -I src/ats_reader.py FILE [FILE ...]      # prints parsed (code, region, n_hours, first, last)
"""

import re
import struct
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


# ---------------------------------------------------------------- OLE2 container
def _ole_stream(data, name="Workbook"):
    if data[:8] != bytes.fromhex("d0cf11e0a1b11ae1"):
        raise ValueError("not an OLE2 file")
    shift = struct.unpack_from("<H", data, 30)[0]
    mshift = struct.unpack_from("<H", data, 32)[0]
    ssz, msz = 1 << shift, 1 << mshift
    n_fat, dir_start, _, cutoff, minifat_start, n_minifat, difat_start, n_difat = \
        struct.unpack_from("<IIIIIIII", data, 44)
    difat = list(struct.unpack_from("<109I", data, 76))
    sect = difat_start
    for _ in range(n_difat):
        off = 512 + sect * ssz if ssz == 512 else (sect + 1) * ssz
        vals = struct.unpack_from(f"<{ssz // 4}I", data, off)
        difat.extend(vals[:-1])
        sect = vals[-1]

    def sector(i):
        off = (i + 1) * ssz
        return data[off:off + ssz]

    fat = []
    for s in difat[:n_fat]:
        fat.extend(struct.unpack(f"<{ssz // 4}I", sector(s)))

    def chain(start, table):
        out, seen = [], set()
        while start < 0xFFFFFFFA and start not in seen:
            seen.add(start)
            out.append(start)
            start = table[start]
        return out

    def read_chain(start):
        return b"".join(sector(s) for s in chain(start, fat))

    dirs = read_chain(dir_start)
    entry = None
    for k in range(0, len(dirs), 128):
        e = dirs[k:k + 128]
        nlen = struct.unpack_from("<H", e, 64)[0]
        nm = e[:max(nlen - 2, 0)].decode("utf-16-le", "replace")
        if k == 0:
            root_start = struct.unpack_from("<I", e, 116)[0]
        if nm == name:
            entry = (struct.unpack_from("<I", e, 116)[0], struct.unpack_from("<Q", e, 120)[0])
    if entry is None:
        raise ValueError(f"no stream {name}")
    start, size = entry
    if size >= cutoff:
        return read_chain(start)[:size]
    # small stream lives in the mini stream
    mini = read_chain(root_start)
    mfat = []
    for s in chain(minifat_start, fat):
        mfat.extend(struct.unpack(f"<{ssz // 4}I", sector(s)))
    out = b"".join(mini[i * msz:(i + 1) * msz] for i in chain(start, mfat))
    return out[:size]


# ---------------------------------------------------------------- BIFF8
def _records(stream):
    pos = 0
    while pos + 4 <= len(stream):
        rid, ln = struct.unpack_from("<HH", stream, pos)
        yield rid, stream[pos + 4:pos + 4 + ln]
        pos += 4 + ln


def _rk(v):
    if v & 2:
        x = float(struct.unpack("<i", struct.pack("<I", v))[0] >> 2)
    else:
        x = struct.unpack("<d", struct.pack("<Q", (v & 0xFFFFFFFC) << 32))[0]
    return x / 100.0 if v & 1 else x


def _sst(chunks):
    """chunks: payloads of SST and its CONTINUE records. Returns list of str."""
    first = chunks[0]
    total = struct.unpack_from("<II", first, 0)[1]
    ci, pos, cur = 0, 8, first
    out = []

    def need(n):
        nonlocal ci, pos, cur
        if pos >= len(cur) and ci + 1 < len(chunks):
            ci += 1
            cur, pos = chunks[ci], 0
        return n

    for _ in range(total):
        need(1)
        nch, flags = struct.unpack_from("<HB", cur, pos)
        pos += 3
        rich = flags & 8 and struct.unpack_from("<H", cur, pos)[0]
        if flags & 8:
            pos += 2
        ext = flags & 4 and struct.unpack_from("<I", cur, pos)[0]
        if flags & 4:
            pos += 4
        chars = []
        left = nch
        wide = flags & 1
        while left > 0:
            need(1)
            avail = (len(cur) - pos) // (2 if wide else 1)
            take = min(left, avail)
            raw = cur[pos:pos + take * (2 if wide else 1)]
            chars.append(raw.decode("utf-16-le" if wide else "cp1251"))
            pos += len(raw)
            left -= take
            if left > 0:                      # string continues in the next CONTINUE record
                ci += 1
                cur, pos = chunks[ci], 0
                wide = cur[0] & 1
                pos = 1
        pos += 4 * (rich or 0) + (ext or 0)
        out.append("".join(chars))
    return out


def _xls_rows(path):
    stream = _ole_stream(Path(path).read_bytes())
    recs = list(_records(stream))
    # shared strings
    chunks, i = [], 0
    for k, (rid, body) in enumerate(recs):
        if rid == 0x00FC:
            chunks = [body]
            j = k + 1
            while j < len(recs) and recs[j][0] == 0x003C:
                chunks.append(recs[j][1])
                j += 1
            break
    sst = _sst(chunks) if chunks else []
    cells, depth, in_first = {}, 0, False
    for rid, b in recs:
        if rid == 0x0809:                     # BOF
            kind = struct.unpack_from("<H", b, 2)[0]
            if kind == 0x0010:                # worksheet
                if in_first:
                    break
                in_first = True
        if not in_first:
            continue
        if rid == 0x00FD:                     # LABELSST
            r, c, _, idx = struct.unpack_from("<HHHI", b)
            cells[(r, c)] = sst[idx]
        elif rid == 0x0203:                   # NUMBER
            r, c, _ = struct.unpack_from("<HHH", b)
            cells[(r, c)] = struct.unpack_from("<d", b, 6)[0]
        elif rid == 0x027E:                   # RK
            r, c, _, v = struct.unpack_from("<HHHI", b)
            cells[(r, c)] = _rk(v)
        elif rid == 0x00BD:                   # MULRK
            r, c0 = struct.unpack_from("<HH", b)
            c1 = struct.unpack_from("<H", b, len(b) - 2)[0]
            for k in range(c1 - c0 + 1):
                cells[(r, c0 + k)] = _rk(struct.unpack_from("<I", b, 4 + 6 * k + 2)[0])
        elif rid == 0x0204:                   # LABEL
            r, c, _, n = struct.unpack_from("<HHHH", b)
            flags = b[10]
            cells[(r, c)] = (b[11:11 + 2 * n].decode("utf-16-le") if flags & 1
                             else b[11:11 + n].decode("cp1251"))
        elif rid == 0x0006:                   # FORMULA, numeric result only
            r, c, _ = struct.unpack_from("<HHH", b)
            if b[12:14] != b"\xff\xff":
                cells[(r, c)] = struct.unpack_from("<d", b, 6)[0]
    if not cells:
        return []
    nrow = max(r for r, _ in cells) + 1
    return [[cells.get((r, c)) for c in range(3)] for r in range(nrow)]


# ---------------------------------------------------------------- xlsx
def _xlsx_rows(path):
    with zipfile.ZipFile(path) as z:
        sst = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
                sst.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    out = []
    for row in sheet.find("m:sheetData", NS):
        vals = {}
        for c in row:
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find("m:v", NS)
            if v is None:
                continue
            t = c.get("t")
            vals[col] = sst[int(v.text)] if t == "s" else (v.text if t == "str" else float(v.text))
        out.append([vals.get(k) for k in "ABC"])
    return out


def rows(path):
    path = Path(path)
    return _xlsx_rows(path) if path.suffix.lower() == ".xlsx" else _xls_rows(path)


DATE = re.compile(r"\d\d\.\d\d\.\d{4}")


def parse(path):
    """-> (region name, [(dd.mm.yyyy, hour 1..24 or more, mw), ...])."""
    region, out = None, []
    for r in rows(path):
        if isinstance(r[0], str) and r[0].strip().startswith("Субъект РФ"):
            region = r[1]
        elif isinstance(r[0], str) and DATE.fullmatch(r[0].strip()) and isinstance(r[1], float) \
                and isinstance(r[2], float):
            out.append((r[0].strip(), int(r[1]), r[2]))
    return region, out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        reg, d = parse(p)
        print(Path(p).name, reg, len(d), d[:1], d[-1:])
