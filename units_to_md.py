#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert law *_units.json (including annex/별표/서식) into Markdown .md files
for use in RAG knowledge bases (ChatGPT Projects, Gemini Gems, etc.).

Ways to reduce the number of files:
  - (default) One .md per law (NO --per-annex)
  - Split into a FEW parts with --shard-chars N (e.g., 300000 for ~300k chars per file)

Usage examples:
  python units_to_md.py faiss_indexes/*_units.json
  python units_to_md.py --out-dir md_out faiss_indexes/산업안전보건기준에*units.json
  python units_to_md.py --shard-chars 300000 --out-dir md_out faiss_indexes/*_units.json
  python units_to_md.py --per-annex --out-dir md_out faiss_indexes/*_units.json

Windows PowerShell:
  python .\\units_to_md.py --out-dir md_out .\\faiss_indexes\\*_units.json
"""

import os, re, json, argparse, glob, sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

def safe_filename(name: str) -> str:
    return re.sub(r'[\/:*?"<>|]+', '_', name).strip()

def annex_no_human(code: Optional[str]) -> str:
    if not code:
        return ""
    code = str(code).zfill(6)
    a = int(code[:4]); b = int(code[4:])
    return f"{a}의{b}" if b else f"{a}"

def detect_base_and_mst_from_filename(path: str) -> Tuple[str, Optional[str]]:
    base = os.path.basename(path)
    m = re.match(r'(.+?)_(\d+)_units\.json$', base)
    if m:
        return m.group(1) + "_" + m.group(2), m.group(2)
    return os.path.splitext(base)[0], None

def unit_heading_md(u: Dict[str, Any]) -> str:
    lvl = (u.get("level") or "").strip()
    title = (u.get("title") or "").strip()
    jo = u.get("jo"); hang = u.get("hang"); mok = u.get("mok")
    if lvl == "조":
        if jo: return f"## 제{jo}조 {title}".rstrip()
        return f"## {title}".rstrip()
    if lvl == "항":
        return f"### 제{hang}항 {title}".rstrip()
    if lvl == "목":
        prefix = f"{mok}. " if mok else ""
        return f"#### {prefix}{title}".rstrip()
    if lvl in ("별표","서식"):
        a_code = (u.get("annex_no") or "")
        a_human = u.get("annex_no_human") or annex_no_human(a_code)
        label = "별표" if lvl == "별표" else "서식"
        num = a_human or a_code or ""
        return f"## {label} {num} {title}".rstrip()
    if lvl == "부칙":
        return f"## 부칙 {title}".rstrip()
    return f"## {title}".rstrip() if title else "## "

def links_md(u: Dict[str, Any]) -> str:
    links = u.get("links") or {}
    parts = []
    if links.get("detail"): parts.append(f"[상세]({links['detail']})")
    if links.get("html"):   parts.append(f"[HTML]({links['html']})")
    if links.get("pdf"):    parts.append(f"[PDF]({links['pdf']})")
    return " / ".join(parts)

def meta_md(u: Dict[str, Any], mst: Optional[str], law_title: Optional[str], base_name: str) -> str:
    parts = []
    if law_title: parts.append(f"법령명: {law_title}")
    if mst: parts.append(f"MST: {mst}")
    lvl = (u.get("level") or "").strip()
    if lvl in ("별표","서식"):
        a_code = (u.get("annex_no") or "")
        a_human = u.get("annex_no_human") or annex_no_human(a_code)
        if a_code: parts.append(f"번호: {a_code} (사람표기: {a_human})")
    vf = u.get("valid_from"); vt = u.get("valid_to")
    if vf or vt: parts.append(f"효력: {vf or '-'} ~ {vt or '-'}")
    src = ((u.get("_annex_meta") or {}).get("extracted_from"))
    if src: parts.append(f"원본: {src}")
    linkline = links_md(u)
    if linkline: parts.append(f"링크: {linkline}")
    return " / ".join(parts)

def table_summaries_from_text(md_text: str, max_lines: int = 2) -> List[str]:
    lines = (md_text or "").splitlines()
    out: List[str] = []; buf: List[str] = []; inside_table = False
    for ln in lines:
        if ln.strip().startswith("|") and ln.strip().endswith("|"):
            inside_table = True; buf.append(ln.strip())
        else:
            if inside_table and buf:
                out.append(" ".join(buf[:max_lines]))
                buf = []; inside_table = False
    if inside_table and buf:
        out.append(" ".join(buf[:max_lines]))
    return out

def render_unit_to_md(u: Dict[str, Any], mst: Optional[str], law_title: Optional[str], base_name: str, add_table_inline_summaries: bool = True) -> str:
    parts: List[str] = []
    parts.append(unit_heading_md(u))
    meta = meta_md(u, mst, law_title, base_name)
    if meta: parts.append(f"> {meta}")
    txt = (u.get("text") or "").strip()
    if add_table_inline_summaries:
        for s in table_summaries_from_text(txt, max_lines=2):
            if s: parts.append(f"- 표요약: {s}")
    if txt: parts.append(txt)
    return "\n\n".join(parts).rstrip() + "\n"

def chunk_units_by_chars(units: List[Dict[str, Any]], mst: Optional[str], law_title: str, base_name: str, max_chars: int) -> List[List[Dict[str, Any]]]:
    """Greedy split by estimated markdown length per unit, to keep each output under max_chars."""
    buckets: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_len = 0
    for u in units:
        # rough estimate
        est = len(render_unit_to_md(u, mst, law_title, base_name))
        if cur and (cur_len + est > max_chars):
            buckets.append(cur); cur = []; cur_len = 0
        cur.append(u); cur_len += est
    if cur:
        buckets.append(cur)
    return buckets

def write_md_law_sharded(units: List[Dict[str, Any]], src_path: str, out_dir: str, shard_chars: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    base_name_full, mst = detect_base_and_mst_from_filename(src_path)
    # law_title detection
    law_title = None
    for u in units:
        if u.get("law_title"):
            law_title = u.get("law_title"); break
    if not law_title:
        law_title = re.sub(r'_\d+$', '', base_name_full)

    # Prepare annex_no_human
    for u in units:
        if u.get("level") in ("별표","서식") and not u.get("annex_no_human"):
            u["annex_no_human"] = annex_no_human(u.get("annex_no"))

    groups = chunk_units_by_chars(units, mst, law_title, base_name_full, shard_chars)
    written: List[str] = []
    for idx, group in enumerate(groups, start=1):
        fname = f"{law_title}_part{idx:02d}.md"
        out_path = os.path.join(out_dir, safe_filename(fname))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# {law_title}\n\n")
            if mst: f.write(f"_MST: {mst}_\n\n")
            for u in group:
                f.write(render_unit_to_md(u, mst, law_title, base_name_full))
                f.write("\n")
        written.append(out_path)
    return written

def write_markdown_for_law(units: List[Dict[str, Any]], src_path: str, out_dir: str, per_annex: bool = False, shard_chars: int = 0) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    base_name_full, mst = detect_base_and_mst_from_filename(src_path)
    law_title = None
    for u in units:
        if u.get("law_title"):
            law_title = u.get("law_title"); break
    if not law_title:
        law_title = re.sub(r'_\d+$', '', base_name_full)

    # Prepare annex_no_human
    for u in units:
        if u.get("level") in ("별표","서식") and not u.get("annex_no_human"):
            u["annex_no_human"] = annex_no_human(u.get("annex_no"))

    written: List[str] = []
    if per_annex:
        annexes = [u for u in units if u.get("level") in ("별표","서식")]
        for u in annexes:
            a_code = u.get("annex_no") or ""
            a_h = u.get("annex_no_human") or annex_no_human(a_code)
            fname = f"{law_title}_별표_{a_h or a_code or 'unknown'}.md"
            out_path = os.path.join(out_dir, safe_filename(fname))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"# {law_title}\n\n")
                if mst: f.write(f"_MST: {mst}_\n\n")
                f.write(render_unit_to_md(u, mst, law_title, base_name_full))
            written.append(out_path)
        return written

    # not per_annex → one file or sharded files
    if shard_chars and shard_chars > 0:
        return write_md_law_sharded(units, src_path, out_dir, shard_chars)

    # single file
    fname = f"{law_title}.md"
    out_path = os.path.join(out_dir, safe_filename(fname))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {law_title}\n\n")
        if mst: f.write(f"_MST: {mst}_\n\n")
        for u in units:
            f.write(render_unit_to_md(u, mst, law_title, base_name_full))
            f.write("\n")
    written.append(out_path)
    return written

def main():
    ap = argparse.ArgumentParser(description="Convert *_units.json to Markdown")
    ap.add_argument("inputs", nargs="+", help="Input units.json files (glob allowed)")
    ap.add_argument("--out-dir", default="md_out", help="Output directory for .md files")
    ap.add_argument("--per-annex", action="store_true", help="Write one .md per annex (별표/서식) instead of one per law")
    ap.add_argument("--min-text-len", type=int, default=0, help="Skip units with text shorter than this length")
    ap.add_argument("--skip-deletion", action="store_true", help="Skip annex units that look like deletion notices")
    ap.add_argument("--annex-only", action="store_true", help="Keep only annex/서식 units (별표/서식) for output")
    ap.add_argument("--shard-chars", type=int, default=0, help="If >0, split each law into multiple .md files under this char size (greedy)")
    args = ap.parse_args()

    paths: List[str] = []
    for pat in args.inputs:
        matches = glob.glob(pat)
        if not matches:
            print(f"[warn] no match for: {pat}")
        paths.extend(matches)
    if not paths:
        print("[error] no input files"); sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)
    total_written: List[str] = []

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                units = json.load(f)
        except Exception as e:
            print(f"[error] failed to read {path}: {e}")
            continue

        if args.min_text_len or args.skip_deletion:
            filtered = []
            for u in units:
                txt = (u.get("text") or "").strip()
                if args.min_text_len and len(txt) < args.min_text_len:
                    continue
                if args.skip_deletion:
                    title = (u.get("title") or "")
                    if ("삭제" in title) and len(txt) < max(120, args.min_text_len):
                        continue
                filtered.append(u)
            units = filtered
        # --annex-only filter
        if args.annex_only:
            units = [u for u in units if (u.get('level') in ('별표','서식'))]

        written = write_markdown_for_law(units, path, args.out_dir, per_annex=args.per_annex, shard_chars=args.shard_chars)
        for w in written:
            print(f"[write] {w}")
        total_written.extend(written)

    print(f"[done] wrote {len(total_written)} files to: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
