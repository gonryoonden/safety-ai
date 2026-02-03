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

def _is_api_url(url: Optional[str]) -> bool:
    if not url:
        return False
    s = str(url)
    return ("/DRF/lawService.do" in s) or ("OC=" in s) or ("/DRF/" in s)

def _pick_source_link(u: Dict[str, Any]) -> Optional[str]:
    for k in ("source_url_abs", "source_url"):
        val = u.get(k)
        if not val:
            continue
        s = str(val).strip()
        if not (s.startswith("http://") or s.startswith("https://")):
            continue
        if _is_api_url(s):
            continue
        return s
    return None

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
    for label, key in (("??", "detail"), ("HTML", "html"), ("PDF", "pdf")):
        url = links.get(key)
        if not url:
            continue
        url = str(url).strip()
        if url.startswith("/"):
            url = "https://www.law.go.kr" + url
        if _is_api_url(url):
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        parts.append(f"[{label}]({url})")
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

def _yaml_quote(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return "null"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val)
    # JSON string is valid YAML scalar for most cases
    if re.search(r"[\s:\[\]\{\}\n-]", s):
        return json.dumps(s, ensure_ascii=False)
    return s

def _yaml_value(val: Any) -> str:
    if isinstance(val, list):
        items = [_yaml_quote(v) for v in val if v is not None and v != ""]
        return "[" + ", ".join(items) + "]"
    return _yaml_quote(val)

def front_matter_md(u: Dict[str, Any]) -> str:
    keys = [
        "law", "mst", "level",
        "article", "article_title", "paragraph",
        "item_path", "item_type",
        "effective_date", "as_of",
        "source_url", "source_anchor",
        "annex_no", "annex_no_human", "annex_id",
        "amended_on", "tags", "refs", "ref_by", "ref_by_display",
        "display_path_norm",
    ]
    lines = ["---"]
    for k in keys:
        if k not in u:
            continue
        v = u.get(k)
        if v is None or v == "" or v == []:
            continue
        lines.append(f"{k}: {_yaml_value(v)}")
    if len(lines) == 1:
        return ""
    lines.append("---")
    return "\n".join(lines)

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

def render_unit_to_md(
    u: Dict[str, Any],
    mst: Optional[str],
    law_title: Optional[str],
    base_name: str,
    add_table_inline_summaries: bool = True,
    front_matter: bool = False,
    add_source_link: bool = False,
) -> str:
    parts: List[str] = []
    if front_matter:
        fm = front_matter_md(u)
        if fm:
            parts.append(fm)
    parts.append(unit_heading_md(u))
    meta = meta_md(u, mst, law_title, base_name)
    if meta:
        parts.append(f"> {meta}")
    txt = (u.get("text") or "").strip()
    if add_table_inline_summaries:
        for s in table_summaries_from_text(txt, max_lines=2):
            if s:
                parts.append(f"- 요약표: {s}")
    if txt:
        parts.append(txt)
    if add_source_link:
        link = _pick_source_link(u)
        if link:
            parts.append(f"원문 확인: {link}")
    return "\n\n".join(parts).rstrip() + "\n"


def chunk_units_by_chars(
    units: List[Dict[str, Any]],
    mst: Optional[str],
    law_title: str,
    base_name: str,
    max_chars: int,
    front_matter: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Greedy split by estimated markdown length per unit, to keep each output under max_chars."""
    buckets: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_len = 0
    for u in units:
        # rough estimate
        est = len(render_unit_to_md(u, mst, law_title, base_name, front_matter=front_matter))
        if cur and (cur_len + est > max_chars):
            buckets.append(cur); cur = []; cur_len = 0
        cur.append(u); cur_len += est
    if cur:
        buckets.append(cur)
    return buckets

def write_md_law_sharded(
    units: List[Dict[str, Any]],
    src_path: str,
    out_dir: str,
    shard_chars: int,
    annex_suffix: str = "",
    annex_only: bool = False,
    front_matter: bool = False,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    base_name_full, mst = detect_base_and_mst_from_filename(src_path)
    # law_title detection
    law_title = None
    for u in units: # Find law_title from the first unit that has it
        if u.get("law_title"): law_title = u.get("law_title"); break
    if not law_title:
        law_title = re.sub(r'_\d+$', '', base_name_full)

    # Prepare annex_no_human
    for u in units:
        if u.get("level") in ("별표","서식") and not u.get("annex_no_human"):
            u["annex_no_human"] = annex_no_human(u.get("annex_no"))

    groups = chunk_units_by_chars(units, mst, law_title, base_name_full, shard_chars, front_matter=front_matter)
    written: List[str] = []
    suffix = annex_suffix if annex_only else ""
    for idx, group in enumerate(groups, start=1):
        fname = f"{law_title}{suffix}_part{idx:02d}.md"
        out_path = os.path.join(out_dir, safe_filename(fname))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# {law_title}\n\n")
            if mst: f.write(f"_MST: {mst}_\n\n")
            for u in group:
                f.write(render_unit_to_md(u, mst, law_title, base_name_full, front_matter=front_matter))
                f.write("\n")
        written.append(out_path)
    return written

def write_markdown_for_law(
    units: List[Dict[str, Any]],
    src_path: str,
    out_dir: str,
    per_annex: bool = False,
    shard_chars: int = 0,
    annex_suffix: str = "",
    annex_only: bool = False,
    front_matter: bool = False,
) -> List[str]:
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
            lvl = u.get("level") or "별표"
            label = "별표" if lvl == "별표" else "서식"
            fname = f"{law_title}_{label}_{a_h or a_code or 'unknown'}.md"
            out_path = os.path.join(out_dir, safe_filename(fname))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"# {law_title}\n\n")
                if mst: f.write(f"_MST: {mst}_\n\n")
                f.write(render_unit_to_md(u, mst, law_title, base_name_full, front_matter=front_matter))
            written.append(out_path)
        return written

    # not per_annex → one file or sharded files
    if shard_chars and shard_chars > 0:
        return write_md_law_sharded(
            units,
            src_path,
            out_dir,
            shard_chars,
            annex_suffix=annex_suffix,
            annex_only=annex_only,
            front_matter=front_matter,
        )

    # single file
    suffix = annex_suffix if annex_only else ""
    fname = f"{law_title}{suffix}.md"
    out_path = os.path.join(out_dir, safe_filename(fname))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {law_title}\n\n")
        if mst: f.write(f"_MST: {mst}_\n\n")
        for u in units:
            f.write(render_unit_to_md(u, mst, law_title, base_name_full, front_matter=front_matter))
            f.write("\n")
    written.append(out_path)
    return written

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Convert *_units.json to Markdown")
    ap.add_argument("inputs", nargs="+", help="Input units.json files (glob allowed)")
    ap.add_argument("--out-dir", default="md_out", help="Output directory for .md files")
    ap.add_argument("--per-annex", action="store_true", help="Write one .md per annex (별표/서식) instead of one per law")
    ap.add_argument("--emit", choices=["full", "annex", "per-annex", "both"], default=None, help="Output mode (overrides --per-annex/--annex-only when set)")
    ap.add_argument("--annex-suffix", default="", help="Suffix for filenames when using --annex-only (and not --per-annex), e.g., _annex")
    ap.add_argument("--min-text-len", type=int, default=0, help="Skip units with text shorter than this length")
    ap.add_argument("--skip-deletion", action="store_true", help="Skip annex units that look like deletion notices")
    ap.add_argument("--annex-only", action="store_true", help="Keep only annex/서식 units (별표/서식) for output")
    ap.add_argument("--shard-chars", type=int, default=0, help="If >0, split each law into multiple .md files under this char size (greedy)")
    fm = ap.add_mutually_exclusive_group()
    fm.add_argument("--front-matter", dest="front_matter", action="store_true", help="Include YAML front matter per unit")
    fm.add_argument("--no-front-matter", dest="front_matter", action="store_false", help="Disable YAML front matter per unit")
    ap.set_defaults(front_matter=True)
    args = ap.parse_args(argv)

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

        def _filter_units(src_units: List[Dict[str, Any]], annex_only: bool = False) -> List[Dict[str, Any]]:
            if not (args.min_text_len or args.skip_deletion or annex_only):
                return src_units
            filtered: List[Dict[str, Any]] = []
            for u in src_units:
                txt = (u.get("text") or "").strip()
                if args.min_text_len and len(txt) < args.min_text_len:
                    continue
                if args.skip_deletion:
                    title = (u.get("title") or "")
                    if ("삭제" in title) and len(txt) < max(120, args.min_text_len):
                        continue
                if annex_only and (u.get('level') not in ('별표','서식')):
                    continue
                filtered.append(u)
            return filtered

        written: List[str] = []
        if args.emit:
            base_units = _filter_units(units, annex_only=False)
            if args.emit in ("full", "both"):
                written.extend(
                    write_markdown_for_law(
                        base_units, path, args.out_dir,
                        per_annex=False,
                        shard_chars=args.shard_chars,
                        annex_suffix=args.annex_suffix,
                        annex_only=False,
                        front_matter=args.front_matter,
                    )
                )
            if args.emit == "annex":
                annex_units = _filter_units(units, annex_only=True)
                written.extend(
                    write_markdown_for_law(
                        annex_units, path, args.out_dir,
                        per_annex=False,
                        shard_chars=args.shard_chars,
                        annex_suffix=args.annex_suffix,
                        annex_only=True,
                        front_matter=args.front_matter,
                    )
                )
            if args.emit in ("per-annex", "both"):
                annex_units = _filter_units(units, annex_only=True)
                written.extend(
                    write_markdown_for_law(
                        annex_units, path, args.out_dir,
                        per_annex=True,
                        shard_chars=args.shard_chars,
                        annex_suffix=args.annex_suffix,
                        annex_only=False,
                        front_matter=args.front_matter,
                    )
                )
        else:
            is_annex_only = args.annex_only
            units_filtered = _filter_units(units, annex_only=is_annex_only)
            written = write_markdown_for_law(
                units_filtered, path, args.out_dir,
                per_annex=args.per_annex,
                shard_chars=args.shard_chars,
                annex_suffix=args.annex_suffix,
                annex_only=is_annex_only,
                front_matter=args.front_matter,
            )
        for w in written:
            print(f"[write] {w}")
        total_written.extend(written)

    print(f"[done] wrote {len(total_written)} files to: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
