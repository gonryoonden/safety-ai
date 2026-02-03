#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_upload.py
- Offline QA (fixtures) -> pass only -> generate upload/debug MD
- Fail-fast: QA fail -> report + exit 1
"""
import argparse
import glob
import inspect
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import units_to_md
import json


def _call_units_to_md(argv_list):
    try:
        sig = inspect.signature(units_to_md.main)
        params = list(sig.parameters.values())
    except Exception:
        params = None
    if not params:
        old_argv = sys.argv[:]
        try:
            sys.argv = ["units_to_md.py"] + argv_list
            return units_to_md.main()
        finally:
            sys.argv = old_argv
    return units_to_md.main(argv_list)


def _resolve_units_json(mst, units_json):
    if units_json:
        if not os.path.exists(units_json):
            raise FileNotFoundError(units_json)
        return units_json
    fixture = os.path.join("fixtures", f"{mst}_units.json")
    if os.path.exists(fixture):
        return fixture
    matches = glob.glob(os.path.join("faiss_indexes", f"*{mst}_units.json"))
    if matches:
        # prefer file with required meta
        def _meta_ok(p):
            try:
                import json
                with open(p, "r", encoding="utf-8") as f:
                    units = json.load(f)
                if not units:
                    return False
                for u in units[:50]:
                    if not u.get("law") or not u.get("mst") or not u.get("source_url") or not u.get("source_anchor"):
                        return False
                    if not u.get("source_url_abs"):
                        return False
                    su = str(u.get("source_url_abs") or u.get("source_url"))
                    if ("/DRF/lawService.do" in su) or ("OC=" in su):
                        return False
                return True
            except Exception:
                return False
        for p in matches:
            if _meta_ok(p):
                return p
        # fallback to most recent
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return matches[0]
    raise FileNotFoundError(f"units json not found for MST {mst}")


def _resolve_law_json(mst, law_json):
    if law_json:
        if not os.path.exists(law_json):
            raise FileNotFoundError(law_json)
        return law_json
    local = os.path.join("laws", "raw", f"mst_{mst}_all.json")
    if os.path.exists(local):
        return local
    raise FileNotFoundError(f"law json not found for MST {mst}")


def _copy_if_exists(src, dst_dir):
    if src and os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def _copy_debug_artifacts(mst, units_json, law_json, debug_dir):
    _copy_if_exists(units_json, debug_dir)
    _copy_if_exists(law_json, debug_dir)
    # faiss artifacts if available
    for name in [
        f"{mst}_faiss_index.bin",
        f"{mst}_faiss_id_map.json",
        f"{mst}_answers.json",
    ]:
        _copy_if_exists(os.path.join("faiss_indexes", name), debug_dir)

def _prefix_full_md(mst: str, out_dir: str):
    if not os.path.isdir(out_dir):
        return
    md_files = [f for f in os.listdir(out_dir) if f.lower().endswith(".md")]
    # full file is the one without annex/format markers
    full_candidates = [f for f in md_files if ("_별표_" not in f and "_서식_" not in f)]
    if len(full_candidates) != 1:
        return
    name = full_candidates[0]
    if name.startswith(f"{mst}_"):
        return
    src = os.path.join(out_dir, name)
    dst = os.path.join(out_dir, f"{mst}_{name}")
    try:
        os.replace(src, dst)
    except Exception:
        pass

def _infer_law_title(units, fallback_base):
    for u in units:
        if u.get("law_title"):
            return u.get("law_title")
    for u in units:
        if u.get("law"):
            return u.get("law")
    return fallback_base



def _clean_compact_dir(mst: str, compact_dir: str):
    if not compact_dir or not os.path.isdir(compact_dir):
        return
    prefix = f"{mst}_"
    for name in os.listdir(compact_dir):
        if name.startswith(prefix) and name.lower().endswith(".md"):
            try:
                os.remove(os.path.join(compact_dir, name))
            except Exception:
                pass
def _write_compact_bundle(units_json, out_dir, mst):
    os.makedirs(out_dir, exist_ok=True)
    with open(units_json, "r", encoding="utf-8") as f:
        units = json.load(f)
    law_title = _infer_law_title(units, fallback_base=mst)

    def _is_annex(u):
        return u.get("level") in ("별표", "서식")

    body_units = [u for u in units if not _is_annex(u)]
    annex_units = [u for u in units if _is_annex(u)]

    body_name = units_to_md.safe_filename(f"{mst}_{law_title}_본문.md")
    annex_name = units_to_md.safe_filename(f"{mst}_{law_title}_별표서식.md")
    body_path = os.path.join(out_dir, body_name)
    annex_path = os.path.join(out_dir, annex_name)

    with open(body_path, "w", encoding="utf-8") as f:
        f.write(f"# {law_title}\n\n")
        for u in body_units:
            f.write(units_to_md.render_unit_to_md(u, mst, law_title, mst, front_matter=False, add_source_link=True))
            f.write("\n")

    with open(annex_path, "w", encoding="utf-8") as f:
        f.write(f"# {law_title} 별표/서식\n\n")
        for u in annex_units:
            f.write(units_to_md.render_unit_to_md(u, mst, law_title, mst, front_matter=False, add_source_link=True))
            f.write("\n")

    return [body_path, annex_path]

def _validate_md_links(md_paths):
    bad = []
    for p in md_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    if line.startswith("원문 확인:"):
                        url = line.split(":", 1)[-1].strip()
                        if not (url.startswith("http://") or url.startswith("https://")):
                            bad.append({"path": p, "line": i, "url": url})
                        if "/DRF/lawService.do" in url or "OC=" in url:
                            bad.append({"path": p, "line": i, "url": url})
        except Exception:
            continue
    return bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mst", required=True, help="MST")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-meta-sample", type=int, default=100)
    ap.add_argument("--n-annex-sample", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--law-json", default=None)
    ap.add_argument("--units-json", default=None)
    ap.add_argument("--qa-mode", choices=["full", "fast"], default="full")
    ap.add_argument("--out-root", default=None, help="Root output dir (upload/debug will be under this)")
    ap.add_argument("--upload-dir", default=None)
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--compact", action="store_true", help="Create compact bundle (본문/별표서식)")
    ap.add_argument("--compact-dir", default=None, help="Output dir for compact bundle")
    args = ap.parse_args()

    mst = str(args.mst)
    units_json = _resolve_units_json(mst, args.units_json)
    law_json = None
    if args.qa_mode != "fast" or args.law_json or os.path.exists(os.path.join("laws", "raw", f"mst_{mst}_all.json")):
        law_json = _resolve_law_json(mst, args.law_json)

    if args.out_root:
        upload_dir = os.path.join(args.out_root, "upload", mst)
        debug_dir = os.path.join(args.out_root, "debug", mst)
    else:
        upload_dir = args.upload_dir or os.path.join("dist", "upload")
        debug_dir = args.debug_dir or os.path.join("dist", "debug")

    os.makedirs(debug_dir, exist_ok=True)
    report_path = os.path.join(debug_dir, f"{mst}_validation_report.json")

    compact_dir_pre = None
    if args.compact:
        if args.compact_dir:
            compact_dir_pre = args.compact_dir
        elif args.out_root:
            compact_dir_pre = os.path.join(args.out_root, "knowledge_upload_compact")
        else:
            compact_dir_pre = os.path.join("dist", "knowledge_upload_compact")
        os.makedirs(compact_dir_pre, exist_ok=True)
        _clean_compact_dir(mst, compact_dir_pre)
    if os.path.exists(report_path):
        try:
            os.remove(report_path)
        except Exception:
            pass

    # 1) QA (offline fixture)
    qa_cmd = [
        sys.executable,
        os.path.join("tools", "qa_check.py"),
        "--mst", mst,
        "--seed", str(args.seed),
        "--n-meta-sample", str(args.n_meta_sample),
        "--n-annex-sample", str(args.n_annex_sample),
        "--top-k", str(args.top_k),
    ]
    if law_json:
        qa_cmd += ["--law-json", law_json]
    qa_cmd += [
        "--units-json", units_json,
        "--report-path", report_path,
        "--mode", args.qa_mode,
    ]
    rc = subprocess.run(qa_cmd).returncode
    if rc != 0:
        return rc

    # 2) Upload MD (no front matter, emit both)
    os.makedirs(upload_dir, exist_ok=True)
    rc = _call_units_to_md([
        "--out-dir", upload_dir,
        "--emit", "both",
        "--no-front-matter",
        units_json,
    ])
    if rc:
        return rc
    _prefix_full_md(mst, upload_dir)

    # 2-1) Compact bundle (본문 + 별표/서식)
    if args.compact:
        compact_dir = args.compact_dir
        if not compact_dir:
            if args.out_root:
                compact_dir = os.path.join(args.out_root, "knowledge_upload_compact", mst)
            else:
                compact_dir = os.path.join("dist", "knowledge_upload_compact", mst)
        compact_paths = _write_compact_bundle(units_json, compact_dir, mst)
        bad = _validate_md_links(compact_paths)
        if bad:
            report = {
                "id_check": {"status": "skipped", "reason": "source_ids_missing"},
                "failures": [
                    {
                        "failure_type": "source_url_invalid",
                        "mst": mst,
                        "crawl_ts": None,
                        "seed": None,
                        "source_anchor": None,
                        "display_path_norm": None,
                        "source_url": None,
                        "level": None,
                        "ref": None,
                        "detail": {"bad_links": bad},
                    }
                ],
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            return 1

    # 3) Debug MD (front matter on)
    rc = _call_units_to_md([
        "--out-dir", debug_dir,
        "--emit", "both",
        "--front-matter",
        units_json,
    ])
    if rc:
        return rc
    _prefix_full_md(mst, debug_dir)

    _copy_debug_artifacts(mst, units_json, law_json, debug_dir)

    print("[done] upload:", os.path.abspath(upload_dir))
    print("[done] debug:", os.path.abspath(debug_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
