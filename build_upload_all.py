#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch build for multiple MSTs:
- QA -> PASS -> upload/debug MD
- FAIL -> report + continue
- Summary JSON + knowledge_upload folder/zip
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
import time
import glob
import hashlib
from typing import Optional, Dict, Any, List

from utils import LawAPIClient, fetch_law_meta_by_name, resolve_law_name
from vector_search_service import build_index_for_mst

ROOT = os.path.dirname(os.path.abspath(__file__))


def _parse_mst_list(arg: str) -> list:
    if not arg:
        return []
    parts = [p.strip() for p in arg.replace(",", " ").split() if p.strip()]
    return parts


def _parse_law_names(arg: str) -> list:
    if not arg:
        return []
    parts = [p.strip() for p in arg.split(",")]
    return [p for p in parts if p]


def _read_mst_file(path: str) -> list:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return _parse_mst_list(txt)


def _collect_md_files(folder: str) -> list:
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(".md")]

def _file_stats(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        return {"chars": len(txt), "lines": txt.count("\n") + 1}
    except Exception:
        return {"chars": None, "lines": None}


def _infer_law_name_from_units(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            units = json.load(f)
        for u in units[:50]:
            if u.get("law_title"):
                return str(u.get("law_title"))
        for u in units[:50]:
            if u.get("law"):
                return str(u.get("law"))
    except Exception:
        return None
    return None


def _load_registry(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"updated_at": None, "laws": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "laws" in data:
            return data
    except Exception:
        pass
    return {"updated_at": None, "laws": []}


def _save_registry(path: str, registry: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def _update_registry(registry: Dict[str, Any], entry: Dict[str, Any]) -> None:
    laws = registry.get("laws") or []
    name = entry.get("law_name")
    updated = False
    for i, it in enumerate(laws):
        if it.get("law_name") == name:
            laws[i] = {**it, **entry}
            updated = True
            break
    if not updated:
        laws.append(entry)
    registry["laws"] = laws
    registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_meta_snapshots(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"snapshots": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "snapshots" in data:
            return data
    except Exception:
        pass
    return {"snapshots": {}}


def _save_meta_snapshots(path: str, data: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _meta_changed(prev: Dict[str, Any], cur: Dict[str, Any]) -> bool:
    if not prev:
        return False
    for k in ("latest_amended", "effective_date", "article_count", "annex_count"):
        pv = prev.get(k)
        cv = cur.get(k)
        if pv is None or cv is None:
            continue
        if str(pv) != str(cv):
            return True
    return False


def _units_meta_ok(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            units = json.load(f)
        if not units:
            return False
        for u in units[:50]:
            if not u.get("law"):
                return False
            if not u.get("mst"):
                return False
            if not u.get("source_url"):
                return False
            if not u.get("source_anchor"):
                return False
            if not u.get("source_url_abs"):
                return False
            su = str(u.get("source_url_abs") or u.get("source_url"))
            if ("/DRF/lawService.do" in su) or ("OC=" in su):
                return False
        return True
    except Exception:
        return False


def _law_hash_from_path(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        payload = json.dumps(data, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        return None


def _units_fast_ok(path: str, law_json_path: Optional[str]) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            units = json.load(f)
        if not units:
            return False
        u0 = units[0]
        units_hash = u0.get("law_hash")
        units_crawl = u0.get("crawl_ts")
        if units_hash and law_json_path and os.path.exists(law_json_path):
            law_hash = _law_hash_from_path(law_json_path)
            if law_hash and units_hash == law_hash:
                return True
        if units_crawl:
            return True
        return False
    except Exception:
        return False


def _law_hash_from_units(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            units = json.load(f)
        if not units:
            return None
        return units[0].get("law_hash")
    except Exception:
        return None


def _copy_upload_to_knowledge(upload_dir: str, knowledge_dir: str, mst: str) -> int:
    if not os.path.isdir(upload_dir):
        return 0
    dst = os.path.join(knowledge_dir, mst)
    os.makedirs(dst, exist_ok=True)
    count = 0
    for name in os.listdir(upload_dir):
        if not name.lower().endswith(".md"):
            continue
        shutil.copy2(os.path.join(upload_dir, name), os.path.join(dst, name))
        count += 1
    return count


def _make_zip(src_dir: str, zip_path: str):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                if not name.lower().endswith(".md"):
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, src_dir)
                zf.write(full, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mst-list", default=None, help="MST list (space/comma separated)")
    ap.add_argument("--mst-file", default=None, help="Text file containing MSTs")
    ap.add_argument("--out-root", default="dist")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-meta-sample", type=int, default=100)
    ap.add_argument("--n-annex-sample", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--zip", action="store_true", help="Create dist/knowledge_upload.zip")
    ap.add_argument("--compact", action="store_true", help="Create compact bundle per MST")
    ap.add_argument("--fast", action="store_true", default=True, help="Fast path (skip rebuild if units are fresh)")
    ap.add_argument("--no-fast", dest="fast", action="store_false", help="Disable fast path")
    ap.add_argument("--force-refresh", action="store_true", help="Force rebuild (override fast path)")
    ap.add_argument("--from-law-names", default=None, help="Comma/space separated law names to resolve MSTs")
    ap.add_argument("--mst-registry", default="mst_registry.json", help="Path to mst_registry.json")
    ap.add_argument("--meta-snapshot", default=None, help="Path to meta snapshot file (default: out_root/meta_snapshots.json)")
    args = ap.parse_args()

    msts: List[str] = []
    if args.mst_list:
        msts.extend(_parse_mst_list(args.mst_list))
    if args.mst_file:
        msts.extend(_read_mst_file(args.mst_file))
    if not msts:
        env_list = os.environ.get("MST_LIST", "")
        msts = _parse_mst_list(env_list)

    registry = _load_registry(args.mst_registry)
    newly_resolved: set = set()
    unresolved_names: List[Dict[str, Any]] = []
    if args.from_law_names:
        law_names = _parse_law_names(args.from_law_names)
        client = LawAPIClient()
        for name in law_names:
            resolved = resolve_law_name(client, name)
            resolved["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if resolved.get("status") == "resolved":
                mst_val = str(resolved.get("mst"))
                if mst_val and mst_val not in msts:
                    msts.append(mst_val)
                # track newly resolved
                prev = None
                for it in registry.get("laws") or []:
                    if it.get("law_name") == name:
                        prev = it
                        break
                if not prev or prev.get("status") != "resolved" or prev.get("mst") != mst_val:
                    newly_resolved.add(mst_val)
                resolved["status"] = "resolved"
                _update_registry(registry, resolved)
            else:
                resolved["status"] = "unresolved"
                unresolved_names.append(resolved)
                _update_registry(registry, resolved)
        _save_registry(args.mst_registry, registry)

    if not msts and not unresolved_names:
        print("[error] no MSTs provided")
        return 2

    summary = {
        "out_root": args.out_root,
        "msts": msts,
        "results": [],
        "unresolved_law_names": unresolved_names,
        "knowledge_upload_dir": os.path.join(args.out_root, "knowledge_upload"),
        "knowledge_upload_zip": os.path.join(args.out_root, "knowledge_upload.zip"),
        "knowledge_upload_compact_dir": os.path.join(args.out_root, "knowledge_upload_compact"),
        "knowledge_upload_compact_zip": os.path.join(args.out_root, "knowledge_upload_compact.zip"),
    }

    knowledge_dir = summary["knowledge_upload_compact_dir"] if args.compact else summary["knowledge_upload_dir"]
    os.makedirs(knowledge_dir, exist_ok=True)

    meta_snapshot_path = args.meta_snapshot or os.path.join(args.out_root, "meta_snapshots.json")
    meta_snapshots = _load_meta_snapshots(meta_snapshot_path)

    any_fail = False
    for item in unresolved_names:
        summary["results"].append({
            "mst": None,
            "status": "SKIP",
            "exit_code": 0,
            "upload_dir": None,
            "debug_dir": None,
            "upload_md_count": 0,
            "knowledge_md_count": 0,
            "report_path": None,
            "id_check_status": "skipped",
            "failure_types": ["unresolved_law_name"],
            "law_name": item.get("law_name"),
            "reason": item.get("reason"),
        })
    client = LawAPIClient()
    for mst in msts:
        mst = str(mst)
        upload_dir = os.path.join(args.out_root, "upload", mst)
        debug_dir = os.path.join(args.out_root, "debug", mst)
        report_path = os.path.join(debug_dir, f"{mst}_validation_report.json")
        force_refresh_mst = args.force_refresh or (mst in newly_resolved)

        law_json_path = os.path.join("laws", "raw", f"mst_{mst}_all.json")

        # ensure units exist
        units_paths = []
        for candidate in [
            os.path.join("fixtures", f"{mst}_units.json"),
            os.path.join("faiss_indexes", f"{mst}_units.json"),
        ]:
            if os.path.exists(candidate):
                units_paths.append(candidate)
        need_rebuild = True
        if units_paths:
            # if existing units missing required meta, rebuild
            need_rebuild = not _units_meta_ok(units_paths[0])
            if args.fast and not force_refresh_mst and not need_rebuild:
                if _units_fast_ok(units_paths[0], law_json_path):
                    need_rebuild = False
                else:
                    need_rebuild = True

        # lightweight meta check (fast path 유지)
        meta_changed = False
        meta_status = None
        meta_snapshot = meta_snapshots.get("snapshots", {}).get(mst, {})
        law_name_for_meta = None
        for it in registry.get("laws") or []:
            if str(it.get("mst")) == mst and it.get("status") == "resolved":
                law_name_for_meta = it.get("resolved_title") or it.get("law_name")
                break
        if not law_name_for_meta and units_paths:
            law_name_for_meta = _infer_law_name_from_units(units_paths[0])

        if args.fast and not force_refresh_mst and law_name_for_meta:
            try:
                meta = fetch_law_meta_by_name(client, law_name_for_meta)
                meta_status = meta.get("status")
                if meta_status == "resolved":
                    if _meta_changed(meta_snapshot, meta):
                        meta_changed = True
                        need_rebuild = True
                    law_hash = _law_hash_from_units(units_paths[0]) if units_paths else None
                    meta_snapshots.setdefault("snapshots", {})[mst] = {
                        "mst": mst,
                        "law_name": law_name_for_meta,
                        "last_checked": meta.get("fetched_at"),
                        "latest_amended": meta.get("latest_amended"),
                        "effective_date": meta.get("effective_date"),
                        "article_count": meta.get("article_count"),
                        "annex_count": meta.get("annex_count"),
                        "law_hash": law_hash,
                        "meta_status": "resolved",
                        "meta_source": "lawSearch",
                    }
                else:
                    meta_status = "unresolved"
            except Exception:
                meta_status = "error"
        if meta_changed:
            force_refresh_mst = True
        if need_rebuild:
            try:
                build_index_for_mst(mst, out_dir="faiss_indexes")
            except Exception as e:
                os.makedirs(debug_dir, exist_ok=True)
                report = {
                    "id_check": {"status": "skipped", "reason": "source_ids_missing"},
                    "failures": [{
                        "failure_type": "precheck_failed",
                        "mst": mst,
                        "crawl_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "seed": args.seed,
                        "source_anchor": None,
                        "display_path_norm": None,
                        "source_url": None,
                        "level": None,
                        "ref": None,
                        "detail": {"reason": "units_build_failed", "error": str(e)},
                    }],
                }
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                summary["results"].append({
                    "mst": mst,
                    "status": "FAIL",
                    "exit_code": 1,
                    "upload_dir": upload_dir,
                    "debug_dir": debug_dir,
                    "upload_md_count": 0,
                    "knowledge_md_count": 0,
                    "report_path": report_path,
                    "id_check_status": "skipped",
                    "failure_types": ["precheck_failed"],
                })
                any_fail = True
                continue

        # ensure law_json exists only when full QA is needed
        qa_mode = "full" if force_refresh_mst else ("fast" if args.fast else "full")
        if qa_mode == "full" and not os.path.exists(law_json_path):
            try:
                os.makedirs(os.path.dirname(law_json_path), exist_ok=True)
                data = client.get_law(mst)
                with open(law_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                os.makedirs(debug_dir, exist_ok=True)
                report = {
                    "id_check": {"status": "skipped", "reason": "source_ids_missing"},
                    "failures": [{
                        "failure_type": "precheck_failed",
                        "mst": mst,
                        "crawl_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "seed": args.seed,
                        "source_anchor": None,
                        "display_path_norm": None,
                        "source_url": None,
                        "level": None,
                        "ref": None,
                        "detail": {"reason": "law_json_fetch_failed", "error": str(e)},
                    }],
                }
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                summary["results"].append({
                    "mst": mst,
                    "status": "FAIL",
                    "exit_code": 1,
                    "upload_dir": upload_dir,
                    "debug_dir": debug_dir,
                    "upload_md_count": 0,
                    "knowledge_md_count": 0,
                    "report_path": report_path,
                    "id_check_status": "skipped",
                    "failure_types": ["precheck_failed"],
                })
                any_fail = True
                continue

        units_json_for_build = None
        candidates = glob.glob(os.path.join("faiss_indexes", f"*{mst}_units.json"))
        if candidates:
            # prefer units that already contain required meta
            for p in candidates:
                if _units_meta_ok(p):
                    units_json_for_build = p
                    break
            if not units_json_for_build:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                units_json_for_build = candidates[0]

        cmd = [
            sys.executable,
            os.path.join(ROOT, "build_upload.py"),
            "--mst", mst,
            "--out-root", args.out_root,
            "--seed", str(args.seed),
            "--n-meta-sample", str(args.n_meta_sample),
            "--n-annex-sample", str(args.n_annex_sample),
            "--top-k", str(args.top_k),
        ]
        if qa_mode == "fast":
            cmd += ["--qa-mode", "fast"]
        if units_json_for_build:
            cmd += ["--units-json", units_json_for_build]
        if os.path.exists(law_json_path):
            cmd += ["--law-json", law_json_path]
        if args.compact:
            cmd += ["--compact", "--compact-dir", summary["knowledge_upload_compact_dir"]]
        rc = subprocess.run(cmd).returncode

        md_count = len(_collect_md_files(upload_dir))
        if args.compact:
            compact_dir = summary["knowledge_upload_compact_dir"]
            compact_files = [f for f in _collect_md_files(compact_dir) if f.startswith(f"{mst}_")]
            knowledge_count = len(compact_files)
        else:
            knowledge_count = _copy_upload_to_knowledge(upload_dir, knowledge_dir, mst) if rc == 0 else 0

        if rc == 0 and mst not in meta_snapshots.get("snapshots", {}):
            law_hash = _law_hash_from_units(units_json_for_build) if units_json_for_build else None
            meta_snapshots.setdefault("snapshots", {})[mst] = {
                "mst": mst,
                "law_name": law_name_for_meta,
                "last_checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "latest_amended": None,
                "effective_date": None,
                "article_count": None,
                "annex_count": None,
                "law_hash": law_hash,
                "meta_status": "initialized",
                "meta_source": "units_only",
            }

        result = {
            "mst": mst,
            "status": "PASS" if rc == 0 else "FAIL",
            "exit_code": rc,
            "upload_dir": upload_dir,
            "debug_dir": debug_dir,
            "upload_md_count": md_count,
            "knowledge_md_count": knowledge_count,
            "report_path": report_path if (rc != 0 and os.path.exists(report_path)) else None,
            "id_check_status": None,
            "failure_types": [],
            "compact_files": [],
            "is_new_mst": mst in newly_resolved,
            "meta_changed": meta_changed,
            "meta_status": meta_status,
            "meta_snapshot_path": meta_snapshot_path,
        }
        if args.compact and rc == 0:
            compact_dir = summary["knowledge_upload_compact_dir"]
            for name in _collect_md_files(compact_dir):
                if not name.startswith(f"{mst}_"):
                    continue
                p = os.path.join(compact_dir, name)
                result["compact_files"].append({"path": p, **_file_stats(p)})
        if result["report_path"]:
            try:
                with open(result["report_path"], "r", encoding="utf-8") as f:
                    report = json.load(f)
                result["id_check_status"] = (report.get("id_check") or {}).get("status")
                failures = report.get("failures") or []
                result["failure_types"] = sorted({f.get("failure_type") for f in failures if f.get("failure_type")})
            except Exception:
                pass

        summary["results"].append(result)
        if rc != 0:
            any_fail = True

    # write summary
    os.makedirs(args.out_root, exist_ok=True)
    summary_path = os.path.join(args.out_root, "summary_build.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _save_meta_snapshots(meta_snapshot_path, meta_snapshots)

    if args.zip:
        zip_path = summary["knowledge_upload_compact_zip"] if args.compact else summary["knowledge_upload_zip"]
        _make_zip(knowledge_dir, zip_path)

    print("[done] summary:", summary_path)
    print("[done] knowledge_upload:", knowledge_dir)
    if args.zip:
        print("[done] knowledge_upload.zip:", zip_path)

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
