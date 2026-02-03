import argparse
import pathlib
import glob
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizers import _normalize_text
from vector_search_service import (
    _make_failure_entry,
    _sanitize_vec,
    _validate_units_integrity,
    embed_text,
)


def _load_units_path(out_dir: str, mst: str, units_json_path: Optional[str] = None) -> str:
    if units_json_path:
        if not os.path.exists(units_json_path):
            raise FileNotFoundError(units_json_path)
        return units_json_path
    pattern = os.path.join(out_dir, f"*{mst}_units.json")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"units json not found for MST {mst} in {out_dir}")
    # prefer exact mst prefix if exists
    for p in matches:
        if os.path.basename(p).startswith(f"{mst}_"):
            return p
    return matches[0]


def _load_law_json(mst: str, law_json_path: Optional[str] = None) -> Optional[Dict]:
    if law_json_path:
        if not os.path.exists(law_json_path):
            raise FileNotFoundError(law_json_path)
        with open(law_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    local_path = os.path.join("laws", "raw", f"mst_{mst}_all.json")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _sample_units(units: List[Dict], seed: int, n: int) -> List[Dict]:
    if n >= len(units):
        return units[:]
    rng = random.Random(seed)
    return rng.sample(units, n)


def _item_path_in_text(item_path: str, text: str) -> bool:
    if not item_path:
        return True
    parts = str(item_path).split("-")
    if not parts:
        return False
    tail = parts[-1]
    if not re.match(r"^[\uac00-\ud7a3]$", tail):
        return False
    head = (text or "").strip()
    return (
        head.startswith(tail + ".")
        or head.startswith(tail + ")")
        or head.startswith(tail + " ")
    )


def _qa_checks(
    units: List[Dict],
    mst: str,
    seed: int,
    n_meta_sample: int,
    n_annex_sample: int,
    top_k: int,
    law_json_path: Optional[str] = None,
    mode: str = "full",
) -> Tuple[List[Dict], Dict]:
    failures: List[Dict] = []
    summary: Dict[str, int] = {}
    crawl_ts = None
    try:
        import time
        crawl_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    except Exception:
        crawl_ts = None

    sample = _sample_units(units, seed, n_meta_sample)

    # 1) required meta check
    missing_required = 0
    for u in sample:
        missing = []
        if not u.get("law"):
            missing.append("law")
        if not u.get("mst"):
            missing.append("mst")
        if not u.get("article"):
            missing.append("article")
        if not u.get("annex_placeholder") and not (u.get("effective_date") or u.get("as_of")):
            missing.append("effective_date/as_of")
        if not u.get("source_url"):
            missing.append("source_url")
        if not u.get("source_url_abs"):
            missing.append("source_url_abs")
        if not u.get("source_anchor"):
            missing.append("source_anchor")
        if missing:
            missing_required += 1
            failures.append(
                _make_failure_entry(
                    "missing_required_meta",
                    mst=mst,
                    crawl_ts=crawl_ts,
                    seed=seed,
                    unit=u,
                    detail={"missing": missing},
                )
            )
    summary["missing_required_meta"] = missing_required

    # 1-1) required meta check (full scan, deterministic)
    full_missing = 0
    for u in units:
        missing = []
        if not u.get("law"):
            missing.append("law")
        if not u.get("mst"):
            missing.append("mst")
        if not u.get("source_url"):
            missing.append("source_url")
        if not u.get("source_url_abs"):
            missing.append("source_url_abs")
        if not u.get("source_anchor"):
            missing.append("source_anchor")
        lvl = u.get("level")
        if lvl in ("조", "항", "목") and not u.get("article"):
            missing.append("article")
        if lvl in ("항", "목") and not u.get("paragraph"):
            missing.append("paragraph")
        if lvl == "목" and not u.get("item_path"):
            missing.append("item_path")
        if not u.get("annex_placeholder") and not (u.get("effective_date") or u.get("as_of")):
            missing.append("effective_date/as_of")
        if missing:
            full_missing += 1
            if full_missing <= 50:
                failures.append(
                    _make_failure_entry(
                        "missing_required_meta",
                        mst=mst,
                        crawl_ts=crawl_ts,
                        seed=seed,
                        unit=u,
                        detail={"missing": missing, "mode": "full"},
                    )
                )
    summary["missing_required_meta_full"] = full_missing

    # 1-2) API URL check (no /DRF/lawService.do or OC= in source_url/source_url_abs)
    api_bad = 0
    for u in units:
        bad = None
        for k in ("source_url", "source_url_abs"):
            v = u.get(k)
            if v and ("/DRF/lawService.do" in str(v) or "OC=" in str(v)):
                bad = {"key": k, "value": v}
                break
        if bad:
            api_bad += 1
            if api_bad <= 50:
                failures.append(
                    _make_failure_entry(
                        "source_url_api",
                        mst=mst,
                        crawl_ts=crawl_ts,
                        seed=seed,
                        unit=u,
                        detail=bad,
                    )
                )
    summary["source_url_api"] = api_bad

    if mode != "fast":
        # 2) header/body mismatch check
        mismatches = 0
        for u in sample:
            lvl = u.get("level")
            header = u.get("display_path_norm") or u.get("display_path") or u.get("path") or ""
            text = u.get("text") or ""
            ok = True
            if lvl == "조":
                art = u.get("article")
                if art and art not in header:
                    ok = False
            elif lvl == "항":
                art = u.get("article")
                par = u.get("paragraph")
                if (art and art not in header) or (par and par not in header):
                    ok = False
            elif lvl == "목":
                art = u.get("article")
                par = u.get("paragraph")
                item_path = u.get("item_path")
                item_type = u.get("item_type")
                if art and art not in header:
                    ok = False
                if par and par not in header:
                    ok = False
                if item_path:
                    expected = f"제{item_path}{item_type}" if item_type else None
                    if expected and expected in header:
                        pass
                    elif str(item_path) in header:
                        pass
                    elif _item_path_in_text(str(item_path), text):
                        pass
                    else:
                        ok = False
            if not ok:
                mismatches += 1
                failures.append(
                    _make_failure_entry(
                        "header_mismatch",
                        mst=mst,
                        crawl_ts=crawl_ts,
                        seed=seed,
                        unit=u,
                    )
                )
        summary["header_mismatch"] = mismatches

        # 3) annex reverse reference check
        annex_fail = 0
        for u in _sample_units(units, seed + 1, n_annex_sample):
            refs = u.get("annex_refs") or []
            if not refs:
                continue
            for r in refs:
                found = False
                for au in units:
                    if au.get("level") in ("별표", "서식"):
                        if (r == au.get("annex_no_human")) or (r == au.get("annex_no")):
                            if u.get("source_anchor") in (au.get("ref_by") or []):
                                found = True
                                break
                if not found:
                    annex_fail += 1
                    failures.append(
                        _make_failure_entry(
                            "annex_ref_fail",
                            mst=mst,
                            crawl_ts=crawl_ts,
                            seed=seed,
                            unit=u,
                            detail={"missing_ref": r},
                        )
                    )
        summary["annex_ref_fail"] = annex_fail

        # 4) EXCEPTION tag check
        exception_missing = 0
        for u in sample:
            text = u.get("text") or ""
            if any(k in text for k in ("다만", "제외", "아니한다")):
                tags = u.get("tags") or []
                if "EXCEPTION" not in tags:
                    exception_missing += 1
                    failures.append(
                        _make_failure_entry(
                            "exception_tag_missing",
                            mst=mst,
                            crawl_ts=crawl_ts,
                            seed=seed,
                            unit=u,
                        )
                    )
        summary["exception_tag_missing"] = exception_missing

        # 5) search normalization check
        pattern = re.compile(r"2\\s*m|2\\s*미터")
        target_idx = None
        base_text = ""
        for i, u in enumerate(units):
            t = u.get("text") or ""
            if pattern.search(t):
                target_idx = i
                base_text = t
                break

        search_fail = 0
        if target_idx is not None:
            def make_query(text: str, variant: str) -> str:
                return re.sub(r"2\\s*m|2\\s*미터", variant, text)

            queries = [
                ("2m", make_query(base_text, "2m")),
                ("2미터", make_query(base_text, "2미터")),
                ("2 미터", make_query(base_text, "2 미터")),
            ]

            def search(query: str, normalize: bool) -> List[int]:
                q = _normalize_text(query) if normalize else query
                v = embed_text(q)
                qv = _sanitize_vec(v)[None, :].astype("float32")
                _, I = index.search(qv, top_k)
                return I[0].tolist()

            improved_any = False
            for label, q in queries:
                raw_ids = search(q, normalize=False)
                norm_ids = search(q, normalize=True)
                raw_rank = raw_ids.index(target_idx) + 1 if target_idx in raw_ids else None
                norm_rank = norm_ids.index(target_idx) + 1 if target_idx in norm_ids else None
                if norm_rank is None:
                    search_fail += 1
                    failures.append(
                        _make_failure_entry(
                            "search_norm_missing",
                            mst=mst,
                            crawl_ts=crawl_ts,
                            seed=seed,
                            unit=units[target_idx],
                            detail={"query_label": label, "raw_rank": raw_rank, "norm_rank": norm_rank},
                        )
                    )
                elif raw_rank is None or (raw_rank and norm_rank and norm_rank < raw_rank):
                    improved_any = True
            if not improved_any:
                search_fail += 1
                failures.append(
                    _make_failure_entry(
                        "search_no_improvement",
                        mst=mst,
                        crawl_ts=crawl_ts,
                        seed=seed,
                        unit=units[target_idx],
                    )
                )
        else:
            summary["search_skipped"] = 1
        summary["search_fail"] = search_fail

        # 6) fail-fast simulation (remove source_url)
        law_json = _load_law_json(mst, law_json_path=law_json_path)
        if law_json:
            units_broken = [dict(u) for u in units]
            units_broken[0].pop("source_url", None)
            report = _validate_units_integrity(
                law_json,
                units_broken,
                None,
                mst=mst,
                crawl_ts=crawl_ts,
                seed=seed,
            )
            if not report.get("failures"):
                failures.append(
                    _make_failure_entry(
                        "fail_fast_not_triggered",
                        mst=mst,
                        crawl_ts=crawl_ts,
                        seed=seed,
                        unit=units_broken[0],
                    )
                )
                summary["fail_fast_fail"] = 1
            else:
                summary["fail_fast_fail"] = 0
                summary["id_check_status"] = report.get("id_check", {}).get("status")
        else:
            summary["fail_fast_fail"] = 1
            failures.append(
                _make_failure_entry(
                    "law_json_missing",
                    mst=mst,
                    crawl_ts=crawl_ts,
                    seed=seed,
                    unit=None,
                )
            )
    else:
        summary["header_mismatch"] = 0
        summary["annex_ref_fail"] = 0
        summary["exception_tag_missing"] = 0
        summary["search_fail"] = 0
        summary["fail_fast_fail"] = 0

    return failures, summary


def main():
    parser = argparse.ArgumentParser(description="QA checks for law RAG units.")
    parser.add_argument("--mst", required=True, help="법령 MST")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-meta-sample", type=int, default=100)
    parser.add_argument("--n-annex-sample", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--out-dir", default="faiss_indexes")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--law-json", default=None, help="law_json fixture path for offline QA")
    parser.add_argument("--units-json", default=None, help="units.json fixture path for offline QA")
    parser.add_argument("--mode", choices=["full", "fast"], default="full", help="QA mode")
    args = parser.parse_args()

    units_path = _load_units_path(args.out_dir, args.mst, units_json_path=args.units_json)
    with open(units_path, "r", encoding="utf-8") as f:
        units = json.load(f)

    failures, summary = _qa_checks(
        units,
        mst=str(args.mst),
        seed=args.seed,
        n_meta_sample=args.n_meta_sample,
        n_annex_sample=args.n_annex_sample,
        top_k=args.top_k,
        law_json_path=args.law_json,
        mode=args.mode,
    )

    report = {
        "mst": str(args.mst),
        "seed": args.seed,
        "summary": summary,
        "failures": failures,
    }

    if failures:
        report_path = args.report_path or os.path.join(args.out_dir, f"qa_report_{args.mst}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[QA] FAIL - report written to {report_path}")
        sys.exit(1)

    print("[QA] PASS")


if __name__ == "__main__":
    main()
