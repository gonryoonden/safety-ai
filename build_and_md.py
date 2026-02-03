#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_and_md.py
1) build_indexes(MST들) → *_units.json 생성
2) units_to_md.py를 즉시 호출해 .md 생성(법령 통합/annex)
   ※ units_to_md.main() 시그니처(무인자/argv 인자형) 모두 호환
"""
import os, sys, glob, argparse, inspect

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vector_search_service import build_indexes  # 1단계
import units_to_md  # 2단계

def resolve_msts(args):
    if args.msts:  # 공백 구분 목록
        return [str(x).strip() for x in args.msts if str(x).strip()]
    v = os.environ.get("MSTS", "")
    return [x.strip() for x in v.split(",") if x.strip()] if v else []

def call_units_to_md(argv_list):
    """
    units_to_md.main() 이
    - (a) 인자 없는 스타일이면: sys.argv 패치 후 호출
    - (b) argv 리스트 받는 스타일이면: 바로 호출
    """
    # a) 인자 유무 확인
    try:
        sig = inspect.signature(units_to_md.main)
        params = list(sig.parameters.values())
    except Exception:
        params = None

    if not params:  # 인자정보를 못 읽거나, 파라미터가 0개로 보이면
        old_argv = sys.argv[:]
        try:
            sys.argv = ["units_to_md.py"] + argv_list
            return units_to_md.main()
        finally:
            sys.argv = old_argv
    else:
        # 파라미터가 1개 이상이면 argv 방식으로 가정
        return units_to_md.main(argv_list)

def run_units_to_md(md_out, out_dir, shard_chars, min_text_len, skip_deletion, annex=False, annex_suffix="_annex", emit=None):
    inputs = glob.glob(os.path.join(out_dir, "*_units.json"))
    if not inputs:
        print("[WARN] *_units.json 파일이 없습니다. out_dir 확인:", out_dir)
        return 0

    argv = ["--out-dir", md_out]
    if shard_chars and int(shard_chars) > 0:
        argv += ["--shard-chars", str(int(shard_chars))]
    if min_text_len and int(min_text_len) > 0:
        argv += ["--min-text-len", str(int(min_text_len))]
    if skip_deletion:
        argv += ["--skip-deletion"]
    if emit:
        argv += ["--emit", str(emit)]
        if str(emit) == "annex":
            argv += ["--annex-suffix", annex_suffix]
    elif annex:
        argv += ["--annex-only", "--annex-suffix", annex_suffix]
    argv += inputs

    print("[INFO] units_to_md.py", " ".join(argv))
    return call_units_to_md(argv)

def main():
    # 필수 환경: LAW_API_OC
    oc = os.environ.get("LAW_API_OC")
    if not oc:
        print("❌ 환경변수 LAW_API_OC 미설정")
        return 2
    print(f"✅ 환경변수 LAW_API_OC = {oc}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--msts", nargs="+", help="크롤링할 MST 목록(공백 구분). 없으면 $MSTS 사용")
    ap.add_argument("--out-dir", default="faiss_indexes")
    ap.add_argument("--md-out", default="md_out")
    ap.add_argument("--emit", choices=["full","annex","both"], default="both")
    ap.add_argument("--shard-chars", type=int, default=0)
    ap.add_argument("--min-text-len", type=int, default=0)
    ap.add_argument("--skip-deletion", action="store_true")
    ap.add_argument("--annex-suffix", default="_annex")
    args = ap.parse_args()

    msts = resolve_msts(args)
    if not msts:
        print("[ERROR] MSTS가 비었습니다. --msts 또는 환경변수 MSTS를 설정하세요.")
        return 2

    # 1) 크롤링/정규화/유닛 생성
    print("[INFO] building indexes for:", msts)
    res = build_indexes(msts, out_dir=args.out_dir)
    print("[INFO] build done:", res)

    # 2) .md 산출
    if args.emit in ("full", "annex", "both"):
        rc = run_units_to_md(
            args.md_out,
            args.out_dir,
            args.shard_chars,
            args.min_text_len,
            args.skip_deletion,
            annex_suffix=args.annex_suffix,
            emit=args.emit,
        )
        if rc:
            return rc

    print("[INFO] all done. md files in:", args.md_out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
