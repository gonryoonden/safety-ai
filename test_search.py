# test_search.py
import argparse, os, glob, json, sys, traceback
import numpy as np

print("[TEST] script start")  # 실행 여부 확인용

# --- FAISS (optional) ---
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# --- 유틸 ---
def guess_base(out_dir: str, mst: str) -> str:
    """out_dir 안에서 *_<mst>_units.json 을 우선 탐색해 base 추정"""
    pats = [
        os.path.join(out_dir, f"*_{mst}_units.json"),
        os.path.join(out_dir, f"{mst}_units.json"),
        os.path.join(out_dir, mst, "units.json"),  # BUNDLE_PER_LAW
    ]
    m = glob.glob(pats[0])
    if m:
        fn = os.path.basename(m[0])
        return fn[:-len("_units.json")]
    if os.path.exists(pats[1]):
        return mst
    if os.path.exists(pats[2]):
        return mst
    return mst

def parse_meta(q: str):
    """질의에서 제n조(의m), 제k항, 제r호 파싱"""
    import re
    CIRCLED = {"①":1,"②":2,"③":3,"④":4,"⑤":5,"⑥":6,"⑦":7,"⑧":8,"⑨":9,"⑩":10,
               "⑪":11,"⑫":12,"⑬":13,"⑭":14,"⑮":15,"⑯":16,"⑰":17,"⑱":18,"⑲":19,"⑳":20}
    q = (q or "").strip()
    # 조(의)
    jo = None
    m = re.search(r"(?:제\s*)?(\d+)\s*조\s*(?:의\s*(\d+))?", q)
    if m:
        jo = f"{m.group(1)}의{m.group(2)}" if m.group(2) else m.group(1)
    # 항
    hn = None
    m = re.search(r"(?:제\s*)?([0-9①-⑳]+)\s*항", q)
    if m:
        tok = m.group(1)
        hn = CIRCLED.get(tok, None)
        if hn is None:
            m2 = re.search(r"(\d+)", tok)
            hn = int(m2.group(1)) if m2 else None
    # 호
    mn = None
    m = re.search(r"(?:제\s*)?(\d+)\s*호", q)
    if m:
        mn = int(m.group(1))
    return jo, hn, mn

def meta_filter(units, jo=None, hang_norm=None, mok_norm=None, top_k=5):
    """조/항/호 관대 매칭 폴백"""
    res = []
    for idx, u in enumerate(units):
        dp = u.get("display_path_norm") or u.get("path") or ""
        # 조: jo 또는 jo_norm 일치, 또는 "제{jo}조" 텍스트 포함 허용
        jo_ok = True
        if jo:
            jo_ok = (u.get("jo") == jo) or (u.get("jo_norm") == jo) or (f"제{jo}조" in dp)
        if not jo_ok:
            continue
        # 항
        if hang_norm is not None and u.get("hang_norm") != hang_norm and f"제{hang_norm}항" not in dp:
            continue
        # 호
        if mok_norm is not None and u.get("mok_norm") != mok_norm and f"제{mok_norm}호" not in dp:
            continue
        res.append((idx, u))
        if len(res) >= top_k:
            break
    return res

# --- 메인 ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="faiss_indexes")
    ap.add_argument("--mst", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    base = args.base or guess_base(args.out_dir, args.mst)

    index_path = os.path.join(args.out_dir, f"{args.mst}_faiss_index.bin")
    units_candidates = [
        os.path.join(args.out_dir, f"{base}_units.json"),
        os.path.join(args.out_dir, base, "units.json"),
        os.path.join(args.out_dir, f"{args.mst}_units.json"),
    ]

    print(f"[DEBUG] out_dir={args.out_dir}, mst={args.mst}, base={base}")
    print(f"[DEBUG] index exists? {os.path.exists(index_path)} ({index_path})")

    units_path = next((p for p in units_candidates if os.path.exists(p)), None)
    print(f"[DEBUG] units found? {bool(units_path)} ({units_path})")
    if not units_path:
        print("[ERROR] units.json 계열을 찾지 못했습니다. out-dir 내용:")
        for f in glob.glob(os.path.join(args.out_dir, "*")):
            print(" -", f)
        sys.exit(2)

    with open(units_path, "r", encoding="utf-8") as f:
        units = json.load(f)
    print(f"[DEBUG] units count: {len(units)}")

    # --- FAISS 경로 ---
    if faiss is not None and os.path.exists(index_path):
        index = faiss.read_index(index_path)
        nt = getattr(index, "ntotal", 0)
        print(f"[DEBUG] index.ntotal: {nt}")

        if nt > 0:
            # 1) 질의 임베딩 (vector_search_service의 안전 함수 재사용)
            from vector_search_service import embed_text as _embed_text, _sanitize_vec  # type: ignore
            k = min(nt, max(args.top_k * 20, args.top_k))
            qv = _sanitize_vec(_embed_text(args.query)).astype("float32")[None, :]

            # 2) 검색 + 디버그
            D, I = index.search(qv, k)
            d0 = D[0].tolist(); i0 = I[0].tolist()
            print(f"[DEBUG] first10 I: {i0[:10]}")
            print(f"[DEBUG] first5 D: {d0[:5]}")
            print(f"[DEBUG] qdim={qv.shape[1]}, index.d={getattr(index, 'd', None)}")
            print("[DEBUG] q finite? ", bool(np.isfinite(qv).all()))

            # 3) 후보 수집(FAISS)
            cand = [(idx, dist, units[idx]) for dist, idx in zip(d0, i0) if idx >= 0]

            # ✅ 3.5) 메타 매칭 결과를 후보 집합에 '강제 합류'
            jo, hn, mn = parse_meta(args.query)

            # meta_filter는 위에 정의되어 있습니다. 넉넉히 최대 200개만 뽑아 병합.
            meta_hits = [i for (i, _) in meta_filter(units, jo, hn, mn, top_k=200)] if (jo or hn is not None or mn is not None) else []
            already = {i for (i, _, _) in cand}
            for i in meta_hits:
                if i not in already:
                    # 거리값은 ∞로 두되, 아래 prio로 1순위에 끌어올립니다.
                    cand.append((i, float("inf"), units[i]))

            # 4) 메타 우선 재정렬
            def prio(i):
                u = units[i]
                score = 0
                if jo:
                    if (u.get("jo") == jo) or (u.get("jo_norm") == jo) or (f"제{jo}조" in (u.get("display_path_norm") or u.get("path") or "")):
                        score += 2
                if hn is not None and u.get("hang_norm") == hn:
                    score += 1
                if mn is not None and u.get("mok_norm") == mn:
                    score += 1
                return score

            cand.sort(key=lambda t: (-prio(t[0]), t[1]))  # 우선순위↓, 거리↑

            if cand:
                results = cand[:args.top_k]
                print(f"[RESULT] [MST={args.mst} base={base}] top{args.top_k} | query: {args.query}")
                for rank, (i, dist, u) in enumerate(results, 1):
                    dp = u.get("display_path_norm") or u.get("path")
                    text = (u.get("text") or "").replace("\n"," ")
                    if len(text) > 160: text = text[:160] + "..."
                    print(f"{rank}. {dp} | dist={dist:.4f}")
                    print(f"   {text}")
                return
            else:
                print("[WARN] FAISS 후보 없음 → 메타 폴백으로 진행")
        else:
            print("[WARN] index.ntotal=0 → 메타 폴백으로 진행")
    else:
        print("[WARN] FAISS 미존재 또는 index 파일 없음 → 메타 폴백으로 진행")

    # --- 메타 폴백 ---
    jo, hn, mn = parse_meta(args.query)
    mf = meta_filter(units, jo, hn, mn, top_k=args.top_k)
    print(f"[RESULT/FALLBACK] [MST={args.mst} base={base}] top{args.top_k} | query: {args.query}")
    if not mf:
        print("메타와 일치하는 후보가 없습니다.")
    for rank, (i, u) in enumerate(mf, 1):
        dp = u.get("display_path_norm") or u.get("path")
        text = (u.get("text") or "").replace("\n"," ")
        if len(text) > 160: text = text[:160] + "..."
        print(f"{rank}. {dp} | faiss_id={i}")
        print(f"   {text}")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("[FATAL] Unhandled exception:", repr(e))
        traceback.print_exc()
        sys.exit(1)

