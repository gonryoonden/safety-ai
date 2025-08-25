# inspect_units.py
import argparse, os, json, glob, re, collections

def find_units_path(out_dir: str, mst: str) -> str | None:
    # 1) 예쁜 파일명 베이스: *_MST_units.json
    for p in glob.glob(os.path.join(out_dir, f"*_{mst}_units.json")):
        return p
    # 2) MST_units.json
    p = os.path.join(out_dir, f"{mst}_units.json")
    if os.path.exists(p): return p
    # 3) BUNDLE_PER_LAW 형태: {out_dir}/{mst}/units.json
    p = os.path.join(out_dir, mst, "units.json")
    if os.path.exists(p): return p
    return None

def norm_jo(val: str | None, fallback_text: str | None = None) -> str | None:
    s = (val or "").strip()
    m = re.search(r'(\d+)(?:\s*의\s*(\d+))?', s)
    if not m and fallback_text:
        t = (fallback_text or "").strip()
        m = re.search(r'제\s*(\d+)\s*조\s*(?:의\s*(\d+))?', t)
    if not m: return None
    return f"{m.group(1)}의{m.group(2)}" if m.group(2) else m.group(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="faiss_indexes")
    ap.add_argument("--mst", required=True)
    ap.add_argument("--jo", required=True, help="예: 3 또는 4의2")
    ap.add_argument("--hang", type=int, default=None)
    ap.add_argument("--mok", type=int, default=None)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    upath = find_units_path(args.out_dir, args.mst)
    if not upath:
        print("[ERROR] units.json을 찾지 못했습니다.")
        return
    units = json.load(open(upath, "r", encoding="utf-8"))
    print(f"[INFO] units loaded: {len(units)} from {upath}")

    # jo 분포 확인
    ctr = collections.Counter(((u.get("jo") or "").strip()) for u in units)
    print("[INFO] jo 상위 10:", ctr.most_common(10))

    # 관대 매칭: jo==args.jo or jo_norm==args.jo or display/path에 '제{jo}조' 포함
    jo_key = str(args.jo)
    wanted = []
    for idx, u in enumerate(units):
        jo_raw = (u.get("jo") or "").strip()
        jo_n = u.get("jo_norm") or norm_jo(jo_raw, u.get("path") or u.get("display_path_norm"))
        dp = u.get("display_path_norm") or u.get("path") or ""
        jo_ok = (jo_raw == jo_key) or (jo_n == jo_key) or (f"제{jo_key}조" in dp)

        if not jo_ok:
            continue

        # 항/호가 지정되면 관대 매칭(정규화 필드 우선, 없으면 문자열 포함)
        if args.hang is not None:
            if u.get("hang_norm") != args.hang and f"제{args.hang}항" not in dp:
                continue
        if args.mok is not None:
            if u.get("mok_norm") != args.mok and f"제{args.mok}호" not in dp:
                continue

        wanted.append((idx, dp, (u.get("text") or "").replace("\n"," ")[:120]))

    print(f"[INFO] 매칭 결과 개수: {len(wanted)}")
    for i,(idx,dp,txt) in enumerate(wanted[:args.show], 1):
        print(f"{i}. [{idx}] {dp}")
        print(f"   {txt}")

if __name__ == "__main__":
    main()
