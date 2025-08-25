# mst_debug.py
import os, json, textwrap
from utils import LawAPIClient
from vector_search_service import fetch_full_law, _iter_clauses

if __name__ == "__main__":
    mst = os.getenv("DEBUG_MST", "272927")
    cli = LawAPIClient()
    data = fetch_full_law(cli, mst)
    clauses = list(_iter_clauses(data))
    print(f"[MST={mst}] 평탄화된 조/항/호 개수: {len(clauses)}")

    # 샘플 3개 출력
    for i, (aid, txt) in enumerate(clauses[:3], 1):
        print(f"\n--- 샘플 {i} / identifier={aid} ---")
        print(textwrap.shorten(txt.replace("\n", " "), width=200, placeholder="..."))

    # 디버그 파일 위치 안내
    print(f"\n디버그 파일: laws/debug_law_{mst}.json")
