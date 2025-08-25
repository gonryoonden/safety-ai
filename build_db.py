import os
import json
import logging
from typing import List

from vector_search_service import build_indexes

logging.basicConfig(level=logging.INFO)


def load_msts_from_env() -> List[str]:
    v = os.environ.get('MSTS')
    if v:
        return [x.strip() for x in v.split(',') if x.strip()]
    # fallback sample
    return [
        '253527',  # sample MST from provided examples
        '266351',
        '261457',
    ]


if __name__ == '__main__':
    msts = load_msts_from_env()
    out_dir = os.environ.get('OUT_DIR', 'faiss_indexes')
    res = build_indexes(msts, out_dir)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    # manifest.json upsert
    from datetime import datetime
    manifest_path = os.path.join(out_dir, "manifest.json")
    now_iso = datetime.now().isoformat(timespec="seconds")
    try:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = []
    except Exception:
        manifest = []
    by_mst = {str(x.get("mst")): x for x in manifest if isinstance(x, dict)}
    for item in res:
        mst = str(item.get("mst"))
        by_mst[mst] = {
            "mst": mst,
            "law_title": item.get("law_title") or item.get("base"),
            "units": int(item.get("units") or 0),
            "build_ts": now_iso,
        }
    new_manifest = [by_mst[k] for k in sorted(by_mst.keys())]
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(new_manifest, f, ensure_ascii=False, indent=2)
    print(f"[INFO] manifest upserted: {manifest_path}")